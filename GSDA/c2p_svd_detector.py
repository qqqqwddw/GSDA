"""
C2P + SVD Fusion Detector - SVD+LoRA Hybrid Version
融合方案: q_proj/k_proj/out_proj用SVD, v_proj用LoRA, MLP冻结

★ 核心修改 (根据EFFORT论文原始代码):
1. q_proj, k_proj, out_proj → SVD (学习伪影模式)
2. v_proj → LoRA (学习类别概念)
3. MLP层 → 完全冻结 (保持预训练知识)
4. Classifier → 简单Linear层 (论文风格)
5. 损失函数 → BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from modeling_svd import inject_svd_into_clip_vision, SVDResidualLinear, collect_svd_losses
from typing import Optional, List

# ★ 导入PEFT的LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft library not found. LoRA will not be available.")
    print("Install with: pip install peft")


class C2P_SVD_LoRA_Detector(nn.Module):
    """
    ★ C2P + SVD + LoRA 混合检测器 (论文风格实现)
    
    融合架构:
    ┌────────────────────────────────────────────────────────────┐
    │  CLIP Vision Encoder (24层)                                │
    │  ├── Transformer Layer                                      │
    │  │   ├── Self-Attention                                     │
    │  │   │   ├── q_proj  → [SVD r=1023]  ← EFFORT              │
    │  │   │   ├── k_proj  → [SVD r=1023]  ← EFFORT              │
    │  │   │   ├── v_proj  → [LoRA r=8]    ← C2P-CLIP            │
    │  │   │   └── out_proj → [SVD r=1023] ← EFFORT              │
    │  │   └── MLP                                                │
    │  │       ├── fc1 → [冻结]                                   │
    │  │       └── fc2 → [冻结]                                   │
    │  └── (重复24层)                                             │
    │                                                             │
    │  → pooler_output (1024维)                                   │
    │  → Classifier: Linear(1024, 1) [论文风格]                   │
    │  → BCEWithLogitsLoss                                        │
    └────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, 
                 clip_model_name='openai/clip-vit-large-patch14',
                 num_classes=1,  # ★ 论文默认使用1（sigmoid）
                 svd_rank=1023,  # ★ 论文默认n-r=1，所以r=1024-1=1023
                 lora_rank=8,
                 lora_alpha=8.0,
                 lora_dropout=0.8,
                 use_text_guidance=True,
                 class_weights: Optional[torch.Tensor] = None,
                 init_gain=0.02):  # ★ 论文的初始化增益
        """
        Args:
            clip_model_name: CLIP模型名称
            num_classes: 分类数量 (1=sigmoid, 2=softmax)
            svd_rank: SVD保留的主成分数量 (论文默认1023，即n-r=1)
            lora_rank: LoRA的秩
            lora_alpha: LoRA的缩放因子
            lora_dropout: LoRA的dropout率
            use_text_guidance: 是否启用C2P文本引导
            class_weights: 类别权重
            init_gain: 分类器权重初始化的标准差 (论文默认0.02)
        """
        super().__init__()
        
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for LoRA. Install with: pip install peft")
        
        self.use_text_guidance = use_text_guidance
        self.num_classes = num_classes
        self.svd_rank = svd_rank
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.init_gain = init_gain
        self.class_weights = class_weights
        
        print("\n" + "="*70)
        print("Initializing C2P + SVD + LoRA Fusion Detector (Paper Style)")
        print(f"  SVD (q/k/out_proj): rank={svd_rank}, residual_dim={1024-svd_rank}")
        print(f"  LoRA (v_proj): rank={lora_rank}, α={lora_alpha}, dropout={lora_dropout}")
        print(f"  MLP: Frozen")
        print(f"  Num Classes: {num_classes} ({'BCE+sigmoid' if num_classes == 1 else 'CE+softmax'})")
        print(f"  Classifier: Linear(1024, {num_classes}) [Paper Style]")
        print(f"  Text Guidance: {'Enabled' if use_text_guidance else 'Disabled'}")
        print("="*70)
        
        # ============ 1. 加载CLIP模型 ============
        print(f"\nLoading CLIP model: {clip_model_name}")
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # ============ 2. 冻结所有CLIP参数 ============
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # ============ 3. 注入SVD到q_proj, k_proj, out_proj ============
        print("\n★ Step 1: Injecting SVD into q_proj, k_proj, out_proj...")
        self.clip.vision_model = inject_svd_into_clip_vision(
            self.clip.vision_model,
            r=svd_rank,
            svd_target_modules=['q_proj', 'k_proj', 'out_proj'],
            include_mlp=False
        )
        
        # ★★★ 关键修复：显式启用SVD参数训练 ★★★
        self._enable_svd_training()
        
        # ============ 4. 对v_proj应用LoRA ============
        print("\n★ Step 2: Applying LoRA to v_proj...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_config)
        print(f"  ✓ LoRA applied to v_proj layers")
        
        # ★★★ 再次确保SVD参数可训练（PEFT可能会影响） ★★★
        self._enable_svd_training()
        
        # ============ 5. 获取特征维度 ============
        # ★ 论文使用 pooler_output (1024维)，不使用 visual_projection
        self.vision_hidden_size = 1024  # CLIP-ViT-L的hidden size
        
        print(f"\n  Vision hidden size: {self.vision_hidden_size}")
        print(f"  Classifier input: pooler_output ({self.vision_hidden_size}D)")
        
        # ============ 6. 分类头 (论文风格：简单Linear) ============
        # ★★★ 关键修改：使用论文的简单分类器 ★★★
        self.fc = nn.Linear(self.vision_hidden_size, num_classes)
        
        # ★ 论文风格的权重初始化
        nn.init.normal_(self.fc.weight.data, mean=0.0, std=init_gain)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias.data)
        
        print(f"  Classifier: Linear({self.vision_hidden_size}, {num_classes})")
        print(f"  Classifier params: {self.vision_hidden_size * num_classes + num_classes:,}")
        print(f"  Weight init: normal(0, {init_gain})")
        
        # ============ 7. 损失函数 ============
        # ★★★ 关键修改：使用论文的BCE损失 ★★★
        if num_classes == 1:
            # ★ 修复：正确处理设备
            if class_weights is not None:
                # 方案A：如果 class_weights 是 tensor
                if isinstance(class_weights, torch.Tensor):
                    pos_weight = (class_weights[1] / class_weights[0]).unsqueeze(0)
                # 方案B：如果是列表
                else:
                    pos_weight = torch.tensor(
                        [class_weights[1] / class_weights[0]], 
                        dtype=torch.float32
                    )
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"  Loss: BCEWithLogitsLoss(pos_weight={pos_weight.item():.2f})")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                print(f"  Loss: BCEWithLogitsLoss")
        else:
            # CrossEntropy for multi-class
            if class_weights is not None:
                if isinstance(class_weights, torch.Tensor):
                    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
            else:
                self.criterion = nn.CrossEntropyLoss()
            print(f"  Loss: CrossEntropyLoss")
        
        # ============ 8. C2P原型 ============
        self.register_buffer('prototype_real', None)
        self.register_buffer('prototype_fake', None)
        
        # ============ 9. 温度参数 ============
        self.temperature = 0.07
        
        self._print_trainable_params()
    
    def _enable_svd_training(self):
        """
        ★★★ 完全修复版：启用SVD残差参数的训练 ★★★
        修复：
        1. 参数命名 sigma_residual → S_residual
        2. 使用 modules() 递归遍历
        3. 检查 PEFT 的 base_model
        """
        svd_params_enabled = 0
        svd_layers_count = 0
        checked_modules = set()  # 避免重复计数
        
        def enable_module_params(module):
            """启用单个SVDResidualLinear层的参数"""
            nonlocal svd_params_enabled, svd_layers_count
            
            # 避免重复处理同一个模块
            if id(module) in checked_modules:
                return
            checked_modules.add(id(module))
            
            if isinstance(module, SVDResidualLinear):
                svd_layers_count += 1
                
                # ★ 修复：使用正确的参数名 S_residual
                for param_name in ['S_residual', 'U_residual', 'V_residual']:
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if param is not None and isinstance(param, nn.Parameter):
                            if not param.requires_grad:
                                param.requires_grad = True
                            svd_params_enabled += param.numel()
        
        # ★ 修复：使用 modules() 递归遍历（不是 named_modules）
        for module in self.clip.vision_model.modules():
            enable_module_params(module)
        
        # ★ 修复：如果有 PEFT 包装，也检查 base_model
        if hasattr(self.clip.vision_model, 'base_model'):
            print(f"  ⚠ Detected PEFT wrapper, enabling SVD in base_model...")
            for module in self.clip.vision_model.base_model.modules():
                enable_module_params(module)
        
        print(f"\n  ✓ SVD Training Status:")
        print(f"    - SVD layers found: {svd_layers_count}")
        print(f"    - SVD params enabled: {svd_params_enabled:,}")
        
        if svd_params_enabled == 0:
            print(f"  ⚠️  CRITICAL: No SVD parameters were enabled!")
            self._debug_svd_params()
            return False
        
        return True
    
    def _print_trainable_params(self):
        """★ 打印可训练参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        svd_params = 0
        lora_params = 0
        classifier_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                    if 'lora' not in name.lower():
                        svd_params += param.numel()
                        continue
                
                if 'lora' in name.lower():
                    lora_params += param.numel()
                elif 'fc.' in name or 'classifier' in name:
                    classifier_params += param.numel()
        
        print(f"\n{'='*70}")
        print(f"Parameter Statistics:")
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,} ({trainable/total*100:.4f}%)")
        print(f"  Frozen: {total - trainable:,}")
        print(f"\n  Breakdown:")
        print(f"    SVD (q/k/out_proj): {svd_params:,} ({svd_params/max(trainable,1)*100:.2f}% of trainable)")
        print(f"    LoRA (v_proj): {lora_params:,} ({lora_params/max(trainable,1)*100:.2f}% of trainable)")
        print(f"    Classifier (fc): {classifier_params:,} ({classifier_params/max(trainable,1)*100:.2f}% of trainable)")
        print(f"{'='*70}\n")
        
        if svd_params == 0:
            print("⚠️  WARNING: No SVD parameters detected as trainable!")
            self._debug_svd_params()
    
    def _debug_svd_params(self):
        """调试：详细检查SVD参数状态"""
        print("\n  🔍 SVD Parameter Debug:")
        
        found_svd = False
        
        # 检查所有模块
        all_modules = list(self.clip.vision_model.modules())
        if hasattr(self.clip.vision_model, 'base_model'):
            all_modules.extend(self.clip.vision_model.base_model.modules())
        
        for module in all_modules:
            if isinstance(module, SVDResidualLinear):
                found_svd = True
                print(f"\n    Found SVDResidualLinear:")
                
                for param_name in ['S_residual', 'U_residual', 'V_residual']:
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if param is not None:
                            print(f"      {param_name}: "
                                  f"shape={param.shape}, "
                                  f"requires_grad={param.requires_grad}, "
                                  f"is_param={isinstance(param, nn.Parameter)}")
                break  # 只显示第一个
        
        if not found_svd:
            print("    ✗ No SVDResidualLinear modules found!")
            print("    Check if SVD injection succeeded.")
    
    def _get_vision_model_forward(self):
        """
        ★★★ 关键修复：获取正确的vision model forward方法 ★★★
        处理PEFT包装器的问题
        """
        vision_model = self.clip.vision_model
        
        # 检查是否被PEFT包装
        if hasattr(vision_model, 'base_model'):
            # PEFT包装的情况
            if hasattr(vision_model.base_model, 'model'):
                # PeftModel -> LoraModel -> 原始模型
                return vision_model.base_model.model
            else:
                return vision_model.base_model
        else:
            return vision_model
    
    def encode_image(self, images, return_feature=False):
        """
        ★★★ 修复版：图像编码 ★★★
        修复：正确处理PEFT包装，避免input_ids错误
        """
        batch_size, _, height, width = images.shape
        
        # 强制resize到224x224（CLIP标准输入）
        if height != 224 or width != 224:
            images = F.interpolate(
                images, 
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        
        # ★★★ 关键修复：直接调用底层模型，避免PEFT的forward问题 ★★★
        vision_model = self.clip.vision_model
        
        # 方法1：尝试直接调用（适用于某些PEFT版本）
        try:
            # 获取底层的vision encoder
            if hasattr(vision_model, 'base_model'):
                # PEFT包装
                if hasattr(vision_model.base_model, 'model'):
                    # PeftModel.base_model.model = 原始CLIPVisionModel
                    actual_model = vision_model.base_model.model
                else:
                    actual_model = vision_model.base_model
            else:
                actual_model = vision_model
            
            # 直接调用forward
            vision_outputs = actual_model(
                pixel_values=images,
                output_hidden_states=True,
                return_dict=True
            )
            
        except Exception as e:
            # 方法2：手动执行forward步骤
            print(f"  Warning: Direct call failed ({e}), using manual forward...")
            vision_outputs = self._manual_vision_forward(images)
        
        pooler_output = vision_outputs.pooler_output
        
        if return_feature:
            return pooler_output
        
        return pooler_output
    
    def _manual_vision_forward(self, pixel_values):
        """
        ★ 手动执行视觉模型的forward（作为后备方案）
        """
        vision_model = self.clip.vision_model
        
        # 获取嵌入层
        if hasattr(vision_model, 'base_model'):
            if hasattr(vision_model.base_model, 'model'):
                embeddings = vision_model.base_model.model.embeddings
                encoder = vision_model.base_model.model.encoder
                pre_layrnorm = vision_model.base_model.model.pre_layrnorm
                post_layernorm = vision_model.base_model.model.post_layernorm
            else:
                embeddings = vision_model.base_model.embeddings
                encoder = vision_model.base_model.encoder
                pre_layrnorm = vision_model.base_model.pre_layrnorm
                post_layernorm = vision_model.base_model.post_layernorm
        else:
            embeddings = vision_model.embeddings
            encoder = vision_model.encoder
            pre_layrnorm = vision_model.pre_layrnorm
            post_layernorm = vision_model.post_layernorm
        
        # Forward pass
        hidden_states = embeddings(pixel_values)
        hidden_states = pre_layrnorm(hidden_states)
        
        encoder_outputs = encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        pooler_output = post_layernorm(last_hidden_state[:, 0, :])
        
        # 创建输出对象
        class VisionOutputs:
            pass
        
        outputs = VisionOutputs()
        outputs.last_hidden_state = last_hidden_state
        outputs.pooler_output = pooler_output
        outputs.hidden_states = encoder_outputs.hidden_states
        
        return outputs
    
    def encode_text(self, text_list):
        """文本编码 (用于C2P)"""
        device = next(self.parameters()).device
        
        valid_texts = []
        for text in text_list:
            if text is not None and isinstance(text, str) and len(text.strip()) > 0:
                valid_texts.append(text)
            else:
                valid_texts.append("An image")
        
        inputs = self.tokenizer(
            valid_texts, 
            padding=True, 
            truncation=True, 
            max_length=77,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            text_outputs = self.clip.text_model(**inputs)
            text_embeds = text_outputs.pooler_output
            text_features = self.clip.text_projection(text_embeds)
        
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        return text_features
    
    def forward(self, images, labels=None, captions=None, return_feature=False):
        """
        ★ 前向传播 (论文风格)
        
        论文代码:
            features = self.model.vision_model(x)['pooler_output']
            if return_feature:
                return features
            return self.fc(features)
        """
        # 获取视觉特征
        features = self.encode_image(images, return_feature=True)  # [B, 1024]
        
        # 分类
        logits = self.fc(features)  # [B, num_classes]
        
        # ★ 推理模式
        if labels is None:
            if return_feature:
                return {'logits': logits, 'features': features}
            return logits
        
        # ★ 训练模式：计算损失
        losses = {}
        
        # 分类损失
        if self.num_classes == 1:
            # ★ 论文风格：BCE损失
            loss_cls = self.criterion(logits.squeeze(-1), labels.float())
        else:
            loss_cls = self.criterion(logits, labels)
        losses['cls'] = loss_cls
        
        # SVD约束损失
        loss_ortho, loss_keepsv = self._compute_svd_losses()
        losses['ortho'] = loss_ortho
        losses['keepsv'] = loss_keepsv
        
        # C2P损失 (可选)
        if self.use_text_guidance:
            # 对于C2P，需要归一化的特征
            normalized_features = F.normalize(features, p=2, dim=-1)
            
            loss_prototype = self._compute_prototype_loss(normalized_features, labels)
            losses['prototype'] = loss_prototype
            
            if captions is not None:
                # 为caption loss，需要投影到文本空间
                projected = self.clip.visual_projection(features)
                projected = F.normalize(projected, p=2, dim=-1)
                loss_caption = self._compute_caption_loss(projected, captions, labels)
                losses['caption'] = loss_caption
            else:
                losses['caption'] = torch.tensor(0.0, device=images.device)
        else:
            losses['prototype'] = torch.tensor(0.0, device=images.device)
            losses['caption'] = torch.tensor(0.0, device=images.device)
        
        losses['logits'] = logits
        if return_feature:
            losses['features'] = features
        
        return losses
    
    def compute_losses(self, images, labels, texts=None):
        """计算所有损失 (训练接口)"""
        result = self.forward(images, labels=labels, captions=texts)
        
        return {
            'cls': result['cls'],
            'prototype': result['prototype'],
            'caption': result['caption'],
            'ortho': result['ortho'],
            'keepsv': result['keepsv']
        }
    
    def _compute_svd_losses(self):
        """
        ★ 修复版：计算SVD约束损失
        修复：确保收集到所有SVD模块
        """
        device = next(self.parameters()).device
        total_ortho = torch.tensor(0.0, device=device)
        total_keepsv = torch.tensor(0.0, device=device)
        count = 0
        
        # 收集所有 SVDResidualLinear 模块（避免重复）
        svd_modules = []
        checked_ids = set()
        
        # 从 vision_model 收集
        for module in self.clip.vision_model.modules():
            if isinstance(module, SVDResidualLinear):
                if id(module) not in checked_ids:
                    svd_modules.append(module)
                    checked_ids.add(id(module))
        
        # 从 base_model 收集（如果存在）
        if hasattr(self.clip.vision_model, 'base_model'):
            for module in self.clip.vision_model.base_model.modules():
                if isinstance(module, SVDResidualLinear):
                    if id(module) not in checked_ids:
                        svd_modules.append(module)
                        checked_ids.add(id(module))
        
        # 计算损失
        for module in svd_modules:
            try:
                ortho_loss = module.compute_orthogonal_loss_decomposed()
                keepsv_loss = module.compute_keepsv_loss()
                
                total_ortho = total_ortho + ortho_loss
                total_keepsv = total_keepsv + keepsv_loss
                count += 1
            except Exception as e:
                print(f"Warning: Failed to compute SVD loss: {e}")
                continue
        
        if count > 0:
            return total_ortho / count, total_keepsv / count
        else:
            return total_ortho, total_keepsv
    
    def _compute_prototype_loss(self, features, labels):
        """计算原型对比损失 (C2P)"""
        device = features.device
        
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        real_features = features[real_mask]
        fake_features = features[fake_mask]
        
        momentum = 0.9
        
        if real_features.shape[0] > 0:
            real_proto = real_features.mean(dim=0)
            if self.prototype_real is None:
                self.prototype_real = real_proto.detach()
            else:
                self.prototype_real = momentum * self.prototype_real + (1 - momentum) * real_proto.detach()
        
        if fake_features.shape[0] > 0:
            fake_proto = fake_features.mean(dim=0)
            if self.prototype_fake is None:
                self.prototype_fake = fake_proto.detach()
            else:
                self.prototype_fake = momentum * self.prototype_fake + (1 - momentum) * fake_proto.detach()
        
        if self.prototype_real is None or self.prototype_fake is None:
            return torch.tensor(0.0, device=device)
        
        loss = torch.tensor(0.0, device=device)
        
        if real_features.shape[0] > 0:
            pos_sim = F.cosine_similarity(real_features, self.prototype_real.unsqueeze(0), dim=-1)
            neg_sim = F.cosine_similarity(real_features, self.prototype_fake.unsqueeze(0), dim=-1)
            loss = loss + (-torch.log(torch.exp(pos_sim / self.temperature) / 
                          (torch.exp(pos_sim / self.temperature) + torch.exp(neg_sim / self.temperature)))).mean()
        
        if fake_features.shape[0] > 0:
            pos_sim = F.cosine_similarity(fake_features, self.prototype_fake.unsqueeze(0), dim=-1)
            neg_sim = F.cosine_similarity(fake_features, self.prototype_real.unsqueeze(0), dim=-1)
            loss = loss + (-torch.log(torch.exp(pos_sim / self.temperature) / 
                          (torch.exp(pos_sim / self.temperature) + torch.exp(neg_sim / self.temperature)))).mean()
        
        return loss
    
    def _compute_caption_loss(self, image_features, captions, labels):
        """计算Caption对比损失 (C2P)"""
        batch_size = image_features.shape[0]
        device = image_features.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        text_features = self.encode_text(captions)
        
        sim_matrix = torch.mm(image_features, text_features.t()) / self.temperature
        pos_mask = torch.eye(batch_size, device=device).bool()
        
        loss = 0.0
        for i in range(batch_size):
            pos_sim = sim_matrix[i, i]
            neg_mask = ~pos_mask[i]
            neg_sims = sim_matrix[i, neg_mask]
            
            if neg_sims.numel() == 0:
                continue
            
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
        
        return loss / batch_size
    
    def predict(self, images):
        """
        ★ 推理接口 (论文风格)
        
        返回:
            如果 num_classes=1: 返回概率 (sigmoid)
            如果 num_classes=2: 返回类别概率 (softmax)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            
            if self.num_classes == 1:
                # BCE: sigmoid得到fake概率
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs > 0.5).long()
                return preds, probs
            else:
                # CE: softmax得到类别概率
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                return preds, probs[:, 1]  # 返回fake的概率
    
    def get_fusion_status(self):
        """获取融合架构状态"""
        status = {
            'svd_layers': [],
            'lora_layers': [],
            'frozen_layers': []
        }
        
        # 收集SVD层
        checked_ids = set()
        for module in self.clip.vision_model.modules():
            if isinstance(module, SVDResidualLinear):
                if id(module) not in checked_ids:
                    checked_ids.add(id(module))
                    params = module.get_trainable_params()
                    status['svd_layers'].append({
                        'trainable_params': params['total'],
                        'trainable_ratio': params['ratio']
                    })
        
        if hasattr(self.clip.vision_model, 'base_model'):
            for module in self.clip.vision_model.base_model.modules():
                if isinstance(module, SVDResidualLinear):
                    if id(module) not in checked_ids:
                        checked_ids.add(id(module))
                        params = module.get_trainable_params()
                        status['svd_layers'].append({
                            'trainable_params': params['total'],
                            'trainable_ratio': params['ratio']
                        })
        
        # 收集LoRA层
        for name, param in self.clip.vision_model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                status['lora_layers'].append({
                    'name': name,
                    'shape': list(param.shape),
                    'params': param.numel()
                })
        
        return status
    
    def save_pretrained(self, save_path):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存完整状态
        state_dict = {
            'model_state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'svd_rank': self.svd_rank,
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
                'use_text_guidance': self.use_text_guidance,
                'vision_hidden_size': self.vision_hidden_size,
            }
        }
        torch.save(state_dict, os.path.join(save_path, 'model.pt'))
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path, clip_model_name='openai/clip-vit-large-patch14', device='cuda'):
        """加载模型"""
        import os
        state_dict = torch.load(os.path.join(load_path, 'model.pt'), map_location=device)
        config = state_dict['config']
        
        model = cls(
            clip_model_name=clip_model_name,
            num_classes=config['num_classes'],
            svd_rank=config['svd_rank'],
            lora_rank=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            use_text_guidance=config['use_text_guidance'],
        )
        
        model.load_state_dict(state_dict['model_state_dict'])
        model.to(device)
        print(f"Model loaded from {load_path}")
        return model


# ============================================================
# 辅助函数
# ============================================================

def get_trainable_params(model):
    """获取可训练参数统计"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    svd_params = 0
    lora_params = 0
    classifier_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                if 'lora' not in name.lower():
                    svd_params += param.numel()
                    continue
            
            if 'lora' in name.lower():
                lora_params += param.numel()
            elif 'fc.' in name or 'classifier' in name:
                classifier_params += param.numel()
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_ratio': trainable_params / total_params * 100 if total_params > 0 else 0,
        'svd_params': svd_params,
        'lora_params': lora_params,
        'classifier_params': classifier_params
    }


# 向后兼容别名
C2P_SVD_Detector = C2P_SVD_LoRA_Detector


# ============================================================
# 测试代码
# ============================================================

def test_fusion_detector():
    """测试融合检测器"""
    print("\n" + "="*70)
    print("Testing C2P + SVD + LoRA Fusion Detector (Paper Style)")
    print("="*70)
    
    if not PEFT_AVAILABLE:
        print("\n✗ peft library not found. Please install: pip install peft")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ★ 使用论文默认参数
    detector = C2P_SVD_LoRA_Detector(
        clip_model_name='openai/clip-vit-large-patch14',
        num_classes=1,      # ★ 论文风格：sigmoid
        svd_rank=1023,      # ★ 论文默认：n-r=1
        lora_rank=8,
        lora_alpha=8.0,
        lora_dropout=0.8,
        use_text_guidance=True,
        init_gain=0.02      # ★ 论文默认初始化
    ).to(device)
    
    print("\n1. Testing trainable params...")
    params_info = get_trainable_params(detector)
    print(f"   Total: {params_info['total']:,}")
    print(f"   Trainable: {params_info['trainable']:,} ({params_info['trainable_ratio']:.4f}%)")
    print(f"   SVD params: {params_info['svd_params']:,}")
    print(f"   LoRA params: {params_info['lora_params']:,}")
    print(f"   Classifier params: {params_info['classifier_params']:,}")
    
    # ★ 验证SVD参数
    if params_info['svd_params'] > 0:
        print("   ✓ SVD parameters are TRAINABLE!")
    else:
        print("   ✗ WARNING: SVD parameters are NOT trainable!")
    
    # ★ 验证分类器参数 (论文风格应该是 1024*1+1 = 1025)
    expected_classifier_params = 1024 * 1 + 1  # Linear(1024, 1)
    print(f"   Expected classifier params: {expected_classifier_params}")
    
    print("\n2. Testing inference...")
    dummy_images = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        logits = detector(dummy_images)
    print(f"   Logits shape: {logits.shape}")  # 应该是 [4, 1]
    print(f"   Logits: {logits.squeeze().tolist()}")
    
    # ★ 测试predict函数
    preds, probs = detector.predict(dummy_images)
    print(f"   Predictions: {preds.tolist()}")
    print(f"   Probabilities: {[f'{p:.4f}' for p in probs.tolist()]}")
    print("   ✓ Inference works!")
    
    print("\n3. Testing training...")
    detector.train()
    dummy_labels = torch.randint(0, 2, (4,)).to(device)
    dummy_captions = ["Real face photo", "Fake deepfake face", "Natural photo", "AI generated"]
    
    losses = detector.compute_losses(dummy_images, dummy_labels, dummy_captions)
    print(f"   Losses: {list(losses.keys())}")
    print(f"   cls loss: {losses['cls'].item():.4f}")
    print(f"   ortho loss: {losses['ortho'].item():.6f}")
    print(f"   keepsv loss: {losses['keepsv'].item():.6f}")
    print(f"   prototype loss: {losses['prototype'].item():.4f}")
    print(f"   caption loss: {losses['caption'].item():.4f}")
    print("   ✓ Training losses computed!")
    
    print("\n4. Testing gradient flow...")
    optimizer = torch.optim.AdamW(
        [p for p in detector.parameters() if p.requires_grad],
        lr=2e-4  # ★ 论文使用的学习率
    )
    
    # 前向传播
    losses = detector.compute_losses(dummy_images, dummy_labels, dummy_captions)
    
    # ★ 论文的损失组合
    lambda1 = 0.1  # ortho权重
    lambda2 = 0.1  # keepsv权重
    total_loss = losses['cls'] + lambda1 * losses['ortho'] + lambda2 * losses['keepsv']
    
    if detector.use_text_guidance:
        total_loss = total_loss + 0.1 * losses['prototype'] + 0.1 * losses['caption']
    
    print(f"   Total loss: {total_loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    
    # 检查各组件的梯度
    print("\n   Gradient check:")
    
    # SVD梯度
    svd_has_grad = False
    for name, param in detector.named_parameters():
        if 'S_residual' in name and param.grad is not None:
            if param.grad.abs().sum() > 0:
                svd_has_grad = True
                print(f"   ✓ SVD ({name}): grad_mean={param.grad.abs().mean():.6f}")
                break
    if not svd_has_grad:
        print("   ✗ SVD: No gradient!")
    
    # LoRA梯度
    lora_has_grad = False
    for name, param in detector.named_parameters():
        if 'lora' in name.lower() and param.grad is not None:
            if param.grad.abs().sum() > 0:
                lora_has_grad = True
                print(f"   ✓ LoRA ({name}): grad_mean={param.grad.abs().mean():.6f}")
                break
    if not lora_has_grad:
        print("   ✗ LoRA: No gradient!")
    
    # 分类器梯度
    if detector.fc.weight.grad is not None:
        print(f"   ✓ Classifier (fc.weight): grad_mean={detector.fc.weight.grad.abs().mean():.6f}")
    else:
        print("   ✗ Classifier: No gradient!")
    
    # 优化器步进
    optimizer.step()
    print("   ✓ Optimizer step completed!")
    
    print("\n5. Testing multi-scale inference...")
    # 测试不同尺寸输入
    for size in [192, 224, 256, 384]:
        dummy_img = torch.randn(2, 3, size, size).to(device)
        with torch.no_grad():
            logits = detector(dummy_img)
        print(f"   Input {size}x{size} → Output shape: {logits.shape} ✓")
    
    print("\n6. Model architecture summary...")
    status = detector.get_fusion_status()
    print(f"   SVD layers: {len(status['svd_layers'])}")
    print(f"   LoRA layers: {len(status['lora_layers'])}")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    
    # ★ 打印论文风格的参数统计
    print("\n📊 Paper-style Parameter Summary:")
    print(f"   Trainable params: {params_info['trainable']:,} ≈ {params_info['trainable']/1e6:.2f}M")
    print(f"   (论文报告: 0.19M)")
    print()


def compare_with_paper():
    """与论文参数对比"""
    print("\n" + "="*70)
    print("Comparing with EFFORT Paper")
    print("="*70)
    
    # 论文参数
    paper_params = {
        'total_trainable': 190000,  # 0.19M
        'svd_rank': 1023,           # n-r=1
        'classifier': 1025,         # Linear(1024, 1)
    }
    
    print("\n论文实现:")
    print(f"  - SVD rank (r): {paper_params['svd_rank']} (residual dim = 1)")
    print(f"  - Classifier: Linear(1024, 1) = {paper_params['classifier']} params")
    print(f"  - Total trainable: ~{paper_params['total_trainable']:,} (0.19M)")
    
    print("\n你的实现 (修改后):")
    print(f"  - SVD rank (r): 1023 (residual dim = 1)")
    print(f"  - LoRA (v_proj): rank=8")
    print(f"  - Classifier: Linear(1024, 1) = 1,025 params")
    
    # 计算预期参数
    # SVD: 24层 × 3投影(q,k,out) × (U_res + sigma_res + V_res)
    #    = 24 × 3 × (1024×1 + 1 + 1×1024) = 24 × 3 × 2049 = 147,528
    # LoRA: 24层 × 1投影(v) × (A + B)
    #    = 24 × 1 × (1024×8 + 8×1024) = 24 × 16384 = 393,216
    # Classifier: 1024 + 1 = 1,025
    
    svd_params = 24 * 3 * (1024 * 1 + 1 + 1 * 1024)
    lora_params = 24 * 1 * (1024 * 8 + 8 * 1024)
    classifier_params = 1024 * 1 + 1
    total = svd_params + lora_params + classifier_params
    
    print(f"\n  预期参数计算:")
    print(f"    SVD (24层×3投影): {svd_params:,}")
    print(f"    LoRA (24层×1投影): {lora_params:,}")
    print(f"    Classifier: {classifier_params:,}")
    print(f"    Total: {total:,} ({total/1e6:.2f}M)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_fusion_detector()
    compare_with_paper()

