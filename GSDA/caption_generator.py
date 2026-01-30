"""
Image Caption Generation for C2P
Based on CLIP features + GPT-2 decoder (ClipCap model)
This provides detailed text descriptions for low-rank learning in C2P
caption_generator.py
修正要点：
1. 智能处理维度不匹配的clip_project层（部分加载权重）
2. 正确处理None输入和错误情况
3. 支持可选的detection transform（fc_path可以是None）
4. 添加ImageCaptionGenerator别名以兼容train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel, CLIPProcessor, CLIPModel
import PIL.Image
import skimage.io as io
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class MLP(nn.Module):
    """Multi-layer perceptron for feature projection"""
    
    def __init__(self, sizes: tuple, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ClipCaptionModel(nn.Module):
    """
    ClipCap model: CLIP image features -> GPT-2 text generation
    ★修改：支持任意 CLIP 特征维度
    """
    
    def __init__(self, prefix_length: int, clip_length: int = 768, prefix_size: int = 768):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.clip_length = clip_length  # ★动态 CLIP 特征长度
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        
        # ★使用动态 clip_length
        self.clip_project = MLP(
            (
                clip_length,  # ★改为动态
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            )
        )
    
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


def generate_caption(model, tokenizer, tokens=None, prompt=None, embed=None, 
                     entry_count=1, entry_length=67, top_p=0.8, temperature=1.0, 
                     stop_token: str = "."):
    """
    Generate text caption from CLIP image features
    """
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
            
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                
                if stop_token_index == next_token.item():
                    break
            
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)
    
    return generated_list[0]


class CaptionGenerator:
    """
    Complete pipeline for generating image captions
    ★修正版：支持任意 CLIP 模型 + 智能权重加载 + 可选 detection transform
    """
    
    def __init__(self, 
                 clip_model_name='openai/clip-vit-large-patch14',
                 clipcap_model_path='/iead/svd-c2p/coco_weights.pt',
                 fc_path=None,  # ★改为 None，默认禁用 detection transform
                 prefix_length=10,
                 device='cuda:0'):
        """
        Args:
            clip_model_name: CLIP model name or path
            clipcap_model_path: ClipCap pretrained weights path
            fc_path: Path or URL to detection FC layer weights
                     ★设为 None 可跳过 detection transform
            prefix_length: GPT-2 prefix length
            device: Computing device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.prefix_length = prefix_length
        self.fc_path = fc_path  # ★可以是 None
        
        print(f"Initializing Caption Generator on {self.device}")
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.clip_model, self.clip_processor = self._load_clip_model(clip_model_name)
        
        # ★获取 CLIP 特征维度
        self.clip_dim = self.clip_model.config.projection_dim  # 768 for ViT-L/14
        print(f"CLIP feature dimension: {self.clip_dim}")
        
        # Load ClipCap model
        print("Loading ClipCap model...")
        self.clipcap_model, self.tokenizer = self._load_clipcap_model(
            clipcap_model_path, prefix_length
        )
        
        # ★提示是否使用 detection transform
        if self.fc_path is None:
            print("⚠ Detection transform disabled (fc_path=None)")
        else:
            print(f"✓ Detection transform enabled: {self.fc_path}")
        
        print("✓ Caption Generator initialized successfully")
    
    def _load_clip_model(self, clip_name):
        """Load CLIP vision model"""
        clipmodel = CLIPModel.from_pretrained(clip_name)
        processor = CLIPProcessor.from_pretrained(clip_name)
        
        # Remove text components (not needed)
        del clipmodel.text_model 
        del clipmodel.text_projection 
        del clipmodel.logit_scale
        
        clipmodel = clipmodel.to(self.device)
        clipmodel.eval()
        
        return clipmodel, processor
    
    def _load_clipcap_model(self, model_path, prefix_length):
        """
        Load ClipCap model with shape compatibility check
        ★修正版：智能处理维度不匹配的层（部分加载权重）
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # ★创建模型（匹配当前 CLIP 维度）
        model = ClipCaptionModel(
            prefix_length, 
            clip_length=self.clip_dim,  # 使用当前CLIP的维度
            prefix_size=self.clip_dim
        )
        
        # ★加载预训练权重
        try:
            if model_path.startswith("http"):
                pretrained = torch.hub.load_state_dict_from_url(
                    model_path, map_location="cpu", progress=True
                )
            else:
                pretrained = torch.load(model_path, map_location="cpu")
        except Exception as e:
            print(f"⚠ Failed to load pretrained weights: {e}")
            print("  Using randomly initialized ClipCap model")
            pretrained = {}
        
        model_dict = model.state_dict()
        
        # ★分层处理（智能加载）
        loaded_keys = []
        skipped_keys = []
        partially_loaded_keys = []
        
        for k, v in pretrained.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    # 完全匹配，直接加载
                    model_dict[k] = v
                    loaded_keys.append(k)
                else:
                    # 维度不匹配
                    skipped_keys.append((k, v.shape, model_dict[k].shape))
                    
                    # ★对于 clip_project 层，尝试部分加载
                    if 'clip_project' in k and 'weight' in k:
                        # 截断或填充权重矩阵
                        min_in = min(v.shape[1], model_dict[k].shape[1])
                        min_out = min(v.shape[0], model_dict[k].shape[0])
                        
                        # 复制重叠部分
                        model_dict[k][:min_out, :min_in] = v[:min_out, :min_in]
                        partially_loaded_keys.append(
                            (k, v.shape, model_dict[k].shape, (min_out, min_in))
                        )
                        print(f"      Partially loaded {k}: {v.shape} → {model_dict[k].shape}")
                    
                    elif 'clip_project' in k and 'bias' in k:
                        # 对于bias也做部分加载
                        min_dim = min(v.shape[0], model_dict[k].shape[0])
                        model_dict[k][:min_dim] = v[:min_dim]
        
        # 加载state_dict
        model.load_state_dict(model_dict)
        
        # ★打印加载摘要
        print(f"\n  Weight Loading Summary:")
        print(f"    ✓ Fully loaded:     {len(loaded_keys)}/{len(pretrained)} layers")
        print(f"    ⚙ Partially loaded: {len(partially_loaded_keys)} layers")
        print(f"    ⚠ Skipped:          {len(skipped_keys) - len(partially_loaded_keys)} layers")
        
        if partially_loaded_keys:
            print(f"\n  Partial Loading Details:")
            for name, old_shape, new_shape, copied_shape in partially_loaded_keys:
                print(f"    {name}:")
                print(f"      Pretrained: {old_shape}")
                print(f"      Current:    {new_shape}")
                print(f"      Copied:     {copied_shape}")
        
        if len(skipped_keys) > len(partially_loaded_keys):
            remaining_skipped = [
                (k, old, new) for k, old, new in skipped_keys 
                if k not in [pk[0] for pk in partially_loaded_keys]
            ]
            print(f"\n  Fully Skipped Layers (first 5):")
            for name, old_shape, new_shape in remaining_skipped[:5]:
                print(f"    {name}: {old_shape} → {new_shape}")
        
        # 冻结 GPT-2 参数
        for param in model.gpt.parameters():
            param.requires_grad = False
        
        model = model.eval().to(self.device)
        
        return model, tokenizer
    
    def extract_clip_features(self, image_path_or_pil):
        """
        Extract CLIP features from image
        ★修正：正确处理 None 输入和错误情况
        """
        with torch.no_grad():
            # Load image
            if isinstance(image_path_or_pil, str):
                try:
                    image = PIL.Image.open(image_path_or_pil).convert('RGB')
                except Exception as e:
                    print(f"⚠ Failed to load image {image_path_or_pil}: {e}")
                    return None
            else:
                image = image_path_or_pil
            
            # ★检查图像是否有效
            if image is None:
                print("⚠ Received None image")
                return None
            
            # Process and extract features
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # ★正确移动到设备
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            
            image_features = self.clip_model.get_image_features(**inputs)
        
        return image_features
    
    def apply_detection_transform(self, image_features, cal_detection_feat=True):
        """
        Apply detection-guided transformation to CLIP features
        ★修正：正确处理 None 的 fc_path 和特征
        """
        # ★如果没有 FC 权重，直接返回
        if self.fc_path is None:
            return image_features
        
        # ★检查特征是否有效
        if image_features is None:
            return None
        
        # Load FC layer weights
        try:
            if self.fc_path.startswith("http"):
                fc_params = torch.hub.load_state_dict_from_url(
                    self.fc_path, map_location="cpu", progress=True
                )
            else:
                fc_params = torch.load(self.fc_path, map_location="cpu")
            
            weight = fc_params['fc.weight'].to(self.device)
            bias = fc_params['fc.bias'].to(self.device)
            
            with torch.no_grad():
                # Get detection probability
                prob = nnf.linear(image_features, weight, bias).sigmoid()
                
                # Apply detection-guided transformation
                if cal_detection_feat:
                    image_features = torch.mul(image_features, weight) + bias
                
                # Normalize
                image_features = image_features / image_features.norm(2, dim=-1, keepdim=True)
        except Exception as e:
            print(f"⚠ Detection transform failed: {e}")
            print("  Continuing with raw CLIP features...")
        
        return image_features
    
    def generate_caption(self, image_path_or_pil, use_detection_transform=False):
        """
        Generate caption for an image
        ★修正：正确处理所有错误情况
        """
        # Extract CLIP features
        image_features = self.extract_clip_features(image_path_or_pil)
        
        # ★检查特征是否有效
        if image_features is None:
            return "[Failed to extract features]"
        
        # Apply detection transformation (可选)
        if use_detection_transform and self.fc_path is not None:
            image_features = self.apply_detection_transform(image_features)
            if image_features is None:
                return "[Detection transform failed]"
        
        # Project to GPT-2 prefix space and generate caption
        try:
            with torch.no_grad():
                prefix_embed = self.clipcap_model.clip_project(image_features).reshape(
                    1, self.prefix_length, -1
                )
            
            # Generate caption
            caption = generate_caption(
                self.clipcap_model, 
                self.tokenizer, 
                embed=prefix_embed,
                entry_length=67,
                top_p=0.8,
                temperature=1.0
            )
        except Exception as e:
            print(f"⚠ Caption generation failed: {e}")
            caption = "[Generation failed]"
        
        return caption
    
    def generate_caption_batch(self, image_paths, use_detection_transform=False):
        """
        Generate captions for multiple images (batch processing)
        """
        captions = []
        for img_path in image_paths:
            caption = self.generate_caption(img_path, use_detection_transform)
            captions.append(caption)
        return captions


# ★添加别名以兼容 train.py 中的 ImageCaptionGenerator 导入
ImageCaptionGenerator = CaptionGenerator


if __name__ == '__main__':
    # Test caption generation
    print("\n" + "="*70)
    print("Testing Caption Generator (CORRECTED)")
    print("="*70 + "\n")
    
    # Initialize generator (without detection transform)
    generator = CaptionGenerator(
        clip_model_name='openai/clip-vit-large-patch14',
        clipcap_model_path='/iead/svd-c2p/coco_weights.pt',
        fc_path=None,  # ★禁用 detection transform
        device='cpu'
    )
    
    print("\n✓ Caption generator ready to use")
    print("\nUsage example:")
    print("  caption = generator.generate_caption('path/to/image.jpg')")
    print("  print(caption)")
    print("\n" + "="*70 + "\n")
