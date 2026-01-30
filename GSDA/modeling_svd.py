"""
SVD Residual Linear Layer - Complete Corrected Version
根据EFFORT论文和参考代码完整修正
modeling_svd.py

★ 新增:支持选择性SVD注入(为LoRA融合准备)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class SVDResidualLinear(nn.Module):
    """
    SVD Residual Linear Layer - EFFORT论文核心实现
    (保持不变,你的实现已经完美)
    """
    
    def __init__(self, original_linear: nn.Linear, r: Optional[int] = None, 
                 r_ratio: Optional[float] = None):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # ============ 1. 执行SVD分解 ============
        with torch.no_grad():
            W = original_linear.weight.data.float().cpu()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            
            full_rank = min(self.in_features, self.out_features)
            self.full_rank = full_rank
            
            if r is not None:
                actual_r = r
            elif r_ratio is not None:
                actual_r = int(full_rank * r_ratio)
            else:
                actual_r = int(full_rank * 0.99)
            
            if actual_r >= full_rank:
                actual_r = full_rank - 1
            if actual_r < 1:
                actual_r = 1
            
            self.r = actual_r
            residual_dim = full_rank - self.r
            
            print(f"    SVDResidualLinear: [{self.out_features}x{self.in_features}], "
                  f"r={self.r}, residual_dim={residual_dim}")
            
            # ============ 2. 分割主成分和残差 ============
            U_main = U[:, :self.r]
            S_main = S[:self.r]
            Vt_main = Vt[:self.r, :]
            W_main = U_main @ torch.diag(S_main) @ Vt_main
            
            U_residual = U[:, self.r:]
            S_residual = S[self.r:]
            Vt_residual = Vt[self.r:, :]
            
            # ============ 3. 注册主成分为buffer ============
            device = original_linear.weight.device
            self.register_buffer('weight_main', W_main.to(device))
            self.register_buffer('weight_original_fnorm_sq', 
                                 torch.tensor(torch.sum(S ** 2).item()))
            self.register_buffer('U_main_ref', U_main.to(device))
            self.register_buffer('V_main_ref', Vt_main.to(device))
            
            if original_linear.bias is not None:
                self.register_buffer('bias', original_linear.bias.data.clone())
            else:
                self.register_buffer('bias', None)
            
            # ============ 4. 注册残差为可训练参数 ============
            if residual_dim > 0:
                self.U_residual = nn.Parameter(U_residual.to(device))
                self.S_residual = nn.Parameter(S_residual.to(device))
                self.V_residual = nn.Parameter(Vt_residual.to(device))
                
                self.register_buffer('U_residual_init', U_residual.clone().to(device))
                self.register_buffer('S_residual_init', S_residual.clone().to(device))
                self.register_buffer('V_residual_init', Vt_residual.clone().to(device))
            else:
                self.register_parameter('U_residual', None)
                self.register_parameter('S_residual', None)
                self.register_parameter('V_residual', None)
                print(f"      Warning: No residual dimensions to train!")
    
    def forward(self, x):
        """前向传播"""
        out_main = F.linear(x, self.weight_main, None)
        
        if self.S_residual is not None and self.S_residual.numel() > 0:
            W_residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            out_residual = F.linear(x, W_residual, None)
            out = out_main + out_residual
        else:
            out = out_main
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def get_weight(self):
        """获取当前完整权重矩阵"""
        if self.S_residual is not None and self.S_residual.numel() > 0:
            W_residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            return self.weight_main + W_residual
        else:
            return self.weight_main
    
    def compute_orthogonal_loss(self):
        """★ 论文公式4:完整版正交性损失"""
        if self.S_residual is None or self.S_residual.numel() == 0:
            return torch.tensor(0.0, device=self.weight_main.device)
        
        device = self.U_residual.device
        dtype = self.U_residual.dtype
        
        U_hat = torch.cat([self.U_main_ref, self.U_residual], dim=1)
        V_hat = torch.cat([self.V_main_ref, self.V_residual], dim=0)
        
        I_full = torch.eye(self.full_rank, device=device, dtype=dtype)
        
        U_orth_matrix = U_hat.t() @ U_hat
        U_orth_error = U_orth_matrix - I_full
        loss_U = torch.sum(U_orth_error ** 2)
        
        V_orth_matrix = V_hat @ V_hat.t()
        V_orth_error = V_orth_matrix - I_full
        loss_V = torch.sum(V_orth_error ** 2)
        
        return loss_U + loss_V
    
    def compute_orthogonal_loss_decomposed(self):
        """★ 分解版正交损失(计算效率更高)"""
        if self.S_residual is None or self.S_residual.numel() == 0:
            return torch.tensor(0.0, device=self.weight_main.device)
        
        device = self.U_residual.device
        dtype = self.U_residual.dtype
        residual_dim = self.full_rank - self.r
        
        U_cross = self.U_main_ref.t() @ self.U_residual
        loss_U_cross = torch.sum(U_cross ** 2)
        
        I_residual = torch.eye(residual_dim, device=device, dtype=dtype)
        U_self = self.U_residual.t() @ self.U_residual
        loss_U_self = torch.sum((U_self - I_residual) ** 2)
        
        V_cross = self.V_residual @ self.V_main_ref.t()
        loss_V_cross = torch.sum(V_cross ** 2)
        
        V_self = self.V_residual @ self.V_residual.t()
        loss_V_self = torch.sum((V_self - I_residual) ** 2)
        
        return loss_U_cross + loss_U_self + loss_V_cross + loss_V_self
    
    def compute_keepsv_loss(self):
        """★ 论文公式6:奇异值保持损失"""
        if self.S_residual is None or self.S_residual.numel() == 0:
            return torch.tensor(0.0, device=self.weight_main.device)
        
        main_fnorm_sq = torch.sum(self.weight_main ** 2)
        
        W_residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        residual_fnorm_sq = torch.sum(W_residual ** 2)
        
        cross_term = 2 * torch.sum(self.weight_main * W_residual)
        
        current_fnorm_sq = main_fnorm_sq + residual_fnorm_sq + cross_term
        
        loss = torch.abs(current_fnorm_sq - self.weight_original_fnorm_sq)
        
        return loss
    
    def get_trainable_params(self):
        """返回可训练参数数量"""
        if self.S_residual is None or self.S_residual.numel() == 0:
            return {'U': 0, 'S': 0, 'V': 0, 'total': 0, 'residual_dim': 0, 'ratio': 0}
        
        residual_dim = self.full_rank - self.r
        n_U = self.out_features * residual_dim
        n_S = residual_dim
        n_V = residual_dim * self.in_features
        total = n_U + n_S + n_V
        original = self.out_features * self.in_features
        
        return {
            'U': n_U,
            'S': n_S,
            'V': n_V,
            'total': total,
            'residual_dim': residual_dim,
            'ratio': total / original * 100
        }
    
    def extra_repr(self):
        info = self.get_trainable_params()
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r={self.r}, residual_dim={info.get("residual_dim", 0)}, '
                f'trainable_ratio={info.get("ratio", 0):.2f}%')


# ============================================================
# ★ 核心修改:支持选择性SVD注入
# ============================================================

def inject_svd_into_clip_vision(vision_model, 
                                 r: int,
                                 svd_target_modules: List[str] = ['q_proj', 'k_proj', 'out_proj'],  # ★ 新增参数
                                 include_mlp: bool = False,
                                 target_layers: Optional[List[int]] = None):
    """
    ★ 修正版:将SVD注入到CLIP Vision Model的指定自注意力层
    
    ★ 关键修改:支持选择性替换投影层,为LoRA融合预留v_proj
    
    Args:
        vision_model: CLIP的vision_model部分
        r: SVD保留的主成分数量
        svd_target_modules: ★ 要应用SVD的投影层列表
            - 默认 ['q_proj', 'k_proj', 'out_proj'] (为v_proj的LoRA预留空间)
            - 如果要全部替换,使用 ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        include_mlp: 是否也替换MLP层 (fc1, fc2)
        target_layers: 要替换的层索引列表,None表示全部
    
    Returns:
        vision_model: 转换后的模型
    """
    print(f"\n{'='*70}")
    print(f"Injecting SVD into CLIP Vision Model (Selective)")
    print(f"  r = {r} (absolute rank)")
    print(f"  svd_target_modules = {svd_target_modules}")  # ★ 打印目标模块
    print(f"  include_mlp = {include_mlp}")
    print(f"  target_layers = {target_layers if target_layers else 'all'}")
    print(f"{'='*70}")
    
    replaced_count = 0
    total_trainable = 0
    total_original = 0
    
    # 获取encoder layers
    if hasattr(vision_model, 'encoder'):
        encoder = vision_model.encoder
    else:
        encoder = vision_model
    
    if hasattr(encoder, 'layers'):
        layers = encoder.layers
    elif hasattr(encoder, 'layer'):
        layers = encoder.layer
    else:
        print("Warning: Could not find encoder layers")
        return vision_model
    
    num_layers = len(layers)
    print(f"  Found {num_layers} encoder layers")
    
    # 遍历每一层
    for layer_idx, layer in enumerate(layers):
        if target_layers is not None and layer_idx not in target_layers:
            continue
        
        print(f"\n  Layer {layer_idx}:")
        
        # ============ 替换Self-Attention投影层 ============
        if hasattr(layer, 'self_attn'):
            self_attn = layer.self_attn
            
            # ★ 只替换指定的投影层
            for proj_name in svd_target_modules:
                if hasattr(self_attn, proj_name):
                    original_layer = getattr(self_attn, proj_name)
                    
                    if isinstance(original_layer, nn.Linear) and not isinstance(original_layer, SVDResidualLinear):
                        original_params = original_layer.weight.numel()
                        total_original += original_params
                        
                        try:
                            svd_layer = SVDResidualLinear(original_layer, r=r)
                            setattr(self_attn, proj_name, svd_layer)
                            
                            trainable = svd_layer.get_trainable_params()['total']
                            total_trainable += trainable
                            replaced_count += 1
                            print(f"    ✓ Replaced {proj_name}")
                        except Exception as e:
                            print(f"    ✗ Failed to replace {proj_name}: {e}")
                else:
                    print(f"    ⚠ {proj_name} not found in self_attn")
            
            # ★ 显式跳过v_proj(如果不在目标列表中)
            if 'v_proj' not in svd_target_modules and hasattr(self_attn, 'v_proj'):
                print(f"    ⊗ Skipped v_proj (reserved for LoRA)")
        
        # ============ 替换MLP层(可选) ============
        if include_mlp and hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            for fc_name in ['fc1', 'fc2']:
                if hasattr(mlp, fc_name):
                    original_layer = getattr(mlp, fc_name)
                    
                    if isinstance(original_layer, nn.Linear) and not isinstance(original_layer, SVDResidualLinear):
                        original_params = original_layer.weight.numel()
                        total_original += original_params
                        
                        try:
                            svd_layer = SVDResidualLinear(original_layer, r=r)
                            setattr(mlp, fc_name, svd_layer)
                            
                            trainable = svd_layer.get_trainable_params()['total']
                            total_trainable += trainable
                            replaced_count += 1
                        except Exception as e:
                            print(f"      ⚠️ Failed to replace {fc_name}: {e}")
    
    compression_ratio = total_trainable / total_original * 100 if total_original > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"SVD Injection Complete!")
    print(f"  Replaced layers: {replaced_count}")
    print(f"  Original params in replaced layers: {total_original:,}")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"  Compression ratio: {compression_ratio:.2f}%")
    print(f"{'='*70}\n")
    
    return vision_model


# ============================================================
# 辅助函数(保持不变)
# ============================================================

def collect_svd_losses(model, loss_type: str = 'decomposed'):
    """收集模型中所有SVDResidualLinear层的损失"""
    total_ortho = 0.0
    total_keepsv = 0.0
    count = 0
    
    for module in model.modules():
        if isinstance(module, SVDResidualLinear):
            if loss_type == 'decomposed':
                ortho_loss = module.compute_orthogonal_loss_decomposed()
            else:
                ortho_loss = module.compute_orthogonal_loss()
            
            keepsv_loss = module.compute_keepsv_loss()
            
            total_ortho = total_ortho + ortho_loss
            total_keepsv = total_keepsv + keepsv_loss
            count += 1
    
    if count > 0:
        return {
            'ortho': total_ortho / count,
            'keepsv': total_keepsv / count,
            'count': count
        }
    else:
        device = next(model.parameters()).device
        return {
            'ortho': torch.tensor(0.0, device=device),
            'keepsv': torch.tensor(0.0, device=device),
            'count': 0
        }


def get_svd_layer_info(model):
    """获取模型中所有SVD层的详细信息"""
    info_list = []
    
    for name, module in model.named_modules():
        if isinstance(module, SVDResidualLinear):
            params = module.get_trainable_params()
            info_list.append({
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'r': module.r,
                'full_rank': module.full_rank,
                'residual_dim': params['residual_dim'],
                'trainable_params': params['total'],
                'trainable_ratio': params['ratio']
            })
    
    return info_list


# 别名
SVDLinear = SVDResidualLinear


if __name__ == "__main__":
    print("modeling_svd.py - Test Mode")
    print("Run test_svd_layer() or test_clip_integration() to verify")
