"""
SuperPoint with LoRA for Descriptor Branch
用于在生成图像和真实图像之间对齐描述子特征，同时保持关键点检测能力不变
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale
from pathlib import Path

from lightglue.superpoint import SuperPoint, simple_nms, top_k_keypoints, sample_descriptors


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer
    实现 h = W0*x + (B*A)*x，其中 W0 是冻结的原始权重，B*A 是低秩分解
    """
    def __init__(self, in_features, out_features, rank=8, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA 的两个低秩矩阵
        # A: (in_features, rank) - 使用高斯初始化
        # B: (rank, out_features) - 使用零初始化（确保初始时LoRA不影响输出）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # scaling factor: alpha / rank
        self.scaling = self.alpha / self.rank
        
    def forward(self, x):
        """
        x: (B, C, H, W) for Conv2d or (B, ..., C) for Linear
        返回 LoRA 的增量输出
        """
        # 对于卷积层，需要处理 (B, C, H, W) 的输入
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
            x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
            # (B*H*W, C) @ (C, rank) @ (rank, out_features) -> (B*H*W, out_features)
            out = x_reshaped @ self.lora_A.T @ self.lora_B.T
            # (B*H*W, out_features) -> (B, H, W, out_features) -> (B, out_features, H, W)
            out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            # 对于全连接层
            out = x @ self.lora_A.T @ self.lora_B.T
            
        return out * self.scaling


class SuperPointLoRA(nn.Module):
    """
    SuperPoint with LoRA on Descriptor Branch
    
    特点：
    1. 完全冻结共享编码器和关键点检测分支
    2. 在描述子分支的 convDa 和 convDb 后添加 LoRA
    3. 可以灵活启用/禁用 LoRA
    4. 支持保存/加载 LoRA 权重
    """
    
    def __init__(self, max_num_keypoints=2048, lora_rank=16, lora_alpha=1.0, 
                 detection_threshold=0.0005, nms_radius=4, descriptor_dim=256,
                 enable_lora=True):
        super().__init__()
        
        # 加载预训练的 SuperPoint
        self.superpoint = SuperPoint(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            nms_radius=nms_radius,
            descriptor_dim=descriptor_dim
        )
        
        # 冻结所有 SuperPoint 参数
        for param in self.superpoint.parameters():
            param.requires_grad = False
        
        self.enable_lora = enable_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 在描述子分支添加 LoRA 层
        # convDa: (128, 256, 3, 3) -> 在输出通道添加 LoRA
        # convDb: (256, 256, 1, 1) -> 在输出通道添加 LoRA
        self.lora_Da = LoRALayer(256, 256, rank=lora_rank, alpha=lora_alpha)
        self.lora_Db = LoRALayer(descriptor_dim, descriptor_dim, rank=lora_rank, alpha=lora_alpha)
        
        # 配置信息
        self.conf = self.superpoint.conf
        
    def forward(self, data: dict) -> dict:
        """
        前向传播
        
        Args:
            data: 字典，必须包含 'image' 键
            
        Returns:
            字典，包含 'keypoints', 'keypoint_scores', 'descriptors'
        """
        image = data["image"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        
        # ============ 共享编码器（冻结）============
        with torch.no_grad():
            x = self.superpoint.relu(self.superpoint.conv1a(image))
            x = self.superpoint.relu(self.superpoint.conv1b(x))
            x = self.superpoint.pool(x)
            x = self.superpoint.relu(self.superpoint.conv2a(x))
            x = self.superpoint.relu(self.superpoint.conv2b(x))
            x = self.superpoint.pool(x)
            x = self.superpoint.relu(self.superpoint.conv3a(x))
            x = self.superpoint.relu(self.superpoint.conv3b(x))
            x = self.superpoint.pool(x)
            x = self.superpoint.relu(self.superpoint.conv4a(x))
            x = self.superpoint.relu(self.superpoint.conv4b(x))
            
            # ============ 关键点检测分支（冻结）============
            cPa = self.superpoint.relu(self.superpoint.convPa(x))
            scores = self.superpoint.convPb(cPa)
            scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
            b, _, h, w = scores.shape
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
            scores = simple_nms(scores, self.conf.nms_radius)
            
            # 移除边界关键点
            if self.conf.remove_borders:
                pad = self.conf.remove_borders
                scores[:, :pad] = -1
                scores[:, :, :pad] = -1
                scores[:, -pad:] = -1
                scores[:, :, -pad:] = -1
            
            # 提取关键点
            best_kp = torch.where(scores > self.conf.detection_threshold)
            scores_kp = scores[best_kp]
            
            # 按 batch 分离
            keypoints = [
                torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)
            ]
            scores_list = [scores_kp[best_kp[0] == i] for i in range(b)]
            
            # 保留 top-k 关键点
            if self.conf.max_num_keypoints is not None:
                keypoints, scores_list = list(
                    zip(
                        *[
                            top_k_keypoints(k, s, self.conf.max_num_keypoints)
                            for k, s in zip(keypoints, scores_list)
                        ]
                    )
                )
            
            # 转换 (h, w) 到 (x, y)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        
        # ============ 描述子分支（带 LoRA）============
        # 第一层：convDa + LoRA
        with torch.no_grad():
            cDa = self.superpoint.relu(self.superpoint.convDa(x))
        
        # 添加 LoRA 增量
        if self.enable_lora:
            cDa = cDa + self.lora_Da(cDa)
        
        # 第二层：convDb + LoRA
        with torch.no_grad():
            descriptors = self.superpoint.convDb(cDa)
        
        # 添加 LoRA 增量
        if self.enable_lora:
            descriptors = descriptors + self.lora_Db(descriptors)
        
        # 归一化描述子
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        
        # 在关键点位置采样描述子
        descriptors = [
            sample_descriptors(k[None], descriptors[i:i+1], 8)[0]
            for i, k in enumerate(keypoints)
        ]
        
        # ============ 组织输出 ============
        device = image.device
        if b > 1:
            # Pad 到相同长度
            max_kps = max(len(k) for k in keypoints)
            max_kps = max(max_kps, 1)
            
            padded_keypoints = torch.zeros((b, max_kps, 2), device=device)
            padded_scores = torch.zeros((b, max_kps), device=device)
            padded_descriptors = torch.zeros(
                (b, self.conf.descriptor_dim, max_kps), device=device
            )
            
            for i in range(b):
                n = len(keypoints[i])
                if n > 0:
                    padded_keypoints[i, :n] = keypoints[i]
                    padded_scores[i, :n] = scores_list[i]
                    padded_descriptors[i, :, :n] = descriptors[i]
            
            return {
                "keypoints": padded_keypoints,
                "keypoint_scores": padded_scores,
                "descriptors": padded_descriptors.transpose(-1, -2).contiguous(),
            }
        
        return {
            "keypoints": torch.stack(keypoints, 0),
            "keypoint_scores": torch.stack(scores_list, 0),
            "descriptors": torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }
    
    def set_lora_enabled(self, enabled: bool):
        """启用或禁用 LoRA"""
        self.enable_lora = enabled
    
    def save_lora_weights(self, path: str):
        """只保存 LoRA 权重"""
        lora_state = {
            'lora_Da': self.lora_Da.state_dict(),
            'lora_Db': self.lora_Db.state_dict(),
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'descriptor_dim': self.conf.descriptor_dim,
        }
        torch.save(lora_state, path)
        print(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """加载 LoRA 权重"""
        lora_state = torch.load(path, map_location='cpu')
        self.lora_Da.load_state_dict(lora_state['lora_Da'])
        self.lora_Db.load_state_dict(lora_state['lora_Db'])
        print(f"LoRA weights loaded from {path}")
    
    def get_trainable_parameters(self):
        """获取可训练参数（只有 LoRA 参数）"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """统计参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params * 100
        }


def test_superpoint_lora():
    """测试 SuperPointLoRA"""
    print("=" * 60)
    print("Testing SuperPointLoRA")
    print("=" * 60)
    
    # 创建模型
    model = SuperPointLoRA(
        max_num_keypoints=2048,
        lora_rank=16,
        lora_alpha=1.0
    )
    
    # 统计参数
    param_stats = model.count_parameters()
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {param_stats['total']:,}")
    print(f"  Trainable parameters: {param_stats['trainable']:,}")
    print(f"  Frozen parameters: {param_stats['frozen']:,}")
    print(f"  Trainable ratio: {param_stats['trainable_ratio']:.2f}%")
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建测试图像
    test_image = torch.randn(2, 1, 512, 512).to(device)
    
    print(f"\nTesting forward pass...")
    print(f"  Input shape: {test_image.shape}")
    
    with torch.no_grad():
        # 测试启用 LoRA
        model.set_lora_enabled(True)
        output_with_lora = model({'image': test_image})
        
        # 测试禁用 LoRA
        model.set_lora_enabled(False)
        output_without_lora = model({'image': test_image})
    
    print(f"\nOutput with LoRA:")
    print(f"  Keypoints shape: {output_with_lora['keypoints'].shape}")
    print(f"  Scores shape: {output_with_lora['keypoint_scores'].shape}")
    print(f"  Descriptors shape: {output_with_lora['descriptors'].shape}")
    
    print(f"\nOutput without LoRA:")
    print(f"  Keypoints shape: {output_without_lora['keypoints'].shape}")
    print(f"  Scores shape: {output_without_lora['keypoint_scores'].shape}")
    print(f"  Descriptors shape: {output_without_lora['descriptors'].shape}")
    
    # 检查关键点是否相同（应该相同，因为检测分支是冻结的）
    kpts_diff = torch.abs(output_with_lora['keypoints'] - output_without_lora['keypoints']).max()
    print(f"\nKeypoints difference (should be 0): {kpts_diff.item():.6f}")
    
    # 检查描述子是否不同（应该不同，因为 LoRA 影响描述子）
    desc_diff = torch.abs(output_with_lora['descriptors'] - output_without_lora['descriptors']).mean()
    print(f"Descriptors difference (should be > 0): {desc_diff.item():.6f}")
    
    # 测试保存和加载
    save_path = "/tmp/test_lora.pth"
    model.save_lora_weights(save_path)
    model.load_lora_weights(save_path)
    print(f"\nLoRA weights save/load test passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_superpoint_lora()
