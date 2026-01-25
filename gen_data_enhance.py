"""
域随机化增强模块 (Domain Randomization Enhancement)

目的: 打破生成数据中不同模态之间的纹理相关性
     迫使模型学习真正的跨模态不变性特征 (几何结构) 而非表面纹理

参考: "Synthetic data in generalizable, learning-based neuroimaging"
增强策略:
  1. Intensity Bias Field (空间渐变亮度不均, 模拟光照不均匀)
  2. Speckle Noise (斑点噪声, OCT/超声成像常见)
  3. Poisson Noise (泊松噪声, 低光条件)
  4. Random Downsampling (小概率随机降采样, 模拟低质量扫描)
  5. Gamma/Contrast/Brightness (传统颜色空间变换)
  6. Gaussian Blur (高斯模糊)
  7. Gaussian Noise (高斯噪声)

来源: train_multimodal_v3_3.py (lines 56-210)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random


def generate_intensity_bias_field(H, W, device, num_control_points=4, strength_range=(0.7, 1.3)):
    """
    生成随机 Intensity Bias Field (空间渐变亮度场)
    模拟真实眼底成像中的光照不均匀性 (如边缘比中心暗)
    
    Args:
        H, W: 图像尺寸
        device: torch 设备
        num_control_points: 控制点数量 (越少越平滑)
        strength_range: 亮度乘数范围
    
    Returns:
        bias_field: [1, 1, H, W] 亮度乘法场
    """
    # 生成低分辨率随机场
    low_res = torch.rand(1, 1, num_control_points, num_control_points, device=device)
    # 缩放到 strength_range
    low_res = low_res * (strength_range[1] - strength_range[0]) + strength_range[0]
    # 上采样到原始分辨率 (双三次插值产生平滑渐变)
    bias_field = F.interpolate(low_res, size=(H, W), mode='bicubic', align_corners=False)
    return bias_field


def apply_speckle_noise(img, intensity_range=(0.0, 0.15)):
    """
    添加斑点噪声 (Speckle Noise)
    常见于 OCT、超声等相干成像系统
    
    公式: out = img + img * noise
    即噪声强度与局部亮度成正比
    """
    intensity = random.uniform(*intensity_range)
    if intensity > 0:
        noise = torch.randn_like(img) * intensity
        img = img + img * noise
    return img


def apply_poisson_noise(img, scale_range=(50, 200)):
    """
    添加泊松噪声 (Shot Noise)
    模拟低光条件下光子计数噪声
    
    Args:
        scale_range: 缩放因子范围, 越小噪声越强
    """
    scale = random.uniform(*scale_range)
    # 转换到 numpy 处理
    img_np = img.cpu().numpy()
    # 缩放到 [0, scale] 范围
    img_scaled = img_np * scale
    # 应用泊松噪声
    noisy = np.random.poisson(img_scaled.clip(0, None)).astype(np.float32)
    # 缩放回 [0, 1]
    noisy = noisy / scale
    return torch.from_numpy(noisy).to(img.device)


def apply_random_downsample(img, factor_range=(2, 4)):
    """
    随机降采样后再上采样回原尺寸
    模拟低分辨率或老旧设备产生的模糊图像
    """
    _, _, H, W = img.shape
    factor = random.randint(*factor_range)
    # 降采样
    low_res = F.interpolate(img, size=(H // factor, W // factor), mode='bilinear', align_corners=False)
    # 上采样回原尺寸
    restored = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)
    return restored


def apply_domain_randomization(img_tensor, gamma_range=(0.7, 1.5), 
                                contrast_range=(0.7, 1.3),
                                brightness_range=(-0.15, 0.15),
                                noise_std_range=(0.0, 0.05),
                                blur_prob=0.3, blur_kernel_range=(3, 7),
                                bias_field_prob=0.5,
                                speckle_prob=0.3,
                                poisson_prob=0.2,
                                downsample_prob=0.03):
    """
    对输入图像张量应用域随机化增强 (增强版)
    
    Args:
        img_tensor: [B, 1, H, W] 形状的图像张量，值域 [0, 1]
        gamma_range: Gamma 变换范围
        contrast_range: 对比度调整范围
        brightness_range: 亮度偏移范围
        noise_std_range: 高斯噪声标准差范围
        blur_prob: 应用模糊的概率
        blur_kernel_range: 模糊核大小范围 (奇数)
        bias_field_prob: 应用 Intensity Bias Field 的概率
        speckle_prob: 应用斑点噪声的概率
        poisson_prob: 应用泊松噪声的概率
        downsample_prob: 应用随机降采样的概率 (默认3%)
    
    Returns:
        增强后的图像张量 [B, 1, H, W]
    """
    B, _, H, W = img_tensor.shape
    device = img_tensor.device
    augmented = img_tensor.clone()
    
    for b in range(B):
        img = augmented[b:b+1]  # [1, 1, H, W]
        
        # 0. [新增] Intensity Bias Field (空间渐变亮度不均)
        if random.random() < bias_field_prob:
            bias_field = generate_intensity_bias_field(H, W, device)
            img = img * bias_field
        
        # 1. Gamma 变换 (模拟不同曝光条件)
        gamma = random.uniform(*gamma_range)
        img = torch.pow(img.clamp(min=1e-8), gamma)
        
        # 2. 对比度调整 (围绕均值缩放)
        contrast = random.uniform(*contrast_range)
        mean_val = img.mean()
        img = (img - mean_val) * contrast + mean_val
        
        # 3. 亮度偏移
        brightness = random.uniform(*brightness_range)
        img = img + brightness
        
        # 4. 高斯噪声
        noise_std = random.uniform(*noise_std_range)
        if noise_std > 0:
            noise = torch.randn_like(img) * noise_std
            img = img + noise
        
        # 5. [新增] 斑点噪声 (Speckle Noise)
        if random.random() < speckle_prob:
            img = apply_speckle_noise(img)
        
        # 6. [新增] 泊松噪声 (Shot Noise)
        if random.random() < poisson_prob:
            img = apply_poisson_noise(img)
        
        # 7. 高斯模糊 (随机应用)
        if random.random() < blur_prob:
            kernel_size = random.choice(range(blur_kernel_range[0], blur_kernel_range[1] + 1, 2))  # 只选奇数
            # 使用 OpenCV 进行模糊 (更高效)
            img_np = img[0, 0].cpu().numpy()
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
            img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        
        # 8. [新增] 随机降采样+上采样 (模拟低分辨率扫描, 小概率)
        if random.random() < downsample_prob:
            img = apply_random_downsample(img)
        
        # 裁剪到有效范围
        augmented[b:b+1] = img.clamp(0, 1)
    
    return augmented


def save_batch_visualization(img0_orig, img1_orig, img0_aug, img1_aug, 
                              save_dir, epoch, step, batch_size, vessel_mask=None):
    """
    保存一个 batch 的可视化结果,用于检查域随机化效果
    
    Args:
        img0_orig: 原始 fix 图像 [B, 1, H, W]
        img1_orig: 原始 moving 图像 [B, 1, H, W]
        img0_aug: 增强后 fix 图像 [B, 1, H, W]
        img1_aug: 增强后 moving 图像 [B, 1, H, W]
        save_dir: 保存目录
        epoch: 当前 epoch
        step: 当前 step (batch index)
        batch_size: batch 大小
        vessel_mask: 可选，血管掩码 [B, 1, H, W]
    """
    import os
    
    vis_dir = os.path.join(save_dir, f'epoch{epoch}_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    for b in range(min(batch_size, img0_orig.shape[0])):
        sample_dir = os.path.join(vis_dir, f'step{step}_sample{b}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 转换为 numpy 并保存
        fix_orig = (img0_orig[b, 0].cpu().numpy() * 255).astype(np.uint8)
        mov_orig = (img1_orig[b, 0].cpu().numpy() * 255).astype(np.uint8)
        fix_aug = (img0_aug[b, 0].cpu().numpy() * 255).astype(np.uint8)
        mov_aug = (img1_aug[b, 0].cpu().numpy() * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(sample_dir, 'fix_original.png'), fix_orig)
        cv2.imwrite(os.path.join(sample_dir, 'moving_original.png'), mov_orig)
        cv2.imwrite(os.path.join(sample_dir, 'fix_augmented.png'), fix_aug)
        cv2.imwrite(os.path.join(sample_dir, 'moving_augmented.png'), mov_aug)
        
        if vessel_mask is not None:
            vessel = (vessel_mask[b, 0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(sample_dir, 'vessel_mask.png'), vessel)
        
        # 创建对比图: 左边原始,右边增强
        comparison_fix = np.hstack([fix_orig, fix_aug])
        comparison_mov = np.hstack([mov_orig, mov_aug])
        cv2.imwrite(os.path.join(sample_dir, 'comparison_fix_orig_vs_aug.png'), comparison_fix)
        cv2.imwrite(os.path.join(sample_dir, 'comparison_moving_orig_vs_aug.png'), comparison_mov)
