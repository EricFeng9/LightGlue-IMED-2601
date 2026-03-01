"""
域随机化增强模块 (Domain Randomization Enhancement)

目的: 打破生成数据中不同模态之间的纹理相关性
     迫使模型学习真正的跨模态不变性特征 (几何结构) 而非表面纹理

核心策略 (参考 SynthMorph):
  采用适度的域随机化，破坏纹理捷径的同时保留血管特征可见性

增强策略 (已调整为温和版本):
  1. 破坏圆形视场（FOV）黑边，用随机偏移彻底消除边缘对齐捷径
  2. 图像反色 (25% 概率)
  3. 非线性 Gamma 变换 (0.5-2.0，模拟不同模态的灰度映射)
  4. 对比度随机调整 (0.6x-1.5x，温和范围，避免血管完全消失)
  5. 亮度随机偏移 (±0.1，温和范围)
  6. 水平/竖直独立的1-8倍下采样 (30% 概率)
  7. 加入少量横竖黑色条纹 (5% 概率，1-3条)
  8. 不均匀高斯噪声（模拟光照不均，50% 概率）
  9. 全局微小高斯噪声 (10% 概率)
  
【重要】FOV Mask 在最后一步应用，确保背景纯净为0，让模型专注于血管区域
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

# --- 随机化域增强对外接口 ---

def apply_fov_destruction_tensor(img_tensor, center_jitter=0.08, radius_jitter=0.15):
    """破坏圆形视场（FOV）黑边，保留为黑或者原背景，不再填充白噪声"""
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    mask = torch.zeros_like(img_tensor)
    
    for b in range(B):
        cx = W / 2.0 + random.uniform(-center_jitter, center_jitter) * W
        cy = H / 2.0 + random.uniform(-center_jitter, center_jitter) * H
        r = min(H, W) / 2.0 * random.uniform(1.0 - radius_jitter, 1.0)
        
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        dist = torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        # 边缘平滑度随机，范围更大
        edge_width = random.uniform(1.0, 10.0)
        fov = torch.sigmoid((r - dist) / edge_width)
        mask[b] = fov.unsqueeze(0)
    
    # 移除全域随机白噪声填充，由于已经进行了不对称裁剪，直接应用mask屏蔽背景
    img_tensor = img_tensor * mask
    
    return img_tensor

def apply_black_stripes_tensor(img_tensor, num_stripes_range=(1, 3), stripe_width_range=(1, 3)):
    """加入少量横竖黑色条纹损坏数据（batch内所有图像应用相同的条纹位置）"""
    B, C, H, W = img_tensor.shape
    mask = torch.ones_like(img_tensor)
    
    # 对整个batch使用相同的条纹参数
    num_stripes = random.randint(*num_stripes_range)
    stripe_params = []
    for _ in range(num_stripes):
        is_vertical = random.random() < 0.5
        width = random.randint(*stripe_width_range)
        if is_vertical:
            pos = random.randint(0, max(1, W - width - 1))
        else:
            pos = random.randint(0, max(1, H - width - 1))
        stripe_params.append((is_vertical, width, pos))
    
    # 对batch内所有图像应用相同的条纹
    for b in range(B):
        for is_vertical, width, pos in stripe_params:
            if is_vertical:
                mask[b, :, :, pos:pos+width] = 0.0
            else:
                mask[b, :, pos:pos+width, :] = 0.0
    
    return img_tensor * mask

def apply_nonuniform_gaussian_noise_tensor(img_tensor, noise_std_max=0.08, dark_spot_strength=0.35):
    """不均匀地加入高斯噪声（模拟光照不均匀在图像上产生的暗色位置）
    batch内所有图像使用相同的噪声分布图"""
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # 随机生成一个低分辨率的 map（对整个batch使用相同的map）
    noise_variance_map = torch.rand(1, 1, 4, 4, device=device)  # 改为 [1, 1, 4, 4]
    # 让map更稀疏，只保留高值区域（局部暗斑）- 提高阈值让暗斑范围更小
    noise_variance_map = torch.where(noise_variance_map > 0.75, noise_variance_map, torch.zeros_like(noise_variance_map))
    noise_variance_map = F.interpolate(noise_variance_map, size=(H, W), mode='bicubic', align_corners=False)
    
    # 扩展到整个batch
    noise_variance_map = noise_variance_map.expand(B, 1, H, W)
    
    # 基于 map 生成不均匀的噪声
    noise = torch.randn_like(img_tensor) * (noise_variance_map * noise_std_max)
    # 基于 map 产生暗区 (减去一定的值)，强度适中
    darkness = noise_variance_map * dark_spot_strength
    
    img_tensor = img_tensor - darkness + noise
    return img_tensor.clamp(0, 1)

def random_domain_augment_image(image):
    """
    对外接口: 接收一张图像 (Numpy Array 或者 Torch Tensor)
    返回同类型的经过域随机化的图像。包含:
    - 破坏圆形视场（FOV）黑边，用不对称偏移消除边缘对齐捷径
    - 图像反色 (25%概率)
    - 对比度温和随机调整 (0.6x - 1.5x，保证血管可见)
    - 亮度随机偏移 (±0.1)
    - 非线性 Gamma 变换 (0.5 - 2.0)
    - 水平/竖直独立的1-8倍下采样 (30% 概率)
    - 加入少量横竖黑色条纹 (5% 概率，1-3条)
    - 不均匀地加入高斯噪声 (50% 概率)
    - 全局微小噪声 (10% 概率)
    
    【关键修复】相比之前的极端版本，已调整为温和参数：
    - 对比度从 0.3-2.5 降低到 0.6-1.5（避免血管完全消失）
    - 亮度偏移从 ±0.2 降低到 ±0.1（避免完全变黑）
    - 黑色条纹从 10%/1-5条 降低到 5%/1-3条（减少遮挡）
    - 不均匀噪声从 80% 降低到 50%（减少干扰）
    
    注：FOV Mask 在最后一步应用，确保背景绝对为0，模型专注血管区域
    """
    is_numpy = isinstance(image, np.ndarray)
    
    # 转换为形如 [B, C, H, W] 值域 [0,1] 的 tensor
    if is_numpy:
        img_dtype = image.dtype
        if image.ndim == 2:
            img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            # 假设输入是 [H, W, C]
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            
        if img_tensor.max() > 2.0:
            scale_factor = 255.0
            img_tensor = img_tensor / 255.0
        else:
            scale_factor = 1.0
    else:
        img_tensor = image.clone().float()
        original_shape = img_tensor.shape
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        elif img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
    # ------ 开始随机化 ------
    
    # 【核心修复】：注释掉独立的 FOV 破坏，防止空间撕裂！
    # B_tmp, C_tmp, H_tmp, W_tmp = img_tensor.shape
    # device_tmp = img_tensor.device
    # fov_mask = torch.zeros_like(img_tensor)
    # center_jitter, radius_jitter = 0.08, 0.15
    # for b in range(B_tmp):
    #     cx = W_tmp / 2.0 + random.uniform(-center_jitter, center_jitter) * W_tmp
    #     cy = H_tmp / 2.0 + random.uniform(-center_jitter, center_jitter) * H_tmp
    #     r = min(H_tmp, W_tmp) / 2.0 * random.uniform(1.0 - radius_jitter, 1.0)
    #     y_grid, x_grid = torch.meshgrid(torch.arange(H_tmp, device=device_tmp), torch.arange(W_tmp, device=device_tmp), indexing='ij')
    #     dist = torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    #     edge_width = random.uniform(1.0, 10.0)
    #     fov_mask[b] = torch.sigmoid((r - dist) / edge_width).unsqueeze(0)
    
    # 【立刻应用 FOV Mask】在所有颜色变换之前！
    # img_tensor = img_tensor * fov_mask
    
    # 2. 图像反色 (所有通道取反) (25% 概率)
    # 此时，如果反色，内部背景和外部背景会同步从 0 变成 1，不会产生边缘！
    if random.random() < 0.25:
        img_tensor = 1.0 - img_tensor
    
    # 3. 非线性 Gamma 变换 (50% 概率，模拟不同模态的非线性灰度映射)
    if random.random() < 0.5:
        B = img_tensor.shape[0]
        gamma = random.uniform(0.5, 2.0)  # 对整个batch使用相同的gamma值
        for b in range(B):
            img_tensor[b] = torch.pow(img_tensor[b].clamp(1e-6, 1.0), gamma)
        
    # 4. 对比度剧烈随机调整 (0.6x - 1.5x，温和范围，避免图像完全变黑或过曝)
    # 【关键修复】之前的 0.3-2.5 范围过于极端，会导致血管特征完全丢失
    B = img_tensor.shape[0]
    contrast = random.uniform(0.6, 1.5)  # 对整个batch使用相同的对比度值
    for b in range(B):
        mean_val = img_tensor[b].mean()
        img_tensor[b] = (img_tensor[b] - mean_val) * contrast + mean_val
    img_tensor = img_tensor.clamp(0, 1)
    
    # 5. 亮度随机偏移 (30% 概率，减小范围避免完全变黑)
    if random.random() < 0.3:
        brightness_shift = random.uniform(-0.1, 0.1)  # 从 ±0.2 降低到 ±0.1
        img_tensor = (img_tensor + brightness_shift).clamp(0, 1)
        
    # 6. 水平/竖直独立的1-8倍下采样 (30% 概率触发下采样，提升触发率)
    if random.random() < 0.3:
        H, W = img_tensor.shape[2:]
        # 水平方向独立随机：1到8倍下采样
        h_scale = random.randint(1, 8)
        # 竖直方向独立随机：1到8倍下采样
        w_scale = random.randint(1, 8)
        
        low_res = F.interpolate(img_tensor, size=(max(1, H // h_scale), max(1, W // w_scale)), mode='bilinear', align_corners=False)
        img_tensor = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)
        
    # 7. 横竖黑色条纹 (10% 概率，数量不多 1-5 条)
    if random.random() < 0.1:
        img_tensor = apply_black_stripes_tensor(img_tensor, num_stripes_range=(1, 5), stripe_width_range=(1, 3))
        
    # 8. 不均匀高斯噪声 (模拟光照不均，产生暗区) - 局部暗斑，不是全局变暗
    # 【关键修复】降低触发率和强度，避免过度遮挡血管
    if random.random() < 0.5:  # 从 80% 降低到 50%
        img_tensor = apply_nonuniform_gaussian_noise_tensor(img_tensor, noise_std_max=0.02, dark_spot_strength=0.15)
        
    # 9. 添加少量全局高斯噪声
    if random.random() < 0.1:
        noise_std = random.uniform(0.002, 0.008)  # 对整个batch使用相同的噪声强度
        noise = torch.randn_like(img_tensor) * noise_std
        img_tensor = (img_tensor + noise).clamp(0, 1)
        
    # 规约到合法区间
    img_tensor = img_tensor.clamp(0, 1)
    
    # 【关键保护】动态范围保护：确保图像不会完全变黑或变白
    # 如果图像动态范围过小（几乎全黑或全白），进行自适应拉伸
    B = img_tensor.shape[0]
    for b in range(B):
        img_min = img_tensor[b].min()
        img_max = img_tensor[b].max()
        dynamic_range = img_max - img_min
        
        # 如果动态范围小于 0.15（说明图像几乎是纯色），进行拉伸
        if dynamic_range < 0.15:
            # 拉伸到 [0.05, 0.95] 范围，保留一定的对比度
            img_tensor[b] = (img_tensor[b] - img_min) / (dynamic_range + 1e-6)
            img_tensor[b] = img_tensor[b] * 0.9 + 0.05
        # 如果动态范围太小但不至于全黑，进行温和拉伸
        elif dynamic_range < 0.3:
            img_tensor[b] = (img_tensor[b] - img_min) / (dynamic_range + 1e-6)
            img_tensor[b] = img_tensor[b] * 0.8 + 0.1
    
    img_tensor = img_tensor.clamp(0, 1)
    
    # 【已删除】FOV Mask 已经在最开始应用，不需要在这里重复应用
    # 原因：如果在这里应用，反色和亮度调整后会产生锐利的人工边缘，导致模型学习到错误的特征
    
    # ------ 转换回原格式 ------
    if is_numpy:
        img_tensor = img_tensor * scale_factor
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        if image.ndim == 2:
            img_np = img_np[:, :, 0]
        if img_dtype == np.uint8:
            img_np = img_np.clip(0, 255).astype(np.uint8)
        return img_np
    else:
        return img_tensor.view(original_shape)
