"""
Training script for SuperPoint LoRA
训练 SuperPoint 描述子 LoRA，使其在生成图像和真实图像之间保持一致性

改进的训练策略：
1. 冻结 SuperPoint 的所有权重（共享编码器 + 关键点检测分支）
2. 只训练描述子分支的 LoRA 参数
3. 聚合 real 和 fake 的关键点，不要求两边关键点对齐
4. 从密集特征图上采样描述子，确保物理位置绝对对齐
5. 使用 InfoNCE Loss + 安全半径避免假负样本
6. 每个epoch验证并可视化，防止特征坍缩
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from scripts.v3.superpoint_detLora import SuperPointLoRA
from data.operation_pre_filtered_cffa_all.operation_pre_filtered_cffa_lora_dataset import CFFALoRADataset


class SuperPointLoRATrainer(pl.LightningModule):
    """SuperPoint LoRA 训练模块"""
    
    def __init__(self, 
                 lora_rank=16,
                 lora_alpha=1.0,
                 learning_rate=1e-4,
                 temperature=0.07,
                 max_num_keypoints=2048,
                 vessel_only=True,
                 safe_radius=8,
                 merge_threshold=2.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SuperPointLoRA(
            max_num_keypoints=max_num_keypoints,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            enable_lora=True
        )
        
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.vessel_only = vessel_only
        self.safe_radius = safe_radius
        self.merge_threshold = merge_threshold
        
        param_stats = self.model.count_parameters()
        logger.info(f"Model parameters: {param_stats['trainable']:,} trainable / {param_stats['total']:,} total "
                   f"({param_stats['trainable_ratio']:.2f}%)")
        logger.info(f"Safe radius: {safe_radius} pixels, Merge threshold: {merge_threshold} pixels")
    
    def forward(self, image):
        return self.model({'image': image})
    
    def get_dense_descriptors(self, image):
        """获取密集描述子特征图"""
        from kornia.color import rgb_to_grayscale
        
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        
        with torch.no_grad():
            x = self.model.superpoint.relu(self.model.superpoint.conv1a(image))
            x = self.model.superpoint.relu(self.model.superpoint.conv1b(x))
            x = self.model.superpoint.pool(x)
            x = self.model.superpoint.relu(self.model.superpoint.conv2a(x))
            x = self.model.superpoint.relu(self.model.superpoint.conv2b(x))
            x = self.model.superpoint.pool(x)
            x = self.model.superpoint.relu(self.model.superpoint.conv3a(x))
            x = self.model.superpoint.relu(self.model.superpoint.conv3b(x))
            x = self.model.superpoint.pool(x)
            x = self.model.superpoint.relu(self.model.superpoint.conv4a(x))
            x = self.model.superpoint.relu(self.model.superpoint.conv4b(x))
        
        with torch.no_grad():
            cDa = self.model.superpoint.relu(self.model.superpoint.convDa(x))
        
        if self.model.enable_lora:
            cDa = cDa + self.model.lora_Da(cDa)
        
        with torch.no_grad():
            descriptors = self.model.superpoint.convDb(cDa)
        
        if self.model.enable_lora:
            descriptors = descriptors + self.model.lora_Db(descriptors)
        
        descriptors = F.normalize(descriptors, p=2, dim=1)
        
        return descriptors
    
    def sample_descriptors_at_keypoints(self, dense_desc, keypoints, img_H, img_W):
        """在关键点位置采样描述子（优化版：直接从原图坐标映射）"""
        if len(keypoints) == 0:
            return torch.empty(0, dense_desc.shape[1], device=dense_desc.device)
        
        # 直接将原始图像坐标 [0, W-1], [0, H-1] 映射到 [-1, 1] 供 grid_sample 使用
        kpts_normalized = keypoints.clone()
        kpts_normalized[:, 0] = (kpts_normalized[:, 0] / (img_W - 1)) * 2 - 1
        kpts_normalized[:, 1] = (kpts_normalized[:, 1] / (img_H - 1)) * 2 - 1
        
        # [1, 1, N, 2] -> 适配 grid_sample 的 grid 格式
        grid = kpts_normalized.unsqueeze(0).unsqueeze(0)
        
        # align_corners=False 通常对特征图下采样的对应更加准确
        sampled = F.grid_sample(
            dense_desc,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        # sampled 形状为 [1, D, 1, N]，转换回 [N, D]
        sampled_desc = sampled.squeeze(2).squeeze(0).t()
        
        return sampled_desc
    

    def filter_keypoints_by_vessel(self, kpts, seg_mask):
        """根据血管分割图过滤关键点"""
        if len(kpts) == 0:
            return torch.tensor([], dtype=torch.long, device=kpts.device)
        
        seg_mask = seg_mask.squeeze(0)
        H, W = seg_mask.shape
        
        kpts_int = kpts.long()
        kpts_int[:, 0] = torch.clamp(kpts_int[:, 0], 0, W - 1)
        kpts_int[:, 1] = torch.clamp(kpts_int[:, 1], 0, H - 1)
        
        valid_mask = seg_mask[kpts_int[:, 1], kpts_int[:, 0]] > 0.5
        
        valid_indices = torch.where(valid_mask)[0]
        return valid_indices
    
    def compute_infonce_loss_with_safe_radius(self, desc_real, desc_fake, keypoints, temperature=0.07):
        """计算 InfoNCE Loss（带安全半径）"""
        N = len(desc_real)
        if N < 2:
            return torch.tensor(0.0, device=desc_real.device)
        
        sim_matrix = torch.matmul(desc_real, desc_fake.t()) / temperature
        
        dist_matrix = torch.cdist(keypoints, keypoints)
        safe_mask = dist_matrix >= self.safe_radius
        
        safe_mask.fill_diagonal_(True)
        
        sim_matrix = sim_matrix.masked_fill(~safe_mask, -1e4)
        
        labels = torch.arange(N, device=desc_real.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """训练步骤（优化版：只使用 Real 关键点，减少 VGG 推理次数）"""
        real_img = batch['real']
        fake_img = batch['fake']
        seg_mask = batch['seg']
        
        B, _, img_H, img_W = real_img.shape
        
        # 1. 带有梯度的密集描述子 (只跑2次 VGG)
        dense_desc_real = self.get_dense_descriptors(real_img)
        dense_desc_fake = self.get_dense_descriptors(fake_img)
        
        # 2. 冻结提取关键点：只提取 REAL 图像的关键点！(节省1次 VGG)
        with torch.no_grad():
            feats_real = self.forward(real_img)
        
        total_loss = 0.0
        num_valid_samples = 0
        total_keypoints = 0
        
        for i in range(B):
            kpts_real = feats_real['keypoints'][i]
            seg = seg_mask[i]
            
            # 过滤掉无效的 padding 点
            valid_real = (kpts_real.sum(dim=1) > 0)
            kpts_real = kpts_real[valid_real]
            
            if len(kpts_real) < 10:
                continue
                
            # 3. 核心：仅使用 Real 的关键点作为绝对物理锚点
            anchor_kpts = kpts_real
            
            # 根据血管分割图进行过滤
            if self.vessel_only:
                vessel_idx = self.filter_keypoints_by_vessel(anchor_kpts, seg)
                if len(vessel_idx) < 10:
                    continue
                anchor_kpts = anchor_kpts[vessel_idx]
                
            total_keypoints += len(anchor_kpts)
            
            # 4. 在 Real 和 Fake 的密集特征图上，根据相同的物理锚点采样描述子
            desc_real_sampled = self.sample_descriptors_at_keypoints(
                dense_desc_real[i:i+1], anchor_kpts, img_H, img_W
            )
            
            desc_fake_sampled = self.sample_descriptors_at_keypoints(
                dense_desc_fake[i:i+1], anchor_kpts, img_H, img_W
            )
            
            # 5. 计算带安全半径的 InfoNCE Loss
            loss = self.compute_infonce_loss_with_safe_radius(
                desc_real_sampled,
                desc_fake_sampled,
                anchor_kpts,
                temperature=self.temperature
            )
            
            total_loss += loss
            num_valid_samples += 1
        
        if num_valid_samples == 0:
            # 如果整个 batch 都没有有效点，返回带梯度的 0 (避免 DDP 报错)
            dummy_loss = (dense_desc_fake * 0).sum()
            return dummy_loss
        
        avg_loss = total_loss / num_valid_samples
        
        self.log('train/loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/num_keypoints', float(total_keypoints) / num_valid_samples, on_step=False, on_epoch=True)
        self.log('train/valid_samples', float(num_valid_samples), on_step=False, on_epoch=True)
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        real_img = batch['real']
        fake_img = batch['fake']
        seg_mask = batch['seg']
        
        B = real_img.shape[0]
        
        dense_desc_real = self.get_dense_descriptors(real_img)
        dense_desc_fake = self.get_dense_descriptors(fake_img)
        
        with torch.no_grad():
            feats_real = self.forward(real_img)
            feats_fake = self.forward(fake_img)
        
        total_loss = 0.0
        num_valid_samples = 0
        total_keypoints = 0
        
        # 保存第一个batch用于可视化
        if batch_idx == 0:
            return {
                'val_loss': torch.tensor(0.0, device=self.device),
                'vis_data': {
                    'real_img': real_img[0],
                    'fake_img': fake_img[0],
                    'seg_mask': seg_mask[0],
                    'feats_real': {k: v[0] for k, v in feats_real.items()},
                    'feats_fake': {k: v[0] for k, v in feats_fake.items()},
                    'dense_desc_real': dense_desc_real[0],
                    'dense_desc_fake': dense_desc_fake[0],
                }
            }
        
        for i in range(B):
            kpts_real = feats_real['keypoints'][i]
            kpts_fake = feats_fake['keypoints'][i]
            seg = seg_mask[i]
            
            valid_real = (kpts_real.sum(dim=1) > 0)
            valid_fake = (kpts_fake.sum(dim=1) > 0)
            
            kpts_real = kpts_real[valid_real]
            kpts_fake = kpts_fake[valid_fake]
            
            if len(kpts_real) == 0 and len(kpts_fake) == 0:
                continue
            
            merged_kpts = self.merge_and_deduplicate_keypoints(kpts_real, kpts_fake)
            
            if len(merged_kpts) == 0:
                continue
            
            if self.vessel_only:
                vessel_idx = self.filter_keypoints_by_vessel(merged_kpts, seg)
                
                if len(vessel_idx) < 10:
                    continue
                
                merged_kpts = merged_kpts[vessel_idx]
            
            if len(merged_kpts) < 10:
                continue
            
            total_keypoints += len(merged_kpts)
            
            desc_real_sampled = self.sample_descriptors_at_keypoints(
                dense_desc_real[i:i+1], merged_kpts
            )
            
            desc_fake_sampled = self.sample_descriptors_at_keypoints(
                dense_desc_fake[i:i+1], merged_kpts
            )
            
            loss = self.compute_infonce_loss_with_safe_radius(
                desc_real_sampled,
                desc_fake_sampled,
                merged_kpts,
                temperature=self.temperature
            )
            
            total_loss += loss
            num_valid_samples += 1
        
        if num_valid_samples == 0:
            avg_loss = torch.tensor(0.0, device=self.device)
        else:
            avg_loss = total_loss / num_valid_samples
        
        self.log('val/loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('val/num_keypoints', total_keypoints / max(num_valid_samples, 1), on_step=False, on_epoch=True, sync_dist=True, logger=True)
        
        return {'val_loss': avg_loss}
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


class VisualizationCallback(Callback):
    """可视化回调：每个epoch可视化训练效果"""
    
    def __init__(self, result_dir):
        super().__init__()
        self.result_dir = Path(result_dir)
        self.vis_dir = self.result_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束时可视化"""
        if trainer.sanity_checking:
            return
        
        epoch = trainer.current_epoch
        
        # 获取验证数据
        val_loader = trainer.val_dataloaders
        batch = next(iter(val_loader))
        
        # 移动到设备
        batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        pl_module.eval()
        with torch.no_grad():
            self.visualize_epoch(pl_module, batch, epoch)
        pl_module.train()
    
    def visualize_epoch(self, model, batch, epoch):
        """可视化一个epoch的结果"""
        real_img = batch['real'][0:1]  # [1, 1, H, W]
        fake_img = batch['fake'][0:1]
        seg_mask = batch['seg'][0:1]
        
        # 提取特征
        feats_real = model.forward(real_img)
        feats_fake = model.forward(fake_img)
        
        # 获取密集描述子
        dense_desc_real = model.get_dense_descriptors(real_img)
        dense_desc_fake = model.get_dense_descriptors(fake_img)
        
        # 转换为numpy
        real_np = (real_img[0, 0].cpu().numpy() * 255).astype(np.uint8)
        fake_np = (fake_img[0, 0].cpu().numpy() * 255).astype(np.uint8)
        seg_np = (seg_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        
        kpts_real = feats_real['keypoints'][0].cpu().numpy()
        kpts_fake = feats_fake['keypoints'][0].cpu().numpy()
        
        # 过滤padding
        valid_real = (kpts_real.sum(axis=1) > 0)
        valid_fake = (kpts_fake.sum(axis=1) > 0)
        kpts_real = kpts_real[valid_real]
        kpts_fake = kpts_fake[valid_fake]
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Real图像 + 关键点
        axes[0, 0].imshow(real_np, cmap='gray')
        axes[0, 0].scatter(kpts_real[:, 0], kpts_real[:, 1], c='red', s=10, alpha=0.5)
        axes[0, 0].set_title(f'Real Image + Keypoints ({len(kpts_real)})')
        axes[0, 0].axis('off')
        
        # 2. Fake图像 + 关键点
        axes[0, 1].imshow(fake_np, cmap='gray')
        axes[0, 1].scatter(kpts_fake[:, 0], kpts_fake[:, 1], c='blue', s=10, alpha=0.5)
        axes[0, 1].set_title(f'Fake Image + Keypoints ({len(kpts_fake)})')
        axes[0, 1].axis('off')
        
        # 3. 血管分割图
        axes[0, 2].imshow(seg_np, cmap='gray')
        axes[0, 2].set_title('Vessel Segmentation')
        axes[0, 2].axis('off')
        
        # 4. 描述子相似度热图（Real）
        desc_real_vis = dense_desc_real[0].cpu().numpy()  # [D, H/8, W/8]
        desc_real_norm = np.linalg.norm(desc_real_vis, axis=0)  # [H/8, W/8]
        axes[1, 0].imshow(desc_real_norm, cmap='viridis')
        axes[1, 0].set_title('Real Descriptor Norm')
        axes[1, 0].axis('off')
        
        # 5. 描述子相似度热图（Fake）
        desc_fake_vis = dense_desc_fake[0].cpu().numpy()
        desc_fake_norm = np.linalg.norm(desc_fake_vis, axis=0)
        axes[1, 1].imshow(desc_fake_norm, cmap='viridis')
        axes[1, 1].set_title('Fake Descriptor Norm')
        axes[1, 1].axis('off')
        
        # 6. 描述子差异热图
        desc_diff = np.abs(desc_real_norm - desc_fake_norm)
        im = axes[1, 2].imshow(desc_diff, cmap='hot')
        axes[1, 2].set_title('Descriptor Difference')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.suptitle(f'Epoch {epoch} - Validation Visualization', fontsize=16)
        plt.tight_layout()
        
        save_path = self.vis_dir / f'epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SuperPoint LoRA')
    
    parser.add_argument('--data_root', type=str, 
                       default='/data/student/Fengjunming/LightGlue/data/operation_pre_filtered_cffa_all')
    parser.add_argument('--modality', type=str, default='both', choices=['cf', 'fa', 'both'])
    parser.add_argument('--img_size', type=int, default=512)
    
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--max_num_keypoints', type=int, default=2048)
    parser.add_argument('--vessel_only', action='store_true', default=True)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--safe_radius', type=int, default=8)
    parser.add_argument('--merge_threshold', type=float, default=2.0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=str, default='0')
    
    parser.add_argument('--name', type=str, default='superpoint_lora')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    pl.seed_everything(42)
    
    result_dir = Path(f'results/superpoint_lora/{args.name}')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    logger.add(result_dir / 'train.log', level='INFO')
    
    logger.info(f"Training SuperPoint LoRA: {args.name}")
    logger.info(f"Arguments: {args}")
    
    train_dataset = CFFALoRADataset(
        root_dir=args.data_root,
        split='train',
        modality=args.modality,
        img_size=args.img_size
    )
    
    val_dataset = CFFALoRADataset(
        root_dir=args.data_root,
        split='val',
        modality=args.modality,
        img_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    model = SuperPointLoRATrainer(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_num_keypoints=args.max_num_keypoints,
        vessel_only=args.vessel_only,
        safe_radius=args.safe_radius,
        merge_threshold=args.merge_threshold
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=result_dir / 'checkpoints',
        filename='epoch{epoch:03d}-loss{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stop = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min',
        verbose=True,
        min_delta=0.0001,
        check_on_train_epoch_end=False
    )
    
    vis_callback = VisualizationCallback(result_dir)
    
    tb_logger = TensorBoardLogger(
        save_dir='logs',
        name=f'superpoint_lora/{args.name}'
    )
    
    if ',' in args.gpus:
        devices = [int(x) for x in args.gpus.split(',')]
    else:
        devices = [int(args.gpus)]
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=devices,
        callbacks=[checkpoint_callback, lr_monitor, early_stop, vis_callback],
        logger=tb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,  # 禁用sanity check，改用完整验证
        gradient_clip_val=1.0,
        precision='16-mixed'
    )
    
    # 训练前先跑一次完整验证作为baseline
    logger.info("Running initial validation as baseline...")
    trainer.validate(model, val_loader)
    
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    
    final_lora_path = result_dir / 'final_lora.pth'
    model.model.save_lora_weights(str(final_lora_path))
    logger.info(f"Final LoRA weights saved to {final_lora_path}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
