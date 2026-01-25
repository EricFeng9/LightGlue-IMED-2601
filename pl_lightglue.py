
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from loguru import logger
import cv2

from lightglue import LightGlue, SuperPoint
from gen_data_enhance import apply_domain_randomization
# 注意：我们这里不直接导入 src 中的绘图工具，因为 src 目录可能缺失
# 可视化逻辑将在 train_lightglue.py 的 Validation Callback 中处理

class PL_LightGlue(pl.LightningModule):
    """
    LightGlue 的 PyTorch Lightning 封装类，用于训练。
    包含特征提取 (SuperPoint, 冻结) 和 特征匹配 (LightGlue, 可训练)。
    """
    def __init__(self, config):
        """
        初始化模型。
        Args:
            config: 配置对象或字典。
                    预期包含:
                    - LR: 学习率
                    - TRAINVAL_DATA_SOURCE: 数据集名称 (用于日志)
                    - MATCHING: 匹配相关配置 (input_dim, descriptor_dim 等)
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # 1. 特征提取器 (Feature Extractor) - 使用 SuperPoint
        # 注意：这里我们使用 .eval() 并冻结参数，因为我们只微调 LightGlue 的匹配逻辑
        # SuperPoint 已经具有很强的角点检测能力，能提取通用的几何特征
        self.extractor = SuperPoint(max_num_keypoints=2048).eval()
        for param in self.extractor.parameters():
            param.requires_grad = False
            
        # 2. 匹配器 (Matcher) - 使用 LightGlue
        # 加载默认配置，如果 config 中有 MATCHING 项则覆盖
        lg_conf = {
            "features": "superpoint",
            "input_dim": 256,
            "descriptor_dim": 256,
            "depth_confidence": -1, # 训练时禁用早停 (Early Stopping)，保证计算图稳定
            "width_confidence": -1, # 禁用宽度剪枝
            "filter_threshold": 0.1,
            "flash": False # 根据环境选择是否开启 Flash Attention
        }
        if hasattr(config, "MATCHING"):
             lg_conf.update(config.MATCHING)
             
        self.matcher = LightGlue(**lg_conf)
        
        # 损失权重
        self.nll_weight = 1.0

    def configure_optimizers(self):
        """配置优化器"""
        # 从配置中获取学习率，默认为 1e-4
        lr = self.config.TRAINER.TRUE_LR if hasattr(self.config, 'TRAINER') else 1e-4
        
        # 只优化 matcher (LightGlue) 的参数
        optimizer = torch.optim.Adam(self.matcher.parameters(), lr=lr)
        
        # 学习率调度器 (ReduceLROnPlateau): 当 val_loss 不再下降时减小学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # 监控验证集 Loss
            },
        }

    def forward(self, batch):
        """
        前向传播逻辑
        1. 提取特征 (如果 batch 中没有预提取好的特征)
        2. 构建 LightGlue 需要的输入字典
        3. 调用 LightGlue 进行匹配
        """
        # 提取特征 (extractor 不计算梯度)
        with torch.no_grad():
            if 'keypoints0' not in batch:
                feats0 = self.extractor({'image': batch['image0']})
                feats1 = self.extractor({'image': batch['image1']})
                # 将提取的特征存回 batch，方便后续使用
                batch = {**batch, 
                         'keypoints0': feats0['keypoints'], 'descriptors0': feats0['descriptors'], 'scores0': feats0['scores'],
                         'keypoints1': feats1['keypoints'], 'descriptors1': feats1['descriptors'], 'scores1': feats1['scores']}
        
        # 整理成 LightGlue 期望的输入格式
        data = {
            'image0': {
                'keypoints': batch['keypoints0'],
                'descriptors': batch['descriptors0'],
                'image': batch['image0']
            },
            'image1': {
                'keypoints': batch['keypoints1'],
                'descriptors': batch['descriptors1'],
                'image': batch['image1']
            }
        }
        
        # 调用 LightGlue
        # 注意：我们修改了 LightGlue 源码，使其在输出中包含 'log_assignment' (对数赋值矩阵)
        return self.matcher(data)

    def _compute_gt_matches(self, kpts0, kpts1, T_0to1, dist_th=3.0):
        """
        计算几何上的“Ground Truth”匹配对。
        原理：利用已知的变换矩阵 T_0to1，将图0的关键点变换到图1坐标系下，
              查找距离最近且小于阈值的点作为真值匹配。
        
        Args:
            kpts0: [B, M, 2] 图0关键点
            kpts1: [B, N, 2] 图1关键点
            T_0to1: [B, 3, 3] 从图0到图1的变换矩阵 (Truth)
            dist_th: 像素距离阈值 (默认 3.0 px)
        Returns:
            matches_gt: [B, M] 对应 kpts0 的真值匹配索引。如果无匹配则为 -1。
        """
        B, M, _ = kpts0.shape
        B, N, _ = kpts1.shape
        device = kpts0.device
        
        # 1. 将 kpts0 变换到 image1 的坐标系
        # 齐次坐标: [x, y, 1]
        kpts0_h = torch.cat([kpts0, torch.ones(B, M, 1, device=device)], dim=-1) # [B, M, 3]
        # 坐标变换: T * kpts0^T
        kpts0_warped_h = torch.matmul(kpts0_h, T_0to1.transpose(1, 2)) 
        # 归一化回 2D 坐标
        kpts0_warped = kpts0_warped_h[..., :2] / (kpts0_warped_h[..., 2:] + 1e-8) # [B, M, 2]
        
        # 2. 计算变换后的 kpts0 与 真实的 kpts1 之间的距离矩阵
        # dist: [B, M, N]
        dist = torch.cdist(kpts0_warped, kpts1)
        
        # 3. 寻找最近邻
        min_dist, matched_indices = torch.min(dist, dim=-1) # [B, M]
        
        # 4. 根据阈值过滤无效匹配
        mask = min_dist < dist_th
        
        # 生成 GT 匹配索引 (-1 代表无匹配/垃圾桶)
        matches_gt = torch.where(mask, matched_indices, torch.tensor(-1, device=device))
        
        return matches_gt

    def _compute_loss(self, outputs, kpts0, kpts1, T_0to1):
        """
        计算负对数似然损失 (Negative Log Likelihood Loss)
        目标是让 LightGlue 预测的匹配概率分布接近几何真值分布。
        """
        # 获取 LightGlue 输出的对数赋值矩阵 (Log Assignment Matrix)
        # 形状: [B, M+1, N+1]，最后一列/行代表“无匹配”(Dustbin)
        scores = outputs['log_assignment'] 
        
        # 计算几何真值匹配
        matches_gt = self._compute_gt_matches(kpts0, kpts1, T_0to1) # [B, M]
        
        B, M, N = scores.shape[0], scores.shape[1]-1, scores.shape[2]-1
        
        # 准备监督信号 (Targets)
        # matches_gt 中的值是 0..N-1 (匹配到的点索引) 或 -1 (无匹配)
        # 在 Assignment Matrix 中，索引 N 对应“无匹配”
        # 所以我们将 -1 映射为 N
        targets = matches_gt.clone()
        targets[targets == -1] = N
        
        # 计算 NLL Loss
        # 我们只看前 M 行 (对应 image0 的所有关键点)
        # scores[:, :M, :] 形状为 [B, M, N+1]
        # targets 形状为 [B, M]，包含了每个关键点应该匹配到的 image1 索引 (或 N)
        # F.nll_loss 需要 (N, C, ...) 格式的输入，所以我们需要转置 scores
        loss = F.nll_loss(scores[:, :M, :].transpose(1, 2), targets)
        
        return loss

    def training_step(self, batch, batch_idx):
        """
        训练步
        """
        # 1. 应用域随机化 (Domain Randomization)
        # 关键点：分别对 image0 和 image1 做独立的增强，打破它们之间的纹理/光照相关性
        # 这迫使模型去学习几何结构，而不是简单的纹理匹配
        batch['image0'] = apply_domain_randomization(batch['image0'])
        batch['image1'] = apply_domain_randomization(batch['image1'])
        
        # 2. 前向传播
        outputs = self(batch)
        
        # 3. 计算损失
        # 使用 dataset 提供的 T_0to1 (真值变换) 来监督
        loss = self._compute_loss(outputs, batch['keypoints0'], batch['keypoints1'], batch['T_0to1'])
        
        # 记录日志
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        验证步
        计算估计的单应矩阵 H_est，供回调函数计算 MACE/MSE 指标
        """
        outputs = self(batch)
        
        # 获取预测的匹配对
        matches0 = outputs['matches0'] # [B, M], image0 中每个点在 image1 中的匹配索引
        
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        
        B = kpts0.shape[0]
        H_ests = []
        
        # 为 Batch 中的每张图计算单应矩阵
        for b in range(B):
            m0 = matches0[b]
            valid = m0 > -1 # 有效匹配
            m_indices_0 = torch.where(valid)[0] # image0 的索引
            m_indices_1 = m0[valid]             # image1 的索引
            
            # 获取匹配点的坐标
            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()
            
            # 使用 RANSAC 计算单应矩阵
            if len(pts0) >= 4:
                try:
                    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
                    if H is None:
                         H = np.eye(3)
                except:
                    H = np.eye(3)
            else:
                H = np.eye(3)
            H_ests.append(H)
            
        return {
            'H_est': H_ests,
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches0': matches0,
            'matches1': outputs['matches1'],
            'loss': 0.0 # 占位符
        }
