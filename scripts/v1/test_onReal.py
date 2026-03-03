import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pprint
from pathlib import Path
from loguru import logger
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.strategies import DDPStrategy
import logging
from types import SimpleNamespace

# 添加父目录到 sys.path 以支持导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入 LightGlue 相关模块
from lightglue import LightGlue, SuperPoint
from lightglue import viz2d

# 导入真实数据集
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset as CFFADataset_Train
from data.CFFA.cffa_dataset import CFFADataset as CFFADataset_Val

# 导入指标计算模块
from scripts.v1.metrics import (
    compute_homography_errors, 
    aggregate_metrics,
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)

# ==========================================
# 配置函数
# ==========================================
def get_default_config():
    """获取默认配置"""
    conf = SimpleNamespace()
    conf.TRAINER = SimpleNamespace()
    conf.TRAINER.CANONICAL_BS = 4
    conf.TRAINER.CANONICAL_LR = 1e-4
    conf.TRAINER.TRUE_LR = 1e-4
    conf.TRAINER.RANSAC_PIXEL_THR = 3.0
    conf.TRAINER.SEED = 66
    conf.TRAINER.WORLD_SIZE = 1
    conf.TRAINER.TRUE_BATCH_SIZE = 4
    conf.TRAINER.PLOT_MODE = 'evaluation'
    
    conf.MATCHING = {
        'features': 'superpoint',
        'input_dim': 256,
        'descriptor_dim': 256,
        'depth_confidence': -1,  # 训练时禁用早停
        'width_confidence': -1,
        'filter_threshold': 0.1,
        'flash': False
    }
    return conf

# ==========================================
# 工具函数
# ==========================================
def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
    """单应矩阵防爆锁"""
    if H is None:
        return False
    if np.isnan(H).any() or np.isinf(H).any():
        return False
    
    det = np.linalg.det(H[:2, :2])
    if det < scale_min or det > scale_max:
        return False
    
    if abs(H[2, 0]) > perspective_threshold or abs(H[2, 1]) > perspective_threshold:
        return False
    
    return True

def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分"""
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    if len(img1.shape) == 3:
        mask1 = np.any(img1 > 10, axis=2)
    else:
        mask1 = img1 > 0
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    valid_mask = mask1 & mask2
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img1, img2
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    filtered_img1[~valid_mask_cropped] = 0
    filtered_img2[~valid_mask_cropped] = 0
    return filtered_img1, filtered_img2

def compute_corner_error(H_est, H_gt, height, width):
    """计算四个角点的平均重投影误差（MACE）"""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace

def create_chessboard(img1, img2, grid_size=4):
    """创建棋盘格对比图"""
    H, W = img1.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

# ==========================================
# 辅助类: RealDatasetWrapper (格式转换)
# ==========================================
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        # 数据集返回的已是归一化到 [0, 1] 的 fix，和 [-1, 1] 的 moving
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
        # 转换为灰度图 [1, H, W]
        if fix_tensor.shape[0] == 3:
            fix_gray = 0.299 * fix_tensor[0] + 0.587 * fix_tensor[1] + 0.114 * fix_tensor[2]
            fix_gray = fix_gray.unsqueeze(0)
        else:
            fix_gray = fix_tensor
            
        if moving_gt_tensor.shape[0] == 3:
            moving_gray = 0.299 * moving_gt_tensor[0] + 0.587 * moving_gt_tensor[1] + 0.114 * moving_gt_tensor[2]
            moving_gray = moving_gray.unsqueeze(0)
        else:
            moving_gray = moving_gt_tensor
            
        if moving_original_tensor.shape[0] == 3:
            moving_orig_gray = 0.299 * moving_original_tensor[0] + 0.587 * moving_original_tensor[1] + 0.114 * moving_original_tensor[2]
            moving_orig_gray = moving_orig_gray.unsqueeze(0)
        else:
            moving_orig_gray = moving_original_tensor
        
        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)
        
        # 数据集内部计算的 T_0to1 是从 Moving 到 Fix 的变换
        # 但 LightGlue 默认输出是从 Image0(Fix) -> Image1(Moving) 的变换
        # 所以这里取逆
        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
            T_fix_to_moving = T_0to1
            
        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'T_0to1': T_fix_to_moving,
            'pair_names': (fix_name, moving_name),
            'dataset_name': 'MultiModal'
        }

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        # 测试阶段使用 CFFA 测试集
        script_dir = Path(__file__).parent.parent.parent
        
        test_data_dir = script_dir / 'data' / 'CFFA'
        test_base = CFFADataset_Val(root_dir=str(test_data_dir), split='val', mode='cf2fa')
        self.test_dataset = RealDatasetWrapper(test_base)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, **self.loader_params)

# ==========================================
# 核心模型: PL_LightGlue_Real
# ==========================================
class PL_LightGlue_Real(pl.LightningModule):
    """LightGlue 的 PyTorch Lightning 封装（用于真实数据训练）"""
    def __init__(self, config, result_dir=None):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.save_hyperparameters({'config': str(config)})
        
        # 1. 特征提取器 (SuperPoint) - 冻结
        self.extractor = SuperPoint(max_num_keypoints=2048).eval()
        for param in self.extractor.parameters():
            param.requires_grad = False
            
        # 2. 匹配器 (LightGlue) - 可训练
        lg_conf = config.MATCHING.copy()
        self.matcher = LightGlue(**lg_conf)
        
        # 用于控制是否强制可视化
        self.force_viz = False
        
        # 用于跨 batch 累积指标，在 on_test_epoch_end 中聚合
        self._test_step_errors = []  # list of errors for AUC calculation

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        lr = self.config.TRAINER.TRUE_LR
        optimizer = torch.optim.Adam(self.matcher.parameters(), lr=lr)
        
        # 使用 ReduceLROnPlateau 调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "combined_auc",  # 监控平均 AUC
            },
        }

    def forward(self, batch):
        """前向传播"""
        # 提取特征
        with torch.no_grad():
            if 'keypoints0' not in batch:
                feats0 = self.extractor({'image': batch['image0']})
                feats1 = self.extractor({'image': batch['image1']})
                batch.update({
                    'keypoints0': feats0['keypoints'], 
                    'descriptors0': feats0['descriptors'], 
                    'scores0': feats0['keypoint_scores'],
                    'keypoints1': feats1['keypoints'], 
                    'descriptors1': feats1['descriptors'], 
                    'scores1': feats1['keypoint_scores']
                })
        
        # LightGlue 匹配
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
        
        return self.matcher(data)

    def _compute_gt_matches(self, kpts0, kpts1, T_0to1, dist_th=3.0):
        """计算几何 Ground Truth 匹配对"""
        B, M, _ = kpts0.shape
        B, N, _ = kpts1.shape
        device = kpts0.device
        
        # 将 kpts0 变换到 image1 的坐标系
        kpts0_h = torch.cat([kpts0, torch.ones(B, M, 1, device=device)], dim=-1)
        kpts0_warped_h = torch.matmul(kpts0_h, T_0to1.transpose(1, 2))
        kpts0_warped = kpts0_warped_h[..., :2] / (kpts0_warped_h[..., 2:] + 1e-8)
        
        # 计算距离矩阵
        dist = torch.cdist(kpts0_warped, kpts1)
        
        # 寻找最近邻
        min_dist, matched_indices = torch.min(dist, dim=-1)
        
        # 根据阈值过滤
        mask = min_dist < dist_th
        matches_gt = torch.where(mask, matched_indices, torch.tensor(-1, device=device))
        
        return matches_gt

    def _compute_loss(self, outputs, kpts0, kpts1, T_0to1):
        """计算负对数似然损失"""
        scores = outputs['log_assignment']
        matches_gt = self._compute_gt_matches(kpts0, kpts1, T_0to1)
        
        B, M, N = scores.shape[0], scores.shape[1]-1, scores.shape[2]-1
        
        targets = matches_gt.clone()
        targets[targets == -1] = N
        
        target_log_probs = torch.gather(scores[:, :M, :], 2, targets.unsqueeze(2)).squeeze(2)
        loss = -target_log_probs.mean()
        
        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤（兼容 metrics.py）"""
        outputs = self(batch)
        
        # 计算测试损失
        loss = self._compute_loss(outputs, batch['keypoints0'], batch['keypoints1'], batch['T_0to1'])
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 获取预测的匹配对
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        
        B = kpts0.shape[0]
        H_ests = []
        
        # 构建用于 metrics.py 的数据格式
        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []
        
        # 为每张图计算单应矩阵
        for b in range(B):
            m0 = matches0[b]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[valid]
            
            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()
            
            # 保存匹配点（用于 metrics.py 计算 AUC）
            if len(pts0) > 0:
                mkpts0_f_list.append(torch.from_numpy(pts0).float())
                mkpts1_f_list.append(torch.from_numpy(pts1).float())
                m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long))
            
            if len(pts0) >= 4:
                try:
                    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, self.config.TRAINER.RANSAC_PIXEL_THR)
                    if H is None:
                        H = np.eye(3)
                except:
                    H = np.eye(3)
            else:
                H = np.eye(3)
            H_ests.append(H)
        
        # 构建 metrics.py 需要的 batch 格式
        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }
        
        # 使用 metrics.py 计算指标
        set_metrics_verbose(True)  # 测试时输出详细日志
        compute_homography_errors(metrics_batch, self.config)
        
        # 累积误差用于后续 AUC 计算
        if len(metrics_batch.get('t_errs', [])) > 0:
            self._test_step_errors.extend(metrics_batch['t_errs'])
        
        return {
            'H_est': H_ests,
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches0': matches0,
            'metrics_batch': metrics_batch
        }

    def on_test_epoch_start(self):
        """每个测试 epoch 开始时重置误差累积列表"""
        self._test_step_errors = []

    def on_test_epoch_end(self):
        """在模型自身 hook 中聚合并 log AUC（按照 metrics.py 的方式）"""
        if self._test_step_errors and len(self._test_step_errors) > 0:
            # 使用 metrics.py 的 error_auc 函数计算 AUC@5, AUC@10, AUC@20
            auc_dict = error_auc(self._test_step_errors, [5, 10, 20])
            auc5 = auc_dict.get('auc@5', 0.0)
            auc10 = auc_dict.get('auc@10', 0.0)
            auc20 = auc_dict.get('auc@20', 0.0)
            
            # 使用 metrics.py 的 compute_auc_rop 函数计算 mAUC
            mauc_dict = compute_auc_rop(self._test_step_errors, limit=25)
            mauc = mauc_dict.get('mAUC', 0.0)
        else:
            auc5 = auc10 = auc20 = mauc = 0.0
        
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        self.log('auc@5',        auc5,         on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@10',       auc10,        on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@20',       auc20,        on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('mAUC',         mauc,         on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('combined_auc', combined_auc, on_epoch=True, prog_bar=True,  logger=True, sync_dist=False)

# ==========================================
# 回调逻辑: TestCallback
# ==========================================
class TestCallback(Callback):
    def __init__(self, args, output_dir):
        super().__init__()
        self.args = args
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []
        self.total_samples = 0
        self.failed_samples = 0

        import csv
        self.csv_path = self.output_dir / "test_metrics.csv"
        with open(self.csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Batch", "Test Loss", "MSE", "MACE", "AUC@5", "AUC@10", "AUC@20", "mAUC", "Match Failure Rate"])

    def on_test_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []
        self.batch_metrics = []
        self.total_samples = 0
        self.failed_samples = 0

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        batch_mses, batch_maces = self._process_batch(trainer, pl_module, batch, outputs, batch_idx, save_images=True)
        self.epoch_mses.extend(batch_mses)
        self.epoch_maces.extend(batch_maces)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.total_samples == 0:
            logger.warning("没有收集到任何测试指标")
            return
        
        # 只在匹配成功的样本上计算 MSE 和 MACE
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses) if self.epoch_mses else 0.0
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else 0.0
        
        # 计算匹配失败率
        match_failure_rate = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0
        
        metrics = trainer.callback_metrics
        
        display_metrics = {
            'mse': avg_mse, 
            'mace': avg_mace,
            'match_failure_rate': match_failure_rate
        }
        
        # 直接从模型累积的 _test_step_errors 计算 AUC（避免 callback_metrics 时序问题）
        if hasattr(pl_module, '_test_step_errors') and pl_module._test_step_errors:
            errors = pl_module._test_step_errors
            auc_dict = error_auc(errors, [5, 10, 20])
            auc5 = auc_dict.get('auc@5', 0.0)
            auc10 = auc_dict.get('auc@10', 0.0)
            auc20 = auc_dict.get('auc@20', 0.0)
            
            mauc_dict = compute_auc_rop(errors, limit=25)
            mauc = mauc_dict.get('mAUC', 0.0)
        else:
            auc5 = auc10 = auc20 = mauc = 0.0
        
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        display_metrics['auc@5'] = auc5
        display_metrics['auc@10'] = auc10
        display_metrics['auc@20'] = auc20
        display_metrics['mAUC'] = mauc
        display_metrics['combined_auc'] = combined_auc
        
        if 'test_loss' in metrics:
            display_metrics['test_loss'] = metrics['test_loss'].item()
        
        inverse_mace = 1.0 / (1.0 + avg_mace) if avg_mace > 0 else 1.0
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        logger.info(f"测试总结 >> {metric_str}")
        logger.info(f"匹配成功样本数: {self.total_samples - self.failed_samples}/{self.total_samples}")
        
        # 保存总结报告
        summary_path = self.output_dir / "test_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"测试总结\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"测试损失: {display_metrics.get('test_loss', 0.0):.6f}\n")
            f.write(f"总样本数: {self.total_samples}\n")
            f.write(f"匹配成功样本数: {self.total_samples - self.failed_samples}\n")
            f.write(f"匹配失败样本数: {self.failed_samples}\n")
            f.write(f"匹配失败率: {match_failure_rate:.4f}\n")
            f.write(f"MSE (仅匹配成功): {avg_mse:.6f}\n")
            f.write(f"MACE (仅匹配成功): {avg_mace:.4f}\n")
            f.write(f"AUC@5: {auc5:.4f}\n")
            f.write(f"AUC@10: {auc10:.4f}\n")
            f.write(f"AUC@20: {auc20:.4f}\n")
            f.write(f"mAUC: {mauc:.4f}\n")
            f.write(f"Combined AUC: {combined_auc:.4f}\n")
            f.write(f"Inverse MACE: {inverse_mace:.6f}\n")
        
        logger.info(f"测试总结已保存到: {summary_path}")

    def _process_batch(self, trainer, pl_module, batch, outputs, batch_idx, save_images=False):
        batch_size = batch['image0'].shape[0]
        mses, maces = [], []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        
        rejected_count = 0
        
        for i in range(batch_size):
            self.total_samples += 1
            H_est = H_ests[i]
            
            # 判断是否匹配失败（单应矩阵无效或接近单位矩阵）
            is_match_failed = False
            if not is_valid_homography(H_est):
                H_est = np.eye(3)
                rejected_count += 1
                is_match_failed = True
            elif np.allclose(H_est, np.eye(3), atol=1e-3):
                is_match_failed = True
            
            # 检查匹配点数量
            if 'matches0' in outputs:
                m0 = outputs['matches0'][i].cpu()
                valid = m0 > -1
                num_matches = torch.sum(valid).item()
                if num_matches < 4:
                    is_match_failed = True
            
            if is_match_failed:
                self.failed_samples += 1
            
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            
            # 只在匹配成功的样本上计算 MSE 和 MACE
            if not is_match_failed:
                try:
                    res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                    mask = (res_f > 0)
                    mse = np.mean((res_f[mask].astype(np.float64) - orig_f[mask].astype(np.float64))**2) if np.any(mask) else 0.0
                except:
                    mse = 0.0
                mses.append(mse)
                maces.append(compute_corner_error(H_est, Ts_gt[i], h, w))
            
            if save_images:
                sample_name = f"batch{batch_idx:04d}_sample{i:02d}_{Path(batch['pair_names'][0][i]).stem}_vs_{Path(batch['pair_names'][1][i]).stem}"
                save_path = self.output_dir / sample_name
                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / "fix.png"), img0)
                cv2.imwrite(str(save_path / "moving_original.png"), img1)
                cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
                cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)
                
                # 绘制关键点和匹配
                img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                
                if 'kpts0' in outputs and 'kpts1' in outputs:
                    kpts0_np = outputs['kpts0'][i].cpu().numpy()
                    kpts1_np = outputs['kpts1'][i].cpu().numpy()
                    
                    # 绘制所有关键点（白色）
                    for pt in kpts0_np:
                        cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    for pt in kpts1_np:
                        cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    
                    # 绘制匹配点（红色）
                    if 'matches0' in outputs:
                        m0 = outputs['matches0'][i].cpu()
                        valid = m0 > -1
                        m_indices_0 = torch.where(valid)[0].numpy()
                        m_indices_1 = m0[valid].numpy()
                        
                        for idx0 in m_indices_0:
                            pt = kpts0_np[idx0]
                            cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        for idx1 in m_indices_1:
                            pt = kpts1_np[idx1]
                            cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        
                        # 使用 viz2d 绘制匹配连线
                        try:
                            fig = plt.figure(figsize=(12, 6))
                            viz2d.plot_images([img0, img1])
                            if len(m_indices_0) > 0:
                                viz2d.plot_matches(kpts0_np[m_indices_0], kpts1_np[m_indices_1], color='lime', lw=0.5)
                            plt.savefig(str(save_path / "matches.png"), bbox_inches='tight', dpi=100)
                            plt.close(fig)
                        except Exception as e:
                            logger.warning(f"绘制匹配图失败: {e}")
                
                cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
                cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)
                
                try:
                    cb = create_chessboard(img1_result, img0)
                    cv2.imwrite(str(save_path / "chessboard.png"), cb)
                except:
                    pass
                
                # 保存单样本指标
                with open(save_path / "metrics.txt", "w") as f:
                    f.write(f"Match Failed: {is_match_failed}\n")
                    if not is_match_failed:
                        f.write(f"MSE: {mses[-1]:.6f}\n")
                        f.write(f"MACE: {maces[-1]:.4f}\n")
                    else:
                        f.write(f"MSE: N/A (match failed)\n")
                        f.write(f"MACE: N/A (match failed)\n")
                    f.write(f"Matches: {len(m_indices_0) if 'matches0' in outputs else 0}\n")
        
        if rejected_count > 0:
            logger.info(f"防爆锁触发: {rejected_count}/{batch_size} 个样本的单应矩阵被重置为单位矩阵")
        
        return mses, maces

# ==========================================
# 参数解析和主函数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="LightGlue CFFA Real-Data Testing")
    parser.add_argument('--name', '-n', type=str, required=True, help='训练模型名称（用于定位checkpoint）')
    parser.add_argument('--test_name', type=str, default='test_results', help='测试名称（用于指定输出子目录）')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径（默认使用best_checkpoint）')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='测试数据集划分')
    return parser.parse_args()

def main():
    args = parse_args()
    args.mode = 'cffa'  # 固定为 CFFA 模式
    
    config = get_default_config()
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)
    
    # 确定checkpoint路径
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # 默认使用best_checkpoint
        ckpt_path = Path(f"results/lightglue_{args.mode}/{args.name}/best_checkpoint/model.ckpt")
    
    if not ckpt_path.exists():
        logger.error(f"检查点不存在: {ckpt_path}")
        logger.info(f"请确保训练模型存在，或使用 --checkpoint 指定有效的检查点路径")
        return
    
    logger.info(f"加载检查点: {ckpt_path}")
    
    # 设置输出目录
    output_dir = Path(f"results/lightglue_{args.mode}/{args.name}/{args.test_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "test_log.txt"
    
    # 配置日志
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="w", backtrace=True, diagnose=False)
    logger.info(f"日志将同时保存到: {log_file}")
    
    # 设置环境变量，让 metrics.py 也写入日志文件
    os.environ['LOFTR_LOG_FILE'] = str(log_file)
    
    # GPU 配置
    if ',' in str(args.gpus):
        gpus_list = [int(x) for x in args.gpus.split(',')]
        _n_gpus = len(gpus_list)
    else:
        try:
            gpus_list = [int(args.gpus)]
            _n_gpus = 1
        except:
            gpus_list = 'auto'
            _n_gpus = 1
    
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    logger.info(f"GPU 配置: devices={gpus_list}, num_gpus={_n_gpus}")
    logger.info(f"测试名称: {args.test_name}")
    logger.info(f"测试数据集划分: {args.split}")
    logger.info(f"输出目录: {output_dir}")
    
    # 从检查点加载模型
    model = PL_LightGlue_Real.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        result_dir=str(output_dir)
    )
    model.eval()
    
    # 初始化数据模块
    data_module = MultimodalDataModule(args, config)
    
    # TensorBoard 日志
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=f"lightglue_test_{args.name}_{args.test_name}")
    
    # Trainer 配置
    trainer_kwargs = {
        'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
        'devices': gpus_list,
        'callbacks': [TestCallback(args, output_dir)],
        'logger': tb_logger,
    }
    
    # 只有在多 GPU 时才添加 strategy
    if _n_gpus > 1:
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    logger.info(f"开始测试 (模型: {args.name} | 测试集: CFFA 真实数据 {args.split} split)")
    trainer.test(model, datamodule=data_module)
    
    logger.info(f"测试完成! 结果已保存到: {output_dir}")

if __name__ == '__main__':
    main()


