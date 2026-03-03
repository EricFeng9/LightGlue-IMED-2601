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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import logging
from types import SimpleNamespace

# 添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入 LightGlue
from lightglue import LightGlue, SuperPoint
from lightglue import viz2d

# 导入数据集
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset

# 导入统一的测试/验证模块
from scripts.v2.test import UnifiedEvaluator

# 导入指标计算
from scripts.v1.metrics import (
    compute_homography_errors, 
    aggregate_metrics,
    set_metrics_verbose
)

# ==========================================
# 配置函数
# ==========================================
def get_default_config():
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
        'depth_confidence': -1,
        'width_confidence': -1,
        'filter_threshold': 0.1,
        'flash': False
    }
    return conf

# ==========================================
# 工具函数
# ==========================================
def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
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
    assert img1.shape[:2] == img2.shape[:2]
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
# 数据集包装类
# ==========================================
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, split_name='unknown'):
        self.base_dataset = base_dataset
        self.split_name = split_name
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
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
            'dataset_name': 'MultiModal',
            'split': self.split_name  # 添加 split 信息
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
        if stage == 'fit' or stage is None:
            script_dir = Path(__file__).parent.parent.parent
            
            # 训练集和验证集都使用 operation_pre_filtered_cffa 数据集
            data_dir = script_dir / 'data' / 'operation_pre_filtered_cffa'
            
            # 训练集
            train_base = CFFADataset(root_dir=str(data_dir), split='train', mode='cf2fa')
            self.train_dataset = RealDatasetWrapper(train_base, split_name='train')
            logger.info(f"训练集加载: {len(self.train_dataset)} 样本")
            
            # 验证集（使用测试集）
            val_base = CFFADataset(root_dir=str(data_dir), split='val', mode='cf2fa')
            self.val_dataset = RealDatasetWrapper(val_base, split_name='test')
            logger.info(f"验证集加载: {len(self.val_dataset)} 样本")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

# ==========================================
# 模型类: PL_LightGlue_Real
# ==========================================
class PL_LightGlue_Real(pl.LightningModule):
    def __init__(self, config, result_dir=None):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.save_hyperparameters({'config': str(config)})
        
        self.extractor = SuperPoint(max_num_keypoints=2048).eval()
        for param in self.extractor.parameters():
            param.requires_grad = False
            
        lg_conf = config.MATCHING.copy()
        self.matcher = LightGlue(**lg_conf)
        
        self.force_viz = False
        
        # 使用统一的评估器
        self.evaluator = UnifiedEvaluator(mode='real', config=config)

    def configure_optimizers(self):
        lr = self.config.TRAINER.TRUE_LR
        optimizer = torch.optim.Adam(self.matcher.parameters(), lr=lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "combined_auc",
                "strict": False,
            },
        }

    def forward(self, batch):
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
        B, M, _ = kpts0.shape
        B, N, _ = kpts1.shape
        device = kpts0.device
        
        kpts0_h = torch.cat([kpts0, torch.ones(B, M, 1, device=device)], dim=-1)
        kpts0_warped_h = torch.matmul(kpts0_h, T_0to1.transpose(1, 2))
        kpts0_warped = kpts0_warped_h[..., :2] / (kpts0_warped_h[..., 2:] + 1e-8)
        
        dist = torch.cdist(kpts0_warped, kpts1)
        min_dist, matched_indices = torch.min(dist, dim=-1)
        mask = min_dist < dist_th
        matches_gt = torch.where(mask, matched_indices, torch.tensor(-1, device=device))
        
        return matches_gt

    def _compute_loss(self, outputs, kpts0, kpts1, T_0to1):
        scores = outputs['log_assignment']
        matches_gt = self._compute_gt_matches(kpts0, kpts1, T_0to1)
        
        B, M, N = scores.shape[0], scores.shape[1]-1, scores.shape[2]-1
        
        targets = matches_gt.clone()
        targets[targets == -1] = N
        
        target_log_probs = torch.gather(scores[:, :M, :], 2, targets.unsqueeze(2)).squeeze(2)
        loss = -target_log_probs.mean()
        
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self._compute_loss(outputs, batch['keypoints0'], batch['keypoints1'], batch['T_0to1'])
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证步骤（使用统一的评估器）"""
        outputs = self(batch)
        
        loss = self._compute_loss(outputs, batch['keypoints0'], batch['keypoints1'], batch['T_0to1'])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 使用统一的评估器
        result = self.evaluator.evaluate_batch(batch, outputs, self)
        
        return result

    def on_validation_epoch_start(self):
        """每个验证 epoch 开始时重置评估器"""
        self.evaluator.reset()

    def on_validation_epoch_end(self):
        """在模型自身 hook 中 log combined_auc，确保 EarlyStopping 能找到该指标"""
        # 使用统一的评估器计算聚合指标
        metrics = self.evaluator.compute_epoch_metrics()
        
        # Log 所有指标
        self.log('auc@5',        metrics['auc@5'],        on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@10',       metrics['auc@10'],       on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@20',       metrics['auc@20'],       on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('mAUC',         metrics.get('mAUC', 0.0), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('combined_auc', metrics['combined_auc'], on_epoch=True, prog_bar=True,  logger=True, sync_dist=False)
        self.log('val_mse',      metrics['mse'],          on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('val_mace',     metrics['mace'],         on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('inverse_mace', metrics['inverse_mace'], on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

# ==========================================
# 回调类: MultimodalValidationCallback
# ==========================================
class MultimodalValidationCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val = -1.0
        self.result_dir = Path(f"results/lightglue_{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []

        import csv
        self.csv_path = self.result_dir / "metrics.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val MSE", "Val MACE", "Val AUC@5", "Val AUC@10", "Val AUC@20", "Val Combined AUC", "Val Inverse MACE"])
        
        self.current_train_metrics = {}
        self.current_val_metrics = {}

    def _try_write_csv(self, epoch):
        if epoch in self.current_train_metrics and epoch in self.current_val_metrics:
            t = self.current_train_metrics.pop(epoch)
            v = self.current_val_metrics.pop(epoch)
            import csv
            with open(self.csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    t.get('loss', ''),
                    v.get('val_loss', ''),
                    v['mse'],
                    v['mace'],
                    v['auc5'],
                    v['auc10'],
                    v['auc20'],
                    v['combined_auc'],
                    v['inverse_mace']
                ])

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 从统一评估器的结果中提取 MSE 和 MACE
        if 'mses' in outputs:
            self.epoch_mses.extend(outputs['mses'])
        if 'maces' in outputs:
            self.epoch_maces.extend(outputs['maces'])

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        display_metrics = {}
        
        if 'train/loss_epoch' in metrics:
            display_metrics['loss'] = metrics['train/loss_epoch'].item()
        
        if display_metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
            logger.info(f"Epoch {epoch} 训练总结 >> {metric_str}")
        
        self.current_train_metrics[epoch] = display_metrics
        self._try_write_csv(epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        # 从 callback_metrics 读取所有指标
        display_metrics = {}
        for k in ['val_loss', 'val_mse', 'val_mace', 'auc@5', 'auc@10', 'auc@20', 'mAUC', 'combined_auc', 'inverse_mace']:
            if k in metrics:
                display_metrics[k] = metrics[k].item()
        
        # 兼容旧的 CSV 格式
        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = display_metrics.get('combined_auc', 0.0)
        avg_mse = display_metrics.get('val_mse', 0.0)
        avg_mace = display_metrics.get('val_mace', 0.0)
        inverse_mace = display_metrics.get('inverse_mace', 0.0)
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        logger.info(f"Epoch {epoch} 验证总结 >> {metric_str}")
        
        self.current_val_metrics[epoch] = {
            'mse': avg_mse,
            'mace': avg_mace,
            'auc5': auc5,
            'auc10': auc10,
            'auc20': auc20,
            'combined_auc': combined_auc,
            'inverse_mace': inverse_mace,
            'val_loss': display_metrics.get('val_loss', 0.0)
        }
        self._try_write_csv(epoch)
        
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
            
        is_best = False
        if combined_auc > self.best_val:
            self.best_val = combined_auc
            is_best = True
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest Combined AUC: {combined_auc:.4f}\n")
                f.write(f"AUC@5: {auc5:.4f}\nAUC@10: {auc10:.4f}\nAUC@20: {auc20:.4f}\n")
                f.write(f"MACE: {avg_mace:.4f}\nMSE: {avg_mse:.6f}\n")
            logger.info(f"发现新的最优模型! Epoch {epoch}, Combined AUC: {combined_auc:.4f}")

        if is_best or (epoch % 5 == 0):
            self._trigger_visualization(trainer, pl_module, is_best, epoch)

    def _trigger_visualization(self, trainer, pl_module, is_best, epoch):
        pl_module.force_viz = True
        target_dir = self.result_dir / (f"epoch{epoch}_best" if is_best else f"epoch{epoch}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        pl_module.eval()
        
        visualized_count = 0
        max_visualize = 20  # 最多可视化 20 个测试集样本
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = pl_module.validation_step(batch, batch_idx)
                
                # 只可视化测试集样本
                batch_size = batch['image0'].shape[0]
                splits = batch.get('split', ['unknown'] * batch_size)
                
                for i in range(batch_size):
                    sample_split = splits[i] if isinstance(splits, list) else splits
                    
                    # 只可视化 test split
                    if sample_split == 'test':
                        self._process_batch_sample(trainer, pl_module, batch, outputs, target_dir, i, batch_idx, sample_split)
                        visualized_count += 1
                        
                        if visualized_count >= max_visualize:
                            break
                
                if visualized_count >= max_visualize:
                    break
        
        logger.info(f"已可视化 {visualized_count} 个测试集样本")
        pl_module.force_viz = False

    def _process_batch_sample(self, trainer, pl_module, batch, outputs, epoch_dir, sample_idx, batch_idx, split):
        """处理并可视化单个样本"""
        H_ests = outputs.get('H_est', [np.eye(3)] * batch['image0'].shape[0])
        H_est = H_ests[sample_idx]
        
        if not is_valid_homography(H_est):
            H_est = np.eye(3)
        
        img0 = (batch['image0'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
        img1 = (batch['image1'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
        img1_gt = (batch['image1_gt'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
        
        h, w = img0.shape
        try:
            H_inv = np.linalg.inv(H_est)
            img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
        except:
            img1_result = img1.copy()
        
        sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}_{split}_{Path(batch['pair_names'][0][sample_idx]).stem}_vs_{Path(batch['pair_names'][1][sample_idx]).stem}"
        save_path = epoch_dir / sample_name
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / "fix.png"), img0)
        cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
        
        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        
        if 'kpts0' in outputs and 'kpts1' in outputs:
            kpts0_np = outputs['kpts0'][sample_idx].cpu().numpy() if hasattr(outputs['kpts0'][sample_idx], 'cpu') else batch['keypoints0'][sample_idx].cpu().numpy()
            kpts1_np = outputs['kpts1'][sample_idx].cpu().numpy() if hasattr(outputs['kpts1'][sample_idx], 'cpu') else batch['keypoints1'][sample_idx].cpu().numpy()
            
            for pt in kpts0_np:
                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            for pt in kpts1_np:
                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            
            if 'matches0' in outputs:
                m0 = outputs['matches0'][sample_idx].cpu() if hasattr(outputs['matches0'][sample_idx], 'cpu') else outputs['matches0'][sample_idx]
                valid = m0 > -1
                m_indices_0 = torch.where(valid)[0].numpy()
                m_indices_1 = m0[valid].numpy()
                
                for idx0 in m_indices_0:
                    pt = kpts0_np[idx0]
                    cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                for idx1 in m_indices_1:
                    pt = kpts1_np[idx1]
                    cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                
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

    def _process_batch(self, trainer, pl_module, batch, outputs, epoch_dir, save_images=False):
        batch_size = batch['image0'].shape[0]
        mses, maces = [], []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        
        rejected_count = 0
        
        for i in range(batch_size):
            H_est = H_ests[i]
            
            if not is_valid_homography(H_est):
                H_est = np.eye(3)
                rejected_count += 1
            
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                mask = (res_f > 0)
                mse = np.mean((res_f[mask].astype(np.float64) - orig_f[mask].astype(np.float64))**2) if np.any(mask) else 0.0
            except:
                mse = 0.0
            mses.append(mse)
            maces.append(compute_corner_error(H_est, Ts_gt[i], h, w))
            
            if save_images:
                sample_name = f"{Path(batch['pair_names'][0][i]).stem}_vs_{Path(batch['pair_names'][1][i]).stem}"
                save_path = epoch_dir / sample_name
                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / "fix.png"), img0)
                cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
                
                img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                
                if 'kpts0' in outputs and 'kpts1' in outputs:
                    kpts0_np = outputs['kpts0'][i].cpu().numpy()
                    kpts1_np = outputs['kpts1'][i].cpu().numpy()
                    
                    for pt in kpts0_np:
                        cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    for pt in kpts1_np:
                        cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    
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
        
        if rejected_count > 0 and save_images:
            logger.info(f"防爆锁触发: {rejected_count}/{batch_size}")
        
        return mses, maces

# ==========================================
# 早停机制
# ==========================================
class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

# ==========================================
# 主函数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="LightGlue CFFA Real-Data Training")
    parser.add_argument('--name', '-n', type=str, default='lightglue_cffa_baseline', help='训练名称')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--start_point', type=str, default=None, help='从检查点恢复')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gpus', type=str, default='1')
    return parser.parse_args()

def main():
    args = parse_args()
    args.mode = 'cffa'
    
    config = get_default_config()
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)
    
    result_dir = Path(f"results/lightglue_{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.txt"
    
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="a", backtrace=True, diagnose=False)
    logger.info(f"日志将同时保存到: {log_file}")
    
    os.environ['LOFTR_LOG_FILE'] = str(log_file)
    
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
    
    model = PL_LightGlue_Real(config, result_dir=str(result_dir))
    data_module = MultimodalDataModule(args, config)
    
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=f"lightglue_{args.name}")
    
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=0,
        monitor='combined_auc',
        mode='max',
        patience=10,
        min_delta=0.0001,
        strict=False  # 首个 epoch 若指标未出现不报错（兼容性保障）
    )
    
    logger.info("早停配置: monitor=combined_auc, start_epoch=0, patience=10, min_delta=0.0001")
    
    if not hasattr(args, 'mode'):
        args.mode = 'cffa'
    
    logger.info(f"GPU 配置: devices={gpus_list}, num_gpus={_n_gpus}")
    logger.info(f"学习率: {config.TRAINER.TRUE_LR:.6f} (scaled from {config.TRAINER.CANONICAL_LR})")
    
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
        'devices': gpus_list,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'callbacks': [
            MultimodalValidationCallback(args), 
            LearningRateMonitor(logging_interval='step'), 
            early_stop_callback
        ],
        'logger': tb_logger,
    }
    
    if _n_gpus > 1:
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    ckpt_path = args.start_point if args.start_point else None
    
    logger.info(f"开始真实数据训练: {args.name}")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()

