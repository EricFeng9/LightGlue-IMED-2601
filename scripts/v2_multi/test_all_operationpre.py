"""
针对 train_onMultiGen_vessels_enhanced.py 训练权重的测试脚本

测试内容：
- 使用 --name 指定的权重，对三个数据集全量数据（train+val 合并）做测试
- --baseline 模式：额外运行 LightGlue 原生预训练权重作为基准
- 最终输出三个数据集上、生成数据训练 vs baseline 的对比表格（CSV）
- 每个数据集输出 5 个样本可视化结果
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger
import argparse
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
import csv

# 添加父目录到 sys.path
# 先添加项目根目录，以便导入 dataset 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
# 再添加 LightGlue 目录，以便导入 lightglue 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lightglue import LightGlue
from lightglue.superpoint import SuperPoint

# 导入 metrics（使用 v2_multi 版本的 metrics，与训练保持一致）
from scripts.v2_multi.metrics import (
    compute_homography_errors,
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)

from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 数据集封装
# ---------------------------------------------------------------------------

class RealDatasetWrapper(torch.utils.data.Dataset):
    """格式转换封装，统一真实数据集输出格式"""
    def __init__(self, base_dataset, split_name='unknown', dataset_name='MultiModal'):
        self.base_dataset = base_dataset
        self.split_name = split_name
        self.dataset_name = dataset_name
        # 目标图像大小（与数据预处理一致）
        self.target_size = 512

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]

        # 获取原始样本（包含GT关键点）
        try:
            if hasattr(self.base_dataset, 'get_raw_sample'):
                raw_sample = self.base_dataset.get_raw_sample(idx)
                # raw_sample[2] 是 fix 图像的关键点，raw_sample[3] 是 moving 图像的关键点
                fix_points = raw_sample[2]  # (N, 2) 关键点坐标
                moving_points = raw_sample[3]  # (N, 2) 关键点坐标

                # 缩放关键点坐标到目标图像大小
                img_fix = raw_sample[0]
                img_moving = raw_sample[1]
                if len(img_fix.shape) == 3:
                    h_fix, w_fix = img_fix.shape[:2]
                else:
                    h_fix, w_fix = img_fix.shape
                if len(img_moving.shape) == 3:
                    h_mov, w_mov = img_moving.shape[:2]
                else:
                    h_mov, w_mov = img_moving.shape

                # 缩放关键点到目标尺寸
                scale_x_fix = self.target_size / float(w_fix)
                scale_y_fix = self.target_size / float(h_fix)
                scale_x_mov = self.target_size / float(w_mov)
                scale_y_mov = self.target_size / float(h_mov)

                if len(fix_points) > 0:
                    fix_points = fix_points.copy()
                    fix_points[:, 0] *= scale_x_fix
                    fix_points[:, 1] *= scale_y_fix
                if len(moving_points) > 0:
                    moving_points = moving_points.copy()
                    moving_points[:, 0] *= scale_x_mov
                    moving_points[:, 1] *= scale_y_mov
            else:
                fix_points = np.array([], dtype=np.float32).reshape(0, 2)
                moving_points = np.array([], dtype=np.float32).reshape(0, 2)
        except Exception:
            fix_points = np.array([], dtype=np.float32).reshape(0, 2)
            moving_points = np.array([], dtype=np.float32).reshape(0, 2)

        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2

        def to_gray(t):
            if t.shape[0] == 3:
                gray = 0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]
                return gray.unsqueeze(0)
            return t

        fix_gray = to_gray(fix_tensor)
        moving_gray = to_gray(moving_gt_tensor)
        moving_orig_gray = to_gray(moving_original_tensor)

        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except Exception:
            T_fix_to_moving = T_0to1

        # 转换 numpy 关键点为 torch tensor
        fix_points_tensor = torch.from_numpy(fix_points).float() if len(fix_points) > 0 else torch.zeros(0, 2)
        moving_points_tensor = torch.from_numpy(moving_points).float() if len(moving_points) > 0 else torch.zeros(0, 2)

        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'T_0to1': T_fix_to_moving,
            'pair_names': (os.path.basename(fix_path), os.path.basename(moving_path)),
            'dataset_name': self.dataset_name,
            'split': self.split_name,
            'gt_pts0': fix_points_tensor,  # GT关键点（固定图，已缩放到512x512）
            'gt_pts1': moving_points_tensor,  # GT关键点（移动图，已缩放到512x512）
        }


def build_full_dataset(dataset_cls, root_dir, mode, dataset_name):
    """将 train + val 拼接为全量数据集"""
    train_base = dataset_cls(root_dir=str(root_dir), split='train', mode=mode)
    val_base = dataset_cls(root_dir=str(root_dir), split='val', mode=mode)
    train_wrapped = RealDatasetWrapper(train_base, split_name='train', dataset_name=dataset_name)
    val_wrapped = RealDatasetWrapper(val_base, split_name='val', dataset_name=dataset_name)
    full = ConcatDataset([train_wrapped, val_wrapped])
    logger.info(f"加载 {dataset_name} 全量数据集: train={len(train_wrapped)}, val={len(val_wrapped)}, total={len(full)}")
    return full


def build_all_dataloaders(args):
    """返回 {dataset_name: DataLoader} 字典，每个数据集独立一个 DataLoader"""
    script_dir = Path(__file__).parent.parent.parent

    loader_params = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'shuffle': False,
    }

    cffa_dir = script_dir.parent / 'dataset' / 'operation_pre_filtered_cffa'
    cfoct_dir = script_dir.parent / 'dataset' / 'operation_pre_filtered_cfoct'
    octfa_dir = script_dir.parent / 'dataset' / 'operation_pre_filtered_octfa'

    datasets = {
        'CFFA': build_full_dataset(CFFADataset, cffa_dir, 'fa2cf', 'CFFA'),
        'CFOCT': build_full_dataset(CFOCTDataset, cfoct_dir, 'oct2cf', 'CFOCT'),
        'OCTFA': build_full_dataset(OCTFADataset, octfa_dir, 'fa2oct', 'OCTFA'),
    }

    return {name: DataLoader(ds, **loader_params) for name, ds in datasets.items()}


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_metrics_for_dataset(evaluator, dataset_name):
    """AUC 使用全部样本（含 Failed/Inaccurate）；MSE/MACE 仅对 Acceptable 样本求平均。
    若某模态上 Trained 的 Inaccurate 更多，则 AUC 可能低于 Baseline，而 MSE/MACE 仍可更优。"""
    if dataset_name not in evaluator.per_dataset_errors or len(evaluator.per_dataset_errors[dataset_name]) == 0:
        return None
    errors = evaluator.per_dataset_errors[dataset_name]
    auc_dict = error_auc(errors, [5, 10, 20])
    mauc_dict = compute_auc_rop(errors, limit=25)
    mses = evaluator.per_dataset_mses.get(dataset_name, [])
    maces = evaluator.per_dataset_maces.get(dataset_name, [])
    failed = evaluator.per_dataset_failed.get(dataset_name, 0)
    inaccurate = evaluator.per_dataset_inaccurate.get(dataset_name, 0)
    acceptable = evaluator.per_dataset_acceptable.get(dataset_name, 0)
    return {
        'dataset': dataset_name,
        'auc@5': auc_dict.get('auc@5', 0.0),
        'auc@10': auc_dict.get('auc@10', 0.0),
        'auc@20': auc_dict.get('auc@20', 0.0),
        'mAUC': mauc_dict.get('mAUC', 0.0),
        'combined_auc': (auc_dict.get('auc@5', 0.0) + auc_dict.get('auc@10', 0.0) + auc_dict.get('auc@20', 0.0)) / 3.0,
        'mse': sum(mses) / len(mses) if mses else 0.0,
        'mace': sum(maces) / len(maces) if maces else 0.0,
        'num_samples': len(errors),
        'failed': failed,
        'inaccurate': inaccurate,
        'acceptable': acceptable,
    }


class UnifiedEvaluator:
    """统一的评估器，支持按数据集分别计算指标"""
    def __init__(self, config=None):
        self.config = config
        self.reset()

    def reset(self):
        self.all_errors = []
        self.all_mses = []
        self.all_maces = []
        self.total_samples = 0
        self.failed_samples = 0
        self.inaccurate_samples = 0
        self.acceptable_samples = 0
        self.per_dataset_errors = {}
        self.per_dataset_mses = {}
        self.per_dataset_maces = {}
        self.per_dataset_samples = {}
        self.per_dataset_failed = {}
        self.per_dataset_inaccurate = {}
        self.per_dataset_acceptable = {}

    def evaluate_batch(self, batch, outputs, pl_module):
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        dataset_names = batch.get('dataset_name', ['unknown'] * kpts0.shape[0])
        B = kpts0.shape[0]

        mkpts0_f_list, mkpts1_f_list, m_bids_list = [], [], []
        for b in range(B):
            m0 = matches0[b]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[valid]
            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()
            if len(pts0) > 0:
                mkpts0_f_list.append(torch.from_numpy(pts0).float())
                mkpts1_f_list.append(torch.from_numpy(pts1).float())
                m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long))

        # 获取 GT 关键点（每个样本一个）
        gt_pts0_list = []
        gt_pts1_list = []
        if 'gt_pts0' in batch and 'gt_pts1' in batch:
            for b in range(B):
                gt_pts0_list.append(batch['gt_pts0'][b])
                gt_pts1_list.append(batch['gt_pts1'][b])

        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name'],
            'gt_pts0': gt_pts0_list,  # List of tensors, 每个元素 (N, 2)
            'gt_pts1': gt_pts1_list,  # List of tensors, 每个元素 (N, 2)
        }

        compute_homography_errors(metrics_batch, self.config if self.config else pl_module.config)

        self.total_samples += B
        failed_mask = metrics_batch.get('failed_mask', [False] * B)
        inaccurate_mask = metrics_batch.get('inaccurate_mask', [False] * B)
        self.failed_samples += int(np.sum(np.array(failed_mask, dtype=np.int64)))
        self.inaccurate_samples += int(np.sum(np.array(inaccurate_mask, dtype=np.int64)))
        self.acceptable_samples += int(B - np.sum(np.array(failed_mask, dtype=np.int64)) - np.sum(np.array(inaccurate_mask, dtype=np.int64)))

        if len(metrics_batch.get('t_errs', [])) > 0:
            self.all_errors.extend(list(metrics_batch['t_errs']))

        batch_mses = list(metrics_batch.get('mse_list', []))
        batch_maces = list(metrics_batch.get('mace_list', []))
        for mse in batch_mses:
            if np.isfinite(mse):
                self.all_mses.append(float(mse))
        for mace in batch_maces:
            if np.isfinite(mace):
                self.all_maces.append(float(mace))

        for b in range(B):
            dataset = dataset_names[b] if isinstance(dataset_names, list) else dataset_names
            if dataset not in self.per_dataset_errors:
                self.per_dataset_errors[dataset] = []
                self.per_dataset_mses[dataset] = []
                self.per_dataset_maces[dataset] = []
                self.per_dataset_samples[dataset] = 0
                self.per_dataset_failed[dataset] = 0
                self.per_dataset_inaccurate[dataset] = 0
                self.per_dataset_acceptable[dataset] = 0

            self.per_dataset_samples[dataset] += 1
            if b < len(failed_mask):
                if failed_mask[b]:
                    self.per_dataset_failed[dataset] += 1
                elif inaccurate_mask[b]:
                    self.per_dataset_inaccurate[dataset] += 1
                else:
                    self.per_dataset_acceptable[dataset] += 1

            if b < len(metrics_batch.get('t_errs', [])):
                self.per_dataset_errors[dataset].append(metrics_batch['t_errs'][b])
            if b < len(batch_mses) and np.isfinite(batch_mses[b]):
                self.per_dataset_mses[dataset].append(float(batch_mses[b]))
            if b < len(batch_maces) and np.isfinite(batch_maces[b]):
                self.per_dataset_maces[dataset].append(float(batch_maces[b]))

        return {
            'H_est': metrics_batch.get('H_est', [np.eye(3)] * B),
            'mses': batch_mses,
            'maces': batch_maces,
            'metrics_batch': metrics_batch,
            'matches0': matches0,
            'kpts0': kpts0,
            'kpts1': kpts1,
        }

    def compute_epoch_metrics(self):
        metrics = {}
        if self.all_errors:
            auc_dict = error_auc(self.all_errors, [5, 10, 20])
            metrics['auc@5'] = auc_dict.get('auc@5', 0.0)
            metrics['auc@10'] = auc_dict.get('auc@10', 0.0)
            metrics['auc@20'] = auc_dict.get('auc@20', 0.0)
            mauc_dict = compute_auc_rop(self.all_errors, limit=25)
            metrics['mAUC'] = mauc_dict.get('mAUC', 0.0)
        else:
            metrics.update({'auc@5': 0.0, 'auc@10': 0.0, 'auc@20': 0.0, 'mAUC': 0.0})

        metrics['combined_auc'] = (metrics['auc@5'] + metrics['auc@10'] + metrics['auc@20']) / 3.0
        metrics['mse'] = sum(self.all_mses) / len(self.all_mses) if self.all_mses else 0.0
        metrics['mace'] = sum(self.all_maces) / len(self.all_maces) if self.all_maces else 0.0
        metrics['inverse_mace'] = 1.0 / (1.0 + metrics['mace']) if metrics['mace'] > 0 else 1.0
        metrics['match_failure_rate'] = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['total_samples'] = self.total_samples
        metrics['failed_samples'] = self.failed_samples
        metrics['success_samples'] = self.total_samples - self.failed_samples
        metrics['inaccurate_samples'] = self.inaccurate_samples
        metrics['acceptable_samples'] = self.acceptable_samples
        metrics['inaccurate_rate'] = self.inaccurate_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['acceptable_rate'] = self.acceptable_samples / self.total_samples if self.total_samples > 0 else 0.0

        metrics['per_dataset'] = {}
        for dataset_name in self.per_dataset_errors:
            ds_metrics = compute_metrics_for_dataset(self, dataset_name)
            if ds_metrics:
                metrics['per_dataset'][dataset_name] = ds_metrics
        return metrics


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _visualize_samples(batch, outputs, output_dir, batch_idx, viz_counters, max_per_dataset=5):
    """对每个数据集最多保存 max_per_dataset 个样本的可视化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        from lightglue import viz2d
        has_viz2d = True
    except Exception:
        has_viz2d = False

    output_dir = Path(output_dir)
    batch_size = batch['image0'].shape[0]
    H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
    dataset_names = batch.get('dataset_name', ['unknown'] * batch_size)

    for sample_idx in range(batch_size):
        dataset_name = dataset_names[sample_idx] if isinstance(dataset_names, list) else dataset_names

        if viz_counters.get(dataset_name, 0) >= max_per_dataset:
            continue

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
        except Exception:
            img1_result = img1.copy()

        pair_names = batch.get('pair_names', None)
        sample_count = viz_counters.get(dataset_name, 0)
        if pair_names:
            sample_name = f"{dataset_name}_sample{sample_count:02d}_{Path(pair_names[0][sample_idx]).stem}_vs_{Path(pair_names[1][sample_idx]).stem}"
        else:
            sample_name = f"{dataset_name}_sample{sample_count:02d}_batch{batch_idx:04d}"

        save_path = output_dir / 'visualizations' / dataset_name / sample_name
        save_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path / "fix.png"), img0)
        cv2.imwrite(str(save_path / "moving_original.png"), img1)
        cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
        cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)

        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        if 'kpts0' in outputs and 'kpts1' in outputs:
            kpts0_np = outputs['kpts0'][sample_idx].cpu().numpy()
            kpts1_np = outputs['kpts1'][sample_idx].cpu().numpy()
            for pt in kpts0_np:
                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            for pt in kpts1_np:
                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

            if 'matches0' in outputs:
                m0 = outputs['matches0'][sample_idx].cpu()
                valid = m0 > -1
                m_indices_0 = torch.where(valid)[0].numpy()
                m_indices_1 = m0[valid].numpy()

                for idx0 in m_indices_0:
                    pt = kpts0_np[idx0]
                    cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                for idx1 in m_indices_1:
                    pt = kpts1_np[idx1]
                    cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

                if has_viz2d and len(m_indices_0) > 0:
                    try:
                        fig = plt.figure(figsize=(12, 6))
                        viz2d.plot_images([img0, img1])
                        viz2d.plot_matches(kpts0_np[m_indices_0], kpts1_np[m_indices_1], color='lime', lw=0.5)
                        plt.savefig(str(save_path / "matches.png"), bbox_inches='tight', dpi=100)
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"绘制匹配图失败: {e}")

        cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
        cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)

        try:
            cv2.imwrite(str(save_path / "chessboard.png"), create_chessboard(img1_result, img0))
            cv2.imwrite(str(save_path / "chessboard_gt_vs_fix.png"), create_chessboard(img1_gt, img0))
            cv2.imwrite(str(save_path / "chessboard_gt_vs_pred.png"), create_chessboard(img1_gt, img1_result))
        except Exception:
            pass

        if 'mses' in outputs and sample_idx < len(outputs['mses']):
            with open(save_path / "metrics.txt", "w") as f:
                f.write(f"MSE: {outputs['mses'][sample_idx]:.6f}\n")
                f.write(f"MACE: {outputs['maces'][sample_idx]:.4f}\n")
                if 'matches0' in outputs:
                    m0 = outputs['matches0'][sample_idx].cpu()
                    f.write(f"Matches: {torch.sum(m0 > -1).item()}\n")

        viz_counters[dataset_name] = sample_count + 1


# ---------------------------------------------------------------------------
# 评估流程
# ---------------------------------------------------------------------------

def run_evaluation_per_dataset(model, dataloaders, config, save_visualizations=False, output_dir=None):
    """
    对每个数据集分别运行评估，返回 {dataset_name: metrics_dict}
    dataloaders: {dataset_name: DataLoader}
    """
    all_results = {}

    for ds_name, dataloader in dataloaders.items():
        logger.info(f"--- 开始评估数据集: {ds_name} ---")
        evaluator = UnifiedEvaluator(config=config)
        viz_counters = {}

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = model(batch)
                result = evaluator.evaluate_batch(batch, outputs, model)

                if save_visualizations and output_dir:
                    _visualize_samples(batch, result, output_dir, batch_idx, viz_counters, max_per_dataset=5)

                if batch_idx % 10 == 0:
                    logger.info(f"  [{ds_name}] 已处理 {batch_idx + 1} 个 batch")

        metrics = evaluator.compute_epoch_metrics()
        # 取本数据集自己的指标（per_dataset 只有当前 ds_name）
        ds_metrics = metrics['per_dataset'].get(ds_name, None)
        if ds_metrics is None:
            # fallback: 用全局聚合指标
            ds_metrics = {
                'dataset': ds_name,
                'auc@5': metrics['auc@5'],
                'auc@10': metrics['auc@10'],
                'auc@20': metrics['auc@20'],
                'mAUC': metrics['mAUC'],
                'combined_auc': metrics['combined_auc'],
                'mse': metrics['mse'],
                'mace': metrics['mace'],
                'num_samples': metrics['total_samples'],
            }
        logger.info(f"  [{ds_name}] AUC@5={ds_metrics['auc@5']:.4f} AUC@10={ds_metrics['auc@10']:.4f} "
                    f"AUC@20={ds_metrics['auc@20']:.4f} mAUC={ds_metrics['mAUC']:.4f} "
                    f"MACE={ds_metrics['mace']:.4f} MSE={ds_metrics['mse']:.6f}")
        all_results[ds_name] = ds_metrics

    return all_results


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

class BaselineLightGlueModel(pl.LightningModule):
    """使用 LightGlue 原生预训练权重的 baseline 模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.extractor = SuperPoint(max_num_keypoints=2048).eval()
        for param in self.extractor.parameters():
            param.requires_grad = False

        lg_conf = config.MATCHING.copy()
        lg_conf['weights'] = 'superpoint_lightglue'
        self.matcher = LightGlue(**lg_conf).eval()
        for param in self.matcher.parameters():
            param.requires_grad = False

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
                    'scores1': feats1['keypoint_scores'],
                })
        data = {
            'image0': {
                'keypoints': batch['keypoints0'],
                'descriptors': batch['descriptors0'],
                'image': batch['image0'],
            },
            'image1': {
                'keypoints': batch['keypoints1'],
                'descriptors': batch['descriptors1'],
                'image': batch['image1'],
            },
        }
        return self.matcher(data)


def load_trained_model(ckpt_path, config, output_dir):
    """加载 train_onMultiGen_vessels_enhanced 训练的模型"""
    import importlib
    module = importlib.import_module('scripts.v2_multi.train_onMultiGen_vessels_enhanced')
    pl_class = getattr(module, 'PL_LightGlue_Gen')
    model = pl_class.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        result_dir=str(output_dir),
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------

DATASET_ORDER = ['CFFA', 'CFOCT', 'OCTFA']
METRIC_KEYS = ['num_samples', 'failed', 'inaccurate', 'acceptable', 'auc@5', 'auc@10', 'auc@20', 'mAUC', 'combined_auc', 'mse', 'mace']
METRIC_DISPLAY = ['Samples', 'Failed', 'Inacc', 'Accept', 'AUC@5', 'AUC@10', 'AUC@20', 'mAUC', 'Combined_AUC', 'MSE', 'MACE']


def save_summary_txt(output_dir, label, results_per_ds):
    """保存单模型测试总结"""
    summary_path = output_dir / f"test_summary_{label}.txt"
    with open(summary_path, "w") as f:
        f.write(f"测试总结 [{label}]\n")
        f.write("=" * 60 + "\n")
        f.write("说明: AUC 使用全部样本(Failed=1e6+Inaccurate+Acceptable)；MSE/MACE 仅对 Acceptable 求平均。\n")
        f.write("若某模态 Trained 的 Inaccurate 多于 Baseline，则可能出现 AUC 更低但 MSE/MACE 更优。\n")
        f.write("-" * 60 + "\n")
        for ds_name in DATASET_ORDER:
            if ds_name not in results_per_ds:
                continue
            m = results_per_ds[ds_name]
            f.write(f"\n{ds_name}: 样本数={m['num_samples']} (Failed: {m.get('failed', 0)}, Inaccurate: {m.get('inaccurate', 0)}, Acceptable: {m.get('acceptable', 0)})\n")
            f.write(f"  AUC@5:       {m['auc@5']:.4f}\n")
            f.write(f"  AUC@10:      {m['auc@10']:.4f}\n")
            f.write(f"  AUC@20:      {m['auc@20']:.4f}\n")
            f.write(f"  mAUC:        {m['mAUC']:.4f}\n")
            f.write(f"  Combined AUC:{m['combined_auc']:.4f}\n")
            f.write(f"  MSE:         {m['mse']:.6f}\n")
            f.write(f"  MACE:        {m['mace']:.4f}\n")
    logger.info(f"测试总结已保存: {summary_path}")
    return summary_path


def save_comparison_csv(output_dir, trained_results, baseline_results=None):
    """
    保存对比 CSV
    列：Dataset | Metric | Trained | Baseline (如有)
    """
    csv_path = output_dir / "comparison_results.csv"

    with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 在 CSV 头部添加元数据注释
        writer.writerow([f"# Generated by test_all_operationpre.py"])

        if baseline_results is not None:
            writer.writerow(['Dataset', 'Metric', 'Trained (MultiGen)', 'Baseline (Pretrained)'])
        else:
            writer.writerow(['Dataset', 'Metric', 'Trained (MultiGen)'])

        for ds_name in DATASET_ORDER:
            trained_m = trained_results.get(ds_name, {})
            if baseline_results is not None:
                baseline_m = baseline_results.get(ds_name, {})

            for mk, md in zip(METRIC_KEYS, METRIC_DISPLAY):
                t_val = trained_m.get(mk, 'N/A')
                if isinstance(t_val, float):
                    t_val = f"{t_val:.4f}" if mk not in ('mse',) else f"{t_val:.6f}"

                if baseline_results is not None:
                    b_val = baseline_m.get(mk, 'N/A')
                    if isinstance(b_val, float):
                        b_val = f"{b_val:.4f}" if mk not in ('mse',) else f"{b_val:.6f}"
                    writer.writerow([ds_name, md, t_val, b_val])
                else:
                    writer.writerow([ds_name, md, t_val])

            writer.writerow([])  # 空行分隔数据集

    logger.info(f"对比 CSV 已保存: {csv_path}")

    # 同时保存一个更直观的"宽表"CSV（数据集为行，指标为列）
    wide_csv_path = output_dir / "comparison_wide.csv"
    with open(wide_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if baseline_results is not None:
            header = ['Dataset', 'Model'] + METRIC_DISPLAY
            writer.writerow(header)
            for ds_name in DATASET_ORDER:
                trained_m = trained_results.get(ds_name, {})
                baseline_m = baseline_results.get(ds_name, {})
                trained_row = [ds_name, 'Trained (MultiGen)']
                baseline_row = [ds_name, 'Baseline (Pretrained)']
                for mk in METRIC_KEYS:
                    tv = trained_m.get(mk, 'N/A')
                    bv = baseline_m.get(mk, 'N/A')
                    fmt = ":.6f" if mk == 'mse' else ":.4f"
                    trained_row.append(f"{tv:{fmt[1:]}}" if isinstance(tv, float) else str(tv))
                    baseline_row.append(f"{bv:{fmt[1:]}}" if isinstance(bv, float) else str(bv))
                writer.writerow(trained_row)
                writer.writerow(baseline_row)
        else:
            header = ['Dataset'] + METRIC_DISPLAY
            writer.writerow(header)
            for ds_name in DATASET_ORDER:
                m = trained_results.get(ds_name, {})
                row = [ds_name]
                for mk in METRIC_KEYS:
                    v = m.get(mk, 'N/A')
                    fmt = ".6f" if mk == 'mse' else ".4f"
                    row.append(f"{v:{fmt}}" if isinstance(v, float) else str(v))
                writer.writerow(row)

    logger.info(f"宽表 CSV 已保存: {wide_csv_path}")


def print_comparison_table(trained_results, baseline_results=None):
    """在日志中打印对比表格"""
    logger.info("说明: AUC 含全部样本；MSE/MCE 仅 Acceptable。某模态若 Trained 的 Inaccurate 更多，可能 AUC 更低但 MSE/MACE 更优。")
    header_parts = ['Dataset'.ljust(8), 'Model'.ljust(24)]
    for md in METRIC_DISPLAY:
        header_parts.append(md.rjust(12))
    logger.info("  ".join(header_parts))
    logger.info("-" * 130)

    for ds_name in DATASET_ORDER:
        trained_m = trained_results.get(ds_name, {})
        row = [ds_name.ljust(8), 'Trained (MultiGen)'.ljust(24)]
        for mk in METRIC_KEYS:
            v = trained_m.get(mk, 'N/A')
            row.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(12))
        logger.info("  ".join(row))

        if baseline_results is not None:
            baseline_m = baseline_results.get(ds_name, {})
            row = [ds_name.ljust(8), 'Baseline (Pretrained)'.ljust(24)]
            for mk in METRIC_KEYS:
                v = baseline_m.get(mk, 'N/A')
                row.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(12))
            logger.info("  ".join(row))
        logger.info("")


# ---------------------------------------------------------------------------
# 参数解析 & 主函数
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="针对 train_onMultiGen_vessels_enhanced 权重的测试脚本（三个全量数据集）"
    )
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='模型名称（results/lightglue_gen/<name>/best_checkpoint/model.ckpt）')
    parser.add_argument('--test_name', '-t', type=str, required=True,
                        help='测试名称（结果保存在 results/lightglue_gen/<name>/<test_name>/）')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='检查点路径（默认 results/lightglue_gen/<name>/best_checkpoint/model.ckpt）')
    parser.add_argument('--baseline', action='store_true',
                        help='额外运行 LightGlue 原生预训练权重作为 baseline 并输出对比表格')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（不指定则自动生成）')
    parser.add_argument('--gpus', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--no_viz', action='store_true', help='禁用可视化')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子（如果未指定则自动生成）
    if args.seed is None:
        import time
        args.seed = int(time.time() * 1000) % (2**31)
        logger.info(f"未指定 seed，自动生成: {args.seed}")
    else:
        if args.seed < 0 or args.seed >= 2**31:
            logger.error(f"seed 必须在 [0, {2**31}) 范围内")
            return
        logger.info(f"使用指定 seed: {args.seed}")

    # 设置所有随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 确定 checkpoint 路径
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(f"results/lightglue_gen/{args.name}/best_checkpoint/model.ckpt")

    if not ckpt_path.exists():
        logger.error(f"检查点不存在: {ckpt_path}")
        logger.info("请确保训练模型存在，或用 --checkpoint 指定有效路径")
        return

    # 输出目录
    output_dir = Path(f"results/lightglue_gen/{args.name}/{args.test_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    log_file = output_dir / "test_log.txt"
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="w")
    logger.info(f"日志保存至: {log_file}")

    # GPU 配置
    if ',' in str(args.gpus):
        gpus_list = [int(x) for x in args.gpus.split(',')]
    else:
        try:
            gpus_list = [int(args.gpus)]
        except Exception:
            gpus_list = [0]

    device = torch.device(f"cuda:{gpus_list[0]}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 获取配置
    import importlib
    module = importlib.import_module('scripts.v2_multi.train_onMultiGen_vessels_enhanced')
    get_default_config = getattr(module, 'get_default_config')
    config = get_default_config()
    config.TRAINER.WORLD_SIZE = len(gpus_list)
    config.TRAINER.TRUE_BATCH_SIZE = len(gpus_list) * args.batch_size

    logger.info(f"模型名称: {args.name}")
    logger.info(f"测试名称: {args.test_name}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Baseline 模式: {args.baseline}")

    # 构建各数据集的 DataLoader（train + val 全量）
    dataloaders = build_all_dataloaders(args)

    set_metrics_verbose(True)

    # -----------------------------------------------------------------------
    # 运行训练模型测试
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1/2: 测试训练模型")
    logger.info("=" * 60)

    trained_model = load_trained_model(ckpt_path, config, output_dir)
    trained_model = trained_model.to(device)

    viz_dir_trained = output_dir / "viz_trained" if not args.no_viz else None
    if viz_dir_trained:
        viz_dir_trained.mkdir(parents=True, exist_ok=True)

    trained_results = run_evaluation_per_dataset(
        trained_model,
        dataloaders,
        config=config,
        save_visualizations=not args.no_viz,
        output_dir=viz_dir_trained,
    )
    save_summary_txt(output_dir, "trained", trained_results)

    # -----------------------------------------------------------------------
    # 运行 Baseline 测试（如有 --baseline）
    # -----------------------------------------------------------------------
    baseline_results = None
    if args.baseline:
        logger.info("=" * 60)
        logger.info("Step 2/2: 测试 Baseline（LightGlue 原生预训练权重）")
        logger.info("=" * 60)

        baseline_model = BaselineLightGlueModel(config)
        baseline_model = baseline_model.to(device)
        baseline_model.eval()

        viz_dir_baseline = output_dir / "viz_baseline" if not args.no_viz else None
        if viz_dir_baseline:
            viz_dir_baseline.mkdir(parents=True, exist_ok=True)

        baseline_results = run_evaluation_per_dataset(
            baseline_model,
            dataloaders,
            config=config,
            save_visualizations=not args.no_viz,
            output_dir=viz_dir_baseline,
        )
        save_summary_txt(output_dir, "baseline", baseline_results)
    else:
        logger.info("未指定 --baseline，跳过 baseline 测试")

    # -----------------------------------------------------------------------
    # 输出对比表格
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("最终对比结果")
    logger.info("=" * 60)
    print_comparison_table(trained_results, baseline_results)
    save_comparison_csv(output_dir, trained_results, baseline_results)

    logger.info(f"所有结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
