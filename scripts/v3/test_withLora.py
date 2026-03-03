"""
统一的测试/验证模块
提供 gen 和 real 两种模式的前向推理和指标计算接口
供训练脚本和独立测试脚本调用，确保指标计算的一致性
支持 SuperPointLoRA 模型
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger

# 添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.v1.metrics import (
    compute_homography_errors, 
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)

from scripts.v3.superpoint_detLora import SuperPointLoRA


def load_lora_weights_to_model(model, lora_name):
    """为模型的SuperPointLoRA提取器加载LoRA权重
    
    Args:
        model: PyTorch Lightning 模型（包含 extractor 属性）
        lora_name: LoRA训练时保存的name，将加载 results/superpoint_lora/{lora_name}/ 下的best checkpoint
                  或者 results/superpoint_lora/{lora_name}/final_lora.pth
    """
    if not hasattr(model, 'extractor'):
        logger.warning("模型没有 extractor 属性，跳过LoRA权重加载")
        return False
    
    # 构建LoRA权重路径
    lora_dir = Path(f"results/superpoint_lora/{lora_name}")
    
    # 优先尝试加载best checkpoint
    checkpoints_dir = lora_dir / "checkpoints"
    
    if checkpoints_dir.exists():
        # 查找best checkpoint
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            # 加载best checkpoint中的LoRA权重
            try:
                checkpoint = torch.load(ckpt_files[0], map_location='cpu')
                if 'model.model.lora_Da.weight' in checkpoint['state_dict']:
                    # 从checkpoint中提取LoRA权重
                    lora_state = {}
                    for key in checkpoint['state_dict']:
                        if 'lora_Da.' in key or 'lora_Db.' in key:
                            lora_state[key.replace('model.model.', '')] = checkpoint['state_dict'][key]
                    
                    # 手动设置LoRA权重
                    state_dict_lora_Da = {k.replace('lora_Da.', ''): v for k, v in lora_state.items() if 'lora_Da.' in k}
                    state_dict_lora_Db = {k.replace('lora_Db.', ''): v for k, v in lora_state.items() if 'lora_Db.' in k}
                    
                    model.extractor.lora_Da.load_state_dict(state_dict_lora_Da)
                    model.extractor.lora_Db.load_state_dict(state_dict_lora_Db)
                    logger.info(f"从checkpoint加载LoRA权重: {ckpt_files[0]}")
                    return True
            except Exception as e:
                logger.warning(f"从checkpoint加载LoRA权重失败: {e}")
    
    # 如果加载checkpoint失败，尝试加载final_lora.pth
    final_lora_path = lora_dir / "final_lora.pth"
    if final_lora_path.exists():
        try:
            model.extractor.load_lora_weights(str(final_lora_path))
            logger.info(f"从final_lora.pth加载LoRA权重: {final_lora_path}")
            return True
        except Exception as e:
            logger.warning(f"从final_lora.pth加载LoRA权重失败: {e}")
    
    logger.warning(f"未找到有效的LoRA权重文件: {lora_dir}")
    return False


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


class UnifiedEvaluator:
    """
    统一的评估器
    支持 gen 和 real 两种模式
    累积所有 batch 的误差后统一计算 AUC
    """
    def __init__(self, mode='gen', config=None):
        """
        Args:
            mode: 'gen' 或 'real'
            config: 配置对象，包含 TRAINER.RANSAC_PIXEL_THR 等参数
        """
        self.mode = mode
        self.config = config
        self.reset()
    
    def reset(self):
        """重置累积的指标"""
        self.all_errors = []  # 累积所有样本的误差用于计算 AUC
        self.all_mses = []    # 累积所有样本的 MSE
        self.all_maces = []   # 累积所有样本的 MACE
        self.total_samples = 0
        self.failed_samples = 0
    
    def evaluate_batch(self, batch, outputs, pl_module):
        """
        评估一个 batch
        
        Args:
            batch: 数据 batch，包含 image0, image1, image1_gt, T_0to1, keypoints0, keypoints1 等
            outputs: 模型输出，包含 matches0 等
            pl_module: PyTorch Lightning 模块（用于获取 config）
            
        Returns:
            dict: 包含 H_est, mses, maces, metrics_batch 等信息
        """
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        
        B = kpts0.shape[0]
        H_ests = []
        batch_mses = []
        batch_maces = []
        
        # 构建用于 metrics.py 的数据格式
        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []
        
        # 为每张图计算单应矩阵
        for b in range(B):
            self.total_samples += 1
            
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
            
            # 计算单应矩阵
            if len(pts0) >= 4:
                try:
                    ransac_thr = self.config.TRAINER.RANSAC_PIXEL_THR if self.config else 3.0
                    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thr)
                    if H is None:
                        H = np.eye(3)
                except:
                    H = np.eye(3)
            else:
                H = np.eye(3)
            
            # 判断是否匹配失败
            is_match_failed = False
            if not is_valid_homography(H):
                H = np.eye(3)
                is_match_failed = True
            elif np.allclose(H, np.eye(3), atol=1e-3):
                is_match_failed = True
            elif len(pts0) < 4:
                is_match_failed = True
            
            if is_match_failed:
                self.failed_samples += 1
            
            H_ests.append(H)
            
            # 计算 MSE 和 MACE（只在匹配成功时）
            if not is_match_failed:
                img0 = (batch['image0'][b, 0].cpu().numpy() * 255).astype(np.uint8)
                img1 = (batch['image1'][b, 0].cpu().numpy() * 255).astype(np.uint8)
                img1_gt = (batch['image1_gt'][b, 0].cpu().numpy() * 255).astype(np.uint8)
                
                h, w = img0.shape
                try:
                    H_inv = np.linalg.inv(H)
                    img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
                except:
                    img1_result = img1.copy()
                
                # 计算 MSE
                try:
                    res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                    mask = (res_f > 0)
                    mse = np.mean((res_f[mask].astype(np.float64) - orig_f[mask].astype(np.float64))**2) if np.any(mask) else 0.0
                except:
                    mse = 0.0
                
                # 计算 MACE
                T_gt = batch['T_0to1'][b].cpu().numpy()
                mace = compute_corner_error(H, T_gt, h, w)
                
                batch_mses.append(mse)
                batch_maces.append(mace)
                self.all_mses.append(mse)
                self.all_maces.append(mace)
        
        # 构建 metrics.py 需要的 batch 格式
        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }
        
        # 使用 metrics.py 计算指标（会填充 t_errs）
        set_metrics_verbose(False)  # 训练时不输出详细日志
        compute_homography_errors(metrics_batch, self.config if self.config else pl_module.config)
        
        # 累积误差用于后续统一计算 AUC
        if len(metrics_batch.get('t_errs', [])) > 0:
            self.all_errors.extend(metrics_batch['t_errs'])
        
        return {
            'H_est': H_ests,
            'mses': batch_mses,
            'maces': batch_maces,
            'metrics_batch': metrics_batch,
            'matches0': matches0,  # 添加 matches0 用于可视化
            'kpts0': kpts0,        # 添加 kpts0 用于可视化
            'kpts1': kpts1         # 添加 kpts1 用于可视化
        }
    
    def compute_epoch_metrics(self):
        """
        计算整个 epoch 的聚合指标
        
        Returns:
            dict: 包含 auc@5, auc@10, auc@20, mAUC, combined_auc, mse, mace, match_failure_rate 等
        """
        metrics = {}
        
        # 计算 AUC（使用所有样本统一计算）
        if self.all_errors and len(self.all_errors) > 0:
            auc_dict = error_auc(self.all_errors, [5, 10, 20])
            metrics['auc@5'] = auc_dict.get('auc@5', 0.0)
            metrics['auc@10'] = auc_dict.get('auc@10', 0.0)
            metrics['auc@20'] = auc_dict.get('auc@20', 0.0)
            
            mauc_dict = compute_auc_rop(self.all_errors, limit=25)
            metrics['mAUC'] = mauc_dict.get('mAUC', 0.0)
        else:
            metrics['auc@5'] = 0.0
            metrics['auc@10'] = 0.0
            metrics['auc@20'] = 0.0
            metrics['mAUC'] = 0.0
        
        metrics['combined_auc'] = (metrics['auc@5'] + metrics['auc@10'] + metrics['auc@20']) / 3.0
        
        # 计算 MSE 和 MACE（只在匹配成功的样本上）
        metrics['mse'] = sum(self.all_mses) / len(self.all_mses) if self.all_mses else 0.0
        metrics['mace'] = sum(self.all_maces) / len(self.all_maces) if self.all_maces else 0.0
        metrics['inverse_mace'] = 1.0 / (1.0 + metrics['mace']) if metrics['mace'] > 0 else 1.0
        
        # 计算匹配失败率
        metrics['match_failure_rate'] = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['total_samples'] = self.total_samples
        metrics['failed_samples'] = self.failed_samples
        metrics['success_samples'] = self.total_samples - self.failed_samples
        
        return metrics


def run_evaluation(pl_module, dataloader, mode='gen', verbose=True):
    """
    运行完整的评估流程
    
    Args:
        pl_module: PyTorch Lightning 模块
        dataloader: 数据加载器
        mode: 'gen' 或 'real'
        verbose: 是否输出详细日志
        
    Returns:
        dict: 评估指标
    """
    evaluator = UnifiedEvaluator(mode=mode, config=pl_module.config)
    
    pl_module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 将 batch 移到设备上
            batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = pl_module(batch)
            
            # 评估
            evaluator.evaluate_batch(batch, outputs, pl_module)
            
            if verbose and batch_idx % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1} 个 batch")
    
    # 计算聚合指标
    metrics = evaluator.compute_epoch_metrics()
    
    if verbose:
        logger.info(f"评估完成: {metrics}")
    
    return metrics
