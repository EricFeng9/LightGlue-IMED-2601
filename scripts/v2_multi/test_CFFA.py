"""
统一的测试/验证模块
提供 gen 和 cffa 两种模式的前向推理和指标计算接口
供训练脚本和独立测试脚本调用，确保指标计算的一致性
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

# 添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.v2.metrics import (
    compute_homography_errors, 
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)


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


class UnifiedEvaluator:
    """
    统一的评估器
    支持 gen 和 cffa 两种模式
    累积所有 batch 的误差后统一计算 AUC
    """
    def __init__(self, mode='gen', config=None):
        """
        Args:
            mode: 'gen' 或 'cffa'
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
        self.inaccurate_samples = 0
        self.acceptable_samples = 0
    
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

        # 构建用于 scripts/v2/metrics.py 的数据格式（跨 batch 拼接匹配点）
        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []
        
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
        
        # 构建 metrics.py 需要的 batch 格式
        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }
        
        # 训练时关闭详细日志；测试阶段由外部显式开启
        split = batch.get('split', None)
        if split is not None:
            split0 = split[0] if isinstance(split, (list, tuple)) else split
            if split0 == 'train':
                set_metrics_verbose(False)

        # 使用 metrics.py 计算指标（会填充 t_errs / mse_list / mace_list 等）
        compute_homography_errors(metrics_batch, self.config if self.config else pl_module.config)

        # --- 累积 epoch 级统计（严格对齐 metrics_cau_principle_0304.md）---
        self.total_samples += B
        failed_mask = metrics_batch.get('failed_mask', [False] * B)
        inaccurate_mask = metrics_batch.get('inaccurate_mask', [False] * B)

        self.failed_samples += int(np.sum(np.array(failed_mask, dtype=np.int64)))
        self.inaccurate_samples += int(np.sum(np.array(inaccurate_mask, dtype=np.int64)))
        self.acceptable_samples += int(B - np.sum(np.array(failed_mask, dtype=np.int64)) - np.sum(np.array(inaccurate_mask, dtype=np.int64)))

        if len(metrics_batch.get('t_errs', [])) > 0:
            self.all_errors.extend(list(metrics_batch['t_errs']))

        # MSE/MACE：仅 Acceptable（metrics.py 已对 failed/inaccurate 置 inf，这里过滤掉）
        batch_mses = list(metrics_batch.get('mse_list', []))
        batch_maces = list(metrics_batch.get('mace_list', []))
        for mse in batch_mses:
            if np.isfinite(mse):
                self.all_mses.append(float(mse))
        for mace in batch_maces:
            if np.isfinite(mace):
                self.all_maces.append(float(mace))
        
        return {
            'H_est': metrics_batch.get('H_est', [np.eye(3)] * B),
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

        metrics['inaccurate_samples'] = self.inaccurate_samples
        metrics['acceptable_samples'] = self.acceptable_samples
        metrics['inaccurate_rate'] = self.inaccurate_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['acceptable_rate'] = self.acceptable_samples / self.total_samples if self.total_samples > 0 else 0.0
        
        return metrics


def run_evaluation(pl_module, dataloader, mode='gen', verbose=True, save_visualizations=False, output_dir=None):
    """
    运行完整的评估流程
    
    Args:
        pl_module: PyTorch Lightning 模块
        dataloader: 数据加载器
        mode: 'gen' 或 'cffa'
        verbose: 是否输出详细日志
        save_visualizations: 是否保存可视化结果
        output_dir: 可视化结果保存目录
        
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
            result = evaluator.evaluate_batch(batch, outputs, pl_module)
            
            # 可视化
            if save_visualizations and output_dir:
                _visualize_batch(batch, result, output_dir, batch_idx)
            
            if verbose and batch_idx % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1} 个 batch")
    
    # 计算聚合指标
    metrics = evaluator.compute_epoch_metrics()
    
    if verbose:
        logger.info(f"评估完成: {metrics}")
    
    return metrics


def _visualize_batch(batch, outputs, output_dir, batch_idx):
    """可视化一个batch的结果"""
    import matplotlib.pyplot as plt
    from lightglue import viz2d
    
    output_dir = Path(output_dir)
    batch_size = batch['image0'].shape[0]
    H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
    
    for sample_idx in range(batch_size):
        H_est = H_ests[sample_idx]
        
        # 启用防爆锁
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
        
        # 构建样本名称
        pair_names = batch.get('pair_names', None)
        if pair_names:
            sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}_{Path(pair_names[0][sample_idx]).stem}_vs_{Path(pair_names[1][sample_idx]).stem}"
        else:
            sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}"
        
        save_path = output_dir / sample_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存基本图像
        cv2.imwrite(str(save_path / "fix.png"), img0)
        cv2.imwrite(str(save_path / "moving_original.png"), img1)
        cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
        cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)
        
        # 绘制关键点和匹配
        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        
        if 'kpts0' in outputs and 'kpts1' in outputs:
            kpts0_np = outputs['kpts0'][sample_idx].cpu().numpy()
            kpts1_np = outputs['kpts1'][sample_idx].cpu().numpy()
            
            # 绘制所有关键点（白色）
            for pt in kpts0_np:
                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            for pt in kpts1_np:
                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            
            # 绘制匹配点（红色）
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
        
        # 创建棋盘格对比图
        try:
            cb = create_chessboard(img1_result, img0)
            cv2.imwrite(str(save_path / "chessboard.png"), cb)

            # 额外保存：moving_gt vs fix 的 chessboard
            cb_gt_vs_fix = create_chessboard(img1_gt, img0)
            cv2.imwrite(str(save_path / "chessboard_gt_vs_fix.png"), cb_gt_vs_fix)

            # 额外保存：moving_gt vs moving_pred 的 chessboard
            cb_gt_vs_pred = create_chessboard(img1_gt, img1_result)
            cv2.imwrite(str(save_path / "chessboard_gt_vs_pred.png"), cb_gt_vs_pred)
        except:
            pass
        
        # 保存单样本指标
        if 'mses' in outputs and sample_idx < len(outputs['mses']):
            mse = outputs['mses'][sample_idx]
            mace = outputs['maces'][sample_idx]
            with open(save_path / "metrics.txt", "w") as f:
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MACE: {mace:.4f}\n")
                if 'matches0' in outputs:
                    m0 = outputs['matches0'][sample_idx].cpu()
                    valid = m0 > -1
                    num_matches = torch.sum(valid).item()
                    f.write(f"Matches: {num_matches}\n")


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
# 主函数
# ==========================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LightGlue 统一测试脚本")
    parser.add_argument('--mode', type=str, required=True, choices=['gen', 'cffa'], 
                        help='测试模式: gen (生成数据) 或 cffa (CFFA真实数据)')
    parser.add_argument('--name', type=str, required=True, 
                        help='模型名称（用于定位 results/lightglue_[mode]/[name] 下的 best_checkpoint）')
    parser.add_argument('--test_name', type=str, required=True, 
                        help='测试名称（结果保存在 results/lightglue_[mode]/[name]/[test_name] 下）')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='检查点路径（默认使用 best_checkpoint/model.ckpt）')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--gpus', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--img_size', type=int, default=512, help='图像大小')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 导入必要的模块（统一使用 CFFA 数据集）
    if args.mode == 'gen':
        from scripts.v2.train_onGen_vessels import (
            PL_LightGlue_Gen, 
            get_default_config
        )
        pl_class = PL_LightGlue_Gen
        mode_dir = 'lightglue_gen'
    else:  # cffa
        from scripts.v2.train_onReal import (
            PL_LightGlue_Real,
            get_default_config
        )
        pl_class = PL_LightGlue_Real
        mode_dir = 'lightglue_cffa'
    
    # 获取配置
    config = get_default_config()
    pl.seed_everything(config.TRAINER.SEED)
    
    # 确定checkpoint路径
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(f"results/{mode_dir}/{args.name}/best_checkpoint/model.ckpt")
    
    if not ckpt_path.exists():
        logger.error(f"检查点不存在: {ckpt_path}")
        logger.info(f"请确保训练模型存在，或使用 --checkpoint 指定有效的检查点路径")
        return
    
    logger.info(f"加载检查点: {ckpt_path}")
    
    # 设置输出目录
    output_dir = Path(f"results/{mode_dir}/{args.name}/{args.test_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    log_file = output_dir / "test_log.txt"
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="w")
    logger.info(f"日志将保存到: {log_file}")
    
    # GPU配置
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
    
    logger.info(f"测试模式: {args.mode}")
    logger.info(f"模型名称: {args.name}")
    logger.info(f"测试名称: {args.test_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"GPU配置: devices={gpus_list}, num_gpus={_n_gpus}")
    
    # 从检查点加载模型
    model = pl_class.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        result_dir=str(output_dir)
    )
    model.eval()
    
    # 统一使用 CFFA 完整数据集（训练集+验证集合并）进行测试
    script_dir = Path(__file__).parent.parent.parent
    cffa_data_dir = script_dir / 'data' / 'CFFA'
    
    # 导入 CFFA 数据集相关模块
    from data.CFFA.cffa_dataset import CFFADataset
    from scripts.v2.train_onReal import RealDatasetWrapper
    
    # 加载训练集
    train_base = CFFADataset(root_dir=str(cffa_data_dir), split='train', mode='cf2fa')
    train_dataset = RealDatasetWrapper(train_base, split_name='test')
    
    # 加载验证集（相当于测试集）
    val_base = CFFADataset(root_dir=str(cffa_data_dir), split='val', mode='cf2fa')
    val_dataset = RealDatasetWrapper(val_base, split_name='test')
    
    # 合并训练集和验证集作为完整测试集
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    
    test_dataloader = torch.utils.data.DataLoader(
        combined_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"CFFA完整数据集: 训练集 {len(train_dataset)} + 验证集 {len(val_dataset)} = {len(combined_dataset)} 样本")
    
    logger.info(f"开始测试 (模式: {args.mode} | 模型: {args.name})")
    
    # 使用 run_evaluation 函数进行测试（带可视化）
    set_metrics_verbose(True)
    metrics = run_evaluation(
        model, 
        test_dataloader, 
        mode=args.mode, 
        verbose=True,
        save_visualizations=True,  # 启用可视化
        output_dir=output_dir       # 保存到输出目录
    )
    
    # 保存测试总结
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("测试总结\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Test Name: {args.test_name}\n")
        f.write(f"Model Name: {args.name}\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"匹配成功样本数: {metrics['success_samples']}\n")
        f.write(f"匹配失败样本数: {metrics['failed_samples']}\n")
        f.write(f"匹配失败率: {metrics['match_failure_rate']:.4f}\n")
        f.write(f"Inaccurate 样本数: {metrics['inaccurate_samples']}\n")
        f.write(f"Acceptable 样本数: {metrics['acceptable_samples']}\n")
        f.write(f"MSE (仅 Acceptable): {metrics['mse']:.6f}\n")
        f.write(f"MACE (仅 Acceptable): {metrics['mace']:.4f}\n")
        f.write(f"AUC@5: {metrics['auc@5']:.4f}\n")
        f.write(f"AUC@10: {metrics['auc@10']:.4f}\n")
        f.write(f"AUC@20: {metrics['auc@20']:.4f}\n")
        f.write(f"mAUC: {metrics['mAUC']:.4f}\n")
        f.write(f"Combined AUC: {metrics['combined_auc']:.4f}\n")
        f.write(f"Inverse MACE: {metrics['inverse_mace']:.6f}\n")
    
    logger.info(f"测试总结已保存到: {summary_path}")
    logger.info(f"测试完成! 结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
