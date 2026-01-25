
import sys
import os
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，防止多线程绘图报错
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
from types import SimpleNamespace
import logging

# 导入本地实现的模块
from pl_lightglue import PL_LightGlue
# from data.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset # Deleted

# 导入真实数据集 (用于验证)
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# LightGlue 可视化工具
from lightglue import viz2d

# --- 配置辅助函数 ---
def get_default_config():
    """
    获取默认配置
    模拟 PL_LightGlue 所需的配置结构，替代原本缺失的 yacs config
    """
    conf = SimpleNamespace()
    conf.TRAINER = SimpleNamespace()
    conf.TRAINER.CANONICAL_BS = 4
    conf.TRAINER.CANONICAL_LR = 1e-4
    conf.TRAINER.EXP_NAME = "lightglue_multimodal"
    conf.TRAINER.TRUE_LR = 1e-4 # 实际学习率将在 main 函数中根据 BatchSize 自动调整
    conf.TRAINER.PLOT_MODE = 'evaluation'
    conf.MATCHING = {
        'input_dim': 256,
        'descriptor_dim': 256
    }
    return conf

# --- 真实数据集包装器 ---
class RealDatasetWrapper(torch.utils.data.Dataset):
    """
    包装真实数据集，使其输出格式与 MultiModalDataset (生成数据) 一致。
    主要作用是：
    1. 统一图像数值范围到 [0, 1]
    2. 统一转为单通道灰度图
    3. 统一返回字典格式
    4. 统一变换矩阵方向
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 真实数据集返回格式: (fix, moving_orig, moving_gt, fix_path, moving_path, T_0to1)
        # 注意：真实数据集的 T_0to1 实际上是从 moving 到 fix 的变换 (fix = H * moving)
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        
        # 将 [-1, 1] 范围转换回 [0, 1]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
        # 转灰度图
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
        
        # 统一变换矩阵方向：LightGlue 计算的是 Image0 -> Image1 的映射
        # 这里 Image0 是 Fix, Image1 是 Moving
        # 真实数据集的 T_0to1 是 Image1 -> Image0 (fix = T * moving)
        # 所以我们需要取逆矩阵来得到 correct 的 T_0to1
        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
            T_fix_to_moving = T_0to1
            
        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray, # 用于计算 MSE
            'T_0to1': T_fix_to_moving, # GT: Fix -> Moving
            'pair_names': (fix_name, moving_name),
            'dataset_name': 'MultiModal'
        }

# --- Data Module ---
# --- Data Module ---
class MultimodalDataModule(pl.LightningDataModule):
    """Lightning 数据模块，负责加载训练和验证数据"""
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def _get_dataset(self, split):
        # 根据模式选择对应的 Dataset
        if self.args.mode == 'cfocta':
            base_dataset = CFOCTADataset(root_dir='data/CF_OCTA_v2_repaired', split=split, mode='cf2octa')
        elif self.args.mode == 'cffa':
            base_dataset = CFFADataset(root_dir='data/operation_pre_filtered_cffa', split=split, mode='fa2cf')
        elif self.args.mode == 'cfoct':
            base_dataset = CFOCTDataset(root_dir='data/operation_pre_filtered_cfoct', split=split, mode='cf2oct')
        elif self.args.mode == 'octfa':
            base_dataset = OCTFADataset(root_dir='data/operation_pre_filtered_octfa', split=split, mode='fa2oct')
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")
        return RealDatasetWrapper(base_dataset)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # 训练集：真实数据集
            self.train_dataset = self._get_dataset('train')
            # 验证集：真实数据集
            self.val_dataset = self._get_dataset('val')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

# --- 工具函数 ---
def compute_tre(H_est, H_gt, height, width, grid_size=10):
    """
    计算 Target Registration Error (TRE) 的近似值。
    由于真实数据集的GT点较少，且 MACE (Corner Error) 在图像边缘容易发散，
    这里改为计算图像内部 ROI 区域内网格点的平均重投影误差。
    """
    # 定义 ROI (Region of Interest) 范围：避开图像边缘 (例如取中心 ~80% 区域)
    margin_w = width * 0.1
    margin_h = height * 0.1
    
    # 生成网格点
    x = np.linspace(margin_w, width - margin_w, grid_size)
    y = np.linspace(margin_h, height - margin_h, grid_size)
    xv, yv = np.meshgrid(x, y)
    
    # [N, 2]
    pts = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)
    
    # 齐次坐标 [N, 3]
    pts_homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    
    # GT 变换后的点
    pts_gt_homo = (H_gt @ pts_homo.T).T
    pts_gt = pts_gt_homo[:, :2] / (pts_gt_homo[:, 2:] + 1e-6)
    
    # 预测变换后的点
    pts_est_homo = (H_est @ pts_homo.T).T
    pts_est = pts_est_homo[:, :2] / (pts_est_homo[:, 2:] + 1e-6)
    
    try:
        # 计算每个点的 L2 距离，然后求平均
        errors = np.sqrt(np.sum((pts_est - pts_gt)**2, axis=1))
        # 排除极大值 (防止溢出或数值不稳定影响整体)
        valid_errors = errors[errors < 1000] 
        if len(valid_errors) > 0:
            tre = np.mean(valid_errors)
        else:
             tre = float('inf')
    except:
        tre = float('inf')
    return tre

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

# --- 验证回调 ---
class MultimodalValidationCallback(Callback):
    """
    Validation Callback
    负责在每个 epoch 结束时：
    1. 保存可视化结果 (配准图、棋盘图、匹配连线图)
    2. 计算并记录评估指标 (MSE, MACE)
    3. 保存最新和最优模型检查点
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val = float('inf')
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_tres = []
        self.cached_results = [] # 缓存当前 epoch 的结果用于后续绘图

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_tres = []
        self.cached_results = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 真实数据：计算指标
        batch_mses, batch_tres = self._compute_batch_metrics(outputs, batch)
        self.epoch_mses.extend(batch_mses)
        self.epoch_tres.extend(batch_tres)
        
        # 缓存数据
        self.cached_results.append({
            'batch': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
            'outputs': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()} if isinstance(outputs, dict) else outputs,
            'mses': batch_mses,
            'tres': batch_tres
        })

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.epoch_mses or trainer.sanity_checking:
            # Sanity check 还是要画一下图确认流程正常
            if trainer.sanity_checking and self.cached_results:
                self._plot_cached_results(trainer, "step0")
            return
            
        epoch = trainer.current_epoch + 1
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_tre = sum(self.epoch_tres) / len(self.epoch_tres)
        
        logger.info(f"Epoch {epoch} Validation: MSE: {avg_mse:.6f}, TRE: {avg_tre:.4f}")
        
        pl_module.log("val_mse", avg_mse, on_epoch=True)
        pl_module.log("val_tre", avg_tre, on_epoch=True)
        
        # 判定是否需要可视化
        is_best = avg_tre < self.best_val
        is_plot_epoch = (epoch % self.args.plot_every_n_epoch == 0)
        
        if is_best or is_plot_epoch:
            sub_dir = f"epoch{epoch}"
            if is_best:
                sub_dir += "_BEST"
            self._plot_cached_results(trainer, sub_dir)

        # 保存最新模型
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
        
        # 只要产生更好的 TRE 就保存最优模型
        if is_best:
            self.best_val = avg_tre
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest TRE: {avg_tre:.6f}\nBest MSE: {avg_mse:.6f}")
            logger.info(f"New Best Model Saved at Epoch {epoch} with TRE {avg_tre:.4f}")

    def _compute_batch_metrics(self, outputs, batch):
        """仅计算指标"""
        batch_size = batch['image0'].shape[0]
        mses, tres = [], []
        H_ests = outputs['H_est']
        Ts_gt = batch['T_0to1'].cpu().numpy()
        
        for i in range(batch_size):
            img0 = batch['image0'][i, 0].cpu().numpy()
            ref_key = 'image1_gt' if 'image1_gt' in batch else 'image1_origin'
            img1_gt = batch[ref_key][i, 0].cpu().numpy()
            H_est = H_ests[i]
            
            # Warp (用于计算 MSE)
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(batch['image1'][i, 0].cpu().numpy(), H_inv, (w, h))
                mse = np.mean((img1_result - img1_gt) ** 2)
            except:
                mse = 1.0
            
            tre = compute_tre(H_est, Ts_gt[i], h, w)
            mses.append(mse)
            tres.append(tre)
        return mses, tres

    def _plot_cached_results(self, trainer, sub_dir_name):
        """执行绘图操作"""
        logger.info(f"Plotting visualization for {sub_dir_name}...")
        for res in self.cached_results:
            batch = res['batch']
            outputs = res['outputs']
            
            epoch_dir = self.result_dir / f"{sub_dir_name}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            
            # 复用之前的绘图逻辑
            self._save_visualizations(batch, outputs, epoch_dir)

    def _save_visualizations(self, batch, outputs, epoch_dir):
        """真正的绘图和写磁盘操作"""
        batch_size = batch['image0'].shape[0]
        H_ests = outputs['H_est']
        pair_names0 = batch['pair_names'][0]
        pair_names1 = batch['pair_names'][1]
        
        # 提取关键点 (如果存在)
        has_kpts = 'kpts0' in outputs and 'kpts1' in outputs

        for i in range(batch_size):
            sample_name = f"{Path(pair_names0[i]).stem}_vs_{Path(pair_names1[i]).stem}"
            save_path = epoch_dir / sample_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            img0 = (batch['image0'][i, 0].numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].numpy() * 255).astype(np.uint8)
            H_est = H_ests[i]
            h, w = img0.shape
            
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            
            cv2.imwrite(str(save_path / "fix.png"), img0)
            cv2.imwrite(str(save_path / "moving.png"), img1)
            cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
            cv2.imwrite(str(save_path / "chessboard.png"), create_chessboard(img1_result, img0))
            
            # --- 新增：可视化特征提取点 ---
            if has_kpts:
                try:
                    kpts0 = outputs['kpts0'][i].numpy()
                    kpts1 = outputs['kpts1'][i].numpy()
                    
                    fig_kpts = plt.figure(figsize=(10, 4))
                    viz2d.plot_images([img0, img1], titles=['Fix Keypoints', 'Moving Keypoints'])
                    viz2d.plot_keypoints([kpts0, kpts1], colors='red', ps=2)
                    plt.savefig(str(save_path / "keypoints.png"), bbox_inches='tight')
                    plt.close(fig_kpts)
                except Exception as e:
                    logger.warning(f"Failed to plot keypoints for {sample_name}: {e}")
            
            # --- 原有：可视化匹配 ---
            fig = None
            try:
                m0 = outputs['matches0'][i]
                valid = m0 > -1
                m_indices_0 = torch.where(valid)[0].numpy()
                m_indices_1 = m0[valid].numpy()
                
                # 如果前面没取 kpts，这里取一下
                if 'kpts0' not in locals():
                    kpts0 = outputs['kpts0'][i].numpy()
                    kpts1 = outputs['kpts1'][i].numpy()
                
                fig = plt.figure(figsize=(10, 4))
                viz2d.plot_images([img0, img1], titles=['Fix', 'Moving'])
                viz2d.plot_matches(kpts0[m_indices_0], kpts1[m_indices_1], color='lime', lw=0.5)
                plt.savefig(str(save_path / "matches.png"), bbox_inches='tight')
            except Exception as e:
                pass
            finally:
                if fig is not None: plt.close(fig)

# --- 自定义早停 ---
class DelayedEarlyStopping(EarlyStopping):
    """自定义早停，在指定的 start_epoch 之后才开始检测"""
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            print(f"DEBUG: Skipping EarlyStopping check at epoch {trainer.current_epoch} (Warmup < {self.start_epoch})")
            return
        super().on_validation_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            print(f"DEBUG: Skipping EarlyStopping check at epoch {trainer.current_epoch} (Warmup < {self.start_epoch})")
            return
        super().on_validation_epoch_end(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return
        super().on_train_epoch_end(trainer, pl_module)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'], help='配准模式')
    parser.add_argument('--name', '-n', type=str, default='lightglue_multimodal_new', help='实验名称')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--vessel_sigma', type=float, default=6.0, help='虽然此处不用mask loss，但Dataset依然需要此参数')
    parser.add_argument('--use_domain_randomization', action='store_true', default=True, help='是否启用域随机化增强')
    # parser.add_argument('--val_on_real', action='store_true', default=True, help='是否在真实数据集上验证')
    parser.add_argument('--data_root', type=str, default="data/FIVES_extract_v2")
    
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--plot_every_n_epoch', type=int, default=5, help='每隔多少 epoch 保存一次可视化结果')
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 配置日志保存 ---
    # 确保实验结果目录存在
    result_dir = Path(f"results/{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 添加文件日志 (追加模式)
    logger.add(result_dir / "train_log.txt", rotation="10 MB", level="INFO", mode="a")

    config = get_default_config()
    
    pl.seed_everything(66)
    
    # 初始化模型
    model = PL_LightGlue(config)
    # 初始化数据
    data_module = MultimodalDataModule(args, config)
    
    # 验证回调
    val_callback = MultimodalValidationCallback(args)
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # 早停策略 (前50 epoch 不停，之后如果 TRE 在 20 epoch 内不下降则停)
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=100, monitor='val_tre', patience=20, mode='min', min_delta=0.01, check_finite=False
    )
    
    # 解析 GPU 配置
    # PL 2.0 推荐使用 accelerator="gpu", devices=[...]
    # 为了兼容以前的用法 (gpus='1' 或 gpus='0,1') 这里的简单处理：
    if ',' in args.gpus:
        devices = [int(x) for x in args.gpus.split(',')]
    else:
        try:
            devices = [int(args.gpus)]
        except ValueError:
             # 如果传入的是 "auto" 或者其他非数字
             devices = "auto"

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        check_val_every_n_epoch=1, # 每 epoch 验证一次，确保 LR Scheduler 有指标可查
        num_sanity_val_steps=2,    # 训练前跑2个 batch 检查流程
        callbacks=[val_callback, lr_monitor, early_stop_callback],
        logger=TensorBoardLogger('logs/tb_logs', name=args.name),
        # replace_sampler_ddp=False, # PL 2.0 不需要显式设置，除非有特殊需求
        strategy="ddp_find_unused_parameters_true" if len(devices) > 1 and isinstance(devices, list) else "auto"
    )
    
    logger.info(f"Starting training: {args.name}")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
