import sys
import os

# 添加项目根目录到 sys.path (必须在导入 src 之前)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # LoFTR-v1 根目录
sys.path.insert(0, project_root)

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
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
import logging

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.lightning.lightning_loftr import PL_LoFTR
from src.utils.plotting import make_matching_figures
from src.utils.metrics import error_auc, set_metrics_verbose

# 导入生成数据集（按 plan_260227.md）
import importlib.util
spec = importlib.util.spec_from_file_location("dataset_module", 
    "/data/student/Fengjunming/LoFTR-v1/data/260227_2_v29_2_1/260227_2_v29_2_1_dataset.py")
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
MultiModalDataset = dataset_module.MultiModalDataset

# 导入真实数据集（用于验证）
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset

# 导入域随机化函数（从 v6 目录）
spec_enhance = importlib.util.spec_from_file_location("gen_data_enhance_module",
    os.path.join(current_dir, "gen_data_enhance.py"))
gen_data_enhance_module = importlib.util.module_from_spec(spec_enhance)
spec_enhance.loader.exec_module(gen_data_enhance_module)
random_domain_augment_image = gen_data_enhance_module.random_domain_augment_image

# ==========================================
# 辅助类: RealDatasetWrapper (格式转换，用于真实数据验证集)
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
            'dataset_name': 'multimodal'  # 修复：使用 supervision.py 支持的名称
        }

# ==========================================
# 数据模块：使用生成数据集训练 + 生成&真实数据集验证
# ==========================================
class VesselGenDataModule(pl.LightningDataModule):
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
            # 训练集：使用生成数据集
            self.train_dataset = MultiModalDataset(
                root_dir='data/260227_2_v29_2_1',
                mode='cffa',
                split='train',
                img_size=self.args.img_size
            )
            
            # 验证集：只使用真实数据（简化流程）
            real_val_base = CFFADataset(
                root_dir='data/operation_pre_filtered_cffa',
                split='val',
                mode='fa2cf'
            )
            self.val_dataset = RealDatasetWrapper(real_val_base)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        # 只返回真实数据验证集
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            shuffle=False, 
            **self.loader_params
        )

# ==========================================
# 辅助工具
# ==========================================
def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
    """
    单应矩阵防爆锁 (Sanity Check for Homography)
    
    用于过滤掉物理上不可能的单应矩阵，避免极端离群值污染 MACE 指标。
    
    Parameters:
        H: [3, 3] 单应矩阵
        scale_min: 最小面积缩放倍数 (默认 0.1，即允许缩小到原来的 1/10)
        scale_max: 最大面积缩放倍数 (默认 10.0，即允许放大到原来的 10 倍)
        perspective_threshold: 透视参数阈值 (H[2,0] 和 H[2,1] 的绝对值上限)
    
    Returns:
        bool: 如果矩阵符合物理常识则返回 True，否则返回 False
    
    检查项：
        1. 矩阵是否包含 NaN 或 Inf
        2. 行列式是否为负 (镜像翻转，不可能)
        3. 行列式是否过大或过小 (极端面积缩放，不可能)
        4. 透视参数是否过大 (角点会被投射到无限远)
    """
    if H is None:
        return False
    if np.isnan(H).any() or np.isinf(H).any():
        return False
    
    # 计算左上角 2x2 仿射矩阵的行列式 (代表面积缩放倍数)
    det = np.linalg.det(H[:2, :2])
    
    # det < 0 表示图像发生了镜像翻转 (不可能)
    # 面积缩小到原来的 1/10 以下，或放大 10 倍以上 (极端拉伸，不可能)
    if det < scale_min or det > scale_max:
        return False
    
    # 检查透视参数 (H[2,0] 和 H[2,1])
    # 如果这两个值太大，图像的角点会被投射到无限远，导致 MACE 爆炸
    if abs(H[2, 0]) > perspective_threshold or abs(H[2, 1]) > perspective_threshold:
        return False
    
    return True

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

# 日志配置
loguru_logger = get_rank_zero_only_logger(logger)
loguru_logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
loguru_logger.add(sys.stderr, format=log_format, level="INFO")

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# ==========================================
# 核心模型: PL_LoFTR_VesselSup (基于 Vessel Segmentation 监督)
# ==========================================
class PL_LoFTR_VesselSup(PL_LoFTR):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, result_dir=None):
        super().__init__(config, pretrained_ckpt, profiler, dump_dir)
        self.result_dir = result_dir
        
        # 🌟 1. 新增：一个极度轻量的辅助分割头
        # 假设 LoFTR coarse level 的特征图通道数是 config.LOFTR.COARSE.D_MODEL (通常是 256)
        coarse_dim = config.LOFTR.COARSE.D_MODEL  # 通常是 256
        self.seg_head = nn.Sequential(
            nn.Conv2d(coarse_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1) # 输出单通道 logits
        )
        
        # 使用 hook 拦截 backbone 的输出，避免修改源码
        self._feat_c0 = None
        self._feat_c1 = None
        self.matcher.backbone.register_forward_hook(self._backbone_hook)

    def _backbone_hook(self, module, input, output):
        # output 是 (feat_c, feat_f)
        feat_c = output[0]
        # forward 会依次传入 image0, image1
        if self._feat_c0 is None:
            self._feat_c0 = feat_c
        elif self._feat_c1 is None:
            self._feat_c1 = feat_c

    def _apply_domain_randomization(self, batch):
        """
        在训练循环中对 image0 (CF) 和 image1 (FA deformed) 进行强烈的域随机化
        使用 v6/gen_data_enhance.py 中的 random_domain_augment_image
        
        注意：域随机化是数据增强操作，不参与梯度计算
        """
        # random_domain_augment_image 接受单张图像 [C, H, W] 或 [H, W] 格式
        # 对 batch 中的每张图像应用域随机化（50% 概率）
        batch_size = batch['image0'].shape[0]
        
        with torch.no_grad():
            for b in range(batch_size):
                # 50% 正常生成数据 + 50% 极端域随机化数据
                if torch.rand(1).item() < 0.5:
                    # 应用域随机化到 image0 (CF Fix)
                    img0 = batch['image0'][b].detach()  # [1, H, W]
                    img0_aug = random_domain_augment_image(img0)
                    batch['image0'][b] = img0_aug.to(img0.device)
                    
                    # 应用域随机化到 image1 (FA deformed Moving)
                    img1 = batch['image1'][b].detach()  # [1, H, W]
                    img1_aug = random_domain_augment_image(img1)
                    batch['image1'][b] = img1_aug.to(img1.device)
        
        return batch
    
    def _warp_image_pytorch(self, image, H_matrix):
        """
        使用 PyTorch 可微分操作进行透视变换
        image: [H, W] tensor
        H_matrix: [3, 3] numpy array (单应矩阵)
        返回: [H, W] tensor (可微分)
        """
        H, W = image.shape
        device = image.device
        
        # 将单应矩阵转换为 torch tensor
        H_tensor = torch.from_numpy(H_matrix).float().to(device)
        
        # 创建归一化的网格坐标 [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 转换到像素坐标
        grid_x_px = (grid_x + 1) * W / 2
        grid_y_px = (grid_y + 1) * H / 2
        
        # 构建齐次坐标 [H, W, 3]
        ones = torch.ones_like(grid_x_px)
        coords = torch.stack([grid_x_px, grid_y_px, ones], dim=-1)  # [H, W, 3]
        
        # 应用单应矩阵变换 [H, W, 3] @ [3, 3]^T -> [H, W, 3]
        coords_warped = torch.matmul(coords, H_tensor.T)
        
        # 归一化齐次坐标
        coords_warped_norm = coords_warped[..., :2] / (coords_warped[..., 2:3] + 1e-8)
        
        # 转换回 [-1, 1] 范围
        grid_warped = torch.zeros_like(coords[..., :2])
        grid_warped[..., 0] = coords_warped_norm[..., 0] * 2 / W - 1  # x
        grid_warped[..., 1] = coords_warped_norm[..., 1] * 2 / H - 1  # y
        
        # 使用 grid_sample 进行双线性插值
        image_batch = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid_batch = grid_warped.unsqueeze(0)  # [1, H, W, 2]
        
        warped = F.grid_sample(
            image_batch, 
            grid_batch, 
            mode='bilinear',  # 使用双线性插值（可微分）
            padding_mode='zeros',
            align_corners=True
        )
        
        return warped.squeeze(0).squeeze(0)  # [H, W]
    
    def _compute_vessel_supervision_loss(self, batch):
        """
        【已废弃】基于 Vessel Segmentation 的监督损失
        
        ⚠️ 致命问题：梯度断裂
        ------------------
        这个函数尝试对匹配点 mkpts0_f 和 mkpts1_f 计算 MSE 损失。
        但是这些匹配点是通过以下不可导操作得到的：
        1. conf_matrix.max() - 取最大值（argmax，不可导）
        2. conf_matrix > threshold - 阈值判断（离散操作，不可导）
        3. 索引提取 - 根据 mask 提取点（index操作，不可导）
        
        结果：虽然能算出 loss 值，但梯度无法反向传播到 CNN 和 Transformer。
        模型的特征提取器收不到任何更新信号，等同于盲目乱猜。
        
        ✅ 正确方案：
        ------------------
        LoFTR 的原生 Loss 已经完美解决了这个问题：
        1. 它在 Coarse Level 对整个特征网格计算交叉熵（不需要取 argmax）
        2. 它使用 T_0to1 对特征网格进行可微分的 Warp 操作
        3. 梯度可以畅通无阻地流回 CNN Backbone 和 Transformer
        4. 完全监督了全局的几何结构
        
        因此，训练时只需要提供正确的 T_0to1，让 LoFTR 原生 Loss 工作即可。
        
        保留此函数仅用于说明问题，实际训练中已不再调用。
        """
        if 'mkpts0_f' not in batch or 'mkpts1_f' not in batch:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 获取匹配点
        mkpts0 = batch['mkpts0_f']  # [N, 2]
        mkpts1 = batch['mkpts1_f']  # [N, 2]
        m_bids = batch['m_bids']    # [N] batch indices
        
        if 'T_0to1' not in batch:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        batch_size = batch['image0'].shape[0]
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0
        
        for b in range(batch_size):
            mask_b = (m_bids == b)
            pts0 = mkpts0[mask_b]  # [M, 2] 保留计算图
            pts1 = mkpts1[mask_b]  # [M, 2] 保留计算图
            
            if len(pts0) == 0:
                continue
            
            # 获取 GT 单应矩阵 [3, 3] (从 image0 到 image1 的绝对真实变换)
            H_gt = batch['T_0to1'][b].to(pts0.device)
            
            # 将 pts0 用真实的 H_gt 投影过去
            pts0_homo = torch.cat([pts0, torch.ones_like(pts0[:, :1])], dim=-1)  # [M, 3]
            pts0_warped_homo = torch.matmul(pts0_homo, H_gt.T)
            pts0_warped = pts0_warped_homo[:, :2] / (pts0_warped_homo[:, 2:] + 1e-8)
            
            # 惩罚预测点 pts1 与真实对应点 pts0_warped 之间的距离
            # ⚠️ 问题：pts1 来自不可导的 argmax，这里的梯度传不回去！
            loss = F.mse_loss(pts1, pts0_warped)
            
            total_loss = total_loss + loss
            valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss / valid_samples
        
    def _trainval_inference(self, batch):
        """
        训练/验证推理：在训练时应用域随机化，然后进行匹配
        """
        self._feat_c0 = None
        self._feat_c1 = None
        
        # 1. 训练时应用域随机化
        if self.training:
            batch = self._apply_domain_randomization(batch)
        
        # 2. 确保图像在 [0, 1] 范围
        for img_key in ['image0', 'image1']:
            if img_key in batch:
                if batch[img_key].min() < 0:
                    batch[img_key] = (batch[img_key] + 1) / 2
                    
        # 3. 调用父类的推理
        super()._trainval_inference(batch)

    def training_step(self, batch, batch_idx):
        # 1. 前向推理，生成计算图图
        self._trainval_inference(batch)
        
        # 捕捉特征预测分割 Mask
        if self._feat_c0 is not None and self._feat_c1 is not None:
            batch['pred_mask0_logits'] = self.seg_head(self._feat_c0)
            batch['pred_mask1_logits'] = self.seg_head(self._feat_c1)
        self._feat_c0 = None
        self._feat_c1 = None

        # 获取原生的 LoFTR 配准 Loss (基于全局匹配)
        loss_loftr = batch.get('loss', torch.tensor(0.0, device=self.device))
        total_loss = loss_loftr
        
        # ==========================================================
        # 🌟 新增：计算辅助分割 Loss (Auxiliary Segmentation Loss)
        # ==========================================================
        loss_seg = torch.tensor(0.0, device=self.device)
        if 'vessel_mask0' in batch and 'pred_mask0_logits' in batch:
            batch_size = batch['image0'].shape[0]
            mask0_full = batch['vessel_mask0'].float()  
            if mask0_full.dim() == 4:
                mask0_full = mask0_full.squeeze(1) # ensure [B, H, W]
            
            # 生成对应的 mask1_full
            mask1_full_list = []
            for b in range(batch_size):
                m0 = mask0_full[b]
                H_mat = batch['T_0to1'][b].cpu().numpy()
                try:
                    H_inv = np.linalg.inv(H_mat)
                except np.linalg.LinAlgError:
                    H_inv = np.eye(3)
                m1 = self._warp_image_pytorch(m0, H_inv)
                m1 = (m1 > 0.5).float()
                mask1_full_list.append(m1)
            mask1_full = torch.stack(mask1_full_list)
            
            # 获取 Coarse 级别的分辨率 (通常是 1/8)
            hw0_c = batch.get('hw0_c', (batch['image0'].shape[2] // 8, batch['image0'].shape[3] // 8))
            hw1_c = batch.get('hw1_c', (batch['image1'].shape[2] // 8, batch['image1'].shape[3] // 8))
            
            # 下采样到特征图尺寸，使用 adaptive_max_pool2d 保留高亮边缘
            m0_c = F.adaptive_max_pool2d(mask0_full.unsqueeze(1), hw0_c) # [B, 1, H_c, W_c]
            m1_c = F.adaptive_max_pool2d(mask1_full.unsqueeze(1), hw1_c) # [B, 1, H_c, W_c]
            
            # ==========================================================
            # 🌟 关键修改：添加 pos_weight 解决严重的正负样本不平衡
            # 因为血管像素只占全图的 5%~10%，如果不加权重，模型会全部预测为 0（背景）来骗取低 Loss
            # 给正样本（即血管 = 1 的位置）加上 15 倍的权重，逼迫网络必须把血管找出来
            # ==========================================================
            pos_weight = torch.tensor([15.0], device=self.device)
            criterion_seg = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # 下面的计算代码保持不变
            loss_seg_0 = criterion_seg(batch['pred_mask0_logits'], m0_c)
            loss_seg_1 = criterion_seg(batch['pred_mask1_logits'], m1_c)
            
            loss_seg = (loss_seg_0 + loss_seg_1) / 2.0
            
            # 记录分割 Loss
            self.log('train/loss_seg', loss_seg, on_step=True, on_epoch=True, prog_bar=True)
            
        # ==========================================================
        # 🌟 联合 Loss：权重融合
        # ==========================================================
        alpha = 1.0 
        total_loss = loss_loftr + alpha * loss_seg
        
        # 记录日志
        self.log('train/loss_loftr', loss_loftr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_total', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # 兼容以前的名字
        self.log('train/loss', total_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # 只在真实数据上验证
        
        # 验证时开启详细日志
        set_metrics_verbose(True)
        
        # 验证时不应用域随机化，直接推理
        self._trainval_inference(batch)
        ret_dict, _ = self._compute_metrics(batch)
        
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if getattr(self, 'force_viz', False):
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        # 提取所有候选匹配点
        all_candidates = []
        if 'conf_matrix' in batch:
            conf_matrix = batch['conf_matrix']
            thr = self.matcher.coarse_matching.thr
            hw0c = batch['hw0_c']
            hw1c = batch['hw1_c']
            scale0 = batch['image0'].shape[2] / hw0c[1]
            scale1 = batch['image1'].shape[2] / hw1c[1]
            
            for b in range(conf_matrix.shape[0]):
                mask0 = conf_matrix[b].max(dim=1)[0] > thr
                indices0 = torch.where(mask0)[0]
                kpts0 = torch.stack([indices0 % hw0c[1], indices0 // hw0c[1]], dim=1).float() * scale0
                
                mask1 = conf_matrix[b].max(dim=0)[0] > thr
                indices1 = torch.where(mask1)[0]
                kpts1 = torch.stack([indices1 % hw1c[1], indices1 // hw1c[1]], dim=1).float() * scale1
                
                all_candidates.append({
                    'kpts0': kpts0.detach().cpu().numpy(),
                    'kpts1': kpts1.detach().cpu().numpy()
                })
        
        # 计算单应矩阵用于可视化和评估
        H_ests = []
        mkpts0 = batch.get('mkpts0_f', torch.empty(0, 2))
        mkpts1 = batch.get('mkpts1_f', torch.empty(0, 2))
        m_bids = batch.get('m_bids', torch.empty(0))
        
        for b in range(batch['image0'].shape[0]):
            mask_b = (m_bids == b)
            pts0 = mkpts0[mask_b].cpu().numpy()
            pts1 = mkpts1[mask_b].cpu().numpy()
            
            if len(pts0) >= 4:
                try:
                    H_est, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
                    if H_est is not None:
                        H_ests.append(H_est)
                    else:
                        H_ests.append(np.eye(3))
                except:
                    H_ests.append(np.eye(3))
            else:
                H_ests.append(np.eye(3))

        return {
            **ret_dict,
            'H_est': H_ests,
            'kpts_candidates': all_candidates,
            'loss_scalars': batch.get('loss_scalars', {}),
            'figures': figures,
        }

# ==========================================
# 回调逻辑: VesselSupervisedValidationCallback (单验证集：仅真实数据)
# ==========================================
class VesselSupervisedValidationCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val = -1.0
        self.result_dir = Path(f"results/{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 只记录真实数据的指标
        self.epoch_mses = []
        self.epoch_maces = []

        import csv
        self.csv_path = self.result_dir / "metrics.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Train Loss Seg", "Train Loss LoFTR", "Val MSE", "Val MACE", "Val AUC@5", "Val AUC@10", "Val AUC@20", "Val Combined AUC", "Val Inverse MACE"])
        
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
                    t.get('loss', t.get('loss_total', '')),
                    t.get('loss_seg', ''),
                    t.get('loss_loftr', ''),
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
        batch_mses, batch_maces = self._process_batch(
            trainer, pl_module, batch, outputs, None, 
            save_images=False
        )
        self.epoch_mses.extend(batch_mses)
        self.epoch_maces.extend(batch_maces)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        display_metrics = {}
        for k in ['loss', 'train/loss', 'train/loss_seg', 'train/loss_loftr']:
            epoch_key = f"{k}_epoch"
            if epoch_key in metrics:
                display_metrics[k.replace('train/', '')] = metrics[epoch_key].item()
        if display_metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
            loguru_logger.info(f"Epoch {epoch} 训练总结 >> {metric_str}")
        
        self.current_train_metrics[epoch] = display_metrics
        self._try_write_csv(epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        # 计算真实数据指标
        if not self.epoch_mses:
            loguru_logger.warning(f"Epoch {epoch}: 验证集没有返回结果")
            return
            
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else float('inf')
        
        display_metrics = {'mse': avg_mse, 'mace': avg_mace}
        
        # 从 metrics 中提取 AUC（单验证集时使用标准键名）
        # 统一使用 [5, 10, 20] 像素阈值
        for k in ['auc@5', 'auc@10', 'auc@20']:
            # 单验证集：优先使用标准键名（不带 dataloader_idx）
            standard_key = f"metrics/{k}"
            key_with_idx = f"{standard_key}/dataloader_idx_0"
            
            if standard_key in metrics:
                display_metrics[k] = metrics[standard_key].item()
            elif key_with_idx in metrics:
                # 向后兼容：如果有带后缀的键（不应该出现在单验证集）
                display_metrics[k] = metrics[key_with_idx].item()
            else:
                # 🔥 致命错误：找不到 AUC 指标
                raise KeyError(f"Epoch {epoch}: 未找到 {k} 指标！尝试了键名: '{standard_key}' 和 '{key_with_idx}'，可用键: {list(metrics.keys())}。请检查真实数据集是否提供了正确的 T_0to1 GT 标签")
        
        # 计算综合 AUC (主要指标)
        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        # ✅ 使用平均AUC作为主要评价指标（越大越好）
        inverse_mace = 1.0 / (1.0 + avg_mace)  # 辅助指标
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        loguru_logger.info(f"Epoch {epoch} 验证总结 >> {metric_str} | combined_auc: {combined_auc:.4f} | inverse_mace: {inverse_mace:.6f}")
        
        self.current_val_metrics[epoch] = {
            'mse': avg_mse,
            'mace': avg_mace,
            'auc5': auc5,
            'auc10': auc10,
            'auc20': auc20,
            'combined_auc': combined_auc,
            'inverse_mace': inverse_mace
        }
        self._try_write_csv(epoch)
        
        # 记录指标到logger（用于早停监控）
        pl_module.log("val_mse", avg_mse, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("val_mace", avg_mace, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("combined_auc", combined_auc, on_epoch=True, prog_bar=True, logger=True)  # 主要指标
        pl_module.log("inverse_mace", inverse_mace, on_epoch=True, prog_bar=False, logger=True)  # 辅助指标
        
        # 保存最新模型
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
            
        # ✅ 评价最优模型（基于平均AUC，越大越好）
        is_best = False
        if combined_auc > self.best_val:
            self.best_val = combined_auc
            is_best = True
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Best Combined AUC: {combined_auc:.4f} [主要指标：越大越好]\n")
                f.write(f"AUC@5: {auc5:.4f}\n")
                f.write(f"AUC@10: {auc10:.4f}\n")
                f.write(f"AUC@20: {auc20:.4f}\n")
                f.write(f"MACE: {avg_mace:.4f}\n")
                f.write(f"Inverse MACE: {inverse_mace:.6f}\n")
                f.write(f"MSE: {avg_mse:.6f}\n")
            loguru_logger.info(f"发现新的最优模型! Epoch {epoch}, Combined AUC: {combined_auc:.4f}")

        if is_best or (epoch % 5 == 0):
            self._trigger_visualization(trainer, pl_module, is_best, epoch)

    def _trigger_visualization(self, trainer, pl_module, is_best, epoch):
        pl_module.force_viz = True
        target_dir = self.result_dir / (f"epoch{epoch}_best" if is_best else f"epoch{epoch}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 只可视化真实数据验证集
        val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx > 5: break  # 只可视化前几个 batch
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = pl_module.validation_step(batch, batch_idx)
                self._process_batch(trainer, pl_module, batch, outputs, target_dir, save_images=True)
        pl_module.force_viz = False

    def _process_batch(self, trainer, pl_module, batch, outputs, epoch_dir, save_images=False):
        batch_size = batch['image0'].shape[0]
        mses, maces = [], []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        
        # 统计防爆锁触发次数（用于调试）
        rejected_count = 0
        
        for i in range(batch_size):
            H_est = H_ests[i]
            
            # ==========================================
            # 🛡️ 启用防爆锁
            # ==========================================
            if not is_valid_homography(H_est):
                # 将预测矩阵强制重置为单位矩阵 (即模型交白卷：我不形变了)
                H_est = np.eye(3)
                rejected_count += 1
            # ==========================================
            
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            
            # 简单计算 MSE（img1_result vs img1_gt）
            img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            try:
                mse = np.mean(((img1_result/255.) - (img1_gt/255.))**2)
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
                cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)
                
                # 绘制候选点和匹配点
                img0_kpts = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_kpts = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                
                if 'kpts_candidates' in outputs and len(outputs['kpts_candidates']) > i:
                    cands = outputs['kpts_candidates'][i]
                    for pt in cands['kpts0']:
                        cv2.circle(img0_kpts, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    for pt in cands['kpts1']:
                        cv2.circle(img1_kpts, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

                if 'm_bids' in batch:
                    mask_i = (batch['m_bids'] == i)
                    kpts0_m = batch['mkpts0_f'][mask_i].cpu().numpy()
                    kpts1_m = batch['mkpts1_f'][mask_i].cpu().numpy()
                    for pt in kpts0_m:
                        cv2.circle(img0_kpts, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                    for pt in kpts1_m:
                        cv2.circle(img1_kpts, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                
                cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_kpts)
                cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_kpts)

                try:
                    cb = create_chessboard(img1_result, img0)
                    cv2.imwrite(str(save_path / "chessboard.png"), cb)
                except: pass
                if 'figures' in outputs and len(outputs['figures'][pl_module.config.TRAINER.PLOT_MODE]) > i:
                    fig = outputs['figures'][pl_module.config.TRAINER.PLOT_MODE][i]
                    fig.savefig(str(save_path / "matches.png"), bbox_inches='tight')
                    plt.close(fig)
        
        # 如果有防爆锁触发，记录一下（可选，方便调试）
        if rejected_count > 0 and save_images:
            loguru_logger.info(f"防爆锁触发: {rejected_count}/{batch_size} 个样本的单应矩阵被重置为单位矩阵")
        
        return mses, maces

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def parse_args():
    parser = argparse.ArgumentParser(description="LoFTR Vessel-Supervised Training (按 plan_260227.md)")
    parser.add_argument('--mode', type=str, default='cffa', choices=['cffa'])
    parser.add_argument('--name', '-n', type=str, default='loftr_vessel_supervised_v1', help='训练名称')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--pretrained_ckpt', type=str, default='weights/outdoor_ds.ckpt')
    parser.add_argument('--start_point', type=str, default=None)
    parser.add_argument('--main_cfg_path', type=str, default=None)
    
    # 早停参数
    parser.add_argument('--early_stop_patience', type=int, default=15, help='早停的耐心值（多少个epoch没有提升就停止）')
    parser.add_argument('--early_stop_start_epoch', type=int, default=0, help='从第几个epoch开始启用早停（之前不会触发早停）')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0001, help='早停的最小改进阈值')
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=200, gpus='1')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_cfg_defaults()
    if args.main_cfg_path: config.merge_from_file(args.main_cfg_path)
    
    # 修复：与 Callback 中的路径保持一致
    result_dir = Path(f"results/{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.txt"
    
    # 重要：先移除之前的所有 handler，然后重新添加（确保在 main 中配置）
    logger.remove()  # 移除所有现有 handler
    logger.add(sys.stderr, format=log_format, level="INFO")  # 重新添加 stderr
    # 添加文件 handler（同步写入，确保日志不丢失）
    logger.add(log_file, format=log_format, level="INFO", mode="a", backtrace=True, diagnose=False)
    logger.info(f"日志将同时保存到: {log_file}")
    
    # 同时配置 loguru_logger（虽然它们是同一个对象，但为了保险起见）
    if loguru_logger is not logger:
        loguru_logger.add(log_file, format=log_format, level="INFO", mode="a", backtrace=True, diagnose=False)
    
    config.DATASET.MGDPT_IMG_RESIZE = args.img_size
    config.LOFTR.RESOLUTION = (8, 2)
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)
    
    _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    model = PL_LoFTR_VesselSup(
        config, 
        pretrained_ckpt=args.pretrained_ckpt, 
        result_dir=str(result_dir)
    )
    
    # 强制全权重加载
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}
        model.matcher.load_state_dict(state_dict, strict=False)
        loguru_logger.info(f"已加载全量预训练权重")
    
    data_module = VesselGenDataModule(args, config)
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=f"vessel_supervised_{args.name}")
    
    # 使用命令行参数配置早停
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=args.early_stop_start_epoch,
        monitor='combined_auc',  # ✅ 监控平均AUC（主要指标）
        mode='max',  # combined_auc 越大越好
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta
    )
    
    loguru_logger.info(f"早停配置: monitor=combined_auc, start_epoch={args.early_stop_start_epoch}, patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}")
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        num_sanity_val_steps=0,  # 启动前不运行验证
        check_val_every_n_epoch=1, # 每一轮都验证
        callbacks=[VesselSupervisedValidationCallback(args), LearningRateMonitor(logging_interval='step'), early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=False) if _n_gpus > 1 else None,
        replace_sampler_ddp=True,
        resume_from_checkpoint=args.start_point
    )
    
    loguru_logger.info(f"开始 Vessel-Supervised 训练 (按 plan_260227.md): {args.name}")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
