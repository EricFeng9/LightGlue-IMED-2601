import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class MultiModalDataset(Dataset):
    """
    多模态眼底图像数据集加载器 (FIVES数据集版本)
    支持 CF, FA, OCT, OCTA 四种模态及其组合配准训练。
    依据 plan.md 实现随机仿射变换 (T) 的在线生成，包含血管分割掩码引导。
    """
    def __init__(self, root_dir, mode='cffa', split='train', img_size=518, df=8, vessel_sigma=6.0):
        """
        Args:
            root_dir (str): 数据集根目录路径。
            mode (str): 配准模式：'cffa', 'cfoct', 'octfa', 'cfocta'。
            split (str): 'train' (训练) 或 'val' (验证) 或 'test' (测试)。
            img_size (int): 图像统一缩放尺寸。
            df (int): 特征图下采样倍率 (默认 8，对应 LoFTR 1/8 粗级特征)。
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.split = split
        self.img_size = img_size
        self.df = df
        self.vessel_sigma = float(vessel_sigma)
        
        # 列出所有子文件夹 (格式: 数字_字母，如 1_A, 600_N)
        # 过滤掉不符合命名规则的文件夹（如 .git, __pycache__ 等）
        def is_valid_dir(d):
            if not d.is_dir():
                return False
            parts = d.name.split('_')
            if len(parts) != 2:
                return False
            try:
                int(parts[0])  # 第一部分必须是数字
                return len(parts[1]) > 0  # 第二部分不能为空
            except ValueError:
                return False
        
        all_dirs = sorted([d for d in self.root_dir.iterdir() if is_valid_dir(d)],
                         key=lambda x: (int(x.name.split('_')[0]), x.name.split('_')[1]))
        
        # 随机打乱 (使用固定种子保证 train 和 val 划分的一致性，且不重叠)
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(all_dirs))
        all_dirs = [all_dirs[i] for i in indices]
        
        # 8:2 划分
        num_total = len(all_dirs)
        num_train = int(num_total * 0.8)
        
        if split == 'train':
            self.dirs = all_dirs[:num_train]
        elif split == 'val':
            self.dirs = all_dirs[num_train:]
        else: # test
            self.dirs = all_dirs[num_train:] # 默认测试集也使用验证集部分数据，除非有独立测试集
        
        # 筛选出包含指定模式所需文件的文件夹
        self.valid_dirs = []
        for d in self.dirs:
            paths = self._get_image_paths(d)
            # paths = (fixed_path, moving_path, vessel_mask_path)
            if paths[0] is not None and paths[1] is not None and paths[2] is not None:
                self.valid_dirs.append((d, paths))
        
        # 验证集子采样：如果是验证集，且样本数大于 12，则随机抽取 12 组
        if self.split == 'val' and len(self.valid_dirs) > 12:
            # 使用独立随机状态进行采样，避免受全局 seed_everything 影响（如果希望每次启动不同）
            # 或者直接采样，受全局种子影响（如果希望运行内和多次运行间稳定）
            # 这里采用直接采样，因为通常希望本次训练中的验证集是固定的
            indices = np.random.choice(len(self.valid_dirs), 12, replace=False)
            self.valid_dirs = [self.valid_dirs[i] for i in indices]
        
        print(f"模式 {mode} ({split} 分割) 已加载 {len(self.valid_dirs)} 对数据")

    def _get_image_paths(self, folder):
        """依据配准模式，在文件夹中检索对应的多模态图像对路径和血管掩码路径"""
        folder_id = folder.name
        fixed_path = None
        moving_path = None
        vessel_mask_path = None
        
        if self.mode == 'cffa':
            # CF (固定) vs FA (待配准)
            cf_options = [f"{folder_id}_cf_512.png", f"{folder_id}_cf.png"]
            fa_options = [f"{folder_id}_fa_gen.png", f"{folder_id}_fa.png"]
            fixed_path = self._find_first_exists(folder, cf_options)
            moving_path = self._find_first_exists(folder, fa_options)
            vessel_mask_path = self._find_first_exists(folder, [f"{folder_id}_seg.png"])
            
        elif self.mode == 'cfoct':
            # CF (固定) vs OCT (待配准)
            cf_options = [f"{folder_id}_cf_512.png", f"{folder_id}_cf.png"]
            oct_options = [f"{folder_id}_oct_gen.png", f"{folder_id}_oct.png"]
            fixed_path = self._find_first_exists(folder, cf_options)
            moving_path = self._find_first_exists(folder, oct_options)
            vessel_mask_path = self._find_first_exists(folder, [f"{folder_id}_seg.png"])
            
        elif self.mode == 'octfa':
            # OCT (固定) vs FA (待配准)
            oct_options = [f"{folder_id}_oct_gen.png", f"{folder_id}_oct.png"]
            fa_options = [f"{folder_id}_fa_gen.png", f"{folder_id}_fa.png"]
            fixed_path = self._find_first_exists(folder, oct_options)
            moving_path = self._find_first_exists(folder, fa_options)
            vessel_mask_path = self._find_first_exists(folder, [f"{folder_id}_seg.png"])
            
        elif self.mode == 'cfocta':
            # CF_clip (固定) vs OCTA (待配准)
            cf_clip_options = [f"{folder_id}_cf_clip_512.png", f"{folder_id}_cf_clip.png"]
            octa_options = [f"{folder_id}_octa_gen.png", f"{folder_id}_octa.png"]
            fixed_path = self._find_first_exists(folder, cf_clip_options)
            moving_path = self._find_first_exists(folder, octa_options)
            vessel_mask_path = self._find_first_exists(folder, [f"{folder_id}_seg_clip.png"])
            
        return fixed_path, moving_path, vessel_mask_path

    def _find_first_exists(self, folder, filenames):
        """辅助函数：按优先级顺序查找第一个存在的文件"""
        for f in filenames:
            path = folder / f
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.valid_dirs)

    def _compute_soft_vessel_weight(self, vessel_mask_bin):
        """
        基于二值血管掩码计算高斯软掩码:
        1. 对反掩码做距离变换，得到每个像素到最近血管的距离 D(x)
        2. 使用高斯核 W(x) = exp(-D(x)^2 / (2*sigma^2)), sigma 以像素为单位
        """
        # vessel_mask_bin: [H, W], 0/1
        inv_mask = (1 - vessel_mask_bin).astype(np.uint8)
        # OpenCV distanceTransform 要求非零为前景，这里用反掩码，保证血管内部 D=0
        dist = cv2.distanceTransform(inv_mask, distanceType=cv2.DIST_L2, maskSize=3)
        sigma = max(self.vessel_sigma, 1e-3)
        # 为避免数值下溢，可裁剪最大距离（3*sigma 之后权重已接近 0）
        max_dist = 3.0 * sigma
        dist = np.clip(dist, 0.0, max_dist)
        weight = np.exp(-(dist ** 2) / (2 * sigma * sigma))
        return weight.astype(np.float32)

    def _get_random_affine(self, rng=None):
        """
        生成随机仿射变换 T，支持大角度旋转和翻转（参考 plan_v2_1.md）
        返回: (M, flip_h, flip_v)
            M: 2x3 仿射矩阵
            flip_h: 是否水平翻转
            flip_v: 是否垂直翻转
        """
        if rng is None:
            rng = np.random
            
        # 旋转角度：均匀采样 [-90, 90] (参考 plan_v2_1.md)
        angle = rng.uniform(-90, 90)
        
        # 缩放因子：0.8~1.2
        scale = rng.uniform(0.8, 1.2)
        
        # 平移量：±20% 的图像尺寸
        tx = rng.uniform(-0.2, 0.2) * self.img_size
        ty = rng.uniform(-0.2, 0.2) * self.img_size
        
        # 随机翻转：20% 概率 (参考 plan_v2_1.md)
        flip_h = rng.rand() < 0.2  # 水平翻转
        flip_v = rng.rand() < 0.2  # 垂直翻转
        
        center = (self.img_size // 2, self.img_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        return M.astype(np.float32), flip_h, flip_v

    def __getitem__(self, idx):
        folder, (fixed_path, moving_path, vessel_mask_path) = self.valid_dirs[idx]
        
        # 以灰度图读取图像
        img0 = cv2.imread(str(fixed_path), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(moving_path), cv2.IMREAD_GRAYSCALE)
        
        # 读取血管分割掩码 (二值图)
        vessel_mask = cv2.imread(str(vessel_mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 统一缩放到目标尺寸
        img0 = cv2.resize(img0, (self.img_size, self.img_size))
        img1 = cv2.resize(img1, (self.img_size, self.img_size))
        vessel_mask = cv2.resize(vessel_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # 二值化掩码 (0/1)
        vessel_mask_bin = (vessel_mask > 127).astype(np.float32)
        # 基于二值掩码计算高斯软掩码 (固定图上的权重)
        vessel_weight0 = self._compute_soft_vessel_weight(vessel_mask_bin)
        
        # 原始移动图 (未形变)
        img1_origin = img1.copy()
        
        if self.split in ['train', 'val']:
            # 训练或验证阶段，对待配准图像 (img1) 和掩码应用随机仿射变换 T
            if self.split == 'val':
                # 验证阶段使用固定的随机种子，保证每次验证时的形变一致
                rng = np.random.RandomState(idx)
                T, flip_h, flip_v = self._get_random_affine(rng)
            else:
                T, flip_h, flip_v = self._get_random_affine()
            
            # 方案B: 建立有效区域掩码 (整个眼底可见区域，用于Transformer注意力)
            # 使用简单的阈值提取眼底轮廓
            valid_mask0_orig = (img0 > 10).astype(np.float32)
            valid_mask1_orig = (img1 > 10).astype(np.float32)
            
            # 同步变换图像和掩码 / 软权重 / 有效区域掩码
            img1_warped = cv2.warpAffine(img1, T, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR)
            mask1_warped = cv2.warpAffine(vessel_mask_bin, T, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
            weight1_warped = cv2.warpAffine(vessel_weight0, T, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR)
            valid_mask1_warped = cv2.warpAffine(valid_mask1_orig, T, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
            
            # 应用翻转 (图像和掩码同步)
            if flip_h:
                img1_warped = cv2.flip(img1_warped, 1)  # 水平翻转
                mask1_warped = cv2.flip(mask1_warped, 1)
                valid_mask1_warped = cv2.flip(valid_mask1_warped, 1)
            if flip_v:
                img1_warped = cv2.flip(img1_warped, 0)  # 垂直翻转
                mask1_warped = cv2.flip(mask1_warped, 0)
                valid_mask1_warped = cv2.flip(valid_mask1_warped, 0)
            
            # 检查掩码质量：如果变换后掩码面积减少超过30%，则标记（但仍使用）
            original_area = np.sum(vessel_mask_bin > 0.5)
            warped_area = np.sum(mask1_warped > 0.5)
            mask_quality = warped_area / (original_area + 1e-6)
            
            # 将仿射矩阵扩充为 3x3 单应矩阵格式，方便后续计算
            H = np.eye(3, dtype=np.float32)
            H[:2, :] = T
            
            # 重要修复：将翻转(Flip)操作同步到单应矩阵 H 中
            if flip_h:
                # 水平翻转矩阵: x' = (W-1) - x
                H_flip_h = np.eye(3, dtype=np.float32)
                H_flip_h[0, 0] = -1
                H_flip_h[0, 2] = self.img_size - 1
                H = H_flip_h @ H
            if flip_v:
                # 垂直翻转矩阵: y' = (H-1) - y
                H_flip_v = np.eye(3, dtype=np.float32)
                H_flip_v[1, 1] = -1
                H_flip_v[1, 2] = self.img_size - 1
                H = H_flip_v @ H
            
            # image0: 固定图, image1: 经过 T 变换后的待配准图
            data = {
                'image0': torch.from_numpy(img0).float()[None] / 255.0,  # [1, H, W]
                'image1': torch.from_numpy(img1_warped).float()[None] / 255.0,
                'image1_origin': torch.from_numpy(img1_origin).float()[None] / 255.0, # 原始移动图
                'mask0': torch.from_numpy(valid_mask0_orig).float()[None],  # [1, H, W] 有效区域掩码
                'mask1': torch.from_numpy(valid_mask1_warped).float()[None],
                'vessel_mask0': torch.from_numpy(vessel_mask_bin).float()[None],  # [1, H, W] 二值血管掩码
                'vessel_mask1': torch.from_numpy(mask1_warped).float()[None],
                'vessel_weight0': torch.from_numpy(vessel_weight0).float()[None],  # [1, H, W] 高斯软掩码 (固定图)
                'vessel_weight1': torch.from_numpy(weight1_warped).float()[None],  # [1, H, W] 高斯软掩码 (变换后)
                'T_0to1': torch.from_numpy(H), # 真值变换矩阵 (image0 到 image1_warped)
                'flip_h': flip_h,  # 是否水平翻转
                'flip_v': flip_v,  # 是否垂直翻转
                'mask_quality': mask_quality,  # 掩码质量指标
            }
        else:
            # 测试阶段直接返回原图对和掩码
            data = {
                'image0': torch.from_numpy(img0).float()[None] / 255.0,
                'image1': torch.from_numpy(img1).float()[None] / 255.0,
                'image1_origin': torch.from_numpy(img1_origin).float()[None] / 255.0,
                # 测试阶段：将血管掩码同时作为有效区域掩码与血管约束
                'mask0': torch.from_numpy(vessel_mask_bin).float()[None],
                'mask1': torch.from_numpy(vessel_mask_bin).float()[None],
                'vessel_mask0': torch.from_numpy(vessel_mask_bin).float()[None],
                'vessel_mask1': torch.from_numpy(vessel_mask_bin).float()[None],
                'vessel_weight0': torch.from_numpy(vessel_weight0).float()[None],
                'vessel_weight1': torch.from_numpy(vessel_weight0).float()[None],
            }
            
        data.update({
            'dataset_name': 'MultiModal',
            'pair_id': idx,
            'pair_names': (fixed_path.name, moving_path.name)
        })
        
        return data

def build_dataset(config, mode='cffa', split='train'):
    """构建数据集的入口函数"""
    return MultiModalDataset(
        root_dir=config['root_dir'],
        mode=mode,
        split=split,
        img_size=config.get('img_size', 512)
    )
