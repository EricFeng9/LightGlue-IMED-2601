import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class MultiModalDataset(Dataset):
    """
    多模态眼底图像数据集加载器 - 按 plan_260227.md 修改
    
    修改要点:
    1. 对 fa 和 seg 同时应用形变，得到 fa_deformed 和 seg_deformed
    2. 返回 cf, fa_deformed, seg_original, seg_deformed
    3. 域随机化将在训练循环中进行，数据集只负责加载和形变
    
    数据集目录结构:
        root_dir/
            9_A_seg_fsgnet/
                cf_gen.png  (Fix image)
                fa_gen.png  (Moving image)
                seg.png     (Vessel segmentation)
    """
    def __init__(self, root_dir, mode='cffa', split='train', img_size=512):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.split = split
        self.img_size = img_size
        
        # 1. 扫描文件夹
        def is_valid_dir(d):
            if not d.is_dir(): return False
            # 260227_2_v29_2_1: 子文件夹格式为 "数字_字母_seg_fsgnet"
            # 如 "9_A_seg_fsgnet", "100_A_seg_fsgnet", "351_A_seg_fsgnet"
            return d.name.endswith('_seg_fsgnet')
        
        # 按数字 ID 排序（提取文件夹名称开头的数字部分）
        def extract_id(folder_name):
            # 从 "100_A_seg_fsgnet" 提取 100
            try:
                return int(folder_name.split('_')[0])
            except:
                return 0
        
        all_dirs = sorted(
            [d for d in self.root_dir.iterdir() if is_valid_dir(d)],
            key=lambda x: extract_id(x.name)
        )
        
        # 2. 划分数据集 (8:2)
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(all_dirs))
        all_dirs = [all_dirs[i] for i in indices]
        
        num_train = int(len(all_dirs) * 0.8)
        
        if split == 'train':
            self.dirs = all_dirs[:num_train]
        else:
            self.dirs = all_dirs[num_train:]
            
        # 3. 验证集不再截断，完整遍历
        print(f"[MultiModalDataset v29_2_1_VesselSup] Mode={mode}, Split={split}, Size={len(self.dirs)}")

    def _get_image_paths(self, folder):
        folder_id = folder.name
        # 定义文件名模板 (适配 260227_2_v29_2_1)
        if self.mode == 'cffa':
            # image0 (Fix): cf_gen.png, image1 (Moving 原图): fa_gen.png
            # 血管分割图: seg.png
            f_names = ["cf_gen.png"]
            m_names = ["fa_gen.png"]
            seg_names = ["seg.png"]
        # 其他模式当前不支持此数据集
        else:
            raise ValueError(f"Unknown mode for 260227_2_v29_2_1 dataset: {self.mode}")
            
        def find(names):
            for n in names:
                if (folder / n).exists(): return folder / n
            return None
            
        # 返回固定图、移动图、血管分割图的路径
        return find(f_names), find(m_names), find(seg_names)

    def _get_random_affine(self, rng=None):
        if rng is None: rng = np.random
        
        angle = rng.uniform(-30, 30)
        scale = rng.uniform(0.9, 1.1)
        tx = rng.uniform(-0.05, 0.05) * self.img_size
        ty = rng.uniform(-0.05, 0.05) * self.img_size
        
        center = (self.img_size // 2, self.img_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # 扩展为 3x3
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = M
        return H

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        folder = self.dirs[idx]
        fixed_path, moving_path, mask_path = self._get_image_paths(folder)
        
        if not fixed_path or not moving_path or not mask_path:
            # Fallback if file missing
            return self.__getitem__((idx + 1) % len(self))

        # 1. 读取图像 (灰度)
        cf_img = cv2.imread(str(fixed_path), cv2.IMREAD_GRAYSCALE)
        fa_img = cv2.imread(str(moving_path), cv2.IMREAD_GRAYSCALE)
        seg_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 2. Resize 到统一尺寸 512x512
        if cf_img.shape[:2] != (self.img_size, self.img_size):
            cf_img = cv2.resize(cf_img, (self.img_size, self.img_size))
        if fa_img.shape[:2] != (self.img_size, self.img_size):
            fa_img = cv2.resize(fa_img, (self.img_size, self.img_size))
        if seg_img.shape[:2] != (self.img_size, self.img_size):
            seg_img = cv2.resize(seg_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # 3. 生成仿射变换 (T)
        if self.split == 'train':
            H_forward = self._get_random_affine() # 随机
        else:
            # 验证集使用固定变换，便于对比
            rng = np.random.RandomState(idx)
            H_forward = self._get_random_affine(rng)
            
        # 4. 应用变换 (Warp) - 对 fa 和 seg 同时变换
        fa_deformed = cv2.warpAffine(fa_img, H_forward[:2], (self.img_size, self.img_size), flags=cv2.INTER_LINEAR)
        seg_deformed = cv2.warpAffine(seg_img, H_forward[:2], (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
        
        # 5. 归一化 [0, 1] 并转 Tensor
        cf_norm = cf_img.astype(np.float32) / 255.0
        fa_deformed_norm = fa_deformed.astype(np.float32) / 255.0
        seg_norm = (seg_img.astype(np.float32) / 255.0 > 0.5).astype(np.float32)  # 二值化
        seg_deformed_norm = (seg_deformed.astype(np.float32) / 255.0 > 0.5).astype(np.float32)  # 二值化

        # 6. 组装数据
        # T_0to1: 从 image0 (cf) 到 image1 (fa_deformed) 的变换
        T_0to1 = H_forward.astype(np.float32)

        pair_names = (fixed_path.name, moving_path.name)
        
        data = {
            # 输入图像 (将在训练循环中进行域随机化)
            'image0': torch.from_numpy(cf_norm).float()[None],              # [1, H, W] CF (Fix)
            'image1': torch.from_numpy(fa_deformed_norm).float()[None],     # [1, H, W] FA deformed (Moving)
            
            # 监督信号: seg 原始和变形后的版本
            'seg_original': torch.from_numpy(seg_norm).float()[None],       # [1, H, W] Seg (与 CF 对齐)
            'seg_deformed': torch.from_numpy(seg_deformed_norm).float()[None],  # [1, H, W] Seg deformed (与 FA deformed 对齐)
            'vessel_mask0': torch.from_numpy(seg_norm).float()[None],              # [1, H, W] Seg (与 CF 对齐)
            'vessel_mask1': torch.from_numpy(seg_deformed_norm).float()[None],     # [1, H, W] Seg deformed (与 FA deformed 对齐)
            
            # 变换矩阵 (从 image0 到 image1)
            'T_0to1': torch.from_numpy(T_0to1).float(),                     # [3, 3]
            
            # 元数据 (修改为 'multimodal' 以兼容 supervision.py)
            'dataset_name': 'multimodal',
            'pair_names': pair_names,
        }
        
        return data
