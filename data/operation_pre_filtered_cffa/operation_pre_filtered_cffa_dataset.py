import os
import glob
import numpy as np
import torch
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ============ 配准与筛选工具函数 (整合自 effective_area_regist_cut.py) ============

def read_points_from_txt(txt_path):
    """从txt文件中读取点位坐标"""
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                coords = line.split()
                if len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y])
    return np.array(points, dtype=np.float32)

def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分，并裁剪使有效区域填满画布"""
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
        return img1, img2, (0, 0)
    
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    
    if len(filtered_img1.shape) == 3:
        filtered_img1[~valid_mask_cropped] = 0
    else:
        filtered_img1[~valid_mask_cropped] = 0
    
    if len(filtered_img2.shape) == 3:
        filtered_img2[~valid_mask_cropped] = 0
    else:
        filtered_img2[~valid_mask_cropped] = 0
    
    return filtered_img1, filtered_img2, (col_min, row_min)

def register_image(cond_img, cond_points, tgt_img, tgt_points):
    """将tgt图配准到cond图的空间"""
    assert len(cond_points) == len(tgt_points), "cond和tgt的点位数量必须一致"
    
    cond_height, cond_width = cond_img.shape[:2]
    H = np.eye(3, dtype=np.float32)
    
    if len(cond_points) >= 4:
        H_est, mask = cv2.findHomography(tgt_points, cond_points, cv2.RANSAC, 5.0)
        
        if H_est is None:
            H_est = cv2.estimateAffinePartial2D(tgt_points, cond_points)[0]
            if H_est is not None:
                H_est = np.vstack([H_est, [0, 0, 1]])
        
        if H_est is not None:
            H = H_est.astype(np.float32)
            registered_img = cv2.warpPerspective(
                tgt_img, 
                H, 
                (cond_width, cond_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            if len(tgt_img.shape) == 3:
                registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
            else:
                registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    else:
        if len(tgt_img.shape) == 3:
            registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
        else:
            registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    
    return registered_img, H

# ============ CF_FA 数据集加载器 ============
SIZE = 512

class CFFADataset(Dataset):
    """
    CF-FA 自动配对数据集 - 支持直接从文件夹读取
    支持配准和有效区域筛选
    返回: cond_original, tgt, cond_path, tgt_path
    """
    def __init__(self, root_dir='/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cffa', split='train', mode='fa2cf'):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        
        self.samples = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        # 1. 搜集所有样本
        all_samples = []
        subdirs = sorted(os.listdir(root_dir))
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            # 寻找配对图像 (01 为 CF, 02 为 FA)
            png_files = glob.glob(os.path.join(subdir_path, "*_01.png"))
            for cf_path in png_files:
                base_name = os.path.basename(cf_path).replace('_01.png', '')
                fa_path = os.path.join(subdir_path, f"{base_name}_02.png")
                cf_pts = os.path.join(subdir_path, f"{base_name}_01.txt")
                fa_pts = os.path.join(subdir_path, f"{base_name}_02.txt")
                
                if os.path.exists(fa_path) and os.path.exists(cf_pts) and os.path.exists(fa_pts):
                    all_samples.append({
                        'fa_path': fa_path,
                        'cf_path': cf_path,
                        'fa_pts': fa_pts,
                        'cf_pts': cf_pts
                    })
        
        # 2. 8:2 随机划分 (固定种子以保证可复现)
        random.Random(42).shuffle(all_samples)
        num_total = len(all_samples)
        num_train = int(num_total * 0.8)
        
        if split == 'train':
            self.samples = all_samples[:num_train]
        else: # val or test
            self.samples = all_samples[num_train:]
        
        print(f"[CFFADataset] {split} set: {len(self.samples)} samples (total {num_total})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fa_path = sample['fa_path']
        cf_path = sample['cf_path']
        fa_pts_path = sample['fa_pts']
        cf_pts_path = sample['cf_pts']
        
        # 1. 加载原始图像
        fa_pil = Image.open(fa_path).convert("RGB")
        cf_pil = Image.open(cf_path).convert("RGB")
        
        fa_np = np.array(fa_pil)
        cf_np = np.array(cf_pil)
        
        # 2. 读取关键点
        try:
            fa_points = read_points_from_txt(fa_pts_path)
            cf_points = read_points_from_txt(cf_pts_path)
        except:
            fa_points = np.array([])
            cf_points = np.array([])
        
        # 3. 确定 fix 和 moving: cffa -> fix=CF, moving=FA
        fix_np = cf_np
        moving_np = fa_np
        fix_points = cf_points
        moving_points = fa_points
        fix_path = cf_path
        moving_path = fa_path
        
        # 4. 计算配准后的moving_gt
        T_0to1 = np.eye(3, dtype=np.float32)
        if len(fix_points) >= 4 and len(moving_points) >= 4:
            moving_gt_np, T_0to1 = register_image(fix_np, fix_points, moving_np, moving_points)
        else:
            moving_gt_np = moving_np.copy()
        
        # 5. 移除裁剪逻辑，直接使用原图进行 Resize
        fix_filtered = fix_np
        moving_gt_filtered = moving_gt_np
        
        # 6. 准备原始moving
        moving_original_pil = Image.fromarray(moving_np).resize((SIZE, SIZE), Image.BICUBIC)
        
        # 7. Resize 到 512x512 并补偿 T_0to1 尺度
        h_orig, w_orig = fix_np.shape[:2]
        h_mov_orig, w_mov_orig = moving_np.shape[:2]
        
        # 尺度补偿：T_scaled = T_fix_scale @ T_orig @ inv(T_mov_scale)
        # 这样矩阵才能在 512x512 空间内自恰
        T_fix_scale = np.array([
            [SIZE / float(w_orig), 0, 0],
            [0, SIZE / float(h_orig), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        T_mov_scale_inv = np.array([
            [float(w_mov_orig) / SIZE, 0, 0],
            [0, float(h_mov_orig) / SIZE, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        T_0to1 = T_fix_scale @ T_0to1 @ T_mov_scale_inv

        fix_pil = Image.fromarray(fix_filtered).resize((SIZE, SIZE), Image.BICUBIC)
        moving_gt_pil = Image.fromarray(moving_gt_filtered).resize((SIZE, SIZE), Image.BICUBIC)
        
        # 8. 转换为 Tensor
        fix_tensor = transforms.ToTensor()(fix_pil)  # [0, 1]
        moving_original_tensor = transforms.ToTensor()(moving_original_pil)  # [0, 1]
        moving_gt_tensor = transforms.ToTensor()(moving_gt_pil)  # [0, 1]
        
        # 9. 归一化到 [-1, 1]
        moving_original_tensor = moving_original_tensor * 2 - 1
        moving_gt_tensor = moving_gt_tensor * 2 - 1
        
        return fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, torch.from_numpy(T_0to1)

    def get_raw_sample(self, idx):
        """返回未配准、未裁剪的原始数据及其关键点"""
        sample = self.samples[idx]
        fa_path, cf_path = sample['fa_path'], sample['cf_path']
        fa_pts_path, cf_pts_path = sample['fa_pts'], sample['cf_pts']

        # 读取原图
        img_fa = cv2.imread(fa_path, cv2.IMREAD_GRAYSCALE)
        img_cf = cv2.imread(cf_path, cv2.IMREAD_GRAYSCALE)

        # 读取原始关键点
        fa_pts = read_points_from_txt(fa_pts_path)
        cf_pts = read_points_from_txt(cf_pts_path)

        if self.mode == 'fa2cf':
            return img_fa, img_cf, fa_pts, cf_pts, fa_path, cf_path
        else: # cf2fa
            return img_cf, img_fa, cf_pts, fa_pts, cf_path, fa_path
