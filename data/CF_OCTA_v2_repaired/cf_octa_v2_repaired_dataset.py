import os
import cv2
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SIZE = 512

# ============ 工具函数 (由 Scripts_v2/v14/registration_cf_octa.py 迁移) ============
def load_affine_matrix(txt_path):
    matrix = []
    if not os.path.exists(txt_path): return None
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                matrix.append([float(x) for x in line.split()])
    return np.array(matrix[:2], dtype=np.float32)

def apply_affine_registration(img_np, affine_matrix, output_size=(512, 512)):
    """
    应用仿射变换配准单张图像。
    改进：直接在原图尺寸上进行变换，避免中间降采样到 256x256 导致的画质损失。
    """
    # 原始矩阵是基于 256x256 尺寸定义的，我们需要将其映射到原图/目标图尺寸
    h, w = img_np.shape[:2]
    
    # 计算从 256x256 到当前图像尺寸的缩放比例
    scale_x = w / 256.0
    scale_y = h / 256.0
    
    # 调整仿射矩阵以适应当前图像尺寸
    M = affine_matrix.copy()
    M[0, 2] *= scale_x
    M[1, 2] *= scale_y
    
    # 转换为 3x3 矩阵返回
    T_0to1 = np.eye(3, dtype=np.float32)
    T_0to1[:2, :] = M
    
    # 执行仿射变换
    # 注意：输出尺寸先设为原图尺寸 (w, h)，保持最高精度
    registered = cv2.warpAffine(
        img_np, 
        M, 
        (w, h), 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # 最后统一 resize 到目标尺寸 (512x512)
    if (w, h) != output_size:
        registered = cv2.resize(registered, output_size, interpolation=cv2.INTER_LINEAR)
        
    return registered, T_0to1

def filter_valid_area(img1, img2):
    """
    筛选有效区域：只保留两张图片都不为纯黑像素的部分。
    针对 CF-OCTA 优化：由于 OCTA 是稀疏图像（血管以外全是黑），
    直接使用像素位与会导致 CF 图被“血管化”。
    改进：使用凸包（Convex Hull）提取两张图的有效视场（FOV），然后取交集。
    """
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    
    def get_fov_mask(img):
        # 转为灰度图并提取非黑区域
        if len(img.shape) == 3:
            gray = np.max(img, axis=2)
        else:
            gray = img
        
        # 使用阈值提取掩码 (阈值设为 5 以过滤微弱噪声)
        mask = (gray > 5).astype(np.uint8) * 255
        
        # 寻找轮廓并提取凸包
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(gray, dtype=bool)
        
        # 获取所有轮廓点的凸包
        all_pts = np.concatenate(contours)
        hull = cv2.convexHull(all_pts)
        
        # 绘制填充后的凸包作为 FOV 掩码
        fov_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(fov_mask, [hull], -1, 255, -1)
        return fov_mask > 0

    mask1 = get_fov_mask(img1)
    mask2 = get_fov_mask(img2)
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
    
    if len(filtered_img1.shape) == 3:
        filtered_img1[~valid_mask_cropped] = 0
    else:
        filtered_img1[~valid_mask_cropped] = 0
    
    if len(filtered_img2.shape) == 3:
        filtered_img2[~valid_mask_cropped] = 0
    else:
        filtered_img2[~valid_mask_cropped] = 0
    
    return filtered_img1, filtered_img2

# ============ CFOCTA 数据集类 ============
class CFOCTADataset(Dataset):
    """
    CFOCTA 数据集加载器 - 支持 CF-OCTA 图像对的配准和加载
    """
    def __init__(self, root_dir='/data/student/Fengjunming/LoFTR/data/CF_OCTA_v2_repaired', split='train', mode='cf2octa'):
        self.root_dir = root_dir
        self.mode = mode
        self.split = split
        
        # 1. 直接根据 split 确定搜索目录
        # 如果 split 是 'test' 或 'val'，从 'ts' 目录读取；如果是 'train'，从 'train' 目录读取
        sdir_name = 'ts' if split in ['test', 'val'] else 'train'
        
        self.samples = []
        gt_cf2octa_dir = os.path.join(root_dir, 'GT_CF_to_OCTA')
        gt_octa2cf_dir = os.path.join(root_dir, 'GT_OCTA_to_CF')
        
        cf_dir = os.path.join(root_dir, f'CF_{sdir_name}')
        octa_dir = os.path.join(root_dir, f'OCTA_{sdir_name}')
        
        if os.path.exists(cf_dir):
            cf_files = sorted([f for f in os.listdir(cf_dir) if f.endswith('.png')])
            for f in cf_files:
                idx = f[:3]
                cf_path = os.path.join(cf_dir, f)
                octa_path = os.path.join(octa_dir, f"{idx}OCTA.png")
                    
                affine_cf2octa = os.path.join(gt_cf2octa_dir, f"{idx}_CF_to_OCTA_affine.txt")
                affine_octa2cf = os.path.join(gt_octa2cf_dir, f"{idx}_OCTA_to_CF_affine.txt")
                
                if os.path.exists(cf_path) and os.path.exists(octa_path):
                    self.samples.append({
                        'cf_path': cf_path,
                        'octa_path': octa_path,
                        'affine_cf2octa': affine_cf2octa,
                        'affine_octa2cf': affine_octa2cf,
                        'id': idx
                    })
        
        print(f"[CFOCTADataset] {split} set ({mode}) from {sdir_name} folder: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. 加载原始图像
        cf_pil = Image.open(s['cf_path']).convert("RGB")
        octa_pil = Image.open(s['octa_path']).convert("RGB")
        
        # CF转为灰度图
        cf_pil = cf_pil.convert("L").convert("RGB")
        
        cf_np = np.array(cf_pil)
        octa_np = np.array(octa_pil)
        
        # 2. 确定 fix 和 moving: cfocta -> fix=CF, moving=OCTA
        fix_np = cf_np
        moving_np = octa_np
        affine_path = s['affine_octa2cf']  # 配准 OCTA 到 CF 空间
        fix_path = s['cf_path']
        moving_path = s['octa_path']
        
        # 3. 计算配准后的moving_gt
        T_0to1 = np.eye(3, dtype=np.float32)
        if affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            if affine_matrix is not None:
                moving_gt_np, T_0to1 = apply_affine_registration(moving_np, affine_matrix, output_size=(SIZE, SIZE))
            else:
                moving_gt_np = moving_np.copy()
        else:
            moving_gt_np = moving_np.copy()
        
        # 4. 移除裁剪逻辑，直接 Resize 并补偿尺度
        fix_filtered = fix_np
        moving_gt_filtered = moving_gt_np
        
        # 准备原始moving
        moving_original_pil = Image.fromarray(moving_np).resize((SIZE, SIZE), Image.BICUBIC)
        
        # 计算尺度补偿 (从原始尺寸到 512)
        h_orig, w_orig = fix_filtered.shape[:2]
        sx = SIZE / float(w_orig)
        sy = SIZE / float(h_orig)
        T_scale = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        T_0to1 = T_scale @ T_0to1

        fix_pil = Image.fromarray(fix_filtered).resize((SIZE, SIZE), Image.BICUBIC)
        moving_gt_pil = Image.fromarray(moving_gt_filtered).resize((SIZE, SIZE), Image.BICUBIC)
        
        # 5. Tensor 转换
        fix_tensor = transforms.ToTensor()(fix_pil)
        moving_original_tensor = transforms.ToTensor()(moving_original_pil)
        moving_gt_tensor = transforms.ToTensor()(moving_gt_pil)
        
        # 6. 归一化
        moving_original_tensor = moving_original_tensor * 2 - 1
        moving_gt_tensor = moving_gt_tensor * 2 - 1
        
        return fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, torch.from_numpy(T_0to1)

    def get_raw_sample(self, idx):
        """返回未配准的原始数据及其关键点"""
        sample = self.samples[idx]
        cf_path = sample['cf_path']
        octa_path = sample['octa_path']
        
        # 定位关键点文件 (x0 y0 x1 y1 格式: CF_x CF_y OCTA_x OCTA_y)
        sdir = 'Ts' if 'ts' in cf_path.lower() else 'train'
        pts_file = os.path.join(self.root_dir, f"Ground_Truth_{sdir}", f"{sample['id']}Fundus_OCTA_points.txt")
        
        img_cf = cv2.imread(cf_path, cv2.IMREAD_GRAYSCALE)
        img_octa = cv2.imread(octa_path, cv2.IMREAD_GRAYSCALE)

        h_cf, w_cf = img_cf.shape[:2]
        h_octa, w_octa = img_octa.shape[:2]

        pts_cf, pts_octa = [], []
        if os.path.exists(pts_file):
            with open(pts_file, 'r') as f:
                for line in f:
                    coords = [float(x) for x in line.strip().split()]
                    if len(coords) == 4:
                        # 核心修正：点位基于 256x256 空间，需映射回图像原始尺寸
                        pts_cf.append([coords[0] * (w_cf / 256.0), coords[1] * (h_cf / 256.0)])
                        pts_octa.append([coords[2] * (w_octa / 256.0), coords[3] * (h_octa / 256.0)])
        
        pts_cf = np.array(pts_cf, dtype=np.float32).reshape(-1, 2)
        pts_octa = np.array(pts_octa, dtype=np.float32).reshape(-1, 2)

        if self.mode == 'cf2octa':
            return img_cf, img_octa, pts_cf, pts_octa, cf_path, octa_path
        else: # octa2cf
            return img_octa, img_cf, pts_octa, pts_cf, octa_path, cf_path
