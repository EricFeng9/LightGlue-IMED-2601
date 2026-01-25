"""
1. SR_ME (Success Rate based on Mean Error) - 平均误差成功率
   - 公式: SR_ME = 1 if (1/N * Σ ||T(p_i) - q_i||) <= ε_ME else 0
   - 说明: 计算所有控制点经过预测变换后的平均重投影误差。如果平均误差小于阈值 ε_ME (5px)，则认为配准成功。
   - 评价: 越大越好 (1.0 代表成功，0.0 代表失败)。

2. SR_MAE (Success Rate based on Maximum Absolute Error) - 最大误差成功率
   - 公式: SR_MAE = 1 if max(||T(p_i) - q_i||) <= ε_MAE else 0
   - 说明: 计算所有控制点中最大的重投影误差。如果最大误差小于阈值 ε_MAE (10px)，则认为配准成功。
   - 评价: 越大越好 (1.0 代表成功，0.0 代表失败)。

3. Rep (Repeatability) - 特征点重复率
   - 公式: Rep = |{p ∈ K0 | dist(H_gt * p, K1) <= ε_Rep}| / |K0|
   - 说明: 衡量在源图中检测到的特征点，经过真值变换后，在目标图中 ε_Rep (5px) 距离范围内也能找到对应检测点的比例。
   - 评价: 越大越好。

4. MIR (Matching Inliers Ratio) - 匹配内点率
   - 公式: MIR = N_inliers / N_matches
   - 说明: 经过 RANSAC 算法剔除误匹配后的内点数量与初始匹配对总数的比值。衡量匹配算法的鲁棒性。
   - 评价: 越大越好。

5. Mean Error / Max Error - 误差数值
   - 公式: 见 SR_ME 和 SR_MAE 的内部项
   - 说明: 控制点的实际重投影误差像素值。
   - 评价: 越小越好。
"""
import numpy as np
import cv2
from scipy.spatial import KDTree

def spatial_binning(pts0, pts1, img_size=(512, 512), grid_size=4, top_n=20):
    """
    【方案 B 改进】空间均匀化 (Spatial Binning)
    """
    h, w = img_size
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    selected_indices = []
    grid = [[] for _ in range(grid_size * grid_size)]
    
    for i, pt in enumerate(pts0):
        gx = min(int(pt[0] / cell_w), grid_size - 1)
        gy = min(int(pt[1] / cell_h), grid_size - 1)
        grid[gy * grid_size + gx].append(i)
        
    for cell_indices in grid:
        if len(cell_indices) == 0:
            continue
        selected_indices.extend(cell_indices[:top_n])
        
    return np.array(selected_indices)

def calculate_metrics(img_origin, img_result, mkpts0, mkpts1, kpts0=None, kpts1=None, ctrl_pts0=None, ctrl_pts1=None, H_gt=None):
    """
    计算四个指标：SR_ME, SR_MAE, Rep, MIR。
    
    参数:
    - img_origin: 源图像 (Image A) [H, W]
    - img_result: 目标图像 (Image B) [H, W]
    - mkpts0: 源图像中匹配成功的特征点 [N, 2] (由配准模型输出的匹配点对)
    - mkpts1: 目标图像中匹配成功的特征点 [N, 2]
    - kpts0: 源图像中所有检测到的特征点 [M0, 2] (用于计算 Rep, 若不提供则默认使用 mkpts0)
    - kpts1: 目标图像中所有检测到的特征点 [M1, 2]
    - ctrl_pts0: 源图像中手动标注的控制点 [6, 2] (用于评估配准精度的真值点)
    - ctrl_pts1: 目标图像中手动标注的控制点 [6, 2]
    - H_gt: 地面真值单应矩阵 [3, 3] (用于计算 Rep)
    
    返回:
    - metrics: 包含四个指标的字典
        - SR_ME: 1.0 (成功) 或 0.0 (失败), 基于平均误差 <= 5px
        - SR_MAE: 1.0 (成功) 或 0.0 (失败), 基于最大误差 <= 10px
        - Rep: 特征点重复率 [0, 1]
        - MIR: 匹配内点率 [0, 1]
        - mean_error: 控制点的平均重投影误差
        - max_error: 控制点的最大重投影误差
    """
    
    # 超参数定义 (依据论文 IR-OCT-OCTA 实验参数)
    EPS_SRME = 5.0    # SR_ME 阈值: 5 像素
    EPS_SRMAE = 10.0  # SR_MAE 阈值: 10 像素
    EPS_REP = 5.0     # Rep 阈值: 5 像素
    EPS_MIR = 5.0     # MIR 阈值 (RANSAC 容差): 5 像素
    
    results = {
        'SR_ME': 0.0,
        'SR_MAE': 0.0,
        'Rep': 0.0,
        'MIR': 0.0,
        'mean_error': float('inf'),
        'max_error': float('inf')
    }
    
    # 转换为 numpy 数组确保兼容性
    mkpts0 = np.array(mkpts0)
    mkpts1 = np.array(mkpts1)
    
    # 1. 估计预测的单应矩阵 H_pred (用于计算 SR 指标)
    H_pred = None
    inliers = None
    if len(mkpts0) >= 4:
        # 【方案 B 改进】空间均匀化 (Spatial Binning)
        # 注意：此处假定 img_origin/img_result 是 512x512，如果不是，建议传入尺寸
        img_size = img_origin.shape[:2] if img_origin is not None else (512, 512)
        bin_indices = spatial_binning(mkpts0, mkpts1, img_size=img_size)
        
        if len(bin_indices) >= 4:
            H_pred, inliers = cv2.findHomography(mkpts0[bin_indices], mkpts1[bin_indices], cv2.RANSAC, EPS_MIR)
        else:
            H_pred, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, EPS_MIR)
    
    # 2. 计算 SR_ME 和 SR_MAE
    if H_pred is not None and ctrl_pts0 is not None and ctrl_pts1 is not None and len(ctrl_pts0) > 0:
        ctrl_pts0 = np.array(ctrl_pts0).reshape(-1, 2)
        ctrl_pts1 = np.array(ctrl_pts1).reshape(-1, 2)
        
        # 将 ctrl_pts0 投影到 Image B 空间
        pts0_h = np.concatenate([ctrl_pts0, np.ones((len(ctrl_pts0), 1))], axis=1)
        pts0_warped_h = (H_pred @ pts0_h.T).T
        pts0_warped = pts0_warped_h[:, :2] / (pts0_warped_h[:, 2:] + 1e-7)
        
        # 计算欧几里得误差 D_ij
        errors = np.linalg.norm(pts0_warped - ctrl_pts1, axis=1)
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        
        results['mean_error'] = mean_err
        results['max_error'] = max_err
        results['SR_ME'] = 1.0 if mean_err <= EPS_SRME else 0.0
        results['SR_MAE'] = 1.0 if max_err <= EPS_SRMAE else 0.0

    # 3. 计算 Rep (Repeatability)
    # 论文逻辑: 在 epsilon 距离内能找到对应点的特征点数 / 检测到的总特征点数
    h_for_rep = H_gt if H_gt is not None else H_pred
    kp0 = np.array(kpts0) if kpts0 is not None else mkpts0
    kp1 = np.array(kpts1) if kpts1 is not None else mkpts1
    
    if h_for_rep is not None and len(kp0) > 0 and len(kp1) > 0:
        kp0_h = np.concatenate([kp0, np.ones((len(kp0), 1))], axis=1)
        kp0_warped_h = (h_for_rep @ kp0_h.T).T
        kp0_warped = kp0_warped_h[:, :2] / (kp0_warped_h[:, 2:] + 1e-7)
        
        tree = KDTree(kp1)
        dist, _ = tree.query(kp0_warped)
        num_found = np.sum(dist <= EPS_REP)
        results['Rep'] = num_found / len(kp0)

    # 4. 计算 MIR (Matching Inliers Ratio)
    # MIR = RANSAC 内点数 / 初始匹配对数 (MNN)
    if len(mkpts0) > 0:
        if inliers is not None:
            num_inliers = np.sum(inliers)
            results['MIR'] = num_inliers / len(mkpts0)
        else:
            results['MIR'] = 0.0
            
    return results

if __name__ == "__main__":
    # --- 模拟测试数据 ---
    # 假设图像尺寸 512x512
    img_size = 512
    
    # 模拟匹配点 (LoFTR 输出)
    # 假设真实位移是 (10, 10)
    mkpts0 = np.random.rand(100, 2) * img_size
    mkpts1 = mkpts0 + 10.0 + np.random.normal(0, 1.0, (100, 2)) # 添加噪声
    
    # 模拟控制点 (手动标注)
    ctrl_pts0 = np.array([
        [100, 100], [400, 100], [100, 400], [400, 400], [256, 256], [200, 300]
    ])
    ctrl_pts1 = ctrl_pts0 + 10.0 # 完美的位移
    
    # 模拟检测到的所有特征点 (用于 Rep)
    kpts0 = np.random.rand(500, 2) * img_size
    kpts1 = kpts0 + 10.0 + np.random.normal(0, 0.5, (500, 2))
    
    # 地面真值单应矩阵
    H_gt = np.eye(3)
    H_gt[0, 2] = 10.0
    H_gt[1, 2] = 10.0
    
    # --- 计算指标 ---
    metrics = calculate_metrics(
        img_origin=None, img_result=None,
        mkpts0=mkpts0, mkpts1=mkpts1,
        kpts0=kpts0, kpts1=kpts1,
        ctrl_pts0=ctrl_pts0, ctrl_pts1=ctrl_pts1,
        H_gt=H_gt
    )
    
    print("计算指标结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
