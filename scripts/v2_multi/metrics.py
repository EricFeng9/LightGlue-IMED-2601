import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
import sys
import os

# 【终极方案】创建一个同时输出到控制台和文件的日志函数
_log_file = None
_verbose_mode = False  # 全局标志：是否输出详细日志（默认关闭，仅在验证时开启）

def set_metrics_verbose(verbose):
    """设置是否输出详细的指标日志（用于区分训练/验证阶段）"""
    global _verbose_mode
    _verbose_mode = verbose

def _dual_log(level, message):
    """同时输出到 loguru 和文件"""
    global _log_file, _verbose_mode
    
    # 如果不在 verbose 模式，只输出 WARNING 和 ERROR
    if not _verbose_mode and level == "INFO":
        return
    
    # 输出到 loguru
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    
    # 同时直接写入文件（绕过 loguru）
    if _log_file is None:
        # 尝试从环境变量获取日志文件路径
        log_path = os.environ.get('LOFTR_LOG_FILE', None)
        if log_path:
            try:
                _log_file = open(log_path, 'a', buffering=1)
            except:
                pass
    
    if _log_file:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_file.write(f"{timestamp} | {level: <8} | {message}\n")
        _log_file.flush()


# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    dataset_name = data['dataset_name'][0].lower()
    # 添加对医学眼底多模态数据集的支持（CFFA, CFOCT, OCTFA, CFOCTA）
    multimodal_datasets = ['multimodal', 'realdataset', 'cffa', 'cfoct', 'octfa', 'cfocta']
    if dataset_name in multimodal_datasets:
        return compute_homography_reprojection_errors(data)
    
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def compute_homography_reprojection_errors(data):
    """
    计算基于单应矩阵的重投影误差 (针对 MultiModal 数据集)
    Update:
        data (dict):{"epi_errs": [M]}
    """
    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    H_0to1 = data['T_0to1'] # [N, 3, 3]

    epi_errs = []
    for bs in range(H_0to1.size(0)):
        mask = m_bids == bs
        if mask.sum() == 0:
            continue
        
        # 将 pts0 投影到 image1
        p0 = pts0[mask]
        p0_h = torch.cat([p0, torch.ones_like(p0[:, :1])], dim=-1)
        p0_warped_h = (H_0to1[bs] @ p0_h.t()).t()
        p0_warped = p0_warped_h[:, :2] / (p0_warped_h[:, 2:] + 1e-7)
        
        # 计算 L2 距离作为误差
        err = torch.norm(p0_warped - pts1[mask], dim=-1)
        epi_errs.append(err)
        
    if len(epi_errs) == 0:
        data.update({'epi_errs': torch.tensor([], device=pts0.device)})
    else:
        data.update({'epi_errs': torch.cat(epi_errs, dim=0)})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    dataset_name = data['dataset_name'][0].lower()
    # 添加对医学眼底多模态数据集的支持（CFFA, CFOCT, OCTFA, CFOCTA）
    multimodal_datasets = ['multimodal', 'realdataset', 'cffa', 'cfoct', 'octfa', 'cfocta']
    if dataset_name in multimodal_datasets:
        return compute_homography_errors(data, config)
    
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


def spatial_binning(pts0, pts1, img_size, grid_size=4, top_n=20, conf=None):
    """
    【方案 B 改进】空间均匀化 (Spatial Binning)
    将图像划分为 grid_size x grid_size 的网格，每个网格内最多保留 Top-N 个匹配点。
    """
    h, w = img_size
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    selected_indices = []
    
    # 模拟网格
    grid = [[] for _ in range(grid_size * grid_size)]
    
    for i, pt in enumerate(pts0):
        gx = min(int(pt[0] / cell_w), grid_size - 1)
        gy = min(int(pt[1] / cell_h), grid_size - 1)
        grid[gy * grid_size + gx].append(i)
        
    for cell_indices in grid:
        if len(cell_indices) == 0:
            continue
        
        if conf is not None:
            # 按置信度排序
            cell_indices = sorted(cell_indices, key=lambda idx: conf[idx], reverse=True)
            
        selected_indices.extend(cell_indices[:top_n])
        
    return np.array(selected_indices)


def compute_corner_error(H_est, H_gt, height, width):
    """四角点平均重投影误差（MACE）"""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_h = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)

    try:
        corners_gt_h = (H_gt @ corners_h.T).T
        corners_gt = corners_gt_h[:, :2] / (corners_gt_h[:, 2:] + 1e-6)
        corners_est_h = (H_est @ corners_h.T).T
        corners_est = corners_est_h[:, :2] / (corners_est_h[:, 2:] + 1e-6)
        return float(np.mean(np.linalg.norm(corners_est - corners_gt, axis=-1)))
    except Exception:
        return float('inf')


def _reprojection_stats(H, pts0, pts1):
    """
    与 metrics_cau_principle_0304.md 对齐：
    - MSE: 特征点坐标 MSE = mean((pts1 - pts1_pred)^2)
    - avg_dist: 匹配点平均重投影误差（L2）
    """
    if H is None or len(pts0) == 0:
        return float('inf'), float('inf'), None

    pts0_homo = pts0.reshape(-1, 1, 2).astype(np.float32)
    try:
        pts1_pred = cv2.perspectiveTransform(pts0_homo, H).reshape(-1, 2)
    except Exception:
        return float('inf'), float('inf'), None

    diff = (pts1 - pts1_pred).astype(np.float64)
    mse = float(np.mean(diff ** 2))
    dis = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    avg_dist = float(dis.mean()) if len(dis) > 0 else float('inf')
    return mse, avg_dist, dis


def _reprojection_stats_gt(H, gt_pts0, gt_pts1):
    """
    基于 GT 关键点计算重投影误差（用于评估指标）

    - MSE: GT关键点坐标 MSE = mean((gt_pts1 - pts1_pred)^2)
    - avg_dist: GT关键点平均重投影误差（L2）
    - dis: 每个GT关键点的重投影误差

    参数:
        H: 估计的单应矩阵 (3x3)
        gt_pts0: GT关键点（固定图上的点）(N, 2)
        gt_pts1: GT关键点（移动图上的点，配对的）(N, 2)

    返回:
        mse, avg_dist, dis
    """
    if H is None or len(gt_pts0) == 0:
        return float('inf'), float('inf'), None

    # 将 gt_pts0 投影到移动图空间
    gt_pts0_homo = gt_pts0.reshape(-1, 1, 2).astype(np.float32)
    try:
        gt_pts1_pred = cv2.perspectiveTransform(gt_pts0_homo, H).reshape(-1, 2)
    except Exception:
        return float('inf'), float('inf'), None

    # 计算投影点与GT关键点之间的误差
    diff = (gt_pts1 - gt_pts1_pred).astype(np.float64)
    mse = float(np.mean(diff ** 2))
    dis = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    avg_dist = float(dis.mean()) if len(dis) > 0 else float('inf')
    return mse, avg_dist, dis


def compute_homography_errors(data, config):
    """
    严格对齐 scripts/v2/metrics_cau_principle_0305.md 的指标口径：
    - Failed: num_matches<4 或 inliers_rate<1e-6 或 H≈I (atol=1e-3) 或 H_est None
    - AUC: 使用 gt_mace（基于GT关键点的误差，失败样本写入 1e6；成功样本包括 inaccurate 也写入 gt_mace）
    - Inaccurate: mae>50 或 mee>20（基于 dis = L2(pts1-pts1_pred)，使用GT关键点）
    - MSE/GT_MACE: 仅 Acceptable（成功且非 inaccurate），否则置 inf，最终聚合时过滤

    使用 GT 关键点计算指标（而非角点或模型匹配点）

    Update:
        data (dict): 会写入
            - "H_est": List[np.ndarray] length B
            - "inliers": List[np.ndarray] length B (bool mask for RANSAC points)
            - "t_errs": List[float] length B (用于 AUC 的 error = gt_mace 或 1e6)
            - "mse_list": List[float] length B (Acceptable 才为有限值)
            - "mace_list": List[float] length B (Acceptable 才为有限值)
            - "failed_mask": List[bool] length B
            - "inaccurate_mask": List[bool] length B
    """
    data.update({
        'R_errs': [],
        't_errs': [],
        'inliers': [],
        'H_est': [],
        'mse_list': [],
        'mace_list': [],
        'failed_mask': [],
        'inaccurate_mask': [],
    })

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    H_gt = data['T_0to1'].cpu().numpy()
    mconf = data.get('mconf')
    if mconf is not None:
        mconf = mconf.cpu().numpy()

    # 获取 GT 关键点（如果存在）
    gt_pts0_batch = data.get('gt_pts0', None)  # List of np.ndarray, 每个元素 (N, 2)
    gt_pts1_batch = data.get('gt_pts1', None)  # List of np.ndarray, 每个元素 (N, 2)

    for bs in range(H_gt.shape[0]):
        mask = m_bids == bs
        num_matches = np.sum(mask)

        if num_matches < 4:
            _dual_log("WARNING", f"⚠️ Batch {bs}: 匹配点数不足 ({num_matches} < 4)")
            data['R_errs'].append(0.0)
            data['t_errs'].append(1e6)
            data['inliers'].append(np.array([]).astype(bool))
            data['H_est'].append(np.eye(3))
            data['mse_list'].append(np.inf)
            data['mace_list'].append(np.inf)
            data['failed_mask'].append(True)
            data['inaccurate_mask'].append(False)
            continue

        # 估计单应矩阵 (对应 plan.md 第四阶段: 几何估计)
        # 【方案 B 改进】进行空间均匀化 (Spatial Binning)
        img_size = data['image0'].shape[2:]
        pts0_batch = pts0[mask]
        pts1_batch = pts1[mask]
        mconf_batch = mconf[mask] if mconf is not None else None

        bin_indices = spatial_binning(pts0_batch, pts1_batch, img_size, grid_size=4, top_n=20, conf=mconf_batch)

        _dual_log("INFO", f"🔍 Batch {bs}: 总匹配点={num_matches}, Spatial Binning后={len(bin_indices)}")

        ransac_thr = float(getattr(getattr(config, 'TRAINER', object()), 'RANSAC_PIXEL_THR', 3.0))
        # 【修复】inliers_rate 分母必须使用全部匹配点数
        denom_inliers = len(pts0_batch)
        if len(bin_indices) >= 4:
            pts0_ransac = pts0_batch[bin_indices]
            pts1_ransac = pts1_batch[bin_indices]
            H_est, inliers = cv2.findHomography(pts0_ransac, pts1_ransac, cv2.RANSAC, ransac_thr)
            # 注意：denom_inliers 保持为 len(pts0_batch)，不是 len(pts0_ransac)
        else:
            H_est, inliers = cv2.findHomography(pts0_batch, pts1_batch, cv2.RANSAC, ransac_thr)

        # --- Failed 判定（严格对齐 0304 原则）---
        is_failed = False
        if H_est is None or (isinstance(H_est, np.ndarray) and (np.isnan(H_est).any() or np.isinf(H_est).any())):
            is_failed = True
        else:
            inliers_arr = inliers.ravel() > 0 if inliers is not None else np.array([], dtype=bool)
            num_inliers = int(np.sum(inliers_arr)) if inliers_arr.size > 0 else 0
            inliers_rate = num_inliers / max(1, int(denom_inliers))
            is_identity = np.allclose(H_est, np.eye(3), atol=1e-3)

            _dual_log(
                "INFO",
                f"✅ Batch {bs}: RANSAC 成功, inliers={num_inliers}/{denom_inliers}, "
                f"inliers_rate={inliers_rate:.2e}, H_est是否单位矩阵={is_identity}"
            )

            if inliers_rate < 1e-6:
                _dual_log("WARNING", f"⚠️ Batch {bs}: inliers_rate < 1e-6，视为失败")
                is_failed = True
            if is_identity:
                _dual_log("WARNING", f"⚠️ Batch {bs}: H_est 接近单位矩阵 (atol=1e-3)，视为失败")
                is_failed = True

        if is_failed:
            _dual_log("WARNING", f"⚠️ Batch {bs}: 匹配失败 (Failed)")
            data['R_errs'].append(0.0)
            data['t_errs'].append(1e6)
            data['inliers'].append(np.array([]).astype(bool))
            data['H_est'].append(np.eye(3))
            data['mse_list'].append(np.inf)
            data['mace_list'].append(np.inf)
            data['failed_mask'].append(True)
            data['inaccurate_mask'].append(False)
            continue

        # --- 成功样本：使用 GT 关键点计算误差 ---
        # 获取本样本的 GT 关键点
        if gt_pts0_batch is None or gt_pts1_batch is None:
            raise ValueError(f"Batch {bs}: 缺少GT关键点数据 (gt_pts0或gt_pts1为None)")

        # 如果 gt_pts0_batch 是 list，每个元素对应一个 batch
        if isinstance(gt_pts0_batch, list):
            gt_pts0 = gt_pts0_batch[bs] if bs < len(gt_pts0_batch) else None
            gt_pts1 = gt_pts1_batch[bs] if bs < len(gt_pts1_batch) else None
        else:
            raise ValueError(f"Batch {bs}: GT关键点格式错误，应为list而非tensor")

        # 检查GT关键点是否存在
        if gt_pts0 is None or gt_pts1 is None:
            raise ValueError(f"Batch {bs}: GT关键点为None")

        # 确保是numpy数组
        if isinstance(gt_pts0, torch.Tensor):
            gt_pts0 = gt_pts0.cpu().numpy()
        if isinstance(gt_pts1, torch.Tensor):
            gt_pts1 = gt_pts1.cpu().numpy()

        # 检查GT关键点数量是否足够
        if len(gt_pts0) < 4 or len(gt_pts1) < 4:
            raise ValueError(f"Batch {bs}: GT关键点数量不足 (fix:{len(gt_pts0)}, moving:{len(gt_pts1)})，需要至少4个关键点")

        # 使用 GT 关键点计算重投影误差
        mse, avg_dist, dis = _reprojection_stats_gt(H_est, gt_pts0, gt_pts1)

        if dis is None or not np.isfinite(avg_dist):
            # 极端数值问题，按失败处理
            _dual_log("WARNING", f"⚠️ Batch {bs}: 重投影误差异常，按失败处理")
            data['R_errs'].append(0.0)
            data['t_errs'].append(1e6)
            data['inliers'].append(np.array([]).astype(bool))
            data['H_est'].append(np.eye(3))
            data['mse_list'].append(np.inf)
            data['mace_list'].append(np.inf)
            data['failed_mask'].append(True)
            data['inaccurate_mask'].append(False)
            continue

        mae = float(np.max(dis)) if len(dis) > 0 else float('inf')
        mee = float(np.median(dis)) if len(dis) > 0 else float('inf')
        is_inaccurate = (mae > 50.0) or (mee > 20.0)

        if is_inaccurate:
            _dual_log("WARNING", f"⚠️ Batch {bs}: Inaccurate (mae={mae:.2f}, mee={mee:.2f})")

        # GT-MACE = GT关键点的平均重投影误差
        gt_mace = float(np.mean(dis)) if len(dis) > 0 else float('inf')

        # 处理极端 mace 值（数值问题时用大值替代）
        if not np.isfinite(gt_mace):
            _dual_log("WARNING", f"⚠️ Batch {bs}: gt_mace 数值异常 ({gt_mace})，用 1e6 替代")
            gt_mace = 1e6

        # AUC: 成功样本（含 inaccurate）使用 gt_mace（基于GT关键点的误差）
        data['R_errs'].append(0.0)
        data['t_errs'].append(gt_mace)
        data['inliers'].append(inliers.ravel() > 0 if inliers is not None else np.array([]).astype(bool))
        data['H_est'].append(H_est)

        # MSE/MACE: 仅 Acceptable 样本统计，否则置 inf
        if is_inaccurate:
            data['mse_list'].append(np.inf)
            data['mace_list'].append(np.inf)
        else:
            data['mse_list'].append(mse)
            data['mace_list'].append(gt_mace)

        data['failed_mask'].append(False)
        data['inaccurate_mask'].append(bool(is_inaccurate))


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # 使用传入的 thresholds（例如 [10, 20, 40]），而不是在函数内部重写
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def compute_auc_rop(errors, limit=25):
    """
    计算 ROP 风格的 mAUC (按照 test_on_CrossModality.py 的方式)
    Args:
        errors (list): 误差列表
        limit (int): 阈值上限，默认25像素
    Returns:
        dict: {'mAUC': float}
    """
    errors = np.array(errors)
    gs_error = np.zeros(limit + 1)
    accum_s = 0
    for i in range(1, limit + 1):
        gs_error[i] = np.sum(errors < i) * 100 / len(errors)
        accum_s = accum_s + gs_error[i]
    
    auc_s = accum_s / (limit * 100)
    return {'mAUC': auc_s}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc（对 MultiModal 场景，误差单位为像素，这里统一使用 [5, 10, 20] 像素阈值）
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    return {**aucs, **precs}
