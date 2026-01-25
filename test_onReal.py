
import sys
import os
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import torch.utils.data as data
import matplotlib.pyplot as plt
from types import SimpleNamespace

# User project imports
from pl_lightglue import PL_LightGlue
from lightglue import viz2d
from measurement import calculate_metrics

# Datasets
from data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# --- Configuration ---
DATA_ROOTS = {
    'cffa': "/data/student/Fengjunming/LightGlue/data/operation_pre_filtered_cffa",
    'cfoct': "/data/student/Fengjunming/LightGlue/data/operation_pre_filtered_cfoct",
    'octfa': "/data/student/Fengjunming/LightGlue/data/operation_pre_filtered_octfa",
    'cfocta': "/data/student/Fengjunming/LightGlue/data/CF_OCTA_v2_repaired"
}

def get_default_config():
    """Configuration for LightGlue model matching the training script."""
    conf = SimpleNamespace()
    conf.TRAINER = SimpleNamespace()
    conf.TRAINER.CANONICAL_BS = 4
    conf.TRAINER.CANONICAL_LR = 1e-4
    conf.TRAINER.EXP_NAME = "lightglue_multimodal"
    conf.TRAINER.TRUE_LR = 1e-4
    conf.MATCHING = {
        'input_dim': 256,
        'descriptor_dim': 256
    }
    return conf

# --- Helper Functions ---

def compute_corner_error(H_est, H_gt, height, width):
    """Calculate MACE (Mean Average Corner Error)."""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    
    # GT warped corners
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    
    # Estimated warped corners
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace

def create_chessboard(img1, img2, grid_size=4):
    """Create checkerboard composite image."""
    H, W = img1.shape
    assert img2.shape == (H, W), "Two images must have the same size"
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

# --- Dataset Wrapper ---

class LightGlueTestDatasetWrapper(data.Dataset):
    """
    Dataset wrapper for testing LightGlue.
    Resizes images to img_size (typically 512x512) and normalizes them.
    Computes keypoint scaling factors.
    """
    def __init__(self, dataset, img_size=512):
        self.dataset = dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Raw data: img0 (fix), img1 (moving), pts0, pts1
        # img0_raw, img1_raw are numpy arrays
        img0_raw, img1_raw, pts0, pts1, path0, path1 = self.dataset.get_raw_sample(idx)
        
        # Ensure float32 and normalize [0, 1]
        if img0_raw.dtype == np.uint8:
            img0_raw = img0_raw.astype(np.float32) / 255.0
        if img1_raw.dtype == np.uint8:
            img1_raw = img1_raw.astype(np.float32) / 255.0
            
        h0, w0 = img0_raw.shape
        h1, w1 = img1_raw.shape
        
        # Resize images
        img0 = cv2.resize(img0_raw, (self.img_size, self.img_size))
        img1 = cv2.resize(img1_raw, (self.img_size, self.img_size))
        
        # Adjust Control Points
        pts0_res = pts0.copy()
        pts1_res = pts1.copy()
        if len(pts0_res) > 0:
            pts0_res[:, 0] *= (self.img_size / w0)
            pts0_res[:, 1] *= (self.img_size / h0)
        if len(pts1_res) > 0:
            pts1_res[:, 0] *= (self.img_size / w1)
            pts1_res[:, 1] *= (self.img_size / h1)
            
        return {
            'image0': torch.from_numpy(img0).float()[None], # [1, H, W]
            'image1': torch.from_numpy(img1).float()[None], # [1, H, W]
            'ctrl_pts0': pts0_res,
            'ctrl_pts1': pts1_res,
            'pair_names': (os.path.basename(path0), os.path.basename(path1)),
            'dataset_name': 'MultiModal',
            # Placeholder for T_0to1, will be calculated from control points in main loop
            'T_0to1': torch.eye(3) 
        }

def main():
    parser = argparse.ArgumentParser(description="LightGlue Multi-Modal Verification Script")
    parser.add_argument('--mode', type=str, required=True, choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    parser.add_argument('--name', type=str, required=True, help='Experiment name (folder in results/)')
    parser.add_argument('--save_dir', type=str, default='test_run')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    # Paths
    ckpt_path = Path(f"results/{args.mode}/{args.name}/best_checkpoint/model.ckpt")
    if not ckpt_path.exists():
        # Try latest checkpoint if best not found
        ckpt_path = Path(f"results/{args.mode}/{args.name}/latest_checkpoint/model.ckpt")
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found at: {ckpt_path}")
            return

    save_root = Path(f"test_results/{args.mode}/{args.save_dir}")
    save_root.mkdir(parents=True, exist_ok=True)
    
    logger.add(save_root / "log.txt", rotation="10 MB")
    logger.info(f"Testing Mode: {args.mode}, Model: {ckpt_path}")

    # Load Model
    config = get_default_config()
    model = PL_LightGlue.load_from_checkpoint(str(ckpt_path), config=config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare Data
    root = DATA_ROOTS[args.mode]
    if args.mode == 'cffa':
        base_ds = CFFADataset(root_dir=root, split='test', mode='cf2fa')
    elif args.mode == 'cfoct':
        base_ds = CFOCTDataset(root_dir=root, split='test', mode='cf2oct')
    elif args.mode == 'octfa':
        base_ds = OCTFADataset(root_dir=root, split='test', mode='oct2fa')
    elif args.mode == 'cfocta':
        base_ds = CFOCTADataset(root_dir=root, split='test', mode='cf2octa')
    
    test_ds = LightGlueTestDatasetWrapper(base_ds, img_size=512)
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    all_metrics = []
    
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        pair_name = f"{Path(batch['pair_names'][0][0]).stem}_vs_{Path(batch['pair_names'][1][0]).stem}"
        sample_dir = save_root / pair_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 1. Forward Pass
        with torch.no_grad():
            outputs = model(batch_gpu)
            
        # 2. Extract Matches & Estimate H
        matches0 = outputs['matches0'][0] # [M]
        kpts0 = batch_gpu['keypoints0'][0] # [M, 2]
        kpts1 = batch_gpu['keypoints1'][0] # [N, 2]
        
        valid = matches0 > -1
        m_indices_0 = torch.where(valid)[0]
        m_indices_1 = matches0[valid]
        
        mkpts0 = kpts0[m_indices_0].cpu().numpy()
        mkpts1 = kpts1[m_indices_1].cpu().numpy()
        
        H_est = np.eye(3)
        if len(mkpts0) >= 4:
            H_found, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.0)
            if H_found is not None:
                H_est = H_found
        
        # 3. Ground Truth H from Control Points
        ctrl_pts0 = batch['ctrl_pts0'][0].numpy()
        ctrl_pts1 = batch['ctrl_pts1'][0].numpy()
        H_gt = np.eye(3)
        if len(ctrl_pts0) >= 4:
            H_gt_found, _ = cv2.findHomography(ctrl_pts0, ctrl_pts1, cv2.RANSAC, 5.0)
            if H_gt_found is not None:
                H_gt = H_gt_found

        # 4. Visualization Images
        img0_np = (batch['image0'][0, 0].numpy() * 255).astype(np.uint8)
        img1_np = (batch['image1'][0, 0].numpy() * 255).astype(np.uint8)
        h, w = img0_np.shape
        
        # Warped results
        try:
            H_est_inv = np.linalg.inv(H_est)
            img1_result = cv2.warpPerspective(img1_np, H_est_inv, (w, h))
        except:
            img1_result = img1_np.copy()
            
        try:
            H_gt_inv = np.linalg.inv(H_gt)
            img1_origin = cv2.warpPerspective(img1_np, H_gt_inv, (w, h))
        except:
            img1_origin = img1_np.copy()
            
        # Save Basic Images
        cv2.imwrite(str(sample_dir / "fix.png"), img0_np)
        cv2.imwrite(str(sample_dir / "moving.png"), img1_np)
        cv2.imwrite(str(sample_dir / "moving_result.png"), img1_result)
        cv2.imwrite(str(sample_dir / "moving_origin.png"), img1_origin)
        
        chessboard = create_chessboard(img1_result, img0_np)
        cv2.imwrite(str(sample_dir / "chessboard.png"), chessboard)
        
        # Save Visualizations (LightGlue style)
        # Matches
        try:
            fig = plt.figure(figsize=(10, 4))
            viz2d.plot_images([img0_np, img1_np], titles=['Fix', 'Moving'])
            viz2d.plot_matches(mkpts0, mkpts1, color='lime', lw=0.5)
            plt.savefig(str(sample_dir / "matches.png"), bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to plot matches: {e}")
            
        # Keypoints (optional, but good for debug)
        try:
            fig_kpts = plt.figure(figsize=(10, 4))
            viz2d.plot_images([img0_np, img1_np], titles=['Fix Keypoints', 'Moving Keypoints'])
            kpts0_np = kpts0.cpu().numpy()
            kpts1_np = kpts1.cpu().numpy()
            viz2d.plot_keypoints([kpts0_np, kpts1_np], colors='red', ps=2)
            plt.savefig(str(sample_dir / "keypoints.png"), bbox_inches='tight')
            plt.close(fig_kpts)
        except Exception as e:
            pass

        # 5. Metrics
        # Basic calculate_metrics used in LoFTR script
        metrics = calculate_metrics(
            img_origin=img0_np, img_result=img1_result,
            mkpts0=mkpts0, mkpts1=mkpts1,
            kpts0=kpts0.cpu().numpy(), kpts1=kpts1.cpu().numpy(),
            ctrl_pts0=ctrl_pts0, ctrl_pts1=ctrl_pts1,
            H_gt=H_gt
        )
        
        # Add MACE (from train_lightglue logic)
        mace = compute_corner_error(H_est, H_gt, h, w)
        metrics['MACE'] = mace
        
        # Standard MSE between result and fixed
        mse_pixel = np.mean(((img1_result / 255.0) - (img0_np / 255.0)) ** 2)
        metrics['MSE'] = mse_pixel

        all_metrics.append(metrics)
        logger.info(f"Sample: {pair_name} | MACE: {mace:.4f} | ME: {metrics['mean_error']:.2f} | MAE: {metrics['max_error']:.2f} | Rep: {metrics['Rep']:.4f} | MIR: {metrics['MIR']:.4f}")

    # 6. Summary
    if all_metrics:
        total = len(all_metrics)
        # Success if mean_error is valid (found reasonable matches)
        valid_metrics = [m for m in all_metrics if np.isfinite(m.get('mean_error', float('inf')))]
        success_count = len(valid_metrics)
        failure_rate = ((total - success_count) / total) * 100
        
        logger.info("="*30)
        logger.info(f"Summary (Total: {total}, Success: {success_count}, Fail Rate: {failure_rate:.2f}%)")
        
        if success_count > 0:
            avg_mace = np.mean([m['MACE'] for m in valid_metrics if np.isfinite(m['MACE'])])
            avg_mse = np.mean([m['MSE'] for m in valid_metrics if np.isfinite(m['MSE'])])
            avg_me = np.mean([m['mean_error'] for m in valid_metrics])
            avg_mae = np.mean([m['max_error'] for m in valid_metrics])
            avg_sr_me = np.mean([m['SR_ME'] for m in valid_metrics])
            avg_sr_mae = np.mean([m['SR_MAE'] for m in valid_metrics])
            avg_rep = np.mean([m['Rep'] for m in valid_metrics])
            avg_mir = np.mean([m['MIR'] for m in valid_metrics])
            
            logger.info(f"  Avg MACE:    {avg_mace:.4f}")
            logger.info(f"  Avg MSE:     {avg_mse:.4f}")
            logger.info(f"  Avg ME:      {avg_me:.2f} px")
            logger.info(f"  Avg MAE:     {avg_mae:.2f} px")
            logger.info(f"  SR_ME:       {avg_sr_me:.4f}")
            logger.info(f"  SR_MAE:      {avg_sr_mae:.4f}")
            logger.info(f"  Repeatability: {avg_rep:.4f}")
            logger.info(f"  MIR:         {avg_mir:.4f}")
        logger.info("="*30)

if __name__ == "__main__":
    main()
