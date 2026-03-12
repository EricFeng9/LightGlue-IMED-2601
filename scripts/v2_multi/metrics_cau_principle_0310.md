# 统一评估指标计算方案

## 文档信息
- **版本**: v3.2 (Seed机制 + num_workers=0)
- **维护者**: Fengjunming
- **最后更新**: 2026-03-11
- **目的**: 保证不同配准模型之间评估流程和指标计算的一致性

---

## ⚠️ 核心要求：所有指标必须基于GT关键点计算

**【强制要求】** 本评估方案要求所有用于评估的数据集**必须包含手工标注的GT关键点**。

- 所有指标（MSE、GT-MACE、AUC、Inaccurate判定等）**必须**基于GT关键点计算
- 如果样本缺少GT关键点，程序会**直接报错并退出**，不存在向后兼容
- 不允许使用模型匹配点或图像角点作为备选计算方式

---

## 一、脚本职责说明

### 1.1 训练脚本

| 脚本 | 用途 | 数据来源 |
|------|------|----------|
| `train_onMultiGen_vessels_enhanced.py` | 使用**生成数据**训练配准模型 | `data/260305_1_v30/` (生成的多模态眼底数据) |
| `train_onReal.py` | 使用**真实数据**训练配准模型 | `data/CFFA/`, `data/CF_OCT/`, `operation_pre_filtered_octfa` (真实医学影像) |

> **注意**: CFFA 和 CFOCT 数据集已更新为使用 `data/CFFA/` 和 `data/CF_OCT/` 目录下的新版本数据集类 (`cffa_dataset.py`, `cf_oct_dataset.py`)。OCTFA 继续使用 `operation_pre_filtered_octfa`。

### 1.2 测试脚本

| 脚本 | 用途 | 评估内容 |
|------|------|----------|
| `test.py` | 单模态评估 | 对比 train on multi gen vs. train on real 在各模态(CFFA/CFOCT/OCTFA)上的表现。**CFFA/CFOCT 使用全部数据 (`split='all'`)** |
| `test_all_operationpre.py` | 全面评估 | 对比 train on multi gen vs. baseline (LightGlue原Pretrained) 在三个真实数据集的全量数据上表现 |

### 1.3 核心模块

| 模块 | 位置 | 用途 |
|------|------|------|
| `metrics.py` | `scripts/v2_multi/metrics.py` | 统一的指标计算核心函数 |
| `test.py` (UnifiedEvaluator) | `scripts/v2_multi/test.py` | 训练/验证/测试共用的评估器 |

---

## 二、数据集脚本返回格式

### 2.1 基础数据集类要求

所有用于训练/测试的数据集类**必须**实现以下接口：

```python
class BaseDataset(Dataset):
    def __getitem__(self, idx):
        """
        返回格式:
        - fix_tensor: 固定图tensor [1, H, W] 或 [3, H, W], 值域 [0,1] 或 [-1,1]
        - moving_original_tensor: 移动图tensor [1, H, W] 或 [3, H, W], 值域 [-1,1]
        - moving_gt_tensor: 配准后的移动图tensor [1, H, W] 或 [3, H, W], 值域 [-1,1]
        - fix_path: 固定图路径 (str)
        - moving_path: 移动图路径 (str)
        - T_0to1: 从固定图到移动图的单应矩阵 (torch.Tensor, 3x3)
        """
        return fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1

    def get_raw_sample(self, idx):
        """
        返回未处理的原始数据（包含GT关键点）
        
        返回格式:
        - img_fix: 固定图原始图像 (numpy array, H x W)
        - img_moving: 移动图原始图像 (numpy array, H x W)
        - fix_points: 固定图上的GT关键点 (numpy array, N x 2) - 配对的关键点
        - moving_points: 移动图上的GT关键点 (numpy array, N x 2) - 与fix_points配对
        - fix_path: 固定图路径
        - moving_path: 移动图路径
        """
        return img_fix, img_moving, fix_points, moving_points, fix_path, moving_path
```

### 2.2 各数据集的GT关键点格式

| 数据集 | 数据集类 | GT关键点文件格式 | 关键点含义 |
|--------|----------|-----------------|-----------|
| `CFFA` | `data.CFFA.cffa_dataset.CFFADataset` | `{id}_01.txt` (CF点), `{id}_02.txt` (FA点) | 每行一个点 `x y`，在原始图像坐标空间中 |
| `CFOCT` | `data.CF_OCT.cf_oct_dataset.CFOCTDataset` | `{id}_01.txt` (OCT点), `{id}_02.txt` (FA点) | 同上 |
| `OCTFA` | `operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset.OCTFADataset` | `{id}_01.txt` (FA点), `{id}_02.txt` (OCT点) | 同上 |
| `CFOCTA` | `data.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset.CFOCTADataset` | `Ground_Truth_Ts/{id}Fundus_OCTA_points.txt` | 格式: `CF_x CF_y OCTA_x OCTA_y` (x0 y0 x1 y1)，基于256x256空间 |

> **注意**: CFFA 和 CFOCT 数据集支持三种 split 模式：
> - `'train'`: 训练集 (80% 的眼睛ID)
> - `'val'`: 验证集 (20% 的眼睛ID)
> - `'all'`: **全部数据** (训练集 + 验证集合并)，用于测试时使用完整数据集

---

## 三、数据集划分模式

### 3.1 split 参数说明

CFFA 和 CFOCT 数据集支持通过 `split` 参数控制数据集划分：

| split 值 | 含义 | 用途 |
|----------|------|------|
| `'train'` | 80% 的眼睛ID | 训练 |
| `'val'` | 20% 的眼睛ID | 验证 |
| `'all'` | 100% 的眼睛ID (全部数据合并) | **测试评估** |

> **【重要】** 测试时使用的数据集应使用 `split='all'`，以利用完整的数据集进行评估，获得更全面的性能指标。

### 3.2 数据集类的 split 实现

```python
class CFFADataset(Dataset):
    def __init__(self, root_dir, split='val', mode='cf2fa', train_ratio=0.8, seed=42):
        # 1. 扫描所有眼睛ID
        all_ids = sorted([d for d in os.listdir(root_dir) if d.isdigit()])
        
        # 2. 按眼睛ID进行划分（固定种子保证可复现）
        random.Random(seed).shuffle(all_ids)
        num_train = int(len(all_ids) * train_ratio)
        
        if split == 'train':
            self.patient_ids = all_ids[:num_train]
        elif split == 'all':
            # 全部数据（训练+验证）
            self.patient_ids = all_ids
        else:  # val
            self.patient_ids = all_ids[num_train:]
```

---

## 四、训练脚本数据封装

### 4.1 RealDatasetWrapper (训练脚本中使用)

训练脚本 (`train_onMultiGen_vessels_enhanced.py`, `train_onReal.py`) 使用 `RealDatasetWrapper` 封装数据集：

```python
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, split_name='unknown', dataset_name='MultiModal'):
        self.base_dataset = base_dataset
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.target_size = 512  # 目标图像大小

    def __getitem__(self, idx):
        # 1. 获取基础数据
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        
        # 2. 获取并缩放GT关键点
        if hasattr(self.base_dataset, 'get_raw_sample'):
            raw_sample = self.base_dataset.get_raw_sample(idx)
            fix_points = raw_sample[2]  # (N, 2)
            moving_points = raw_sample[3]  # (N, 2)
            
            # 缩放到目标尺寸
            h_fix, w_fix = raw_sample[0].shape[:2]
            h_mov, w_mov = raw_sample[1].shape[:2]
            
            if len(fix_points) > 0:
                fix_points = fix_points.copy()
                fix_points[:, 0] *= self.target_size / w_fix
                fix_points[:, 1] *= self.target_size / h_fix
            if len(moving_points) > 0:
                moving_points = moving_points.copy()
                moving_points[:, 0] *= self.target_size / w_mov
                moving_points[:, 1] *= self.target_size / h_mov
        
        # 3. 转换并返回
        return {
            'image0': fix_gray,           # [1, 512, 512]
            'image1': moving_orig_gray,   # [1, 512, 512]
            'image1_gt': moving_gray,    # [1, 512, 512]
            'T_0to1': T_0to1,            # [3, 3]
            'pair_names': (fix_name, moving_name),
            'dataset_name': self.dataset_name,
            'split': self.split_name,
            'gt_pts0': fix_points_tensor,    # (N, 2) torch.Tensor
            'gt_pts1': moving_points_tensor,  # (N, 2) torch.Tensor
        }
```

### 4.2 batch 格式 (传入模型)

```python
batch = {
    'image0': torch.Tensor,           # [B, 1, 512, 512] 固定图
    'image1': torch.Tensor,           # [B, 1, 512, 512] 移动图(原始/变形)
    'image1_gt': torch.Tensor,        # [B, 1, 512, 512] 移动图(GT/配准后)
    'T_0to1': torch.Tensor,          # [B, 3, 3] GT单应矩阵
    'pair_names': tuple of str,      # (fix_name, moving_name) for each sample
    'dataset_name': str or list,     # 'CFFA'/'CFOCT'/'OCTFA'/'CFOCTA'
    'split': str,                     # 'train'/'val'
    'gt_pts0': torch.Tensor,          # [B, N, 2] GT关键点(固定图)
    'gt_pts1': torch.Tensor,          # [B, N, 2] GT关键点(移动图,配对)
}
```

---

## 五、指标计算流程

### 5.1 核心函数: compute_homography_errors

位于 `scripts/v2_multi/metrics.py`，是所有指标计算的核心。

**调用位置**:
- 训练脚本: `PL_LightGlue_Gen.validation_step()` 中调用
- 测试脚本: `UnifiedEvaluator.evaluate_batch()` 中调用

**输入数据**:
```python
data = {
    'mkpts0_f': torch.Tensor,    # 模型匹配点(固定图) [M, 2]
    'mkpts1_f': torch.Tensor,    # 模型匹配点(移动图) [M, 2]
    'm_bids': torch.Tensor,      # 匹配点对应的batch索引 [M]
    'T_0to1': torch.Tensor,      # GT单应矩阵 [B, 3, 3]
    'image0': torch.Tensor,     # 固定图 [B, 1, H, W]
    'dataset_name': list,        # 数据集名称列表
    'gt_pts0': list,             # GT关键点列表 [B] x (N, 2)
    'gt_pts1': list,             # GT关键点列表 [B] x (N, 2)
}
config = {
    'TRAINER.RANSAC_PIXEL_THR': 3.0,  # RANSAC阈值
}
```

### 5.2 指标计算详细逻辑

#### 步骤1: RANSAC估计单应矩阵

```python
# 对每个batch样本独立计算
for bs in range(B):
    # 1. 提取当前样本的匹配点
    mask = m_bids == bs
    pts0_batch = mkpts0_f[mask]  # 当前样本的匹配点
    pts1_batch = mkpts1_f[mask]
    
    # 2. 空间均匀化 (Spatial Binning)
    bin_indices = spatial_binning(pts0_batch, pts1_batch, img_size, 
                                  grid_size=4, top_n=20)
    
    # 3. RANSAC估计单应矩阵
    H_est, inliers = cv2.findHomography(pts0_ransac, pts1_ransac, 
                                         cv2.RANSAC, ransac_thr=3.0)
```

#### 步骤2: Failed判定

```python
# 失败条件(满足任一即失败):
is_failed = False
if num_matches < 4:  # 匹配点不足
    is_failed = True
if H_est is None or has_nan_or_inf(H_est):  # 单应矩阵无效
    is_failed = True
if inliers_rate < 1e-6:  # 内点率过低
    is_failed = True
if np.allclose(H_est, np.eye(3), atol=1e-3):  # 接近单位矩阵
    is_failed = True

# 失败样本处理
if is_failed:
    t_err = 1e6      # 用于AUC计算
    mse = np.inf     # 无效
    gt_mace = np.inf # 无效
    failed_mask = True
```

#### 步骤3: 成功样本 - GT关键点重投影误差计算

```python
# 获取当前样本的GT关键点
gt_pts0 = gt_pts0_list[bs]  # (N, 2) 固定图上的GT关键点
gt_pts1 = gt_pts1_list[bs]  # (N, 2) 移动图上的GT关键点

# 【强制要求】所有指标必须基于GT关键点计算
# 如果没有GT关键点，直接报错退出
if gt_pts0 is None or gt_pts1 is None:
    raise ValueError(f"Batch {bs}: 缺少GT关键点数据 (gt_pts0或gt_pts1为None)")

if len(gt_pts0) < 4 or len(gt_pts1) < 4:
    raise ValueError(f"Batch {bs}: GT关键点数量不足 (fix:{len(gt_pts0)}, moving:{len(gt_pts1)})，需要至少4个关键点")

# 1. 用估计的单应矩阵投影GT关键点
gt_pts0_homo = gt_pts0.reshape(-1, 1, 2).astype(np.float32)
gt_pts1_pred = cv2.perspectiveTransform(gt_pts0_homo, H_est).reshape(-1, 2)

# 2. 计算重投影误差
diff = gt_pts1 - gt_pts1_pred
dis = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)  # 每个GT关键点的误差

# 3. 计算MSE和MACE
mse = np.mean(diff ** 2)
gt_mace = np.mean(dis)  # GT-MACE: GT关键点的平均重投影误差

# 4. Inaccurate判定（基于GT关键点）
mae = np.max(dis)     # 最大误差
mee = np.median(dis) # 中位误差
is_inaccurate = (mae > 50.0) or (mee > 20.0)
```

> **【重要】** 不允许缺少GT关键点的样本。所有指标（MSE、GT-MACE、AUC、Inaccurate判定）必须基于手工标注的GT关键点计算。如果样本没有GT关键点，程序会直接报错并退出，而非向后兼容使用模型匹配点或图像角点。

#### 步骤4: 聚合指标计算

```python
# AUC计算 (使用所有样本)
# Failed样本: error = 1e6
# Success样本: error = gt_mace (包括inaccurate)
error_auc(all_errors, [5, 10, 20])

# mAUC计算 (ROP风格)
compute_auc_rop(all_errors, limit=25)

# MSE/GT-MACE (仅Acceptable样本)
# Acceptable = Success - Inaccurate
mse_list = [m for m in mse_list if np.isfinite(m)]
gt_mace_list = [m for m in gt_mace_list if np.isfinite(m)]
avg_mse = np.mean(mse_list)
avg_gt_mace = np.mean(gt_mace_list)
```

---

## 六、训练/验证/测试中的指标使用

### 6.1 训练阶段 (training)

**位置**: `train_onMultiGen_vessels_enhanced.py` / `train_onReal.py`

```python
class PL_LightGlue_Gen(pl.LightningModule):
    def __init__(self, config, result_dir=None):
        # 初始化评估器
        self.evaluator = UnifiedEvaluator(mode='gen', config=config)
    
    def validation_step(self, batch, batch_idx):
        # 1. 前向传播
        outputs = self(batch)
        
        # 2. 计算损失(用于训练)
        loss = self._compute_loss(outputs, ...)
        self.log('val_loss', loss, ...)
        
        # 3. 使用评估器计算指标
        result = self.evaluator.evaluate_batch(batch, outputs, self)
        return result
    
    def on_validation_epoch_end(self):
        # 4. 聚合epoch指标
        metrics = self.evaluator.compute_epoch_metrics()
        
        # 5. 记录指标用于早停和日志
        self.log('auc@5', metrics['auc@5'], ...)
        self.log('auc@10', metrics['auc@10'], ...)
        self.log('auc@20', metrics['auc@20'], ...)
        self.log('mAUC', metrics['mAUC'], ...)
        self.log('combined_auc', metrics['combined_auc'], ...)  # 用于早停
        self.log('val_mse', metrics['mse'], ...)
        self.log('val_mace', metrics['mace'], ...)
        
        # 6. 重置评估器
        self.evaluator.reset()
```

### 6.2 测试阶段 (evaluation)

**位置**: `test.py` / `test_all_operationpre.py`

```python
def run_evaluation(pl_module, dataloader, config):
    # 1. 初始化评估器
    evaluator = UnifiedEvaluator(config=config)
    
    # 2. 遍历所有batch
    for batch_idx, batch in enumerate(dataloader):
        outputs = pl_module(batch)
        
        # 3. 评估每个batch
        result = evaluator.evaluate_batch(batch, outputs, pl_module)
    
    # 4. 聚合所有指标
    metrics = evaluator.compute_epoch_metrics()
    
    # 5. 按数据集分别统计
    for dataset_name in evaluator.per_dataset_errors:
        ds_metrics = compute_metrics_for_dataset(evaluator, dataset_name)
        metrics['per_dataset'][dataset_name] = ds_metrics
    
    return metrics
```

---

## 七、输出指标说明

### 7.1 指标定义

| 指标 | 计算方法 | 用途 |
|------|----------|------|
| **AUC@5/10/20** | 误差<5/10/20像素的样本比例曲线下面积 | 配准精度 |
| **mAUC** | ROP风格mAUC,阈值上限25像素 | 综合配准精度 |
| **Combined AUC** | (AUC@5 + AUC@10 + AUC@20)/3 | 综合指标,用于早停 |
| **MSE** | GT关键点重投影坐标均方误差 | 点位精度(仅Acceptable) |
| **GT-MACE** | GT关键点重投影平均误差 | 关键点精度(仅Acceptable) |
| **Failed** | 匹配点数<4/内点率过低/H无效 | 匹配失败 |
| **Inaccurate** | mae>50 或 mee>20 | 配准不准但未失败 |
| **Acceptable** | Success - Inaccurate | 成功且精度足够 |

### 7.2 样本分类

```
总样本 = Failed + Success
Success = Acceptable + Inaccurate
```

- **Failed**: 匹配失败,error=1e6
- **Inaccurate**: 配准不准,仍计入AUC但MSE/GT-MACE为inf
- **Acceptable**: 配准成功且准确,MSE/GT-MACE有效

---

## 八、关键代码位置索引

### 8.1 metrics.py 关键函数

| 函数 | 位置 | 功能 |
|------|------|------|
| `compute_homography_errors` | L328-520 | 核心指标计算 |
| `_reprojection_stats_gt` | L294-325 | GT关键点重投影误差 |
| `_reprojection_stats` | L272-291 | 模型匹配点重投影误差 |
| `spatial_binning` | L225-254 | 空间均匀化 |
| `error_auc` | L454-471 | AUC计算 |
| `compute_auc_rop` | L474-491 | mAUC计算 |

### 8.2 test.py UnifiedEvaluator

| 方法 | 位置 | 功能 |
|------|------|------|
| `__init__` | L213-223 | 初始化 |
| `reset` | L225-239 | 重置累积指标 |
| `evaluate_batch` | L261-312 | 评估单个batch |
| `compute_epoch_metrics` | L347-387 | 聚合epoch指标 |

### 8.3 训练脚本验证流程

| 脚本 | 验证调用位置 |
|------|--------------|
| `train_onMultiGen_vessels_enhanced.py` | L502 `validation_step` → L518 `evaluator.evaluate_batch` |
| `train_onReal.py` | 类似位置 |

---

## 九、数据流总结图

```
                    ┌─────────────────────────────────────┐
                    │         数据集类                     │
                    │  __getitem__() → 基础数据            │
                    │  get_raw_sample() → GT关键点        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │    RealDatasetWrapper               │
                    │  - 图像预处理(Resize to 512x512)    │
                    │  - GT关键点缩放                      │
                    │  - 转换为Tensor                      │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      DataLoader (batch)              │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      配准模型 (LightGlue)           │
                    │  - 特征提取                         │
                    │  - 匹配                             │
                    │  → matches0, mkpts0_f, mkpts1_f    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  UnifiedEvaluator.evaluate_batch()  │
                    │  → compute_homography_errors()     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │    compute_homography_errors()     │
                    │  1. RANSAC估计H                    │
                    │  2. Failed判定                     │
                    │  3. GT关键点重投影误差             │
                    │  4. Inaccurate判定                 │
                    │  → t_errs, mse_list, mace_list     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  UnifiedEvaluator.compute_epoch_   │
                    │  metrics()                         │
                    │  → AUC@5/10/20, mAUC, MSE, MACE   │
                    └─────────────────────────────────────┘
```

---

## 十、附录: 统一参数标准

| 参数 | 标准值 | 说明 |
|------|--------|------|
| RANSAC阈值 | 3.0 | 像素 |
| AUC阈值 | [5, 10, 20] | 像素 |
| mAUC上限 | 25 | 像素 |
| Inaccurate MAE阈值 | 50 | 像素 |
| Inaccurate MEE阈值 | 20 | 像素 |
| inliers_rate失败阈值 | 1e-6 | - |
| Spatial Binning grid | 4 | - |
| Spatial Binning top_n | 20 | - |
| 目标图像大小 | 512x512 | - |
| Failed t_err | 1e6 | 用于AUC |



---

## 十一、随机种子 (Seed) 机制

### 11.1 目的
为了确保训练、验证和测试结果的可复现性，必须设置随机种子。

### 11.2 支持 seed 的脚本

| 脚本 | seed 参数 | 说明 |
|------|-----------|------|
| `test.py` | `--seed` | 测试脚本，支持指定 seed |
| `test_all_operationpre.py` | `--seed` | 全面测试脚本，支持指定 seed |
| `train_onMultiGen_vessels_enhanced.py` | `--seed` | 训练脚本，支持指定 seed |
| `train_onReal.py` | `--seed` | 训练脚本，支持指定 seed |

### 11.3 seed 使用方法

```bash
# 指定 seed（结果可复现）
python scripts/v2_multi/test.py -s train_onMultiGen_vessels_enhanced \
    -n 260309_4_v30_Multi_Hpatience_enhanced -t test_1_cfoct \
    --seed 42 --num_workers 0

# 不指定 seed（自动生成，基于毫秒级时间戳）
python scripts/v2_multi/test.py -s train_onMultiGen_vessels_enhanced \
    -n 260309_4_v30_Multi_Hpatience_enhanced -t test_1_cfoct \
    --num_workers 0
```

### 11.4 seed 验证规则

- seed 必须在 `[0, 2^31)` 范围内
- 如果不指定 seed，系统自动生成：`int(time.time() * 1000) % (2**31)`

### 11.5 seed 设置内容

设置 seed 时，会同时设置以下随机源：

```python
import random
import numpy as np
import torch

random.seed(seed)           # Python 随机库
np.random.seed(seed)        # NumPy
torch.manual_seed(seed)     # PyTorch CPU
torch.cuda.manual_seed_all(seed)  # PyTorch GPU
torch.backends.cudnn.deterministic = True  # CUDA 卷积确定性
torch.backends.cudnn.benchmark = False     # 禁用 benchmark
```

### 11.6 注意事项

即使设置了 seed，由于以下原因，**仍可能存在微小差异**：

1. **CUDA 原子操作的非确定性**：某些 CUDA 操作执行顺序可能有微小差异
2. **GPU 并行计算顺序**：相同 tensor 操作的执行顺序可能因并行方式不同
3. **特征点检测算法**：SuperPoint 内部可能有一些随机操作

通常这种差异很小（在小数点后几位），属于正常现象。配合 `--num_workers 0` 使用可最大化结果稳定性。

---

## 十二、num_workers 参数

### 12.1 默认值

所有训练和测试脚本的 `num_workers` 默认值设为 **0**，以减少多进程数据加载带来的不确定性。

### 12.2 各脚本默认值

| 脚本 | num_workers 默认值 |
|------|-------------------|
| `test.py` | 0 |
| `test_all_operationpre.py` | 0 |
| `train_onMultiGen_vessels_enhanced.py` | 0 |
| `train_onReal.py` | 0 |

### 12.3 说明

- `num_workers=0` 表示在主进程中串行加载数据，消除了多进程调度的随机性
- 牺牲一点数据加载速度，换取更高的结果可复现性
- 特别是在调试和对比实验时，强烈建议使用 `num_workers=0`

