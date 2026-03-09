# 评估指标计算原则（统一标准版）

## 文档信息
- **版本**: v2.1 (统一标准，修复后)
- **参考实现**: LightGlue v2_multi (修复后)
- **维护者**: Fengjunming
- **最后更新**: 2026-03-09

---

## 一、总体架构

本项目采用**统一评估框架**，确保所有模型（LightGlue、MambaGlue、SuperGlue）在指标计算上完全一致。

### 核心组件
- **UnifiedEvaluator**：统一评估器类，负责累积误差并计算聚合指标
- **compute_homography_errors**：底层指标计算函数
- **evaluate_batch**：批次级评估接口
- **compute_epoch_metrics**：Epoch级聚合指标计算

---

## 二、评估流程

### 阶段 1：特征提取与匹配
```
输入：image0 (固定图), image1 (移动图)
输出：matches0 (匹配索引), keypoints0, keypoints1
```

### 阶段 2：单应矩阵估计（RANSAC）
```python
# 对每个样本独立计算
if num_matches >= 4:
    H_est, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thr=3.0)
else:
    H_est = np.eye(3)  # 匹配失败
```

### 阶段 3：误差计算

#### 3.1 数据集名称判断（必须包含所有医学眼底数据集）
```python
# 【关键】必须包含所有数据集名称，否则会走错分支
dataset_name = data['dataset_name'][0].lower()
multimodal_datasets = ['multimodal', 'realdataset', 'cffa', 'cfoct', 'octfa', 'cfocta']
if dataset_name in multimodal_datasets:
    return compute_homography_errors(data, config)  # 使用单应矩阵误差
else:
    return compute_pose_errors(data, config)  # 使用本质矩阵误差
```

#### 3.2 Failed 判断
```python
# 【关键】inliers_rate 计算必须使用全部匹配点数作为分母
inliers_count = np.sum(inliers.ravel() > 0)
inliers_rate = inliers_count / len(pts0)  # 必须用全部匹配点

# 失败条件（满足任一即失败）
is_failed = False
if num_matches < 4:
    is_failed = True
if H_est is None or np.isnan(H_est).any() or np.isinf(H_est).any():
    is_failed = True
if inliers_rate < 1e-6:
    is_failed = True
if np.allclose(H_est, np.eye(3), atol=1e-3):
    is_failed = True

# 失败样本处理
if is_failed:
    R_err = 0.0
    t_err = 1e6  # 【关键】必须是1e6，不是np.inf
    mse = np.inf
    mace = np.inf
    failed_mask = True
```

#### 3.3 MSE 计算（特征点坐标 MSE）
```python
# 与 test_on_CrossModality.py 的 cal_MSE 对齐
pts0_homo = pts0.reshape(-1, 1, 2).astype(np.float32)
pts1_pred = cv2.perspectiveTransform(pts0_homo, H_est).reshape(-1, 2)
mse = np.mean((pts1 - pts1_pred) ** 2)
```

#### 3.4 avg_dist / dis 计算
```python
# 用于判断 Inaccurate
dis = pts1 - pts1_pred
dis = np.sqrt(dis[:, 0] ** 2 + dis[:, 1] ** 2)
avg_dist = dis.mean()
```

#### 3.5 Inaccurate 判断
```python
mae = dis.max()   # 最大误差
mee = np.median(dis)  # 中位误差

# 不准确判定：mae > 50 或 mee > 20
is_inaccurate = (mae > 50.0) or (mee > 20.0)
```

#### 3.6 MACE 计算（角点误差）
```python
# 【关键】角点坐标定义必须使用 [w, h]，不是 [w-1, h-1]
corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
corners_h = np.concatenate([corners, np.ones((4, 1))], axis=-1)

# 真值单应矩阵投影
corners_gt_h = (H_gt @ corners_h.T).T
corners_gt = corners_gt_h[:, :2] / (corners_gt_h[:, 2:] + 1e-7)

# 估计单应矩阵投影
corners_est_h = (H_est @ corners_h.T).T
corners_est = corners_est_h[:, :2] / (corners_est_h[:, 2:] + 1e-7)

# MACE = 平均角点误差
mace = np.mean(np.linalg.norm(corners_est - corners_gt, axis=-1))
```

---

## 三、关键原则

### 1. 误差累积策略
- **AUC 相关**：累积所有样本的真实配准误差（MACE）
  - Failed 样本：error = **1e6**（不是 np.inf）
  - Success 样本（包括 inaccurate）：error = mace
  - 在 Epoch 结束后统一计算 AUC
- **MSE/MACE**：仅统计 Acceptable 样本（mae ≤ 50 且 mee ≤ 20）
  - Failed 样本：用 inf 填充，最后过滤掉
  - Inaccurate 样本：用 inf 填充，最后过滤掉

### 2. 匹配失败判定
满足以下任一条件视为**失败**（Failed）：
- 匹配点数 < 4
- H_est 为 None 或包含 nan/inf
- inliers rate < 1e-6
- H_est 接近单位矩阵（np.allclose(H, I, atol=1e-3)）

### 3. 不准确判定
满足以下任一条件视为**Inaccurate**（不准确但未失败）：
- mae（最大误差）> 50 像素
- mee（中位误差）> 20 像素

### 4. Acceptable 判定
- **Acceptable** = 成功样本 - Inaccurate 样本
- 即 mae ≤ 50 且 mee ≤ 20 的样本

### 5. 监控指标选择
- **训练监控**：combined_auc（平均 AUC，越高越好）
- **学习率调度**：ReduceLROnPlateau 监控 combined_auc
- **早停机制**：监控 combined_auc，patience=10，min_delta=0.0001

---

## 四、统一标准对照表

| 项目 | 标准值 | LightGlue | MambaGlue | SuperGlue | 状态 |
|------|--------|-----------|-----------|-----------|------|
| 数据集名称列表 | 6个 | ✅ | ✅ | ✅ | ✅ 已修复 |
| Failed → t_errs | 1e6 | ✅ | ✅ | ✅ | ✅ 已修复 |
| Failed → R_errs | 0.0 | ✅ | ✅ | ✅ | ✅ 已修复 |
| inliers_rate 分母 | len(pts0) | ✅ | ✅ | ✅ | ✅ 已修复 |
| 角点坐标 | [w,h] | ✅ | ✅ | ✅ | ✅ 已修复 |
| MSE 字段名 | mse_list | ✅ | ✅ | ✅ | ✅ 已修复 |
| MACE 字段名 | mace_list | ✅ | ✅ | ✅ | ✅ 已修复 |

---

## 五、返回值结构标准

### 5.1 compute_homography_errors 返回值
```python
data.update({
    'R_errs': [],        # 旋转误差（像素配准场景固定为0.0）
    't_errs': [],        # 平移误差（用于AUC，MACE或1e6）
    'inliers': [],       # 内点掩码
    'H_est': [],         # 估计的单应矩阵
    'mse_list': [],      # MSE（仅Acceptable样本有效）
    'mace_list': [],     # MACE（仅Acceptable样本有效）
    'failed_mask': [],   # 失败标记
    'inaccurate_mask': [],  # 不准确标记
})
```

### 5.2 compute_pose_errors 返回值
```python
data.update({
    'R_errs': [],
    't_errs': [],
    'inliers': [],
    'failed_mask': [],
    'inaccurate_mask': [],
})
```python
def compute_pose_errors(data, config):
    dataset_name = data['dataset_name'][0].lower()
    # 【必须】包含所有数据集名称
    multimodal_datasets = ['multimodal', 'realdataset', 'cffa', 'cfoct', 'octfa', 'cfocta']
    if dataset_name in multimodal_datasets:
        return compute_homography_errors(data, config)
    # ... 其他数据集使用本质矩阵
```

### 5.2 compute_homography_errors 返回值结构
```python
data.update({
    'R_errs': [],        # 旋转误差（像素配准场景固定为0.0）
    't_errs': [],        # 平移误差（用于AUC，MACE或1e6）
    'inliers': [],       # 内点掩码
    'H_est': [],         # 估计的单应矩阵
    'mse_list': [],      # MSE（仅Acceptable样本有效）
    'mace_list': [],    # MACE（仅Acceptable样本有效）
    'failed_mask': [],   # 失败标记
    'inaccurate_mask': [],  # 不准确标记
})
```

### 5.3 Spatial Binning 参数
```python
def spatial_binning(pts0, pts1, img_size, grid_size=4, top_n=20, conf=None):
    """
    空间均匀化：网格4x4，每格最多20个点
    仅用于RANSAC估计H，不影响匹配点
    """
```

---

## 六、聚合指标计算

### AUC 计算
```python
def error_auc(errors, thresholds):
    """
    thresholds: [5, 10, 20] 像素
    errors: 包含所有样本，Failed样本为1e6
    """
```

### mAUC 计算（ROP风格）
```python
def compute_auc_rop(errors, limit=25):
    """
    阈值上限：25像素
    """
```

---

## 七、常见问题

### Q1: 为什么 AUC 和 MSE/MACE 的样本数不一致？
**A**: AUC 包含所有样本（失败样本 error=1e6），MSE/MACE 仅包含匹配成功样本（过滤掉 inf）。

### Q2: 为什么 Failed 样本用 1e6 而非 np.inf？
**A**: np.inf 无法参与数值积分（np.trapz 会报错或得到错误结果），而 1e6 是一个足够大的数值，可以正常参与 AUC 计算。

### Q3: 为什么 inliers_rate 必须用 len(pts0) 作为分母？
**A**: 这是与 test_on_CrossModality.py 对齐的标准做法。文档明确要求 `inliers_rate = inliers_count / len(pts0)`。

### Q4: 为什么角点坐标用 [w, h] 而非 [w-1, h-1]？
**A**: 图像角点坐标应该是图像边界外侧，右下角坐标为 [w, h]（超出图像范围1像素），这样才能正确计算四个角点的投影误差。使用 [w-1, h-1] 会导致角点位于图像内部，与标准定义不符。

---

## 八、附录：标准参数汇总

| 参数 | 标准值 |
|------|--------|
| RANSAC 阈值 | 3.0 |
| AUC 阈值 | [5, 10, 20] |
| mAUC 上限 | 25 |
| Inaccurate MAE 阈值 | 50 |
| Inaccurate MEE 阈值 | 20 |
| H≈I 判定 atol | 1e-3 |
| inliers_rate 失败阈值 | 1e-6 |
| Spatial Binning grid | 4 |
| Spatial Binning top_n | 20 |
| 角点坐标 | [[0,0], [w,0], [w,h], [0,h]] |
| Failed t_errs | 1e6 |
| Failed R_errs | 0.0 |
