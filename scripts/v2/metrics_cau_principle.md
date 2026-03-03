# LightGlue 评估指标计算原则

## 一、总体架构

本项目采用**统一评估框架**（`scripts/v2/test.py`），确保训练（`train_onGen_vessels.py`、`train_onReal.py`）和测试（`test.py`）阶段的指标计算完全一致。

### 核心组件
- **UnifiedEvaluator**：统一评估器类，负责累积误差并计算聚合指标
- **compute_homography_errors**：底层指标计算函数（来自 `scripts/v1/metrics.py`）
- **evaluate_batch**：批次级评估接口
- **compute_epoch_metrics**：Epoch级聚合指标计算

---

## 二、评估流程（五阶段）

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

**防爆锁机制**：
- 检查 `H_est` 是否为 None、包含 NaN/Inf
- 检查行列式是否在合理范围 [0.1, 10.0]
- 检查透视分量是否过大（阈值 0.005）
- 不合格的矩阵强制替换为单位矩阵

### 阶段 3：图像配准与质量评估
```python
# 仅对匹配成功的样本计算
H_inv = np.linalg.inv(H_est)
img1_result = cv2.warpPerspective(img1, H_inv, (w, h))

# MSE：配准结果与真值的像素级误差
res_f, orig_f = filter_valid_area(img1_result, img1_gt)
mse = np.mean((res_f - orig_f)**2)  # 仅在有效区域计算

# MACE：四角点平均重投影误差
corners = [[0,0], [w,0], [w,h], [0,h]]
corners_gt = H_gt @ corners
corners_est = H_est @ corners
mace = np.mean(||corners_est - corners_gt||)
```

### 阶段 4：几何误差计算（用于 AUC）
```python
# 在 metrics.py 的 compute_homography_errors 中执行
# 使用 Spatial Binning 优化 RANSAC
bin_indices = spatial_binning(pts0, pts1, img_size, grid_size=4, top_n=20)

# 计算角点误差作为 t_errs（用于 AUC 计算）
corners_gt = H_gt @ corners
corners_est = H_est @ corners
t_err = np.mean(||corners_est - corners_gt||)  # 单位：像素
```

**关键修正**：
- `R_errs` 固定为 0.0（眼底配准无旋转误差概念）
- `t_errs` 使用角点误差（Corner Error），而非内点残差
- 这确保了 AUC 计算反映**估计H与真值H的偏差**

### 阶段 5：Epoch级聚合指标
```python
# 累积所有样本的 t_errs 后统一计算
all_errors = [t_err_sample1, t_err_sample2, ...]

# AUC 计算（阈值单位：像素）
auc@5  = error_auc(all_errors, threshold=5)
auc@10 = error_auc(all_errors, threshold=10)
auc@20 = error_auc(all_errors, threshold=20)
combined_auc = (auc@5 + auc@10 + auc@20) / 3

# mAUC（ROP风格，0-25像素）
mAUC = compute_auc_rop(all_errors, limit=25)

# MSE/MACE（仅匹配成功样本）
avg_mse = mean(all_mses)
avg_mace = mean(all_maces)
inverse_mace = 1.0 / (1.0 + avg_mace)

# 匹配失败率
match_failure_rate = failed_samples / total_samples
```

---

## 三、关键原则

### 1. 误差累积策略
- **AUC 相关**：累积所有样本的 `t_errs`（包括失败样本的 inf），在 Epoch 结束后统一计算
- **MSE/MACE**：仅累积匹配成功样本的值，失败样本不参与计算

### 2. 匹配失败判定
满足以下任一条件视为失败：
- 匹配点数 < 4
- `H_est` 未通过防爆锁检查
- `H_est` 接近单位矩阵（`np.allclose(H, I, atol=1e-3)`）

### 3. 有效区域过滤
```python
# filter_valid_area：仅保留两张图都非黑色的像素
mask = (img1 > 10) & (img2 > 10)
# 裁剪到最小包围矩形，避免黑边干扰 MSE 计算
```

### 4. Spatial Binning（空间均匀化）
```python
# 将图像划分为 4x4 网格，每格最多保留 Top-20 匹配点
# 目的：避免匹配点聚集导致 RANSAC 偏差
grid_size = 4
top_n = 20
```

### 5. 监控指标选择
- **训练监控**：`combined_auc`（平均 AUC，越高越好）
- **学习率调度**：ReduceLROnPlateau 监控 `combined_auc`
- **早停机制**：监控 `combined_auc`，patience=10，min_delta=0.0001

---

## 四、三个脚本的一致性验证

| 项目 | test.py | train_onGen_vessels.py | train_onReal.py |
|------|---------|------------------------|-----------------|
| 评估器 | UnifiedEvaluator | UnifiedEvaluator | UnifiedEvaluator |
| RANSAC阈值 | 3.0 | 3.0 | 3.0 |
| 防爆锁 | ✅ | ✅ | ✅ |
| Spatial Binning | ✅ | ✅ | ✅ |
| AUC阈值 | [5,10,20] | [5,10,20] | [5,10,20] |
| mAUC上限 | 25 | 25 | 25 |
| MSE计算 | filter_valid_area | filter_valid_area | filter_valid_area |
| MACE计算 | compute_corner_error | compute_corner_error | compute_corner_error |
| 失败样本处理 | t_err=inf, 不计入MSE/MACE | t_err=inf, 不计入MSE/MACE | t_err=inf, 不计入MSE/MACE |

**结论**：三个脚本的评估指标计算方式完全一致，均通过 `UnifiedEvaluator` 调用 `metrics.py` 的底层函数。

---

## 五、日志与调试

### 详细日志控制
```python
# 训练时：关闭详细日志（避免刷屏）
set_metrics_verbose(False)

# 测试时：开启详细日志（便于调试）
set_metrics_verbose(True)
```

### 关键日志输出
- `🔍 Batch X: 总匹配点=N, Spatial Binning后=M`
- `✅ Batch X: RANSAC 成功, inliers=N/M`
- `⚠️ Batch X: H_est 接近单位矩阵!`
- `⚠️ Batch X: Inliers 数量较少 (N)`

---

## 六、特殊说明

### 1. 课程学习（仅 train_onGen_vessels.py）
```python
# 血管引导的加权损失（不影响评估指标）
vessel_loss_weight: 10.0 -> 1.0 (Epoch 0-100)
```

### 2. 数据集差异
- **Gen模式**：训练集=生成数据，验证集=CFFA真实数据
- **Real模式**：训练集=CFFA训练集，验证集=CFFA测试集

### 3. 可视化触发
- 最优模型（`combined_auc` 提升时）
- 每5个Epoch
- 仅可视化验证集中的测试样本（`split='test'`），最多20个

---

## 七、代码引用路径

```
scripts/v2/test.py
├── UnifiedEvaluator.evaluate_batch()
│   ├── cv2.findHomography()  # RANSAC估计H
│   ├── is_valid_homography()  # 防爆锁
│   ├── filter_valid_area()    # MSE计算
│   ├── compute_corner_error() # MACE计算
│   └── compute_homography_errors()  # 调用metrics.py
│       ├── spatial_binning()  # 空间均匀化
│       └── 计算t_errs（角点误差）
└── UnifiedEvaluator.compute_epoch_metrics()
    ├── error_auc()           # AUC@5/10/20
    ├── compute_auc_rop()     # mAUC
    └── 聚合MSE/MACE/失败率
```

---

## 八、常见问题

### Q1: 为什么 AUC 和 MSE/MACE 的样本数不一致？
**A**: AUC 包含所有样本（失败样本 t_err=inf），MSE/MACE 仅包含匹配成功样本。

### Q2: 为什么 t_errs 使用角点误差而非内点残差？
**A**: 内点残差反映 RANSAC 拟合质量，角点误差反映估计H与真值H的偏差，后者更符合配准任务的评估目标。

### Q3: Spatial Binning 会影响最终配准结果吗？
**A**: 不会。Spatial Binning 仅用于 RANSAC 估计 H，不影响匹配点的生成和最终的图像变换。

### Q4: 如何确保训练和测试指标一致？
**A**: 统一使用 `UnifiedEvaluator`，避免在不同脚本中重复实现指标计算逻辑。

---

**文档版本**: v1.0  
**最后更新**: 2026-03-03  
**维护者**: Fengjunming
