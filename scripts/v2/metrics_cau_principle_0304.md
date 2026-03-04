# 评估指标计算原则

## 一、总体架构

本项目采用**统一评估框架**（`scripts/v1/test.py`），确保训练（`train_onGen_vessels.py`、`train_onReal.py`）和测试（`test.py`）阶段的指标计算完全一致，并与 `test_on_CrossModality.py` 对齐。

### 核心组件
- **UnifiedEvaluator**：统一评估器类，负责累积误差并计算聚合指标
- **compute_homography_errors**：底层指标计算函数（来自 `scripts/v1/metrics.py`）
- **evaluate_batch**：批次级评估接口
- **compute_epoch_metrics**：Epoch级聚合指标计算

---

## 二、评估流程（与 test_on_CrossModality.py 对齐）

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

### 阶段 3：误差计算（与 test_on_CrossModality.py 完全对齐）

#### 3.1 Failed 判断
```python
# 与 test_on_CrossModality.py 对齐：inliers < 1e-6 视为失败
inliers_count = np.sum(inliers.ravel() > 0)
inliers_rate = inliers_count / len(pts0)
if inliers_rate < 1e-6:
    # 失败样本
    failed_samples += 1
    mse = inf
    all_errors.append(1e6)  # 用于 AUC 计算
```

#### 3.2 MSE 计算（特征点坐标 MSE）
```python
# 与 test_on_CrossModality.py 的 cal_MSE 对齐
# 使用 cv2.perspectiveTransform 将 pts0 用 H 变换，然后计算 MSE
pts0_homo = pts0.reshape(-1, 1, 2).astype(np.float32)
pts1_pred = cv2.perspectiveTransform(pts0_homo, H).reshape(-1, 2)

# 计算 MSE（特征点坐标 MSE）
mse = np.mean((pts1 - pts1_pred) ** 2)
```

#### 3.3 avg_dist 计算（用于 AUC）
```python
# 与 test_on_CrossModality.py 对齐：使用匹配点的平均重投影误差
dis = (pts1 - pts1_pred) ** 2
dis = np.sqrt(dis[:, 0] + dis[:, 1])
avg_dist = dis.mean()
all_errors.append(avg_dist)  # 用于 AUC 计算
```

#### 3.4 Inaccurate 判断
```python
# 与 test_on_CrossModality.py 对齐
mae = dis.max()   # 最大误差
mee = np.median(dis)  # 中位误差

# 不准确判定：mae > 50 或 mee > 20
if mae > 50.0 or mee > 20.0:
    inaccurate_samples += 1
```

#### 3.5 MACE 计算（角点误差）
```python
# 四角点平均重投影误差
corners = [[0,0], [w,0], [w,h], [0,h]]
corners_gt = H_gt @ corners
corners_est = H_est @ corners
mace = np.mean(||corners_est - corners_gt||)
```

---

## 三、关键原则

### 1. 误差累积策略
- **AUC 相关**：累积所有样本的误差
  - Failed 样本：error = 1e6
  - Success 样本（包括 inaccurate）：error = avg_dist
  - 在 Epoch 结束后统一计算 AUC
- **MSE/MACE**：仅统计 Acceptable 样本（mae ≤ 50 且 mee ≤ 20）
  - Failed 样本：用 inf 填充，最后过滤掉
  - Inaccurate 样本：用 inf 填充，最后过滤掉（与 test_on_CrossModality.py 对齐）

### 2. 匹配失败判定（与 test_on_CrossModality.py 对齐）
满足以下任一条件视为**失败**（Failed）：
- inliers rate < 1e-6
- 匹配点数 < 4
- `H_est` 接近单位矩阵（`np.allclose(H, I, atol=1e-3)`）

### 3. 不准确判定（与 test_on_CrossModality.py 对齐）
满足以下任一条件视为**Inaccurate**（不准确但未失败）：
- `mae`（最大误差）> 50 像素
- `mee`（中位误差）> 20 像素

### 4. Acceptable 判定
- **Acceptable** = 成功样本 - Inaccurate 样本
- 即 mae ≤ 50 且 mee ≤ 20 的样本

### 5. 监控指标选择
- **训练监控**：`combined_auc`（平均 AUC，越高越好）
- **学习率调度**：ReduceLROnPlateau 监控 `combined_auc`
- **早停机制**：监控 `combined_auc`，patience=10，min_delta=0.0001

---

## 四、三个脚本的一致性验证（与 test_on_CrossModality.py 对齐）

| 项目 | test.py | train_onGen_vessels.py | train_onReal.py | test_on_CrossModality.py |
|------|---------|------------------------|-----------------|--------------------------|
| 评估器 | UnifiedEvaluator | UnifiedEvaluator | UnifiedEvaluator | 独立实现 |
| RANSAC阈值 | 3.0 | 3.0 | 3.0 | - |
| AUC阈值 | [5,10,20] | [5,10,20] | [5,10,20] | [5,10,20] |
| mAUC上限 | 25 | 25 | 25 | 25 |
| **MSE计算** | **特征点坐标 MSE** | **特征点坐标 MSE** | **特征点坐标 MSE** | **特征点坐标 MSE** |
| **AUC依据** | **avg_dist** | **avg_dist** | **avg_dist** | **avg_dist** |
| **Failed判定** | **inliers < 1e-6** | **inliers < 1e-6** | **inliers < 1e-6** | **inliers < 1e-6** |
| MACE计算 | compute_corner_error | compute_corner_error | compute_corner_error | - |
| 失败样本处理 | error=1e6, MSE/MACE=inf | error=1e6, MSE/MACE=inf | error=1e6, MSE/MACE=inf | error=big_num |
| Inaccurate判定 | ✅ mae>50 or mee>20 | ✅ mae>50 or mee>20 | ✅ mae>50 or mee>20 | ✅ mae>50 or mee>20 |

**结论**：所有脚本的评估指标计算方式完全一致。

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

### 1. MSE 的两种定义
| 定义 | 计算方式 | 应用场景 |
|------|----------|----------|
| **图像像素 MSE** | 比较 warped 图像和目标图像的像素值 | 评估配准的视觉质量 |
| **特征点坐标 MSE** | 用 H 变换 pts0，与 pts1 计算 MSE | 评估匹配精度（与 test_on_CrossModality.py 对齐） |

**本项目使用特征点坐标 MSE**，与 `test_on_CrossModality.py` 的 `cal_MSE` 函数保持一致。

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
scripts/v1/test.py
├── UnifiedEvaluator.evaluate_batch()
│   ├── cv2.findHomography()  # RANSAC估计H
│   ├── cv2.perspectiveTransform()  # 计算 pts1_pred
│   ├── MSE = np.mean((pts1 - pts1_pred) ** 2)  # 特征点坐标MSE
│   ├── avg_dist = dis.mean()  # 用于AUC计算
│   ├── mae/mee 计算  # 用于判断inaccurate
│   └── compute_corner_error()  # MACE计算
└── UnifiedEvaluator.compute_epoch_metrics()
    ├── error_auc()           # AUC@5/10/20
    ├── compute_auc_rop()     # mAUC
    └── 聚合MSE/MACE/失败率/inaccurate/acceptable
```

---

## 八、常见问题

### Q1: 为什么 AUC 和 MSE/MACE 的样本数不一致？
**A**: AUC 包含所有样本（失败样本 error=1e6），MSE/MACE 仅包含匹配成功样本（过滤掉 inf）。

### Q2: 为什么使用 avg_dist 而非角点误差计算 AUC？
**A**: 与 `test_on_CrossModality.py` 保持一致。avg_dist 反映匹配点的重投影误差，更直接反映配准质量。

### Q3: Spatial Binning 会影响最终配准结果吗？
**A**: 不会。Spatial Binning 仅用于 RANSAC 估计 H，不影响匹配点的生成和最终的图像变换。

### Q4: 如何确保训练和测试指标一致？
**A**: 统一使用 `UnifiedEvaluator`，避免在不同脚本中重复实现指标计算逻辑。

---

**文档版本**: v1.1
**最后更新**: 2026-03-04
**维护者**: Fengjunming
