# Plan: LightGlue with Vessel-Guided Curriculum Learning
我现在在做一个课题，目标是目标是基于我们已有的生成数据集（结构完全相同且对齐的cf-oct-fa图像对）训练出一个支持cf-fa，cf-oct（cf均为fix）的多模态配准模型。我们目前在用fractMorph模型进行2D化改造后的模型，但是fractMorph2D只能针对一些微小血管形变进行对齐和修复，但是针对真实情况不对齐图像中出现的血管大尺度旋转（如15 30度的旋转角度）、大的位置偏移的情况下配准效果较差，1是原先的逐像素点移动的模式很差2是我们修改成让模型直接预测放射参数，效果也很差，几乎不收敛。我现在的想法是改造Lightglue，学习cf图像和oct/fa图像之间的关键点匹配，然后利用这些匹配出来的关键点计算放射矩阵，对跨模态图像进行配准。
多模态数据集data\FIVES_extract_v2\FIVES_extract_v2.py
真实数据集：
data\operation_pre_filtered_cffa\operation_pre_filtered_cffa_dataset.py
data\operation_pre_filtered_cfoct\operation_pre_filtered_cfoct_dataset.py
data\operation_pre_filtered_octfa\operation_pre_filtered_octfa_dataset.py



## 1. 核心问题
现有的LightGlue/LoFTR模型在Sim-to-Real (CF vs OCT/FA) 跨模态配准任务中面临 **Mode Collapse**。
*   **现象**: 所有匹配点都集中在图像顶部或特定区域，完全忽略血管结构。
*   **原因**: 模态差异巨大，背景纹理不一致，模型无法找到共性特征，只能寻找局部极值。
*   **机会**: 血管结构是唯一的跨模态共性特征。我们拥有生成数据的血管Mask (GT)。

## 2. 解决方案: 血管引导的课程学习 (Vessel-Guided Curriculum Learning)
将LoFTR的Attention Bias思想迁移到LightGlue的稀疏匹配框架中。通过 **Weighted NLL Loss** 强制模型关注血管上的关键点。

### 核心机制
利用 `vessel_mask0` (Fix图的血管掩码) 对 Loss 进行加权。
*   **In-Vessel Points**: 落在血管上的关键点，匹配错误的惩罚极高 (High Weight)。
*   **Background Points**: 背景区域的关键点，维持正常权重 (Weight = 1.0)。

### 三阶段课程 (Curriculum Schedule)
通过 `CurriculumScheduler` Callback 动态调整权重参数 `vessel_weight`。

#### Phase 1: 教学期 (Teaching Phase) [Epoch 0 - 20]
*   **策略**: 强迫关注血管。
*   `vessel_weight = 10.0`
*   **目标**: 让模型迅速学会 "血管特征才是匹配的关键"，忽略干扰性极强的背景噪声。

#### Phase 2: 断奶期 (Weaning Phase) [Epoch 20 - 50]
*   **策略**: 线性衰减权重。
*   `vessel_weight`: 线性从 10.0 降至 1.0。
*   **目标**: 随着模型特征提取能力的增强，逐渐减少人为干预，防止过拟合到Mask本身。

#### Phase 3: 独立期 (Independence Phase) [Epoch 50+]
*   **策略**: 自由探索。
*   `vessel_weight = 1.0` (即关闭特殊加权)。
*   **目标**: 此时模型应已具备良好的初始化，允许其利用背景上下文或其他微细纹理进行精细化匹配。

## 3. 实现细节

### A. 模型修改 (`pl_lightglue.py`)
在 `_compute_loss` 中增加 Weighted Loss 逻辑:
1.  接收 `batch['vessel_mask0']`。
2.  将 `kpts0` 坐标映射到 Mask 上进行采样，判断每个点是否在血管内 (`is_vessel`)。
3.  计算权重向量 `weights = is_vessel * (vessel_weight - 1.0) + 1.0`。
4.  将权重应用到 NLL Loss。

### B. 训练脚本 (`train_lightglue_onGen_v2.py`)
1.  实现 `CurriculumScheduler` 回调函数。
2.  在每个 Epoch 开始时更新 `pl_module.vessel_weight`。
3.  确保 DataModule 传递 `vessel_mask0`。

### C. 兼容性
*   若 Batch 中无 Mask (如真实数据)，则默认 `vessel_weight=1.0`，逻辑退化为原始 LightGlue。
*   SuperPoint 提取器本身冻结，假设其能在血管处提取到角点 (通常边缘处角点响应很强)。
