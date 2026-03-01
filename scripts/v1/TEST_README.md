# 测试脚本使用说明

## 概述

本目录包含两个测试脚本，用于评估训练好的 LightGlue 模型：

1. `test_onPureGen.py` - 测试在生成数据上训练的模型 (Gen mode)
2. `test_onReal.py` - 测试在真实数据上训练的模型 (CFFA mode)

## 功能特点

- **自动加载检查点**: 默认使用训练模型的 `best_checkpoint`
- **完整的前向推理**: 与训练时相同的推理流程
- **详细的可视化**: 保存匹配点、关键点、配准结果等
- **全面的指标计算**: MSE, MACE, AUC@5/10/20 等
- **灵活的输出管理**: 通过 `--test_name` 指定输出子目录

## 使用方法

### 1. test_onPureGen.py (生成数据训练模型测试)

```bash
# 基本用法 - 测试 best checkpoint
python scripts/v1/test_onPureGen.py --name 260228_1_v2_onGen_fixdataset

# 指定测试名称和数据集划分
python scripts/v1/test_onPureGen.py \
    --name 260228_1_v2_onGen_fixdataset \
    --test_name val_test_20260301 \
    --split val

# 使用自定义检查点
python scripts/v1/test_onPureGen.py \
    --name 260228_1_v2_onGen_fixdataset \
    --checkpoint results/lightglue_gen/260228_1_v2_onGen_fixdataset/latest_checkpoint/model.ckpt \
    --test_name latest_test

# 使用多个GPU
python scripts/v1/test_onPureGen.py \
    --name 260228_1_v2_onGen_fixdataset \
    --gpus "0,1"
```

### 2. test_onReal.py (真实数据训练模型测试)

```bash
# 基本用法 - 测试 best checkpoint
python scripts/v1/test_onReal.py --name 260228_1_v1_fixdataset

# 指定测试名称和数据集划分
python scripts/v1/test_onReal.py \
    --name 260228_1_v1_fixdataset \
    --test_name val_test_20260301 \
    --split val

# 使用自定义检查点
python scripts/v1/test_onReal.py \
    --name 260228_1_v1_fixdataset \
    --checkpoint results/lightglue_cffa/260228_1_v1_fixdataset/latest_checkpoint/model.ckpt \
    --test_name latest_test
```

## 参数说明

### 必需参数

- `--name` / `-n`: 训练模型名称，用于定位 checkpoint 目录

### 可选参数

- `--test_name`: 测试名称，指定输出子目录 (默认: `test_results`)
- `--checkpoint`: 检查点路径 (默认: 使用 `best_checkpoint/model.ckpt`)
- `--split`: 测试数据集划分 (默认: `val`, 可选: `train`, `val`, `test`)
- `--batch_size`: 批次大小 (默认: `4`)
- `--num_workers`: 数据加载线程数 (默认: `8`)
- `--img_size`: 图像大小 (默认: `512`)
- `--gpus`: GPU 设备 (默认: `0`, 多GPU示例: `"0,1"`)

## 输出结构

测试结果保存在以下目录结构中：

```
results/lightglue_{mode}/{name}/{test_name}/
├── test_log.txt                          # 测试日志
├── test_summary.txt                      # 测试指标总结
├── test_metrics.csv                      # 批次级指标
├── batch0000_sample00_xxx_vs_xxx/        # 每个样本的详细结果
│   ├── fix.png                           # 固定图 (参考图)
│   ├── moving_original.png               # 原始移动图 (未配准)
│   ├── moving_result.png                 # 配准结果
│   ├── moving_gt.png                     # Ground Truth
│   ├── fix_with_kpts.png                 # 带关键点的固定图
│   ├── moving_with_kpts.png              # 带关键点的移动图
│   ├── matches.png                       # 匹配可视化
│   ├── chessboard.png                    # 棋盘格对比图
│   └── metrics.txt                       # 单样本指标
└── ...
```

其中 `{mode}` 为:
- `gen` - 对应 test_onPureGen.py
- `cffa` - 对应 test_onReal.py

## 输出文件说明

### test_summary.txt
包含整体测试指标：
- Test Loss: 测试损失
- MSE: 均方误差
- MACE: 平均角点误差
- AUC@5/10/20: 不同阈值下的AUC
- Combined AUC: 平均AUC
- Inverse MACE: 归一化的MACE指标

### 单样本文件夹
每个测试样本都会保存完整的可视化结果：
- 原始图像和配准结果
- 关键点检测结果
- 匹配点可视化
- 棋盘格对比图
- 单样本指标

## 示例工作流

```bash
# 1. 测试生成数据训练的模型
python scripts/v1/test_onPureGen.py \
    --name 260228_1_v2_onGen_fixdataset \
    --test_name final_eval \
    --split val

# 2. 测试真实数据训练的模型
python scripts/v1/test_onReal.py \
    --name 260228_1_v1_fixdataset \
    --test_name final_eval \
    --split val

# 3. 查看结果
# - 生成数据模型: results/lightglue_gen/260228_1_v2_onGen_fixdataset/final_eval/
# - 真实数据模型: results/lightglue_cffa/260228_1_v1_fixdataset/final_eval/
```

## 注意事项

1. **Checkpoint 路径**: 确保指定的模型名称对应的 checkpoint 存在
2. **数据集要求**: 测试脚本会自动加载 CFFA 数据集的指定划分
3. **GPU 内存**: 根据可用显存调整 `--batch_size`
4. **输出目录**: 如果测试名称相同，会覆盖之前的结果

## 与训练脚本的对应关系

- `test_onPureGen.py` ↔ `train_onPureGen.py`, `train_onPureGen_v2.py`
  - 训练: 生成数据
  - 测试: CFFA 真实数据
  
- `test_onReal.py` ↔ `test_onReal.py`
  - 训练: 真实数据
  - 测试: CFFA 真实数据

## 技术细节

- **模型加载**: 使用 PyTorch Lightning 的 `load_from_checkpoint`
- **前向推理**: 与训练时完全相同的推理流程
- **指标计算**: 复用 `scripts/v1/metrics.py` 模块
- **防爆锁机制**: 自动检测和处理异常的单应矩阵
- **日志同步**: 测试日志同时输出到终端和文件
