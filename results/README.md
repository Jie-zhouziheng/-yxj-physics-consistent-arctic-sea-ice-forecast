# Results Directory

本目录用于存放 **所有实验的评估结果（evaluation outputs）**，
以 **CSV 表格形式** 记录模型在 test 集（以及必要时 val 集）上的性能指标。

这些文件是论文中 **所有定量结果（tables / figures）** 的**唯一来源**。

> ⚠️ 本目录 **不包含模型权重**（请见 `experiments/`），
> 仅包含由权重推理后得到的 **指标统计结果**。

------

## 目录定位原则

1. **结果只 append，不覆盖**
    - 每次实验都会向 CSV 追加新行
    - 便于回溯历史实验与对比不同模型/超参
2. **CSV 是论文级别的“事实记录”**
    - 论文表格应由 CSV 直接聚合生成
    - 不手工“挑结果”
3. **所有结果都可由脚本复现**
    - 对应训练 / 评估脚本位于 `scripts/`

------

## 当前结果文件说明

### 1️⃣ `models_test_lead1.csv`

**用途**
记录 **单模型、单次运行** 在 test 集上的 **全局指标**。

**典型字段**

| 列名           | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| `model`        | 模型名称（如 `convlstm_phase1` / `phase3_season_aware_phys`） |
| `split`        | 数据划分（通常为 `test`）                                    |
| `input_window` | 输入时间窗口长度（如 12 个月）                               |
| `lead_time`    | 预测提前量（1–6 个月）                                       |
| `mask`         | 评估掩膜（如 `ice15`）                                       |
| `mae`          | 掩膜区域平均 MAE                                             |
| `rmse`         | 掩膜区域平均 RMSE                                            |

**说明**

- 每一行对应 **一个模型配置 + 一个 seed**
- multi-seed 实验会产生多行，后续可计算 mean ± std
- 这是论文 **主表（overall performance）** 的直接来源

------

### 2️⃣ `monthly_test_lead1.csv`

**用途**
记录模型在 **test 集上按月份划分的误差指标**，
用于分析 **季节依赖性误差结构**，特别是：

> **春季可预报性屏障（Spring Predictability Barrier, SPB）**

**典型字段**

| 列名           | 含义                       |
| -------------- | -------------------------- |
| `model`        | 模型或 sweep tag           |
| `split`        | 数据划分（test）           |
| `input_window` | 输入窗口                   |
| `lead_time`    | 预测提前量                 |
| `mask`         | 掩膜类型                   |
| `month`        | 月份（1–12）或聚合标签     |
| `n_samples`    | 该月份在 test 集中的样本数 |
| `mae`          | 该月份平均 MAE             |
| `rmse`         | 该月份平均 RMSE            |

**特殊行说明**

- `month = 0`
    → 全年 test 集整体指标（与 `models_test_lead1.csv` 数值一致）
- `month = spring_3_6`
    → 春季（3–6 月）聚合结果，用于直接量化 SPB 改善程度

**说明**

- 每个模型会写入 **13 + 1 行**：
    - 12 个月
    - 1 个全年
    - 1 个 spring 聚合
- 多个模型 / sweep 实验通过 `model` 字段区分

------

### 3️⃣ `phase3_sweep.csv`

**用途**
记录 Phase 3（物理一致性模型）的 **超参数扫描结果**。

**典型字段**

| 列名              | 含义               |
| ----------------- | ------------------ |
| `model`           | 模型类型           |
| `epochs`          | 训练轮数           |
| `batch_size`      | batch size         |
| `lr`              | 学习率             |
| `embed_channels`  | 编码器通道数       |
| `hidden_channels` | 隐状态通道数       |
| `lam_tv`          | 空间平滑正则权重   |
| `lam_time`        | 时间一致性正则权重 |
| `best_val_rmse`   | 验证集最优 RMSE    |
| `test_mae`        | test MAE           |
| `test_rmse`       | test RMSE          |
| `ckpt_path`       | 对应模型权重路径   |

**说明**

- 每一行对应 **一次完整训练 + 最优模型评估**
- 与 `monthly_test_lead*.csv` 中的 `model=tag` 一一对应
- 用于论文中的：
    - 超参敏感性分析
    - 正则项有效性讨论

------

## 与论文结构的对应关系

| 论文部分                        | 对应 CSV                              |
| ------------------------------- | ------------------------------------- |
| Overall performance table       | `models_test_lead*.csv`               |
| Seasonal / monthly analysis     | `monthly_test_lead*.csv`              |
| Ablation / regularization study | `phase3_sweep.csv`                    |
| Spring barrier discussion       | `monthly_test_lead*.csv (spring_3_6)` |

------

## 结果复现说明

所有 CSV 均由以下脚本生成（append 模式）：

| 脚本                          | 作用                      |
| ----------------------------- | ------------------------- |
| `10_train_convlstm_phase1.py` | Phase 1 baseline          |
| `11_train_phase2.py`          | Season-aware model        |
| `12_train_phase3.py`          | Physics-consistent model  |
| `13_sweep_phase3.py`          | 超参数扫描                |
| `eval_utils.py`               | monthly / spring 评估函数 |

------

## 后续规划（建议）

- 按 lead-time 拆分结果文件：
    - `models_test_lead{1..6}.csv`
    - `monthly_test_lead{1..6}.csv`
- 增加 `seed` 列，支持自动统计 mean ± std
- 将 `scripts/results/` 迁移为顶层 `results/` 目录