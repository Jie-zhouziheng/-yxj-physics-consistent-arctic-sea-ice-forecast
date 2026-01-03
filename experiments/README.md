# Experiments Directory

本目录用于存放 **模型训练过程中产生的实验产物**，
包括 **模型权重（checkpoints）** 以及与之对应的实验配置标识。

这里的内容是论文实验结果的**直接证据（evidence）**，
所有模型性能指标（CSV）均由这些权重在 `scripts/` 中的评估脚本生成。

------

## 目录定位与原则

**experiments/** 目录遵循以下原则：

1. **只存放“可复现实验产物”**
    - 模型权重（`.pt`）
    - 与训练强相关、但不适合放在 `results/` 的中间文件
2. **不存放原始数据或评估 CSV**
    - 指标结果统一放在 `results/`（或当前的 `scripts/results/`）
3. **文件命名必须自描述**
    - 单看文件名即可判断：模型类型 / lead-time / seed / 关键超参

------

## 当前实验阶段说明

截至当前版本，实验主要分为三个阶段（Phase）：

### Phase 1 — ConvLSTM baseline

- 目的：建立轻量级深度学习基线
- 特点：
    - 仅使用 SIC
    - 单一 ConvLSTM
    - 无显式物理约束
- 对应脚本：
    - `scripts/10_train_convlstm_phase1.py`

### Phase 2 — Season-aware temporal modeling

- 目的：针对春季转折阶段（SPB）引入季节感知结构
- 特点：
    - 冬季：时间聚合
    - 春季：轻量级时序建模
- 对应脚本：
    - `scripts/11_train_phase2.py`

### Phase 3 — Season-aware + physics-consistent model

- 目的：在 Phase 2 基础上引入物理一致性约束
- 特点：
    - 空间平滑（TV-like regularization）
    - 时间平滑（single-step temporal consistency）
- 对应脚本：
    - `scripts/12_train_phase3.py`
    - 超参数扫描：`scripts/13_sweep_phase3.py`

------

## Checkpoint 命名规范（强烈建议遵循）

模型权重文件（`.pt`）建议采用如下命名格式：

```text
{model}_lead{L}_seed{S}_tv{λ_tv}_time{λ_time}.pt
```

示例：

```text
phase3_lead1_seed0_tv0.005_time0.0005.pt
phase3_lead1_seed1_tv0.005_time0.0005.pt
```

其中：

- `model`：模型阶段（`convlstm_phase1 / phase2 / phase3`）
- `L`：预测提前量（lead time）
- `S`：随机种子（seed）
- `λ_tv`：空间平滑正则权重
- `λ_time`：时间一致性正则权重

> 该命名方式用于支持 **multi-seed / multi-lead / sweep 实验**，
> 并避免权重被意外覆盖。

------

## 与结果文件（CSV）的对应关系

- **模型权重**：`experiments/*.pt`
- **评估指标**：`results/*.csv`（或当前 `scripts/results/`）

对应关系如下：

| 权重文件                  | 对应结果                  |
| ------------------------- | ------------------------- |
| `phase3_lead1_seed0_*.pt` | `models_test_lead1.csv`   |
| 同一模型多 seed           | CSV 中多行（seed 列区分） |
| sweep 实验权重            | `phase3_sweep.csv`        |
| 每个权重的月度结果        | `monthly_test_lead*.csv`  |

------

## 可复现实验流程（推荐）

1. 使用 `scripts/` 中的训练脚本生成 checkpoint（保存到本目录）
2. 使用统一评估函数在 **test split** 上计算指标
3. 将结果 append 写入 CSV（不覆盖历史记录）
4. 论文表格由 CSV 聚合（mean ± std）生成

------

## 注意事项

- 本目录不保证向后兼容旧实验
- 当模型结构或损失函数发生改变时，**必须重新训练并生成新权重**
- 请勿手动修改 `.pt` 文件名中的关键信息（seed / lead / λ）

------

## 后续规划

- 按 lead-time（1–6）分子目录组织实验
- 支持多源物理变量（SIT / SAT）的独立实验组
- 与 `configs/` 目录深度绑定，实现完全配置驱动的实验管理

