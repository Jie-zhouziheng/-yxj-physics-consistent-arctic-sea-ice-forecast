**Project Handover Prompt — Physics-Consistent Arctic Sea Ice Forecasting**

------

## 一、项目背景（请务必先读）

你正在接手一个**可复现、轻量级、物理一致性的北极海冰浓度（SIC）季节预测研究项目**，目标是完成一篇**可发表论文（lead time = 1–6 months）**。

### 核心约束与设计理念

- **算力受限**（单 GPU / 小 batch）
- **工程优先（engineering-first）**
- **强可复现性**（固定随机种子、append-only 结果）
- **物理一致性 > 复杂网络**
- **模块化**：新增功能不能破坏已有实验结果

------

## 二、当前仓库状态（非常重要）

### 仓库路径

```
physics-consistent-arctic-sea-ice-forecast/
```

### 关键目录

```
src/
  datasets/
    SICWindowDataset.py     # SIC 主数据集（不可随意修改）
    sat_scalar.py           # ✅ 新增：SAT anomaly → scalar 查表
  models/
    convlstm_baseline.py
    season_aware_phase2.py
    season_aware_phase3_sat.py  # ✅ 新增：SAT 调制版 Phase3
  losses/
    physics.py              # TV / temporal smooth
  utils/
    repro.py                # seed_everything / generator
    config.py               # YAML config loader（已存在）

scripts/
  10_train_convlstm_phase1.py   # ✅ 已固定随机种子
  11_train_phase2.py            # ✅ 已固定随机种子
  12_train_phase3.py            # ✅ physics-only phase3（已 config 化）
  14_train_phase3_sat.py        # ✅ 新增：SAT-conditioned phase3
  eval_utils.py                 # monthly / spring eval（append）

configs/
  phase3_phys.yaml              # Phase3 (physics-only)
  （SAT 的 config 尚未拆出）

data/raw/
  nsidc_sic/                     # SIC 月平均数据
  era5_sat/
    era5_t2m_monthly_1979_2022_arctic.nc
    era5_sat_anom_monthly_1979_2022_arctic.nc  # ✅ 已修复并生成

experiments/
  phase3_sat/
    phase3_sat_leadX_seedY.pt

scripts/results/
  models_test_lead{L}.csv
  monthly_test_lead{L}.csv
```

------

## 三、已经完成的关键工作（请不要重复）

### ✅ 1. 随机种子与可复现性

- 所有 Phase1 / Phase2 / Phase3 / sweep / SAT 版本：
    - `seed_everything(seed, deterministic=True)`
    - DataLoader 使用 `generator=g`
- 实验结果**可稳定复现**

------

### ✅ 2. Phase1–Phase3 基线已完成（lead=1）

- Phase1：ConvLSTM baseline
- Phase2：Season-aware（冬季压缩 + 春季重点建模）
- Phase3：Season-aware + Physics（TV + temporal smooth）
- 统一输出：
    - overall metrics
    - monthly metrics
    - spring_3_6 metrics

------

### ✅ 3. 多提前量实验（lead = 1–6）框架已就绪

- Dataset 已原生支持 `lead_time`
- 训练脚本支持 `--lead`
- 输出文件按 `lead` 分开 append（不覆盖）

------

### ✅ 4. ERA5 SAT anomaly 数据已正确处理

SAT anomaly 文件内容 **已验证正确**：

```
Dimensions: (time=528, latitude=121, longitude=1440)
Variable: sat_anom (time, lat, lon)
Time: 1979-01 → 2022-12（月平均）
```

------

### ✅ 5. 新增：SAT-conditioned Phase3（不破坏旧代码）

#### 设计选择（请保持一致）

- **SAT 形式**：每个样本一个 scalar（lat/lon 加权平均）

- **时间对齐**：使用 `t_out` 对应月份的 SAT anomaly

- **作用位置**：

    - 不改输入
    - 不改 loss
    - **只调制 season-aware 的融合权重 `alpha`**

- **调制方式**（稳定、可解释）：

    ```
    alpha = alpha_season + sat_k * tanh(sat / sat_scale)
    ```

#### 新增文件

- `src/datasets/sat_scalar.py`
- `src/models/season_aware_phase3_sat.py`
- `scripts/14_train_phase3_sat.py`

👉 **这是论文“多源观测 / 再分析数据接入”的核心技术贡献之一**

------

## 四、严格遵守的工程原则（⚠️ 不可破坏）

1. ❌ 不要修改旧实验脚本来“顺便支持新功能”
2. ❌ 不要 overwrite CSV / ckpt
3. ✅ 所有结果必须 append
4. ✅ 新想法 = 新脚本 / 新模型
5. ✅ 任何新增模块都要能 **单独跑通**
6. ✅ 任何复杂化设计都必须有“论文解释价值”

------

## 五、当前最重要的待办任务（按优先级）

### 🔴 P0（必须完成，论文主线）

#### 1️⃣ 完整跑 **lead = 1–6**

- 对以下模型：
    - Phase1 baseline
    - Phase2 season-aware
    - Phase3 physics-only
    - Phase3 + SAT
- 至少 **seed = 0 / 1 / 2**
- 汇总：
    - overall mean ± std
    - spring_3_6 mean ± std

------

#### 2️⃣ 把 SAT 模型结果正式写进论文

- 在 **Methodology** 中说明：
    - SAT anomaly 构建方式
    - 为什么用 scalar（稳定 / 低算力 / 可解释）
- 在 **Experiments / Discussion** 中：
    - 对比 Phase3 vs Phase3+SAT
    - 特别强调 **春季（MAMJ）改进**

------

### 🟠 P1（强烈建议）

#### 3️⃣ 将 SAT 超参拆到 YAML config

- `sat_k`
- `sat_scale`
- `alpha_spring`
- `alpha_other`
- 便于 sweep + ablation

------

#### 4️⃣ 可视化（论文证据）

- 月度 RMSE 曲线（lead=1–6）
- 春季典型样本：
    - GT / Phase3 / Phase3+SAT / diff

------

### 🟡 P2（加分项）

#### 5️⃣ 多源扩展预留

- SIT（PIOMAS）：
    - 同样走 scalar 或 very-low-res 路线
- 保持与 SAT 完全一致的接口设计

------

## 六、你现在可以如何继续（建议路径）

> 如果你是新的 AI，请按这个顺序行动：

1. **不要改任何已存在代码**
2. 跑 `scripts/14_train_phase3_sat.py`，完成 lead=1–6
3. 汇总 CSV → 生成论文表格
4. 只在确认结果稳定后，才考虑更复杂的多源设计

------

## 七、项目目标（最终）

> 一篇 reviewer 看了会说：

- “工程设计很干净”
- “物理约束合理”
- “多源数据接入不过拟合、很克制”
- “在算力受限条件下仍有科学价值”