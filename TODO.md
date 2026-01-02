 6 个TODO项目（都偏工程/实验规范 + 论文呈现），要求“可复现、可解释、可发表”。

清单表（从最值得做 → 可选加分），每项都写清楚**目的 / 具体动作 / 产出物**

## 精修待办清单（建议表）

| 优先级 | 事项                                           | 目的                                | 你需要做的最小动作                                           | 产出物                                          |
| ------ | ---------------------------------------------- | ----------------------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| P0     | 固定随机种子（Phase1/2/3 + sweep）             | 结果可复现，避免 reviewer 质疑      | 在每个脚本 `main()` 开头加 `set_seed(seed)`；同时设置 `cudnn.deterministic` | 训练日志稳定；CSV 中记录 seed                   |
| P0     | 统一实验记录与命名                             | 避免覆盖 ckpt；便于追溯每条结果     | ckpt 名包含 model/lead/mask/seed/lam；输出目录结构一致       | `experiments/` 内可追溯权重；results CSV 可追溯 |
| P0     | 指标体系补齐：monthly + spring_3_6（你已完成） | 支撑 SPB 主线论证                   | 对 Phase1/Phase2/Phase3 都输出 monthly CSV（append）         | `monthly_test_lead1.csv`（含所有模型）          |
| P1     | 配置文件（configs）+ 单一入口脚本              | 扫参/复现实验更方便；论文方法更规范 | 用 JSON/YAML 定义：data、model、train、loss、eval；脚本读取 config | `configs/*.yaml` + `scripts/run.py`             |
| P1     | 多 seed 稳健性（至少 3 seeds）                 | 给出均值±方差，更“论文级”           | 对最终模型（以及 Phase1 baseline）跑 seed=0/1/2；记录 mean±std | 表格：overall 与 spring 的 mean±std             |
| P1     | 可视化验证（不是“好看”，是“证据”）             | 证明物理一致性改善冰缘/融化区       | 固定 2–3 个春季样本，画 GT/Pred/Diff；或画 RMSE month curve  | 1 张月度曲线 + 1–2 张空间图                     |
| P2     | Loss/正则项尺度检查与消融表                    | 解释 Phase3 为什么有时变差          | 记录 l_pred/l_tv/l_time 的均值；加入消融：λ=0、只TV、只time  | 1 个 ablation 表 + 一段解释                     |
| P2     | 数据与预处理声明（mask、归一化、缺测）         | 避免 reviewer “数据细节”攻击点      | README/Method 加：mask 构建、值域、nan 处理、split 年份      | Methods/Appendix 一小节                         |

------

## 你现在“已经有初步数据”，下一步最建议先做什么？

我建议按这个顺序（最省力、收益最大）：

1. **P0：固定 seed + 命名规范**（立刻做，不改变模型，只改变可复现性）
2. **P0：把 Phase1/2/3 的 monthly 都 append 到同一个 CSV**
    - 这样你立刻能回答：SPB（3–6 月）到底改善了吗？
3. **P1：最终模型跑 3 个 seed**（只对“你准备写进论文的最终模型”做，不用全家桶都跑）
4. **P1：画一张 Month-RMSE 曲线**（强烈建议，写论文最有用）

## 基于你现有的结果，我建议你论文里至少需要两张表：

### 表 A：Overall（全局）对比（你已有）

包含：

- Persistence / Climatology
- ConvLSTM Phase1
- Phase2
- Phase3（最终）

### 表 B：Spring (3–6) 指标对比（核心表）

从 `monthly_test_lead1.csv` 里筛 `month="spring_3_6"`，同样列出上面模型。

> 有了表 B，你的 SPB 叙事就“落地”了。