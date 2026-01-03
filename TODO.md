TODO项目（都偏工程/实验规范 + 论文呈现），要求“可复现、可解释、可发表”。 

清单表（从最值得做 → 可选加分），每项都写清楚**目的 / 具体动作 / 产出物**

| 优先级 | 事项                                           | 目的                                         | 你需要做的最小动作                                           | 产出物                                    |
| ------ | ---------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| **P0** | 固定随机种子（Phase1/2/3 + sweep）（✔ 已完成） | 保证结果可复现，避免 reviewer 质疑           | 在每个脚本 `main()` 开头加入 `set_seed(seed)`；设置 `torch.backends.cudnn.deterministic=True` | 训练日志稳定；CSV 中记录 `seed`           |
| **P0** | 统一实验记录与命名规范（✔ 已完成）             | 避免 ckpt 覆盖；便于回溯每条结果             | ckpt 名包含 `model/lead/mask/seed/lam_tv/lam_time`；统一目录结构 | `experiments/` 可追溯；results CSV 可审计 |
| **P0** | 指标体系补齐：monthly + spring_3_6（✔ 已完成） | 明确服务 SPB（春季可预报性屏障）主线         | 对 Phase1/Phase2/Phase3 全部输出 monthly CSV（append 模式）  | `monthly_test_lead*.csv`                  |
| **P0** | **多提前量实验：lead = 1–6（✔ 已完成）**       | 回到论文核心问题（季节尺度预测）             | 复用现有 pipeline，将 `lead_time=1` 扩展为 `{1,…,6}`；先 Phase1→Phase3 最终模型 | 主表：lead–RMSE 曲线；spring 子表         |
| **P1** | 配置文件（configs）+ 单一入口脚本（✔ 已完成）  | 扫参与复现实验更规范；方法描述更清晰         | 用 YAML/JSON 定义 data/model/train/loss/eval；`scripts/run.py --config` | `configs/*.yaml` + 统一入口               |
| **P1** | **多源观测 / 再分析数据接入**                  | 提升物理信息量，支撑“physics-consistent”叙事 | 逐步加入 SIT proxy、SAT anomaly（先对齐到 SIC 网格与时间）；作为附加通道 | 新数据 loader + ablation 表               |
| **P1** | 多 seed 稳健性（≥3 seeds）                     | 给出 mean ± std，更“论文级”                  | 对最终模型（及 Phase1 baseline）跑 seed=0/1/2                | overall & spring 的 mean±std 表           |
| **P1** | 可视化验证（不是“好看”，是“证据”）             | 证明模型在春季/冰缘区的改进                  | 固定 2–3 个春季样本，画 GT / Pred / Diff；或画 month-wise RMSE 曲线 | 1 张月度曲线 + 1–2 张空间图               |
| **P2** | Loss / 正则项尺度检查与消融                    | 解释 Phase3 有时变差的原因                   | 记录 `l_pred / l_tv / l_time` 的量级；对 λ=0 / 只TV / 只Time 做消融 | 1 个 ablation 表 + 解释段落               |
| **P2** | 数据与预处理声明（mask / 归一化 / 缺测）       | 避免 reviewer 从“数据细节”攻击               | 在 Methods / Appendix 写清：mask 构建、值域、nan 处理、年份划分 | Methods 或 Appendix 一小节                |