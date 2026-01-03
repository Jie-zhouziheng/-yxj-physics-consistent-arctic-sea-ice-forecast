# Core source code structure

本目录包含项目的工程化核心实现，所有模型训练与实验均应依赖此处代码，
而非 notebooks 中的临时代码。

## Directory structure

```text
src/
├── datasets/   # 数据读取、样本构造、滑动窗口、lead-time 与 mask/split 定义
├── models/     # baseline 模型与物理一致性模型
├── losses/     # 物理约束损失函数（空间平滑、时间一致性等）
├── utils/      # 通用工具函数（reproducibility, evaluation helpers, logging）
└── train.py    # 训练入口原型（部分实验当前通过 scripts/ 中的脚本运行）
```