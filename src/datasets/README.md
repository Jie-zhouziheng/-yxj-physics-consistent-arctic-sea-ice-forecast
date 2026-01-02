# datasets 目录是干什么的？

一句话：这里负责把 raw 的 NetCDF（月文件）变成**训练 / 评估可直接使用的样本**。

本项目中，所有 baseline 和模型都 **必须** 通过本目录提供的接口访问数据，
不允许直接读取 raw NetCDF 文件。

---

## 数据处理流程概览

datasets 层可以理解为两步：

```
raw NetCDF（月文件）
↓
统一读取为单月 SIC 数组
↓
滑动窗口切分为训练 / 评估样本

```

---

## 第 1 步：读一个月（`sic_reader.py`）

每个 `.nc` 文件里的变量名可能不同，例如：

- 1979 年：`N07_ICECON`
- 2000 年：`F13_ICECON`

但它们在物理意义上都是 **海冰浓度（SIC）**。

` sic_reader.py ` 做的事情是：

- 自动识别唯一的 `*_ICECON` 变量
- 去掉 `time=1` 这一维
- 返回一个统一格式的二维数组

**输出约定（固定）：**

- 单月 SIC：`(y, x)`  
- 在 NSIDC-0051 v2 中：`(448, 304)`
- 数值范围：`[0, 1]`

---

## 第 2 步：切成训练样本（`sic_dataset.py`）

模型不是一次只看一个月，而是使用一段历史进行预测。

我们用两个参数来定义样本构造方式：

- `input_window`：输入使用的历史月份数
- `lead_time`：预测提前量（月）

### 示例

当：

- `input_window = 3`
- `lead_time = 1`

则一个样本为：

- 输入 `X`：`[t-2, t-1, t]` 三个月的 SIC
- 输出 `Y`：`t+1` 月的 SIC

`SICWindowDataset` 返回：

- `X`：`(input_window, y, x)`
- `Y`：`(y, x)`
- `meta`：包含该样本对应的时间信息（输入末月、预测目标月）

---

## 固定的数据划分协议（非常重要）

本项目使用 **固定的时间划分**，对所有 baseline 和模型保持一致：

```text
Train : 1979–2010
Val   : 2011–2016
Test  : 2017–2022
```

该划分通过 `split_index_by_year` 实现，任何实验都不应自行修改年份规则。

------

## 标准使用方式（推荐）

```python
from pathlib import Path
from src.datasets import (
    build_index,
    SICWindowDataset,
    split_index_by_year,
)

# 1) 构建全量时间索引
index_all = build_index(Path("data/raw/nsidc_sic"), hemisphere="N")

# 2) 按固定规则切分
index_train = split_index_by_year(index_all, split="train")
index_val   = split_index_by_year(index_all, split="val")
index_test  = split_index_by_year(index_all, split="test")

# 3) 构造 Dataset
ds_train = SICWindowDataset(index_train, input_window=12, lead_time=1)
ds_val   = SICWindowDataset(index_val,   input_window=12, lead_time=1)
ds_test  = SICWindowDataset(index_test,  input_window=12, lead_time=1)

```

## 设计约定

- 所有实验必须使用相同的 Dataset 构造逻辑
- baseline 与深度学习模型共享同一数据接口
- notebooks 和 scripts 只负责调用，不承载核心数据逻辑

## 有效冰区掩膜（Effective Ice Mask）

为避免陆地和常年无冰海域对评估指标的稀释，
本项目在评估阶段使用固定的有效冰区掩膜。

掩膜定义为：

- 在训练期（1979–2010）内
- 任一月份海冰浓度 ≥ 0.15 的格点

该掩膜在所有 baseline 与模型评估中保持一致，
仅用于指标计算，不影响数据读取与模型输入。


