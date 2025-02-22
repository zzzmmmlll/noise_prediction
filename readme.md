## Python 版本要求

此项目需要 Python 3.10 及以上版本。可以通过以下命令检查当前安装的 Python 版本：

```bash
python --version
```

然后，您可以使用 `pip` 安装所需的所有依赖项：

1. 克隆此项目：

```bash
git clone https://github.com/zzzmmmlll/noise_prediction.git
cd noise_prediction
```
2. 安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

# CSV 文件处理与模型训练使用文档

该工程包含两个脚本：`main.py` 和 `model.py`。`main.py` 负责处理 CSV 文件，训练模型，并生成预测结果；`model.py` 包含实际的模型训练和预测逻辑。

## 功能概述

1. **遍历 CSV 文件**：脚本会递归遍历 `data` 文件夹中的所有 CSV 文件。
2. **模型训练与预测**：每个 CSV 文件会被用来训练模型，并生成预测结果。
3. **结果保存**：每个 CSV 文件的结果（包括预测结果、训练模型和图像）会保存到 `result` 文件夹中，结构保持与 `data` 文件夹相同。

## 目录结构

```python
your-project/
├── data/                             # 原始 CSV 数据文件夹
│   ├── q0/                           # 一个子文件夹，存放相关 CSV 文件
│   │   ├── gate_error_gate_sx0_q0.csv
│   │   └── another_data.csv
│   ├── q1/
│   │   └── some_data.csv
│   └── raw_data.csv
├── model.py                          # 模型训练和预测逻辑脚本
├── program/
│   └── main.py                       # 主程序，用于处理 CSV 文件并生成结果
├── result/                           # 存储结果的文件夹
│   ├── q0/
│   │   ├── gate_error_gate_sx0_q0/   # 该 CSV 对应的结果文件夹
│   │   │   ├── predictions.csv
│   │   │   ├── result_plot.png
│   │   └── another_data/
│   │       ├── predictions.csv
│   │       ├── result_plot.png
│   ├── q1/
│   │   └── some_data/
│   │       ├── predictions.csv
│   │       ├── result_plot.png
│   └── ...
└── README.md                         # 项目说明文件
```


## 使用方法

### 步骤 1：准备数据

将所有的 CSV 文件放入 `data` 文件夹中，支持嵌套文件夹。

### 步骤 2：运行主脚本

执行 `main.py` 来开始处理数据：

```bash
python main.py
```

## 结果文件夹（`result`）

`result` 文件夹用于存放模型训练和预测的结果。该文件夹包含根据输入的 CSV 数据生成的预测结果文件、图像以及模型的检查点（checkpoint）。每个数据文件会生成一个独立的文件夹，文件夹名称与数据文件的相对路径保持一致。文件夹内的内容包括：

1. **`predictions.csv`**：
   - 该文件包含预测结果，包括每个数据点的真实值、预测值以及对应的误差。
   - 每个数据文件会生成一个独立的 `predictions.csv` 文件。

2. **`result_plot.png`**：
   - 该图像文件展示了真实值和预测值的比较。通常用于评估模型的性能，帮助用户直观地了解模型的预测能力。
   - 图像内容包括真实值与预测值的折线图。

3. **`checkpoints/` 文件夹**：
   - 用于存放模型训练过程中的中间结果（checkpoint）。在训练过程中，如果模型达到某个连续训练次数限制，当前模型会保存为一个 checkpoint 文件，以便稍后使用或恢复。
   - `checkpoints/` 文件夹包含了所有训练过程中的模型文件。

### `result` 文件夹结构示例：

```text
result/
│
├── q0/                           # 文件夹名称与原数据文件的相对路径相对应
│   ├── gate_error_gate_sx0_q0/    # 该数据的预测结果和模型保存目录
│   │   ├── predictions.csv        # 该数据的预测结果文件
│   │   ├── result_plot.png        # 预测结果的可视化图像
│   │   └── checkpoints/           # 存储模型训练过程中的 checkpoint 文件
│   │       └── checkpoint_model_1.pkl
│   │       └── checkpoint_model_2.pkl
│   └── another_data/              # 如果有多个数据源，类似的目录结构也会在 result 下创建
│       ├── predictions.csv
│       ├── result_plot.png
│       └── checkpoints/
│           └── checkpoint_model_1.pkl
