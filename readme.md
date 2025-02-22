## Python 版本要求

此项目需要 Python 3.10 及以上版本。可以通过以下命令检查当前安装的 Python 版本：

```bash
python --version
```

然后，您可以使用 `pip` 安装所需的所有依赖项：

1. 克隆此项目：

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
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
│   │   │   └── trained_model.pkl
│   │   └── another_data/
│   │       ├── predictions.csv
│   │       ├── result_plot.png
│   │       └── trained_model.pkl
│   ├── q1/
│   │   └── some_data/
│   │       ├── predictions.csv
│   │       ├── result_plot.png
│   │       └── trained_model.pkl
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
