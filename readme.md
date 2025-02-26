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
```

# 训练过程思路

该模型的训练过程旨在实现 **增量学习** 和 **时间序列预测**。我们基于 **SGDRegressor**（随机梯度下降回归）模型，通过增量训练和误差反馈机制，不断调整和优化模型的预测结果。

以下是详细的训练过程思路：

## 1. 数据预处理

### 1.1 数据读取与初步处理
首先，程序从指定的 CSV 文件中读取数据。该文件包含两列：
- `query_time`：时间戳，表示数据采集的时间。
- `value`：该时间点的目标值，即模型需要预测的数值。

**数据初步处理**：
- `query_time` 列被转换为标准的 **datetime** 类型，确保其格式符合时间戳要求，方便后续操作。
- 如果 `value` 列存在缺失值（NaN），则会删除这些包含缺失值的行。

### 1.2 处理 `value = 1` 的数据
在数据处理过程中，有些 `value` 为 1 的数据可能并不是我们需要的有效数据。因此，我们需要对这些 `value` 为 1 的数据进行处理。我们用 **最接近的非 1 数据值** 替代这些值，确保数据的连贯性和准确性。

### 1.3 数据归一化
由于不同的数据特征可能具有不同的量纲或范围，因此需要对数据进行归一化处理，确保所有输入数据在训练时的尺度一致。归一化的方式是将数据转化为均值为 0，标准差为 1 的分布。公式如下：

$$
X_{scaled} = \frac{X - \mu}{\sigma}
$$

其中，`μ` 为数据的均值，`σ` 为数据的标准差。

## 2. 时间序列数据集构建

### 2.1 时间序列数据集构建
由于数据是时间序列数据，模型需要考虑历史数据对未来的影响。为此，使用 **滑动窗口**（Sliding Window）技术来构建训练数据集。具体操作如下：
- 每个窗口包含 `time_steps` 个数据点。
- 使用前 `time_steps` 个数据点来预测下一个数据点的目标值。

### 2.2 划分训练集和标签
在构建窗口化数据集后：
- `X`：表示每个窗口的数据（前 `time_steps` 个数据点）。
- `y`：表示每个窗口后面的目标值（即 `time_steps` 后的下一个数据点）。

### 2.3 划分训练集和测试集
将数据划分为训练集和测试集：
- `X_train, y_train`：用于训练模型的数据和目标值。
- `X_test, y_test`：用于评估模型的测试数据。

## 3. 模型初始化

### 3.1 初始化 SGDRegressor
使用 **SGDRegressor**（随机梯度下降回归）模型进行训练。该模型适用于大规模数据集和增量学习，能够通过 **梯度下降** 优化损失函数。模型初始化时设置以下参数：
- `max_iter=3000`：最大迭代次数为 3000。
- `tol=1e-4`：容忍误差小于 `1e-4` 时停止训练。
- `learning_rate='adaptive'`：自适应学习率，能够根据损失函数变化动态调整学习率。
- `eta0=0.001`：初始学习率设置为 0.001。

## 4. 模型训练

### 4.1 初始训练
使用训练集的前 `train_size` 个数据点进行初始训练。

### 4.2 增量训练
一旦初始模型训练完成，模型将进入增量训练阶段。在每次有新数据到来时，模型会执行以下操作：
- 预测新数据点，并计算预测值与真实值之间的误差。
- 如果误差超过设定的阈值 `error_threshold`，或者预测值超出预定的有效范围 `min_value` 和 `max_value`，模型将进行调整：
  - **重新训练**：当误差过大时，使用所有训练数据重新训练模型。
  - **加载 Checkpoint 模型**：如果模型预测值超出预定范围，则会尝试加载 **checkpoint** 文件夹中的最新模型进行预测和调整。如果加载的模型符合要求，则继续增量训练；否则，删除该模型并继续使用历史模型进行调整。

### 4.3 模型更新与保存
每次增量训练后，模型将保存为一个新的版本：
- 保存训练好的模型到 `checkpoints/` 文件夹。
- 如果连续训练次数超过 `max_continuous_training`，会将模型保存为一个新的 checkpoint。

### 4.4 预测结果与保存
每次训练后，模型将对新数据进行预测，并保存结果：
- 预测结果将保存为 `predictions.csv`。
- 预测结果的可视化图表（`result_plot.png`）也将被保存，展示实际值与预测值的对比。

### 4.5 预测值裁剪
为了防止预测值超出设定的有效范围（`min_value` 和 `max_value`），每次预测后，模型输出的预测值将通过 `np.clip()` 函数裁剪到设定范围内。

## 5. Checkpoint 机制

### 5.1 Checkpoint 文件夹
在模型训练过程中，所有的模型和训练结果都会保存在 `checkpoints/` 文件夹中。如果模型预测误差过大或出现异常预测，我们会尝试加载该文件夹中的最新模型进行预测和调整。如果加载的模型表现不佳，模型会删除该模型，并继续使用历史模型进行调整。

### 5.2 增量训练与模型存储
每次增量训练后，模型都会保存最新的状态，并存储为 `.pkl` 文件。这些文件存储在 `checkpoints/` 文件夹中，并在后续的训练中进行加载和使用。

## 6. 总结

### 6.1 增量训练的优势
增量训练的最大优势是避免了从头开始训练整个数据集。当新数据到来时，模型能够快速适应变化，提高效率。此外，当预测误差过大时，模型能够自动回滚到之前的版本，避免不必要的过拟合或误差。

### 6.2 模型的稳定性
通过设置误差阈值、模型预测范围和 checkpoint 机制，模型能够保持较高的稳定性。即使出现过度拟合或预测误差较大的情况，模型也能有效回退并恢复较为稳定的状态。

### 6.3 结果输出
训练完成后，模型的预测结果和训练的模型将被保存在 `result/` 文件夹中，包括：
- 预测结果的 CSV 文件。
- 预测结果的图表（`result_plot.png`）。
- 训练后的模型存储在 `checkpoints/` 文件夹中。

这种训练过程能够确保模型对新数据的适应性，同时保持较高的预测精度和模型稳定性。
