# 在这里需要引入必要的函数和库
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
import csv
import os
import joblib
import matplotlib.pyplot as plt

max_continuous_training=70
continuous_train_count=0

def load_data(file_path):
    """读取并处理数据"""
    data = pd.read_csv(file_path)
    data['query_time'] = pd.to_datetime(data['query_time'])
    values = data['value'].values
    return data, values


def normalize(data, mean_value, std_value):
    """自定义标准化函数"""
    return (data - mean_value) / std_value


def denormalize(data, mean_value, std_value):
    """自定义反标准化函数"""
    return data * std_value + mean_value


def create_dataset(data, time_steps=1):
    """创建时间序列数据集"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])  # 时间序列窗口数据
        y.append(data[i + time_steps, 0])      # 当前时刻的值
    return np.array(X), np.array(y)


def save_results_to_csv(date, real_value, predicted_value, error, file_name="predictions.csv"):
    """保存预测结果到CSV文件"""
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['query_time', 'real_value', 'predicted_value', 'error'])  # 写入表头
        writer.writerow([date, real_value, predicted_value, error])  # 写入数据


def save_model(model, model_filename):
    """保存训练好的模型"""
    joblib.dump(model, model_filename)
    print(f"模型已保存为：{model_filename}")


def load_model(model_filename):
    """加载保存的模型"""
    return joblib.load(model_filename)


def train_or_retrain_model(X_train, y_train, X_new, y_new, model, checkpoint_dir, error_threshold=0.005, min_value=0.001, max_value=0.125):
    """增量训练模型或从checkpoint加载模型"""
    global continuous_train_count

    # 用当前模型进行预测
    predicted_new = model.predict(X_new)
    error_new = np.abs(predicted_new - y_new)

    # 如果误差超过阈值，检查checkpoint模型
    if error_new > error_threshold or predicted_new[0] < min_value or predicted_new[0] > max_value:
        print(f"预测误差超出阈值，尝试加载checkpoint模型...")

        # 按时间顺序加载所有checkpoint模型
        checkpoint_files = sorted(os.listdir(checkpoint_dir), reverse=True)  # 按最新的模型排序
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

            # 加载checkpoint模型
            checkpoint_model = load_model(checkpoint_path)

            # 用checkpoint模型进行预测
            predicted_new_checkpoint = checkpoint_model.predict(X_new)
            error_checkpoint = np.abs(predicted_new_checkpoint - y_new)

            # 如果checkpoint模型的误差小于阈值，则采用该模型
            if error_checkpoint <= error_threshold and predicted_new_checkpoint[0] >= min_value and predicted_new_checkpoint[0] <= max_value:
                print(f"加载checkpoint模型 {checkpoint_file}，误差在阈值内")

                # 进行增量训练
                checkpoint_model.partial_fit(X_new, np.array([y_new]))
                save_model(checkpoint_model, checkpoint_path)  # 保存更新后的模型
                return checkpoint_model

            else:
                print(f"加载的checkpoint模型 {checkpoint_file} 不符合条件，删除此模型...")
                os.remove(checkpoint_path)  # 删除不符合条件的模型

        # 如果没有符合条件的模型，则重新训练
        print("没有符合条件的模型，重新训练模型...")
        model.fit(X_train, y_train)
        save_model(model, os.path.join(checkpoint_dir, f"model_{len(checkpoint_files) + 1}.pkl"))  # 保存新模型
        continuous_train_count = 0  # 重置连续训练计数
        return model
    else:
        # 如果误差在阈值内，就继续增量训练
        model.partial_fit(X_new, np.array([y_new]))
        continuous_train_count += 1
        if continuous_train_count >= max_continuous_training:
            print(f"连续训练超过 {max_continuous_training} 次，保存checkpoint模型...")
            save_model(model, os.path.join(checkpoint_dir, f"checkpoint_model_{continuous_train_count}.pkl"))
            continuous_train_count = 0  # 重置计数器
        return model


def train_and_predict(data, values, train_size, min_value, max_value, error_threshold, checkpoint_dir,prediction_csv_file):
    """主训练和预测过程"""
    # 归一化数据
    mean_value = np.mean(values)
    std_value = np.std(values)
    scaled_values = normalize(values, mean_value, std_value)

    # 创建窗口化数据集
    X, y = create_dataset(scaled_values.reshape(-1, 1), time_steps=30)

    # 划分训练集和测试集
    X_train, y_train = X[:train_size], y[:train_size]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))  # 重塑训练数据
    X_test, y_test = X[train_size:], y[train_size:]

    # 初始化SGDRegressor模型
    model = SGDRegressor(max_iter=3000, tol=1e-4, learning_rate='adaptive', eta0=0.001, alpha=0.1)

    # 用前train_size行数据训练模型
    model.fit(X_train, y_train)

    # 模拟训练过程
    for i in range(train_size, len(X)):
        X_new, y_new = X[i].reshape(1, -1), y[i]
        start_idx = max(0, i - 30)
        X_retrain, y_retrain = X[start_idx:i + 1], y[start_idx:i + 1]

        # 用当前模型对新数据进行预测并计算误差
        model = train_or_retrain_model(X_retrain, y_retrain, X_new, y_new, model, checkpoint_dir, error_threshold,min_value,max_value)

        predicted_new = model.predict(X_new)
        predicted_new = denormalize(predicted_new, mean_value, std_value)
        y_actual_new = denormalize(np.array([y_new]), mean_value, std_value)

        # 计算新数据的预测误差
        error_new = np.abs(predicted_new - y_actual_new)

        # 裁剪预测值到设定的区间 [min_value, max_value]
        predicted_new = np.clip(predicted_new, min_value, max_value)

        # 输出当前预测误差
        print(f"当前误差: {error_new[0]}")

        # 保存当前预测结果
        predicted_date = data['query_time'][train_size + i]
        save_results_to_csv(predicted_date, y_actual_new[0], predicted_new[0], error_new[0],prediction_csv_file)

def create_and_save_picture(data_path,output_path):
    
    # 读取数据
    data = pd.read_csv(data_path)

    # 绘制实际值和预测值的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(data['query_time'], data['real_value'], label='Target', marker='o')
    plt.plot(data['query_time'], data['predicted_value'], label='Predict', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Target vs Predict')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def model(csv_file_path='gate_error_gate_ecr0_1_q0.csv', prediction_csv_file="predictions.csv", checkpoint_dir='checkpoints', picture_output_path="result_plot.png", train_size=30, error_threshold=0.003, max_continuous_training=70):
    """主函数，便于其他脚本调用"""

    global continuous_train_count

    # 读取数据
    data, values = load_data(csv_file_path)

    # 动态计算 min_value 和 max_value
    max_value = np.max(values) + error_threshold
    min_value = max(np.min(values) - error_threshold, 0)

    continuous_train_count = 0
    # 训练和预测
    train_and_predict(data, values, train_size, min_value, max_value, error_threshold, checkpoint_dir,prediction_csv_file)
    
    # 生成并保存预测图像
    create_and_save_picture(prediction_csv_file, picture_output_path)

if __name__ == "__main__":
    # 运行主函数，传入默认参数
    model()
