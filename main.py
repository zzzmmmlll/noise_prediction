import os
import shutil
import pandas as pd
from model import model  # 导入model1.py中的main函数

def process_csv_file(csv_file_path, result_base_dir, **kwargs):
    """处理单个 CSV 文件，生成预测结果、模型和图片"""
    
  # 获取文件相对路径
    relative_path = os.path.relpath(csv_file_path, 'data')  # 相对路径
    
    # 获取文件夹路径，并去除 .csv 后缀作为文件夹名称
    relative_folder_path = os.path.dirname(relative_path)
    file_name_without_extension = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    # 创建目标文件夹路径
    result_dir = os.path.join(result_base_dir, relative_folder_path, file_name_without_extension)

    print(result_dir)
    # 创建目标文件夹
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 设置预测结果保存路径
    prediction_csv_file = os.path.join(result_dir, 'predictions.csv')
    print(prediction_csv_file)
    # 设置图片保存路径
    picture_output_path = os.path.join(result_dir, 'result_plot.png')
    
    # 设置模型保存路径
    model_filename = os.path.join(result_dir, 'trained_model.pkl')
    
    # 创建 checkpoints 文件夹
    checkpoint_dir = os.path.join(result_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 调用 model1.py 中的 main 函数，进行训练和预测
    kwargs['csv_file_path'] = csv_file_path
    kwargs['prediction_csv_file'] = prediction_csv_file
    kwargs['checkpoint_dir'] = checkpoint_dir  # 使用为该文件生成的 checkpoint 文件夹
    kwargs['picture_output_path'] = picture_output_path
    # 调用 model1 中的 main 函数进行训练、预测和图像保存
    model(**kwargs)

def process_directory(directory_path, result_base_dir, **kwargs):
    """递归遍历文件夹，处理其中所有的 CSV 文件"""
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                print(f"正在处理文件: {csv_file_path}")
                process_csv_file(csv_file_path, result_base_dir, **kwargs)

def main():
    """主函数，遍历 data 文件夹并处理所有 CSV 文件"""
    
    # 设置需要传递给 model1.py 的其他参数
    kwargs = {
        'train_size': 30,  # 示例参数，用户可以根据需要调整
        'error_threshold': 0.003,
        'max_continuous_training': 70
    }
    
    # 设置数据和结果的基础目录
    data_dir = 'data'  # 输入文件夹
    result_base_dir = 'result'  # 输出文件夹
    
    # 清空 result 文件夹（如果需要）
    if os.path.exists(result_base_dir):
        shutil.rmtree(result_base_dir)
    os.makedirs(result_base_dir)
    
    # 递归处理 data 文件夹中的所有 CSV 文件
    process_directory(data_dir, result_base_dir, **kwargs)

if __name__ == "__main__":
    main()
