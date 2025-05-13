import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_yolo_training_curves(csv_files, model_labels, title_prefix="YOLO Training"):
    """
    读取多个Ultralytics训练生成的CSV文件，并在同一图上绘制Loss和mAP曲线。

    Args:
        csv_files (list): 包含所有模型results.csv文件路径的列表。
        model_labels (list): 对应csv_files中每个文件的模型标签或名称列表。
        title_prefix (str): 图表标题的前缀。
    """

    if len(csv_files) != len(model_labels):
        raise ValueError("CSV文件列表和模型标签列表的长度必须一致！")

    # --- 绘制 Loss 图 ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1) # 创建第一个子图用于 Loss

    for i, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
            print(f"警告: 文件不存在 - {csv_file}，跳过。")
            continue

        df = pd.read_csv(csv_file)

        # 检查必需的列是否存在
        required_loss_cols = ['epoch', 'train/loss', 'val/loss']
        if not all(col in df.columns for col in required_loss_cols):
            print(f"警告: 文件 {csv_file} 缺少 Loss 相关的列，跳过 Loss 绘图。")
            continue

        # 绘制训练 Loss
        plt.plot(df['epoch'], df['train/loss'], label=f'{model_labels[i]} Train Loss', linestyle='-')
        # 绘制验证 Loss
        plt.plot(df['epoch'], df['val/loss'], label=f'{model_labels[i]} Val Loss', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6) # 添加网格线

    # --- 绘制 mAP 图 ---
    plt.subplot(1, 2, 2) # 创建第二个子图用于 mAP

    for i, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
             # 警告已在 Loss 图中打印，这里不再重复
            continue

        df = pd.read_csv(csv_file)

        # 检查必需的列是否存在
        required_map_cols = ['epoch', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']
        if not all(col in df.columns for col in required_map_cols):
            print(f"警告: 文件 {csv_file} 缺少 mAP 相关的列，跳过 mAP 绘图。")
            continue

        # 绘制 mAP@0.5
        plt.plot(df['epoch'], df['metrics/mAP_0.5'], label=f'{model_labels[i]} mAP@0.5', linestyle='-')
        # 绘制 mAP@0.5:0.95
        plt.plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label=f'{model_labels[i]} mAP@0.5:0.95', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title(f'{title_prefix} mAP Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6) # 添加网格线

    # 调整布局，避免重叠
    plt.tight_layout()

    # 显示图表
    plt.show()

# --- 示例使用 ---

# 替换为你的CSV文件路径列表
# 假设你有两个模型的训练结果文件：run1/results.csv 和 run2/results.csv
example_csv_files = [
    'path/to/your/run1/results.csv',
    'path/to/your/run2/results.csv',
    'path/to/your/run3/results.csv', # 可以添加更多文件
]

# 对应上面文件的模型标签或名称列表
example_model_labels = [
    'Model A',
    'Model B',
    'Model C', # 对应上面添加的文件
]

# 调用函数绘制图表
# plot_yolo_training_curves(example_csv_files, example_model_labels, title_prefix="My Custom YOLO Training")

# 注意：在实际使用时，请将 example_csv_files 和 example_model_labels 替换为你自己的文件路径和标签。
# 确保文件路径正确，并且文件存在。
