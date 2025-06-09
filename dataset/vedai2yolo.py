import os
import shutil
import random
from pathlib import Path

# --- 1. 配置你的路径和参数 ---

# 输入目录
SOURCE_IMAGES_DIR = Path("VEDAI/images")
SOURCE_LABELS_DIR = Path("VEDAI/labels")

# 输出目录
OUTPUT_DIR = Path("output_datasets")

# 数据集划分比例（例如，0.2 表示 20% 的数据作为验证集）
VAL_SPLIT_RATIO = 0.2

# ⚠️【重要】请在这里修改为你自己的类别名称！顺序必须和YOLO标签中的类别索引一致。
CLASS_NAMES = ['person', 'car', 'bicycle'] # 这是一个例子，请务必修改

# --- 2. 脚本主逻辑 (通常无需修改) ---

def create_yolo_datasets():
    """
    主函数，用于创建和组织三个YOLO数据集。
    """
    print("🚀 开始处理数据集...")

    # 检查输入目录是否存在
    if not SOURCE_IMAGES_DIR.is_dir() or not SOURCE_LABELS_DIR.is_dir():
        print(f"❌ 错误：找不到源文件夹 '{SOURCE_IMAGES_DIR}' 或 '{SOURCE_LABELS_DIR}'。")
        print("请确保脚本与 'images' 和 'labels' 文件夹在同一目录下。")
        return

    # 从标签文件中获取所有文件名的“主干”部分 (例如 '00000000')
    base_filenames = sorted([p.stem for p in SOURCE_LABELS_DIR.glob("*.txt")])
    
    if not base_filenames:
        print("❌ 错误：在 'labels' 文件夹中没有找到 .txt 标注文件。")
        return
        
    print(f"✅ 找到了 {len(base_filenames)} 个标注文件，对应 {len(base_filenames) * 2} 张图像。")

    # 随机打乱并划分训练集和验证集
    random.shuffle(base_filenames)
    split_index = int(len(base_filenames) * (1 - VAL_SPLIT_RATIO))
    train_files = base_filenames[:split_index]
    val_files = base_filenames[split_index:]

    print(f"📊 数据集划分: {len(train_files)} 个用于训练, {len(val_files)} 个用于验证。")

    # 定义要创建的数据集
    datasets_to_create = {
        "dataset_visible": ["co"],
        "dataset_thermal": ["ir"],
        "dataset_combined": ["co", "ir"]
    }

    # 为每个数据集类型进行处理
    for dataset_name, suffixes in datasets_to_create.items():
        print(f"\nProcessing dataset: {dataset_name}...")
        dataset_path = OUTPUT_DIR / dataset_name
        
        # 创建 train/val 文件夹结构
        train_images_path = dataset_path / "images" / "train"
        val_images_path = dataset_path / "images" / "val"
        train_labels_path = dataset_path / "labels" / "train"
        val_labels_path = dataset_path / "labels" / "val"
        
        for p in [train_images_path, val_images_path, train_labels_path, val_labels_path]:
            p.mkdir(parents=True, exist_ok=True)
            
        # 处理文件并复制
        process_split(train_files, suffixes, train_images_path, train_labels_path, "train")
        process_split(val_files, suffixes, val_images_path, val_labels_path, "val")

        # 创建 dataset.yaml 文件
        create_yaml_file(dataset_path, CLASS_NAMES)
        
    print("\n🎉 所有数据集处理完成！")
    print(f"请检查 '{OUTPUT_DIR}' 文件夹。")


def process_split(file_list, suffixes, dest_img_path, dest_label_path, split_name):
    """
    根据给定的文件列表和后缀，复制图像和标签文件。
    """
    count = 0
    for base_name in file_list:
        source_label_file = SOURCE_LABELS_DIR / f"{base_name}.txt"
        
        if not source_label_file.exists():
            continue

        for suffix in suffixes:
            source_image_file = SOURCE_IMAGES_DIR / f"{base_name}_{suffix}.png"
            if source_image_file.exists():
                # 复制并重命名图像和标签文件，使它们一一对应
                dest_image_file = dest_img_path / f"{base_name}_{suffix}.png"
                dest_label_file = dest_label_path / f"{base_name}_{suffix}.txt" # 标签文件名与图像名匹配
                
                shutil.copy(source_image_file, dest_image_file)
                shutil.copy(source_label_file, dest_label_file)
                count += 1
    print(f"  -> 成功复制了 {count} 张图像及其标签到 {split_name} 集。")


def create_yaml_file(dataset_path, class_names):
    """
    为数据集创建 .yaml 配置文件。
    """
    yaml_content = f"""
# YOLOv8 Dataset Configuration File

path: {dataset_path.resolve()}  # 数据集根目录 (绝对路径)
train: images/train  # 训练集图片目录 (相对于 'path')
val: images/val      # 验证集图片目录 (相对于 'path')

# Classes
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(dataset_path / "dataset.yaml", "w", encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"  -> 已在 '{dataset_path}' 中创建 dataset.yaml。")


if __name__ == "__main__":
    create_yolo_datasets()