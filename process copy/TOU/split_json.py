import json
import random
import os
import argparse
from pathlib import Path

def split_dataset(input_json_path, train_json_path, test_json_path, train_ratio=0.8, random_seed=42):
    """
    将数据集划分为训练集和测试集
    
    参数:
        input_json_path: 输入的JSON文件路径
        train_json_path: 输出的训练集JSON文件路径
        test_json_path: 输出的测试集JSON文件路径
        train_ratio: 训练集占总数据的比例，默认为0.8（80%）
        random_seed: 随机种子，用于保证可重复性
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 读取输入的JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 获取所有图像组
    image_groups = dataset.get("image_groups", [])
    
    # 确保有数据可分割
    if not image_groups:
        print("错误: 输入的JSON文件中没有找到图像组数据!")
        return
    
    # 随机打乱图像组顺序
    random.shuffle(image_groups)
    
    # 计算训练集大小
    train_size = int(len(image_groups) * train_ratio)
    
    # 划分训练集和测试集
    train_groups = image_groups[:train_size]
    test_groups = image_groups[train_size:]
    
    # 创建训练集和测试集的JSON对象
    train_dataset = {
        "image_groups": train_groups,
        "split": "train",
        "num_groups": len(train_groups)
    }
    
    test_dataset = {
        "image_groups": test_groups,
        "split": "test",
        "num_groups": len(test_groups)
    }
    
    # 写入训练集JSON文件
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)
    
    # 写入测试集JSON文件
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)
    
    # 打印结果信息
    print(f"数据集划分完成!")
    print(f"总图像组数量: {len(image_groups)}")
    print(f"训练集图像组数量: {len(train_groups)} ({len(train_groups)/len(image_groups)*100:.1f}%)")
    print(f"测试集图像组数量: {len(test_groups)} ({len(test_groups)/len(image_groups)*100:.1f}%)")
    print(f"训练集已保存至: {train_json_path}")
    print(f"测试集已保存至: {test_json_path}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将数据集划分为训练集和测试集')
    parser.add_argument('--input', type=str, default="/data/ss/benchmark/json/TOU/dataset_full.json",
                        help='输入的JSON文件路径 (默认: dataset_images.json)')
    parser.add_argument('--train_output', type=str, default="/data/ss/benchmark/json/TOU/train_dataset.json",
                        help='输出的训练集JSON文件路径 (默认: train_dataset.json)')
    parser.add_argument('--test_output', type=str, default="/data/ss/benchmark/json/TOU/test_dataset.json",
                        help='输出的测试集JSON文件路径 (默认: test_dataset.json)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集占总数据的比例 (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用数据集划分函数
    split_dataset(
        args.input,
        args.train_output,
        args.test_output,
        args.train_ratio,
        args.seed
    )