import os
import json
import random
import argparse
from pathlib import Path

def generate_anomaly_detection_swift_datasets(train_json_path, test_json_path, output_dir, num_images=5, random_seed=42):
    """
    读取训练集和测试集JSON文件，为异常图像检测任务生成MS-SWIFT格式的数据集
    使用更通用的格式，将图像和文本分开
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义异常图像检测任务的提示模板
    prompt_template = (
        "I've shown you {num_images} images of locations taken over time.\n\n"
        "In a normal temporal sequence, all images should show the same location as it changes over time.\n"
        "However, one of these images may be an anomaly - it either shows a different location or was taken at an inconsistent time point that breaks the logical progression.\n\n"
        "Please identify which image (if any) is the anomaly that doesn't belong in this temporal sequence.\n\n"
        "Answer with the number of the anomalous image (1 to {num_images}), or answer \"None\" if all images appear to form a consistent temporal sequence of the same location."
    )
    
    # 处理训练集和测试集
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 获取所有图像组
        image_groups = dataset.get("image_groups", [])
        
        # 过滤出包含足够图像的组
        valid_groups = [group for group in image_groups if len(group.get("images", [])) >= num_images]
        
        if len(valid_groups) < 2:
            print(f"警告: 在{split}集中没有足够的图像组用于生成异常检测任务!")
            continue
            
        # 生成异常图像检测任务数据
        anomaly_data = []
        
        for group_idx, group in enumerate(valid_groups):
            images = group.get("images", [])
            
            # 确保图像数量足够
            if len(images) < num_images:
                continue
                
            # 随机决定是否插入异常
            has_anomaly = random.random() > 0.5
            
            if has_anomaly:
                # 选择正常序列的图像
                selected_normal_images = random.sample(images, num_images - 1)
                
                # 按文件名排序，确定真实时间顺序
                image_filenames = [os.path.basename(img) for img in selected_normal_images]
                sorted_indices = sorted(range(len(selected_normal_images)), key=lambda i: image_filenames[i])
                normal_images = [selected_normal_images[i] for i in sorted_indices]
                
                # 从其他图像组中选择一张图像作为异常
                other_groups = valid_groups.copy()
                other_groups.pop(group_idx)  # 移除当前组
                
                # 随机选择另一个组
                anomaly_group = random.choice(other_groups)
                anomaly_image = random.choice(anomaly_group.get("images", []))
                
                # 随机决定异常图像的插入位置
                anomaly_position = random.randint(0, num_images - 1)
                
                # 构建最终图像序列
                final_images = normal_images.copy()
                final_images.insert(anomaly_position, anomaly_image)
                
                # 如果序列长度超过num_images，截断
                if len(final_images) > num_images:
                    final_images = final_images[:num_images]
                    
                # 异常图像的位置编号（从1开始）
                anomaly_idx = anomaly_position + 1
                correct_answer = str(anomaly_idx)
            else:
                # 无异常情况：选择连续的图像序列
                selected_images = random.sample(images, num_images)
                
                # 按文件名排序，确定真实时间顺序
                image_filenames = [os.path.basename(img) for img in selected_images]
                sorted_indices = sorted(range(len(selected_images)), key=lambda i: image_filenames[i])
                final_images = [selected_images[i] for i in sorted_indices]
                
                correct_answer = "None"
            
            # 填充提示模板中的参数
            formatted_prompt = prompt_template.format(num_images=num_images)
            
            # 构造MS-SWIFT格式（将图像和文本分开）
            swift_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt
                    },
                    {
                        "role": "assistant",
                        "content": correct_answer
                    }
                ],
                "images": final_images
            }
            
            anomaly_data.append(swift_sample)
        
        # 保存MS-SWIFT格式数据集
        output_path = os.path.join(output_dir, f"anomaly_detection_swift_{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anomaly_data, f, indent=2, ensure_ascii=False)
        
        print(f"生成MS-SWIFT格式异常图像检测任务{split}集: {output_path}")
        print(f"包含{len(anomaly_data)}个样本")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='为异常图像检测任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/openbayes/home/0917/json/TOU/train_dataset.json",
                        help='训练集JSON文件路径 (默认: train_dataset.json)')
    parser.add_argument('--test_json', type=str, default="/openbayes/home/0917/json/TOU/test_dataset.json",
                        help='测试集JSON文件路径 (默认: test_dataset.json)')
    parser.add_argument('--output_dir', type=str, default="/openbayes/home/0917/json/TOU/task_datasets",
                        help='输出目录 (默认: task_datasets)')
    parser.add_argument('--num_images', type=int, default=5,
                        help='每个任务实例使用的图片数量 (默认: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 生成MS-SWIFT格式异常图像检测任务数据集
    generate_anomaly_detection_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.num_images,
        args.seed
    )