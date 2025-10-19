import os
import json
import random
import argparse
from pathlib import Path

def generate_is_true_order_swift_datasets(train_json_path, test_json_path, output_dir, min_images=3, max_images=7, random_seed=42):
    """
    读取训练集和测试集JSON文件，为序列合理性判别任务生成MS-SWIFT格式的数据集
    
    参数:
        train_json_path: 训练集JSON文件路径
        test_json_path: 测试集JSON文件路径
        output_dir: 输出目录
        min_images: 每个任务实例最少使用的图片数量，默认为3
        max_images: 每个任务实例最多使用的图片数量，默认为7
        random_seed: 随机种子，用于保证可重复性
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义序列合理性判别任务的提示模板
    prompt_template = (
        "You are given several pictures in a sequence: Picture 1, Picture 2, ..., Picture N.\n\n"
        "Please determine whether these pictures are presented in the correct temporal order.\n\n"
        "Answer with \"True\" if the sequence is in the correct chronological order, otherwise answer with \"False\"."
    )
    
    # 处理训练集和测试集
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 获取图像组
        image_groups = dataset.get("image_groups", [])
        
        # 生成序列合理性判别任务数据
        is_true_order_data = []
        
        for group in image_groups:
            images = group.get("images", [])
            
            # 确保图像数量足够
            if len(images) < min_images:
                continue
            
            # 随机选择3-7张图片
            num_images = random.randint(min_images, min(max_images, len(images)))
            selected_images = random.sample(images, num_images)
            
            # 按文件名排序，这才是"真正的时间顺序"
            image_filenames = [os.path.basename(img) for img in selected_images]
            true_order = sorted(range(len(selected_images)), key=lambda i: image_filenames[i])
            ordered_images = [selected_images[i] for i in true_order]
            
            # 随机决定是否打乱顺序
            if random.random() > 0.5:
                # 使用正确的时间顺序
                final_images = ordered_images[:]
                answer = True
            else:
                # 确保打乱后的顺序不等于正确顺序
                shuffled = ordered_images[:]
                while shuffled == ordered_images:  # 循环直到确实打乱了
                    random.shuffle(shuffled)
                final_images = shuffled
                answer = False
            
            # 构建带编号的图片提示部分
            image_prompt = ""
            for i in range(len(final_images)):
                image_prompt += f"Picture {i+1}: <image> "
            
            # 转换为字符串格式
            response = "True" if answer else "False"
            
            # 构造MS-SWIFT格式
            swift_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": image_prompt.strip() + "\n\n" + prompt_template
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ],
                "images": final_images
            }
            
            is_true_order_data.append(swift_sample)
        
        # 保存MS-SWIFT格式数据集
        output_path = os.path.join(output_dir, f"is_true_order_swift_{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(is_true_order_data, f, indent=2, ensure_ascii=False)
        
        print(f"生成MS-SWIFT格式序列合理性判别任务{split}集: {output_path}")
        print(f"包含{len(is_true_order_data)}个样本")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='为序列合理性判别任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/openbayes/home/0917/json/TOU/train_dataset.json",
                        help='训练集JSON文件路径 (默认: train_dataset.json)')
    parser.add_argument('--test_json', type=str, default="/openbayes/home/0917/json/TOU/test_dataset.json",
                        help='测试集JSON文件路径 (默认: test_dataset.json)')
    parser.add_argument('--output_dir', type=str, default="/openbayes/home/0917/json/TOU/task_datasets",
                        help='输出目录 (默认: task_datasets)')
    parser.add_argument('--min_images', type=int, default=3,
                        help='每个任务实例最少使用的图片数量 (默认: 3)')
    parser.add_argument('--max_images', type=int, default=7,
                        help='每个任务实例最多使用的图片数量 (默认: 7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 生成MS-SWIFT格式序列合理性判别任务数据集
    generate_is_true_order_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.min_images,
        args.max_images,
        args.seed
    )