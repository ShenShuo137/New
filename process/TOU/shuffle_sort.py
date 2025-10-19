import os
import json
import random
import argparse
from pathlib import Path

def generate_shuffle_sort_swift_datasets(train_json_path, test_json_path, output_dir, num_images=5, random_seed=42):
    """
    读取训练集和测试集JSON文件，为乱序重排任务生成MS-SWIFT格式的数据集
    
    参数:
        train_json_path: 训练集JSON文件路径
        test_json_path: 测试集JSON文件路径
        output_dir: 输出目录
        num_images: 每个任务实例使用的图片数量，默认为5
        random_seed: 随机种子，用于保证可重复性
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义乱序重排任务的提示模板
    prompt_template = (
        "You are given 5 pictures in random order: Picture 1, Picture 2, Picture 3, Picture 4, and Picture 5.\n\n"
        "Your task is to determine their correct chronological order based on visual content.\n\n"
        "Please output five integers representing the correct temporal order of the pictures.\n\n"
        "Each integer should be between 1 and 5, indicating the index of the picture in the original input list "
        "(1 = Picture 1, 2 = Picture 2, ..., 5 = Picture 5).\n\n"
        "The output should be a permutation of 1 to 5, indicating the order in which the input pictures occurred in time.\n\n"
        "For example, if the correct chronological order is: Picture 4, Picture 2, Picture 3, Picture 1, Picture 5, "
        "output only like this: `4 2 3 1 5`. Do not provide any explanations or additional text."
    )
    
    # 处理训练集和测试集
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 获取图像组
        image_groups = dataset.get("image_groups", [])
        
        # 生成乱序重排任务数据
        shuffle_sort_data = []
        
        for group in image_groups:
            images = group.get("images", [])
            
            # 确保图像数量足够
            if len(images) < num_images:
                continue
            
            # 随机选择指定数量的图像
            selected_images = random.sample(images, num_images)
            
            # 假设文件名按字母顺序排列就是时间顺序
            # 这里使用文件路径的基本名称(如A.png, B.png等)来排序
            image_filenames = [os.path.basename(img) for img in selected_images]
            true_order = sorted(range(len(selected_images)), key=lambda i: image_filenames[i])
            
            # 创建打乱的图像序列
            shuffled_images = selected_images.copy()
            random.shuffle(shuffled_images)
            
            # 计算答案 - 即将打乱的序列恢复到正确顺序的索引
            # 对于每个正确位置，找出应该放置哪个打乱后的图像
            shuffled_to_true = {}
            for i, img in enumerate(shuffled_images):
                for j, true_img in enumerate(selected_images):
                    if img == true_img:
                        shuffled_to_true[j] = i
                        break
            
            # 构建答案数组 - 答案是一个包含1到num_images的数组
            # 表示正确顺序中的第i个位置应该放置打乱后序列中的第answer[i]个图像
            answer = [shuffled_to_true[i] + 1 for i in true_order]
            
            # 构建图片提示部分
            image_prompt = ""
            for i in range(len(shuffled_images)):
                image_prompt += f"Picture {i+1}: <image> "
            
            # 将answer数组转换为空格分隔的字符串
            response = " ".join(str(x) for x in answer)
            
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
                "images": shuffled_images
            }
            
            shuffle_sort_data.append(swift_sample)
        
        # 保存MS-SWIFT格式数据集
        output_path = os.path.join(output_dir, f"shuffle_sort_swift_{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(shuffle_sort_data, f, indent=2, ensure_ascii=False)
        
        print(f"生成MS-SWIFT格式乱序重排任务{split}集: {output_path}")
        print(f"包含{len(shuffle_sort_data)}个样本")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='为乱序重排任务生成MS-SWIFT格式数据集')
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
    
    # 生成MS-SWIFT格式乱序重排任务数据集
    generate_shuffle_sort_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.num_images,
        args.seed
    )