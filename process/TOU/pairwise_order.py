import os
import json
import random
import argparse
from pathlib import Path

def generate_pairwise_order_swift_datasets(train_json_path, test_json_path, output_dir, random_seed=42):
    """
    读取训练集和测试集JSON文件，为配对顺序判断任务生成MS-SWIFT格式的数据集
    
    参数:
        train_json_path: 训练集JSON文件路径
        test_json_path: 测试集JSON文件路径
        output_dir: 输出目录
        random_seed: 随机种子，用于保证可重复性
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义配对顺序判断任务的提示模板
    prompt_template = (
        "You are given two pictures: Picture 1 and Picture 2.\n\n"
        "Please determine whether their temporal order is correct.\n"
        "Picture 1 appears earlier than Picture 2 in time.\n"
        "Answer with 'True' if the order is correct, otherwise answer with 'False'."
    )
    
    # 处理训练集和测试集
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 获取图像组
        image_groups = dataset.get("image_groups", [])
        
        # 生成配对顺序判断任务数据
        pairwise_order_data = []
        
        for group in image_groups:
            images = group.get("images", [])
            
            # 确保图像数量足够
            if len(images) < 2:
                continue
            
            # 随机选择2张图像
            selected_images = random.sample(images, 2)
            
            # 假设文件名按字母顺序排列就是时间顺序
            # 这里使用文件路径的基本名称(如A.png, B.png等)来排序
            image_filenames = [os.path.basename(img) for img in selected_images]
            is_ordered = image_filenames[0] < image_filenames[1]
            
            # 随机决定是保持正确顺序还是反转顺序
            if random.random() > 0.5:
                ordered_images = selected_images[:]  # 正序
                answer = is_ordered  # 如果本来就是正序的，答案是True；否则是False
            else:
                ordered_images = selected_images[::-1]  # 反序
                answer = not is_ordered  # 如果本来是正序的，反转后答案是False；否则是True
            
            # 构建图片提示部分
            image_prompt = "Picture 1: <image> Picture 2: <image>"
            
            # 转换为字符串格式
            response = "True" if answer else "False"
            
            # 构造MS-SWIFT格式
            swift_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": image_prompt + "\n\n" + prompt_template
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ],
                "images": ordered_images
            }
            
            pairwise_order_data.append(swift_sample)
        
        # 保存MS-SWIFT格式数据集
        output_path = os.path.join(output_dir, f"pairwise_order_swift_{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairwise_order_data, f, indent=2, ensure_ascii=False)
        
        print(f"生成MS-SWIFT格式配对顺序判断任务{split}集: {output_path}")
        print(f"包含{len(pairwise_order_data)}个样本")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='为配对顺序判断任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/openbayes/home/0917/json/TOU/train_dataset.json",
                        help='训练集JSON文件路径 (默认: train_dataset.json)')
    parser.add_argument('--test_json', type=str, default="/openbayes/home/0917/json/TOU/test_dataset.json",
                        help='测试集JSON文件路径 (默认: test_dataset.json)')
    parser.add_argument('--output_dir', type=str, default="/openbayes/home/0917/json/TOU/task_datasets",
                        help='输出目录 (默认: task_datasets)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 生成MS-SWIFT格式配对顺序判断任务数据集
    generate_pairwise_order_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.seed
    )