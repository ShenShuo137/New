import os
import json
import random
import argparse
from pathlib import Path

def generate_pairwise_order_swift_datasets(train_json_path, test_json_path, output_dir, random_seed=42):
    """
    读取训练集和测试集JSON文件，为配对顺序判断任务生成MS-SWIFT格式的数据集。
    此版本重构了核心逻辑以提高代码可读性。
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
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        image_groups = dataset.get("image_groups", [])
        
        pairwise_order_data = []
        
        for group in image_groups:
            images = group.get("images", [])
            
            if len(images) < 2:
                continue
            
            # ==================== LOGIC REFACTORING START ====================
            # 核心逻辑重构，使代码更清晰
            
            # 1. 随机选择2张不同的图像
            img1, img2 = random.sample(images, 2)
            
            # 2. 根据文件名确定它们的真实时间顺序
            #    我们总是能明确地找出哪张更早，哪张更晚
            if os.path.basename(img1) < os.path.basename(img2):
                earlier_image, later_image = img1, img2
            else:
                earlier_image, later_image = img2, img1

            # 3. 随机决定是生成一个正确顺序的样本，还是一个错误顺序的样本
            if random.random() > 0.5:
                # 生成正确顺序的样本 (早 -> 晚)
                # 这个顺序符合Prompt描述 "Picture 1 appears earlier than Picture 2"，所以答案是 True
                final_images = [earlier_image, later_image]
                correct_answer = "True"
            else:
                # 生成错误顺序的样本 (晚 -> 早)
                # 这个顺序不符合Prompt描述，所以答案是 False
                final_images = [later_image, earlier_image]
                correct_answer = "False"
            # ===================== LOGIC REFACTORING END =====================

            # 构建带编号的图片提示部分
            # 对于只有两张图的固定任务，直接写字符串比循环更高效
            image_placeholders = "Picture 1: <image> Picture 2: <image>"
            
            # 组合成最终的用户输入
            final_user_prompt = image_placeholders + "\n\n" + prompt_template
            
            # 构造MS-SWIFT格式
            swift_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": final_user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": correct_answer
                    }
                ],
                "images": final_images
            }
            
            pairwise_order_data.append(swift_sample)
        
        # 保存MS-SWIFT格式数据集
        output_path = os.path.join(output_dir, f"pairwise_order_swift_{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairwise_order_data, f, indent=2, ensure_ascii=False)
        
        print(f"生成MS-SWIFT格式配对顺序判断任务{split}集: {output_path}")
        print(f"包含{len(pairwise_order_data)}个样本")

if __name__ == "__main__":
    # 命令行参数部分无需修改
    parser = argparse.ArgumentParser(description='为配对顺序判断任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/data/ss/benchmark/json/TOU/train_dataset.json",
                        help='训练集JSON文件路径 (默认: train_dataset.json)')
    parser.add_argument('--test_json', type=str, default="/data/ss/benchmark/json/TOU/test_dataset.json",
                        help='测试集JSON文件路径 (默认: test_dataset.json)')
    parser.add_argument('--output_dir', type=str, default="/data/ss/benchmark/json/TOU/task_datasets",
                        help='输出目录 (默认: task_datasets)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    generate_pairwise_order_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.seed
    )