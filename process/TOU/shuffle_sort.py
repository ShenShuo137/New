import os
import json
import random
import argparse
from pathlib import Path

def generate_shuffle_sort_swift_datasets(train_json_path, test_json_path, output_dir, num_images=5, random_seed=42):
    """
    读取训练集和测试集JSON文件，为乱序重排任务生成MS-SWIFT格式的数据集。
    此版本重构了核心逻辑，修复了硬编码的prompt，并统一了代码风格。
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== BUG FIX & REFINEMENT START ====================
    # 1. 修复硬编码Bug: 将prompt中的数字 "5" 替换为可格式化的占位符 {num_images}
    prompt_template = (
        "You are given {num_images} pictures in random order: {picture_list}.\n\n"
        "Your task is to determine their correct chronological order based on visual content.\n\n"
        "Please output {num_images} integers representing the correct temporal order of the pictures.\n\n"
        "Each integer should be between 1 and {num_images}, indicating the index of the picture in the original input list "
        "({index_mapping}).\n\n"
        "The output should be a permutation of 1 to {num_images}, indicating the order in which the input pictures occurred in time.\n\n"
        "For example, if the correct chronological order is: Picture 4, Picture 2, Picture 3, Picture 1, Picture 5, "
        "output only like this: `4 2 3 1 5`. Do not provide any explanations or additional text."
    )
    # ===================== BUG FIX & REFINEMENT END =====================
    
    # 处理训练集和测试集
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        image_groups = dataset.get("image_groups", [])
        
        shuffle_sort_data = []
        
        for group in image_groups:
            images = group.get("images", [])
            
            if len(images) < num_images:
                continue
            
            # ==================== LOGIC REFACTORING START ====================
            # 重构核心逻辑，使其更清晰
            
            # 1. 随机选择图片，并立刻按文件名确定它们的“真实”时间顺序
            selected_images = random.sample(images, num_images)
            ordered_images = sorted(selected_images, key=lambda img: os.path.basename(img))
            
            # 2. 创建一个打乱后的副本作为给模型的输入，并确保它与真实顺序不同
            shuffled_images = ordered_images[:]
            while shuffled_images == ordered_images:
                random.shuffle(shuffled_images)

            # 3. 计算正确答案 (新逻辑)
            #   a. 创建一个映射，记录每一张图片在“打乱后”输入列表中的位置 (1-based index)
            #      例如: { 'C.jpg': 1, 'A.jpg': 2, 'B.jpg': 3 }
            shuffled_pos_map = {img_path: i + 1 for i, img_path in enumerate(shuffled_images)}
            
            #   b. 遍历“真实顺序”列表，从映射中查找每张图片在打乱后列表中的位置
            #      例如: 真实的顺序是 A, B, C. 它们在打乱后列表中的位置是 2, 3, 1
            answer_list = [shuffled_pos_map[img] for img in ordered_images]
            
            #   c. 将答案转换为模型期望的字符串格式
            response = " ".join(map(str, answer_list))
            # ===================== LOGIC REFACTORING END =====================

            # 统一代码风格: 使用.join()方法创建图片占位符
            placeholder_parts = [f"Picture {i+1}: <image>" for i in range(len(shuffled_images))]
            image_placeholders = " ".join(placeholder_parts)
            
            # 动态填充Prompt模板
            picture_list_str = ", ".join([f"Picture {i+1}" for i in range(num_images)])
            index_mapping_str = ", ".join([f"{i+1} = Picture {i+1}" for i in range(num_images)])
            formatted_prompt = prompt_template.format(
                num_images=num_images,
                picture_list=picture_list_str,
                index_mapping=index_mapping_str
            )

            # 组合成最终的用户输入
            final_user_prompt = image_placeholders + "\n\n" + formatted_prompt
            
            # 构造MS-SWIFT格式
            swift_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": final_user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ],
                "images": shuffled_images
            }
            
            shuffle_sort_data.append(swift_sample)
        
        output_path = os.path.join(output_dir, f"shuffle_sort_swift_{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(shuffle_sort_data, f, indent=2, ensure_ascii=False)
        
        print(f"生成MS-SWIFT格式乱序重排任务{split}集: {output_path}")
        print(f"包含{len(shuffle_sort_data)}个样本")

if __name__ == "__main__":
    # 命令行参数部分无需修改
    parser = argparse.ArgumentParser(description='为乱序重排任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/data/ss/benchmark/json/TOU/train_dataset.json",
                        help='训练集JSON文件路径 (默认: train_dataset.json)')
    parser.add_argument('--test_json', type=str, default="/data/ss/benchmark/json/TOU/test_dataset.json",
                        help='测试集JSON文件路径 (默认: test_dataset.json)')
    parser.add_argument('--output_dir', type=str, default="/data/ss/benchmark/json/TOU/task_datasets",
                        help='输出目录 (默认: task_datasets)')
    parser.add_argument('--num_images', type=int, default=5,
                        help='每个任务实例使用的图片数量 (默认: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    generate_shuffle_sort_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.num_images,
        args.seed
    )