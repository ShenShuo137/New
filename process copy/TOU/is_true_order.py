import os
import json
import random
import argparse
from pathlib import Path

def generate_is_true_order_swift_datasets(train_json_path, test_json_path, output_dir, min_images=3, max_images=7, random_seed=42):
    """
    读取训练集和测试集JSON文件，为序列合理性判别任务生成MS-SWIFT格式的数据集。
    此版本确保了代码风格的统一性。
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
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        image_groups = dataset.get("image_groups", [])
        
        is_true_order_data = []
        
        for group in image_groups:
            images = group.get("images", [])
            
            if len(images) < min_images:
                continue
            
            num_images = random.randint(min_images, min(max_images, len(images)))
            selected_images = random.sample(images, num_images)
            
            image_filenames = [os.path.basename(img) for img in selected_images]
            true_order_indices = sorted(range(len(selected_images)), key=lambda i: image_filenames[i])
            ordered_images = [selected_images[i] for i in true_order_indices]
            
            # 50%的概率使用正确顺序，50%的概率使用打乱的顺序
            if random.random() > 0.5:
                final_images = ordered_images
                correct_answer = "True"
            else:
                shuffled_images = ordered_images[:]
                # 确保打乱后的顺序与原始顺序不同
                while shuffled_images == ordered_images:
                    random.shuffle(shuffled_images)
                final_images = shuffled_images
                correct_answer = "False"
            
            # ==================== STYLE REFINEMENT START ====================
            # 使用列表推导式和.join()方法，与前一个脚本风格保持一致
            placeholder_parts = [f"Picture {i+1}: <image>" for i in range(len(final_images))]
            image_placeholders = " ".join(placeholder_parts)
            
            # 组合成最终的用户输入
            final_user_prompt = image_placeholders + "\n\n" + prompt_template
            # ===================== STYLE REFINEMENT END =====================

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
            
            is_true_order_data.append(swift_sample)
        
        # 保存MS-SWIFT格式数据集
        output_path = os.path.join(output_dir, f"is_true_order_swift_{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(is_true_order_data, f, indent=2, ensure_ascii=False)
        
        print(f"生成MS-SWIFT格式序列合理性判别任务{split}集: {output_path}")
        print(f"包含{len(is_true_order_data)}个样本")

if __name__ == "__main__":
    # 命令行参数部分无需修改
    parser = argparse.ArgumentParser(description='为序列合理性判别任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/data/ss/benchmark/json/TOU/train_dataset.json",
                        help='训练集JSON文件路径 (默认: train_dataset.json)')
    parser.add_argument('--test_json', type=str, default="/data/ss/benchmark/json/TOU/test_dataset.json",
                        help='测试集JSON文件路径 (默认: test_dataset.json)')
    parser.add_argument('--output_dir', type=str, default="/data/ss/benchmark/json/TOU/task_datasets",
                        help='输出目录 (默认: task_datasets)')
    parser.add_argument('--min_images', type=int, default=3,
                        help='每个任务实例最少使用的图片数量 (默认: 3)')
    parser.add_argument('--max_images', type=int, default=7,
                        help='每个任务实例最多使用的图片数量 (默认: 7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    generate_is_true_order_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.min_images,
        args.max_images,
        args.seed
    )