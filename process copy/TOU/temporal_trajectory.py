import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def generate_temporal_reasoning_swift_datasets(train_json_path, test_json_path, output_dir, random_seed=42):
    """
    读取训练集和测试集JSON文件，生成时间推理任务的MS-SWIFT格式数据集。
    此版本添加了与图像对应的<image>占位符到prompt中。
    """
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    task_types = [
        {
            "name": "interpolation",
            "prompt_template": (
                "You are given {n} images. The first {context_n} images (Image 1 to Image {context_n}) correspond to positions {positions} in a complete, evenly spaced time sequence.\n\n"
                "The last image (Image {target_image_idx}) is from a time point missing from the initial sequence.\n\n"
                "Which position does the last image belong to?\n"
                "A. {option_A}\n"
                "B. {option_B}\n\n"
                "Please answer with the corresponding letter only."
            )
        },
        {
            "name": "extrapolation",
            "prompt_template": (
                "You are given {n} images. The first {context_n} images (Image 1 to Image {context_n}) correspond to positions {positions} in a complete, evenly spaced time sequence.\n\n"
                "The last image (Image {target_image_idx}) is from a time point either before or after this sequence.\n\n"
                "Which position does the last image belong to?\n"
                "A. {option_A}\n"
                "B. {option_B}\n\n"
                "Please answer with the corresponding letter only."
            )
        }
    ]
    
    TARGET_COUNTS = {"train": 800, "test": 200}
    
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        print(f"\n正在处理 {split} 集...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except FileNotFoundError:
            print(f"错误: 找不到输入文件 {json_path}。")
            continue
        
        image_groups = dataset.get("image_groups", [])
        reasoning_data = []
        
        # 使用tqdm显示进度条
        pbar = tqdm(total=TARGET_COUNTS.get(split, len(image_groups)), desc=f"生成 {split} 样本")
        
        # 持续生成直到满足数量要求或遍历完所有组
        while len(reasoning_data) < TARGET_COUNTS.get(split, float('inf')) and image_groups:
            # 为了能pop，需要先打乱
            if len(reasoning_data) == 0:
                 random.shuffle(image_groups)

            if not image_groups: break
            group = image_groups.pop()

            images = group.get("images", [])
            
            if len(images) < 5:
                continue
            
            sorted_images = sorted(images, key=lambda p: os.path.basename(p))

            seq_length = random.randint(5, min(len(sorted_images), 10))
            start_idx = random.randint(0, len(sorted_images) - seq_length)
            selected_images = sorted_images[start_idx:start_idx+seq_length]
            
            if len(selected_images) < 5:
                continue

            task_name = random.choice(["interpolation", "extrapolation"])
            task_info = next(t for t in task_types if t['name'] == task_name)

            if task_name == "interpolation":
                context_indices = [0, 2, 4]
                context_positions = [1, 3, 5]
                
                is_pos_2 = random.choice([True, False])
                target_image = selected_images[1] if is_pos_2 else selected_images[3]
                correct_pos = 2 if is_pos_2 else 4
                distractor_pos = 4 if is_pos_2 else 2
                
                context_images = [selected_images[i] for i in context_indices]
                final_images = context_images + [target_image]

                options = [str(correct_pos), str(distractor_pos)]
                random.shuffle(options)
                correct_answer = chr(ord('A') + options.index(str(correct_pos)))
                
                text_prompt = task_info["prompt_template"].format(
                    n=len(final_images),
                    context_n=len(context_images),
                    positions=", ".join(map(str, context_positions)),
                    target_image_idx=len(final_images),
                    option_A=options[0],
                    option_B=options[1]
                )

            else: # extrapolation
                context_indices = [1, 2, 3]
                context_positions = [2, 3, 4]

                predict_future = random.choice([True, False])
                target_image = selected_images[4] if predict_future else selected_images[0]
                correct_pos = 5 if predict_future else 1
                distractor_pos = 1 if predict_future else 5

                context_images = [selected_images[i] for i in context_indices]
                final_images = context_images + [target_image]

                options = [str(correct_pos), str(distractor_pos)]
                random.shuffle(options)
                correct_answer = chr(ord('A') + options.index(str(correct_pos)))

                text_prompt = task_info["prompt_template"].format(
                    n=len(final_images),
                    context_n=len(context_images),
                    positions=", ".join(map(str, context_positions)),
                    target_image_idx=len(final_images),
                    option_A=options[0],
                    option_B=options[1]
                )
            
            # ==================== MODIFICATION START ====================
            # 1. 创建占位符。注意使用 "Image" 而不是 "Picture" 来匹配prompt文本
            placeholder_parts = [f"Image {i+1}: <image>" for i in range(len(final_images))]
            image_placeholders = " ".join(placeholder_parts)

            # 2. 将占位符与文本提示结合，形成最终的用户输入
            final_user_prompt = image_placeholders + "\n\n" + text_prompt
            
            # 3. 构造与之前脚本完全兼容的MS-SWIFT格式
            swift_sample = {
                "messages": [
                    {"role": "user", "content": final_user_prompt},
                    {"role": "assistant", "content": correct_answer}
                ],
                "images": final_images
            }
            # ===================== MODIFICATION END =====================
            
            reasoning_data.append(swift_sample)
            pbar.update(1)

        pbar.close()

        # 检查最终数量并保存
        target_count = TARGET_COUNTS.get(split)
        final_data = reasoning_data
        if target_count and len(final_data) > target_count:
            final_data = final_data[:target_count] # 如果超出，截取到目标数量
        
        if not final_data:
             print(f"警告: 未能为 {split} 集生成任何样本。请检查输入数据和逻辑。")
             continue

        output_path = Path(output_dir) / f"temporal_trajectory_swift_{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功生成时间推理任务({split}集): {output_path}")
        print(f"包含 {len(final_data)} 个样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为时间推理任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/data/ss/benchmark/json/TOU/train_dataset.json")
    parser.add_argument('--test_json', type=str, default="/data/ss/benchmark/json/TOU/test_dataset.json")
    parser.add_argument('--output_dir', type=str, default="/data/ss/benchmark/json/TOU/task_datasets")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_temporal_reasoning_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.seed
    )