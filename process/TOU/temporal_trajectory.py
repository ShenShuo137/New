import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def generate_temporal_reasoning_swift_datasets(train_json_path, test_json_path, output_dir, random_seed=42):
    """
    读取训练集和测试集JSON文件，生成时间推理任务（包括内插和外推）的MS-SWIFT格式数据集。
    此版本根据新的任务定义进行了修改：为给定的目标图片选择正确的时间点。
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
        
        for group in tqdm(image_groups, desc=f"处理 {split} 图像组"):
            images = group.get("images", [])
            
            if len(images) < 5:
                continue
            
            sorted_images = sorted(images, key=lambda p: os.path.basename(p))

            if len(sorted_images) > 10:
                seq_length = random.randint(5, 10)
                start_idx = random.randint(0, len(sorted_images) - seq_length)
                selected_images = sorted_images[start_idx:start_idx+seq_length]
            else:
                selected_images = sorted_images
            
            if len(selected_images) < 5:
                continue

            task_name = random.choice(["interpolation", "extrapolation"])
            task_info = next(t for t in task_types if t['name'] == task_name)

            # --- 修正点: 严格按照 "提供 1, 3, 5，从 2 或 4 里面选" 的逻辑重写内插任务 ---
            if task_name == "interpolation":
                # 提供位置 1, 3, 5 的图像作为上下文
                context_indices = [0, 2, 4]
                context_positions = [1, 3, 5]
                
                # 随机决定目标图片是位置2还是位置4
                is_pos_2 = random.choice([True, False])
                if is_pos_2:
                    target_image = selected_images[1] # 目标是位置2的图片
                    correct_pos = 2
                    distractor_pos = 4
                else:
                    target_image = selected_images[3] # 目标是位置4的图片
                    correct_pos = 4
                    distractor_pos = 2
                
                context_images = [selected_images[i] for i in context_indices]
                final_images = context_images + [target_image]

                options = [str(correct_pos), str(distractor_pos)]
                random.shuffle(options)
                correct_answer = chr(ord('A') + options.index(str(correct_pos)))
                
                prompt = task_info["prompt_template"].format(
                    n=len(final_images), # 总共4张图
                    context_n=len(context_images), # 3张上下文图
                    positions=", ".join(map(str, context_positions)),
                    target_image_idx=len(final_images),
                    option_A=options[0],
                    option_B=options[1]
                )

            else: # extrapolation
                # 此部分逻辑符合 "3上下文+1目标" 的要求，保持不变
                context_indices = [1, 2, 3]
                context_positions = [2, 3, 4]

                predict_future = random.choice([True, False])
                
                if predict_future:
                    target_image = selected_images[4] # 目标是位置5
                    correct_pos = 5
                    distractor_pos = 1
                else:
                    target_image = selected_images[0] # 目标是位置1
                    correct_pos = 1
                    distractor_pos = 5

                context_images = [selected_images[i] for i in context_indices]
                final_images = context_images + [target_image]

                options = [str(correct_pos), str(distractor_pos)]
                random.shuffle(options)
                correct_answer = chr(ord('A') + options.index(str(correct_pos)))

                prompt = task_info["prompt_template"].format(
                    n=len(final_images), # 总共4张图
                    context_n=len(context_images), # 3张上下文图
                    positions=", ".join(map(str, context_positions)),
                    target_image_idx=len(final_images),
                    option_A=options[0],
                    option_B=options[1]
                )

            swift_sample = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": correct_answer}
                ],
                "images": final_images
            }
            reasoning_data.append(swift_sample)
        
        target_count = TARGET_COUNTS.get(split)
        if target_count is None:
            final_reasoning_data = reasoning_data
        else:
            print(f"从 {split} 集中生成了 {len(reasoning_data)} 个合格样本。")
            final_reasoning_data = reasoning_data
            if len(reasoning_data) > target_count:
                print(f"样本数超过目标 {target_count}，将随机采样至目标数量。")
                random.shuffle(final_reasoning_data)
                final_reasoning_data = final_reasoning_data[:target_count]
            elif len(reasoning_data) < target_count:
                print(f"警告: 样本数 ({len(reasoning_data)}) 少于目标数 ({target_count})。将使用所有可用样本。")
        
        output_path = Path(output_dir) / f"temporal_trajectory_swift_{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_reasoning_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功生成时间推理任务({split}集): {output_path}")
        print(f"包含 {len(final_reasoning_data)} 个样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为时间推理任务生成MS-SWIFT格式数据集')
    parser.add_argument('--train_json', type=str, default="/openbayes/home/0917/json/TOU/train_dataset.json")
    parser.add_argument('--test_json', type=str, default="/openbayes/home/0917/json/TOU/test_dataset.json")
    parser.add_argument('--output_dir', type=str, default="/openbayes/home/0917/json/TOU/task_datasets")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_temporal_reasoning_swift_datasets(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.seed
    )