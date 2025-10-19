import os
import json
import random
import argparse
from pathlib import Path

def generate_anomaly_detection_swift_datasets_overwrite(
    train_json_path, 
    test_json_path, 
    output_dir, 
    num_images=5, 
    random_seed=42
):
    """
    读取训练集和测试集JSON，为异常图像检测任务生成MS-SWIFT格式的数据集。
    *** 此版本确保每个样本都有一个异常，从而完全移除'None'答案，并直接覆盖旧文件。***
    """
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # 更新后的、更确定的任务指令
    prompt_template = (
        "I've shown you {num_images} images of locations taken over time.\n\n"
        "In a normal temporal sequence, all images should show the same location as it changes over time.\n"
        "However, one of these images is an anomaly - it either shows a different location or was taken at an inconsistent time point that breaks the logical progression.\n\n"
        "Please identify the anomalous image that doesn't belong in this temporal sequence.\n\n"
        "Answer with the number of the anomalous image (1 to {num_images})."
    )
    
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        image_groups = dataset.get("image_groups", [])
        valid_groups = [group for group in image_groups if len(group.get("images", [])) >= num_images]
        
        if len(valid_groups) < 2:
            print(f"❌ 错误: 在 {split} 集中没有足够的图像组 (至少需要2组) 来生成此任务!")
            continue
            
        anomaly_data = []
        
        for group_idx, group in enumerate(valid_groups):
            images = group.get("images", [])
            
            # 1. 创建一个有序的正常序列作为基础
            selected_images = random.sample(images, num_images)
            image_filenames = [os.path.basename(img) for img in selected_images]
            sorted_indices = sorted(range(len(selected_images)), key=lambda i: image_filenames[i])
            final_images = [selected_images[i] for i in sorted_indices]
            
            # 2. 强制引入一个异常
            anomaly_position_to_replace = random.randint(0, num_images - 1)
            correct_answer = str(anomaly_position_to_replace + 1)
            
            other_groups = valid_groups.copy()
            other_groups.pop(group_idx)
            
            anomaly_group = random.choice(other_groups)
            if not anomaly_group.get("images", []):
                continue
            anomaly_image = random.choice(anomaly_group.get("images", []))
            
            # 3. 在选定位置用异常图片替换正常图片
            final_images[anomaly_position_to_replace] = anomaly_image
            
            # 4. 构建样本格式
            placeholder_parts = [f"Picture {i+1}: <image>" for i in range(len(final_images))]
            image_placeholders = " ".join(placeholder_parts)
            text_prompt = prompt_template.format(num_images=len(final_images))
            final_user_prompt = image_placeholders + "\n\n" + text_prompt
            
            swift_sample = {
                "messages": [
                    {"role": "user", "content": final_user_prompt},
                    {"role": "assistant", "content": correct_answer}
                ],
                "images": final_images
            }
            
            anomaly_data.append(swift_sample)
        
        # --- MODIFICATION: 恢复原始文件名以实现覆盖 ---
        output_filename = f"anomaly_detection_swift_{split}.json"
        output_path = os.path.join(output_dir, output_filename)
        # ----------------------------------------------
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anomaly_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已生成并覆盖了MS-SWIFT格式异常检测任务 {split} 集: {output_path}")
        print(f"   包含 {len(anomaly_data)} 个样本 (所有样本均已修正，不含'None'答案)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为异常图像检测任务生成不含'None'答案的MS-SWIFT格式数据集，并覆盖旧文件。")
    parser.add_argument('--train_json', type=str, default="/data/ss/benchmark/json/TOU/train_dataset.json", help='训练集JSON文件路径')
    parser.add_argument('--test_json', type=str, default="/data/ss/benchmark/json/TOU/test_dataset.json", help='测试集JSON文件路径')
    parser.add_argument('--output_dir', type=str, default="/data/ss/benchmark/json/TOU/task_datasets", help='输出目录')
    parser.add_argument('--num_images', type=int, default=5, help='每个任务实例使用的图片数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    generate_anomaly_detection_swift_datasets_overwrite(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.num_images,
        args.seed
    )