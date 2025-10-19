import os
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

def get_date_diff(date_str1, date_str2):
    """计算两个日期字符串 ('YYYY-MM-DD') 之间的天数差"""
    try:
        d1 = datetime.strptime(date_str1, '%Y-%m-%d')
        d2 = datetime.strptime(date_str2, '%Y-%m-%d')
        return abs((d2 - d1).days)
    except ValueError:
        # 简化错误处理，避免过多打印
        return -1

def generate_interval_extremum_identification_mcq(train_json_path, test_json_path, output_dir, random_seed=42):
    """
    为连续跨度识别 (interval_extremum_identification) 任务生成MS-SWIFT格式数据集 (选择题版本)。
    此版本已适配4张图片的数据集。
    """
    print("开始为 TSU 任务 'interval_extremum_identification' (选择题版, 4图版) 生成数据集...")
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 新增：定义目标样本数量 ---
    TARGET_COUNTS = {"train": 800, "test": 200}
    
    prompt_template = (
        "You are given four pictures in chronological order.\n\n"
        "Your task is to identify which adjacent pair has the {extremum_type} time span.\n"
        "A. Between Picture 1 and Picture 2\n"
        "B. Between Picture 2 and Picture 3\n"
        "C. Between Picture 3 and Picture 4\n\n"
        "Please answer with the corresponding letter only."
    )
    
    for split, json_path in [("train", train_json_path), ("test", test_json_path)]:
        print(f"\n正在处理 {split} 集...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except FileNotFoundError:
            print(f"错误: 找不到输入文件 {json_path}。请确保文件路径正确。")
            continue
        
        image_groups = dataset.get("image_groups", [])
        task_data = []
        
        # --- 以下部分保持不变，用于生成所有可能的样本 ---
        for group in tqdm(image_groups, desc=f"处理 {split} 图像组"):
            images = group.get("images", [])
            
            if len(images) < 4:
                continue
            
            selected_images = images[:4]
            
            paths = [img['path'] for img in selected_images]
            dates = [img['date'] for img in selected_images]
            
            spans = [get_date_diff(dates[i], dates[i+1]) for i in range(len(dates) - 1)]
            
            if any(s < 0 for s in spans) or all(s == 0 for s in spans):
                continue

            extremum_type = random.choice(["maximum", "minimum"])
            
            if extremum_type == "maximum":
                target_span_value = max(spans)
            else:
                non_zero_spans = [s for s in spans if s > 0]
                if not non_zero_spans:
                    continue
                target_span_value = min(non_zero_spans)

            if spans.count(target_span_value) > 1:
                continue

            target_index = spans.index(target_span_value)
            response = chr(ord('A') + target_index)

            image_prompt = (
                "Picture 1: <image> Picture 2: <image> Picture 3: <image> "
                "Picture 4: <image>"
            )
            
            final_prompt_text = prompt_template.format(extremum_type=extremum_type)
            
            swift_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": image_prompt.strip() + "\n\n" + final_prompt_text
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ],
                "images": paths
            }
            task_data.append(swift_sample)
            
        # --- 新增：最终采样逻辑 ---
        target_count = TARGET_COUNTS[split]
        print(f"从 {split} 集中生成了 {len(task_data)} 个合格样本。")
        
        final_task_data = task_data
        if len(task_data) > target_count:
            print(f"样本数超过目标 {target_count}，将随机采样至目标数量。")
            random.shuffle(final_task_data)
            final_task_data = final_task_data[:target_count]
        elif len(task_data) < target_count:
             print(f"警告: 样本数 ({len(task_data)}) 少于目标数 ({target_count})。将使用所有可用样本。")
            
        output_path = Path(output_dir) / f"interval_extremum_identification_mcq_4img_swift_{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_task_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功生成4图版任务集: {output_path}")
        print(f"包含 {len(final_task_data)} 个样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为TSU-连续跨度识别任务(4图版)生成选择题格式的MS-SWIFT数据集')
    parser.add_argument('--train_json', type=str, default="/data/ss/benchmark/json/TSU/train_with_dates.json",
                        help='包含日期信息的训练集JSON文件路径')
    parser.add_argument('--test_json', type=str, default="/data/ss/benchmark/json/TSU/test_with_dates.json",
                        help='包含日期信息的测试集JSON文件路径')
    parser.add_argument('--output_dir', type=str, default="/data/ss/benchmark/json/TSU/task_datasets",
                        help='输出任务JSON文件的目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    generate_interval_extremum_identification_mcq(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.seed
    )