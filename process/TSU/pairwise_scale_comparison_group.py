# 文件名: generate_pairwise_scale_comparison_cross_group.py

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
        # 简化错误处理
        return -1

def generate_pairwise_scale_comparison_cross_group_mcq(train_json_path, test_json_path, output_dir, random_seed=42):
    """
    为时间跨度比较任务生成MS-SWIFT格式数据集 (跨组选择题版本)。
    """
    print("开始为 TSU 任务 'pairwise_scale_comparison' (跨组选择题版) 生成数据集...")
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 新增：定义目标样本数量 ---
    TARGET_COUNTS = {"train": 800, "test": 200}
    
    prompt_template = (
        "You are given four pictures. The first two pictures are from one location, and the last two are from another.\n\n"
        "Your task is to compare the time span between the first pair of pictures (Picture 1 and Picture 2) "
        "and the time span between the second pair of pictures (Picture 3 and Picture 4).\n\n"
        "Which pair of pictures has a larger time span?\n"
        "A. The first pair (Picture 1 and Picture 2)\n"
        "B. The second pair (Picture 3 and Picture 4)\n\n"
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
        
        # 筛选出至少有2张图片的合格图像组
        eligible_groups = [g for g in dataset.get("image_groups", []) if len(g.get("images", [])) >= 2]
        if len(eligible_groups) < 2:
            print(f"警告: {split} 集中合格的图像组少于2个，无法生成跨组样本。")
            continue

        task_data = []
        # 我们生成与合格组数量相当的样本，以保持数据规模
        for group_A in tqdm(eligible_groups, desc=f"为 {split} 集生成跨组样本"):
            # 随机选择一个与 group_A 不同的 group_B
            group_B = random.choice(eligible_groups)
            while group_B['group_id'] == group_A['group_id']:
                group_B = random.choice(eligible_groups)

            # 从 group_A 中随机抽样一对图像
            pair_A = random.sample(group_A['images'], 2)
            pair_A.sort(key=lambda img: img['date']) # 按日期排序以正确计算跨度

            # 从 group_B 中随机抽样一对图像
            pair_B = random.sample(group_B['images'], 2)
            pair_B.sort(key=lambda img: img['date'])

            paths = [pair_A[0]['path'], pair_A[1]['path'], pair_B[0]['path'], pair_B[1]['path']]
            dates = [pair_A[0]['date'], pair_A[1]['date'], pair_B[0]['date'], pair_B[1]['date']]

            span1 = get_date_diff(dates[0], dates[1])
            span2 = get_date_diff(dates[2], dates[3])
            
            # 确保跨度有效且不相等，避免歧义
            if span1 <= 0 or span2 <= 0 or span1 == span2:
                continue
            
            response = "A" if span1 > span2 else "B"
            
            image_prompt = "Picture 1: <image> Picture 2: <image> Picture 3: <image> Picture 4: <image>"
            
            swift_sample = {
                "messages": [
                    { "role": "user", "content": image_prompt.strip() + "\n\n" + prompt_template },
                    { "role": "assistant", "content": response }
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
            
        output_path = Path(output_dir) / f"pairwise_scale_comparison_cross_group_mcq_swift_{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_task_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功生成跨组版任务集: {output_path}")
        print(f"包含 {len(final_task_data)} 个样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为TSU-跨度比较任务生成跨组选择题格式的MS-SWIFT数据集')
    parser.add_argument('--train_json', type=str, default="/data/ss/benchmark/json/TSU/train_with_dates.json")
    parser.add_argument('--test_json', type=str, default="/data/ss/benchmark/json/TSU/test_with_dates.json")
    parser.add_argument('--output_dir', type=str, default="/data/ss/benchmark/json/TSU/task_datasets")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_pairwise_scale_comparison_cross_group_mcq(args.train_json, args.test_json, args.output_dir, args.seed)