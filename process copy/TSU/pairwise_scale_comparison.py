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
        print(f"日期格式错误: {date_str1} 或 {date_str2}")
        return -1

def generate_pairwise_scale_comparison_swift_datasets_mcq(train_json_path, test_json_path, output_dir, random_seed=42):
    """
    为时间跨度比较任务生成MS-SWIFT格式数据集 (选择题版本)。
    """
    print("开始为 TSU 任务 'pairwise_scale_comparison' (选择题版) 生成数据集...")
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # 新的Prompt模板，包含选项和占位符要求
    prompt_template = (
        "You are given four pictures, presented in their correct chronological order.\n\n"
        "Your task is to compare the time span between the first two pictures (Picture 1 and Picture 2) "
        "and the time span between the last two pictures (Picture 3 and Picture 4).\n\n"
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
        
        image_groups = dataset.get("image_groups", [])
        task_data = []
        
        for group in tqdm(image_groups, desc=f"处理 {split} 图像组"):
            images = group.get("images", [])
            
            if len(images) < 4:
                continue
            
            start_index = random.randint(0, len(images) - 4)
            selected_images = images[start_index : start_index + 4]
            
            paths = [img['path'] for img in selected_images]
            dates = [img['date'] for img in selected_images]
            
            span1 = get_date_diff(dates[0], dates[1])
            span2 = get_date_diff(dates[2], dates[3])
            
            if span1 < 0 or span2 < 0 or span1 == span2:
                continue
            
            # 确定答案为 "A" 或 "B"
            response = "A" if span1 > span2 else "B"
            
            # 构建带占位符的图片提示
            image_prompt = "Picture 1: <image> Picture 2: <image> Picture 3: <image> Picture 4: <image> "
            
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
                "images": paths
            }
            task_data.append(swift_sample)
            
        output_path = Path(output_dir) / f"pairwise_scale_comparison_mcq_swift_{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功生成选择题版任务集: {output_path}")
        print(f"包含 {len(task_data)} 个样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为TSU-跨度比较任务生成选择题格式的MS-SWIFT数据集')
    parser.add_argument('--train_json', type=str, default="tsu_initial_jsons/train_with_dates.json",
                        help='包含日期信息的训练集JSON文件路径')
    parser.add_argument('--test_json', type=str, default="tsu_initial_jsons/test_with_dates.json",
                        help='包含日期信息的测试集JSON文件路径')
    parser.add_argument('--output_dir', type=str, default="task_datasets",
                        help='输出任务JSON文件的目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    generate_pairwise_scale_comparison_swift_datasets_mcq(
        args.train_json,
        args.test_json,
        args.output_dir,
        args.seed
    )