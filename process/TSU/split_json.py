import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def create_tsu_initial_datasets(dataset_root, output_dir, total_samples=1000, train_split=0.8, random_seed=42):
    """
    遍历TAMMs数据集目录，提取图像路径和日期，生成用于TSU任务的初始训练集和测试集JSON文件。

    参数:
        dataset_root: TAMMs数据集的根目录 (包含airport_hangar, space_facility等文件夹)。
        output_dir: 输出JSON文件的目录。
        total_samples: 抽取的总样本数。
        train_split: 训练集所占的比例。
        random_seed: 随机种子。
    """
    print("开始扫描数据集...")
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    all_image_groups = []
    
    # 使用 pathlib.Path 兼容不同操作系统
    root_path = Path(dataset_root)

    # tqdm 用于显示进度条
    # 我们要找的是包含 .jpg 文件的最底层目录，这些就是我们的“观测点”
    group_dirs = []
    for dirpath, _, filenames in os.walk(root_path):
        # 如果当前目录直接包含 .jpg 文件，我们认为它是一个 "image_group"
        if any(f.endswith('.jpg') for f in filenames):
            group_dirs.append(Path(dirpath))

    print(f"找到了 {len(group_dirs)} 个潜在的图像组。正在处理...")

    for group_path in tqdm(group_dirs, desc="处理图像组"):
        current_group_images = []
        
        # 查找当前组目录下的所有jpg文件
        jpg_files = list(group_path.glob('*.jpg'))
        
        for img_path in jpg_files:
            json_path = img_path.with_suffix('.json')
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    # 从 timestamp 中提取日期部分 (YYYY-MM-DD)
                    timestamp = meta_data.get("timestamp")
                    if timestamp:
                        date_str = timestamp.split('T')[0]
                        
                        # 存储相对路径或绝对路径，这里使用绝对路径转字符串
                        current_group_images.append({
                            "path": str(img_path.resolve()),
                            "date": date_str
                        })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"警告: 无法处理文件 {json_path} 或缺少 'timestamp' 键. 错误: {e}")
            else:
                print(f"警告: 找不到对应的JSON文件: {json_path}")

        # 确保一个组里有足够多的图片（例如，至少4张，因为有任务需要）
        if len(current_group_images) >= 4:
            # 按日期排序，这对于后续任务很方便
            current_group_images.sort(key=lambda x: x['date'])
            all_image_groups.append({
                "group_id": group_path.name,
                "images": current_group_images
            })

    print(f"成功处理了 {len(all_image_groups)} 个有效的图像组。")
    
    # --- 抽样和分割 ---
    if len(all_image_groups) < total_samples:
        print(f"警告: 有效图像组数量 ({len(all_image_groups)}) 少于目标样本数 ({total_samples})。将使用所有有效组。")
        total_samples = len(all_image_groups)

    # 随机打乱所有组
    random.shuffle(all_image_groups)
    
    # 抽取指定数量的样本
    sampled_groups = all_image_groups[:total_samples]
    
    # 分割训练集和测试集
    split_index = int(total_samples * train_split)
    train_groups = sampled_groups[:split_index]
    test_groups = sampled_groups[split_index:]
    
    print(f"数据集分割完成: {len(train_groups)} 个训练样本, {len(test_groups)} 个测试样本。")

    # --- 保存文件 ---
    train_output_path = Path(output_dir) / "train_with_dates.json"
    test_output_path = Path(output_dir) / "test_with_dates.json"
    
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump({"image_groups": train_groups}, f, indent=2, ensure_ascii=False)
    print(f"训练集描述文件已保存到: {train_output_path}")

    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump({"image_groups": test_groups}, f, indent=2, ensure_ascii=False)
    print(f"测试集描述文件已保存到: {test_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从TAMMs数据集生成TSU任务的初始JSON文件')
    parser.add_argument('--dataset_root', type=str, default="/openbayes/home/dataset_256x256_20250723_154123/TAMMs",
                        help='TAMMs数据集的根目录 (例如 D:\\dataset\\TAMMs\\TAMMs)')
    parser.add_argument('--output_dir', type=str, default="/openbayes/home/0917/json/TSU",
                        help='输出JSON文件的目录 (默认: tsu_initial_jsons)')
    parser.add_argument('--total_samples', type=int, default=1250,
                        help='要抽取的总样本数 (默认: 1000)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='训练集所占的比例 (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    create_tsu_initial_datasets(
        args.dataset_root,
        args.output_dir,
        args.total_samples,
        args.train_split,
        args.seed
    )
