import os
import json
import glob
from pathlib import Path

def generate_dataset_json(dataset_root_path, output_json_path):
    """
    生成包含所有图像组及其图像路径的JSON文件
    
    参数:
        dataset_root_path: 数据集的根目录路径
        output_json_path: 输出JSON文件的路径
    """
    # 转换为绝对路径以确保一致性
    dataset_root_path = os.path.abspath(dataset_root_path)
    
    # 存储所有图像组的列表
    image_groups = []
    
    # 获取所有子文件夹(如data-xiongan-withsum等)
    subdirs = [d for d in os.listdir(dataset_root_path) 
               if os.path.isdir(os.path.join(dataset_root_path, d))]
    
    # 遍历每个子文件夹
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_root_path, subdir)
        
        # 获取子文件夹中的所有序号文件夹
        number_dirs = [d for d in os.listdir(subdir_path) 
                      if os.path.isdir(os.path.join(subdir_path, d))]
        
        # 遍历每个序号文件夹
        for number_dir in number_dirs:
            number_dir_path = os.path.join(subdir_path, number_dir)
            
            # 获取所有图像文件(假设是png格式，如有其他格式可以添加)
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(number_dir_path, ext)))
            
            # 过滤掉非图像文件(如txt文件)
            image_files = [f for f in image_files if not f.lower().endswith('.txt')]
            
            # 如果有图像文件，则创建一个图像组
            if image_files:
                # 计算相对于dataset根目录的路径作为group_id
                rel_path = os.path.relpath(number_dir_path, dataset_root_path)
                
                # 创建图像组对象
                group = {
                    "group_id": rel_path,
                    "images": sorted(image_files)  # 排序以确保顺序一致
                }
                
                image_groups.append(group)
    
    # 创建最终的JSON对象
    dataset_json = {
        "image_groups": image_groups
    }
    
    # 写入JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
    
    print(f"已生成JSON文件: {output_json_path}")
    print(f"共包含 {len(image_groups)} 个图像组")

if __name__ == "__main__":
    # 设置数据集根目录和输出JSON文件路径
    # 根据您的描述，路径可能会变化，所以这里使用相对路径
    dataset_root = "/data/ss/benchmark/dataset/dataset/dataset"  # 您可以根据实际情况修改此路径
    output_json = "/data/ss/benchmark/json/TOU/dataset_full.json"
    
    generate_dataset_json(dataset_root, output_json)