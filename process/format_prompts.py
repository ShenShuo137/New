# python ~/benchmark/process/format_prompts.py ~/benchmark/json/
import os
import json
import argparse
import shutil
import datetime

def process_json_files(directory: str, backup: bool):
    """
    统一修改目录下所有 _train.json 和 _test.json 文件中的prompt格式。

    规则: 将包含 "Picture 1: <image>..." 的占位符行替换为纯粹的 "<image><image>..." 堆叠。
    这样做可以为所有任务创建一个统一的数据源格式，同时满足MS-SWIFT微调和InternVL推理的需求。
    文件将被直接覆盖，但提供备份选项。
    """
    
    # 1. 创建备份目录 (如果需要)
    backup_dir = None
    if backup:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(directory, f"backup_originals_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        print(f"🗂️ 备份已启用。原始文件将备份至: {backup_dir}")

    # 2. 遍历目录
    for root, _, files in os.walk(directory):
        if backup_dir and os.path.samefile(root, backup_dir):
            continue

        for filename in files:
            # 同时匹配 _train.json 和 _test.json
            if not (filename.endswith("_test.json") or filename.endswith("_train.json")):
                continue

            file_path = os.path.join(root, filename)
            print(f"\nProcessing: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ❌ 错误: 无法读取或解析文件。详情: {e}")
                continue

            modified = False
            
            # 3. 遍历文件中的每一个样本
            for sample in data:
                if not isinstance(sample, dict) or "messages" not in sample or len(sample["messages"]) == 0:
                    continue
                
                original_content = sample["messages"][0].get("content", "")
                
                # 安全地分离占位符行和问题主体
                if '\n\n' not in original_content:
                    continue

                header, body = original_content.split('\n\n', 1)
                
                # --- 核心修改：对所有文件应用统一规则 ---
                if "<image>" in header and "Picture" in header: # 确保只处理未被修改过的文件
                    image_count = header.count("<image>")
                    if image_count > 0:
                        # 生成 <image> 堆叠
                        new_header = "<image>" * image_count
                        new_content = new_header + "\n\n" + body
                        
                        sample["messages"][0]["content"] = new_content
                        modified = True
                        # 使用统一的日志信息
                        print(f"  - [统一格式] 将占位符替换为 {image_count}x'<image>'.")
                # ----------------------------------------------

            # 4. 如果有修改，则执行备份和覆盖
            if modified:
                try:
                    if backup:
                        shutil.copy2(file_path, backup_dir)
                        print(f"  - ✅ 已备份原始文件。")
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"  - ✅ 已成功修改并覆盖文件。")

                except Exception as e:
                    print(f"  ❌ 错误: 写入或备份文件时失败。详情: {e}")
            else:
                print("  - ⚠️ 文件内容无需修改，已跳过。")


    print("\n\n处理完成！所有文件均已更新为统一的 '<image>' 堆叠格式。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        统一修改所有 *_train.json 和 *_test.json 文件中的Prompt格式。
        规则: 将 'Picture N: <image>...' 行替换为纯粹的 '<image>' 堆叠格式，
        以创建一个兼容多种模型的标准数据集。
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'directory', 
        type=str, 
        help='要处理的JSON文件所在的根目录。脚本将递归遍历此目录。'
    )
    
    parser.add_argument(
        '--no-backup', 
        action='store_true', 
        help='禁用备份功能。强烈建议首次运行时保留备份。'
    )
    
    args = parser.parse_args()
    
    # 在执行前，建议先将之前的修改还原
    print("--- 准备开始统一化Prompt格式 ---")
    print(f"目标目录: {args.directory}")
    print(f"启用备份: {not args.no_backup}")
    print("本脚本会将所有 *_train.json 和 *_test.json 文件中的 'Picture N: <image>'")
    print("行统一替换为 '<image><image>...' 格式。")
    print("-" * 50)
    
    process_json_files(args.directory, not args.no_backup)
