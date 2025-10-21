import os
import json
import re  # 导入正则表达式库
import torch
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

def set_random_seed(seed):
    """设置随机种子以确保结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_image_reordering_task(
    test_json_path,
    model_base_path,
    tsv_output_dir,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    评估模型在“图像乱序重排”任务上的表现。
    - 使用成对排序的正负比（PNR）作为核心评估指标。
    - 自动处理任意数量的图片。
    - 健壮地解析模型的数字序列输出。
    """
    # === 1. 加载模型 & 处理器 (无变化) ===
    print("🚀 开始加载模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )
    if lora_ckpt_path:
        print(f"🔄 正在从 {lora_ckpt_path} 加载并合并LoRA权重...")
        model = PeftModel.from_pretrained(model, lora_ckpt_path).merge_and_unload()
        print("✅ LoRA权重已成功合并。")
    processor = AutoProcessor.from_pretrained(model_base_path, min_pixels=256*28*28, max_pixels=1280*28*28)
    print("✅ 模型和处理器加载完成。")

    # === 2. 加载测试数据 (无变化) ===
    print(f"📚 正在从 {test_json_path} 加载测试数据...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ 成功加载 {len(test_data)} 条测试样本。")

    # === 3. 推理主循环 ===
    results = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for entry in tqdm(test_data, desc="🧠 Evaluating Image Reordering"):
        image_paths = entry.get("images", [])
        if not image_paths:
            print("⚠️ 警告: 跳过一个样本，因为它不包含任何图片。")
            continue

        try:
            # --- MODIFICATION 1: 直接从数据中读取prompt，而不是硬编码 ---
            prompt_text = entry["messages"][0]["content"]
            gt_answer = entry["messages"][1]["content"].strip()
        except (KeyError, IndexError) as e:
            print(f"⚠️ 警告: 跳过一个样本，因为其 'messages' 结构不正确。错误: {e}")
            continue

        messages = [{"role": "user", "content": [{"type": "image", "image": p} for p in image_paths] + [{"type": "text", "text": prompt_text}]}]

        try:
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, return_tensors="pt").to(device)
        except Exception as e:
            print(f"❌ 处理输入时发生错误，图片路径: {image_paths}。错误详情: {e}")
            continue

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
        
        model_output = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        
        # --- MODIFICATION 2: 使用正则表达式健壮地提取数字序列 ---
        # 无论模型输出 "The order is 4 2 3 1 5" 还是 "4 2 3 1 5"，都能正确提取
        predicted_numbers = re.findall(r'\d+', model_output)
        predicted_answer = " ".join(predicted_numbers)

        results.append({
            "ground_truth": gt_answer,
            "prediction": predicted_answer,
            "raw_output": model_output,
            "timestamp": timestamp_now,
            "images": "|".join(image_paths)
        })

    # === 4. 保存结果并计算PNR得分 ===
    if not results:
        print("❌ 评估结束，但没有有效的评估结果可以保存。")
        return None, 0.0

    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True)
    tsv_save_path = os.path.join(tsv_output_dir, f"image_reordering_eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    # --- MODIFICATION 3: 采用参考代码中正确且高效的PNR计算逻辑，并增加健壮性检查 ---
    total_pos, total_neg = 0, 0

    for _, row in df.iterrows():
        gt_str, pred_str = row["ground_truth"], row["prediction"]
        try:
            gt_order = [int(x) for x in gt_str.split()]
            pred_order = [int(x) for x in pred_str.split()]

            # 健壮性检查：如果预测结果不是一个正确的排列组合，则跳过此样本
            if sorted(pred_order) != sorted(gt_order):
                print(f"⚠️ 警告: 预测结果 '{pred_str}' 不是基准答案 '{gt_str}' 的有效排列，已跳过此样本的PNR计算。")
                # 对于无效的排列，可以认为所有对都是错误的
                num_pairs = len(gt_order) * (len(gt_order) - 1) // 2
                total_neg += num_pairs
                continue

            # 构建预测排名字典：{图片ID: 预测的顺序(0-based)}，用于快速查找
            pred_rank = {img_id: rank for rank, img_id in enumerate(pred_order)}

            # 遍历所有在正确顺序中的图像对 (a, b)，其中 a 在 b 之前
            for i in range(len(gt_order)):
                for j in range(i + 1, len(gt_order)):
                    img_a = gt_order[i]
                    img_b = gt_order[j]
                    
                    # 检查在预测的顺序中，a 是否仍在 b 之前
                    if pred_rank[img_a] < pred_rank[img_b]:
                        total_pos += 1  # 正序对
                    else:
                        total_neg += 1  # 逆序对
        except (ValueError, KeyError) as e:
            print(f"❌ 错误: 处理PNR计算时发生错误。基准: '{gt_str}', 预测: '{pred_str}'. 错误详情: {e}")
            continue

    # 计算全局PNR
    if total_neg == 0:
        global_pnr = float('inf') if total_pos > 0 else 0.0
    else:
        global_pnr = total_pos / total_neg

    print("\n" + "="*50)
    print(f"✅ 推理完成，结果已保存至：{tsv_save_path}")
    print(f"📊 全局 PNR (正序对 / 逆序对): {global_pnr:.4f}")
    print(f"   - 总正序对数: {total_pos}")
    print(f"   - 总逆序对数: {total_neg}")
    print("="*50 + "\n")

    return df, global_pnr

# === 使用示例 ===
if __name__ == "__main__":
    set_random_seed(42)

    # --- 请根据你的实际情况修改以下路径 ---
    test_json_path = "/data/ss/benchmark/json/TOU/task_datasets/shuffle_sort_swift_test.json"  # <--- 在这里填入你的测试JSON路径
    model_base_path = "/data/ss/benchmark/model/Qwen2.5-vl"        # <--- 你的模型基础路径
    tsv_output_dir = "/data/ss/benchmark/result"                  # <--- 你的结果输出目录
    lora_ckpt_path = None # 如果有LoRA权重，请指定路径，例如: "/path/to/your/checkpoint-xxx"

    if os.path.exists(test_json_path):
        evaluate_image_reordering_task(
            test_json_path, 
            model_base_path, 
            tsv_output_dir,
            lora_ckpt_path=lora_ckpt_path
        )
    else:
        print(f"❌ 错误: 找不到测试文件 '{test_json_path}'。")