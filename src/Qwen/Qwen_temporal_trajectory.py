import os
import json
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

def evaluate_temporal_interpolation_task(
    test_json_path,
    model_base_path,
    tsv_output_dir,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    评估模型在“时间序列插值”任务上的表现。
    该任务给定4张图片，前3张位于1,3,5位，要求模型判断第4张图片属于2位还是4位。
    """
    # === 1. 加载模型 & 处理器 (无变化) ===
    print("🚀 开始加载模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )
    if lora_ckpt_path:
        model = PeftModel.from_pretrained(model, lora_ckpt_path).merge_and_unload()
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

    for entry in tqdm(test_data, desc="🧠 Evaluating Temporal Interpolation"):
        image_paths = entry.get("images", [])
        
        # --- MODIFICATION 1: 确保此任务有且仅有4张图片 ---
        if len(image_paths) != 4:
            print(f"⚠️ 警告: 跳过一个样本，因为它包含 {len(image_paths)} 张图片，而不是此任务要求的4张。")
            continue

        try:
            prompt_text = entry["messages"][0]["content"]
            gt_answer = entry["messages"][1]["content"].strip()
        except (KeyError, IndexError) as e:
            print(f"⚠️ 警告: 跳过一个样本，因为其 'messages' 结构不正确。错误: {e}")
            continue

        # 构造输入 (与之前修正版相同，保留<image>占位符)
        messages = [{"role": "user", "content": [{"type": "image", "image": p} for p in image_paths] + [{"type": "text", "text": prompt_text}]}]
        
        try:
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, return_tensors="pt").to(device)
        except Exception as e:
            print(f"❌ 处理输入时发生错误，图片路径: {image_paths}。错误详情: {e}")
            continue

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
        
        generated_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        model_output = generated_text.strip()
        
        # MCQ任务，直接取第一个大写字母作为预测答案
        predicted_answer = model_output[:1].upper() if model_output else ""

        results.append({
            "ground_truth": gt_answer, "prediction": predicted_answer, "raw_output": model_output,
            "timestamp": timestamp_now, "images": "|".join(image_paths)
        })

    # === 4. 保存和评估结果 ===
    if not results:
        print("❌ 评估结束，但没有有效的评估结果可以保存。")
        return None, 0, {}
        
    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True)
    # --- MODIFICATION 2: 更新输出文件名 ---
    tsv_save_path = os.path.join(tsv_output_dir, f"temporal_interpolation_eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    correct_predictions = (df["ground_truth"] == df["prediction"]).sum()
    overall_accuracy = correct_predictions / len(df) if len(df) > 0 else 0.0

    print("\n" + "="*50)
    print(f"✅ 推理完成，结果已保存至：{tsv_save_path}")
    print(f"📊 整体准确率: {overall_accuracy:.4f} ({correct_predictions}/{len(df)})")
    print("\n📊 各选项详细表现:")
    
    # --- MODIFICATION 3: 为此任务定制准确的选项标签 ---
    option_mapping = {
        'A': 'Position 4',
        'B': 'Position 2'
    }
    option_accuracies = {}

    for option, label in option_mapping.items():
        gt_mask = df["ground_truth"] == option
        count = gt_mask.sum()
        if count > 0:
            correct_for_option = ((df["ground_truth"] == option) & (df["prediction"] == option)).sum()
            accuracy = correct_for_option / count
            stats = {'accuracy': accuracy, 'count': int(count), 'correct': int(correct_for_option)}
        else:
            stats = {'accuracy': 0.0, 'count': 0, 'correct': 0}
        
        option_accuracies[label] = stats
        print(f"  - 选项 {option} ({label}): {stats['accuracy']:.4f} (预测正确 {stats['correct']} / 总数 {stats['count']})")
    print("="*50 + "\n")

    return df, overall_accuracy, option_accuracies

# === 使用示例 ===
if __name__ == "__main__":
    set_random_seed(42)

    # --- 请根据你的实际情况修改以下路径 ---
    # 这是你的时间插值任务的测试集JSON文件
    test_json_path = "/data/ss/benchmark/json/TOU/task_datasets/temporal_trajectory_swift_test.json" # <--- 在这里填入正确的路径
    
    # 基础模型路径
    model_base_path = "/data/ss/benchmark/model/Qwen2.5-vl"
    
    # 评估结果输出目录
    tsv_output_dir = "/data/ss/benchmark/result"
    
    # LoRA ckpt路径（如果需要）
    lora_ckpt_path = None

    # --- 运行评估 ---
    if os.path.exists(test_json_path):
        evaluate_temporal_interpolation_task(
            test_json_path, 
            model_base_path, 
            tsv_output_dir,
            lora_ckpt_path=lora_ckpt_path
        )
    else:
        print(f"❌ 错误: 找不到测试文件 '{test_json_path}'。")
