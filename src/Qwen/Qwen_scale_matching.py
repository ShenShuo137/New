import os
import json
import re
import torch
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# --- CORE MODIFICATION 1: Ensure PeftModel is imported ---
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel # <-- Enabled for LoRA support
from qwen_vl_utils import process_vision_info

def set_random_seed(seed):
    """设置随机种子以确保结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_model_output(generated_text):
    """清理模型输出，移除 <think></think> 标签和多余的换行符。"""
    cleaned = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
    lines = cleaned.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line
    return cleaned.strip()

# --- CORE MODIFICATION 2: Add lora_ckpt_path back to the function signature ---
def evaluate_mcq_task_mimo_vl(
    task_name: str,
    test_json_path: str,
    model_base_path: str,
    tsv_output_dir: str,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    一个通用的函数，用于评估 MiMo-VL 模型在任何多项选择题（MCQ）任务上的表现。
    - 支持可选的 LoRA 权重加载。
    """
    print(f"\n{'='*20} Starting MiMo-VL Evaluation for Task: {task_name.upper()} {'='*20}")
    
    # === 1. 加载模型 (使用Qwen的类加载MiMo-VL权重) ===
    print(f"🚀 开始加载 MiMo-VL 模型 (兼容Qwen架构) from '{model_base_path}'...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_base_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_base_path)

        # --- CORE MODIFICATION 3: Restore optional LoRA loading logic ---
        if lora_ckpt_path:
            print(f"🔄 正在从 {lora_ckpt_path} 加载并合并LoRA权重...")
            model = PeftModel.from_pretrained(model, lora_ckpt_path).merge_and_unload()
            print("✅ LoRA权重已成功合并。")

        print("✅ MiMo-VL 模型 (+LoRA, if applicable) 和处理器加载完成。")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}。请检查路径 '{model_base_path}' / '{lora_ckpt_path}' 是否正确。")
        return

    # === 2. 加载测试数据 (无变化) ===
    print(f"📚 正在从 {test_json_path} 加载测试数据...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ 成功加载 {len(test_data)} 条测试样本。")

    # === 3. 推理主循环 (无变化) ===
    results = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for entry in tqdm(test_data, desc=f"🧠 Evaluating {task_name} (MiMo-VL)"):
        image_paths = entry.get("images", [])
        if not image_paths: continue

        try:
            prompt_text = entry["messages"][0]["content"]
            gt_answer = entry["messages"][1]["content"].strip()
            prompt_text += "\n<no_think>\n"
        except (KeyError, IndexError):
            continue

        messages = [{"role": "user", "content": [{"type": "image", "image": p} for p in image_paths] + [{"type": "text", "text": prompt_text}]}]
        
        try:
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(device)
        except Exception as e:
            print(f"❌ 处理输入时发生错误: {image_paths}。错误: {e}")
            continue

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        raw_output = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        model_output = clean_model_output(raw_output)
        
        predicted_answer = model_output[:1].upper() if model_output else ""

        results.append({
            "ground_truth": gt_answer, "prediction": predicted_answer, "raw_output": raw_output,
            "timestamp": timestamp_now, "images": "|".join(image_paths)
        })

    # === 4. 保存与评估 (无变化) ===
    if not results:
        print(f"❌ 任务 '{task_name}' 评估结束，但没有有效的评估结果。")
        del model, processor
        torch.cuda.empty_cache()
        return

    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True)
    tsv_save_path = os.path.join(tsv_output_dir, f"Mimo-VL_{task_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    correct_predictions = (df["ground_truth"] == df["prediction"]).sum()
    total_samples = len(df)
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    print("\n" + "="*50)
    print(f"✅ 任务 '{task_name}' (MiMo-VL) 推理完成，结果已保存至：{tsv_save_path}")
    print(f"📊 整体准确率: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    print("\n📊 各选项详细表现:")
    possible_options = sorted(df["ground_truth"].unique())
    for option in possible_options:
        gt_mask = df["ground_truth"] == option
        count = gt_mask.sum()
        correct_for_option = (gt_mask & (df["prediction"] == option)).sum()
        accuracy = correct_for_option / count if count > 0 else 0.0
        print(f"  - 选项 {option}: {accuracy:.4f} (预测正确 {correct_for_option} / 总数 {count})")
    print("="*50 + "\n")
    
    del model, processor
    torch.cuda.empty_cache()

# === 使用示例 ===
if __name__ == "__main__":
    set_random_seed(42)

    # --- 全局配置 ---
    MODEL_BASE_PATH = "/data/ss/benchmark/model/Qwen2.5-vl"
    TSV_OUTPUT_DIR = "/data/ss/benchmark/result"
    
    # --- CORE MODIFICATION 4: Restore lora_ckpt_path variable for easy testing ---
    # To run with LoRA, set this to your checkpoint path, e.g., "/path/to/your/checkpoint-xxx"
    lora_ckpt_path = None
    
    MCQ_TASKS = {
        
        'scale_matching_within_group_mcq_swift_test': '/data/ss/benchmark/json/TSU/task_datasets/scale_matching_within_group_mcq_swift_test.json',
        
    }

    # --- 循环运行所有定义的任务 ---
    for task_name, json_path in MCQ_TASKS.items():
        if os.path.exists(json_path):
            evaluate_mcq_task_mimo_vl(
                task_name=task_name,
                test_json_path=json_path,
                model_base_path=MODEL_BASE_PATH,
                tsv_output_dir=TSV_OUTPUT_DIR,
                lora_ckpt_path=lora_ckpt_path # <-- Pass the argument to the function
            )
        else:
            print(f"❌ 警告: 找不到任务 '{task_name}' 的测试文件 '{json_path}'，已跳过。")