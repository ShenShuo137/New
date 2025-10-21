import os
import json
import re
import torch
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# --- CORE MODIFICATION 1: Import new libraries, including PeftModel ---
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel # <-- Added for LoRA support

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
def evaluate_image_reordering_task_mimo_vl(
    test_json_path,
    model_base_path,
    tsv_output_dir,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    评估 MiMo-VL 模型在“图像乱序重排”任务上的表现，使用PNR作为核心指标。
    - 支持可选的 LoRA 权重加载。
    """
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
        return None, 0.0

    # === 2. 加载测试数据 (无变化) ===
    print(f"📚 正在从 {test_json_path} 加载测试数据...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ 成功加载 {len(test_data)} 条测试样本。")

    # === 3. 推理主循环 (无变化) ===
    results = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for entry in tqdm(test_data, desc="🧠 Evaluating Image Reordering (MiMo-VL)"):
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
        
        predicted_numbers = re.findall(r'\d+', model_output)
        predicted_answer = " ".join(predicted_numbers)

        results.append({
            "ground_truth": gt_answer, "prediction": predicted_answer, "raw_output": raw_output,
            "timestamp": timestamp_now, "images": "|".join(image_paths)
        })

    # === 4. 保存结果并计算PNR得分 (无变化) ===
    if not results:
        print("❌ 评估结束，但没有有效的评估结果可以保存。")
        return None, 0.0

    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True)
    tsv_save_path = os.path.join(tsv_output_dir, f"Mimo-VL_image_reordering_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    total_pos, total_neg = 0, 0
    for _, row in df.iterrows():
        gt_str, pred_str = row["ground_truth"], row["prediction"]
        try:
            gt_order = [int(x) for x in gt_str.split()]
            pred_order = [int(x) for x in pred_str.split()]

            if sorted(pred_order) != sorted(gt_order):
                num_pairs = len(gt_order) * (len(gt_order) - 1) // 2
                total_neg += num_pairs
                continue

            pred_rank = {img_id: rank for rank, img_id in enumerate(pred_order)}
            for i in range(len(gt_order)):
                for j in range(i + 1, len(gt_order)):
                    img_a, img_b = gt_order[i], gt_order[j]
                    if pred_rank.get(img_a, -1) < pred_rank.get(img_b, -1):
                        total_pos += 1
                    else:
                        total_neg += 1
        except (ValueError, KeyError) as e:
            print(f"❌ 错误: 处理PNR计算时发生错误。GT: '{gt_str}', Pred: '{pred_str}'. 详情: {e}")
            continue
    
    global_pnr = (total_pos / total_neg) if total_neg > 0 else (float('inf') if total_pos > 0 else 0.0)

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

    # --- 全局配置 ---
    test_json_path = "/data/ss/benchmark/json/TOU/task_datasets/shuffle_sort_swift_test.json"
    MODEL_BASE_PATH = "/data/ss/benchmark/model/Mimo-VL"
    TSV_OUTPUT_DIR = "/data/ss/benchmark/result"

    # --- CORE MODIFICATION 4: Restore lora_ckpt_path variable for easy testing ---
    # To run with LoRA, set this to your checkpoint path, e.g., "/path/to/your/checkpoint-xxx"
    lora_ckpt_path = None

    # --- 运行评估 ---
    if os.path.exists(test_json_path):
        evaluate_image_reordering_task_mimo_vl(
            test_json_path, 
            MODEL_BASE_PATH, 
            TSV_OUTPUT_DIR,
            lora_ckpt_path=lora_ckpt_path # <-- Pass the argument to the function
        )
    else:
        print(f"❌ 错误: 找不到测试文件 '{test_json_path}'。")