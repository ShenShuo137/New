import os
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# --- CORE MODIFICATION 1: Import new libraries, including PeftModel ---
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from peft import PeftModel # <-- Added for LoRA support

def set_random_seed(seed):
    """设置随机种子以确保结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pil_image(image_path):
    """加载单张图片为PIL.Image对象。"""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ 加载图片时发生错误: {image_path}. 错误: {e}")
        return None

# --- CORE MODIFICATION 2: Add lora_ckpt_path back to the function signature ---
def evaluate_anomaly_detection_task_minicpm(
    test_json_path,
    model_base_path,
    tsv_output_dir,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    评估 MiniCPM-V 模型在“时间序列异常检测”任务上的表现。
    - 支持可选的 LoRA 权重加载。
    """
    # === 1. 加载 MiniCPM-V 模型和 Tokenizer ===
    print("🚀 开始加载 MiniCPM-V 模型和 Tokenizer...")
    try:
        model = AutoModel.from_pretrained(
            model_base_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_base_path, trust_remote_code=True)
        
        # --- CORE MODIFICATION 3: Restore optional LoRA loading logic ---
        if lora_ckpt_path:
            print(f"🔄 正在从 {lora_ckpt_path} 加载并合并LoRA权重...")
            model = PeftModel.from_pretrained(model, lora_ckpt_path).merge_and_unload()
            print("✅ LoRA权重已成功合并。")
            
        print("✅ MiniCPM-V 模型 (+LoRA, if applicable) 和 Tokenizer 加载完成。")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}。请检查路径 '{model_base_path}' / '{lora_ckpt_path}' 是否正确。")
        return None, 0, {}

    # === 2. 加载测试数据 (无变化) ===
    print(f"📚 正在从 {test_json_path} 加载测试数据...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ 成功加载 {len(test_data)} 条测试样本。")

    # === 3. 推理主循环 (无变化) ===
    results = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for entry in tqdm(test_data, desc="🧠 Evaluating Anomaly Detection (MiniCPM-V)"):
        image_paths = entry.get("images", [])
        if not image_paths: continue

        try:
            question = entry["messages"][0]["content"]
            gt_answer = entry["messages"][1]["content"].strip()
            question = question.replace('<image>', '').lstrip('\n')
        except (KeyError, IndexError):
            continue

        images = [load_pil_image(p) for p in image_paths if p]
        if None in images:
            print(f"⚠️ 跳过样本，因为其中的一张或多张图片无法加载: {image_paths}")
            continue
        msgs = [{'role': 'user', 'content': images + [question]}]

        try:
            with torch.no_grad():
                response = model.chat(msgs=msgs, tokenizer=tokenizer, max_new_tokens=40) # Increased token limit slightly
            model_output = response.strip()
        except Exception as e:
            print(f"❌ 模型推理时发生错误。跳过此样本。错误: {e}")
            continue
        
        # --- CORE MODIFICATION: Enhanced parsing logic to handle natural language answers ---
        predicted_answer = ""
        lower_output = model_output.lower()

        if "none" in lower_output:
            predicted_answer = "None"
        else:
            # Define mapping for English ordinal/cardinal numbers to digits
            word_to_digit = {
                "first": "1", "one": "1",
                "second": "2", "two": "2",
                "third": "3", "three": "3",
                "fourth": "4", "four": "4",
                "fifth": "5", "five": "5",
                "sixth": "6", "six": "6"
                # Add more if needed
            }
            
            found_positions = {}
            # First, try to find the earliest occurrence of a number word
            for word, digit in word_to_digit.items():
                pos = lower_output.find(word)
                if pos != -1:
                    found_positions[pos] = digit # Store position and corresponding digit
            
            if found_positions:
                # If words were found, get the digit corresponding to the one that appeared first
                first_pos = min(found_positions.keys())
                predicted_answer = found_positions[first_pos]
            else:
                # Fallback: if no number words are found, find the first digit in the output
                for char in model_output:
                    if char.isdigit():
                        predicted_answer = char
                        break
        # --- End of enhanced parsing logic ---

        results.append({
            "ground_truth": gt_answer, "prediction": predicted_answer, "raw_output": model_output,
            "timestamp": timestamp_now, "images": "|".join(image_paths)
        })

    # === 4. 保存和评估结果 (无变化) ===
    if not results:
        print("❌ 评估结束，但没有有效的评估结果可以保存。")
        return None, 0, {}
        
    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True)
    tsv_save_path = os.path.join(tsv_output_dir, f"MiniCPM_anomaly_detection_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    correct_predictions = (df["ground_truth"] == df["prediction"]).sum()
    overall_accuracy = correct_predictions / len(df) if len(df) > 0 else 0.0

    print("\n" + "="*50)
    print(f"✅ 推理完成，结果已保存至：{tsv_save_path}")
    print(f"📊 整体准确率: {overall_accuracy:.4f} ({correct_predictions}/{len(df)})")
    print("\n📊 各类答案详细表现:")
    
    option_accuracies = {}
    available_answers = sorted(df["ground_truth"].unique())
    for answer in available_answers:
        gt_mask = df["ground_truth"] == answer
        count = gt_mask.sum()
        correct_for_answer = (gt_mask & (df["prediction"] == answer)).sum()
        accuracy = correct_for_answer / count if count > 0 else 0.0
        stats = {'accuracy': accuracy, 'count': int(count), 'correct': int(correct_for_answer)}
        option_accuracies[answer] = stats
        print(f"  - 答案为 '{answer}': {stats['accuracy']:.4f} (预测正确 {stats['correct']} / 总数 {stats['count']})")
    print("="*50 + "\n")

    return df, overall_accuracy, option_accuracies

# === 使用示例 ===
if __name__ == "__main__":
    set_random_seed(42)

    test_json_path = "/data/ss/benchmark/json/TOU/task_datasets/anomaly_detection_swift_test.json"
    MODEL_BASE_PATH = "/data/ss/benchmark/model/MiniCPM"
    TSV_OUTPUT_DIR = "/data/ss/benchmark/result"

    # --- CORE MODIFICATION 4: Restore lora_ckpt_path variable for easy testing ---
    # To run with LoRA, set this to your checkpoint path, e.g., "/path/to/your/checkpoint-xxx"
    lora_ckpt_path = "/data/ss/benchmark/lora/MiniCPM/v2-20251016-165915/checkpoint-450"

    # --- 运行评估 ---
    if os.path.exists(test_json_path):
        evaluate_anomaly_detection_task_minicpm(
            test_json_path, 
            MODEL_BASE_PATH, 
            TSV_OUTPUT_DIR,
            lora_ckpt_path=lora_ckpt_path # <-- Pass the argument to the function
        )
    else:
        print(f"❌ 错误: 找不到测试文件 '{test_json_path}'。")