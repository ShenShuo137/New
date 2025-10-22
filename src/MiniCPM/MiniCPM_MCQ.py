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
def evaluate_mcq_task_minicpm(
    task_name: str,
    test_json_path: str,
    model_base_path: str,
    tsv_output_dir: str,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    一个通用的函数，用于评估 MiniCPM-V 模型在任何多项选择题（MCQ）任务上的表现。
    - 支持可选的 LoRA 权重加载。
    """
    print(f"\n{'='*20} Starting MiniCPM-V Evaluation for Task: {task_name.upper()} {'='*20}")
    
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
        return

    # === 2. 加载测试数据 (无变化) ===
    print(f"📚 正在从 {test_json_path} 加载测试数据...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ 成功加载 {len(test_data)} 条测试样本。")

    # === 3. 推理主循环 (无变化) ===
    results = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for entry in tqdm(test_data, desc=f"🧠 Evaluating {task_name} (MiniCPM-V)"):
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
                response = model.chat(msgs=msgs, tokenizer=tokenizer)
            model_output = response.strip()
        except Exception as e:
            print(f"❌ 模型推理时发生错误。跳过此样本。错误: {e}")
            continue
        
        predicted_answer = model_output[:1].upper() if model_output else ""

        results.append({
            "ground_truth": gt_answer, "prediction": predicted_answer, "raw_output": model_output,
            "timestamp": timestamp_now, "images": "|".join(image_paths)
        })

    # === 4. 保存与评估 (无变化) ===
    if not results:
        print(f"❌ 任务 '{task_name}' 评估结束，但没有有效的评估结果。")
        del model, tokenizer
        torch.cuda.empty_cache()
        return

    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True)
    tsv_save_path = os.path.join(tsv_output_dir, f"MiniCPM_{task_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    correct_predictions = (df["ground_truth"] == df["prediction"]).sum()
    total_samples = len(df)
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    print("\n" + "="*50)
    print(f"✅ 任务 '{task_name}' (MiniCPM-V) 推理完成，结果已保存至：{tsv_save_path}")
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
    
    del model, tokenizer
    torch.cuda.empty_cache()

# === 使用示例 ===
if __name__ == "__main__":
    set_random_seed(42)

    # --- 全局配置 ---
    MODEL_BASE_PATH = "/data/ss/benchmark/model/MiniCPM"
    TSV_OUTPUT_DIR = "/data/ss/benchmark/result"
    
    # --- CORE MODIFICATION 4: Restore lora_ckpt_path variable for easy testing ---
    # To run with LoRA, set this to your checkpoint path, e.g., "/path/to/your/checkpoint-xxx"
    lora_ckpt_path = "/data/ss/benchmark/lora/MiniCPM/v2-20251016-165915/checkpoint-450"
    
    MCQ_TASKS = {
        'anchored_scale_prediction': '/data/ss/benchmark/json/TSU/task_datasets/anchored_scale_prediction_mcq_swift_test.json',
        'interval_extremum_identification': '/data/ss/benchmark/json/TSU/task_datasets/interval_extremum_identification_mcq_4img_swift_test.json',
        'pairwise_scale_comparison': '/data/ss/benchmark/json/TSU/task_datasets/pairwise_scale_comparison_within_group_mcq_swift_test.json',
        'scale_matching_within_group_mcq_swift_test': '/data/ss/benchmark/json/TSU/task_datasets/scale_matching_within_group_mcq_swift_test.json',
        'temporal_trajectory_swift_test': '/data/ss/benchmark/json/TOU/task_datasets/temporal_trajectory_swift_test.json',
    }

    # --- 循环运行所有定义的任务 ---
    for task_name, json_path in MCQ_TASKS.items():
        if os.path.exists(json_path):
            evaluate_mcq_task_minicpm(
                task_name=task_name,
                test_json_path=json_path,
                model_base_path=MODEL_BASE_PATH,
                tsv_output_dir=TSV_OUTPUT_DIR,
                lora_ckpt_path=lora_ckpt_path # <-- Pass the argument to the function
            )
        else:
            print(f"❌ 警告: 找不到任务 '{task_name}' 的测试文件 '{json_path}'，已跳过。")