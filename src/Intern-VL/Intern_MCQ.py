import os
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# --- CORE MODIFICATION 1: Import new libraries, including PeftModel ---
from modelscope import AutoModel, AutoTokenizer
from internvl_utils import load_image
from peft import PeftModel # <-- Re-added for LoRA support

def set_random_seed(seed):
    """设置随机种子以确保结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- CORE MODIFICATION 2: Add lora_ckpt_path back to the function signature ---
def evaluate_mcq_task_internvl(
    task_name: str,
    test_json_path: str,
    model_base_path: str,
    tsv_output_dir: str,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    一个通用的函数，用于评估 InternVL 模型在任何多项选择题（MCQ）任务上的表现。
    - 支持可选的 LoRA 权重加载。
    """
    print(f"\n{'='*20} Starting MCQ Task: {task_name.upper()} {'='*20}")
    
    # === 1. 加载模型 & Tokenizer (InternVL 方式) ===
    print("🚀 开始加载 InternVL 模型和 Tokenizer...")
    model = AutoModel.from_pretrained(
        model_base_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2",
        trust_remote_code=True, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_base_path, trust_remote_code=True)

    # --- CORE MODIFICATION 3: Restore optional LoRA loading logic ---
    if lora_ckpt_path:
        print(f"🔄 正在从 {lora_ckpt_path} 加载并合并LoRA权重...")
        model = PeftModel.from_pretrained(model, lora_ckpt_path).merge_and_unload()
        print("✅ LoRA权重已成功合并。")
        
    print("✅ InternVL 模型 (+LoRA, if applicable) 和 Tokenizer 加载完成。")


    # === 2. 加载测试数据 (无变化) ===
    print(f"📚 正在从 {test_json_path} 加载测试数据...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ 成功加载 {len(test_data)} 条测试样本。")

    # === 3. 推理主循环 (无变化) ===
    results = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for entry in tqdm(test_data, desc=f"🧠 Evaluating {task_name} (InternVL)"):
        image_paths = entry.get("images", [])
        if not image_paths:
            continue

        try:
            question = entry["messages"][0]["content"]
            gt_answer = entry["messages"][1]["content"].strip()
        except (KeyError, IndexError):
            continue

        try:
            pixel_values_list = [load_image(p).to(torch.bfloat16) for p in image_paths]
            num_patches_list = [p.size(0) for p in pixel_values_list]
            pixel_values = torch.cat(pixel_values_list, dim=0).to(device)
        except Exception as e:
            print(f"❌ 处理图像时发生错误: {image_paths}。跳过此样本。错误详情: {e}")
            continue

        generation_config = dict(max_new_tokens=10, do_sample=False)
        
        with torch.no_grad():
            response = model.chat(
                tokenizer, pixel_values=pixel_values, question=question,
                generation_config=generation_config, num_patches_list=num_patches_list
            )
        model_output = response.strip()
        
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
    tsv_save_path = os.path.join(tsv_output_dir, f"InternVL_{task_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    correct_predictions = (df["ground_truth"] == df["prediction"]).sum()
    total_samples = len(df)
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    print("\n" + "="*50)
    print(f"✅ 任务 '{task_name}' 推理完成，结果已保存至：{tsv_save_path}")
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
    MODEL_BASE_PATH = "/data/ss/benchmark/model/Intern-VL"
    TSV_OUTPUT_DIR = "/data/ss/benchmark/result"
    
    # --- CORE MODIFICATION 4: Restore lora_ckpt_path variable and complete task list ---
    # To run with LoRA, set this to your checkpoint path, e.g., "/path/to/your/checkpoint-xxx"
    #lora_ckpt_path = "/data/ss/benchmark/lora/Intern-VL/v3-20251011-112954/checkpoint-450" 
    lora_ckpt_path = None
    # --- 定义所有要运行的MCQ任务 ---
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
            evaluate_mcq_task_internvl(
                task_name=task_name,
                test_json_path=json_path,
                model_base_path=MODEL_BASE_PATH,
                tsv_output_dir=TSV_OUTPUT_DIR,
                lora_ckpt_path=lora_ckpt_path # <-- Pass the argument to the function
            )
        else:
            print(f"❌ 警告: 找不到任务 '{task_name}' 的测试文件 '{json_path}'，已跳过。")