import os
import json
import re
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
def evaluate_image_reordering_task_internvl(
    test_json_path,
    model_base_path,
    tsv_output_dir,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    评估 InternVL 模型在“图像乱序重排”任务上的表现，使用PNR作为核心指标。
    - 支持可选的 LoRA 权重加载。
    """
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

    for entry in tqdm(test_data, desc="🧠 Evaluating Image Reordering (InternVL)"):
        image_paths = entry.get("images", [])
        if not image_paths:
            continue

        try:
            prompt_text = entry["messages"][0]["content"]
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

        generation_config = dict(max_new_tokens=64, do_sample=False)
        
        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer, pixel_values=pixel_values, question=prompt_text,
                generation_config=generation_config, num_patches_list=num_patches_list,
                history=None, return_history=False
            )
        model_output = response.strip()
        
        predicted_numbers = re.findall(r'\d+', model_output)
        predicted_answer = " ".join(predicted_numbers)

        results.append({
            "ground_truth": gt_answer, "prediction": predicted_answer, "raw_output": model_output,
            "timestamp": timestamp_now, "images": "|".join(image_paths)
        })

    # === 4. 保存结果并计算PNR得分 (无变化) ===
    if not results:
        print("❌ 评估结束，但没有有效的评估结果可以保存。")
        return None, 0.0

    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True)
    tsv_save_path = os.path.join(tsv_output_dir, f"InternVL_image_reordering_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
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

    test_json_path = "/data/ss/benchmark/json/TOU/task_datasets/shuffle_sort_swift_test.json"
    model_base_path = "/data/ss/benchmark/model/Intern-VL"
    tsv_output_dir = "/data/ss/benchmark/result"

    # --- CORE MODIFICATION 4: Restore lora_ckpt_path variable for easy testing ---
    # To run with LoRA, set this to your checkpoint path, e.g., "/path/to/your/checkpoint-xxx"
    #lora_ckpt_path = "/data/ss/benchmark/lora/Intern-VL/v3-20251011-112954/checkpoint-450" 
    lora_ckpt_path = None
    if os.path.exists(test_json_path):
        evaluate_image_reordering_task_internvl(
            test_json_path, 
            model_base_path, 
            tsv_output_dir,
            lora_ckpt_path=lora_ckpt_path # <-- Pass the argument to the function
        )
    else:
        print(f"❌ 错误: 找不到测试文件 '{test_json_path}'。")