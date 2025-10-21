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

def evaluate_timespan_prediction_task(
    test_json_path,
    model_base_path,
    tsv_output_dir,
    lora_ckpt_path=None,
    device="cuda"
):
    """
    评估模型在时间跨度预测任务上的表现。
    该任务给定4张图片和前两个时间间隔，要求模型预测第三个时间间隔的类别。
    """
    # === 1. 加载模型 & 处理器 ===
    print("🚀 开始加载模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    if lora_ckpt_path:
        print(f"🔄 正在从 {lora_ckpt_path} 加载并合并LoRA权重...")
        model = PeftModel.from_pretrained(model, lora_ckpt_path)
        model = model.merge_and_unload()
        print("✅ LoRA权重已成功合并。")

    processor = AutoProcessor.from_pretrained(model_base_path, min_pixels=256*28*28, max_pixels=1280*28*28)
    print("✅ 模型和处理器加载完成。")

    # === 2. 加载测试数据 ===
    print(f"📚 正在从 {test_json_path} 加载测试数据...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ 成功加载 {len(test_data)} 条测试样本。")

    # === 3. 准备结果列表 ===
    results = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # === 4. 推理主循环 ===
    for entry in tqdm(test_data, desc="🧠 Evaluating Timespan Prediction Task"):
        image_paths = entry.get("images", [])
        
        if len(image_paths) != 4:
            print(f"⚠️ 警告: 跳过一个样本，因为它包含 {len(image_paths)} 张图片，而不是4张。")
            continue

        try:
            # Prompt位于第一个message中
            prompt_with_placeholders = entry["messages"][0]["content"]
            # 正确答案位于第二个message中，直接就是选项字母
            gt_answer = entry["messages"][1]["content"].strip()
        except (KeyError, IndexError) as e:
            print(f"⚠️ 警告: 跳过一个样本，因为其 'messages' 结构不正确。错误: {e}")
            continue

        # 移除Prompt中的<image>占位符，processor会自动处理文本和图像的关联
        prompt_text = prompt_with_placeholders.replace("<image>", "").strip()

        # 构造模型输入格式
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": p} for p in image_paths] +
                           [{"type": "text", "text": prompt_text}]
            }
        ]

        # 处理输入
        try:
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            ).to(device)
        except Exception as e:
            print(f"❌ 处理输入时发生错误，图片路径: {image_paths}。错误详情: {e}")
            continue

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10) # 答案很短，10个token足够
        
        # 解码并提取预测答案
        generated_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        model_output = generated_text.strip()
        
        # 预测的答案应该是输出的第一个非空字符，并转为大写以保证一致性
        predicted_answer = model_output[:1].upper() if model_output else ""

        # 记录结果
        results.append({
            "ground_truth": gt_answer,
            "prediction": predicted_answer,
            "raw_output": model_output,
            "timestamp": timestamp_now,
            "images": "|".join(image_paths)
        })

    # === 5. 保存结果到TSV ===
    if not results:
        print("❌ 评估结束，但没有有效的评估结果可以保存。请检查测试数据和模型。")
        return None, 0, {}
        
    df = pd.DataFrame(results)
    os.makedirs(tsv_output_dir, exist_ok=True) # 确保输出目录存在
    tsv_save_path = os.path.join(tsv_output_dir, f"timespan_pred_eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    df.to_csv(tsv_save_path, sep="\t", index=False)

    # === 6. 计算和打印准确率 ===
    correct_predictions = (df["ground_truth"] == df["prediction"]).sum()
    overall_accuracy = correct_predictions / len(df) if len(df) > 0 else 0.0

    # 计算各个选项的准确率 (A, B, C)
    option_mapping = {'A': 'Days', 'B': 'Months', 'C': 'Years'}
    option_accuracies = {}
    
    print("\n" + "="*50)
    print(f"✅ 推理完成，结果已保存至：{tsv_save_path}")
    print(f"📊 整体准确率: {overall_accuracy:.4f} ({correct_predictions}/{len(df)})")
    print("\n📊 各选项详细表现:")
    for option, label in option_mapping.items():
        gt_mask = df["ground_truth"] == option
        count = gt_mask.sum()
        if count > 0:
            correct_for_option = ((df["ground_truth"] == option) & (df["prediction"] == option)).sum()
            accuracy = correct_for_option / count
            stats = {'accuracy': accuracy, 'count': int(count), 'correct': int(correct_for_option)}
        else:
            stats = {'accuracy': 0.0, 'count': 0, 'correct': 0}
        
        option_accuracies[f"{option}_{label}"] = stats
        print(f"  - 选项 {option} ({label}): {stats['accuracy']:.4f} (预测正确 {stats['correct']} / 总数 {stats['count']})")
    print("="*50 + "\n")


    return df, overall_accuracy, option_accuracies

# === 使用示例 ===
if __name__ == "__main__":
    set_random_seed(42)  # 设置固定随机种子以保证复现性

    # --- 请根据你的实际情况修改以下路径 ---
    # 这是你的生成脚本产生的测试集JSON文件
    test_json_path = "/data/ss/benchmark/json/TSU/task_datasets/pairwise_scale_comparison_within_group_mcq_swift_test.json"
    
    # 这是你的基础模型路径
    model_base_path = "/data/ss/benchmark/model/Qwen2.5-vl"
    
    # 这是评估结果（TSV文件）的输出目录
    tsv_output_dir = "/data/ss/benchmark/result"
    
    # 如果要评估LoRA微调后的模型，请取消下面的注释并提供正确的checkpoint路径
    # lora_ckpt_path = "/path/to/your/lora_checkpoint" 
    lora_ckpt_path = None

    # --- 运行评估 ---
    if os.path.exists(test_json_path):
        evaluate_timespan_prediction_task(
            test_json_path, 
            model_base_path, 
            tsv_output_dir,
            lora_ckpt_path=lora_ckpt_path
        )
    else:
        print(f"❌ 错误: 找不到测试文件 '{test_json_path}'。请确保路径正确并已生成该文件。")
