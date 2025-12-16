#!/usr/bin/env python
# coding: utf-8
"""
简化版Fusion Alpaca数据集生成器
instruction=question, input为空, output=fusion_reply
"""

import json
import pandas as pd
from pathlib import Path

# 配置
BASE_DIR = Path(r"D:\project7\10000final")
BASE_DIR1 = Path(r"D:\qwensft\uploadjson")

INPUT_JSON = BASE_DIR / "doubao-pro-32k_answers_2+1-2-1-9400.json"
SCORES_JSON = BASE_DIR / "grades_doubao-pro-256k_answers_2+1-2-1-9400.json"
OUTPUT_DATASET = BASE_DIR1 / "alpaca_dataset_2+1_sft_withoutthinking.json"
OUTPUT_EXCEL = BASE_DIR1 / "alpaca_dataset_2+1_info.xlsx"

ANALYZE_COUNT = 3000

def load_scores(scores_path):
    """加载评分数据"""
    with scores_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    question_scores = {}
    for item in data.get("detailed_results", []):
        question = item.get("question", "")
        if question:
            avg_scores = item.get("avg_scores", {})
            question_scores[question] = {
                "avg_score_50": avg_scores.get("total", 0),
                "avg_logic": avg_scores.get("logic", 0),
                "avg_depth": avg_scores.get("depth", 0),
                "avg_innovation": avg_scores.get("innovation", 0),
                "avg_accuracy": avg_scores.get("accuracy", 0),
                "avg_completeness": avg_scores.get("completeness", 0),
                "avg_score_100": item.get("avg_score_100", 0),
                "num_valid_trials": item.get("num_valid_trials", 0)
            }
    
    print(f"加载了 {len(question_scores)} 个问题的评分")
    return question_scores

def create_alpaca_sample(item):
    """创建Alpaca格式样本"""
    return {
        "instruction": item.get("question", ""),
        "input": "",
        "output": item.get("fusion_reply", "")
    }

def main():
    print("开始生成Fusion Alpaca数据集...")
    
    # 加载数据
    scores = load_scores(SCORES_JSON)
    
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    
    print(f"加载了 {len(data)} 条数据")
    
    # 过滤有效数据
    valid_items = []
    for item in data:
        question = item.get("question", "")
        reply = item.get("fusion_reply", "")
        
        # 检查必需字段和长度
        if question and reply and len(question) >= 10 and len(reply) >= 100 and question in scores:
            item.update(scores[question])
            valid_items.append(item)
    
    print(f"通过筛选: {len(valid_items)} 条")
    
    # 按分数排序取TOP
    valid_items.sort(key=lambda x: x.get("avg_score_50", 0), reverse=True)
    process_items = valid_items[:ANALYZE_COUNT]
    
    # 生成Alpaca数据集
    alpaca_dataset = []
    excel_data = []
    
    for idx, item in enumerate(process_items, 1):
        sample = create_alpaca_sample(item)
        if sample["instruction"] and sample["output"]:
            alpaca_dataset.append(sample)
            
            excel_data.append({
                "序号": idx,
                "问题": item.get("question", ""),
                "评分(50分制)": item.get("avg_score_50", 0),
                "评分(100分制)": item.get("avg_score_100", 0),
                "逻辑": item.get("avg_logic", 0),
                "深度": item.get("avg_depth", 0),
                "创新": item.get("avg_innovation", 0),
                "准确": item.get("avg_accuracy", 0),
                "完整": item.get("avg_completeness", 0),
                "Instruction长度": len(sample["instruction"]),
                "Output长度": len(sample["output"]),
                "A1长度": len(item.get("A1_third_answer", "")),
                "A2长度": len(item.get("A2_combination_reply", "")),
                "Fusion长度": len(item.get("fusion_reply", ""))
            })
    
    # 保存结果
    with OUTPUT_DATASET.open("w", encoding="utf-8") as f:
        json.dump(alpaca_dataset, f, ensure_ascii=False, indent=2)
    
    if excel_data:
        df = pd.DataFrame(excel_data)
        df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"完成! 生成了 {len(alpaca_dataset)} 个样本")
    print(f"JSON: {OUTPUT_DATASET}")
    print(f"Excel: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()