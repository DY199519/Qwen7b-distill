#!/usr/bin/env python
# coding: utf-8
"""
简化版RLHF数据集生成器
使用combination_1_reply字段，保留RLHF格式不变
"""
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# ==================== 配置部分 ====================
BASE_DIR = Path(r"D:\project7\10000final")
BASE_DIR1 = Path(r"D:\qwensft\uploadjson")
FUSION_JSON = BASE_DIR / "deepseek_answers_without_summary3+1-1-9400.json"
SCORES_JSON = BASE_DIR / "grades-3+1-1-9400.json"
OUTPUT_BASE = BASE_DIR1 / "rlhf_train_top1000-3+1"
TOP_N = 1000
# ================================================

def load_scores_data(scores_path: Path) -> Dict[str, Dict]:
    """加载评分数据"""
    with scores_path.open("r", encoding="utf-8") as f:
        scores_data = json.load(f)
    
    question_scores = {}
    detailed_results = scores_data.get("detailed_results", [])
    
    for item in detailed_results:
        question = item.get("question", "")
        if question:
            avg_scores = item.get("avg_scores", {})
            avg_score_50 = avg_scores.get("total", 0)
            
            question_scores[question] = {
                "avg_score_50": avg_score_50,
                "avg_scores": avg_scores,
                "num_valid_trials": item.get("num_valid_trials", 0),
                "avg_logic": avg_scores.get("logic", 0),
                "avg_depth": avg_scores.get("depth", 0),
                "avg_innovation": avg_scores.get("innovation", 0),
                "avg_accuracy": avg_scores.get("accuracy", 0),
                "avg_completeness": avg_scores.get("completeness", 0),
            }
    
    return question_scores

def create_rlhf_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    将数据项转换为RLHF格式
    使用question和combination_1_reply
    """
    problem_text = item.get("question", "")
    solution_text = item.get("combination_1_reply", "")
    
    # 构建messages
    messages = [
        {
            "role": "user",
            "content": problem_text
        },
        {
            "role": "assistant", 
            "content": solution_text
        }
    ]
    
    # 构建RLHF格式的样本
    rlhf_sample = {
        "problem": {"Value": problem_text},
        "solution": {"Value": solution_text},
        "messages": {"Value": messages}
    }
    
    return rlhf_sample

def generate_dataset():
    """生成RLHF格式的数据集"""
    
    # 加载评分数据
    question_scores = load_scores_data(SCORES_JSON)
    
    # 加载数据
    with FUSION_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return 1
    
    # 匹配评分并过滤数据
    items_with_scores = []
    required_fields = ["question", "combination_1_reply"]
    
    for item in items:
        question = item.get("question", "")
        
        # 检查是否有评分
        if question not in question_scores:
            continue
        
        # 检查必要字段
        missing_fields = [f for f in required_fields if not item.get(f)]
        if missing_fields:
            continue
        
        # 检查内容长度
        question_len = len(item.get("question", ""))
        reply_len = len(item.get("combination_1_reply", ""))
        
        if question_len < 10 or reply_len < 100:
            continue
        
        # 添加评分信息
        score_info = question_scores[question]
        item.update(score_info)
        items_with_scores.append(item)
    
    # 按评分排序，取TOP N
    items_with_scores.sort(key=lambda x: x["avg_score_50"], reverse=True)
    items_to_process = items_with_scores[:TOP_N]
    
    # 转换为RLHF格式
    rlhf_dataset = []
    for idx, item in enumerate(items_to_process, 1):
        try:
            rlhf_sample = create_rlhf_sample(item)
            
            if rlhf_sample["problem"]["Value"] and rlhf_sample["solution"]["Value"]:
                rlhf_dataset.append(rlhf_sample)
                
        except Exception as e:
            continue
    
    # 确保输出目录存在
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    json_file = OUTPUT_BASE.with_suffix('.json')
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(rlhf_dataset, f, ensure_ascii=False, indent=2)
    
    # 保存为Parquet格式
    parquet_file = OUTPUT_BASE.with_suffix('.parquet')
    df_data = []
    for item in rlhf_dataset:
        df_data.append({
            "problem": item["problem"]["Value"],
            "solution": item["solution"]["Value"],
            "messages": item["messages"]["Value"]
        })
    df = pd.DataFrame(df_data)
    df.to_parquet(parquet_file, index=False)
    
    # 输出Excel
    excel_data = []
    for idx, item in enumerate(items_to_process, 1):
        excel_data.append({
            "序号": idx,
            "问题": item["question"],
            "总分(50分制)": item["avg_score_50"],
            "逻辑": item.get("avg_logic", 0),
            "深度": item.get("avg_depth", 0),
            "创新": item.get("avg_innovation", 0),
            "准确": item.get("avg_accuracy", 0),
            "完整": item.get("avg_completeness", 0),
            "Question长度": len(item.get("question", "")),
            "Reply长度": len(item.get("combination_1_reply", "")),
        })
    
    df_excel = pd.DataFrame(excel_data)
    excel_file = OUTPUT_BASE.with_suffix('.xlsx')
    df_excel.to_excel(excel_file, index=False)
    
    print(f"完成! 生成了 {len(rlhf_dataset)} 个RLHF样本")
    
    return 0

def main():
    return generate_dataset()

if __name__ == "__main__":
    main()