#!/usr/bin/env python
# coding: utf-8
"""
Simplified RLHF Dataset Generator
Use the combination_1_reply field while keeping the RLHF format unchanged
"""
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# ==================== Configuration Section ====================
BASE_DIR = Path(r"D:\project7\10000final")
BASE_DIR1 = Path(r"D:\qwensft\uploadjson")
FUSION_JSON = BASE_DIR / "deepseek_answers_without_summary3+1-1-9400.json"
SCORES_JSON = BASE_DIR / "grades-3+1-1-9400.json"
OUTPUT_BASE = BASE_DIR1 / "rlhf_train_top1000-3+1"
TOP_N = 1000
# =============================================================

def load_scores_data(scores_path: Path) -> Dict[str, Dict]:
    """Load scoring data"""
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
    Convert data items to RLHF format
    Using question and combination_1_reply
    """
    problem_text = item.get("question", "")
    solution_text = item.get("combination_1_reply", "")
    
    # Construct messages
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
    
    # Construct RLHF format sample
    rlhf_sample = {
        "problem": {"Value": problem_text},
        "solution": {"Value": solution_text},
        "messages": {"Value": messages}
    }
    
    return rlhf_sample

def generate_dataset():
    """Generate dataset in RLHF format"""
    
    # Load scoring data
    question_scores = load_scores_data(SCORES_JSON)
    
    # Load data
    with FUSION_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return 1
    
    # Match scores and filter data
    items_with_scores = []
    required_fields = ["question", "combination_1_reply"]
    
    for item in items:
        question = item.get("question", "")
        
        # Check if there is a score
        if question not in question_scores:
            continue
        
        # Check required fields
        missing_fields = [f for f in required_fields if not item.get(f)]
        if missing_fields:
            continue
        
        # Check content length
        question_len = len(item.get("question", ""))
        reply_len = len(item.get("combination_1_reply", ""))
        
        if question_len < 10 or reply_len < 100:
            continue
        
        # Add score information
        score_info = question_scores[question]
        item.update(score_info)
        items_with_scores.append(item)
    
    # Sort by score, take TOP N
    items_with_scores.sort(key=lambda x: x["avg_score_50"], reverse=True)
    items_to_process = items_with_scores[:TOP_N]
    
    # Convert to RLHF format
    rlhf_dataset = []
    for idx, item in enumerate(items_to_process, 1):
        try:
            rlhf_sample = create_rlhf_sample(item)
            
            if rlhf_sample["problem"]["Value"] and rlhf_sample["solution"]["Value"]:
                rlhf_dataset.append(rlhf_sample)
                
        except Exception as e:
            continue
    
    # Ensure output directory exists
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON format
    json_file = OUTPUT_BASE.with_suffix('.json')
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(rlhf_dataset, f, ensure_ascii=False, indent=2)
    
    # Save as Parquet format
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
    
    # Output Excel
    excel_data = []
    for idx, item in enumerate(items_to_process, 1):
        excel_data.append({
            "Serial Number": idx,
            "Question": item["question"],
            "Total Score (50-point scale)": item["avg_score_50"],
            "Logic": item.get("avg_logic", 0),
            "Depth": item.get("avg_depth", 0),
            "Innovation": item.get("avg_innovation", 0),
            "Accuracy": item.get("avg_accuracy", 0),
            "Completeness": item.get("avg_completeness", 0),
            "Question Length": len(item.get("question", "")),
            "Reply Length": len(item.get("combination_1_reply", "")),
        })
    
    df_excel = pd.DataFrame(excel_data)
    excel_file = OUTPUT_BASE.with_suffix('.xlsx')
    df_excel.to_excel(excel_file, index=False)
    
    print(f"Completed! Generated {len(rlhf_dataset)} RLHF samples")
    
    return 0

def main():
    return generate_dataset()

if __name__ == "__main__":
    main()
