#!/usr/bin/env python
# coding: utf-8
"""
Simplified Fusion Alpaca Dataset Generator
instruction=question, input is empty, output=fusion_reply
"""

import json
import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path(r"D:\project7\10000final")
BASE_DIR1 = Path(r"D:\qwensft\uploadjson")

INPUT_JSON = BASE_DIR / "doubao-pro-32k_answers_2+1-2-1-9400.json"
SCORES_JSON = BASE_DIR / "grades_doubao-pro-256k_answers_2+1-2-1-9400.json"
OUTPUT_DATASET = BASE_DIR1 / "alpaca_dataset_2+1_sft_withoutthinking.json"
OUTPUT_EXCEL = BASE_DIR1 / "alpaca_dataset_2+1_info.xlsx"

ANALYZE_COUNT = 3000

def load_scores(scores_path):
    """Load scoring data"""
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
    
    print(f"Loaded scores for {len(question_scores)} questions")
    return question_scores

def create_alpaca_sample(item):
    """Create Alpaca format sample"""
    return {
        "instruction": item.get("question", ""),
        "input": "",
        "output": item.get("fusion_reply", "")
    }

def main():
    print("Starting Fusion Alpaca dataset generation...")
    
    # Load data
    scores = load_scores(SCORES_JSON)
    
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    
    print(f"Loaded {len(data)} data entries")
    
    # Filter valid data
    valid_items = []
    for item in data:
        question = item.get("question", "")
        reply = item.get("fusion_reply", "")
        
        # Check required fields and lengths
        if question and reply and len(question) >= 10 and len(reply) >= 100 and question in scores:
            item.update(scores[question])
            valid_items.append(item)
    
    print(f"Passed filtering: {len(valid_items)} entries")
    
    # Sort by score and take top entries
    valid_items.sort(key=lambda x: x.get("avg_score_50", 0), reverse=True)
    process_items = valid_items[:ANALYZE_COUNT]
    
    # Generate Alpaca dataset
    alpaca_dataset = []
    excel_data = []
    
    for idx, item in enumerate(process_items, 1):
        sample = create_alpaca_sample(item)
        if sample["instruction"] and sample["output"]:
            alpaca_dataset.append(sample)
            
            excel_data.append({
                "Serial Number": idx,
                "Question": item.get("question", ""),
                "Score (50-point scale)": item.get("avg_score_50", 0),
                "Score (100-point scale)": item.get("avg_score_100", 0),
                "Logic": item.get("avg_logic", 0),
                "Depth": item.get("avg_depth", 0),
                "Innovation": item.get("avg_innovation", 0),
                "Accuracy": item.get("avg_accuracy", 0),
                "Completeness": item.get("avg_completeness", 0),
                "Instruction Length": len(sample["instruction"]),
                "Output Length": len(sample["output"]),
                "A1 Length": len(item.get("A1_third_answer", "")),
                "A2 Length": len(item.get("A2_combination_reply", "")),
                "Fusion Length": len(item.get("fusion_reply", ""))
            })
    
    # Save results
    with OUTPUT_DATASET.open("w", encoding="utf-8") as f:
        json.dump(alpaca_dataset, f, ensure_ascii=False, indent=2)
    
    if excel_data:
        df = pd.DataFrame(excel_data)
        df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"Completed! Generated {len(alpaca_dataset)} samples")
    print(f"JSON: {OUTPUT_DATASET}")
    print(f"Excel: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
