#!/usr/bin/env python
# coding: utf-8
"""
direct_rlhf_generator.py
Directly generate RLHF format dataset from raw data and scoring data
Using combination_1_prompt and combination_1_reply fields
"""
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys
import pandas as pd

# ==================== Configuration Section - Modify here ====================
BASE_DIR = Path(r"D:\project7\10000final")
BASE_DIR1 = Path(r"D:\qwensft\uploadjson")
FUSION_JSON = BASE_DIR / "deepseek_answers_without_summary3+1-1-9400.json"
SCORES_JSON = BASE_DIR / "grades-3+1-1-9400.json"
OUTPUT_BASE = BASE_DIR / "rlhf_train_top100-3+1"  # Base filename, will generate .json and .parquet
TOP_N = 100  # Take top N highest scored items
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_scores_data(scores_path: Path) -> Dict[str, Dict]:
    """Load scoring data"""
    try:
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
        
        logger.info(f"Successfully loaded scoring data for {len(question_scores)} questions")
        
        # Print score statistics
        if question_scores:
            scores_list = [v["avg_score_50"] for v in question_scores.values()]
            non_zero_scores = [s for s in scores_list if s > 0]
            if non_zero_scores:
                logger.info(f"Score statistics: Highest {max(non_zero_scores):.2f}, Lowest {min(non_zero_scores):.2f}, Average {sum(non_zero_scores)/len(non_zero_scores):.2f}")
        
        return question_scores
        
    except Exception as e:
        logger.error(f"Failed to load scoring data: {e}")
        return {}

def create_rlhf_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert data item to RLHF format
    Use combination_1_prompt and combination_1_reply concatenation as solution
    """
    # problem is the original question
    problem_text = item.get("question", "")
    
    # Get combination_1_prompt and combination_1_reply
    combination_prompt = item.get("combination_1_prompt", "")
    combination_reply = item.get("combination_1_reply", "")
    
    # Concatenate solution, clearly marking both parts (consistent with original format)
    if combination_prompt and combination_reply:
        solution_text = (
            "【Combination Prompt】\n"
            f"{combination_prompt}\n\n"
            "【Combination Reply】\n"
            f"{combination_reply}"
        )
    elif combination_reply:
        solution_text = f"【Combination Reply】\n{combination_reply}"
    else:
        solution_text = ""
    
    # Construct messages - containing complete conversation
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
    """Generate RLHF format dataset"""
    
    # Check if files exist
    if not FUSION_JSON.exists():
        logger.error(f"Data file does not exist: {FUSION_JSON}")
        return 1
    
    if not SCORES_JSON.exists():
        logger.error(f"Scoring data file does not exist: {SCORES_JSON}")
        return 1
    
    # Load scoring data
    logger.info("Loading scoring data...")
    question_scores = load_scores_data(SCORES_JSON)
    if not question_scores:
        logger.error("Failed to load scoring data")
        return 1
    
    # Load data
    logger.info("Loading data...")
    with FUSION_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        logger.error(f"Unsupported data format: {type(data)}")
        return 1
    
    logger.info(f"Loaded {len(items)} data entries")
    
    # Match scores and filter data
    items_with_scores = []
    required_fields = ["question", "combination_1_reply"]  # Modified required fields
    
    no_score_count = 0
    missing_fields_count = 0
    too_short_count = 0
    
    for item in items:
        question = item.get("question", "")
        
        # Check if score exists
        if question not in question_scores:
            no_score_count += 1
            continue
        
        # Check required fields
        missing_fields = [f for f in required_fields if not item.get(f)]
        if missing_fields:
            missing_fields_count += 1
            if missing_fields_count <= 3:  # Only print first 3
                logger.debug(f"Missing fields {missing_fields}: {question[:50]}...")
            continue
        
        # Check content length
        question_len = len(item.get("question", ""))
        reply_len = len(item.get("combination_1_reply", ""))
        
        if question_len < 10 or reply_len < 100:
            too_short_count += 1
            continue
        
        # Add score information
        score_info = question_scores[question]
        item.update(score_info)
        items_with_scores.append(item)
    
    logger.info(f"\nData filtering statistics:")
    logger.info(f"  - Total data: {len(items)}")
    logger.info(f"  - No score: {no_score_count}")
    logger.info(f"  - Missing fields: {missing_fields_count}")
    logger.info(f"  - Content too short: {too_short_count}")
    logger.info(f"  - Valid data: {len(items_with_scores)}")
    
    if not items_with_scores:
        logger.error("No valid data!")
        return 1
    
    # Sort by score, take TOP N
    items_with_scores.sort(key=lambda x: x["avg_score_50"], reverse=True)
    items_to_process = items_with_scores[:TOP_N]
    
    logger.info(f"\nWill process top {len(items_to_process)} highest-scoring entries")
    logger.info("\nTOP 5 high-score questions (50-point scale):")
    for i, item in enumerate(items_to_process[:5], 1):
        score = item["avg_score_50"]
        question = item["question"][:80]
        logger.info(f"  {i}. Score: {score:.2f}/50 - {question}...")
    
    # Convert to RLHF format
    rlhf_dataset = []
    for idx, item in enumerate(items_to_process, 1):
        try:
            rlhf_sample = create_rlhf_sample(item)
            
            if rlhf_sample["problem"]["Value"] and rlhf_sample["solution"]["Value"]:
                rlhf_dataset.append(rlhf_sample)
                
                if idx == 1:
                    logger.info("\n✅ First RLHF sample example:")
                    logger.info("-" * 60)
                    logger.info(f"Problem (original question): {rlhf_sample['problem']['Value'][:200]}...")
                    logger.info("-" * 60)
                    logger.info(f"Solution structure preview:")
                    solution_preview = rlhf_sample['solution']['Value'][:500]
                    logger.info(solution_preview + "...")
                    logger.info("-" * 60)
                
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {e}")
            continue
    
    if not rlhf_dataset:
        logger.error("Failed to generate any valid RLHF samples")
        return 1
    
    # Ensure output directory exists
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON format
    json_file = OUTPUT_BASE.with_suffix('.json')
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(rlhf_dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ JSON format saved to: {json_file}")
    
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
    logger.info(f"✅ Parquet format saved to: {parquet_file}")
    
    logger.info(f"\n✅ Successfully generated {len(rlhf_dataset)} RLHF training samples")
    
    # Statistics
    logger.info("\n📊 Dataset statistics:")
    problem_lengths = [len(item["problem"]["Value"]) for item in rlhf_dataset]
    solution_lengths = [len(item["solution"]["Value"]) for item in rlhf_dataset]
    logger.info(f"  - Number of samples: {len(rlhf_dataset)}")
    logger.info(f"  - Average problem length: {sum(problem_lengths)/len(problem_lengths):.0f} characters")
    logger.info(f"  - Average solution length: {sum(solution_lengths)/len(solution_lengths):.0f} characters")
    logger.info(f"  - Problem min/max: {min(problem_lengths)}/{max(problem_lengths)} characters")
    logger.info(f"  - Solution min/max: {min(solution_lengths)}/{max(solution_lengths)} characters")
    
    # Output Excel: containing detailed scoring information
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
            "Prompt Length": len(item.get("combination_1_prompt", "")),
            "Reply Length": len(item.get("combination_1_reply", "")),
            "Total Solution Length": len(item.get("combination_1_prompt", "")) + len(item.get("combination_1_reply", "")) + 50  # Including tag length
        })
    
    df_excel = pd.DataFrame(excel_data)
    excel_file = OUTPUT_BASE.with_suffix('.xlsx')
    df_excel.to_excel(excel_file, index=False)
    logger.info(f"✅ Excel format saved to: {excel_file}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate RLHF dataset")
    parser.add_argument("--top", type=int, help="Override default TOP_N value")
    parser.add_argument("--output", type=str, help="Override default output base filename")
    
    args = parser.parse_args()
    
    # Override default values if command line arguments are provided
    if args.top:
        global TOP_N
        TOP_N = args.top
    
    if args.output:
        global OUTPUT_BASE
        OUTPUT_BASE = Path(args.output)
    
    logger.info("=" * 70)
    logger.info("RLHF Dataset Generator (Combination_1 version)")
    logger.info("=" * 70)
    logger.info(f"Data file: {FUSION_JSON}")
    logger.info(f"Scoring data: {SCORES_JSON}")
    logger.info(f"Output files: {OUTPUT_BASE}.json / .parquet / .xlsx")
    logger.info(f"Processing quantity: TOP {TOP_N}")
    logger.info("=" * 70)
    
    return generate_dataset()

if __name__ == "__main__":
    sys.exit(main())
