#!/usr/bin/env python
# coding: utf-8
"""
direct_rlhf_generator.py
Directly generate an RLHF - format dataset from the original fusion data and scoring data.
"""
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys
import pandas as pd
# ==================== configuration ====================
BASE_DIR = Path(r"D:\project7\10000final")
FUSION_JSON = BASE_DIR / "doubao-pro-32k_answers_2+1-2-1-9400.json"
SCORES_JSON = BASE_DIR / "grades_doubao-pro-256k_answers_2+1-2-1-9400.json"
OUTPUT_BASE = BASE_DIR / "rlhf_train_top100" # basic filename，generate .json and .parquet format
TOP_N = 100 # get the top N with the high score
# ===========================================================
# configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
def load_scores_data(scores_path: Path) -> Dict[str, Dict]:
    """load the scoring data"""
    try:
        with scores_path.open("r", encoding="utf-8") as f:
            scores_data = json.load(f)
       
        question_scores = {}
        detailed_results = scores_data.get("detailed_results", [])
       
        for item in detailed_results:
            question = item.get("question", "")
            if question:
                avg_score_50 = item.get("avg_scores", {}).get("total", 0)
                question_scores[question] = {
                    "avg_score_50": avg_score_50,
                    "avg_scores": item.get("avg_scores", {}),
                    "num_valid_trials": item.get("num_valid_trials", 0),
                }
       
        logger.info(f"successfully load  {len(question_scores)} data")
        return question_scores
       
    except Exception as e:
        logger.error(f"failure: {e}")
        return {}
def create_rlhf_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """

    """
    # problem is the original question
    problem_text = item.get("question", "")
   
    # get fusion_prompt和fusion_reply
    fusion_prompt = item.get("fusion_prompt", "")
    fusion_reply = item.get("fusion_reply", "")
   
    # concatenate solution
    if fusion_prompt and fusion_reply:
        solution_text = (
            "【Fusion Prompt】\n"
            f"{fusion_prompt}\n\n"
            "【Fusion Reply】\n"
            f"{fusion_reply}"
        )
    elif fusion_reply:
        solution_text = f"【Fusion Reply】\n{fusion_reply}"
    else:
        solution_text = ""
   
    # build messages
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
   
    # create RLHF demo
    rlhf_sample = {
        "problem": {"Value": problem_text},
        "solution": {"Value": solution_text},
        "messages": {"Value": messages} # 
    }
   
    return rlhf_sample
def generate_dataset():
   
    # file is or not?
    if not FUSION_JSON.exists():
        logger.error(f"Fusion N/A: {FUSION_JSON}")
        return 1
   
    if not SCORES_JSON.exists():
        logger.error(f"Scoring data N/A: {SCORES_JSON}")
        return 1
   
    # loading data
    logger.info("Loading...")
    question_scores = load_scores_data(SCORES_JSON)
    if not question_scores:
        logger.error("do not load")
        return 1
   
    # loading fusion data
    logger.info("Loading fusion data...")
    with FUSION_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
   
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        logger.error(f"do not support : {type(data)}")
        return 1
   
    logger.info(f"loaded {len(items)}  fusion data")
   
    # match the scoring data and select
    items_with_scores = []
    required_fields = ["question", "fusion_reply"]
   
    for item in items:
        question = item.get("question", "")
       
        # determine scoring 
        if question not in question_scores:
            continue
           
        # 
        if not all(item.get(field) for field in required_fields):
            continue
       
        # len(question)
        question_len = len(item.get("question", ""))
        fusion_reply_len = len(item.get("fusion_reply", ""))
       
        if question_len < 10 or fusion_reply_len < 100:
            continue
       
        # append scoring info
        item["avg_score_50"] = question_scores[question]["avg_score_50"]
        items_with_scores.append(item)
   
    logger.info(f"get {len(items_with_scores)} valid data (with scores and complete fields)")
   
    #  rank and TOP N
    items_with_scores.sort(key=lambda x: x["avg_score_50"], reverse=True)
    items_to_process = items_with_scores[:TOP_N]
   
    logger.info(f"\nProcess the {len(items_to_process)} pieces of data with the highest scores")
    logger.info("\nTOP 5 High - score questions (on a 50 - point scale):")
    for i, item in enumerate(items_to_process[:5], 1):
        score = item["avg_score_50"]
        question = item["question"][:80]
        logger.info(f" {i}. scoring: {score:.2f}/50 - {question}...")
   
    # switch to RLHF 
    rlhf_dataset = []
    for idx, item in enumerate(items_to_process, 1):
        try:
            rlhf_sample = create_rlhf_sample(item)
           
            if rlhf_sample["problem"]["Value"] and rlhf_sample["solution"]["Value"]:
                rlhf_dataset.append(rlhf_sample)
               
                if idx == 1:
                    logger.info("\n✅ The first RLHF sample example:")
                    logger.info("-" * 60)
                    logger.info(f"Problem (original ques): {rlhf_sample['problem']['Value'][:200]}...")
                    logger.info("-" * 60)
               
        except Exception as e:
            logger.error(f"Failed to process the {idx}th sample: {e}")
            continue
   
    if not rlhf_dataset:
        logger.error("Failed to generate any valid RLHF samples")
        return 1
   
    # Ensure the output directory exists
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
   
    # save to JSON
    json_file = OUTPUT_BASE.with_suffix('.json')
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(rlhf_dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ JSON has saved: {json_file}")
   
    # save to Parquet
    parquet_file = OUTPUT_BASE.with_suffix('.parquet')
    # switch to DataFrame
    df_data = []
    for item in rlhf_dataset:
        df_data.append({
            "problem": item["problem"]["Value"],
            "solution": item["solution"]["Value"],
            "messages": item["messages"]["Value"]
        })
    df = pd.DataFrame(df_data)
    df.to_parquet(parquet_file, index=False)
    logger.info(f"✅ Parquet has saved: {parquet_file}")
   
    logger.info(f"\n✅ Successfully generated {len(rlhf_dataset)} RLHF training samples")
   
    # statistical info
    logger.info("\n📊 dataset:")
    problem_lengths = [len(item["problem"]["Value"]) for item in rlhf_dataset]
    solution_lengths = [len(item["solution"]["Value"]) for item in rlhf_dataset]
    logger.info(f" - Sample size: {len(rlhf_dataset)}")
    logger.info(f" - Problem average len: {sum(problem_lengths)/len(problem_lengths):.0f} character")
    logger.info(f" - Solution average len: {sum(solution_lengths)/len(solution_lengths):.0f} character")
   
    # output Excel: or each question and its corresponding score, arrange them in descending order of scores
    scores_data = []
    for item in items_with_scores:
        scores_data.append({
            "question": item["question"],
            "avg_score_50": item["avg_score_50"]
        })
    df_scores = pd.DataFrame(scores_data)
    excel_file = OUTPUT_BASE.with_suffix('.xlsx')
    df_scores.to_excel(excel_file, index=False)
    logger.info(f"✅saved to: {excel_file}")
   
    return 0
def main():
    parser = argparse.ArgumentParser(description="generate RLHF dataset")
    parser.add_argument("--top", type=int, help="Override the default TOP_N value")
    parser.add_argument("--output", type=str, help="Override the default base output file name")
   
    args = parser.parse_args()
   
    if args.top:
        global TOP_N
        TOP_N = args.top
   
    if args.output:
        global OUTPUT_BASE
        OUTPUT_BASE = Path(args.output)
   
    logger.info("=" * 70)
    logger.info("RLHF data generator")
    logger.info("=" * 70)
    logger.info(f"Fusion data: {FUSION_JSON}")
    logger.info(f"scoring dataset: {SCORES_JSON}")
    logger.info(f"output file: {OUTPUT_BASE}.json 和 {OUTPUT_BASE}.parquet")
    logger.info(f"process data: TOP {TOP_N}")
    logger.info("=" * 70)
   
    return generate_dataset()
if __name__ == "__main__":

    sys.exit(main())
