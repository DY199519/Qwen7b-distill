#!/usr/bin/env python
# coding: utf-8
"""
direct_rlhf_generator.py
直接从原始fusion数据和评分数据生成RLHF格式数据集
"""
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys
import pandas as pd
# ==================== 配置部分 - 修改这里 ====================
BASE_DIR = Path(r"D:\project7\10000final")
FUSION_JSON = BASE_DIR / "doubao-pro-32k_answers_2+1-2-1-9400.json"
SCORES_JSON = BASE_DIR / "grades_doubao-pro-256k_answers_2+1-2-1-9400.json"
OUTPUT_BASE = BASE_DIR / "rlhf_train_top100" # 基础文件名，会生成 .json 和 .parquet
TOP_N = 100 # 取评分最高的前N个
# ===========================================================
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
def load_scores_data(scores_path: Path) -> Dict[str, Dict]:
    """加载评分数据"""
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
       
        logger.info(f"成功加载了 {len(question_scores)} 个问题的评分数据")
        return question_scores
       
    except Exception as e:
        logger.error(f"加载评分数据失败: {e}")
        return {}
def create_rlhf_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    将fusion数据项转换为RLHF格式
    """
    # problem就是原始的question
    problem_text = item.get("question", "")
   
    # 获取fusion_prompt和fusion_reply
    fusion_prompt = item.get("fusion_prompt", "")
    fusion_reply = item.get("fusion_reply", "")
   
    # 拼接solution，明确标记两部分
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
   
    # 构建messages - 包含完整对话
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
        "messages": {"Value": messages} # 直接存储列表，不是JSON字符串
    }
   
    return rlhf_sample
def generate_dataset():
    """生成RLHF格式的数据集"""
   
    # 检查文件是否存在
    if not FUSION_JSON.exists():
        logger.error(f"Fusion数据文件不存在: {FUSION_JSON}")
        return 1
   
    if not SCORES_JSON.exists():
        logger.error(f"评分数据文件不存在: {SCORES_JSON}")
        return 1
   
    # 加载评分数据
    logger.info("正在加载评分数据...")
    question_scores = load_scores_data(SCORES_JSON)
    if not question_scores:
        logger.error("未能加载评分数据")
        return 1
   
    # 加载fusion数据
    logger.info("正在加载fusion数据...")
    with FUSION_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
   
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        logger.error(f"不支持的数据格式: {type(data)}")
        return 1
   
    logger.info(f"加载了 {len(items)} 条fusion数据")
   
    # 匹配评分并过滤数据
    items_with_scores = []
    required_fields = ["question", "fusion_reply"]
   
    for item in items:
        question = item.get("question", "")
       
        # 检查是否有评分
        if question not in question_scores:
            continue
           
        # 检查必要字段
        if not all(item.get(field) for field in required_fields):
            continue
       
        # 检查内容长度
        question_len = len(item.get("question", ""))
        fusion_reply_len = len(item.get("fusion_reply", ""))
       
        if question_len < 10 or fusion_reply_len < 100:
            continue
       
        # 添加评分信息
        item["avg_score_50"] = question_scores[question]["avg_score_50"]
        items_with_scores.append(item)
   
    logger.info(f"找到 {len(items_with_scores)} 条有效数据（有评分且字段完整）")
   
    # 按评分排序，取TOP N
    items_with_scores.sort(key=lambda x: x["avg_score_50"], reverse=True)
    items_to_process = items_with_scores[:TOP_N]
   
    logger.info(f"\n将处理得分最高的 {len(items_to_process)} 条数据")
    logger.info("\nTOP 5 高分问题（50分制）：")
    for i, item in enumerate(items_to_process[:5], 1):
        score = item["avg_score_50"]
        question = item["question"][:80]
        logger.info(f" {i}. 得分: {score:.2f}/50 - {question}...")
   
    # 转换为RLHF格式
    rlhf_dataset = []
    for idx, item in enumerate(items_to_process, 1):
        try:
            rlhf_sample = create_rlhf_sample(item)
           
            if rlhf_sample["problem"]["Value"] and rlhf_sample["solution"]["Value"]:
                rlhf_dataset.append(rlhf_sample)
               
                if idx == 1:
                    logger.info("\n✅ 第一个RLHF样本示例:")
                    logger.info("-" * 60)
                    logger.info(f"Problem (原始问题): {rlhf_sample['problem']['Value'][:200]}...")
                    logger.info("-" * 60)
               
        except Exception as e:
            logger.error(f"处理第 {idx} 个样本失败: {e}")
            continue
   
    if not rlhf_dataset:
        logger.error("未能生成任何有效的RLHF样本")
        return 1
   
    # 确保输出目录存在
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
   
    # 保存为JSON格式
    json_file = OUTPUT_BASE.with_suffix('.json')
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(rlhf_dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ JSON格式已保存到: {json_file}")
   
    # 保存为Parquet格式
    parquet_file = OUTPUT_BASE.with_suffix('.parquet')
    # 将嵌套的字典结构展平为DataFrame
    df_data = []
    for item in rlhf_dataset:
        df_data.append({
            "problem": item["problem"]["Value"],
            "solution": item["solution"]["Value"],
            "messages": item["messages"]["Value"] # 直接存储列表，Arrow会自动处理
        })
    df = pd.DataFrame(df_data)
    df.to_parquet(parquet_file, index=False)
    logger.info(f"✅ Parquet格式已保存到: {parquet_file}")
   
    logger.info(f"\n✅ 成功生成 {len(rlhf_dataset)} 个RLHF训练样本")
   
    # 统计信息
    logger.info("\n📊 数据集统计:")
    problem_lengths = [len(item["problem"]["Value"]) for item in rlhf_dataset]
    solution_lengths = [len(item["solution"]["Value"]) for item in rlhf_dataset]
    logger.info(f" - 样本数量: {len(rlhf_dataset)}")
    logger.info(f" - Problem平均长度: {sum(problem_lengths)/len(problem_lengths):.0f} 字符")
    logger.info(f" - Solution平均长度: {sum(solution_lengths)/len(solution_lengths):.0f} 字符")
   
    # 输出 Excel: 每道题和评分，按照评分从高往下
    scores_data = []
    for item in items_with_scores:
        scores_data.append({
            "question": item["question"],
            "avg_score_50": item["avg_score_50"]
        })
    df_scores = pd.DataFrame(scores_data)
    excel_file = OUTPUT_BASE.with_suffix('.xlsx')
    df_scores.to_excel(excel_file, index=False)
    logger.info(f"✅ Excel格式已保存到: {excel_file}")
   
    return 0
def main():
    parser = argparse.ArgumentParser(description="生成RLHF数据集")
    parser.add_argument("--top", type=int, help="覆盖默认的TOP_N值")
    parser.add_argument("--output", type=str, help="覆盖默认的输出基础文件名")
   
    args = parser.parse_args()
   
    # 如果提供了命令行参数，覆盖默认值
    if args.top:
        global TOP_N
        TOP_N = args.top
   
    if args.output:
        global OUTPUT_BASE
        OUTPUT_BASE = Path(args.output)
   
    logger.info("=" * 70)
    logger.info("RLHF数据集生成器")
    logger.info("=" * 70)
    logger.info(f"Fusion数据: {FUSION_JSON}")
    logger.info(f"评分数据: {SCORES_JSON}")
    logger.info(f"输出文件: {OUTPUT_BASE}.json 和 {OUTPUT_BASE}.parquet")
    logger.info(f"处理数量: TOP {TOP_N}")
    logger.info("=" * 70)
   
    return generate_dataset()
if __name__ == "__main__":
    sys.exit(main())