#!/usr/bin/env python
# coding: utf-8
"""
pairwise_grade_answers_retry.py
-------------------------------
读取 grouped_answers.json
对每个核心问题的 basic_answers 进行两两配对，
调用 o3 模型评分。若解析失败（任一关键项缺失），
自动让模型重新回答，直到成功或达到最大重试次数。
成功标准：
    • 能抓到两行各 6 个数字的分数
    • 能抓到 AB / BA 胜负行
"""

import json, itertools, re, os, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import httpx
from tqdm import tqdm

# ========== OpenAI 初始化 ====================================================
httpx_client = httpx.Client(verify=False)
os.environ["OPENAI_API_KEY"] = "sk-gwwbtmiiMKmF9h3P858dCaC14dB94bCc9bD728BaA6Bf082d"
os.environ["OPENAI_BASE_URL"] = "https://api.vansai.cn/v1"
client = OpenAI(http_client=httpx_client)

# ========== 路径设置 ========================================================
INPUT_PATH  = r"D:\project\grouped_answers.json"
OUTPUT_DIR  = r"D:\project"
OUTPUT_NAME = "pairwise_grades_retry_basic.json"

# ========== Prompt 模板 =====================================================
PROMPT_TMPL = """你是一个专业答题评审员，请对两个答案进行比较 按照以下 5 个维度给每个答案打分： 
1. 逻辑性   2. 深度   3. 创新性   4. 准确性   5. 完整性
每维度满分 100，总分 500。  

### 核心问题
{core_question}

### 回答
A:
{answer_a}

B:
{answer_b}

### 输出要求
- 先输出 2 行分数，每行对应 A、B，格式：总分 逻辑 深度 创新 准确 完整 （仅数字、空格）
- 接着单独一行输出胜负，内容为 AB 或 BA
- 最后一段给出评分理由并引用依据
严格遵守格式，现在开始：
"""

# ---------------------------------------------------------------------------
# 工具：安全提取文本摘要，用于日志
# ---------------------------------------------------------------------------
def safe_extract(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.S)
    return (m.group(1).strip()[:80] if m else "未找到")

# ---------------------------------------------------------------------------
# 解析 GPT 输出
# ---------------------------------------------------------------------------
def parse_response(raw: str) -> Tuple[dict, dict, List[str], str]:
    """若格式不合规就抛 ValueError"""
    keys = ["total", "logic", "depth", "innovation", "accuracy", "completeness"]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # 捕获分数行（至少 6 个数字视为一行）
    score_rows = []
    for l in lines:
        nums = re.findall(r'\d+', l)
        if len(nums) >= 6:
            score_rows.append([int(n) for n in nums[:6]])
        if len(score_rows) == 2:
            break
    if len(score_rows) != 2:
        raise ValueError("找不到两行完整分数")

    # 捕获胜负行
    winner_line = next((l for l in lines if re.fullmatch(r'[ABab]{2}', l)), "")
    if not winner_line:
        raise ValueError("未找到胜负行 AB/BA")

    # 剩余作为评论
    win_idx = lines.index(winner_line)
    commentary = "\n".join(lines[win_idx + 1:]).strip()

    return (
        dict(zip(keys, score_rows[0])),
        dict(zip(keys, score_rows[1])),
        list(winner_line.upper()),
        commentary
    )

# ---------------------------------------------------------------------------
# GPT 调用 + 自动重试
# ---------------------------------------------------------------------------
def ask_and_parse(prompt: str,
                  model: str = "gpt-4o",
                  max_attempts: int = 6,
                  backoff_base: int = 2):
    """循环调用 GPT，直到 parse_response 成功或超出最大次数"""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
            parsed = parse_response(raw)     # 尝试解析
            return *parsed, raw              # 成功则返回 unpack
        except Exception as e:
            wait = backoff_base ** attempt
            print(f"⚠️ 尝试 {attempt}/{max_attempts} 失败：{e} —— {wait}s later retry")
            time.sleep(wait)
    raise RuntimeError("达到最大重试次数仍未获得合规回答")

# ---------------------------------------------------------------------------
# 把 answers 标准化为 [[model,text], ...]
# ---------------------------------------------------------------------------
def normalize(arr):
    if not arr:
        return []
    if isinstance(arr[0], list):
        return arr
    if isinstance(arr[0], str) and len(arr) % 2 == 0:
        return [[arr[i], arr[i+1]] for i in range(0, len(arr), 2)]
    if isinstance(arr[0], str):
        return [[f"model_{i+1}", t] for i, t in enumerate(arr)]
    if isinstance(arr[0], dict):
        return [[d.get("model", f"model_{i+1}"), d.get("text", "")] for i, d in enumerate(arr)]
    return []

# ---------------------------------------------------------------------------
# 评分一个配对
# ---------------------------------------------------------------------------
def grade_pair(core_q, model_a, text_a, model_b, text_b):
    prompt = PROMPT_TMPL.format(core_question=core_q, answer_a=text_a, answer_b=text_b)

    # 日志摘要
    print("\n" + "-"*60)
    print(f"Q: {core_q[:50]}...")
    print(f"A by {model_a[:20]} | B by {model_b[:20]}")

    scores_a, scores_b, winner, commentary, raw = ask_and_parse(prompt)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "winner_order": winner,
        "commentary": commentary,
        "gpt_raw": raw[:800] + "..." if len(raw) > 800 else raw
    }

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        grouped = json.load(f)

    results, total_pairs = {}, 0
    for v in grouped.values():
        total_pairs += len(list(itertools.combinations(normalize(v.get("basic_answers")), 2)))
    print(f"总 basic 对比数：{total_pairs}\n")

    done = 0
    for cq, bundle in tqdm(grouped.items(), desc="核心问题"):
        basics = normalize(bundle.get("basic_answers"))
        pair_res = []
        for (m1, t1), (m2, t2) in itertools.combinations(basics, 2):
            pair_res.append(grade_pair(cq, m1, t1, m2, t2))
            done += 1
            print(f"✓ {done}/{total_pairs} 完成\n")
        results[cq] = {"basic_pairs": pair_res}

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    out_path = Path(OUTPUT_DIR) / OUTPUT_NAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 全部完成，结果写入 {out_path}")

if __name__ == "__main__":
    main()
