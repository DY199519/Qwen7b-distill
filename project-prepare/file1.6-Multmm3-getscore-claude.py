#!/usr/bin/env python
# coding: utf-8
"""
pairwise_grade_top234.py
------------------------
从 `claude_combined_answers.json` 读取每个问题的 top2 / top3 / top4 / direct 回答，
两两配对（共 6 组），调用 gpt-4o 进行 5×100 打分并判定胜负。
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
INPUT_PATH  = r"D:\project\claude_combined_answers.json"
OUTPUT_DIR  = r"D:\project"
OUTPUT_NAME = "pairwise_grades_top234_claude.json"

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

# ========== 解析 GPT 响应 ===================================================
def parse_response(raw: str) -> Tuple[dict, dict, List[str], str]:
    keys = ["total", "logic", "depth", "innovation", "accuracy", "completeness"]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    score_rows = []

    for l in lines:
        nums = re.findall(r"\d+", l)
        if len(nums) >= 6:
            score_rows.append([int(n) for n in nums[:6]])
        if len(score_rows) == 2:
            break
    if len(score_rows) != 2:
        raise ValueError("找不到两行完整分数")

    winner_line = next((l for l in lines if re.fullmatch(r"[ABab]{2}", l)), "")
    if not winner_line:
        raise ValueError("未找到胜负行 AB/BA")
    win_idx = lines.index(winner_line)
    commentary = "\n".join(lines[win_idx + 1:]).strip()

    return (
        dict(zip(keys, score_rows[0])),
        dict(zip(keys, score_rows[1])),
        list(winner_line.upper()),
        commentary
    )

# ========== GPT 打分并自动重试 ===============================================
def ask_and_parse(prompt: str,
                  model: str = "gpt-4o",
                  max_attempts: int = 6,
                  backoff_base: int = 2):
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
            parsed = parse_response(raw)
            return *parsed, raw
        except Exception as e:
            wait = backoff_base ** attempt
            print(f"⚠️ 尝试 {attempt}/{max_attempts} 失败：{e} —— {wait}s later retry")
            time.sleep(wait)
    raise RuntimeError("达到最大重试次数仍未获得合规回答")

# ========== 单个配对评分 =====================================================
def grade_pair(core_q: str, model_a: str, text_a: str, model_b: str, text_b: str):
    prompt = PROMPT_TMPL.format(core_question=core_q, answer_a=text_a, answer_b=text_b)
    print("\n" + "-"*60)
    print(f"Q: {core_q[:80]}...")
    print(f"A by {model_a} | B by {model_b}")
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

# ========== 主流程 ===========================================================
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = {}

    # 动态统计总配对数
    total_pairs = 0
    valid_items = []

    for entry in items:
        candidates = [
            ("top2", entry.get("top2_reply", "")),
            ("top3", entry.get("top3_reply", "")),
            ("top4", entry.get("top4_reply", "")),
            ("direct", entry.get("direct_reply", ""))  # ✅ 修复字段名
        ]
        answers = [(name, text.strip()) for name, text in candidates if text and text.strip()]
        if len(answers) >= 2:
            valid_items.append((entry, answers))
            total_pairs += len(answers) * (len(answers) - 1) // 2

    print(f"🌟 共 {len(valid_items)} 道题进入配对评分，总计 {total_pairs} 个配对")

    done = 0
    with tqdm(total=total_pairs, desc="配对评分进度") as pbar:
        for entry, answers in valid_items:
            question = entry.get("question") or entry.get("direct_prompt")
            ans_pairs = []

            # 优先处理 direct vs others
            direct_item = next(((name, text) for name, text in answers if name == "direct"), None)
            others = [(name, text) for name, text in answers if name != "direct"]

            if direct_item:
                for other in others:
                    m1, t1 = direct_item
                    m2, t2 = other
                    print(f"🔎 优先处理：{m1} vs {m2}")
                    ans_pairs.append(grade_pair(question, m1, t1, m2, t2))
                    done += 1
                    pbar.update(1)

            for (m1, t1), (m2, t2) in itertools.combinations(others, 2):
                print(f"🔄 常规处理：{m1} vs {m2}")
                ans_pairs.append(grade_pair(question, m1, t1, m2, t2))
                done += 1
                pbar.update(1)

            results[question] = {"top_pairs": ans_pairs}

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    out_path = Path(OUTPUT_DIR) / OUTPUT_NAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 全部完成，{done} 个配对已评分，结果写入：{out_path}")

# ========== 启动 ============================================================
if __name__ == "__main__":
    main()
