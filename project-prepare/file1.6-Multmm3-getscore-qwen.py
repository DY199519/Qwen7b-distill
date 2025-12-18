#!/usr/bin/env python
# coding: utf-8
"""
pairwise_grade_top234.py
------------------------
Read top2 / top3 / top4 answers for each question from `qwen_combined_answers.json`,
perform all pairwise comparison scoring (3 groups in total), call GPT-4o for scoring and determine the winner.
"""

import json, itertools, re, os, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import httpx
from tqdm import tqdm

# ========== OpenAI Initialization ====================================================
httpx_client = httpx.Client(verify=False)
os.environ["OPENAI_API_KEY"] = "sk-gwwbtmiiMKmF9h3P858dCaC14dB94bCc9bD728BaA6Bf082d"
os.environ["OPENAI_BASE_URL"] = "https://api.vansai.cn/v1"
client = OpenAI(http_client=httpx_client)

# ========== Path Settings ========================================================
INPUT_PATH  = r"D:\project\qwen_combined_answers.json"
OUTPUT_DIR  = r"D:\project"
OUTPUT_NAME = "pairwise_grades_top234_qwen.json"

# ========== Prompt Template =====================================================
PROMPT_TMPL = """You are a professional answer reviewer. Please compare two answers and score each answer according to the following 5 dimensions:
1. Logicality  2. Depth  3. Innovation  4. Accuracy  5. Completeness
Each dimension is scored out of 100, with a total score of 500.

### Core Question
{core_question}

### Answers
A:
{answer_a}

B:
{answer_b}

### Output Requirements
- First, output 2 lines of scores, each corresponding to A and B, in the format: total logicality depth innovation accuracy completeness (only numbers and spaces)
- Then, output the winner in a separate line, which should be AB or BA
- Finally, provide a paragraph explaining the scoring reasons with references
Strictly follow the format. Now start:
"""

# ========== Utility Functions ========================================================
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
        raise ValueError("Could not find two complete lines of scores")

    winner_line = next((l for l in lines if re.fullmatch(r"[ABab]{2}", l)), "")
    if not winner_line:
        raise ValueError("Could not find winner line AB/BA")
    win_idx = lines.index(winner_line)
    commentary = "\n".join(lines[win_idx + 1:]).strip()

    return (
        dict(zip(keys, score_rows[0])),
        dict(zip(keys, score_rows[1])),
        list(winner_line.upper()),
        commentary
    )

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
            print(f"⚠️ Attempt {attempt}/{max_attempts} failed: {e} —— Retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Reached maximum number of retries without obtaining a valid response")

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

# ========== Main Process ===========================================================
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = {}

    # Analyze each question and count total pairs
    total_pairs = 0
    valid_items = []

    for entry in items:
        candidates = [
            ("top2", entry.get("top2_reply", "")),
            ("top3", entry.get("top3_reply", "")),
            ("top4", entry.get("top4_reply", ""))
            # direct_reply has been removed
        ]
        answers = [(name, text.strip()) for name, text in candidates if text and text.strip()]
        if len(answers) >= 2:
            valid_items.append((entry, answers))
            total_pairs += len(answers) * (len(answers) - 1) // 2

    print(f"🌟 A total of {len(valid_items)} questions entered pairwise scoring, with {total_pairs} pairs in total")

    done = 0
    with tqdm(total=total_pairs, desc="Pairwise Scoring Progress") as pbar:
        for entry, answers in valid_items:
            question = entry.get("question") or entry.get("direct_prompt")
            ans_pairs = []
            for (m1, t1), (m2, t2) in itertools.combinations(answers, 2):
                ans_pairs.append(grade_pair(question, m1, t1, m2, t2))
                done += 1
                pbar.update(1)
            results[question] = {"top_pairs": ans_pairs}

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    out_path = Path(OUTPUT_DIR) / OUTPUT_NAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 All completed, {done} pairs have been scored. Results written to: {out_path}")

# ========== Startup ============================================================
if __name__ == "__main__":
    main()
