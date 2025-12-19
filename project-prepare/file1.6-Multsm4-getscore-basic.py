#!/usr/bin/env python
# coding: utf-8
"""
pairwise_grade_answers_retry.py
-------------------------------
Reads grouped_answers.json
Pairs each core question's basic_answers with each other,
Calls the o3 model for grading. If parsing fails (any key item is missing),
Automatically makes the model retry until success or maximum retries are reached.
Success criteria:
    • Able to capture two lines each with 6 numerical scores
    • Able to capture the AB/BA winner line
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
INPUT_PATH  = r"D:\project\grouped_answers.json"
OUTPUT_DIR  = r"D:\project"
OUTPUT_NAME = "pairwise_grades_retry_basic.json"

# ========== Prompt Template =====================================================
PROMPT_TMPL = """You are a professional answer reviewer. Please compare two answers and score each answer according to the following 5 dimensions:
1. Logicality 2. Depth 3. Innovation 4. Accuracy 5. Completeness
Each dimension is scored out of 100, with a total of 500.

### Core Question
{core_question}

### Answers
A:
{answer_a}

B:
{answer_b}

### Output Requirements
- First, output 2 lines of scores, each corresponding to A and B, in the format: total logicality depth innovation accuracy completeness (numbers and spaces only)
- Then, output the winner in a separate line, which should be AB or BA
- Finally, provide a paragraph explaining the scoring reasons with references
Strictly follow the format. Now begin:
"""

# ---------------------------------------------------------------------------
# Tool: Safely extract text summary for logging
# ---------------------------------------------------------------------------
def safe_extract(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.S)
    return (m.group(1).strip()[:80] if m else "Not found")

# ---------------------------------------------------------------------------
# Parse GPT output
# ---------------------------------------------------------------------------
def parse_response(raw: str) -> Tuple[dict, dict, List[str], str]:
    """Raises ValueError if format is invalid"""
    keys = ["total", "logic", "depth", "innovation", "accuracy", "completeness"]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Capture score lines (at least 6 numbers considered a line)
    score_rows = []
    for l in lines:
        nums = re.findall(r'\d+', l)
        if len(nums) >= 6:
            score_rows.append([int(n) for n in nums[:6]])
        if len(score_rows) == 2:
            break
    if len(score_rows) != 2:
        raise ValueError("Could not find two complete score lines")

    # Capture winner line
    winner_line = next((l for l in lines if re.fullmatch(r'[ABab]{2}', l)), "")
    if not winner_line:
        raise ValueError("Could not find winner line AB/BA")

    # Remaining content as commentary
    win_idx = lines.index(winner_line)
    commentary = "\n".join(lines[win_idx + 1:]).strip()

    return (
        dict(zip(keys, score_rows[0])),
        dict(zip(keys, score_rows[1])),
        list(winner_line.upper()),
        commentary
    )

# ---------------------------------------------------------------------------
# GPT call + automatic retry
# ---------------------------------------------------------------------------
def ask_and_parse(prompt: str,
                  model: str = "gpt-4o",
                  max_attempts: int = 6,
                  backoff_base: int = 2):
    """Loop calling GPT until parse_response succeeds or max attempts exceeded"""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
            parsed = parse_response(raw)     # Attempt parsing
            return *parsed, raw              # Return unpacked if successful
        except Exception as e:
            wait = backoff_base ** attempt
            print(f"⚠️ Attempt {attempt}/{max_attempts} failed: {e} —— Retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Reached maximum retries without obtaining valid response")

# ---------------------------------------------------------------------------
# Standardize answers to [[model,text], ...] format
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
# Grade a pair
# ---------------------------------------------------------------------------
def grade_pair(core_q, model_a, text_a, model_b, text_b):
    prompt = PROMPT_TMPL.format(core_question=core_q, answer_a=text_a, answer_b=text_b)

    # Log summary
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
# Main process
# ---------------------------------------------------------------------------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        grouped = json.load(f)

    results, total_pairs = {}, 0
    for v in grouped.values():
        total_pairs += len(list(itertools.combinations(normalize(v.get("basic_answers")), 2)))
    print(f"Total basic comparisons: {total_pairs}\n")

    done = 0
    for cq, bundle in tqdm(grouped.items(), desc="Core questions"):
        basics = normalize(bundle.get("basic_answers"))
        pair_res = []
        for (m1, t1), (m2, t2) in itertools.combinations(basics, 2):
            pair_res.append(grade_pair(cq, m1, t1, m2, t2))
            done += 1
            print(f"✓ {done}/{total_pairs} completed\n")
        results[cq] = {"basic_pairs": pair_res}

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    out_path = Path(OUTPUT_DIR) / OUTPUT_NAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 All completed. Results written to {out_path}")

if __name__ == "__main__":
    main()
