#!/usr/bin/env python
# coding: utf-8
"""
fusion_reply_grade.py
------------------------------------
Read fused answer JSON, automatically grade fusion_reply and continuously save progress.
"""

import json, re, os, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import httpx
from openai import OpenAI
from tqdm import tqdm

# ========== File path configuration (easy to modify) ==========================================
INPUT_PATH = r"D:\project7\MM\result\2+1\doubao-pro-32k_answers_2+1-2-7800-8100.json"
OUTPUT_DIR = r"D:\project7\MM\result"
OUTPUT_FILENAME = "grades_doubao-pro-256k_answers_2+1-2-7800-8100.json"  # Custom output file name

# ========== Other configuration options =====================================================
SAVE_INTERVAL = 1  # Save every N questions

# ========== OpenAI initialization ====================================================
httpx_client = httpx.Client(verify=False)
os.environ["OPENAI_API_KEY"]  = "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa"
os.environ["OPENAI_BASE_URL"] = "https://vir.vimsai.com/v1"
client = OpenAI(http_client=httpx_client)

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# ========== Prompt template =====================================================
PROMPT_TMPL = """
You are a professional answer reviewer. Please grade the following answer according to the following 5 dimensions:
1. Logicality  2. Depth  3. Innovation  4. Accuracy  5. Completeness
Each dimension is scored out of 5, with a total score of 25.

Scoring format example (strictly copy the numbers and spaces):
15 3 3 3 3 3
(Follow this line with the scoring reason paragraphs)

### Question
{question}

### Answer
{answer}

### Output requirements
- The first line should **only contain 6 numbers**, separated by spaces, in the order: total score, logic, depth, innovation, accuracy, completeness
- Do not write any text, units or punctuation
- Start writing detailed scoring reasons from the second line (at least 2 paragraphs)

1. Logicality —— Whether the argument structure and causal chain are rigorous;
2. Depth —— Whether academic concepts / data / multi-angle analysis are cited;
3. Innovation —— Whether new viewpoints or insights that are not clichés are put forward;
4. Accuracy —— Whether facts, data and concepts are correct;
5. Completeness —— Whether all key points of the question are fully answered.

**Hard and fast rules for scoring** (must be implemented, please be cautious with high scores):
| Single dimension score | Evaluation criteria (examples) |
|----------|-----------------|
| 5 | Almost no flaws, only minor details can be criticized |
| 4  | 1–2 minor flaws |
| 3  | **Obvious flaws** or missing key points |
| 2  | Multiple flaws, more than 2 argument/factual errors |
| 0–1  | Key logic is invalid, or factual errors are serious |
Strictly follow the format, now start:
"""

# ---------------------------------------------------------------------------
def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Read JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return []

# ---------------------------------------------------------------------------
def load_existing_results(output_file: Path) -> Tuple[Dict[str, Any] | None, set]:
    """Load existing scoring progress"""
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            done = {r["question"] for r in data.get("detailed_results", [])}
            print(f"📂 Existing progress: {len(done)} questions")
            return data, done
        except Exception as e:
            print(f"⚠️ Failed to read progress file: {e}")
    return None, set()

# ---------------------------------------------------------------------------
def save_progress(data: Dict[str, Any], output_file: Path):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Progress saved to {output_file}")
    except Exception as e:
        print(f"❌ Save failed: {e}")

# ---------------------------------------------------------------------------
def parse_response(raw: str) -> Tuple[Dict[str, int], str]:
    """Parse GPT output"""
    keys = ["total", "logic", "depth", "innovation", "accuracy", "completeness"]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Find the score string
    score_line = next((l for l in lines if len(re.findall(r"\d+", l)) >= 6), None)
    if not score_line:
        raise ValueError("Could not find complete score line")
    nums = list(map(int, re.findall(r"\d+", score_line)[:6]))

    commentary = "\n".join(lines[lines.index(score_line) + 1:]).strip()
    if not commentary:
        raise ValueError("Missing scoring reasons")

    return dict(zip(keys, nums)), commentary

# ---------------------------------------------------------------------------
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
            scores, detail = parse_response(raw)
            return scores, detail, raw
        except Exception as e:
            wait = backoff_base ** attempt
            print(f"⚠️ Attempt {attempt}/{max_attempts} failed: {e}, retrying in {wait}s")
            time.sleep(wait)
    return None

# ---------------------------------------------------------------------------
def grade_single(question: str, answer: str, trials: int = 3):
    prompt = PROMPT_TMPL.format(question=question, answer=answer)
    all_scores, all_cmts, raws = [], [], []

    for t in range(trials):
        res = ask_and_parse(prompt)
        if not res:
            print(f"  Trial {t+1} scoring failed")
            continue
        score, cmt, raw = res
        all_scores.append(score); all_cmts.append(cmt); raws.append(raw)
        print(f"  Trial {t+1} score: {score['total']}/50")

    if not all_scores:
        return None

    avg = {k: round(sum(s[k] for s in all_scores) / len(all_scores), 2)
           for k in all_scores[0]}
    avg100 = round(avg["total"] * 2, 2)

    return {
        "question": question,
        "avg_scores": avg,
        "avg_score_100": avg100,
        "num_valid_trials": len(all_scores),
        "all_scores": all_scores,
        "all_commentaries": all_cmts,
        "all_gpt_raws": raws
    }

# ---------------------------------------------------------------------------
def grade_fusion_replies(records: List[Dict[str, Any]]):
    print(f"\n===== Grading fusion_reply =====")
    
    # Use configured output file name
    output_file = Path(OUTPUT_DIR) / OUTPUT_FILENAME
    
    prev, done_set = load_existing_results(output_file)
    results = prev.get("detailed_results", []) if prev else []

    # Filter records with fusion_reply
    items = [d for d in records if "fusion_reply" in d and d["fusion_reply"]]
    pending = [d for d in items if d["question"] not in done_set]
    
    print(f"Total {len(items)} questions | Pending grading {len(pending)} questions")

    # Main loop
    all_totals, all_totals100 = [], []

    # Incorporate old scores
    if prev:
        all_totals = [r["avg_scores"]["total"] for r in results]
        all_totals100 = [r["avg_score_100"] for r in results]

    for idx, item in enumerate(pending, 1):
        q = item["question"]
        a = item["fusion_reply"]
        
        print(f"\n[{idx}/{len(pending)}] {q[:40]}...")
        res = grade_single(q, a)
        
        if res:
            # Save additional metadata
            res["type"] = "fusion_reply"
            if "third_model" in item:
                res["third_model"] = item["third_model"]
            if "A1_third_answer" in item:
                res["has_third_answer"] = True
            if "A2_combination_reply" in item:
                res["has_combination_reply"] = True
                
            results.append(res)
            all_totals.append(res["avg_scores"]["total"])
            all_totals100.append(res["avg_score_100"])

        if idx % SAVE_INTERVAL == 0:
            stats = {
                "type": "fusion_reply",
                "input_file": INPUT_PATH,
                "total_questions": len(items),
                "valid_grades": len(all_totals),
                "total_average": round(sum(all_totals)/len(all_totals), 2),
                "total_average_100": round(sum(all_totals100)/len(all_totals100), 2)
            }
            save_progress({"statistics": stats, "detailed_results": results}, output_file)

    # Final statistics
    if all_totals:
        stats = {
            "type": "fusion_reply",
            "input_file": INPUT_PATH,
            "total_questions": len(items),
            "valid_grades": len(all_totals),
            "total_average": round(sum(all_totals)/len(all_totals), 2),
            "total_average_100": round(sum(all_totals100)/len(all_totals100), 2)
        }
        save_progress({"statistics": stats, "detailed_results": results}, output_file)
        print(f"\n📊 fusion_reply average {stats['total_average']}/50 "
              f"(100-point scale {stats['total_average_100']})")

# ---------------------------------------------------------------------------
def main():
    data = read_json_file(INPUT_PATH)
    if not data:
        return
    
    grade_fusion_replies(data)

# ---------------------------------------------------------------------------
if __name__ == "__main__":

    main()
