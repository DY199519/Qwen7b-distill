#!/usr/bin/env python
# coding: utf-8
"""
flatten_and_group_by_model.py
-----------------------------
Read one or two JSON files (supporting the structure {"model_name":..., "results":[...]}),
filter specified models, and group by core_question:
    * basic_answers       -> list [model_name, text]
    * answers_with_context-> list [model_name, text]
Write to grouped_answers.json
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# ========= Path Configuration ========================================================
INPUT_FILE_1 = r"D:\project\finalmut_fixed_all.json"
INPUT_FILE_2 = r""   # Set to "" if not available

OUTPUT_DIR   = r"D:\project"
OUTPUT_NAME  = "grouped_answers.json"

SKIP_MODEL   = "qwen3-235b-a22b"                       # Model to be excluded
# ==========================================================================


# ---------- Flatten nested results ------------------------------------------
def flatten_records(raw: List[dict]) -> List[dict]:
    flat: List[dict] = []
    for item in raw:
        if "core_question" in item:          # Already a flat entry
            flat.append(item)
        else:                                # Outer layer + inner results
            model = item.get("model_name", "unknown_model")
            for res in item.get("results", []):
                rec = res.copy()
                rec.setdefault("model_name", model)
                flat.append(rec)
    return flat


# ---------- Read and flatten -----------------------------------------------------
def load_json_file(path_str: str) -> List[dict]:
    if not path_str:
        return []
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return flatten_records(data if isinstance(data, list) else [data])


# ---------- Grouping: retain model + text ---------------------------------------
def regroup_split(list_a: List[dict],
                  list_b: List[dict]) -> Dict[str, Dict[str, List[List[str]]]]:
    """
    Returns in the form:
    {
      core_q1: {
        "basic_answers": [ [model, text], ... ],
        "answers_with_context": [ [model, text], ... ]
      },
      ...
    }
    """
    grouped: Dict[str, Dict[str, List[List[str]]]] = defaultdict(
        lambda: {"basic_answers": [], "answers_with_context": []})

    for ent in list_a + list_b:
        model = ent.get("model_name")
        if model == SKIP_MODEL:
            continue

        cq = ent.get("core_question")

        ba = ent.get("basic_answer")
        if ba:
            grouped[cq]["basic_answers"].append([model, ba])

        awc = ent.get("answer_with_context")
        if awc:
            grouped[cq]["answers_with_context"].append([model, awc])

    return grouped


# ---------- Main process ---------------------------------------------------------
def main() -> None:
    data1 = load_json_file(INPUT_FILE_1)
    data2 = load_json_file(INPUT_FILE_2)

    grouped = regroup_split(data1, data2)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / OUTPUT_NAME
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"Completed! Written to {out_path}  —— Total {len(grouped)} core_question entries")


if __name__ == "__main__":
    main()
