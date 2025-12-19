#!/usr/bin/env python
# coding: utf-8
"""
generate_multi_answers.py
-------------------------
Reads generated_results_multi_model.json,
generates (or reuses) basic_answer and answer_with_context for each model/each core question,
and saves them to final_ans_multi_sm.json.

If the corresponding basic_answer already exists in grouped_answers.json, it will be reused
to avoid duplicate generation; the API will only be called when it is missing.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI
import httpx


# ========= Paths & Constants ========================================================
INPUT_FILE     = Path(r"D:\project\generated_results_multi_model.json")
BASIC_ANS_FILE = Path(r"D:\project\grouped_answers.json")           # ← Existing basic answers
OUTPUT_DIR     = Path(r"D:\project")
OUTPUT_FILE    = OUTPUT_DIR / "final_ans_multi_sm.json"


# ========= Model Configuration (Complete) ===================================================
models_config: Dict[str, Dict[str, str]] = {
    "gemini-2.5-flash-preview-04-17-thinking": {
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/"
    },
    "grok-3-beta": {
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/"
    },
    "doubao-pro-256k": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "moonshot-v1-8k": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "deepseek-v3": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "hunyuan-turbo": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "qwen2.5-72b-instruct": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    }
}


# ========= Utility Functions ===========================================================
def load_basic_answer_map(filepath: Path) -> Dict[str, Dict[str, str]]:
    """Reads grouped_answers.json and returns a mapping: {core_question: {model_name: basic_answer, ...}, ...}"""
    if not filepath.exists():
        return {}

    with filepath.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    q2model2ans: Dict[str, Dict[str, str]] = {}
    for core_q, blocks in raw.items():
        basic_list = blocks.get("basic_answers", [])
        for model_name, answer in basic_list:
            q2model2ans.setdefault(core_q, {})[model_name] = answer
    return q2model2ans


def get_completion(client: OpenAI,
                   model_name: str,
                   prompt: str,
                   attempt: int = 1,
                   max_attempts: int = 10) -> str:
    """OpenAI API request with retry mechanism."""
    try:
        rsp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return rsp.choices[0].message.content
    except Exception as e:
        if attempt < max_attempts:
            print(f"⚠️ Request failed, retrying ({attempt}/{max_attempts})... Reason: {e}")
            time.sleep(2)
            return get_completion(client, model_name, prompt,
                                  attempt + 1, max_attempts)
        print(f"❌ Final failure: {e}")
        return f"ERROR: {str(e)}"


# ========= Main Process =============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    basic_answer_map = load_basic_answer_map(BASIC_ANS_FILE)
    all_results = []

    for model_block in raw_data:
        model_name = model_block["model_name"]
        print(f"\n{'='*60}\n🔍 Starting processing model: {model_name}\n{'='*60}")

        config = models_config.get(model_name)
        if not config:
            print(f"⚠️ No model configuration found, skipping: {model_name}")
            continue

        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            http_client=httpx.Client(verify=False)
        )

        model_results = []

        for entry in model_block["results"]:
            core_q_data = entry.get("core_question", [])
            if not core_q_data or not isinstance(core_q_data, list):
                continue

            core_question = core_q_data[0]
            sum_list_data = entry.get("sum_list", [])
            print(f"\n➡️ Question: {core_question}")

            # -------- Construct context information (skip parts exceeding 30 characters) --------
            context_parts = []
            for grp in sum_list_data:
                if not grp or not isinstance(grp, list) or len(grp) < 2:
                    continue
                part = f"{grp[0]} field: {'、'.join(grp[1:])}"  #保留中文顿号，因为是内容分隔符
                if len(part) > 30:
                    print(f"⚠️ Skipping overlong factor: {part}")
                    continue
                context_parts.append(part)
            context_str = "；".join(context_parts)  #保留中文分号，因为是内容分隔符

            # -------- 1. basic answer --------
            basic_ans = basic_answer_map.get(core_question, {}).get(model_name)
            if basic_ans is None:
                prompt_basic = f"Please answer: \"{core_question}\"."
                basic_ans = get_completion(client, model_name, prompt_basic)
                print("    (No existing basic_answer found, API called to supplement)")

            # -------- 2. answer_with_context --------
            prompt_context = (f"Please answer: \"{core_question}\", combining the following information, "
                              f"answer with reference to the following important factors, sorted by importance: {context_str}.")
            context_ans = get_completion(client, model_name, prompt_context)

            # -------- Record results --------
            model_results.append({
                "core_question": core_question,
                "basic_answer": basic_ans,
                "answer_with_context": context_ans,
                "context_info": context_str
            })

        all_results.append({
            "model_name": model_name,
            "results": model_results
        })
        print(f"\n✅ All questions processed for model {model_name}!")

    # -------- Save to file --------
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 All models processed, results saved to: {OUTPUT_FILE}")


# ========= Entry Point ==============================================================
if __name__ == "__main__":
    main()
