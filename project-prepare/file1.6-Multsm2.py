#!/usr/bin/env python
# coding: utf-8
"""
generate_multi_answers.py
-------------------------
读取 generated_results_multi_model.json，
对每个模型 / 每个核心问题生成（或复用） basic_answer 和
answer_with_context，并保存到 final_ans_multi_sm.json。

若 grouped_answers.json 中已存在对应 basic_answer，则直接复用，
避免重复生成；只有缺失时才会调用 API。
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI
import httpx


# ========= 路径 & 常量 ========================================================
INPUT_FILE     = Path(r"D:\project\generated_results_multi_model.json")
BASIC_ANS_FILE = Path(r"D:\project\grouped_answers.json")           # ← 已有 basic 答案
OUTPUT_DIR     = Path(r"D:\project")
OUTPUT_FILE    = OUTPUT_DIR / "final_ans_multi_sm.json"


# ========= 模型配置（完整） ===================================================
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


# ========= 工具函数 ===========================================================
def load_basic_answer_map(filepath: Path) -> Dict[str, Dict[str, str]]:
    """读取 grouped_answers.json，返回映射：{核心问题: {model_name: basic_answer, ...}, ...}"""
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
    """带重试的 OpenAI API 请求。"""
    try:
        rsp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return rsp.choices[0].message.content
    except Exception as e:
        if attempt < max_attempts:
            print(f"⚠️ 请求失败，重试中 ({attempt}/{max_attempts})... 原因：{e}")
            time.sleep(2)
            return get_completion(client, model_name, prompt,
                                  attempt + 1, max_attempts)
        print(f"❌ 最终失败：{e}")
        return f"ERROR: {str(e)}"


# ========= 主流程 =============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    basic_answer_map = load_basic_answer_map(BASIC_ANS_FILE)
    all_results = []

    for model_block in raw_data:
        model_name = model_block["model_name"]
        print(f"\n{'='*60}\n🔍 开始处理模型：{model_name}\n{'='*60}")

        config = models_config.get(model_name)
        if not config:
            print(f"⚠️ 未找到模型配置，跳过：{model_name}")
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
            print(f"\n➡️ 问题：{core_question}")

            # -------- 构造 context 信息（跳过超过30字符的部分） --------
            context_parts = []
            for grp in sum_list_data:
                if not grp or not isinstance(grp, list) or len(grp) < 2:
                    continue
                part = f"{grp[0]}领域：{'、'.join(grp[1:])}"
                if len(part) > 30:
                    print(f"⚠️ 跳过过长因素：{part}")
                    continue
                context_parts.append(part)
            context_str = "；".join(context_parts)

            # -------- 1. basic answer --------
            basic_ans = basic_answer_map.get(core_question, {}).get(model_name)
            if basic_ans is None:
                prompt_basic = f"请回答：\"{core_question}\"。"
                basic_ans = get_completion(client, model_name, prompt_basic)
                print("    （未找到现成 basic_answer，已调用 API 补充）")

            # -------- 2. answer_with_context --------
            prompt_context = (f"请回答：\"{core_question}\"，结合以下信息，"
                              f"参考以下重要因素进行作答，因素按重要性排序：{context_str}。")
            context_ans = get_completion(client, model_name, prompt_context)

            # -------- 记录结果 --------
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
        print(f"\n✅ 模型 {model_name} 全部问题处理完成！")

    # -------- 保存到文件 --------
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 所有模型处理完成，结果已保存至：{OUTPUT_FILE}")


# ========= 入口 ==============================================================
if __name__ == "__main__":
    main()
