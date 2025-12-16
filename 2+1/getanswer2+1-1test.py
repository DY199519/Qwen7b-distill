#!/usr/bin/env python
# coding: utf-8
"""
multmm2_run2plus2_topic_first.py
--------------------------------
按"题目 -> 模型"顺序处理：
  1. 读取第二个文件生成的 JSON 数据
  2. 针对同一道题，依次调用所有模型
  3. 每 SAVE_INTERVAL 题把各模型的进度文件写盘
  4. 使用第二个文件中的 prompt 字段作为输入
  5. 从原始JSON文件中获取direct reply，不重复调用API

修改：适配第二个文件的输出格式，从原始数据中提取答案
"""

import csv, json, time, re, traceback
from pathlib import Path
from openai import OpenAI

# ========== 输出配置（放在最前面，方便修改） ==========
OUTPUT_DIR = Path(r"D:\qwensft\2+1")  # <-- 修改这里设置输出目录
OUTPUT_FILE_SUFFIX = "2+1-1-test"  # <-- 输出文件后缀，例如: gemini-2.5-flash_answers_2+1-1.json
# =====================================================

# ===== 0. 路径配置 ===========================================================
BASE_DIR   = Path(r"D:\qwensft\2+1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取各个组合的 JSON 文件
PROMPT_FILES = [
    BASE_DIR / "finalprompt_combination_1_2+1_test-1.json",
    # BASE_DIR / "finalprompt_combination_2_2+1.json",
    # BASE_DIR / "finalprompt_combination_3_2+1.json",
]

# 原始问题和答案的JSON文件
ORIGINAL_JSON = Path(r"D:\qwensft\testquestion\multi_model_answersTest500.json")
SAVE_INTERVAL = 1  # 每 N 题保存一次

# ===== 1. 模型账户配置 =======================================================
MODEL_CFGS = [
    {
        "model_name": "gemini-2.5-flash",
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/"
    },
    # {
    #     "model_name": "moonshot-v1-8k",
    #     "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
    #     "base_url": "https://api.vansai.cn/v1",
    # }
]

# ===== 2. 辅助函数 ==================================================

def print_model_info(combo: str, cur: str):
    """简化的组合信息打印"""
    print(f"  └─ 处理组合: {combo}")

# ===== 3. IO & GPT 调用 ======================================================

def load_original_answers(path: Path):
    """从原始JSON文件中加载所有模型的答案"""
    if not path.exists():
        print(f"⚠️ 原始答案文件不存在: {path}")
        return {}
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        answers_cache = {}
        if "questions" in data:
            for question, question_data in data["questions"].items():
                if "answers" in question_data:
                    answers_cache[question] = {}
                    for model, model_answers in question_data["answers"].items():
                        if model_answers and len(model_answers) > 0:
                            # 取第一个答案
                            answer_text = model_answers[0].get("answer", "").strip()
                            if answer_text:
                                answers_cache[question][model] = answer_text
        
        print(f"✓ 成功加载原始答案，包含 {len(answers_cache)} 个问题")
        return answers_cache
        
    except Exception as e:
        print(f"❌ 读取原始答案文件失败: {e}")
        return {}

def get_direct_answer(question: str, model_name: str, original_answers: dict):
    """从原始答案中获取指定模型对问题的直接回答"""
    if question not in original_answers:
        print(f"  ⚠️ 问题不在原始答案中: {question[:50]}...")
        return ""
    
    question_answers = original_answers[question]
    
    # 尝试精确匹配
    if model_name in question_answers:
        return question_answers[model_name]
    
    # 尝试模糊匹配
    for model, answer in question_answers.items():
        if model_name.lower() in model.lower() or model.lower() in model_name.lower():
            print(f"  📋 模糊匹配: {model_name} -> {model}")
            return answer
    
    print(f"  ⚠️ 未找到模型 {model_name} 的答案")
    return ""

# ---- 新增：答案质量检查函数 ----
END_PUNCT = ('。', '！', '？', '.', '!', '?')
MIN_GOOD_LENGTH = 200  # 触发重试的最小长度

def is_low_quality_answer(text: str) -> bool:
    if not text:
        return True
    text = text.strip()
    if len(text) < MIN_GOOD_LENGTH:
        return True
    if not text.endswith(END_PUNCT):
        return True
    return False

def ask(api: OpenAI, model: str, prompt: str, retry: int = 3, pause: int = 2):
    last_txt = ""
    for i in range(retry):
        try:
            rsp = api.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            txt = (rsp.choices[0].message.content or "").strip()
            last_txt = txt

            if not is_low_quality_answer(txt):
                return txt

            print(f"⚠️ {model} 第 {i+1} 次返回过短/无结尾标点（len={len(txt)}），重试...")
        except Exception as e:
            print(f"❌ {model} 第 {i+1} 次失败: {e}")

        time.sleep(pause)

    return last_txt

def load_progress(file: Path):
    if not file.exists():
        return {}
    try:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {row["question"]: row for row in data}
    except Exception as e:
        print(f"⚠️ 读取进度失败: {e}")
        return {}

def save_progress(rows: list, file: Path):
    try:
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(file)
        print(f"💾 保存 {file.name} （{len(rows)} 条）")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

# ===== 4. 读取第二个文件生成的 prompt 数据 ====================================

def load_prompts_from_files(prompt_files):
    """读取第二个文件生成的 JSON 文件
    返回 {question: {combination: entry_data}}
    """
    q2data = {}
    
    for file_path in prompt_files:
        if not file_path.exists():
            print(f"⚠️ 文件不存在: {file_path}")
            continue
            
        print(f"📖 读取文件: {file_path}")
        
        try:
            with file_path.open("r", encoding="utf-8") as f:
                entries = json.load(f)
            
            for entry in entries:
                q = entry["question"]
                combo = entry["combination"]
                
                if q not in q2data:
                    q2data[q] = {}
                
                # 存储整个条目数据
                q2data[q][combo] = entry
                
            print(f"  └─ 读取 {len(entries)} 条记录")
            
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")
    
    return q2data

# ===== 5. 脚本入口：按题目遍历 ==============================================
if __name__ == "__main__":
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 输出文件后缀: {OUTPUT_FILE_SUFFIX}")
    print("-" * 60)
    
    # 1) 读取第二个文件生成的数据
    q2data = load_prompts_from_files(PROMPT_FILES)
    all_questions = sorted(q2data.keys())
    print(f"\n📚 题目数: {len(all_questions)}")
    
    # 2) 读取原始答案数据
    original_answers = load_original_answers(ORIGINAL_JSON)

    # 3) 为每个模型准备 API、进度文件、行缓存
    model_env = {}
    for cfg in MODEL_CFGS:
        name = cfg["model_name"]
        output_file = OUTPUT_DIR / f"{name}_answers_{OUTPUT_FILE_SUFFIX}.json"
        model_env[name] = {
            "api": OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]),
            "out": output_file,
            "rows": [],  # 新增/累积结果
            "done": load_progress(output_file),
        }

    processed = 0

    # ------- 主循环：题目优先 -----------------
    for qi, q in enumerate(all_questions, 1):
        print(f"\n📝 [{qi}/{len(all_questions)}] {q[:60]}…")
        
        # 获取该问题的所有组合数据
        q_combinations = q2data[q]

        for cfg in MODEL_CFGS:
            mname = cfg["model_name"]
            env = model_env[mname]

            # 已有则跳过
            if q in env["done"]:
                continue

            api = env["api"]

            print(f"🤖 处理 {mname}")
            # 从原始答案中获取该模型对问题的直接回答
            direct_answer = get_direct_answer(q, mname, original_answers)
            item = {
                "question": q,
                "direct_prompt": q,
                "direct_reply": direct_answer,
            }

            # 处理该题的所有组合
            for combo, entry_data in q_combinations.items():
                print_model_info(combo, mname)
                
                # 使用第二个文件中的 prompt 字段
                prompt_text = entry_data["prompt"]
                reply = ask(api, mname, prompt_text)
                
                # 保存 prompt 和回复
                item[f"{combo}_prompt"] = prompt_text
                item[f"{combo}_reply"] = reply
            
            # 在所有组合处理完后，添加 third_model 和 third_answer
            # 从任意一个组合中获取（因为同一个问题的所有组合应该有相同的 third 信息）
            if q_combinations:
                first_combo_data = next(iter(q_combinations.values()))
                if "third_model" in first_combo_data:
                    item["third_model"] = first_combo_data["third_model"]
                if "third_answer" in first_combo_data:
                    item["third_answer"] = first_combo_data["third_answer"]

            env["rows"].append(item)
            env["done"][q] = item  # 标记完成

        processed += 1
        # ---- SAVE_INTERVAL ----
        if processed % SAVE_INTERVAL == 0:
            for mname, env in model_env.items():
                save_progress(env["rows"] + list(env["done"].values()), env["out"])

    # 4) 全部完成后保存一次
    for mname, env in model_env.items():
        final_rows = env["rows"] + list(env["done"].values())
        save_progress(final_rows, env["out"])

    print("\n🎉 按题目顺序全部处理完毕，文件保存在:", OUTPUT_DIR)
