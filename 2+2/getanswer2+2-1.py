#!/usr/bin/env python
# coding: utf-8
"""
unified_gemini_run.py
---------------------
修改版：去掉 combination 逻辑，统一使用 gemini-2.5-flash 模型
新增：答案质量检查（字数和标点符号检查）
"""

import csv, json, time, traceback
from pathlib import Path
from openai import OpenAI

# ===== 0. 路径配置 ===========================================================
BASE_DIR = Path(r"D:\project7\prompt")
BASE_DIR_1= Path(r"D:\project7\MM\result")
OUTPUT_DIR=Path(r"D:\project7\MM\result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件定义（移到最前面）
OUTPUT_FILE = OUTPUT_DIR / "answer-2+2-1-1-700.json"

PROMPT_CSV   = BASE_DIR_1 / "final_prompt_2+2-1-700.csv"
GROUPED_JSON = BASE_DIR / "multi_model_answers-1-700.json"
SAVE_INTERVAL = 10  # 每 N 题保存一次

# ===== 1. 模型配置 =======================================================
# 只保留 gemini 模型配置
MODEL_NAME = "gemini-2.5-flash"
API_KEY = "sk-eU0JtXoQSn3wSM0yA981lTMrUEDD31vtxAtFLA2ub6lwi3dd"
BASE_URL = "https://api.aigptapi.com/v1/"

# 答案质量检查参数
MIN_ANSWER_LENGTH = 10  # 最小字数要求
VALID_END_PUNCTUATION = {'.', '。', '!', '！', '?', '？', ')', '）', '"', '"', "'", "'"}  # 有效的结尾标点

# ===== 2. IO & GPT 调用 ======================================================
def load_cache(path: Path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
        cache = {}
        for q, entry in data.items():
            for m, ans in entry.get("basic_answers", []):
                cache.setdefault(q, {})[m] = ans
        return cache
    except Exception as e:
        print(f"⚠️ 读取 cache 失败: {e}")
        return {}

def check_answer_quality(answer: str) -> tuple[bool, str]:
    """
    检查答案质量
    返回: (是否合格, 问题描述)
    """
    if not answer:
        return False, "答案为空"
    
    # 检查字数
    if len(answer.strip()) < MIN_ANSWER_LENGTH:
        return False, f"答案过短（少于{MIN_ANSWER_LENGTH}字）"
    
    # 检查结尾标点
    last_char = answer.strip()[-1] if answer.strip() else ''
    if last_char not in VALID_END_PUNCTUATION:
        return False, f"答案未以标点符号结尾（最后字符: '{last_char}'）"
    
    return True, "合格"

def ask(api: OpenAI, model: str, prompt: str, retry=3, pause=2):
    """
    调用 API 获取回答，并进行质量检查
    """
    for i in range(retry):
        try:
            rsp = api.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            txt = rsp.choices[0].message.content.strip()
            
            # 检查答案质量
            is_valid, issue = check_answer_quality(txt)
            if not is_valid:
                print(f"  ⚠️ 答案质量问题: {issue}")
                if i < retry - 1:
                    print(f"  🔄 重试中...")
                    time.sleep(pause)
                    continue
                else:
                    print(f"  ❌ 多次尝试后仍有质量问题，使用当前结果")
            
            return txt
            
        except Exception as e:
            print(f"❌ {model} 第 {i+1} 次失败: {e}")
            time.sleep(pause)
    return ""

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

def save_progress(results: list, file: Path):
    """保存进度"""
    try:
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(file)
        print(f"💾 保存 {file.name} （{len(results)} 条）")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

# ===== 3. 脚本入口：统一使用 gemini 处理所有问题 ===============================
if __name__ == "__main__":
    print(f"📁 输出文件: {OUTPUT_FILE}")
    print(f"📏 答案质量要求: 最少{MIN_ANSWER_LENGTH}字，需以标点符号结尾")
    
    # 1) 预读 CSV，构建 {question: [prompts]}
    q2prompts = {}
    with PROMPT_CSV.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"]
            prompt = row["prompt"]
            q2prompts.setdefault(q, []).append(prompt)

    all_questions = sorted(q2prompts.keys())
    print(f"📚 题目数: {len(all_questions)}")

    # 2) 初始化 API 对象
    api = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"🤖 使用模型: {MODEL_NAME}")
    
    # 3) 加载已有进度
    existing_results = []
    processed_questions = set()
    
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            existing_results = json.load(f)
            processed_questions = {item["question"] for item in existing_results}
        print(f"📊 已有进度: {len(processed_questions)} 题")

    cache = load_cache(GROUPED_JSON)
    processed = 0
    skipped = 0
    quality_issues = 0

    # ------- 主循环：题目优先 -----------------
    for qi, q in enumerate(all_questions, 1):
        print(f"\n📝 [{qi}/{len(all_questions)}] {q[:60]}…")
        
        # 已处理过则跳过
        if q in processed_questions:
            print(f"  ⏭️ 已处理过，跳过")
            skipped += 1
            continue

        item = {
            "question": q,
            "model": MODEL_NAME,
            "prompts": q2prompts[q],
            "answers": []
        }

        # 先获取直接回答（使用原始问题）
        print(f"  └─ 获取直接回答...")
        direct_answer = cache.get(q, {}).get(MODEL_NAME) or ask(api, MODEL_NAME, q)
        item["direct_answer"] = direct_answer
        
        # 检查直接回答的质量
        is_valid, issue = check_answer_quality(direct_answer)
        if not is_valid:
            quality_issues += 1
            item["direct_answer_quality_issue"] = issue
        
        # 处理该题的所有 prompts
        for i, prompt in enumerate(q2prompts[q], 1):
            print(f"  └─ 处理 prompt {i}/{len(q2prompts[q])}")
            
            reply = ask(api, MODEL_NAME, prompt)
            
            # 检查回答质量
            is_valid, issue = check_answer_quality(reply)
            answer_data = {
                "prompt_index": i,
                "prompt": prompt,
                "reply": reply
            }
            
            if not is_valid:
                quality_issues += 1
                answer_data["quality_issue"] = issue
            
            item["answers"].append(answer_data)
            
            # 短暂延迟，避免请求过快
            if i < len(q2prompts[q]):
                time.sleep(0.5)

        existing_results.append(item)
        processed += 1
        processed_questions.add(q)
        
        # ---- SAVE_INTERVAL ----
        if processed > 0 and processed % SAVE_INTERVAL == 0:
            print(f"\n💾 达到保存间隔，保存进度...")
            save_progress(existing_results, OUTPUT_FILE)

    # 4) 全部完成后保存一次
    print(f"\n🏁 处理完成！新处理 {processed} 题，跳过 {skipped} 题")
    save_progress(existing_results, OUTPUT_FILE)
    print(f"✅ 总计 {len(existing_results)} 条记录")

    # 统计信息
    total_prompts = sum(len(item["answers"]) for item in existing_results)
    print(f"\n📊 统计信息：")
    print(f"  - 总题目数: {len(existing_results)}")
    print(f"  - 总 prompt 数: {total_prompts}")
    print(f"  - 平均每题 prompts: {total_prompts/len(existing_results):.2f}")
    print(f"  - 使用模型: {MODEL_NAME}")
    print(f"  - 质量问题数: {quality_issues}")

    # 质量问题汇总
    if quality_issues > 0:
        print(f"\n⚠️ 发现 {quality_issues} 个答案质量问题")
        print("  可在输出文件中查看具体问题详情（quality_issue 字段）")

    print(f"\n🎉 处理完毕，文件保存在: {OUTPUT_FILE}")