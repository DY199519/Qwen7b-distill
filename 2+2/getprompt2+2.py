#!/usr/bin/env python
# coding: utf-8
"""
multmm1_build_prompts.py  ——  占位符统一为 A1/A2
-------------------------------------------------
读取 JSON，组合不同模型答案，按配对生成 prompt 并写 CSV。
添加了模糊匹配功能，支持部分匹配模型名称。
修改版：添加空答案检查，如果任意一个答案为空则跳过该题目。
"""

import json, csv
from pathlib import Path

# === 路径配置 ===
BASE_DIR = Path(r"D:\project7\prompt")
BASE_DIR_1= Path(r"D:\project7\MM\result")

JSON_PATH   = BASE_DIR_1 / "multi_model_answer-1-700.json"
PROMPT_FILE = BASE_DIR / "prompt2+2.txt"
OUT_CSV     = BASE_DIR_1 / "final_prompt_2+2-1-700.csv"

# === 组合 & 配对 ===
MODEL_COMBINATIONS = {
    "combination_1": ["gemini-2.5", "grok-3", "doubao-pro-256k"],
    # "combination_4": ["gemini-2.5", "moonshot-v1-8k", "Yi-9B"],
    # "combination_5": ["moonshot-v1-8k", "Yi-9B", "vucina-7b"]
}

ANSWER_PAIRINGS = {
    "combination_1": [{"name": "A1A2", "indices": [0, 1]},
                      {"name": "A1A3", "indices": [0, 2]}],
    # "combination_4": [{"name": "A1B1", "indices": [0, 1]},
    #                   {"name": "A1B2", "indices": [0, 2]}],
    # "combination_5": [{"name": "B1B2", "indices": [0, 1]},
    #                   {"name": "B1C2", "indices": [0, 2]}]
}

# === 读取 prompt 模板 ===
def load_prompt_template() -> str:
    try:
        with PROMPT_FILE.open("r", encoding="utf-8") as f:
            print(f"✓ 使用外部模板：{PROMPT_FILE}")
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ 未找到模板，改用内置默认模板")
        return (
            '请回答："{q}"。\n'
            '硬性约束\n'
            '1. 仅可使用下方 A1 与 A2 中出现的事实或观点；\n'
            '2. 禁止引入任何未在两份回答中显式出现的信息、数据或推论；\n'
            '3. 如信息不足以回答，输出"无法回答"；\n'
            '4. 除衔接词（例如"因此""此外"）外，不得新增内容。\n\n'
            '【任务步骤】\n'
            '① 读取 A1 与 A2，提炼关键信息；\n'
            '② 去重、归类并精简整合，生成最终回答。\n\n'
            '【输出格式】\n'
            '- 关键信息要点：\n'
            '  …\n'
            '  …\n'
            '- 精简整合后的最终回答：\n'
            '  …\n\n'
            'A1：\n{A1}\n\n'
            'A2：\n{A2}'
        )

# === 提取指定模型答案（支持模糊匹配）===
def extract_answers(qdata: dict, models: list[str]):
    """
    提取指定模型的答案，支持模糊匹配
    """
    answers, names = [], []
    all_model_names = list(qdata.get("answers", {}).keys())
    
    # 打印可用的模型名称（调试用）
    if all_model_names:
        print(f"  可用模型: {', '.join(all_model_names)}")
    
    for m in models:
        txt = ""
        actual_model_name = m
        
        # 先尝试精确匹配
        if m in all_model_names:
            answer_list = qdata.get("answers", {}).get(m, [])
            if answer_list and len(answer_list) > 0:
                txt = answer_list[0].get("answer", "").strip()
            print(f"  ✓ 精确匹配: {m}")
        else:
            # 模糊匹配 - 检查是否为子串关系
            matched = False
            for actual_name in all_model_names:
                # 双向检查：m是actual_name的子串，或actual_name是m的子串
                if m.lower() in actual_name.lower() or actual_name.lower() in m.lower():
                    answer_list = qdata.get("answers", {}).get(actual_name, [])
                    if answer_list and len(answer_list) > 0:
                        txt = answer_list[0].get("answer", "").strip()
                        actual_model_name = actual_name
                        print(f"  ✓ 模糊匹配: {m} → {actual_name}")
                        matched = True
                        break
            
            if not matched:
                print(f"  ✗ 未找到匹配: {m}")
        
        if txt:
            answers.append(txt)
            names.append(actual_model_name)
        else:
            print(f"  ⚠️ 模型 {m} 无有效答案")
    
    return answers, names

# === 主构造函数 ===
def build_records(questions: dict, tpl: str):
    rows = []
    total_questions = len(questions)
    skipped_empty = 0  # 统计因空答案跳过的题目数
    
    for combo, models in MODEL_COMBINATIONS.items():
        pairings = ANSWER_PAIRINGS.get(combo, [{"name": "default", "indices": [0, 1]}])
        print(f"\n处理组合: {combo}")
        print(f"期望模型: {models}")
        print(f"配对方案: {[p['name'] for p in pairings]}")
        
        processed_count = 0
        combo_skipped_empty = 0
        
        for q_idx, (q, qdata) in enumerate(questions.items(), 1):
            print(f"\n[{q_idx}/{total_questions}] 问题: {q[:60]}...")
            
            ans_list, model_names = extract_answers(qdata, models)
            
            print(f"  找到答案: {len(ans_list)} 个")
            if model_names:
                print(f"  实际模型: {model_names}")
            
            if len(ans_list) < 2:
                print(f"  ⚠️ 答案不足（需要至少2个），跳过此问题")
                continue
            
            # 为每个配对生成prompt
            question_has_valid_pair = False
            for p in pairings:
                pair_name = p["name"]
                i, j = p["indices"]
                
                if i >= len(ans_list) or j >= len(ans_list):
                    print(f"  ⚠️ 配对 {pair_name} 索引越界 ({i},{j})，跳过")
                    continue
                
                # 检查答案是否为空
                if not ans_list[i] or not ans_list[j]:
                    print(f"  ❌ 配对 {pair_name} 包含空答案，跳过")
                    if not ans_list[i]:
                        print(f"     - 答案1 (索引{i}, {model_names[i]}) 为空")
                    if not ans_list[j]:
                        print(f"     - 答案2 (索引{j}, {model_names[j]}) 为空")
                    combo_skipped_empty += 1
                    continue
                
                try:
                    prompt = tpl.format(q=q, A1=ans_list[i], A2=ans_list[j])
                    
                    rows.append({
                        "question": q,
                        "prompt": prompt,
                        "model": f"{model_names[i]},{model_names[j]}",
                        "version": f"{combo}_{pair_name}",
                        "combination": f"{combo}_{pair_name}"
                    })
                    
                    print(f"  ✓ 生成配对: {pair_name} ({model_names[i]} + {model_names[j]})")
                    processed_count += 1
                    question_has_valid_pair = True
                    
                except KeyError as e:
                    print(f"  ❌ 模板格式错误: {e}")
                    # 尝试使用备用格式
                    try:
                        prompt = tpl.format(q=q, ctx=f"{ans_list[i]}\n---\n{ans_list[j]}")
                        rows.append({
                            "question": q,
                            "prompt": prompt,
                            "model": f"{model_names[i]},{model_names[j]}",
                            "version": f"{combo}_{pair_name}",
                            "combination": f"{combo}_{pair_name}"
                        })
                        print(f"  ✓ 使用备用格式生成配对: {pair_name}")
                        processed_count += 1
                        question_has_valid_pair = True
                    except:
                        print(f"  ❌ 无法生成配对 {pair_name}")
            
            if not question_has_valid_pair:
                print(f"  ⚠️ 此问题没有有效的配对，完全跳过")
        
        print(f"\n组合 {combo} 处理完成：")
        print(f"  - 生成记录: {processed_count} 条")
        print(f"  - 因空答案跳过: {combo_skipped_empty} 个配对")
        skipped_empty += combo_skipped_empty
    
    print(f"\n📊 总计生成 {len(rows)} 条记录")
    print(f"   因空答案跳过 {skipped_empty} 个配对")
    return rows

# === 入口 ===
def main():
    print("=" * 60)
    print("多模型答案组合生成器 (支持模糊匹配 + 空答案检查)")
    print("=" * 60)
    
    # 加载prompt模板
    tpl = load_prompt_template()
    
    # 读取JSON数据
    try:
        with JSON_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", {})
        print(f"\n✓ 成功读取 {len(questions)} 个问题")
    except Exception as e:
        print(f"\n❌ 数据读取失败： {e}")
        return
    
    # 构建记录
    records = build_records(questions, tpl)
    
    if not records:
        print("\n⚠️ 无数据生成，请检查模型名称是否正确或答案是否为空")
        return
    
    # 写入CSV
    try:
        with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f,
                fieldnames=["question", "prompt", "model", "version", "combination"])
            writer.writeheader()
            writer.writerows(records)
        print(f"\n✅ 已写入 {OUT_CSV}")
        print(f"   包含 {len(records)} 条记录")
        
        # 统计不同组合的记录数
        combo_stats = {}
        for r in records:
            combo = r["combination"]
            combo_stats[combo] = combo_stats.get(combo, 0) + 1
        
        print("\n📊 各组合记录数：")
        for combo, count in sorted(combo_stats.items()):
            print(f"   - {combo}: {count} 条")
            
    except Exception as e:
        print(f"\n❌ 写入失败: {e}")

if __name__ == "__main__":
    main()