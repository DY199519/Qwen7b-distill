#!/usr/bin/env python
# coding: utf-8
"""
multmm1_build_prompts.py
------------------------
读取JSON文件并生成prompt：
  · 读取包含questions和answers的JSON文件
  · 按不同模型组合提取答案
  · 检查答案质量，过滤不合格的答案
  · 从外部文件读取prompt模板
  · 为每个组合生成独立的JSON文件
  · 将组合中最后一个模型的答案单独保存
"""

import json, csv
from pathlib import Path
import re
from typing import Tuple, List, Dict, Any
from datetime import datetime

# ========== 输出配置（放在最前面，方便修改） ==========
OUTPUT_DIR = Path(r"D:\qwensft\2+1")  # <-- 修改这里设置输出目录
OUTPUT_FILE_PREFIX = "finalprompt"  # <-- 输出文件前缀
OUTPUT_FILE_SUFFIX = "2+1_test-1"  # <-- 输出文件后缀
# =====================================================

# === 1. 路径配置 ===
BASE_DIR = Path(r"D:\project7\prompt")
json_path = Path(r"D:\qwensft\testquestion\multi_model_answersTest500.json")

# Prompt 文件路径
PROMPT_FILE = BASE_DIR / "prompt-2+1-1.txt"

# 模型组合配置
MODEL_COMBINATIONS = {
    "combination_1": ["gemini", "grok", "doubao"],
    # "combination_2": ["moonshot", "Yi", "gpt"],
    # "combination_3": ["llama", "vucina"],
}

# === 2. 质量检查参数 ===
MIN_ANSWER_LENGTH = 100  # 最小答案长度
MIN_COMPLETE_LENGTH = 50  # 完整性最小长度

# === 3. 答案质量检查函数 ===
def check_answer_quality(answer_text: str) -> Tuple[bool, str]:
    """
    检查答案质量（使用与之前相同的标准）
    返回: (是否合格, 问题描述)
    """
    # 检查是否为空
    if not answer_text or answer_text.strip() == "":
        return False, "空答案"
    
    answer_text = answer_text.strip()
    
    # 检查长度
    if len(answer_text) < MIN_COMPLETE_LENGTH:
        return False, f"答案过短({len(answer_text)}字符)"
    
    # 简单检查：是否以常见的完整标点结尾
    if answer_text.endswith(('。', '！', '？', '.', '!', '?')):
        return True, "完整"
    
    # 如果没有标点结尾，检查长度
    if len(answer_text) < MIN_ANSWER_LENGTH:
        return False, f"无结尾标点且较短({len(answer_text)}字符)"
    
    # 长答案但无标点，也视为不完整
    return False, f"无结尾标点({len(answer_text)}字符)"

# === 4. 读取 Prompt 模板 ===
def load_prompt_template():
    """从文件读取 prompt 模板"""
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            template = f.read().strip()
            print(f"✓ 成功读取 prompt 模板：{PROMPT_FILE}")
            return template
    except FileNotFoundError:
        print(f"⚠️ 警告：未找到 prompt 文件：{PROMPT_FILE}")
        # 使用默认模板作为备份
        default_template = '请回答："{q}"，基于以下回答对你的答案进行完善：{ctx}。'
        print(f"  使用默认 prompt 模板")
        return default_template

# === 5. 工具函数 ===
def fuzzy_match_model(model_pattern, available_models):
    """模糊匹配模型名称，返回匹配的模型列表"""
    matched_models = []
    for model in available_models:
        if model_pattern.lower() in model.lower():
            matched_models.append(model)
    return matched_models

def extract_answers_with_quality_check(question_data, model_patterns):
    """
    从问题数据中提取指定模型模式的答案，并进行质量检查
    返回: (答案列表, 找到的模型列表, 质量问题列表)
    """
    answers = []
    found_models = []
    quality_issues = []
    
    if "answers" in question_data:
        available_models = list(question_data["answers"].keys())
        
        for pattern in model_patterns:
            # 使用模糊匹配找到符合模式的模型
            matched_models = fuzzy_match_model(pattern, available_models)
            
            # 从匹配的模型中选择第一个有效答案
            found_valid = False
            for model in matched_models:
                if model in question_data["answers"]:
                    model_answers = question_data["answers"][model]
                    if model_answers and len(model_answers) > 0:
                        # 只取第一个答案
                        answer_text = model_answers[0].get("answer", "").strip()
                        
                        # 质量检查
                        is_quality_good, issue_desc = check_answer_quality(answer_text)
                        
                        if is_quality_good:
                            answers.append(answer_text)
                            found_models.append(model)
                            found_valid = True
                            break
                        else:
                            quality_issues.append({
                                "model": model,
                                "issue": issue_desc,
                                "answer_preview": answer_text[:50] + "..." if len(answer_text) > 50 else answer_text
                            })
            
            # 如果这个模式没有找到合格的答案，记录问题
            if not found_valid:
                quality_issues.append({
                    "model_pattern": pattern,
                    "issue": "未找到质量合格的答案"
                })
    
    return answers, found_models, quality_issues

def build_records(questions_data, prompt_template, combo_name, model_patterns):
    """为单个组合构造记录列表，包含质量检查"""
    rows = []
    combo_count = 0
    skipped_count = 0
    quality_issues_summary = {}
    
    print(f"\n  开始质量检查...")
    
    for question, question_data in questions_data.items():
        # 提取当前组合模型的答案并进行质量检查
        answers, found_models, quality_issues = extract_answers_with_quality_check(
            question_data, model_patterns
        )
        
        # 记录质量问题
        if quality_issues:
            quality_issues_summary[question] = quality_issues
        
        # 新的prompt格式需要至少2个质量合格的答案
        if len(answers) < 2:
            skipped_count += 1
            continue
        
        # 生成 prompt
        try:
            if len(answers) >= 2:
                prompt = prompt_template.format(q=question, A1=answers[0], A2=answers[1])
            else:
                continue
        except KeyError as e:
            # 如果模板格式不匹配，尝试旧格式
            ctx = "\n".join(f"回答{i+1}：{ans}" for i, ans in enumerate(answers[:2]))
            try:
                prompt = prompt_template.format(q=question, ctx=ctx)
            except:
                print(f"  ⚠️ 警告：prompt模板格式不匹配，跳过问题：{question[:50]}...")
                skipped_count += 1
                continue
        
        # 构建记录
        record = {
            "question": question,
            "prompt": prompt,
            "model": ",".join(found_models[:2]),
            "version": f"{combo_name}_{min(len(answers), 2)}_answers",
            "combination": combo_name,
            "answer_quality": "checked"  # 标记已通过质量检查
        }
        
        # 如果有第三个答案
        if len(answers) >= 3 and len(found_models) >= 3:
            record["third_model"] = found_models[2]
            record["third_answer"] = answers[2]
        
        rows.append(record)
        combo_count += 1
    
    print(f"  · {combo_name} 生成 {combo_count} 条记录")
    print(f"  · 因质量问题跳过 {skipped_count} 条记录")
    
    # 如果有质量问题，输出详细报告
    if quality_issues_summary:
        issue_count = len(quality_issues_summary)
        print(f"  · 发现 {issue_count} 个问题存在质量问题")
        
        # 保存质量问题报告
        quality_report_file = OUTPUT_DIR / f"quality_report_{combo_name}.json"
        with quality_report_file.open("w", encoding="utf-8") as f:
            json.dump({
                "combination": combo_name,
                "total_questions": len(questions_data),
                "questions_with_issues": issue_count,
                "skipped_questions": skipped_count,
                "generated_prompts": combo_count,
                "quality_issues": quality_issues_summary
            }, f, ensure_ascii=False, indent=2)
        print(f"  · 质量报告已保存到: {quality_report_file}")
    
    return rows

# === 6. 主程序 ===
print("📖 开始处理数据...")
print(f"📁 输出目录: {OUTPUT_DIR}")

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取 prompt 模板
prompt_template = load_prompt_template()

# 读取 JSON 数据
print(f"\n📖 读取 JSON 文件：{json_path}")
try:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # 检查数据结构
    if "questions" in data:
        questions_data = data["questions"]
        print(f"  · 找到 {len(questions_data)} 个问题")
    else:
        print("❌ 错误：JSON文件中没有找到 'questions' 字段")
        exit(1)
        
except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {json_path}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"❌ 错误：JSON解析失败：{e}")
    exit(1)

# === 7. 生成记录并写入 JSON ===
print("\n⚙️ 生成 prompt 记录...")

total_count = 0
total_skipped = 0

# 为每个组合生成独立的JSON文件
for combo_name, model_patterns in MODEL_COMBINATIONS.items():
    print(f"\n📋 处理组合 {combo_name}: {', '.join(model_patterns)}")
    
    # 生成当前组合的记录
    combo_rows = build_records(questions_data, prompt_template, combo_name, model_patterns)
    
    if combo_rows:
        # 为每个组合创建独立的JSON文件
        out_json = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_{combo_name}_{OUTPUT_FILE_SUFFIX}.json"
        
        # 写入JSON文件
        print(f"📝 写入 JSON 文件：{out_json}")
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(combo_rows, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 成功写入 {len(combo_rows)} 条记录到 {out_json.name}")
        total_count += len(combo_rows)
    else:
        print(f"⚠️ 警告：{combo_name} 没有生成任何记录（所有答案都未通过质量检查）")

# 生成总体统计报告
summary_file = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_summary_{OUTPUT_FILE_SUFFIX}.json"
with summary_file.open("w", encoding="utf-8") as f:
    json.dump({
        "total_questions": len(questions_data),
        "total_prompts_generated": total_count,
        "combinations_processed": len(MODEL_COMBINATIONS),
        "quality_check_enabled": True,
        "min_answer_length": MIN_ANSWER_LENGTH,
        "min_complete_length": MIN_COMPLETE_LENGTH,
        "timestamp": datetime.now().isoformat()
    }, f, ensure_ascii=False, indent=2)

print(f"\n📊 总计生成 {total_count} 条记录，分布在 {len(MODEL_COMBINATIONS)} 个文件中")
print(f"📊 总体统计已保存到: {summary_file}")
print("\n🎉 完成！")