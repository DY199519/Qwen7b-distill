#!/usr/bin/env python
# coding: utf-8
"""
multmm1_build_prompts_txt.py
----------------------------
读取JSON文件并生成prompt：
  · 读取包含questions和answers的JSON文件
  · 模糊匹配包含gemini, grok, doubao的模型
  · 从外部文件读取prompt模板
  · 生成prompt并写入TXT文件，每个prompt为一段
  · 添加纠错机制，检查答案的有效性
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

# === 1. 路径配置 ===
BASE_DIR = Path(r"D:\project7\prompt")
BASE_DIR_1 = Path(r"D:\project7")
BASE_DIR_2 = Path(r"D:\project7\MM\result")

# 使用第二个JSON文件
json_path = BASE_DIR_1 / "multi_model_answers9400-10000.json"

# Prompt 文件路径
PROMPT_FILE = BASE_DIR_2 / "prompt-3+1withoutsummary.txt"

# 输出文件（改为txt）
OUT_TXT = BASE_DIR_2 / "final_prompt_3+1-Test.txt"
ERROR_LOG = BASE_DIR_2 / "error_log.txt"  # 错误日志文件

# === 2. 纠错配置 ===
class ErrorChecker:
    """答案纠错检查器"""
    
    # 最小答案长度
    MIN_ANSWER_LENGTH = 10
    
    # 最大答案长度（可能是错误）
    MAX_ANSWER_LENGTH = 10000
    
    # 常见错误模式
    ERROR_PATTERNS = [
        r'^error:',  # 以error开头
        r'^exception:',  # 以exception开头
        r'^\s*$',  # 纯空白
        r'^null$',  # null值
        r'^undefined$',  # undefined值
        r'^N/A$',  # N/A
        r'^\[.*error.*\]$',  # 包含error的方括号内容
        r'^\{.*error.*\}$',  # 包含error的花括号内容
    ]
    
    # 可疑模式（警告但不过滤）
    WARNING_PATTERNS = [
        r'^.{1,9}$',  # 过短的答案（小于10字符）
        r'^\d+$',  # 纯数字
        r'^[^\u4e00-\u9fa5a-zA-Z]+$',  # 没有中文或英文字母
        r'(.)\1{10,}',  # 重复字符超过10次
    ]
    
    @classmethod
    def check_context_format(cls, context: str) -> Tuple[bool, List[str]]:
        """
        检查生成的上下文格式是否正确
        确保每个"回答X："后面都有实际内容
        """
        issues = []
        
        # 提取所有回答部分
        pattern = r'回答(\d+)：(.*?)(?=回答\d+：|$)'
        matches = re.findall(pattern, context, re.DOTALL)
        
        if not matches:
            issues.append("未找到标准的'回答X：'格式")
            return False, issues
        
        # 检查每个回答
        for num, content in matches:
            content = content.strip()
            if not content:
                issues.append(f"回答{num}为空")
            elif len(content) < cls.MIN_ANSWER_LENGTH:
                issues.append(f"回答{num}内容过短 ({len(content)}字符)")
        
        # 检查回答编号是否连续
        numbers = [int(num) for num, _ in matches]
        numbers.sort()
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            issues.append(f"回答编号不连续: {numbers}")
        
        return len(issues) == 0, issues
    
    @classmethod
    def check_answer(cls, answer: str, question: str = "") -> Tuple[bool, str, List[str]]:
        """
        检查答案是否有效
        返回: (是否有效, 清理后的答案, 错误/警告列表)
        """
        errors = []
        warnings = []
        
        # 基本检查
        if not answer:
            errors.append("答案为空")
            return False, "", errors
        
        # 类型检查
        if not isinstance(answer, str):
            errors.append(f"答案类型错误: {type(answer)}")
            return False, "", errors
        
        # 清理答案（去除首尾空白）
        cleaned_answer = answer.strip()
        
        # 长度检查
        if len(cleaned_answer) < cls.MIN_ANSWER_LENGTH:
            warnings.append(f"答案过短 ({len(cleaned_answer)} 字符)")
        elif len(cleaned_answer) > cls.MAX_ANSWER_LENGTH:
            warnings.append(f"答案过长 ({len(cleaned_answer)} 字符)")
        
        # 错误模式检查
        for pattern in cls.ERROR_PATTERNS:
            if re.match(pattern, cleaned_answer, re.IGNORECASE):
                errors.append(f"匹配错误模式: {pattern}")
                return False, cleaned_answer, errors
        
        # 警告模式检查
        for pattern in cls.WARNING_PATTERNS:
            if re.match(pattern, cleaned_answer, re.IGNORECASE):
                warnings.append(f"匹配可疑模式: {pattern}")
        
        # 特殊字符检查
        if cleaned_answer.count('\n') > 50:
            warnings.append("包含过多换行符")
        
        if cleaned_answer.count(' ') / len(cleaned_answer) > 0.5:
            warnings.append("空格比例过高")
        
        # 返回结果
        is_valid = len(errors) == 0
        all_issues = errors + warnings
        
        return is_valid, cleaned_answer, all_issues

# === 3. 读取 Prompt 模板 ===
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

# === 4. 工具函数 ===
def extract_answers_fuzzy(question_data: Dict, question: str, error_log: List[Dict]) -> Tuple[List[str], List[str], Dict]:
    """
    模糊匹配包含 gemini, grok, doubao 的模型并提取答案
    添加纠错机制
    """
    answers = []
    found_models = []
    target_keywords = ['gemini', 'grok', 'doubao']
    
    # 统计信息
    stats = {
        'total_models': 0,
        'matched_models': 0,
        'valid_answers': 0,
        'invalid_answers': 0,
        'warnings': 0
    }
    
    if "answers" in question_data:
        for model_name, model_answers in question_data["answers"].items():
            stats['total_models'] += 1
            
            # 模糊匹配：检查模型名是否包含目标关键词
            model_lower = model_name.lower()
            for keyword in target_keywords:
                if keyword in model_lower:
                    stats['matched_models'] += 1
                    
                    if model_answers and len(model_answers) > 0:
                        # 只取第一个答案
                        raw_answer = model_answers[0].get("answer", "")
                        
                        # 纠错检查
                        is_valid, cleaned_answer, issues = ErrorChecker.check_answer(raw_answer, question)
                        
                        if is_valid:
                            if cleaned_answer:  # 再次确认清理后的答案不为空
                                answers.append(cleaned_answer)
                                found_models.append(model_name)
                                stats['valid_answers'] += 1
                                
                                # 如果有警告，记录但不阻止使用
                                if issues:
                                    stats['warnings'] += 1
                                    error_log.append({
                                        'type': 'warning',
                                        'question': question[:50] + '...' if len(question) > 50 else question,
                                        'model': model_name,
                                        'issues': issues,
                                        'answer_preview': cleaned_answer[:50] + '...' if len(cleaned_answer) > 50 else cleaned_answer
                                    })
                        else:
                            stats['invalid_answers'] += 1
                            error_log.append({
                                'type': 'error',
                                'question': question[:50] + '...' if len(question) > 50 else question,
                                'model': model_name,
                                'issues': issues,
                                'raw_answer': raw_answer[:50] + '...' if len(raw_answer) > 50 else raw_answer
                            })
                    break  # 找到匹配就跳出内层循环
    
    return answers, found_models, stats

def build_prompts(questions_data: Dict, prompt_template: str) -> Tuple[List[str], List[Dict]]:
    """构造prompt列表，返回prompt列表和错误日志"""
    prompts = []
    error_log = []
    
    # 全局统计
    global_stats = {
        'total_questions': 0,
        'matched_questions': 0,
        'skipped_questions': 0,
        'total_errors': 0,
        'total_warnings': 0,
        'context_format_errors': 0
    }
    
    print(f"\n📋 开始处理数据...")
    
    for question, question_data in questions_data.items():
        global_stats['total_questions'] += 1
        
        # 提取模糊匹配的答案（带纠错）
        answers, found_models, stats = extract_answers_fuzzy(question_data, question, error_log)
        
        # 更新全局统计
        global_stats['total_errors'] += stats['invalid_answers']
        global_stats['total_warnings'] += stats['warnings']
        
        if not answers:
            global_stats['skipped_questions'] += 1
            continue
        
        global_stats['matched_questions'] += 1
        
        # 构建上下文
        ctx = "\n".join(f"回答{i+1}：{ans}" for i, ans in enumerate(answers))
        
        # 检查上下文格式
        ctx_valid, ctx_issues = ErrorChecker.check_context_format(ctx)
        if not ctx_valid:
            global_stats['context_format_errors'] += 1
            error_log.append({
                'type': 'context_error',
                'question': question[:50] + '...' if len(question) > 50 else question,
                'model': ",".join(found_models),
                'issues': ctx_issues,
                'context_preview': ctx[:100] + '...' if len(ctx) > 100 else ctx
            })
            # 如果上下文格式有严重问题，跳过这个问题
            if any("为空" in issue for issue in ctx_issues):
                global_stats['skipped_questions'] += 1
                continue
        
        # 生成 prompt
        try:
            prompt = prompt_template.format(q=question, ctx=ctx)
            prompts.append(prompt)
        except Exception as e:
            error_log.append({
                'type': 'prompt_generation_error',
                'question': question[:50] + '...' if len(question) > 50 else question,
                'model': ",".join(found_models),
                'issues': [f"Prompt生成失败: {str(e)}"],
                'context_preview': ctx[:100] + '...' if len(ctx) > 100 else ctx
            })
            continue
        
        # 显示进度
        if global_stats['matched_questions'] % 10 == 0:
            print(f"  · 已处理 {global_stats['matched_questions']} 个匹配的问题")
    
    # 打印统计信息
    print(f"\n📊 处理统计：")
    print(f"  · 总问题数: {global_stats['total_questions']}")
    print(f"  · 匹配问题数: {global_stats['matched_questions']}")
    print(f"  · 跳过问题数: {global_stats['skipped_questions']}")
    print(f"  · 错误答案数: {global_stats['total_errors']}")
    print(f"  · 警告数: {global_stats['total_warnings']}")
    print(f"  · 上下文格式错误: {global_stats['context_format_errors']}")
    print(f"  · 生成prompt数: {len(prompts)}")
    
    return prompts, error_log

def save_error_log(error_log: List[Dict], filepath: Path):
    """保存错误日志"""
    if not error_log:
        print("  · 没有错误或警告需要记录")
        return
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== 答案纠错日志 ===\n\n")
        
        # 分别统计错误、警告和上下文错误
        errors = [e for e in error_log if e['type'] == 'error']
        warnings = [e for e in error_log if e['type'] == 'warning']
        context_errors = [e for e in error_log if e['type'] == 'context_error']
        prompt_errors = [e for e in error_log if e['type'] == 'prompt_generation_error']
        
        # 写入错误
        if errors:
            f.write(f"### 错误 ({len(errors)} 项) ###\n\n")
            for i, error in enumerate(errors, 1):
                f.write(f"{i}. 问题: {error['question']}\n")
                f.write(f"   模型: {error['model']}\n")
                f.write(f"   错误: {', '.join(error['issues'])}\n")
                f.write(f"   原始答案: {error.get('raw_answer', 'N/A')}\n")
                f.write("-" * 50 + "\n")
        
        # 写入上下文格式错误
        if context_errors:
            f.write(f"\n### 上下文格式错误 ({len(context_errors)} 项) ###\n\n")
            for i, error in enumerate(context_errors, 1):
                f.write(f"{i}. 问题: {error['question']}\n")
                f.write(f"   模型: {error['model']}\n")
                f.write(f"   问题: {', '.join(error['issues'])}\n")
                f.write(f"   上下文预览: {error.get('context_preview', 'N/A')}\n")
                f.write("-" * 50 + "\n")
        
        # 写入Prompt生成错误
        if prompt_errors:
            f.write(f"\n### Prompt生成错误 ({len(prompt_errors)} 项) ###\n\n")
            for i, error in enumerate(prompt_errors, 1):
                f.write(f"{i}. 问题: {error['question']}\n")
                f.write(f"   模型: {error['model']}\n")
                f.write(f"   错误: {', '.join(error['issues'])}\n")
                f.write("-" * 50 + "\n")
        
        # 写入警告
        if warnings:
            f.write(f"\n### 警告 ({len(warnings)} 项) ###\n\n")
            for i, warning in enumerate(warnings, 1):
                f.write(f"{i}. 问题: {warning['question']}\n")
                f.write(f"   模型: {warning['model']}\n")
                f.write(f"   警告: {', '.join(warning['issues'])}\n")
                f.write(f"   答案预览: {warning.get('answer_preview', 'N/A')}\n")
                f.write("-" * 50 + "\n")
    
    print(f"  · 错误日志已保存到: {filepath}")
    print(f"    - 答案错误: {len(errors)} 项")
    print(f"    - 上下文格式错误: {len(context_errors)} 项")
    print(f"    - Prompt生成错误: {len(prompt_errors)} 项")
    print(f"    - 警告: {len(warnings)} 项")

# === 5. 主程序 ===
def main():
    print("📖 开始处理数据...")
    print(f"  · 工作目录：{BASE_DIR}")
    print(f"  · 模糊匹配模型：gemini, grok, doubao")
    print(f"  · 输出格式：TXT文件（每个prompt为一段）")
    
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
            return
            
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON解析失败：{e}")
        return
    
    # === 生成prompts并写入TXT文件 ===
    print("\n⚙️ 生成 prompt 列表...")
    all_prompts, error_log = build_prompts(questions_data, prompt_template)
    
    if all_prompts:
        # 写入TXT文件
        print(f"\n📝 写入 TXT 文件：{OUT_TXT}")
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(all_prompts):
                f.write(prompt)
                # 如果不是最后一个prompt，添加分隔线
                if i < len(all_prompts) - 1:
                    f.write("\n-------------------\n")
        
        print(f"✅ 成功写入 {len(all_prompts)} 个prompt")
        
        # 保存错误日志
        print(f"\n📝 保存错误日志...")
        save_error_log(error_log, ERROR_LOG)
        
        # 统计信息
        total_chars = sum(len(prompt) for prompt in all_prompts)
        avg_chars = total_chars // len(all_prompts) if all_prompts else 0
        print(f"\n📊 内容统计：")
        print(f"  · 总prompt数量: {len(all_prompts)}")
        print(f"  · 总字符数: {total_chars:,}")
        print(f"  · 平均每个prompt: {avg_chars} 字符")
    else:
        print("⚠️ 警告：没有生成任何prompt")
    
    print("\n🎉 完成！")

if __name__ == "__main__":
    main()