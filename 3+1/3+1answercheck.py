#!/usr/bin/env python
# coding: utf-8
"""
grade_quality_separator.py
--------------------------
根据答案质量检查结果，将评分文件分成两部分：
1. 有质量问题题目的评分（不可靠）
2. 质量良好题目的评分（可靠）
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from datetime import datetime

# ========= 配置参数 ===========================================================
# 输入路径
INPUT_DIR = Path(r"D:\project7\MM\result")

# 答案文件（用于检查质量）
ANSWERS_FILE = INPUT_DIR / "multi_model_answer-1-700.json"

# 评分文件（需要根据质量检查结果分离）
GRADES_FILE = INPUT_DIR / "3+1" / "grades-3+1-700-1700.json"

# 输出目录
OUTPUT_DIR = INPUT_DIR / "quality_separated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
PROBLEMATIC_GRADES_FILE = OUTPUT_DIR / "grades_with_quality_issues.json"
GOOD_GRADES_FILE = OUTPUT_DIR / "grades_without_issues.json"
QUALITY_CHECK_REPORT = OUTPUT_DIR / "quality_check_report.json"
PROBLEMATIC_ANSWERS_FILE = OUTPUT_DIR / "answers_with_quality_issues.json"  # 新增：有问题的答案文件

# 质量检查参数
MIN_ANSWER_LENGTH = 100  # 最小答案长度
MIN_COMPLETE_LENGTH = 50  # 完整性最小长度

# 需要检查的模型
MODELS_TO_CHECK = [
    "gemini-2.5-flash",
    "grok-3", 
    "doubao-pro-256k",
    "deepseek-v3"
]

# 错误模式
ERROR_PATTERNS = [
    r'^error:',
    r'^exception:',
    r'^\s*$',
    r'^null$',
    r'^undefined$',
    r'^N/A$',
    r'request failed',
    r'rate limit',
    r'timeout',
    r'\[ERROR',
    r'API调用失败'
]

# ========= 工具函数 ===========================================================
def load_json_file(file_path: Path) -> Dict[str, Any]:
    """加载JSON文件"""
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return {}
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文件失败 {file_path}: {e}")
        return {}

def save_json_file(data: Any, file_path: Path, description: str = ""):
    """保存JSON文件"""
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ {description}已保存到: {file_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def check_answer_quality(answer_text: str) -> Tuple[bool, str]:
    """
    检查答案质量（使用第一个脚本的标准）
    返回: (是否有问题, 问题描述)
    """
    # 检查是否为空
    if not answer_text or answer_text.strip() == "":
        return True, "空答案"
    
    answer_text = answer_text.strip()
    
    # 检查长度
    if len(answer_text) < MIN_COMPLETE_LENGTH:
        return True, f"答案过短({len(answer_text)}字符)"
    
    # 简单检查：是否以常见的完整标点结尾
    if answer_text.endswith(('。', '！', '？', '.', '!', '?')):
        return False, "完整"
    
    # 如果没有标点结尾，检查长度
    if len(answer_text) < MIN_ANSWER_LENGTH:
        return True, f"无结尾标点且较短({len(answer_text)}字符)"
    
    # 长答案但无标点，也视为不完整（与第一个脚本一致）
    return True, f"无结尾标点({len(answer_text)}字符)"

def check_answers_file(file_path: Path) -> Tuple[Set[str], Dict[str, List[Dict]], Dict[str, Any]]:
    """
    检查答案文件，找出有质量问题的题目
    返回: (有问题的题目集合, 问题详情, 原始questions数据)
    """
    print(f"\n🔍 开始检查答案文件: {file_path}")
    
    # 加载数据
    data = load_json_file(file_path)
    if not data:
        return set(), {}, {}
    
    # 获取questions部分
    questions_data = data.get("questions", {})
    if not questions_data:
        print("❌ 未找到questions字段")
        return set(), {}, {}
    
    print(f"📊 找到 {len(questions_data)} 个问题")
    
    # 检查每个题目
    problematic_questions = set()
    problem_details = {}
    
    total_checks = 0
    total_issues = 0
    
    for question, question_data in questions_data.items():
        answers = question_data.get("answers", {})
        question_has_issue = False
        question_issues = []
        
        # 检查每个模型的答案
        for model in MODELS_TO_CHECK:
            total_checks += 1
            
            if model not in answers:
                question_has_issue = True
                question_issues.append({
                    "model": model,
                    "issue": "答案缺失",
                    "details": "该模型没有答案"
                })
                total_issues += 1
            else:
                answer_list = answers[model]
                if not answer_list or len(answer_list) == 0:
                    question_has_issue = True
                    question_issues.append({
                        "model": model,
                        "issue": "空答案列表",
                        "details": "答案列表为空"
                    })
                    total_issues += 1
                else:
                    answer_text = answer_list[0].get("answer", "")
                    has_problem, problem_desc = check_answer_quality(answer_text)
                    
                    if has_problem:
                        question_has_issue = True
                        question_issues.append({
                            "model": model,
                            "issue": problem_desc,
                            "answer_length": len(answer_text),
                            "answer_preview": answer_text[:100] + "..." if len(answer_text) > 100 else answer_text
                        })
                        total_issues += 1
        
        # 记录有问题的题目
        if question_has_issue:
            problematic_questions.add(question)
            problem_details[question] = question_issues
    
    print(f"\n📊 检查完成:")
    print(f"  · 总检查项: {total_checks}")
    print(f"  · 发现问题: {total_issues}")
    print(f"  · 有问题的题目: {len(problematic_questions)}")
    print(f"  · 没问题的题目: {len(questions_data) - len(problematic_questions)}")
    
    return problematic_questions, problem_details, questions_data

def separate_grades_by_quality(grades_file: Path, 
                             problematic_questions: Set[str],
                             problem_details: Dict[str, List[Dict]],
                             questions_data: Dict[str, Any]) -> Tuple[int, int]:
    """
    根据质量问题分离评分文件，并保存有问题的答案
    返回: (有问题的评分数, 没问题的评分数)
    """
    print(f"\n📂 开始处理评分文件: {grades_file}")
    
    # 加载评分数据
    grades_data = load_json_file(grades_file)
    if not grades_data:
        return 0, 0
    
    # 获取统计信息和详细结果
    statistics = grades_data.get("statistics", {})
    detailed_results = grades_data.get("detailed_results", [])
    
    print(f"📊 找到 {len(detailed_results)} 个评分结果")
    
    # 分离数据
    problematic_grades = []
    good_grades = []
    problematic_answers = {}  # 新增：收集有问题的答案
    
    for grade_item in detailed_results:
        question = grade_item.get("question", "")
        
        if question in problematic_questions:
            # 添加质量问题信息
            grade_item["quality_issues"] = problem_details.get(question, [])
            grade_item["has_quality_issues"] = True
            problematic_grades.append(grade_item)
            
            # 收集有问题的答案数据
            if question in questions_data:
                problematic_answers[question] = questions_data[question]
        else:
            grade_item["has_quality_issues"] = False
            good_grades.append(grade_item)
    
    print(f"\n📊 分离结果:")
    print(f"  · 有质量问题的评分: {len(problematic_grades)}")
    print(f"  · 质量良好的评分: {len(good_grades)}")
    print(f"  · 有质量问题的答案: {len(problematic_answers)}")
    
    # 保存有问题的答案（使用原始格式）
    if problematic_answers:
        # 计算有问题答案的汇总信息
        all_models = set()
        total_answer_count = 0
        for q_data in problematic_answers.values():
            answers = q_data.get("answers", {})
            for model in answers:
                all_models.add(model)
                total_answer_count += len(answers[model])
        
        problematic_answers_data = {
            "questions": problematic_answers,
            "summary": {
                "total_questions": len(problematic_answers),
                "total_models": len(all_models),
                "total_answers": total_answer_count,
                "models": sorted(list(all_models)),
                "quality_check_time": datetime.now().isoformat(),
                "quality_issues_summary": {
                    "total_issues": sum(len(issues) for issues in problem_details.values()),
                    "issues_by_question": {q: len(issues) for q, issues in problem_details.items() if q in problematic_answers}
                }
            }
        }
        save_json_file(problematic_answers_data, PROBLEMATIC_ANSWERS_FILE, "有质量问题的答案")
    
    # 重新计算统计信息
    def recalculate_stats(grades_list, original_stats):
        if not grades_list:
            return {}
        
        all_scores = [g["avg_scores"]["total"] for g in grades_list]
        all_scores_100 = [g["avg_score_100"] for g in grades_list]
        
        new_stats = original_stats.copy()
        new_stats["total_questions"] = len(grades_list)
        new_stats["valid_grades"] = len(grades_list)
        new_stats["total_average"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
        new_stats["total_average_100"] = round(sum(all_scores_100) / len(all_scores_100), 2) if all_scores_100 else 0
        
        # 重新计算分数分布
        new_stats["score_distribution"] = {
            "0-20": len([s for s in all_scores if s < 20]),
            "20-30": len([s for s in all_scores if 20 <= s < 30]),
            "30-40": len([s for s in all_scores if 30 <= s < 40]),
            "40-50": len([s for s in all_scores if 40 <= s <= 50])
        }
        
        new_stats["separation_time"] = datetime.now().isoformat()
        
        return new_stats
    
    # 保存有问题的评分
    if problematic_grades:
        problematic_data = {
            "metadata": {
                "source_grades_file": str(grades_file),
                "source_answers_file": str(ANSWERS_FILE),
                "separation_reason": "答案质量问题导致评分不可靠",
                "total_issues": sum(len(issues) for issues in problem_details.values()),
                "separation_time": datetime.now().isoformat()
            },
            "statistics": recalculate_stats(problematic_grades, statistics),
            "detailed_results": problematic_grades
        }
        save_json_file(problematic_data, PROBLEMATIC_GRADES_FILE, "有质量问题的评分")
    
    # 保存质量良好的评分
    if good_grades:
        good_data = {
            "metadata": {
                "source_grades_file": str(grades_file),
                "source_answers_file": str(ANSWERS_FILE),
                "separation_reason": "答案质量良好，评分可靠",
                "separation_time": datetime.now().isoformat()
            },
            "statistics": recalculate_stats(good_grades, statistics),
            "detailed_results": good_grades
        }
        save_json_file(good_data, GOOD_GRADES_FILE, "质量良好的评分")
    
    return len(problematic_grades), len(good_grades)

def generate_quality_report(problematic_questions: Set[str],
                          problem_details: Dict[str, List[Dict]],
                          problematic_count: int,
                          good_count: int):
    """生成质量检查报告"""
    # 统计问题类型
    issue_types = {}
    model_issues = {}
    
    for question, issues in problem_details.items():
        for issue in issues:
            # 统计问题类型
            issue_type = issue.get("issue", "未知")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            # 统计模型问题
            model = issue.get("model", "未知")
            model_issues[model] = model_issues.get(model, 0) + 1
    
    report = {
        "check_time": datetime.now().isoformat(),
        "files_checked": {
            "answers_file": str(ANSWERS_FILE),
            "grades_file": str(GRADES_FILE)
        },
        "summary": {
            "total_questions_with_issues": len(problematic_questions),
            "total_individual_issues": sum(len(issues) for issues in problem_details.values()),
            "grades_affected": problematic_count,
            "grades_reliable": good_count,
            "reliability_rate": f"{(good_count / (problematic_count + good_count) * 100):.2f}%" if (problematic_count + good_count) > 0 else "0%"
        },
        "issue_types": issue_types,
        "issues_by_model": model_issues,
        "sample_issues": list(problem_details.items())[:10]  # 前10个问题的示例
    }
    
    save_json_file(report, QUALITY_CHECK_REPORT, "质量检查报告")
    
    # 打印报告摘要
    print("\n" + "="*60)
    print("📊 质量检查报告摘要")
    print("="*60)
    print(f"\n🔍 问题类型分布:")
    for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  · {issue_type}: {count}")
    
    print(f"\n🤖 各模型问题数:")
    for model, count in sorted(model_issues.items(), key=lambda x: x[1], reverse=True):
        print(f"  · {model}: {count}")
    
    print(f"\n📈 评分可靠性:")
    print(f"  · 可靠评分: {good_count} ({report['summary']['reliability_rate']})")
    print(f"  · 不可靠评分: {problematic_count}")

def main():
    """主函数"""
    print("🚀 启动评分质量分离器...")
    print(f"📁 答案文件: {ANSWERS_FILE}")
    print(f"📁 评分文件: {GRADES_FILE}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    
    # 步骤1: 检查答案文件，找出有问题的题目
    problematic_questions, problem_details, questions_data = check_answers_file(ANSWERS_FILE)
    
    if not problematic_questions:
        print("\n✅ 所有题目的答案质量都良好，无需分离评分文件")
        return
    
    # 步骤2: 根据质量问题分离评分文件，并保存有问题的答案
    problematic_count, good_count = separate_grades_by_quality(
        GRADES_FILE, 
        problematic_questions, 
        problem_details,
        questions_data  # 传递原始questions数据
    )
    
    # 步骤3: 生成质量检查报告
    generate_quality_report(
        problematic_questions,
        problem_details,
        problematic_count,
        good_count
    )
    
    print(f"\n✅ 处理完成！")
    print(f"  · 有问题的评分: {PROBLEMATIC_GRADES_FILE}")
    print(f"  · 可靠的评分: {GOOD_GRADES_FILE}")
    print(f"  · 有问题的答案: {PROBLEMATIC_ANSWERS_FILE}")
    print(f"  · 质量报告: {QUALITY_CHECK_REPORT}")

if __name__ == "__main__":
    main()