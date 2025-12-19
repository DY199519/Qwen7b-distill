#!/usr/bin/env python
# coding: utf-8
"""
grade_quality_separator.py
--------------------------
Based on answer quality check results, split the grading file into two parts:
1. Grades for questions with quality issues (unreliable)
2. Grades for questions with good quality (reliable)
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from datetime import datetime

# ========= Configuration Parameters ===========================================================
# Input path
INPUT_DIR = Path(r"D:\project7\MM\result")

# Answer file (for quality check)
ANSWERS_FILE = INPUT_DIR / "multi_model_answer-1-700.json"

# Grading file (needs to be separated based on quality check results)
GRADES_FILE = INPUT_DIR / "3+1" / "grades-3+1-700-1700.json"

# Output directory
OUTPUT_DIR = INPUT_DIR / "quality_separated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
PROBLEMATIC_GRADES_FILE = OUTPUT_DIR / "grades_with_quality_issues.json"
GOOD_GRADES_FILE = OUTPUT_DIR / "grades_without_issues.json"
QUALITY_CHECK_REPORT = OUTPUT_DIR / "quality_check_report.json"
PROBLEMATIC_ANSWERS_FILE = OUTPUT_DIR / "answers_with_quality_issues.json"  # New: file for problematic answers

# Quality check parameters
MIN_ANSWER_LENGTH = 100  # Minimum answer length
MIN_COMPLETE_LENGTH = 50  # Minimum length for completeness

# Models to check
MODELS_TO_CHECK = [
    "gemini-2.5-flash",
    "grok-3", 
    "doubao-pro-256k",
    "deepseek-v3"
]

# Error patterns
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
    r'API调用失败'  # Kept in Chinese as it's a specific error message pattern
]

# ========= Utility Functions ===========================================================
def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file"""
    if not file_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return {}
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load file {file_path}: {e}")
        return {}

def save_json_file(data: Any, file_path: Path, description: str = ""):
    """Save data to a JSON file"""
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ {description}saved to: {file_path}")
    except Exception as e:
        print(f"❌ Save failed: {e}")

def check_answer_quality(answer_text: str) -> Tuple[bool, str]:
    """
    Check answer quality (using standards from the first script)
    Returns: (has_issue, issue_description)
    """
    # Check if empty
    if not answer_text or answer_text.strip() == "":
        return True, "Empty answer"
    
    answer_text = answer_text.strip()
    
    # Check length
    if len(answer_text) < MIN_COMPLETE_LENGTH:
        return True, f"Answer too short ({len(answer_text)} characters)"
    
    # Simple check: whether it ends with common punctuation marks
    if answer_text.endswith(('。', '！', '？', '.', '!', '?')):
        return False, "Complete"
    
    # If no punctuation at the end, check length
    if len(answer_text) < MIN_ANSWER_LENGTH:
        return True, f"No ending punctuation and short ({len(answer_text)} characters)"
    
    # Long answer without punctuation is also considered incomplete (consistent with first script)
    return True, f"No ending punctuation ({len(answer_text)} characters)"

def check_answers_file(file_path: Path) -> Tuple[Set[str], Dict[str, List[Dict]], Dict[str, Any]]:
    """
    Check answer file to find questions with quality issues
    Returns: (set of problematic questions, issue details, original questions data)
    """
    print(f"\n🔍 Starting to check answer file: {file_path}")
    
    # Load data
    data = load_json_file(file_path)
    if not data:
        return set(), {}, {}
    
    # Get questions section
    questions_data = data.get("questions", {})
    if not questions_data:
        print("❌ No 'questions' field found")
        return set(), {}, {}
    
    print(f"📊 Found {len(questions_data)} questions")
    
    # Check each question
    problematic_questions = set()
    problem_details = {}
    
    total_checks = 0
    total_issues = 0
    
    for question, question_data in questions_data.items():
        answers = question_data.get("answers", {})
        question_has_issue = False
        question_issues = []
        
        # Check answers from each model
        for model in MODELS_TO_CHECK:
            total_checks += 1
            
            if model not in answers:
                question_has_issue = True
                question_issues.append({
                    "model": model,
                    "issue": "Missing answer",
                    "details": "No answer from this model"
                })
                total_issues += 1
            else:
                answer_list = answers[model]
                if not answer_list or len(answer_list) == 0:
                    question_has_issue = True
                    question_issues.append({
                        "model": model,
                        "issue": "Empty answer list",
                        "details": "Answer list is empty"
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
        
        # Record problematic questions
        if question_has_issue:
            problematic_questions.add(question)
            problem_details[question] = question_issues
    
    print(f"\n📊 Check completed:")
    print(f"  · Total checks: {total_checks}")
    print(f"  · Issues found: {total_issues}")
    print(f"  · Problematic questions: {len(problematic_questions)}")
    print(f"  · Problem-free questions: {len(questions_data) - len(problematic_questions)}")
    
    return problematic_questions, problem_details, questions_data

def separate_grades_by_quality(grades_file: Path, 
                             problematic_questions: Set[str],
                             problem_details: Dict[str, List[Dict]],
                             questions_data: Dict[str, Any]) -> Tuple[int, int]:
    """
    Separate grading file based on quality issues and save problematic answers
    Returns: (count of problematic grades, count of good grades)
    """
    print(f"\n📂 Starting to process grading file: {grades_file}")
    
    # Load grading data
    grades_data = load_json_file(grades_file)
    if not grades_data:
        return 0, 0
    
    # Get statistics and detailed results
    statistics = grades_data.get("statistics", {})
    detailed_results = grades_data.get("detailed_results", [])
    
    print(f"📊 Found {len(detailed_results)} grading results")
    
    # Separate data
    problematic_grades = []
    good_grades = []
    problematic_answers = {}  # New: collect problematic answers
    
    for grade_item in detailed_results:
        question = grade_item.get("question", "")
        
        if question in problematic_questions:
            # Add quality issue information
            grade_item["quality_issues"] = problem_details.get(question, [])
            grade_item["has_quality_issues"] = True
            problematic_grades.append(grade_item)
            
            # Collect problematic answer data
            if question in questions_data:
                problematic_answers[question] = questions_data[question]
        else:
            grade_item["has_quality_issues"] = False
            good_grades.append(grade_item)
    
    print(f"\n📊 Separation results:")
    print(f"  · Grades with quality issues: {len(problematic_grades)}")
    print(f"  · Grades with good quality: {len(good_grades)}")
    print(f"  · Answers with quality issues: {len(problematic_answers)}")
    
    # Save problematic answers (using original format)
    if problematic_answers:
        # Calculate summary information for problematic answers
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
        save_json_file(problematic_answers_data, PROBLEMATIC_ANSWERS_FILE, "Answers with quality issues")
    
    # Recalculate statistics
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
        
        # Recalculate score distribution
        new_stats["score_distribution"] = {
            "0-20": len([s for s in all_scores if s < 20]),
            "20-30": len([s for s in all_scores if 20 <= s < 30]),
            "30-40": len([s for s in all_scores if 30 <= s < 40]),
            "40-50": len([s for s in all_scores if 40 <= s <= 50])
        }
        
        new_stats["separation_time"] = datetime.now().isoformat()
        
        return new_stats
    
    # Save problematic grades
    if problematic_grades:
        problematic_data = {
            "metadata": {
                "source_grades_file": str(grades_file),
                "source_answers_file": str(ANSWERS_FILE),
                "separation_reason": "Scores unreliable due to answer quality issues",
                "total_issues": sum(len(issues) for issues in problem_details.values()),
                "separation_time": datetime.now().isoformat()
            },
            "statistics": recalculate_stats(problematic_grades, statistics),
            "detailed_results": problematic_grades
        }
        save_json_file(problematic_data, PROBLEMATIC_GRADES_FILE, "Grades with quality issues")
    
    # Save good quality grades
    if good_grades:
        good_data = {
            "metadata": {
                "source_grades_file": str(grades_file),
                "source_answers_file": str(ANSWERS_FILE),
                "separation_reason": "Answers of good quality, scores reliable",
                "separation_time": datetime.now().isoformat()
            },
            "statistics": recalculate_stats(good_grades, statistics),
            "detailed_results": good_grades
        }
        save_json_file(good_data, GOOD_GRADES_FILE, "Grades with good quality")
    
    return len(problematic_grades), len(good_grades)

def generate_quality_report(problematic_questions: Set[str],
                          problem_details: Dict[str, List[Dict]],
                          problematic_count: int,
                          good_count: int):
    """Generate quality check report"""
    # Count issue types
    issue_types = {}
    model_issues = {}
    
    for question, issues in problem_details.items():
        for issue in issues:
            # Count issue types
            issue_type = issue.get("issue", "Unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            # Count model issues
            model = issue.get("model", "Unknown")
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
        "sample_issues": list(problem_details.items())[:10]  # Samples of first 10 issues
    }
    
    save_json_file(report, QUALITY_CHECK_REPORT, "Quality check report")
    
    # Print report summary
    print("\n" + "="*60)
    print("📊 Quality Check Report Summary")
    print("="*60)
    print(f"\n🔍 Issue type distribution:")
    for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  · {issue_type}: {count}")
    
    print(f"\n🤖 Issues by model:")
    for model, count in sorted(model_issues.items(), key=lambda x: x[1], reverse=True):
        print(f"  · {model}: {count}")
    
    print(f"\n📈 Grade reliability:")
    print(f"  · Reliable grades: {good_count} ({report['summary']['reliability_rate']})")
    print(f"  · Unreliable grades: {problematic_count}")

def main():
    """Main function"""
    print("🚀 Starting grade quality separator...")
    print(f"📁 Answer file: {ANSWERS_FILE}")
    print(f"📁 Grade file: {GRADES_FILE}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Step 1: Check answer file to find problematic questions
    problematic_questions, problem_details, questions_data = check_answers_file(ANSWERS_FILE)
    
    if not problematic_questions:
        print("\n✅ All answers are of good quality, no need to separate grade files")
        return
    
    # Step 2: Separate grade file based on quality issues and save problematic answers
    problematic_count, good_count = separate_grades_by_quality(
        GRADES_FILE, 
        problematic_questions, 
        problem_details,
        questions_data  # Pass original questions data
    )
    
    # Step 3: Generate quality check report
    generate_quality_report(
        problematic_questions,
        problem_details,
        problematic_count,
        good_count
    )
    
    print(f"\n✅ Processing completed!")
    print(f"  · Problematic grades: {PROBLEMATIC_GRADES_FILE}")
    print(f"  · Reliable grades: {GOOD_GRADES_FILE}")
    print(f"  · Problematic answers: {PROBLEMATIC_ANSWERS_FILE}")
    print(f"  · Quality report: {QUALITY_CHECK_REPORT}")

if __name__ == "__main__":
    main()
