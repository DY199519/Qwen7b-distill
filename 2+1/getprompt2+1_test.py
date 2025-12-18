#!/usr/bin/env python
# coding: utf-8
"""
multmm1_build_prompts.py
------------------------
Read JSON files and generate prompts:
  · Read JSON files containing questions and answers
  · Extract answers according to different model combinations
  · Check answer quality and filter out不合格的答案
  · Read prompt templates from external files
  · Generate independent JSON files for each combination
  · Save the answer from the last model in the combination separately
"""

import json, csv
from pathlib import Path
import re
from typing import Tuple, List, Dict, Any
from datetime import datetime

# ========== Output Configuration (placed at the top for easy modification) ==========
OUTPUT_DIR = Path(r"D:\qwensft\2+1")  # <-- Modify here to set the output directory
OUTPUT_FILE_PREFIX = "finalprompt"  # <-- Output file prefix
OUTPUT_FILE_SUFFIX = "2+1_test-1"  # <-- Output file suffix
# =====================================================

# === 1. Path Configuration ===
BASE_DIR = Path(r"D:\project7\prompt")
json_path = Path(r"D:\qwensft\testquestion\multi_model_answersTest500.json")

# Prompt file path
PROMPT_FILE = BASE_DIR / "prompt-2+1-1.txt"

# Model combination configuration
MODEL_COMBINATIONS = {
    "combination_1": ["gemini", "grok", "doubao"],
    # "combination_2": ["moonshot", "Yi", "gpt"],
    # "combination_3": ["llama", "vucina"],
}

# === 2. Quality Check Parameters ===
MIN_ANSWER_LENGTH = 100  # Minimum answer length
MIN_COMPLETE_LENGTH = 50  # Minimum length for completeness

# === 3. Answer Quality Check Function ===
def check_answer_quality(answer_text: str) -> Tuple[bool, str]:
    """
    Check answer quality (using the same standards as before)
    Returns: (is_qualified, issue_description)
    """
    # Check if empty
    if not answer_text or answer_text.strip() == "":
        return False, "Empty answer"
    
    answer_text = answer_text.strip()
    
    # Check length
    if len(answer_text) < MIN_COMPLETE_LENGTH:
        return False, f"Answer too short ({len(answer_text)} characters)"
    
    # Simple check: whether it ends with common complete punctuation
    if answer_text.endswith(('。', '！', '？', '.', '!', '?')):
        return True, "Complete"
    
    # If no punctuation at the end, check length
    if len(answer_text) < MIN_ANSWER_LENGTH:
        return False, f"No ending punctuation and short ({len(answer_text)} characters)"
    
    # Long answer without punctuation is also considered incomplete
    return False, f"No ending punctuation ({len(answer_text)} characters)"

# === 4. Read Prompt Template ===
def load_prompt_template():
    """Read prompt template from file"""
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            template = f.read().strip()
            print(f"✓ Successfully read prompt template: {PROMPT_FILE}")
            return template
    except FileNotFoundError:
        print(f"⚠️ Warning: Prompt file not found: {PROMPT_FILE}")
        # Use default template as backup
        default_template = 'Please answer: "{q}", improve your answer based on the following responses: {ctx}.'
        print(f"  Using default prompt template")
        return default_template

# === 5. Utility Functions ===
def fuzzy_match_model(model_pattern, available_models):
    """Fuzzy match model names and return the list of matched models"""
    matched_models = []
    for model in available_models:
        if model_pattern.lower() in model.lower():
            matched_models.append(model)
    return matched_models

def extract_answers_with_quality_check(question_data, model_patterns):
    """
    Extract answers of specified model patterns from question data and perform quality checks
    Returns: (list of answers, list of found models, list of quality issues)
    """
    answers = []
    found_models = []
    quality_issues = []
    
    if "answers" in question_data:
        available_models = list(question_data["answers"].keys())
        
        for pattern in model_patterns:
            # Use fuzzy matching to find models matching the pattern
            matched_models = fuzzy_match_model(pattern, available_models)
            
            # Select the first valid answer from matched models
            found_valid = False
            for model in matched_models:
                if model in question_data["answers"]:
                    model_answers = question_data["answers"][model]
                    if model_answers and len(model_answers) > 0:
                        # Take only the first answer
                        answer_text = model_answers[0].get("answer", "").strip()
                        
                        # Quality check
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
            
            # If no qualified answer found for this pattern, record the issue
            if not found_valid:
                quality_issues.append({
                    "model_pattern": pattern,
                    "issue": "No qualified answer found"
                })
    
    return answers, found_models, quality_issues

def build_records(questions_data, prompt_template, combo_name, model_patterns):
    """Construct record list for a single combination, including quality checks"""
    rows = []
    combo_count = 0
    skipped_count = 0
    quality_issues_summary = {}
    
    print(f"\n  Starting quality check...")
    
    for question, question_data in questions_data.items():
        # Extract answers for current combination models and perform quality checks
        answers, found_models, quality_issues = extract_answers_with_quality_check(
            question_data, model_patterns
        )
        
        # Record quality issues
        if quality_issues:
            quality_issues_summary[question] = quality_issues
        
        # The new prompt format requires at least 2 quality-qualified answers
        if len(answers) < 2:
            skipped_count += 1
            continue
        
        # Generate prompt
        try:
            if len(answers) >= 2:
                prompt = prompt_template.format(q=question, A1=answers[0], A2=answers[1])
            else:
                continue
        except KeyError as e:
            # If template format does not match, try old format
            ctx = "\n".join(f"Answer {i+1}: {ans}" for i, ans in enumerate(answers[:2]))
            try:
                prompt = prompt_template.format(q=question, ctx=ctx)
            except:
                print(f"  ⚠️ Warning: Prompt template format does not match, skipping question: {question[:50]}...")
                skipped_count += 1
                continue
        
        # Build record
        record = {
            "question": question,
            "prompt": prompt,
            "model": ",".join(found_models[:2]),
            "version": f"{combo_name}_{min(len(answers), 2)}_answers",
            "combination": combo_name,
            "answer_quality": "checked"  # Marked as passed quality check
        }
        
        # If there is a third answer
        if len(answers) >= 3 and len(found_models) >= 3:
            record["third_model"] = found_models[2]
            record["third_answer"] = answers[2]
        
        rows.append(record)
        combo_count += 1
    
    print(f"  · {combo_name} generated {combo_count} records")
    print(f"  · Skipped {skipped_count} records due to quality issues")
    
    # If there are quality issues, output detailed report
    if quality_issues_summary:
        issue_count = len(quality_issues_summary)
        print(f"  · Found {issue_count} questions with quality issues")
        
        # Save quality issue report
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
        print(f"  · Quality report saved to: {quality_report_file}")
    
    return rows

# === 6. Main Program ===
print("📖 Starting data processing...")
print(f"📁 Output directory: {OUTPUT_DIR}")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read prompt template
prompt_template = load_prompt_template()

# Read JSON data
print(f"\n📖 Reading JSON file: {json_path}")
try:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Check data structure
    if "questions" in data:
        questions_data = data["questions"]
        print(f"  · Found {len(questions_data)} questions")
    else:
        print("❌ Error: 'questions' field not found in JSON file")
        exit(1)
        
except FileNotFoundError:
    print(f"❌ Error: File not found {json_path}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Error: JSON parsing failed: {e}")
    exit(1)

# === 7. Generate Records and Write to JSON ===
print("\n⚙️ Generating prompt records...")

total_count = 0
total_skipped = 0

# Generate independent JSON files for each combination
for combo_name, model_patterns in MODEL_COMBINATIONS.items():
    print(f"\n📋 Processing combination {combo_name}: {', '.join(model_patterns)}")
    
    # Generate records for current combination
    combo_rows = build_records(questions_data, prompt_template, combo_name, model_patterns)
    
    if combo_rows:
        # Create independent JSON file for each combination
        out_json = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_{combo_name}_{OUTPUT_FILE_SUFFIX}.json"
        
        # Write to JSON file
        print(f"📝 Writing to JSON file: {out_json}")
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(combo_rows, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully wrote {len(combo_rows)} records to {out_json.name}")
        total_count += len(combo_rows)
    else:
        print(f"⚠️ Warning: {combo_name} did not generate any records (all answers failed quality checks)")

# Generate overall statistics report
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

print(f"\n📊 Total {total_count} records generated, distributed across {len(MODEL_COMBINATIONS)} files")
print(f"📊 Overall statistics saved to: {summary_file}")

print("\n🎉 Completed!")
