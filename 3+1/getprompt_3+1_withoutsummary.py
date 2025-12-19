#!/usr/bin/env python
# coding: utf-8
"""
multmm1_build_prompts_txt.py
----------------------------
Read JSON files and generate prompts:
  · Read JSON files containing questions and answers
  · Fuzzy match models containing gemini, grok, doubao
  · Read prompt templates from external files
  · Generate prompts and write to TXT files, each prompt as a paragraph
  · Add error correction mechanism to check the validity of answers
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

# === 1. Path Configuration ===
BASE_DIR = Path(r"D:\project7\prompt")
BASE_DIR_1 = Path(r"D:\project7")
BASE_DIR_2 = Path(r"D:\project7\MM\result")

# Use the second JSON file
json_path = BASE_DIR_1 / "multi_model_answers9400-10000.json"

# Prompt file path
PROMPT_FILE = BASE_DIR_2 / "prompt-3+1withoutsummary.txt"

# Output file (changed to txt)
OUT_TXT = BASE_DIR_2 / "final_prompt_3+1-Test.txt"
ERROR_LOG = BASE_DIR_2 / "error_log.txt"  # Error log file

# === 2. Error Correction Configuration ===
class ErrorChecker:
    """Answer error correction checker"""
    
    # Minimum answer length
    MIN_ANSWER_LENGTH = 10
    
    # Maximum answer length (may be an error)
    MAX_ANSWER_LENGTH = 10000
    
    # Common error patterns
    ERROR_PATTERNS = [
        r'^error:',  # Starts with error
        r'^exception:',  # Starts with exception
        r'^\s*$',  # Pure whitespace
        r'^null$',  # null value
        r'^undefined$',  # undefined value
        r'^N/A$',  # N/A
        r'^\[.*error.*\]$',  # Content in square brackets containing error
        r'^\{.*error.*\}$',  # Content in curly brackets containing error
    ]
    
    # Suspicious patterns (warn but not filter)
    WARNING_PATTERNS = [
        r'^.{1,9}$',  # Overly short answers (less than 10 characters)
        r'^\d+$',  # Pure numbers
        r'^[^\u4e00-\u9fa5a-zA-Z]+$',  # No Chinese or English letters
        r'(.)\1{10,}',  # Repeated characters more than 10 times
    ]
    
    @classmethod
    def check_context_format(cls, context: str) -> Tuple[bool, List[str]]:
        """
        Check if the generated context format is correct
        Ensure each "Answer X:" has actual content after it
        """
        issues = []
        
        # Extract all answer parts
        pattern = r'Answer(\d+)：(.*?)(?=Answer\d+：|$)'
        matches = re.findall(pattern, context, re.DOTALL)
        
        if not matches:
            issues.append("Standard 'Answer X:' format not found")
            return False, issues
        
        # Check each answer
        for num, content in matches:
            content = content.strip()
            if not content:
                issues.append(f"Answer {num} is empty")
            elif len(content) < cls.MIN_ANSWER_LENGTH:
                issues.append(f"Answer {num} is too short ({len(content)} characters)")
        
        # Check if answer numbers are consecutive
        numbers = [int(num) for num, _ in matches]
        numbers.sort()
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            issues.append(f"Answer numbers are not consecutive: {numbers}")
        
        return len(issues) == 0, issues
    
    @classmethod
    def check_answer(cls, answer: str, question: str = "") -> Tuple[bool, str, List[str]]:
        """
        Check if the answer is valid
        Returns: (is_valid, cleaned_answer, error/warning list)
        """
        errors = []
        warnings = []
        
        # Basic check
        if not answer:
            errors.append("Answer is empty")
            return False, "", errors
        
        # Type check
        if not isinstance(answer, str):
            errors.append(f"Answer type error: {type(answer)}")
            return False, "", errors
        
        # Clean answer (remove leading and trailing whitespace)
        cleaned_answer = answer.strip()
        
        # Length check
        if len(cleaned_answer) < cls.MIN_ANSWER_LENGTH:
            warnings.append(f"Answer is too short ({len(cleaned_answer)} characters)")
        elif len(cleaned_answer) > cls.MAX_ANSWER_LENGTH:
            warnings.append(f"Answer is too long ({len(cleaned_answer)} characters)")
        
        # Error pattern check
        for pattern in cls.ERROR_PATTERNS:
            if re.match(pattern, cleaned_answer, re.IGNORECASE):
                errors.append(f"Matches error pattern: {pattern}")
                return False, cleaned_answer, errors
        
        # Warning pattern check
        for pattern in cls.WARNING_PATTERNS:
            if re.match(pattern, cleaned_answer, re.IGNORECASE):
                warnings.append(f"Matches suspicious pattern: {pattern}")
        
        # Special character check
        if cleaned_answer.count('\n') > 50:
            warnings.append("Contains too many line breaks")
        
        if cleaned_answer.count(' ') / len(cleaned_answer) > 0.5:
            warnings.append("Excessive proportion of spaces")
        
        # Return result
        is_valid = len(errors) == 0
        all_issues = errors + warnings
        
        return is_valid, cleaned_answer, all_issues

# === 3. Read Prompt Template ===
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

# === 4. Utility Functions ===
def extract_answers_fuzzy(question_data: Dict, question: str, error_log: List[Dict]) -> Tuple[List[str], List[str], Dict]:
    """
    Fuzzy match models containing gemini, grok, doubao and extract answers
    Add error correction mechanism
    """
    answers = []
    found_models = []
    target_keywords = ['gemini', 'grok', 'doubao']
    
    # Statistical information
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
            
            # Fuzzy match: check if model name contains target keywords
            model_lower = model_name.lower()
            for keyword in target_keywords:
                if keyword in model_lower:
                    stats['matched_models'] += 1
                    
                    if model_answers and len(model_answers) > 0:
                        # Take only the first answer
                        raw_answer = model_answers[0].get("answer", "")
                        
                        # Error correction check
                        is_valid, cleaned_answer, issues = ErrorChecker.check_answer(raw_answer, question)
                        
                        if is_valid:
                            if cleaned_answer:  # Confirm again that the cleaned answer is not empty
                                answers.append(cleaned_answer)
                                found_models.append(model_name)
                                stats['valid_answers'] += 1
                                
                                # If there are warnings, record them but do not prevent usage
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
                    break  # Exit inner loop once a match is found
    
    return answers, found_models, stats

def build_prompts(questions_data: Dict, prompt_template: str) -> Tuple[List[str], List[Dict]]:
    """Construct prompt list, return prompt list and error log"""
    prompts = []
    error_log = []
    
    # Global statistics
    global_stats = {
        'total_questions': 0,
        'matched_questions': 0,
        'skipped_questions': 0,
        'total_errors': 0,
        'total_warnings': 0,
        'context_format_errors': 0
    }
    
    print(f"\n📋 Starting data processing...")
    
    for question, question_data in questions_data.items():
        global_stats['total_questions'] += 1
        
        # Extract fuzzy matched answers (with error correction)
        answers, found_models, stats = extract_answers_fuzzy(question_data, question, error_log)
        
        # Update global statistics
        global_stats['total_errors'] += stats['invalid_answers']
        global_stats['total_warnings'] += stats['warnings']
        
        if not answers:
            global_stats['skipped_questions'] += 1
            continue
        
        global_stats['matched_questions'] += 1
        
        # Build context
        ctx = "\n".join(f"Answer {i+1}：{ans}" for i, ans in enumerate(answers))
        
        # Check context format
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
            # If there are serious issues with context format, skip this question
            if any("is empty" in issue for issue in ctx_issues):
                global_stats['skipped_questions'] += 1
                continue
        
        # Generate prompt
        try:
            prompt = prompt_template.format(q=question, ctx=ctx)
            prompts.append(prompt)
        except Exception as e:
            error_log.append({
                'type': 'prompt_generation_error',
                'question': question[:50] + '...' if len(question) > 50 else question,
                'model': ",".join(found_models),
                'issues': [f"Prompt generation failed: {str(e)}"],
                'context_preview': ctx[:100] + '...' if len(ctx) > 100 else ctx
            })
            continue
        
        # Display progress
        if global_stats['matched_questions'] % 10 == 0:
            print(f"  · Processed {global_stats['matched_questions']} matched questions")
    
    # Print statistics
    print(f"\n📊 Processing statistics：")
    print(f"  · Total questions: {global_stats['total_questions']}")
    print(f"  · Matched questions: {global_stats['matched_questions']}")
    print(f"  · Skipped questions: {global_stats['skipped_questions']}")
    print(f"  · Invalid answers: {global_stats['total_errors']}")
    print(f"  · Warnings: {global_stats['total_warnings']}")
    print(f"  · Context format errors: {global_stats['context_format_errors']}")
    print(f"  · Generated prompts: {len(prompts)}")
    
    return prompts, error_log

def save_error_log(error_log: List[Dict], filepath: Path):
    """Save error log"""
    if not error_log:
        print("  · No errors or warnings to record")
        return
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== Answer Error Correction Log ===\n\n")
        
        # Separate statistics for errors, warnings and context errors
        errors = [e for e in error_log if e['type'] == 'error']
        warnings = [e for e in error_log if e['type'] == 'warning']
        context_errors = [e for e in error_log if e['type'] == 'context_error']
        prompt_errors = [e for e in error_log if e['type'] == 'prompt_generation_error']
        
        # Write errors
        if errors:
            f.write(f"### Errors ({len(errors)} items) ###\n\n")
            for i, error in enumerate(errors, 1):
                f.write(f"{i}. Question: {error['question']}\n")
                f.write(f"   Model: {error['model']}\n")
                f.write(f"   Errors: {', '.join(error['issues'])}\n")
                f.write(f"   Raw answer: {error.get('raw_answer', 'N/A')}\n")
                f.write("-" * 50 + "\n")
        
        # Write context format errors
        if context_errors:
            f.write(f"\n### Context Format Errors ({len(context_errors)} items) ###\n\n")
            for i, error in enumerate(context_errors, 1):
                f.write(f"{i}. Question: {error['question']}\n")
                f.write(f"   Model: {error['model']}\n")
                f.write(f"   Issues: {', '.join(error['issues'])}\n")
                f.write(f"   Context preview: {error.get('context_preview', 'N/A')}\n")
                f.write("-" * 50 + "\n")
        
        # Write Prompt generation errors
        if prompt_errors:
            f.write(f"\n### Prompt Generation Errors ({len(prompt_errors)} items) ###\n\n")
            for i, error in enumerate(prompt_errors, 1):
                f.write(f"{i}. Question: {error['question']}\n")
                f.write(f"   Model: {error['model']}\n")
                f.write(f"   Errors: {', '.join(error['issues'])}\n")
                f.write("-" * 50 + "\n")
        
        # Write warnings
        if warnings:
            f.write(f"\n### Warnings ({len(warnings)} items) ###\n\n")
            for i, warning in enumerate(warnings, 1):
                f.write(f"{i}. Question: {warning['question']}\n")
                f.write(f"   Model: {warning['model']}\n")
                f.write(f"   Warnings: {', '.join(warning['issues'])}\n")
                f.write(f"   Answer preview: {warning.get('answer_preview', 'N/A')}\n")
                f.write("-" * 50 + "\n")
    
    print(f"  · Error log saved to: {filepath}")
    print(f"    - Answer errors: {len(errors)} items")
    print(f"    - Context format errors: {len(context_errors)} items")
    print(f"    - Prompt generation errors: {len(prompt_errors)} items")
    print(f"    - Warnings: {len(warnings)} items")

# === 5. Main Program ===
def main():
    print("📖 Starting data processing...")
    print(f"  · Working directory: {BASE_DIR}")
    print(f"  · Fuzzy matching models: gemini, grok, doubao")
    print(f"  · Output format: TXT file (each prompt as a paragraph)")
    
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
            return
            
    except FileNotFoundError:
        print(f"❌ Error: File not found {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: JSON parsing failed: {e}")
        return
    
    # === Generate prompts and write to TXT file ===
    print("\n⚙️ Generating prompt list...")
    all_prompts, error_log = build_prompts(questions_data, prompt_template)
    
    if all_prompts:
        # Write to TXT file
        print(f"\n📝 Writing to TXT file: {OUT_TXT}")
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(all_prompts):
                f.write(prompt)
                # Add separator if not the last prompt
                if i < len(all_prompts) - 1:
                    f.write("\n-------------------\n")
        
        print(f"✅ Successfully wrote {len(all_prompts)} prompts")
        
        # Save error log
        print(f"\n📝 Saving error log...")
        save_error_log(error_log, ERROR_LOG)
        
        # Statistical information
        total_chars = sum(len(prompt) for prompt in all_prompts)
        avg_chars = total_chars // len(all_prompts) if all_prompts else 0
        print(f"\n📊 Content statistics：")
        print(f"  · Total prompt count: {len(all_prompts)}")
        print(f"  · Total characters: {total_chars:,}")
        print(f"  · Average per prompt: {avg_chars} characters")
    else:
        print("⚠️ Warning: No prompts were generated")
    
    print("\n🎉 Completed!")

if __name__ == "__main__":
    main()
