#!/usr/bin/env python
# coding: utf-8
"""
multmm2_run234.py
-----------------
Reads prompt CSV files with combination fields and executes model calls.
Does not save progress; checks in real-time if each question already exists in the answer file.
Unified JSON output is written to OUTPUT_DIR.
Enhanced features:
- Data integrity checks
- Answer quality verification
- Detailed error logging
- Intelligent retry mechanism
"""

import csv, json, time
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# ========== 0. Path Configuration =======================================================
# Output file path - placed at the top
OUTPUT_FILE = Path(r"D:\project7\MM\result\3+1\deepseek_answers_without_summary3+1-9400-10000.json")

BASE_DIR_1   = Path(r"D:\project7\MM\3+1")           # Root directory for data files
BASE_DIR = Path(r"D:\project7\prompt")
OUTPUT_DIR = Path(r"D:\project7\MM\result")            # <-- Only modify this to change output location
OUTPUT_DIR_1 = Path(r"D:\project7\MM\result\3+1")            # <-- Only modify this to change output location

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_1.mkdir(parents=True, exist_ok=True)

# Modified to read CSV files with combination
PROMPT_CSV = OUTPUT_DIR / "final_prompt_3+1-9400-10000.csv"

GROUPED_JSON = OUTPUT_DIR / "multi_model_answer9400-10000.json"

# Run log file
RUN_LOG = OUTPUT_DIR_1 / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ========== 1. Model List =======================================================
MODEL_CFGS = [
    {
        "model_name": "deepseek-v3",
        "api_key": "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa",
        "base_url": "https://usa.vimsai.com/v1",
        "timeout": 60,  # Timeout in seconds
        "max_retry": 3,  # Maximum number of retries
    }
    # {
    #     "model_name": "qwen2.5-72b-instruct",
    #     "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
    #     "base_url": "https://api.vansai.cn/v1",
    #     "timeout": 60,
    #     "max_retry": 3,
    # },
]

# ========== 2. Answer Validation Class ====================================================
class AnswerValidator:
    """Answer validator"""
    
    # Minimum answer length
    MIN_ANSWER_LENGTH = 5
    
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
    ]
    
    @classmethod
    def validate_answer(cls, answer: str, question: str = "") -> Tuple[bool, List[str]]:
        """Validate if the answer is valid"""
        issues = []
        
        if not answer:
            issues.append("Answer is empty")
            return False, issues
        
        if not isinstance(answer, str):
            issues.append(f"Answer type error: {type(answer)}")
            return False, issues
        
        answer = answer.strip()
        
        # Length check
        if len(answer) < cls.MIN_ANSWER_LENGTH:
            issues.append(f"Answer too short ({len(answer)} characters)")
        
        # Error pattern check
        for pattern in cls.ERROR_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                issues.append(f"Matches error pattern: {pattern}")
                return False, issues
        
        return len(issues) == 0, issues

# ========== 3. Log Recorder ====================================================
class Logger:
    """Simple log recorder"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = datetime.now()
        self._write(f"=== Run started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    def _write(self, message: str):
        """Write to log"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    
    def info(self, message: str):
        """Record information"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{timestamp}] INFO: {message}")
        print(f"📝 {message}")
    
    def error(self, message: str):
        """Record error"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{timestamp}] ERROR: {message}")
        print(f"❌ {message}")
    
    def warning(self, message: str):
        """Record warning"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{timestamp}] WARNING: {message}")
        print(f"⚠️ {message}")
    
    def summary(self, stats: dict):
        """Record statistical summary"""
        elapsed = datetime.now() - self.start_time
        self._write(f"\n=== Run Statistics ===")
        self._write(f"Total time elapsed: {elapsed}")
        for key, value in stats.items():
            self._write(f"{key}: {value}")
        self._write("=" * 50)

# ========== 4. Utility Functions =======================================================
def find_existing_answer(question: str, output_file: Path) -> dict:
    """
    Find the answer for a specified question in the output file
    Returns: The found answer item, or None if not found
    """
    if not output_file.exists():
        return None
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            if item.get('question') == question:
                return item
                
    except Exception as e:
        print(f"❌ Failed to read answer file: {e}")
        
    return None

def is_answer_complete_and_valid(item: dict) -> Tuple[bool, List[str]]:
    """
    Check if the answer item is complete and valid
    Returns: (Is valid, list of issues)
    """
    if not item:
        return False, ["Answer item is empty"]
    
    issues = []
    
    # Check direct_reply
    if not item.get('direct_reply'):
        issues.append("Missing direct_reply")
    else:
        is_valid, sub_issues = AnswerValidator.validate_answer(item['direct_reply'])
        if not is_valid:
            issues.extend([f"direct_reply: {issue}" for issue in sub_issues])
    
    # Check default_reply
    if not item.get('default_reply'):
        issues.append("Missing default_reply")
    else:
        is_valid, sub_issues = AnswerValidator.validate_answer(item['default_reply'])
        if not is_valid:
            issues.extend([f"default_reply: {issue}" for issue in sub_issues])
    
    return len(issues) == 0, issues

def ask(api: OpenAI, model: str, prompt: str, logger: Logger, 
        timeout: int = 60, max_retry: int = 3, pause: float = 2.0) -> Tuple[str, bool, List[str]]:
    """
    Call the model API
    Returns: (Answer, success status, list of errors)
    """
    errors = []
    
    for i in range(1, max_retry + 1):
        try:
            # Set timeout
            rsp = api.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout
            )
            answer = rsp.choices[0].message.content.strip()
            
            # Validate answer
            is_valid, issues = AnswerValidator.validate_answer(answer, prompt[:50])
            
            if is_valid and answer:
                return answer, True, []
            else:
                errors.append(f"Attempt {i} - Answer validation failed: {', '.join(issues)}")
                logger.warning(f"Answer validation failed: {issues}")
                
        except Exception as e:
            error_msg = f"Attempt {i} failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        if i < max_retry:
            time.sleep(pause * i)  # Increasing wait time
    
    return "", False, errors

def append_to_file(item: dict, output_file: Path, logger: Logger):
    """
    Append new answer to file
    """
    try:
        # Read existing data
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        # Add new item
        data.append(item)
        
        # Write back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Answer saved to file (Total: {len(data)})")
        
    except Exception as e:
        logger.error(f"Failed to save answer: {e}")

def validate_csv_data(csv_path: Path, logger: Logger) -> bool:
    """Validate the integrity of CSV data"""
    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                logger.error("CSV file is empty")
                return False
            
            # Check required columns
            required_columns = ['question', 'prompt']
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            if missing_columns:
                logger.error(f"CSV missing required columns: {missing_columns}")
                return False
            
            # Check data integrity
            empty_questions = 0
            empty_prompts = 0
            
            for row in rows:
                if not row.get('question', '').strip():
                    empty_questions += 1
                if not row.get('prompt', '').strip():
                    empty_prompts += 1
            
            if empty_questions > 0:
                logger.warning(f"Found {empty_questions} empty questions")
            if empty_prompts > 0:
                logger.warning(f"Found {empty_prompts} empty prompts")
            
            logger.info(f"CSV data validation completed: {len(rows)} records")
            return True
            
    except Exception as e:
        logger.error(f"CSV validation failed: {e}")
        return False

# ========== 5. Main Execution Function =====================================================
def run_batch(model_cfg: dict, csv_path: Path):
    name = model_cfg["model_name"]
    logger = Logger(RUN_LOG)
    
    logger.info(f"Starting model run: {name}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    
    # Validate CSV data
    if not validate_csv_data(csv_path, logger):
        logger.error("CSV data validation failed, exiting run")
        return
    
    # Initialize API
    try:
        api = OpenAI(
            api_key=model_cfg["api_key"], 
            base_url=model_cfg["base_url"]
        )
    except Exception as e:
        logger.error(f"API initialization failed: {e}")
        return

    # --- Read CSV ---
    questions = []
    question_prompts = {}  # Directly store mapping of questions and prompts
    
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        logger.info(f"CSV column names: {reader.fieldnames}")
        
        for row in reader:
            q = row["question"]
            prompt = row["prompt"]
            question_prompts[q] = prompt
            questions.append(q)
    
    total_questions = len(questions)
    
    # Statistical information
    stats = {
        "Model": name,
        "Total questions": total_questions,
        "Skipped": 0,
        "Newly generated": 0,
        "Regenerated": 0,
        "Successful": 0,
        "Failed": 0,
        "API calls": 0
    }
    
    logger.info(f"=== 🚀 {name} ===")
    logger.info(f"Total questions: {stats['Total questions']}")
    
    # Error records
    failed_questions = []
    
    # Process each question
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{total_questions}] Checking question: {q[:60]}…")
        
        # Find existing answer
        existing_item = find_existing_answer(q, OUTPUT_FILE)
        
        if existing_item:
            # Check if answer is complete and valid
            is_valid, issues = is_answer_complete_and_valid(existing_item)
            
            if is_valid:
                print(f"  ✅ Skipped (valid answer exists)")
                stats['Skipped'] += 1
                continue
            else:
                print(f"  🔄 Needs regeneration (issues: {', '.join(issues)})")
                stats['Regenerated'] += 1
        else:
            print(f"  🆕 Generating new answer")
            stats['Newly generated'] += 1
        
        # Generate answer
        # direct/basic
        stats['API calls'] += 1
        direct_answer = ""
        
        # Try multiple times to get valid answer
        max_attempts = 5  # Maximum 5 attempts
        attempt = 0
        success = False
        
        while attempt < max_attempts and not success:
            attempt += 1
            if attempt > 1:
                logger.info(f"Attempt {attempt} to generate direct answer...")
            
            # Get direct answer
            direct_answer, success, errors = ask(
                api, name, q, logger,
                timeout=model_cfg.get('timeout', 60),
                max_retry=model_cfg.get('max_retry', 3)
            )
            
            if success:
                break
            else:
                logger.warning(f"Attempt {attempt} failed: {errors}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)  # Increasing wait time
        
        if not success:
            failed_questions.append({
                'question': q,
                'type': 'direct',
                'errors': errors,
                'attempts': attempt
            })
            stats['Failed'] += 1
            logger.error(f"Question '{q[:50]}...' still failed after {attempt} attempts")
            # Record even if failed for later processing
            direct_answer = f"[ERROR after {attempt} attempts]"
        
        item = {
            "question": q, 
            "direct_prompt": q, 
            "direct_reply": direct_answer,
            "timestamp": datetime.now().isoformat(),
            "attempts": attempt
        }
        
        # Process default prompt/reply
        if q in question_prompts:
            ptxt = question_prompts[q]
            print(f"  · Processing default prompt...")
            
            if ptxt:
                stats['API calls'] += 1
                
                #同样尝试多次
                default_attempt = 0
                default_success = False
                default_reply = ""
                
                while default_attempt < max_attempts and not default_success:
                    default_attempt += 1
                    if default_attempt > 1:
                        logger.info(f"default - Attempt {default_attempt}...")
                    
                    default_reply, default_success, errors = ask(
                        api, name, ptxt, logger,
                        timeout=model_cfg.get('timeout', 60),
                        max_retry=model_cfg.get('max_retry', 3)
                    )
                    
                    if default_success:
                        break
                    else:
                        if default_attempt < max_attempts:
                            time.sleep(5 * default_attempt)
                
                if not default_success:
                    failed_questions.append({
                        'question': q,
                        'type': 'default',
                        'errors': errors,
                        'attempts': default_attempt
                    })
                    stats['Failed'] += 1
                    default_reply = f"[ERROR after {default_attempt} attempts]"
            else:
                default_reply = ""
            
            item["default_prompt"] = ptxt
            item["default_reply"] = default_reply
        
        # Check if all responses were successfully obtained
        all_success = True
        for key in item:
            if key.endswith('_reply') and '[ERROR' in str(item.get(key, '')):
                all_success = False
                break
        
        if all_success:
            stats['Successful'] += 1
        
        # Append to file
        append_to_file(item, OUTPUT_FILE, logger)
        
        print(f"  ✅ Answer saved")
        
        # Display current statistics
        processed = stats['Skipped'] + stats['Newly generated'] + stats['Regenerated']
        success_rate = (stats['Successful'] / (stats['Newly generated'] + stats['Regenerated']) * 100) if (stats['Newly generated'] + stats['Regenerated']) > 0 else 100
        logger.info(f"Progress: {processed}/{total_questions} (Skipped: {stats['Skipped']}, Newly generated: {stats['Newly generated']}, Regenerated: {stats['Regenerated']}, Success rate: {success_rate:.1f}%)")

    # Save failure records
    if failed_questions:
        failed_file = OUTPUT_DIR_1 / f"failed_questions_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        failed_file.write_text(
            json.dumps(failed_questions, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logger.warning(f"Failure records saved to: {failed_file}")
    
    # Update statistics
    stats['Final success rate'] = f"{(stats['Successful'] / (stats['Newly generated'] + stats['Regenerated']) * 100):.1f}%" if (stats['Newly generated'] + stats['Regenerated']) > 0 else "100%"
    
    # Record final statistics
    logger.summary(stats)
    
    print(f"\n✅ {name} completed!")
    print(f"  · Total questions: {stats['Total questions']}")
    print(f"  · Skipped: {stats['Skipped']}")
    print(f"  · Newly generated: {stats['Newly generated']}")
    print(f"  · Regenerated: {stats['Regenerated']}")
    print(f"  · Successful: {stats['Successful']}")
    print(f"  · Failed: {stats['Failed']}")
    print(f"  · Success rate: {stats['Final success rate']}")

# ========== 6. Execution Loop =======================================================
print(f"📁 Output file: {OUTPUT_FILE}")
print(f"📄 Input CSV: {PROMPT_CSV}")
print(f"📝 Run log: {RUN_LOG}")
print("-" * 60)

for cfg in MODEL_CFGS:
    run_batch(cfg, PROMPT_CSV)

print(f"\n🎉 All completed!")
print(f"📁 Results saved in: {OUTPUT_FILE}")
print(f"📝 Run log: {RUN_LOG}")
