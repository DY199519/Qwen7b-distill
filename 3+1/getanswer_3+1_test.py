#!/usr/bin/env python
# coding: utf-8
"""
txt_qa_processor.py
-----------------
Reads JSON format question files, matches corresponding prompts in TXT format prompt files, 
executes model calls, and generates TXT format answer files
Each prompt is strictly separated by -------------------
Output format: Question: XXXX Reply: XXXXX
"""

import time
import json
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# ========== 0. Path Configuration =======================================================
# Control the number of questions to generate (None means generate all, number means generate only first N questions)
LIMIT_QUESTIONS = 2  # Example: 10 means generate only first 10 questions, None means generate all

# Input and output file paths
JSON_FILE = Path(r"D:\qwensft\testquestion\multi_model_answersTest500.json")  # JSON question file path
TXT_FILE = Path(r"D:\qwensft\uploadjson\final_prompt_3+1-Test.txt")  # TXT prompt file path
OUTPUT_FILE = Path(r"D:\qwensft\uploadjson\final_answer_3+1-Test.txt")  # Modify to your output file path

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Run log file
RUN_LOG = OUTPUT_FILE.parent / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ========== 1. Model Configuration =======================================================
MODEL_CFG = {
    "model_name": "deepseek-v3",
    "api_key": "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa",
    "base_url": "https://usa.vimsai.com/v1",
    "timeout": 60,  # Timeout in seconds
    "max_retry": 3,  # Maximum number of retries
}

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
    def validate_answer(cls, answer: str) -> Tuple[bool, List[str]]:
        """Validate if the answer is valid"""
        issues = []
        
        if not answer:
            issues.append("Answer is empty")
            return False, issues
        
        if not isinstance(answer, str):
            issues.append(f"Invalid answer type: {type(answer)}")
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
def read_json_questions(file_path: Path, logger: Logger) -> List[str]:
    """
    Read question list from JSON file
    Extract questions in order from the questions field
    """
    if not file_path.exists():
        logger.error(f"JSON file does not exist: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions from JSON
        questions = []
        if 'questions' in data:
            for question_key in data['questions'].keys():
                questions.append(question_key)
        
        logger.info(f"Successfully read {len(questions)} questions")
        return questions
        
    except Exception as e:
        logger.error(f"Failed to read JSON file: {e}")
        return []

def read_txt_prompts(file_path: Path, logger: Logger) -> Dict[str, str]:
    """
    Read prompts from TXT file and create a mapping from questions to prompts
    Each prompt is separated by -------------------
    """
    if not file_path.exists():
        logger.error(f"TXT file does not exist: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split using -------------------
        prompts = content.split('-------------------')
        
        # Create question to prompt mapping
        question_to_prompt = {}
        for prompt in prompts:
            prompt = prompt.strip()
            if prompt:
                # Extract question from prompt
                question = extract_question_from_prompt(prompt)
                if question:
                    question_to_prompt[question] = prompt
        
        logger.info(f"Successfully read {len(question_to_prompt)} prompts")
        return question_to_prompt
        
    except Exception as e:
        logger.error(f"Failed to read TXT file: {e}")
        return {}

def extract_question_from_prompt(prompt: str) -> str:
    """
    Extract question part from prompt
    Assume question is in the first line or line containing "问题" (question) keyword
    """
    lines = prompt.split('\n')
    
    # Look for line containing "问题" (question)
    for line in lines:
        if '问题' in line and '"' in line:
            # Extract content within quotes
            match = re.search(r'"([^"]+)"', line)
            if match:
                return match.group(1)
    
    # If not found, return first 100 characters as identifier
    return prompt[:100] if len(prompt) > 100 else prompt

def find_matching_prompt(question: str, question_to_prompt: Dict[str, str], logger: Logger) -> Optional[str]:
    """
    Find matching prompt in prompt mapping
    """
    # Direct match
    if question in question_to_prompt:
        return question_to_prompt[question]
    
    # Fuzzy match (remove spaces and punctuation)
    normalized_question = re.sub(r'[^\w]', '', question)
    for prompt_question, prompt in question_to_prompt.items():
        normalized_prompt_question = re.sub(r'[^\w]', '', prompt_question)
        if normalized_question == normalized_prompt_question:
            logger.info(f"Fuzzy match successful: {question[:50]}...")
            return prompt
    
    # Partial match (inclusion relationship)
    for prompt_question, prompt in question_to_prompt.items():
        if question in prompt_question or prompt_question in question:
            logger.info(f"Partial match successful: {question[:50]}...")
            return prompt
    
    logger.warning(f"No matching prompt found: {question[:50]}...")
    return None

def ask(api: OpenAI, model: str, prompt: str, logger: Logger, 
        timeout: int = 60, max_retry: int = 3, pause: float = 2.0) -> Tuple[str, bool, List[str]]:
    """
    Call model API
    Returns: (answer, success status, error list)
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
            is_valid, issues = AnswerValidator.validate_answer(answer)
            
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

def append_to_output(question: str, answer: str, output_file: Path, logger: Logger):
    """
    Append question-answer pair to output file
    Format: Question: XXXX  Reply:XXXXX
    """
    try:
        # Construct output content
        output_content = f"Question: {question}\nReference Reply: {answer}\n"
        
        # Add separator if file exists and has content
        if output_file.exists() and output_file.stat().st_size > 0:
            output_content = "-------------------\n" + output_content
        
        # Append to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(output_content)
        
        logger.info(f"Answer saved to file")
        
    except Exception as e:
        logger.error(f"Failed to save answer: {e}")

def check_existing_answers(output_file: Path, logger: Logger) -> set:
    """
    Check existing questions in output file
    Returns set of answered questions
    """
    existing_questions = set()
    
    if not output_file.exists():
        return existing_questions
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split existing answers
        answers = content.split('-------------------')
        
        for answer in answers:
            if 'Question: ' in answer:
                # Extract question
                match = re.search(r'Question: (.+?)(?:\n|$)', answer)
                if match:
                    question = match.group(1).strip()
                    existing_questions.add(question)
        
        logger.info(f"Found {len(existing_questions)} existing answers")
        
    except Exception as e:
        logger.error(f"Failed to read existing answers: {e}")
    
    return existing_questions

# ========== 5. Main Execution Function =====================================================
def run_txt_processing():
    """Main processing function"""
    logger = Logger(RUN_LOG)
    
    logger.info(f"Starting processing")
    logger.info(f"JSON question file: {JSON_FILE}")
    logger.info(f"TXT prompt file: {TXT_FILE}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info(f"Model: {MODEL_CFG['model_name']}")
    
    # Show question limit setting
    if LIMIT_QUESTIONS is not None:
        logger.info(f"⚠️ Generation limit set: only process first {LIMIT_QUESTIONS} questions")
    else:
        logger.info(f"Processing all questions (no limit)")
    
    # Read JSON questions
    questions = read_json_questions(JSON_FILE, logger)
    if not questions:
        logger.error("No questions read")
        return
    
    # Read TXT prompts
    question_to_prompt = read_txt_prompts(TXT_FILE, logger)
    if not question_to_prompt:
        logger.error("No prompts read")
        return
    
    # Apply question count limit
    original_count = len(questions)
    if LIMIT_QUESTIONS is not None and LIMIT_QUESTIONS > 0:
        questions = questions[:LIMIT_QUESTIONS]
        logger.info(f"Applying limit: selecting first {len(questions)} questions from {original_count}")
    
    # Check existing answers
    existing_questions = check_existing_answers(OUTPUT_FILE, logger)
    
    # Initialize API
    try:
        api = OpenAI(
            api_key=MODEL_CFG["api_key"], 
            base_url=MODEL_CFG["base_url"]
        )
        logger.info("API initialized successfully")
    except Exception as e:
        logger.error(f"API initialization failed: {e}")
        return
    
    # Statistical information
    stats = {
        "Model": MODEL_CFG["model_name"],
        "Original question count": original_count,
        "Processed question count": len(questions),
        "Question limit": LIMIT_QUESTIONS if LIMIT_QUESTIONS else "No limit",
        "Skipped count": 0,
        "Success count": 0,
        "Failure count": 0,
        "Unmatched count": 0,
        "API call count": 0
    }
    
    failed_prompts = []
    unmatched_questions = []
    
    # Process each question
    for idx, question in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] Processing question: {question[:60]}...")
        
        # Check if already exists
        if question in existing_questions:
            print(f"  ✅ Skipped (answer already exists)")
            stats['Skipped count'] += 1
            continue
        
        # Find matching prompt
        prompt = find_matching_prompt(question, question_to_prompt, logger)
        if not prompt:
            print(f"  ❌ No matching prompt found")
            stats['Unmatched count'] += 1
            unmatched_questions.append(question)
            continue
        
        # Call API to get answer
        stats['API call count'] += 1
        
        max_attempts = 5
        attempt = 0
        success = False
        answer = ""
        
        while attempt < max_attempts and not success:
            attempt += 1
            if attempt > 1:
                logger.info(f"Attempt {attempt}...")
            
            answer, success, errors = ask(
                api, 
                MODEL_CFG["model_name"], 
                prompt, 
                logger,
                timeout=MODEL_CFG.get('timeout', 60),
                max_retry=MODEL_CFG.get('max_retry', 3)
            )
            
            if success:
                break
            else:
                logger.warning(f"Attempt {attempt} failed: {errors}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)
        
        if success:
            stats['Success count'] += 1
            # Save to file
            append_to_output(question, answer, OUTPUT_FILE, logger)
            print(f"  ✅ Successfully generated and saved answer")
        else:
            stats['Failure count'] += 1
            failed_prompts.append({
                'question': question,
                'prompt': prompt[:200],
                'errors': errors,
                'attempts': attempt
            })
            logger.error(f"Question '{question[:50]}...' failed after {attempt} attempts")
            
            # Record even if failed
            append_to_output(question, f"[Generation failed: {errors[-1] if errors else 'Unknown error'}]", OUTPUT_FILE, logger)
        
        # Show progress
        processed = idx
        processed_count = processed - stats['Skipped count'] - stats['Unmatched count']
        success_rate = (stats['Success count'] / processed_count * 100) if processed_count > 0 else 100
        logger.info(f"Progress: {processed}/{len(questions)} (Success rate: {success_rate:.1f}%)")
        
        # Short delay to avoid too frequent requests
        time.sleep(1)
    
    # Save failed records
    if failed_prompts:
        failed_file = OUTPUT_FILE.parent / f"failed_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(failed_file, 'w', encoding='utf-8') as f:
            for item in failed_prompts:
                f.write(f"Question: {item['question']}\n")
                f.write(f"Number of attempts: {item['attempts']}\n")
                f.write(f"Errors: {item['errors']}\n")
                f.write("-------------------\n")
        logger.warning(f"Failed records saved to: {failed_file}")
    
    # Save unmatched questions
    if unmatched_questions:
        unmatched_file = OUTPUT_FILE.parent / f"unmatched_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(unmatched_file, 'w', encoding='utf-8') as f:
            for question in unmatched_questions:
                f.write(f"{question}\n")
        logger.warning(f"Unmatched questions saved to: {unmatched_file}")
    
    # Update statistics
    actual_processed = stats['Processed question count'] - stats['Skipped count'] - stats['Unmatched count']
    stats['Final success rate'] = f"{(stats['Success count'] / actual_processed * 100):.1f}%" if actual_processed > 0 else "100%"
    
    # Record final statistics
    logger.summary(stats)
    
    print(f"\n🎉 Processing completed!")
    print(f"  · Original total: {stats['Original question count']}")
    print(f"  · Processed count: {stats['Processed question count']} (Limit: {stats['Question limit']})")
    print(f"  · Skipped: {stats['Skipped count']}")
    print(f"  · Unmatched: {stats['Unmatched count']}")
    print(f"  · Success: {stats['Success count']}")
    print(f"  · Failure: {stats['Failure count']}")
    print(f"  · Success rate: {stats['Final success rate']}")

# ========== 6. Execution Entry =======================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TXT Format Q&A Processing Script (JSON Questions + TXT Prompt Matching)")
    print("=" * 60)
    
    # Check if input files exist
    if not JSON_FILE.exists():
        print(f"❌ JSON question file does not exist: {JSON_FILE}")
    elif not TXT_FILE.exists():
        print(f"❌ TXT prompt file does not exist: {TXT_FILE}")
    else:
        run_txt_processing()
        print(f"\n📁 Results saved in: {OUTPUT_FILE}")
        print(f"📝 Run log: {RUN_LOG}")
