#!/usr/bin/env python
# coding: utf-8
"""
multmm2_run2plus2_topic_first_modified.py
----------------------------------------
Modified version: Extract third_answer from JSON format as A1, combination_1_reply as A2
Use new prompt template
Add answer quality check function
"""

import csv, json, time, re, traceback
from pathlib import Path
from openai import OpenAI
from datetime import datetime

# ===== 0. Output path and filename configuration ======================================================
BASE_DIR   = Path(r"D:\project7\MM\result\2+1")
BASE_DIR_1   = Path(r"D:\project7\prompt")
OUTPUT_DIR = Path(r"D:\project7\MM\result\2+1")  # You can easily modify the output path here

# ========== Output filename configuration - Modify here! ==========
# Method 1: Use a fixed suffix
OUTPUT_SUFFIX = "answers_2+1-2-7800-8100"  # Modify this value to change the output filename

# Method 2: Use timestamp
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# OUTPUT_SUFFIX = f"answers_{timestamp}"

# Method 3: Use fully custom name
# OUTPUT_FILENAME_TEMPLATE = "{model_name}_fusion_result_v2.json"  # {model_name} will be replaced with the model name

# Method 4: Specify output filename for each model individually (add output_filename field in MODEL_CFGS)
# ================================================

# Read JSON file containing combination_1_reply and third_answer
ANSWER_FILE = BASE_DIR / "gemini-2.5-flash_answers_2+1-1-7800-8100.json"  # File containing questions, combination_1_reply and third_answer
PROMPT_FILE = BASE_DIR_1 / "prompt-2+1-2.txt"  # Prompt template file
SAVE_INTERVAL = 1  # Save every N questions

# ===== 1. Model account configuration =======================================================
MODEL_CFGS = [
    {
        "model_name": "doubao-pro-32k",
        "api_key": "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa",
        "base_url": "https://vir.vimsai.com/v1",
        # "output_filename": "doubao_custom_output.json"  # Optional: Specify custom filename for specific model
    }
]

# ===== 2. Answer quality check configuration ===================================================
MIN_ANSWER_LENGTH = 10  # Minimum answer length
VALID_ENDINGS = ['。', '！', '？', '.', '!', '?', ')', '）', '"', '"', "'", "'"]  # Valid ending punctuation marks
MAX_RETRIES = 3  # Maximum number of retries

def check_answer_quality(answer: str, question: str = ""):
    """
    Check answer quality
    Returns: (is_valid, error_message)
    """
    if not answer or not answer.strip():
        return False, "Answer is empty"
    
    answer = answer.strip()
    
    # Check length
    if len(answer) < MIN_ANSWER_LENGTH:
        return False, f"Answer is too short ({len(answer)} characters, minimum {MIN_ANSWER_LENGTH} required)"
    
    # Check if ends with appropriate punctuation
    if not any(answer.endswith(ending) for ending in VALID_ENDINGS):
        return False, f"Answer may be truncated, ending character: '{answer[-1] if answer else 'N/A'}'"
    
    # Check for obvious truncation signs
    truncation_signs = ['...', '……', '[未完成]', '[截断]', '(未完', '（未完']
    if any(sign in answer for sign in truncation_signs):
        return False, "Answer contains truncation signs"
    
    # Check if answer is overly repetitive (may indicate generation anomaly)
    words = answer.split()
    if len(words) > 5:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        max_freq = max(word_freq.values())
        if max_freq > len(words) * 0.5:  # If a word appears more than 50% of the time
            return False, "Answer content is overly repetitive"
    
    return True, "Quality check passed"

# ===== 3. Read prompt template ====================================================
def load_prompt_template(template_path: Path):
    """Read prompt template file"""
    try:
        with template_path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ Failed to read prompt template: {e}")
        # If reading fails, use default template
        return """Read the summary answer A2 from the other two models and improve your previous answer A1. If there are still gaps, you can supplement with your common knowledge or public information, and indicate the source in parentheses (common knowledge / public information).
Question:
{q}
A1:
{A1}
A2:
{A2}
【Task Description】
Do not show the intermediate extraction process. Do not include any introductory statements at the beginning.
【Output Requirements】
- Clear and organized, may use numbering or paragraphing;
- Avoid redundancy, keep concise."""

# Load prompt template
PROMPT_TEMPLATE = load_prompt_template(PROMPT_FILE)

# ===== 4. Auxiliary functions ==========================================================

def get_output_filename(model_name: str, cfg: dict):
    """Generate output filename based on configuration"""
    # Prioritize custom filename in model configuration
    if "output_filename" in cfg:
        return cfg["output_filename"]
    
    # Use global template (if defined)
    if 'OUTPUT_FILENAME_TEMPLATE' in globals():
        return OUTPUT_FILENAME_TEMPLATE.format(model_name=model_name)
    
    # Default to suffix method
    return f"{model_name}_{OUTPUT_SUFFIX}.json"

def ask(api: OpenAI, model: str, prompt: str, retry: int = 3, pause: int = 2, question: str = ""):
    """Call API with quality check"""
    for i in range(retry):
        try:
            rsp = api.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            txt = rsp.choices[0].message.content.strip()
            
            # Perform quality check
            is_valid, error_msg = check_answer_quality(txt, question)
            
            if is_valid:
                print(f"    ✅ Answer quality check passed (length: {len(txt)} characters)")
                return txt
            else:
                print(f"    ⚠️ Quality check failed on attempt {i+1}: {error_msg}")
                if i < retry - 1:  # If not the last attempt
                    print(f"    🔄 Retrying...")
                    time.sleep(pause)
                    continue
                else:
                    print(f"    ❌ Reached maximum retries, returning current answer")
                    return txt  # Return even if quality is poor to avoid complete failure
                    
        except Exception as e:
            print(f"❌ {model} API call failed on attempt {i+1}: {e}")
            if i < retry - 1:
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
        print(f"⚠️ Failed to read progress: {e}")
        return {}

def save_progress(done_dict: dict, file: Path):
    """Save progress, only save records in done dictionary"""
    try:
        rows = list(done_dict.values())
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(file)
        print(f"💾 Saved {file.name} ({len(rows)} entries)")
    except Exception as e:
        print(f"❌ Save failed: {e}")

def load_answer_data(answer_path: Path):
    """Read JSON file containing questions, combination_1_reply and third_answer"""
    q2data = {}
    
    # Check if file exists
    if not answer_path.exists():
        print(f"❌ File does not exist: {answer_path}")
        return q2data
        
    with answer_path.open("r", encoding="utf-8") as f:
        # Determine if file is JSON array or JSON lines
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON array
            entries = json.load(f)
        else:
            # JSON lines
            entries = [json.loads(line) for line in f if line.strip()]
    
    print(f"📖 Read {len(entries)} records")
    
    for i, entry in enumerate(entries):
        try:
            q = entry["question"]
            combination_reply = entry.get("combination_1_reply", "")
            third_answer = entry.get("third_answer", "")
            third_model = entry.get("third_model", "")
            
            # Debug information
            if i == 0:  # Only print field information for the first record
                print(f"🔍 Fields of first record: {list(entry.keys())}")
                if "third_answer" in entry:
                    print(f"   ✓ Found third_answer (length: {len(third_answer)})")
                else:
                    print(f"   ❌ third_answer not found")
                if "combination_1_reply" in entry:
                    print(f"   ✓ Found combination_1_reply (length: {len(combination_reply)})")
                else:
                    print(f"   ❌ combination_1_reply not found")
            
            # Only add if both answers exist
            if combination_reply and third_answer:
                q2data[q] = {
                    "combination_reply": combination_reply,  # A2
                    "third_answer": third_answer,  # A1
                    "third_model": third_model
                }
            else:
                if i < 3:  # Only print warning for first few entries
                    print(f"⚠️ Record {i+1} missing required fields: combination_reply={bool(combination_reply)}, third_answer={bool(third_answer)}")
                    
        except KeyError as e:
            print(f"⚠️ Record {i+1} missing field {e}: {list(entry.keys())}")
            continue
    
    print(f"✅ Successfully loaded data for {len(q2data)} questions")
    return q2data

# ===== 5. Script entry point ==========================================================
if __name__ == "__main__":
    # 0) Check if template file exists
    if not PROMPT_FILE.exists():
        print(f"⚠️ Prompt template file does not exist: {PROMPT_FILE}")
        print("📝 Please create prompt-2+1-2.txt file containing {q}, {A1} and {A2} placeholders")
    else:
        print(f"✅ Loaded prompt template: {PROMPT_FILE}")
        print(f"📋 Answer quality check configuration: Minimum length={MIN_ANSWER_LENGTH}, Maximum retries={MAX_RETRIES}")
    
    # 1) Read answer data
    q2data = load_answer_data(ANSWER_FILE)
    all_questions = sorted(q2data.keys())
    print(f"📚 Number of questions: {len(all_questions)}")
    
    # 2) Prepare API, progress files, and caches for each model
    model_env = {}
    for cfg in MODEL_CFGS:
        name = cfg["model_name"]
        output_filename = get_output_filename(name, cfg)
        output_path = OUTPUT_DIR / output_filename
        
        model_env[name] = {
            "api": OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]),
            "out": output_path,
            "done": load_progress(output_path),
        }
        # Print existing progress
        existing_count = len(model_env[name]["done"])
        if existing_count > 0:
            print(f"📊 {name} existing progress: {existing_count} questions")
        print(f"📄 {name} output file: {output_filename}")
    
    processed = 0
    skipped = 0
    quality_failures = 0
    
    # ------- Main loop: Question priority -----------------
    for qi, q in enumerate(all_questions, 1):
        print(f"\n📝 [{qi}/{len(all_questions)}] {q[:60]}…")
        
        # Get data for this question
        data = q2data[q]
        a1 = data["third_answer"]  # third_answer as A1
        a2 = data["combination_reply"]  # combination_1_reply as A2
        source_model = data["third_model"]
        
        question_processed = False
        
        for cfg in MODEL_CFGS:
            mname = cfg["model_name"]
            env = model_env[mname]
            
            # Skip if already processed
            if q in env["done"]:
                print(f"  ⏭️ {mname} already processed, skipping")
                skipped += 1
                continue
            
            api = env["api"]
            
            print(f"  🤖 Calling {mname}")
            
            # Build prompt using new template
            prompt = PROMPT_TEMPLATE.format(q=q, A1=a1, A2=a2)
            
            # Call model (includes quality check)
            reply = ask(api, mname, prompt, question=q)
            
            # Record quality check result
            if reply:
                is_valid, quality_msg = check_answer_quality(reply, q)
                if not is_valid:
                    quality_failures += 1
                    print(f"    ⚠️ Final answer quality issue: {quality_msg}")
            
            # Save result
            item = {
                "question": q,
                "third_model": source_model,
                "A1_third_answer": a1,
                "A2_combination_reply": a2,
                "fusion_prompt": prompt,
                "fusion_reply": reply,
                "quality_check": check_answer_quality(reply, q)[1] if reply else "Generation failed"
            }
            
            # Add directly to done dictionary, not using rows
            env["done"][q] = item
            question_processed = True
        
        if question_processed:
            processed += 1
            
        # ---- SAVE_INTERVAL ----
        if processed > 0 and processed % SAVE_INTERVAL == 0:
            print(f"\n💾 Reached save interval, saving progress...")
            for mname, env in model_env.items():
                save_progress(env["done"], env["out"])
    
    # 3) Save once after all completion
    print(f"\n🏁 Processing complete!")
    print(f"📊 Statistics:")
    print(f"   - Newly processed: {processed} questions")
    print(f"   - Skipped: {skipped} questions") 
    print(f"   - Quality issues: {quality_failures} questions")
    
    for mname, env in model_env.items():
        save_progress(env["done"], env["out"])
        print(f"✅ {mname} total {len(env['done'])} records")
    

    print(f"\n🎉 All processed in question order, files saved to: {OUTPUT_DIR}")
