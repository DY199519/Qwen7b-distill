#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Swift model batch inference script (supports checkpoint resumption)
Used to run Swift inference on AutoDL servers, process interdisciplinary questions, and save results
"""

import subprocess
import json
import time
import os
import sys
from datetime import datetime
import signal
import re
import hashlib

# Path to swift executable
SWIFT_PATH = "/root/miniconda3/bin/swift"

def get_question_hash(question):
    """
    Generate a unique identifier for the question
    """
    return hashlib.md5(question.encode('utf-8')).hexdigest()[:8]

def load_questions_from_file(file_path):
    """
    Load question list from file
    Supports two formats:
    1. Python list format (as in example)
    2. Plain text format (one question per line)
    """
    print(f"Debug: Attempting to access file: {file_path}")
    print(f"Debug: File exists: {os.path.exists(file_path)}")
    print(f"Debug: Current working directory: {os.getcwd()}")
    
    # Process file path
    if os.path.dirname(file_path):
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            print(f"Debug: Directory contents: {os.listdir(dir_path)}")
    
    questions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Debug: File length: {len(content)} chars")
        print(f"Debug: First 100 chars: {content[:100]}...")
        
        # Method 1: Try executing Python code (most reliable method)
        if 'high_quality_crossdisciplinary_questions' in content:
            try:
                # Create a local namespace to execute the code
                local_namespace = {}
                exec(content, {}, local_namespace)
                
                if 'high_quality_crossdisciplinary_questions' in local_namespace:
                    questions_data = local_namespace['high_quality_crossdisciplinary_questions']
                    # Extract the first item of each element (question text)
                    questions = [item[0] for item in questions_data if isinstance(item, list) and len(item) > 0]
                    print(f"✓ Successfully loaded {len(questions)} questions via exec method")
                    
                    # Verify question format and quantity
                    print(f"  Original list length: {len(questions_data)}")
                    print(f"  Extracted question count: {len(questions)}")
                    
                    # Display first 3 and last 3 questions for verification
                    if questions:
                        print("  First 3 questions:")
                        for i, q in enumerate(questions[:3], 1):
                            print(f"    {i}. {q[:60]}...")
                        if len(questions) > 3:
                            print("  Last 3 questions:")
                            for i, q in enumerate(questions[-3:], len(questions)-2):
                                print(f"    {i}. {q[:60]}...")
                    
                    # Check if all questions end with a question mark
                    non_question_count = sum(1 for q in questions if not (q.endswith('?') or q.endswith('？')))
                    if non_question_count > 0:
                        print(f"  Warning: {non_question_count} questions do not end with a question mark")
                    
                    return questions
            except Exception as e:
                print(f"exec method failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Method 2: If exec fails, do not use regular expressions (as they easily lose data)
        # Instead, parse the Python list manually
        if not questions and 'high_quality_crossdisciplinary_questions = [' in content:
            try:
                # Find the start and end of the list
                start_idx = content.find('high_quality_crossdisciplinary_questions = [')
                if start_idx != -1:
                    # Find list content from start position
                    list_start = content.find('[', start_idx) + 1
                    
                    # Calculate bracket balance to find list end
                    bracket_count = 1
                    idx = list_start
                    while bracket_count > 0 and idx < len(content):
                        if content[idx] == '[':
                            bracket_count += 1
                        elif content[idx] == ']':
                            bracket_count -= 1
                        idx += 1
                    
                    list_content = content[list_start:idx-1]
                    
                    # Use ast.literal_eval for safer parsing
                    import ast
                    try:
                        # Construct complete Python expression
                        full_list = ast.literal_eval('[' + list_content + ']')
                        questions = [item[0] for item in full_list if isinstance(item, list) and len(item) > 0]
                        print(f"✓ Successfully loaded {len(questions)} questions via ast.literal_eval")
                    except:
                        print("ast.literal_eval method also failed")
            except Exception as e:
                print(f"Manual parsing method failed: {e}")
        
        # Method 3: Plain text format (one question per line)
        if not questions:
            lines = content.strip().split('\n')
            # Filter out empty lines and comment lines
            potential_questions = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('high_quality'):
                    # Check if it looks like a question (contains Chinese and ends with a question mark)
                    if '？' in line or '?' in line:
                        potential_questions.append(line)
            
            if potential_questions:
                questions = potential_questions
                print(f"Loaded {len(questions)} questions via plain text method")
        
        if not questions:
            print("❌ Warning: Failed to parse any questions")
        elif len(questions) < 200:
            print(f"⚠ Warning: Only loaded {len(questions)} questions, expected 200")
        
        return questions
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Failed to load question file: {e}")
        import traceback
        traceback.print_exc()
        return []

def validate_and_fix_question(question, input_file, question_index):
    """
    Verify if the question is complete (should end with a question mark), re-read if incomplete
    """
    if question.endswith('?') or question.endswith('？'):
        return question
    
    print(f"  ⚠ Question incomplete, re-reading: {question[:50]}...")
    
    # Re-read complete question
    try:
        questions = load_questions_from_file(input_file)
        if 0 <= question_index < len(questions):
            full_question = questions[question_index]
            if full_question.endswith('?') or full_question.endswith('？'):
                print(f"  ✓ Successfully re-read complete question")
                return full_question
            else:
                print(f"  ⚠ Re-read question still incomplete")
                return full_question
    except Exception as e:
        print(f"  ✗ Failed to re-read question: {e}")
    
    return question

def load_existing_results(output_file):
    """
    Load existing results, return set of processed questions
    """
    processed_questions = {}
    json_file = output_file.replace('.txt', '.json')
    
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                for result in results:
                    question_hash = get_question_hash(result['question'])
                    processed_questions[question_hash] = result
                print(f"Loaded {len(processed_questions)} processed questions from cache")
        except Exception as e:
            print(f"Error loading cache file: {e}")
    
    return processed_questions

def save_single_result(result, output_file, all_results):
    """
    Save a single result (append mode)
    """
    try:
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Swift Model Inference Results\n")
            f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Questions: {len(all_results)}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, res in enumerate(all_results, 1):
                f.write(f"Question {idx}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Question: {res['question']}\n\n")
                f.write(f"Answer:\n{res['answer']}\n")
                f.write(f"\nTimestamp: {res['timestamp']}\n")
                f.write("=" * 80 + "\n\n")
        
        print(f"  ✓ Results saved (Total: {len(all_results)})")
        
    except Exception as e:
        print(f"  ✗ Error saving results: {e}")

def extract_clean_answer(output):
    """
    Extract clean answer from output, separate command line output
    Returns: (clean_answer, command_output)
    """
    lines = output.split('\n')
    answer_lines = []
    command_lines = []
    start_collecting = False
    
    for line in lines:
        # Detect command line output
        if line.startswith('run sh:') or \
           line.startswith('Downloading Model') or \
           '/root/miniconda3' in line or \
           'swift/cli/infer.py' in line or \
           line.startswith('[') or \
           'swift' in line.lower():
            command_lines.append(line)
            continue
        
        # Detect answer start marker
        if '<<<' in line:
            start_collecting = True
            continue
        
        # Detect answer end or invalid content
        if line == "exit" or line == "--------------------------------------------------":
            continue
        
        # Collect answer content
        if start_collecting or ("assistant:" in line.lower()):
            if "assistant:" in line.lower():
                start_collecting = True
                continue
            answer_lines.append(line)
    
    clean_answer = '\n'.join(answer_lines).strip()
    command_output = '\n'.join(command_lines).strip()
    
    # If answer is empty, try another parsing method
    if not clean_answer:
        # Find content between <<< and next <<<
        match = re.search(r'<<<\s*(.*?)(?:<<<|$)', output, re.DOTALL)
        if match:
            clean_answer = match.group(1).strip()
            # Remove trailing separator
            clean_answer = clean_answer.replace('--------------------------------------------------', '').strip()
    
    return clean_answer, command_output

def run_single_inference(question, model_path, adapters_path, timeout=120):
    """
    Run inference for a single question
    """
    cmd = f"""(echo "{question.replace('"', '\\"')}"; echo "exit") | CUDA_VISIBLE_DEVICES=0 {SWIFT_PATH} infer \
        --model {model_path} \
        --adapters {adapters_path} \
        --max_new_tokens 2048 \
        --temperature 0.7 \
        --top_p 0.9 \
        --stream true"""
    
    try:
        print(f"  Executing command: {SWIFT_PATH} infer ...")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8'
        )
        
        output = result.stdout
        if result.returncode != 0:
            print(f"  ✗ Inference error: {result.stderr}")
            return None, f"Error: {result.stderr}"
        
        # Extract clean answer and command output
        clean_answer, command_output = extract_clean_answer(output)
        
        return clean_answer, command_output
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ Inference timed out ({timeout} seconds)")
        return None, "Inference timed out"
    except Exception as e:
        print(f"  ✗ Inference error: {e}")
        return None, f"Error: {str(e)}"

def run_batch_inference_with_checkpoint(questions, model_path, adapters_path, output_file, input_file):
    """
    Batch inference with checkpoint resumption support
    """
    processed = load_existing_results(output_file)
    all_results = list(processed.values())
    
    total_questions = len(questions)
    already_processed = 0
    to_process = []
    
    for i, question in enumerate(questions):
        question_hash = get_question_hash(question)
        if question_hash in processed:
            already_processed += 1
            print(f"Question {i+1}: Already processed, skipping")
        else:
            to_process.append((i, question))
    
    print("\n" + "=" * 50)
    print(f"Total questions: {total_questions}")
    print(f"Already processed: {already_processed}")
    print(f"Remaining to process: {len(to_process)}")
    print("=" * 50 + "\n")
    
    if not to_process:
        print("All questions have been processed!")
        return all_results
    
    for idx, (question_index, question) in enumerate(to_process, 1):
        question_num = question_index + 1
        print(f"\nProcessing question {question_num}/{total_questions} (remaining {idx}/{len(to_process)}):")
        
        # Validate and fix question
        question = validate_and_fix_question(question, input_file, question_index)
        print(f"  Question: {question[:100]}...")
        
        start_time = time.time()
        
        # Maximum 3 retries
        max_retries = 3
        retry_count = 0
        answer = None
        command_output = ""
        
        while retry_count < max_retries:
            answer, command_output = run_single_inference(question, model_path, adapters_path)
            
            # Check answer validity
            if answer and len(answer) >= 20:
                print(f"  ✓ Valid answer (length: {len(answer)} characters)")
                break
            else:
                retry_count += 1
                if answer:
                    print(f"  ⚠ Answer too short ({len(answer)} characters), retrying {retry_count}/{max_retries}")
                else:
                    print(f"  ⚠ Answer generation failed, retrying {retry_count}/{max_retries}")
                
                if retry_count < max_retries:
                    time.sleep(3)  # Wait 3 seconds before retrying
        
        # If still failed after retries, use error message
        if not answer or len(answer) < 20:
            answer = f"Inference failed: Answer incomplete or too short (failed after {max_retries} retries)"
            print(f"  ✗ Final failure: Unable to generate valid answer")
        
        elapsed_time = time.time() - start_time
        
        result = {
            'question_id': question_num,
            'question': question,
            'question_hash': get_question_hash(question),
            'answer': answer,
            'command_output': command_output,  # Store command line output separately
            'timestamp': datetime.now().isoformat(),
            'processing_time': f"{elapsed_time:.2f} seconds",
            'retry_count': retry_count  # Record number of retries
        }
        
        all_results.append(result)
        processed[result['question_hash']] = result
        
        save_single_result(result, output_file, all_results)
        
        print(f"  ✓ Completed (time taken: {elapsed_time:.2f} seconds, retries: {retry_count})")
        print(f"  Progress: {already_processed + idx}/{total_questions} ({(already_processed + idx)*100/total_questions:.1f}%)")
        
        if idx < len(to_process):
            time.sleep(2)
    
    return all_results

def check_swift_installation():
    """
    Check if Swift is properly installed
    """
    try:
        print(f"Debug: Checking swift at {SWIFT_PATH}")
        print(f"Debug: Python executable: {sys.executable}")
        print(f"Debug: PATH: {os.environ.get('PATH', 'Not set')}")
        result = subprocess.run(
            ["/root/miniconda3/bin/pip", "show", "ms-swift"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = re.search(r"Version: (.+)", result.stdout)
            if version:
                print(f"✓ Swift installed: ms-swift {version.group(1)}")
                return True
            else:
                print("✗ Invalid ms-swift installation information")
                return False
        else:
            print(f"✗ ms-swift not installed: {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"✗ pip command not found: /root/miniconda3/bin/pip")
        return False
    except Exception as e:
        print(f"✗ Swift check failed: {e}")
        return False

def main():
    """
    Main function
    """
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    adapters_path = "/root/autodl-tmp/output/sft3+1/checkpoint-1500"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "/root/200.txt"
    
    output_file = "/root/inference_results-sft-3+1.txt"
    
    print("=" * 50)
    print("Swift Batch Inference Script (with checkpoint resumption)")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Adapters: {adapters_path}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 50)
    
    if not check_swift_installation():
        print("\nPlease install Swift first:")
        print(f"/root/miniconda3/bin/pip install ms-swift -U")
        return
    
    cuda_available = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_available:
        print(f"✓ CUDA device: {cuda_available}")
    else:
        print("⚠ CUDA_VISIBLE_DEVICES not set, will use default GPU 0")
    
    questions = load_questions_from_file(input_file)
    
    if not questions:
        print("No valid questions found, exiting program")
        return
    
    print(f"\nPreparing to process {len(questions)} questions...")
    
    print("Starting processing (non-interactive mode, will continue automatically)")
    
    results = run_batch_inference_with_checkpoint(
        questions, 
        model_path, 
        adapters_path, 
        output_file,
        input_file  # Pass input file path for re-reading
    )
    
    print("\n" + "=" * 50)
    print(f"Inference completed!")
    print(f"Total processed questions: {len(results)}")
    print(f"Success: {len([r for r in results if 'error' not in r['answer'] and 'timeout' not in r['answer'] and 'failed' not in r['answer']])}")
    print(f"Failed: {len([r for r in results if 'error' in r['answer'] or 'timeout' in r['answer'] or 'failed' in r['answer']])}")
    print(f"Result file: {output_file}")
    print(f"JSON file: {output_file.replace('.txt', '.json')}")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUser interrupted program")
        print("Progress has been saved, next run will resume from checkpoint")
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
