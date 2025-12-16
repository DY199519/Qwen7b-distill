#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
grpo模型批量推理脚本（支持断点续传）
用于在AutoDL服务器上运行grpo推理，处理跨学科问题并保存结果
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
    生成问题的唯一标识符
    """
    return hashlib.md5(question.encode('utf-8')).hexdigest()[:8]

def load_questions_from_file(file_path):
    """
    从文件中加载问题列表
    支持两种格式：
    1. Python列表格式（如示例）
    2. 纯文本格式（每行一个问题）
    """
    print(f"Debug: Attempting to access file: {file_path}")
    print(f"Debug: File exists: {os.path.exists(file_path)}")
    print(f"Debug: Current working directory: {os.getcwd()}")
    
    # 处理文件路径
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
        
        # 方法1: 尝试执行Python代码（最可靠的方法）
        if 'high_quality_crossdisciplinary_questions' in content:
            try:
                # 创建一个局部命名空间来执行代码
                local_namespace = {}
                exec(content, {}, local_namespace)
                
                if 'high_quality_crossdisciplinary_questions' in local_namespace:
                    questions_data = local_namespace['high_quality_crossdisciplinary_questions']
                    # 提取每个元素的第一项（问题文本）
                    questions = [item[0] for item in questions_data if isinstance(item, list) and len(item) > 0]
                    print(f"✓ 成功通过exec方法加载 {len(questions)} 个问题")
                    
                    # 验证问题格式和数量
                    print(f"  原始列表长度: {len(questions_data)}")
                    print(f"  提取的问题数: {len(questions)}")
                    
                    # 显示前3个和后3个问题作为验证
                    if questions:
                        print("  前3个问题:")
                        for i, q in enumerate(questions[:3], 1):
                            print(f"    {i}. {q[:60]}...")
                        if len(questions) > 3:
                            print("  后3个问题:")
                            for i, q in enumerate(questions[-3:], len(questions)-2):
                                print(f"    {i}. {q[:60]}...")
                    
                    # 检查是否所有问题都以问号结尾
                    non_question_count = sum(1 for q in questions if not (q.endswith('?') or q.endswith('？')))
                    if non_question_count > 0:
                        print(f"  警告: {non_question_count} 个问题不以问号结尾")
                    
                    return questions
            except Exception as e:
                print(f"exec方法失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 方法2: 如果exec失败，不使用正则表达式（因为容易丢失数据）
        # 而是手动解析Python列表
        if not questions and 'high_quality_crossdisciplinary_questions = [' in content:
            try:
                # 找到列表的开始和结束
                start_idx = content.find('high_quality_crossdisciplinary_questions = [')
                if start_idx != -1:
                    # 从开始位置找到列表内容
                    list_start = content.find('[', start_idx) + 1
                    
                    # 计算括号平衡来找到列表结束
                    bracket_count = 1
                    idx = list_start
                    while bracket_count > 0 and idx < len(content):
                        if content[idx] == '[':
                            bracket_count += 1
                        elif content[idx] == ']':
                            bracket_count -= 1
                        idx += 1
                    
                    list_content = content[list_start:idx-1]
                    
                    # 使用ast.literal_eval更安全地解析
                    import ast
                    try:
                        # 构造完整的Python表达式
                        full_list = ast.literal_eval('[' + list_content + ']')
                        questions = [item[0] for item in full_list if isinstance(item, list) and len(item) > 0]
                        print(f"✓ 成功通过ast.literal_eval加载 {len(questions)} 个问题")
                    except:
                        print("ast.literal_eval方法也失败")
            except Exception as e:
                print(f"手动解析方法失败: {e}")
        
        # 方法3: 纯文本格式（每行一个问题）
        if not questions:
            lines = content.strip().split('\n')
            # 过滤掉空行和注释行
            potential_questions = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('high_quality'):
                    # 检查是否像是问题（包含中文并以问号结尾）
                    if '？' in line or '?' in line:
                        potential_questions.append(line)
            
            if potential_questions:
                questions = potential_questions
                print(f"通过纯文本方式加载 {len(questions)} 个问题")
        
        if not questions:
            print("❌ 警告：未能解析出任何问题")
        elif len(questions) < 200:
            print(f"⚠ 警告：只加载了 {len(questions)} 个问题，预期是200个")
        
        return questions
    
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return []
    except Exception as e:
        print(f"加载问题文件失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def validate_and_fix_question(question, input_file, question_index):
    """
    验证问题是否完整（应该以问号结尾），如果不完整则重新读取
    """
    if question.endswith('?') or question.endswith('？'):
        return question
    
    print(f"  ⚠ 问题不完整，重新读取: {question[:50]}...")
    
    # 重新读取完整问题
    try:
        questions = load_questions_from_file(input_file)
        if 0 <= question_index < len(questions):
            full_question = questions[question_index]
            if full_question.endswith('?') or full_question.endswith('？'):
                print(f"  ✓ 成功重新读取完整问题")
                return full_question
            else:
                print(f"  ⚠ 重新读取的问题仍不完整")
                return full_question
    except Exception as e:
        print(f"  ✗ 重新读取问题失败: {e}")
    
    return question

def load_existing_results(output_file):
    """
    加载已存在的结果，返回已处理的问题集合
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
                print(f"从缓存加载了 {len(processed_questions)} 个已处理的问题")
        except Exception as e:
            print(f"加载缓存文件时出错: {e}")
    
    return processed_questions

def save_single_result(result, output_file, all_results):
    """
    保存单个结果（追加模式）
    """
    try:
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Swift模型推理结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总问题数: {len(all_results)}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, res in enumerate(all_results, 1):
                f.write(f"问题 {idx}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"问题: {res['question']}\n\n")
                f.write(f"回答:\n{res['answer']}\n")
                f.write(f"\n时间戳: {res['timestamp']}\n")
                f.write("=" * 80 + "\n\n")
        
        print(f"  ✓ 结果已保存（总计: {len(all_results)} 个）")
        
    except Exception as e:
        print(f"  ✗ 保存结果时出错: {e}")

def extract_clean_answer(output):
    """
    从输出中提取干净的答案，分离命令行输出
    返回: (clean_answer, command_output)
    """
    lines = output.split('\n')
    answer_lines = []
    command_lines = []
    start_collecting = False
    
    for line in lines:
        # 检测命令行输出
        if line.startswith('run sh:') or \
           line.startswith('Downloading Model') or \
           '/root/miniconda3' in line or \
           'swift/cli/infer.py' in line or \
           line.startswith('[') or \
           'swift' in line.lower():
            command_lines.append(line)
            continue
        
        # 检测答案开始标记
        if '<<<' in line:
            start_collecting = True
            continue
        
        # 检测答案结束或无效内容
        if line == "exit" or line == "--------------------------------------------------":
            continue
        
        # 收集答案内容
        if start_collecting or ("assistant:" in line.lower()):
            if "assistant:" in line.lower():
                start_collecting = True
                continue
            answer_lines.append(line)
    
    clean_answer = '\n'.join(answer_lines).strip()
    command_output = '\n'.join(command_lines).strip()
    
    # 如果答案为空，尝试另一种解析方式
    if not clean_answer:
        # 查找 <<< 和下一个 <<< 之间的内容
        match = re.search(r'<<<\s*(.*?)(?:<<<|$)', output, re.DOTALL)
        if match:
            clean_answer = match.group(1).strip()
            # 移除结尾的分隔线
            clean_answer = clean_answer.replace('--------------------------------------------------', '').strip()
    
    return clean_answer, command_output

def run_single_inference(question, model_path, adapters_path, timeout=120):
    """
    运行单个问题的推理
    """
    cmd = f"""(echo "{question.replace('"', '\\"')}"; echo "exit") | CUDA_VISIBLE_DEVICES=0 {SWIFT_PATH} infer \
        --model {model_path} \
        --adapters {adapters_path} \
        --max_new_tokens 2048 \
        --temperature 0.7 \
        --top_p 0.9 \
        --stream true"""
    
    try:
        print(f"  执行命令: {SWIFT_PATH} infer ...")
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
            print(f"  ✗ 推理错误: {result.stderr}")
            return None, f"错误: {result.stderr}"
        
        # 提取干净的答案和命令行输出
        clean_answer, command_output = extract_clean_answer(output)
        
        return clean_answer, command_output
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ 推理超时（{timeout}秒）")
        return None, "推理超时"
    except Exception as e:
        print(f"  ✗ 推理错误: {e}")
        return None, f"错误: {str(e)}"

def run_batch_inference_with_checkpoint(questions, model_path, adapters_path, output_file, input_file):
    """
    批量推理，支持断点续传
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
            print(f"问题 {i+1}: 已处理，跳过")
        else:
            to_process.append((i, question))
    
    print("\n" + "=" * 50)
    print(f"总问题数: {total_questions}")
    print(f"已处理: {already_processed}")
    print(f"待处理: {len(to_process)}")
    print("=" * 50 + "\n")
    
    if not to_process:
        print("所有问题都已处理完成！")
        return all_results
    
    for idx, (question_index, question) in enumerate(to_process, 1):
        question_num = question_index + 1
        print(f"\n处理问题 {question_num}/{total_questions} (待处理 {idx}/{len(to_process)}):")
        
        # 验证并修复问题
        question = validate_and_fix_question(question, input_file, question_index)
        print(f"  问题: {question[:100]}...")
        
        start_time = time.time()
        
        # 最多重试3次
        max_retries = 3
        retry_count = 0
        answer = None
        command_output = ""
        
        while retry_count < max_retries:
            answer, command_output = run_single_inference(question, model_path, adapters_path)
            
            # 检查答案有效性
            if answer and len(answer) >= 20:
                print(f"  ✓ 答案有效（长度: {len(answer)}字符）")
                break
            else:
                retry_count += 1
                if answer:
                    print(f"  ⚠ 答案过短（{len(answer)}字符），重试 {retry_count}/{max_retries}")
                else:
                    print(f"  ⚠ 答案生成失败，重试 {retry_count}/{max_retries}")
                
                if retry_count < max_retries:
                    time.sleep(3)  # 等待3秒再重试
        
        # 如果重试后仍然失败，使用错误信息
        if not answer or len(answer) < 20:
            answer = f"推理失败：答案生成不完整或过短（重试{max_retries}次后仍失败）"
            print(f"  ✗ 最终失败：无法生成有效答案")
        
        elapsed_time = time.time() - start_time
        
        result = {
            'question_id': question_num,
            'question': question,
            'question_hash': get_question_hash(question),
            'answer': answer,
            'command_output': command_output,  # 单独存储命令行输出
            'timestamp': datetime.now().isoformat(),
            'processing_time': f"{elapsed_time:.2f}秒",
            'retry_count': retry_count  # 记录重试次数
        }
        
        all_results.append(result)
        processed[result['question_hash']] = result
        
        save_single_result(result, output_file, all_results)
        
        print(f"  ✓ 完成 (用时: {elapsed_time:.2f}秒, 重试: {retry_count}次)")
        print(f"  进度: {already_processed + idx}/{total_questions} ({(already_processed + idx)*100/total_questions:.1f}%)")
        
        if idx < len(to_process):
            time.sleep(2)
    
    return all_results

def check_swift_installation():
    """
    检查Swift是否正确安装
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
                print(f"✓ Swift已安装: ms-swift {version.group(1)}")
                return True
            else:
                print("✗ ms-swift 安装信息无效")
                return False
        else:
            print(f"✗ ms-swift 未安装: {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"✗ 找不到pip命令: /root/miniconda3/bin/pip")
        return False
    except Exception as e:
        print(f"✗ Swift检查失败: {e}")
        return False

def main():
    """
    主函数
    """
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    adapters_path = "/root/autodl-tmp/output/grpo3+1/checkpoint-300"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "/root/200.txt"
    
    output_file = "/root/inference_results_grpo3+1.txt"
    
    print("=" * 50)
    print("Swift批量推理脚本（支持断点续传）")
    print("=" * 50)
    print(f"模型: {model_path}")
    print(f"适配器: {adapters_path}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("=" * 50)
    
    if not check_swift_installation():
        print("\n请先安装Swift:")
        print(f"/root/miniconda3/bin/pip install ms-swift -U")
        return
    
    cuda_available = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_available:
        print(f"✓ CUDA设备: {cuda_available}")
    else:
        print("⚠ 未设置CUDA_VISIBLE_DEVICES，将使用默认GPU 0")
    
    questions = load_questions_from_file(input_file)
    
    if not questions:
        print("未找到有效问题，退出程序")
        return
    
    print(f"\n准备处理 {len(questions)} 个问题...")
    
    print("开始处理（非交互模式，自动继续）")
    
    results = run_batch_inference_with_checkpoint(
        questions, 
        model_path, 
        adapters_path, 
        output_file,
        input_file  # 传递输入文件路径用于重新读取
    )
    
    print("\n" + "=" * 50)
    print(f"推理完成！")
    print(f"总处理问题数: {len(results)}")
    print(f"成功: {len([r for r in results if '错误' not in r['answer'] and '超时' not in r['answer'] and '失败' not in r['answer']])}")
    print(f"失败: {len([r for r in results if '错误' in r['answer'] or '超时' in r['answer'] or '失败' in r['answer']])}")
    print(f"结果文件: {output_file}")
    print(f"JSON文件: {output_file.replace('.txt', '.json')}")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        print("进度已保存，下次运行将从断点继续")
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)