#!/usr/bin/env python
# coding: utf-8
"""
txt_qa_processor.py
-----------------
读取JSON格式的问题文件，在TXT格式的prompt文件中匹配对应的prompt，执行模型调用，生成TXT格式的答案文件
每个prompt用-------------------严格分开
输出格式为：问题：XXXX 回复:XXXXX
"""

import time
import json
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# ========== 0. 路径配置 =======================================================
# 控制生成题目数量（None表示全部生成，数字表示只生成前N道题）
LIMIT_QUESTIONS = 2  # 例如：10 表示只生成前10道题，None 表示全部生成

# 输入输出文件路径
JSON_FILE = Path(r"D:\qwensft\testquestion\multi_model_answersTest500.json")  # JSON问题文件路径
TXT_FILE = Path(r"D:\qwensft\uploadjson\final_prompt_3+1-Test.txt")  # TXT prompt文件路径
OUTPUT_FILE = Path(r"D:\qwensft\uploadjson\final_answer_3+1-Test.txt")  # 修改为你的输出文件路径

# 确保输出目录存在
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 运行日志文件
RUN_LOG = OUTPUT_FILE.parent / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ========== 1. 模型配置 =======================================================
MODEL_CFG = {
    "model_name": "deepseek-v3",
    "api_key": "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa",
    "base_url": "https://usa.vimsai.com/v1",
    "timeout": 60,  # 超时时间（秒）
    "max_retry": 3,  # 最大重试次数
}

# ========== 2. 答案验证类 ====================================================
class AnswerValidator:
    """答案验证器"""
    
    # 最小答案长度
    MIN_ANSWER_LENGTH = 5
    
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
    ]
    
    @classmethod
    def validate_answer(cls, answer: str) -> Tuple[bool, List[str]]:
        """验证答案是否有效"""
        issues = []
        
        if not answer:
            issues.append("答案为空")
            return False, issues
        
        if not isinstance(answer, str):
            issues.append(f"答案类型错误: {type(answer)}")
            return False, issues
        
        answer = answer.strip()
        
        # 长度检查
        if len(answer) < cls.MIN_ANSWER_LENGTH:
            issues.append(f"答案过短 ({len(answer)} 字符)")
        
        # 错误模式检查
        for pattern in cls.ERROR_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                issues.append(f"匹配错误模式: {pattern}")
                return False, issues
        
        return len(issues) == 0, issues

# ========== 3. 日志记录器 ====================================================
class Logger:
    """简单的日志记录器"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = datetime.now()
        self._write(f"=== 运行开始: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    def _write(self, message: str):
        """写入日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    
    def info(self, message: str):
        """记录信息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{timestamp}] INFO: {message}")
        print(f"📝 {message}")
    
    def error(self, message: str):
        """记录错误"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{timestamp}] ERROR: {message}")
        print(f"❌ {message}")
    
    def warning(self, message: str):
        """记录警告"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{timestamp}] WARNING: {message}")
        print(f"⚠️ {message}")
    
    def summary(self, stats: dict):
        """记录统计摘要"""
        elapsed = datetime.now() - self.start_time
        self._write(f"\n=== 运行统计 ===")
        self._write(f"总耗时: {elapsed}")
        for key, value in stats.items():
            self._write(f"{key}: {value}")
        self._write("=" * 50)

# ========== 4. 工具函数 =======================================================
def read_json_questions(file_path: Path, logger: Logger) -> List[str]:
    """
    读取JSON文件中的问题列表
    从questions字段中按顺序提取问题
    """
    if not file_path.exists():
        logger.error(f"JSON文件不存在: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从JSON中提取问题
        questions = []
        if 'questions' in data:
            for question_key in data['questions'].keys():
                questions.append(question_key)
        
        logger.info(f"成功读取 {len(questions)} 个问题")
        return questions
        
    except Exception as e:
        logger.error(f"读取JSON文件失败: {e}")
        return []

def read_txt_prompts(file_path: Path, logger: Logger) -> Dict[str, str]:
    """
    读取TXT文件中的prompts，并建立问题到prompt的映射
    每个prompt用-------------------分开
    """
    if not file_path.exists():
        logger.error(f"TXT文件不存在: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用-------------------分割
        prompts = content.split('-------------------')
        
        # 建立问题到prompt的映射
        question_to_prompt = {}
        for prompt in prompts:
            prompt = prompt.strip()
            if prompt:
                # 从prompt中提取问题
                question = extract_question_from_prompt(prompt)
                if question:
                    question_to_prompt[question] = prompt
        
        logger.info(f"成功读取 {len(question_to_prompt)} 个prompts")
        return question_to_prompt
        
    except Exception as e:
        logger.error(f"读取TXT文件失败: {e}")
        return {}

def extract_question_from_prompt(prompt: str) -> str:
    """
    从prompt中提取问题部分
    假设问题在prompt的第一行或者包含"问题"关键字的行
    """
    lines = prompt.split('\n')
    
    # 查找包含"问题"的行
    for line in lines:
        if '问题' in line and '"' in line:
            # 提取引号内的内容
            match = re.search(r'"([^"]+)"', line)
            if match:
                return match.group(1)
    
    # 如果没找到，返回prompt的前100个字符作为标识
    return prompt[:100] if len(prompt) > 100 else prompt

def find_matching_prompt(question: str, question_to_prompt: Dict[str, str], logger: Logger) -> Optional[str]:
    """
    在prompt映射中查找匹配的prompt
    """
    # 直接匹配
    if question in question_to_prompt:
        return question_to_prompt[question]
    
    # 模糊匹配（去除空格和标点符号）
    normalized_question = re.sub(r'[^\w]', '', question)
    for prompt_question, prompt in question_to_prompt.items():
        normalized_prompt_question = re.sub(r'[^\w]', '', prompt_question)
        if normalized_question == normalized_prompt_question:
            logger.info(f"模糊匹配成功: {question[:50]}...")
            return prompt
    
    # 部分匹配（包含关系）
    for prompt_question, prompt in question_to_prompt.items():
        if question in prompt_question or prompt_question in question:
            logger.info(f"部分匹配成功: {question[:50]}...")
            return prompt
    
    logger.warning(f"未找到匹配的prompt: {question[:50]}...")
    return None

def ask(api: OpenAI, model: str, prompt: str, logger: Logger, 
        timeout: int = 60, max_retry: int = 3, pause: float = 2.0) -> Tuple[str, bool, List[str]]:
    """
    调用模型API
    返回: (答案, 是否成功, 错误列表)
    """
    errors = []
    
    for i in range(1, max_retry + 1):
        try:
            # 设置超时
            rsp = api.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout
            )
            answer = rsp.choices[0].message.content.strip()
            
            # 验证答案
            is_valid, issues = AnswerValidator.validate_answer(answer)
            
            if is_valid and answer:
                return answer, True, []
            else:
                errors.append(f"第{i}次尝试 - 答案验证失败: {', '.join(issues)}")
                logger.warning(f"答案验证失败: {issues}")
                
        except Exception as e:
            error_msg = f"第{i}次尝试失败: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        if i < max_retry:
            time.sleep(pause * i)  # 递增等待时间
    
    return "", False, errors

def append_to_output(question: str, answer: str, output_file: Path, logger: Logger):
    """
    将问答对追加到输出文件
    格式：问题：XXXX  回复:XXXXX
    """
    try:
        # 构建输出内容
        output_content = f"问题：{question}\参考回复：{answer}\n"
        
        # 如果文件已存在且有内容，添加分隔符
        if output_file.exists() and output_file.stat().st_size > 0:
            output_content = "-------------------\n" + output_content
        
        # 追加到文件
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(output_content)
        
        logger.info(f"已保存答案到文件")
        
    except Exception as e:
        logger.error(f"保存答案失败: {e}")

def check_existing_answers(output_file: Path, logger: Logger) -> set:
    """
    检查输出文件中已存在的问题
    返回已回答问题的集合
    """
    existing_questions = set()
    
    if not output_file.exists():
        return existing_questions
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割已有的回答
        answers = content.split('-------------------')
        
        for answer in answers:
            if '问题：' in answer:
                # 提取问题
                match = re.search(r'问题：(.+?)(?:\n|$)', answer)
                if match:
                    question = match.group(1).strip()
                    existing_questions.add(question)
        
        logger.info(f"发现 {len(existing_questions)} 个已存在的答案")
        
    except Exception as e:
        logger.error(f"读取已有答案失败: {e}")
    
    return existing_questions

# ========== 5. 主执行函数 =====================================================
def run_txt_processing():
    """主处理函数"""
    logger = Logger(RUN_LOG)
    
    logger.info(f"开始处理")
    logger.info(f"JSON问题文件: {JSON_FILE}")
    logger.info(f"TXT prompt文件: {TXT_FILE}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    logger.info(f"模型: {MODEL_CFG['model_name']}")
    
    # 显示题目限制设置
    if LIMIT_QUESTIONS is not None:
        logger.info(f"⚠️ 设置生成限制：仅处理前 {LIMIT_QUESTIONS} 道题")
    else:
        logger.info(f"处理所有题目（无限制）")
    
    # 读取JSON问题
    questions = read_json_questions(JSON_FILE, logger)
    if not questions:
        logger.error("没有读取到任何问题")
        return
    
    # 读取TXT prompts
    question_to_prompt = read_txt_prompts(TXT_FILE, logger)
    if not question_to_prompt:
        logger.error("没有读取到任何prompts")
        return
    
    # 应用题目数量限制
    original_count = len(questions)
    if LIMIT_QUESTIONS is not None and LIMIT_QUESTIONS > 0:
        questions = questions[:LIMIT_QUESTIONS]
        logger.info(f"应用限制：从 {original_count} 道题中选择前 {len(questions)} 道题")
    
    # 检查已存在的答案
    existing_questions = check_existing_answers(OUTPUT_FILE, logger)
    
    # 初始化API
    try:
        api = OpenAI(
            api_key=MODEL_CFG["api_key"], 
            base_url=MODEL_CFG["base_url"]
        )
        logger.info("API初始化成功")
    except Exception as e:
        logger.error(f"API初始化失败: {e}")
        return
    
    # 统计信息
    stats = {
        "模型": MODEL_CFG["model_name"],
        "原始题目数": original_count,
        "处理题目数": len(questions),
        "题目限制": LIMIT_QUESTIONS if LIMIT_QUESTIONS else "无限制",
        "跳过数": 0,
        "成功数": 0,
        "失败数": 0,
        "未匹配数": 0,
        "API调用次数": 0
    }
    
    failed_prompts = []
    unmatched_questions = []
    
    # 处理每个问题
    for idx, question in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] 处理问题: {question[:60]}...")
        
        # 检查是否已存在
        if question in existing_questions:
            print(f"  ✅ 跳过（已有答案）")
            stats['跳过数'] += 1
            continue
        
        # 查找匹配的prompt
        prompt = find_matching_prompt(question, question_to_prompt, logger)
        if not prompt:
            print(f"  ❌ 未找到匹配的prompt")
            stats['未匹配数'] += 1
            unmatched_questions.append(question)
            continue
        
        # 调用API获取答案
        stats['API调用次数'] += 1
        
        max_attempts = 5
        attempt = 0
        success = False
        answer = ""
        
        while attempt < max_attempts and not success:
            attempt += 1
            if attempt > 1:
                logger.info(f"第 {attempt} 次尝试...")
            
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
                logger.warning(f"第 {attempt} 次尝试失败: {errors}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)
        
        if success:
            stats['成功数'] += 1
            # 保存到文件
            append_to_output(question, answer, OUTPUT_FILE, logger)
            print(f"  ✅ 成功生成并保存答案")
        else:
            stats['失败数'] += 1
            failed_prompts.append({
                'question': question,
                'prompt': prompt[:200],
                'errors': errors,
                'attempts': attempt
            })
            logger.error(f"问题 '{question[:50]}...' 在 {attempt} 次尝试后失败")
            
            # 即使失败也记录
            append_to_output(question, f"[生成失败: {errors[-1] if errors else 'Unknown error'}]", OUTPUT_FILE, logger)
        
        # 显示进度
        processed = idx
        processed_count = processed - stats['跳过数'] - stats['未匹配数']
        success_rate = (stats['成功数'] / processed_count * 100) if processed_count > 0 else 100
        logger.info(f"进度: {processed}/{len(questions)} (成功率: {success_rate:.1f}%)")
        
        # 短暂延迟，避免请求过快
        time.sleep(1)
    
    # 保存失败记录
    if failed_prompts:
        failed_file = OUTPUT_FILE.parent / f"failed_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(failed_file, 'w', encoding='utf-8') as f:
            for item in failed_prompts:
                f.write(f"问题: {item['question']}\n")
                f.write(f"尝试次数: {item['attempts']}\n")
                f.write(f"错误: {item['errors']}\n")
                f.write("-------------------\n")
        logger.warning(f"失败记录已保存到: {failed_file}")
    
    # 保存未匹配问题记录
    if unmatched_questions:
        unmatched_file = OUTPUT_FILE.parent / f"unmatched_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(unmatched_file, 'w', encoding='utf-8') as f:
            for question in unmatched_questions:
                f.write(f"{question}\n")
        logger.warning(f"未匹配问题已保存到: {unmatched_file}")
    
    # 更新统计
    actual_processed = stats['处理题目数'] - stats['跳过数'] - stats['未匹配数']
    stats['最终成功率'] = f"{(stats['成功数'] / actual_processed * 100):.1f}%" if actual_processed > 0 else "100%"
    
    # 记录最终统计
    logger.summary(stats)
    
    print(f"\n🎉 处理完成！")
    print(f"  · 原始总数: {stats['原始题目数']}")
    print(f"  · 处理数量: {stats['处理题目数']} (限制: {stats['题目限制']})")
    print(f"  · 跳过: {stats['跳过数']}")
    print(f"  · 未匹配: {stats['未匹配数']}")
    print(f"  · 成功: {stats['成功数']}")
    print(f"  · 失败: {stats['失败数']}")
    print(f"  · 成功率: {stats['最终成功率']}")

# ========== 6. 执行入口 =======================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TXT格式问答处理脚本 (JSON问题 + TXT Prompt匹配)")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not JSON_FILE.exists():
        print(f"❌ JSON问题文件不存在: {JSON_FILE}")
    elif not TXT_FILE.exists():
        print(f"❌ TXT prompt文件不存在: {TXT_FILE}")
    else:
        run_txt_processing()
        print(f"\n📁 结果保存在: {OUTPUT_FILE}")
        print(f"📝 运行日志: {RUN_LOG}")