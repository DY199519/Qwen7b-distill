#!/usr/bin/env python
# coding: utf-8
"""
multmm2_run234.py
-----------------
读取带有combination字段的prompt CSV文件，执行模型调用。
不保存进度，每道题都去答案文件里实时查找是否已存在。
输出 JSON 统一写入 OUTPUT_DIR。
增强功能：
- 数据完整性检查
- 回答质量验证
- 详细的错误日志
- 智能重试机制
"""

import csv, json, time
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# ========== 0. 路径配置 =======================================================
# 输出文件路径 - 放到最前面
OUTPUT_FILE = Path(r"D:\project7\MM\result\3+1\deepseek_answers_without_summary3+1-9400-10000.json")

BASE_DIR_1   = Path(r"D:\project7\MM\3+1")           # 数据文件所在根目录
BASE_DIR = Path(r"D:\project7\prompt")
OUTPUT_DIR = Path(r"D:\project7\MM\result")            # <-- 只改这里即可换输出位置
OUTPUT_DIR_1 = Path(r"D:\project7\MM\result\3+1")            # <-- 只改这里即可换输出位置

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_1.mkdir(parents=True, exist_ok=True)

# 修改为读取带combination的CSV文件
PROMPT_CSV = OUTPUT_DIR / "final_prompt_3+1-9400-10000.csv"

GROUPED_JSON = OUTPUT_DIR / "multi_model_answer9400-10000.json"

# 运行日志文件
RUN_LOG = OUTPUT_DIR_1 / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ========== 1. 模型列表 =======================================================
MODEL_CFGS = [
    {
        "model_name": "deepseek-v3",
        "api_key": "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa",
        "base_url": "https://usa.vimsai.com/v1",
        "timeout": 60,  # 超时时间（秒）
        "max_retry": 3,  # 最大重试次数
    }
    # {
    #     "model_name": "qwen2.5-72b-instruct",
    #     "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
    #     "base_url": "https://api.vansai.cn/v1",
    #     "timeout": 60,
    #     "max_retry": 3,
    # },
]

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
    def validate_answer(cls, answer: str, question: str = "") -> Tuple[bool, List[str]]:
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
def find_existing_answer(question: str, output_file: Path) -> dict:
    """
    在输出文件中查找指定问题的答案
    返回: 找到的答案项，如果没找到返回None
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
        print(f"❌ 读取答案文件失败: {e}")
        
    return None

def is_answer_complete_and_valid(item: dict) -> Tuple[bool, List[str]]:
    """
    检查答案项是否完整且有效
    返回: (是否有效, 问题列表)
    """
    if not item:
        return False, ["答案项为空"]
    
    issues = []
    
    # 检查direct_reply
    if not item.get('direct_reply'):
        issues.append("缺少direct_reply")
    else:
        is_valid, sub_issues = AnswerValidator.validate_answer(item['direct_reply'])
        if not is_valid:
            issues.extend([f"direct_reply: {issue}" for issue in sub_issues])
    
    # 检查default_reply
    if not item.get('default_reply'):
        issues.append("缺少default_reply")
    else:
        is_valid, sub_issues = AnswerValidator.validate_answer(item['default_reply'])
        if not is_valid:
            issues.extend([f"default_reply: {issue}" for issue in sub_issues])
    
    return len(issues) == 0, issues

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
            is_valid, issues = AnswerValidator.validate_answer(answer, prompt[:50])
            
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

def append_to_file(item: dict, output_file: Path, logger: Logger):
    """
    将新答案追加到文件
    """
    try:
        # 读取现有数据
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        # 添加新项
        data.append(item)
        
        # 写回文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已保存答案到文件（总数: {len(data)}）")
        
    except Exception as e:
        logger.error(f"保存答案失败: {e}")

def validate_csv_data(csv_path: Path, logger: Logger) -> bool:
    """验证CSV数据的完整性"""
    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                logger.error("CSV文件为空")
                return False
            
            # 检查必需的列
            required_columns = ['question', 'prompt']
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            if missing_columns:
                logger.error(f"CSV缺少必需列: {missing_columns}")
                return False
            
            # 检查数据完整性
            empty_questions = 0
            empty_prompts = 0
            
            for row in rows:
                if not row.get('question', '').strip():
                    empty_questions += 1
                if not row.get('prompt', '').strip():
                    empty_prompts += 1
            
            if empty_questions > 0:
                logger.warning(f"发现 {empty_questions} 个空问题")
            if empty_prompts > 0:
                logger.warning(f"发现 {empty_prompts} 个空prompt")
            
            logger.info(f"CSV数据验证完成：{len(rows)} 条记录")
            return True
            
    except Exception as e:
        logger.error(f"CSV验证失败：{e}")
        return False

# ========== 5. 主执行函数 =====================================================
def run_batch(model_cfg: dict, csv_path: Path):
    name = model_cfg["model_name"]
    logger = Logger(RUN_LOG)
    
    logger.info(f"开始运行模型: {name}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    
    # 验证CSV数据
    if not validate_csv_data(csv_path, logger):
        logger.error("CSV数据验证失败，退出运行")
        return
    
    # 初始化API
    try:
        api = OpenAI(
            api_key=model_cfg["api_key"], 
            base_url=model_cfg["base_url"]
        )
    except Exception as e:
        logger.error(f"API初始化失败: {e}")
        return

    # --- 读取 CSV ---
    questions = []
    question_prompts = {}  # 直接存储问题和prompt的映射
    
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        logger.info(f"CSV列名: {reader.fieldnames}")
        
        for row in reader:
            q = row["question"]
            prompt = row["prompt"]
            question_prompts[q] = prompt
            questions.append(q)
    
    total_questions = len(questions)
    
    # 统计信息
    stats = {
        "模型": name,
        "总问题数": total_questions,
        "跳过数": 0,
        "新生成数": 0,
        "重新生成数": 0,
        "成功数": 0,
        "失败数": 0,
        "API调用次数": 0
    }
    
    logger.info(f"=== 🚀 {name} ===")
    logger.info(f"总问题数：{stats['总问题数']}")
    
    # 错误记录
    failed_questions = []
    
    # 处理每道题
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{total_questions}] 检查问题: {q[:60]}…")
        
        # 查找现有答案
        existing_item = find_existing_answer(q, OUTPUT_FILE)
        
        if existing_item:
            # 检查答案是否完整且有效
            is_valid, issues = is_answer_complete_and_valid(existing_item)
            
            if is_valid:
                print(f"  ✅ 跳过（已有有效答案）")
                stats['跳过数'] += 1
                continue
            else:
                print(f"  🔄 需要重新生成（问题: {', '.join(issues)}）")
                stats['重新生成数'] += 1
        else:
            print(f"  🆕 生成新答案")
            stats['新生成数'] += 1
        
        # 生成答案
        # direct/basic
        stats['API调用次数'] += 1
        direct_answer = ""
        
        # 尝试多次获取有效答案
        max_attempts = 5  # 最多尝试5次
        attempt = 0
        success = False
        
        while attempt < max_attempts and not success:
            attempt += 1
            if attempt > 1:
                logger.info(f"第 {attempt} 次尝试生成direct答案...")
            
            # 获取direct答案
            direct_answer, success, errors = ask(
                api, name, q, logger,
                timeout=model_cfg.get('timeout', 60),
                max_retry=model_cfg.get('max_retry', 3)
            )
            
            if success:
                break
            else:
                logger.warning(f"第 {attempt} 次尝试失败: {errors}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)  # 递增等待时间
        
        if not success:
            failed_questions.append({
                'question': q,
                'type': 'direct',
                'errors': errors,
                'attempts': attempt
            })
            stats['失败数'] += 1
            logger.error(f"问题 '{q[:50]}...' 在 {attempt} 次尝试后仍然失败")
            # 即使失败也记录，方便后续处理
            direct_answer = f"[ERROR after {attempt} attempts]"
        
        item = {
            "question": q, 
            "direct_prompt": q, 
            "direct_reply": direct_answer,
            "timestamp": datetime.now().isoformat(),
            "attempts": attempt
        }
        
        # 处理 default prompt/reply
        if q in question_prompts:
            ptxt = question_prompts[q]
            print(f"  · 处理 default prompt...")
            
            if ptxt:
                stats['API调用次数'] += 1
                
                # 同样尝试多次
                default_attempt = 0
                default_success = False
                default_reply = ""
                
                while default_attempt < max_attempts and not default_success:
                    default_attempt += 1
                    if default_attempt > 1:
                        logger.info(f"default - 第 {default_attempt} 次尝试...")
                    
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
                    stats['失败数'] += 1
                    default_reply = f"[ERROR after {default_attempt} attempts]"
            else:
                default_reply = ""
            
            item["default_prompt"] = ptxt
            item["default_reply"] = default_reply
        
        # 检查是否所有回答都获取成功
        all_success = True
        for key in item:
            if key.endswith('_reply') and '[ERROR' in str(item.get(key, '')):
                all_success = False
                break
        
        if all_success:
            stats['成功数'] += 1
        
        # 追加到文件
        append_to_file(item, OUTPUT_FILE, logger)
        
        print(f"  ✅ 已保存答案")
        
        # 显示当前统计
        processed = stats['跳过数'] + stats['新生成数'] + stats['重新生成数']
        success_rate = (stats['成功数'] / (stats['新生成数'] + stats['重新生成数']) * 100) if (stats['新生成数'] + stats['重新生成数']) > 0 else 100
        logger.info(f"进度: {processed}/{total_questions} (跳过: {stats['跳过数']}, 新生成: {stats['新生成数']}, 重新生成: {stats['重新生成数']}, 成功率: {success_rate:.1f}%)")

    # 保存失败记录
    if failed_questions:
        failed_file = OUTPUT_DIR_1 / f"failed_questions_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        failed_file.write_text(
            json.dumps(failed_questions, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logger.warning(f"失败记录已保存到: {failed_file}")
    
    # 更新统计
    stats['最终成功率'] = f"{(stats['成功数'] / (stats['新生成数'] + stats['重新生成数']) * 100):.1f}%" if (stats['新生成数'] + stats['重新生成数']) > 0 else "100%"
    
    # 记录最终统计
    logger.summary(stats)
    
    print(f"\n✅ {name} 完成！")
    print(f"  · 总题数: {stats['总问题数']}")
    print(f"  · 跳过: {stats['跳过数']}")
    print(f"  · 新生成: {stats['新生成数']}")
    print(f"  · 重新生成: {stats['重新生成数']}")
    print(f"  · 成功: {stats['成功数']}")
    print(f"  · 失败: {stats['失败数']}")
    print(f"  · 成功率: {stats['最终成功率']}")

# ========== 6. 执行循环 =======================================================
print(f"📁 输出文件: {OUTPUT_FILE}")
print(f"📄 输入CSV: {PROMPT_CSV}")
print(f"📝 运行日志: {RUN_LOG}")
print("-" * 60)

for cfg in MODEL_CFGS:
    run_batch(cfg, PROMPT_CSV)

print(f"\n🎉 全部完成！")
print(f"📁 结果保存在: {OUTPUT_FILE}")
print(f"📝 运行日志: {RUN_LOG}")