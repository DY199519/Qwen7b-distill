#!/usr/bin/env python
# coding: utf-8
"""
single_question_grade_combination.py
------------------------------------
读取多-combination 答案 JSON，针对指定组合自动打分并持续保存进度。
增强功能：
- 数据质量检查
- 自动重新评分异常结果
- 详细的错误日志
- 评分一致性验证
"""

import json, re, os, time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import httpx
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import statistics

# ========== 配置选项 =========================================================
# 支持多种字段名的评分
FIELDS_TO_GRADE = ["3+1_reply", "default_reply"]  # 按优先级排序的字段名列表
SAVE_INTERVAL = 1  # 每 N 题保存一次

# 评分质量阈值
MIN_VALID_TRIALS = 2  # 最少需要成功的评分次数
MAX_SCORE_VARIANCE = 5  # 多次评分的最大方差（用于检测不一致）
SUSPICIOUS_SCORE_THRESHOLD = 10  # 低于此分数视为可疑，需要重新评分

# ========== OpenAI 初始化 ====================================================
httpx_client = httpx.Client(verify=False)
os.environ["OPENAI_API_KEY"]  = "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa"
os.environ["OPENAI_BASE_URL"] = "https://vir.vimsai.com/v1"
client = OpenAI(http_client=httpx_client)

# ========== 路径设置 =========================================================
INPUT_PATH = r"D:\project7\MM\result\3+1\deepseek_answers_without_summary3+1-9400-10000.json"
OUTPUT_DIR = r"D:\project7\MM\result\3+1"
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = Path(OUTPUT_DIR) / "grades-3+1-9400-10000.json"

# 日志文件
LOG_FILE = Path(OUTPUT_DIR) / f"grading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ========== 日志记录器 =======================================================
class Logger:
    """简单的日志记录器"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = datetime.now()
        self._write(f"=== 评分开始: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    def _write(self, message: str):
        """写入日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    
    def info(self, message: str):
        """记录信息"""
        self._write(f"INFO: {message}")
        print(f"📝 {message}")
    
    def error(self, message: str):
        """记录错误"""
        self._write(f"ERROR: {message}")
        print(f"❌ {message}")
    
    def warning(self, message: str):
        """记录警告"""
        self._write(f"WARNING: {message}")
        print(f"⚠️ {message}")
    
    def success(self, message: str):
        """记录成功"""
        self._write(f"SUCCESS: {message}")
        print(f"✅ {message}")

# ========== 评分质量检查器 ===================================================
class ScoreValidator:
    """评分质量验证器"""
    
    @staticmethod
    def validate_single_score(scores: Dict[str, int]) -> Tuple[bool, List[str]]:
        """验证单次评分的有效性"""
        issues = []
        
        # 检查分数范围
        if not (0 <= scores["total"] <= 50):
            issues.append(f"总分异常: {scores['total']}")
        
        for key in ["logic", "depth", "innovation", "accuracy", "completeness"]:
            if key not in scores:
                issues.append(f"缺少{key}分数")
            elif not (0 <= scores[key] <= 10):
                issues.append(f"{key}分数异常: {scores[key]}")
        
        # 检查总分是否等于各项之和
        expected_total = sum(scores.get(k, 0) for k in ["logic", "depth", "innovation", "accuracy", "completeness"])
        if scores["total"] != expected_total:
            issues.append(f"总分({scores['total']})与各项之和({expected_total})不符")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_multiple_scores(all_scores: List[Dict[str, int]]) -> Tuple[bool, List[str]]:
        """验证多次评分的一致性"""
        issues = []
        
        if len(all_scores) < MIN_VALID_TRIALS:
            issues.append(f"有效评分次数不足: {len(all_scores)} < {MIN_VALID_TRIALS}")
            return False, issues
        
        # 计算总分的方差
        totals = [s["total"] for s in all_scores]
        if len(totals) > 1:
            variance = statistics.variance(totals)
            if variance > MAX_SCORE_VARIANCE:
                issues.append(f"评分一致性差，方差: {variance:.2f} > {MAX_SCORE_VARIANCE}")
        
        # 检查是否有异常低分
        avg_total = sum(totals) / len(totals)
        if avg_total < SUSPICIOUS_SCORE_THRESHOLD:
            issues.append(f"平均分过低: {avg_total:.2f} < {SUSPICIOUS_SCORE_THRESHOLD}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_grading_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证完整的评分结果"""
        issues = []
        
        # 检查必要字段
        required_fields = ["question", "avg_scores", "avg_score_100", "num_valid_trials", "all_scores"]
        for field in required_fields:
            if field not in result:
                issues.append(f"缺少必要字段: {field}")
        
        # 检查评分次数
        if result.get("num_valid_trials", 0) < MIN_VALID_TRIALS:
            issues.append(f"有效评分次数不足")
        
        # 检查平均分计算
        if "avg_score_100" in result and "avg_scores" in result:
            expected_100 = result["avg_scores"]["total"] * 2
            if abs(result["avg_score_100"] - expected_100) > 0.1:
                issues.append(f"百分制分数计算错误")
        
        return len(issues) == 0, issues

# ========== Prompt 模板 =====================================================
PROMPT_TMPL = """
你是一个专业答题评审员，请对以下答案进行评分，按照以下 5 个维度打分：
1. 逻辑性   2. 深度   3. 创新性   4. 准确性   5. 完整性
每维度满分 5，总分 25。

评分格式示例（严格照抄数字和空格）：
15 3 3 3 3 3
（此行后面紧跟评分理由段落）

### 问题
{question}

### 回答
{answer}

### 输出要求
- 第一行 **只写 6 个数字**，用空格分隔，顺序是：总分 逻辑 深度 创新 准确 完整
- 不要写任何文字、单位或标点
- 第二行开始写详细评分理由（至少 2 段）

1. 逻辑性 —— 论证结构、因果链条是否严谨；  
2. 深度   —— 是否引用学术概念 / 数据 / 多角度分析；  
3. 创新性 —— 是否提出新观点或非陈词滥调的洞见；  
4. 准确性 —— 事实、数据、概念是否正确；  
5. 完整性 —— 是否充分回答题干所有要点。

**打分硬性规则**（一定要执行,请谨慎打高分）：  
| 单维得分 | 评价基准（示例） |  
|----------|-----------------|  
| 5 | 几乎无缺陷，仅可挑细节 |  
| 4  | 有 1–2 处轻微缺陷 |  
| 3  | 出现 **明显缺陷** 或遗漏要点 |  
| 2  | 多处缺陷，论证/事实错误 >2 处 |  
| 0–1  | 关键逻辑不成立，或事实错误严重 |
严格遵守格式，现在开始：
"""

# ---------------------------------------------------------------------------
def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """读取 JSON 文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
    return []

# ---------------------------------------------------------------------------
def load_existing_results(output_file: Path, logger: Logger) -> Tuple[Dict[str, Any] | None, set, List[str]]:
    """
    加载已有评分进度并检查质量
    返回: (完整数据, 有效完成集合, 需要重新评分的问题列表)
    """
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            detailed_results = data.get("detailed_results", [])
            valid_done = set()
            need_regrade = []
            
            logger.info(f"检查已有评分质量...")
            
            for result in detailed_results:
                question = result["question"]
                is_valid, issues = ScoreValidator.validate_grading_result(result)
                
                if is_valid:
                    # 再检查多次评分的一致性
                    if "all_scores" in result:
                        consistency_valid, consistency_issues = ScoreValidator.validate_multiple_scores(result["all_scores"])
                        if not consistency_valid:
                            is_valid = False
                            issues.extend(consistency_issues)
                
                if is_valid:
                    valid_done.add(question)
                else:
                    need_regrade.append(question)
                    logger.warning(f"问题 '{question[:40]}...' 需要重新评分: {', '.join(issues)}")
            
            logger.info(f"评分质量检查完成：")
            logger.info(f"  · 有效评分: {len(valid_done)}")
            logger.info(f"  · 需要重评: {len(need_regrade)}")
            
            return data, valid_done, need_regrade
            
        except Exception as e:
            logger.error(f"读取进度文件失败: {e}")
    
    return None, set(), []

# ---------------------------------------------------------------------------
def save_progress(data: Dict[str, Any], output_file: Path, logger: Logger):
    """保存进度，包含备份机制"""
    try:
        # 创建备份
        if output_file.exists():
            backup_file = output_file.with_suffix('.backup.json')
            output_file.rename(backup_file)
        
        # 保存新文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.success(f"进度已保存至 {output_file}")
        
        # 删除备份
        if output_file.with_suffix('.backup.json').exists():
            output_file.with_suffix('.backup.json').unlink()
            
    except Exception as e:
        logger.error(f"保存失败: {e}")
        # 恢复备份
        backup_file = output_file.with_suffix('.backup.json')
        if backup_file.exists():
            backup_file.rename(output_file)
            logger.info("已从备份恢复")

# ---------------------------------------------------------------------------
def parse_response(raw: str, logger: Logger) -> Tuple[Dict[str, int], str]:
    """解析 GPT 输出，增加错误处理"""
    keys = ["total", "logic", "depth", "innovation", "accuracy", "completeness"]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # 找分数字串
    score_line = None
    for line in lines:
        # 尝试找到包含6个数字的行
        numbers = re.findall(r'\b\d+\b', line)
        if len(numbers) >= 6:
            score_line = line
            break
    
    if not score_line:
        raise ValueError("找不到完整分数行")
    
    nums = list(map(int, re.findall(r'\b\d+\b', score_line)[:6]))
    
    # 验证分数
    scores = dict(zip(keys, nums))
    is_valid, issues = ScoreValidator.validate_single_score(scores)
    if not is_valid:
        logger.warning(f"分数验证失败: {', '.join(issues)}")
        raise ValueError(f"分数验证失败: {', '.join(issues)}")

    # 提取评论
    score_line_idx = lines.index(score_line)
    commentary = "\n".join(lines[score_line_idx + 1:]).strip()
    if not commentary:
        raise ValueError("缺少评分理由")

    return scores, commentary

# ---------------------------------------------------------------------------
def ask_and_parse(prompt: str,
                  logger: Logger,
                  model: str = "gpt-4o",
                  max_attempts: int = 6,
                  backoff_base: int = 2):
    """调用API并解析结果，增加错误处理"""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
            scores, detail = parse_response(raw, logger)
            return scores, detail, raw
        except Exception as e:
            wait = backoff_base ** attempt
            logger.warning(f"第 {attempt}/{max_attempts} 次失败: {e}，{wait}s 后重试")
            time.sleep(wait)
    return None

# ---------------------------------------------------------------------------
def grade_single(question: str, answer: str, logger: Logger, trials: int = 3, max_retries: int = 2):
    """
    对单个问题进行评分，增加重试机制
    max_retries: 如果所有trials都失败，最多重试的轮数
    """
    prompt = PROMPT_TMPL.format(question=question, answer=answer)
    
    for retry in range(max_retries + 1):
        if retry > 0:
            logger.info(f"第 {retry + 1} 轮重试...")
            time.sleep(5 * retry)  # 递增等待
        
        all_scores, all_cmts, raws = [], [], []
        
        for t in range(trials):
            res = ask_and_parse(prompt, logger)
            if not res:
                logger.warning(f"  第 {t+1} 次评分失败")
                continue
            
            score, cmt, raw = res
            all_scores.append(score)
            all_cmts.append(cmt)
            raws.append(raw)
            logger.info(f"  第 {t+1} 次得分：{score['total']}/50")
        
        # 检查评分一致性
        if len(all_scores) >= MIN_VALID_TRIALS:
            is_valid, issues = ScoreValidator.validate_multiple_scores(all_scores)
            if is_valid:
                # 计算平均分
                avg = {k: round(sum(s[k] for s in all_scores) / len(all_scores), 2)
                       for k in all_scores[0]}
                avg100 = round(avg["total"] * 2, 2)
                
                return {
                    "question": question,
                    "avg_scores": avg,
                    "avg_score_100": avg100,
                    "num_valid_trials": len(all_scores),
                    "all_scores": all_scores,
                    "all_commentaries": all_cmts,
                    "all_gpt_raws": raws,
                    "retry_count": retry
                }
            else:
                logger.warning(f"评分一致性检查失败: {', '.join(issues)}")
    
    logger.error(f"经过 {max_retries + 1} 轮尝试仍无法获得有效评分")
    return None

# ---------------------------------------------------------------------------
def get_field_value(record: Dict[str, Any], fields: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    从记录中获取字段值，按优先级尝试不同的字段名
    返回: (字段值, 使用的字段名)
    """
    for field in fields:
        if field in record and record[field] and record[field].strip():
            return record[field], field
    return None, None

# ---------------------------------------------------------------------------
def grade_replies(records: List[Dict[str, Any]], logger: Logger):
    """评分reply字段，支持多种字段名"""
    logger.info(f"===== 评分 {'/'.join(FIELDS_TO_GRADE)} 字段 =====")

    # 输出文件名
    prev, valid_done_set, need_regrade = load_existing_results(OUTPUT_FILE, logger)
    
    # 获取已有的有效结果
    existing_results = []
    if prev:
        for result in prev.get("detailed_results", []):
            if result["question"] not in need_regrade:
                existing_results.append(result)
    
    # 筛选有效数据并记录使用的字段
    items = []
    field_usage = {}  # 记录每个问题使用的字段名
    
    for record in records:
        value, field_used = get_field_value(record, FIELDS_TO_GRADE)
        if value:
            items.append(record)
            field_usage[record["question"]] = field_used
    
    if not items:
        logger.error(f"未找到包含 {'/'.join(FIELDS_TO_GRADE)} 字段的有效数据")
        return
    
    # 统计字段使用情况
    field_counts = {}
    for field in field_usage.values():
        field_counts[field] = field_counts.get(field, 0) + 1
    
    logger.info("字段使用统计：")
    for field, count in field_counts.items():
        logger.info(f"  · {field}: {count} 题")

    # 计算待处理项
    # 1. 全新的问题
    new_questions = [d for d in items if d["question"] not in valid_done_set and d["question"] not in need_regrade]
    # 2. 需要重新评分的问题
    regrade_questions = [d for d in items if d["question"] in need_regrade]
    
    pending = new_questions + regrade_questions
    
    logger.info(f"数据统计：")
    logger.info(f"  · 总题数: {len(items)}")
    logger.info(f"  · 已有效完成: {len(valid_done_set)}")
    logger.info(f"  · 需要重评: {len(regrade_questions)}")
    logger.info(f"  · 全新题目: {len(new_questions)}")
    logger.info(f"  · 待处理总数: {len(pending)}")

    # 主循环
    results = existing_results.copy()
    all_totals = [r["avg_scores"]["total"] for r in results]
    all_totals100 = [r["avg_score_100"] for r in results]
    
    # 按字段分类的统计
    field_stats = {field: {"count": 0, "total_score": 0} for field in FIELDS_TO_GRADE}
    
    failed_questions = []
    regraded_count = 0

    for idx, item in enumerate(pending, 1):
        q = item["question"]
        field_used = field_usage[q]
        a = item[field_used]
        
        is_regrade = q in need_regrade
        
        if is_regrade:
            logger.info(f"\n🔄 [{idx}/{len(pending)}] 重新评分 ({field_used}): {q[:40]}...")
            regraded_count += 1
        else:
            logger.info(f"\n[{idx}/{len(pending)}] 评分 ({field_used}): {q[:40]}...")
        
        res = grade_single(q, a, logger)
        
        if res:
            res["field_graded"] = field_used  # 记录实际使用的字段
            res["is_regraded"] = is_regrade
            res["grading_timestamp"] = datetime.now().isoformat()
            
            results.append(res)
            all_totals.append(res["avg_scores"]["total"])
            all_totals100.append(res["avg_score_100"])
            
            # 更新字段统计
            field_stats[field_used]["count"] += 1
            field_stats[field_used]["total_score"] += res["avg_scores"]["total"]
            
            # 再次验证结果
            is_valid, issues = ScoreValidator.validate_grading_result(res)
            if not is_valid:
                logger.warning(f"评分结果验证失败: {', '.join(issues)}")
        else:
            failed_questions.append({
                "question": q,
                "field_used": field_used,
                "reason": "无法获得有效评分",
                "timestamp": datetime.now().isoformat()
            })

        # 定期保存
        if idx % SAVE_INTERVAL == 0:
            # 计算各字段的平均分
            field_averages = {}
            for field, stats in field_stats.items():
                if stats["count"] > 0:
                    field_averages[field] = {
                        "count": stats["count"],
                        "average": round(stats["total_score"] / stats["count"], 2)
                    }
            
            stats = {
                "fields_graded": FIELDS_TO_GRADE,
                "total_questions": len(items),
                "valid_grades": len(all_totals),
                "failed_grades": len(failed_questions),
                "regraded_count": regraded_count,
                "total_average": round(sum(all_totals)/len(all_totals), 2) if all_totals else 0,
                "total_average_100": round(sum(all_totals100)/len(all_totals100), 2) if all_totals100 else 0,
                "field_statistics": field_averages,
                "last_update": datetime.now().isoformat()
            }
            save_progress({"statistics": stats, "detailed_results": results}, OUTPUT_FILE, logger)

    # 最终统计和保存
    if all_totals:
        # 计算各字段的最终平均分
        field_averages = {}
        for field in FIELDS_TO_GRADE:
            field_results = [r for r in results if r.get("field_graded") == field]
            if field_results:
                field_scores = [r["avg_scores"]["total"] for r in field_results]
                field_averages[field] = {
                    "count": len(field_results),
                    "average": round(sum(field_scores) / len(field_scores), 2),
                    "average_100": round(sum(field_scores) / len(field_scores) * 2, 2)
                }
        
        stats = {
            "fields_graded": FIELDS_TO_GRADE,
            "total_questions": len(items),
            "valid_grades": len(all_totals),
            "failed_grades": len(failed_questions),
            "regraded_count": regraded_count,
            "total_average": round(sum(all_totals)/len(all_totals), 2),
            "total_average_100": round(sum(all_totals100)/len(all_totals100), 2),
            "field_statistics": field_averages,
            "score_distribution": {
                "0-20": len([s for s in all_totals if s < 20]),
                "20-30": len([s for s in all_totals if 20 <= s < 30]),
                "30-40": len([s for s in all_totals if 30 <= s < 40]),
                "40-50": len([s for s in all_totals if 40 <= s <= 50])
            },
            "completion_time": datetime.now().isoformat()
        }
        
        final_data = {
            "statistics": stats,
            "detailed_results": results
        }
        
        # 保存失败记录
        if failed_questions:
            final_data["failed_questions"] = failed_questions
            
            # 单独保存失败记录文件
            failed_file = Path(OUTPUT_DIR) / f"failed_grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_questions, f, ensure_ascii=False, indent=2)
            logger.warning(f"失败记录已保存到: {failed_file}")
        
        save_progress(final_data, OUTPUT_FILE, logger)
        
        logger.success(f"\n📊 评分完成！")
        logger.success(f"  · 总平均分: {stats['total_average']}/50 ({stats['total_average_100']}分)")
        logger.success(f"  · 有效评分: {stats['valid_grades']}")
        logger.success(f"  · 失败: {stats['failed_grades']}")
        logger.success(f"  · 重新评分: {stats['regraded_count']}")
        
        # 显示各字段统计
        logger.success(f"\n📈 各字段统计：")
        for field, field_stat in field_averages.items():
            logger.success(f"  · {field}: {field_stat['count']} 题, 平均 {field_stat['average']}/50 ({field_stat['average_100']}分)")

# ---------------------------------------------------------------------------
def main():
    logger = Logger(LOG_FILE)
    logger.info("开始评分任务")
    
    data = read_json_file(INPUT_PATH)
    if not data:
        logger.error("读取数据文件失败")
        return

    # 对reply字段进行评分（支持多种字段名）
    grade_replies(data, logger)
    
    # 计算总耗时
    elapsed = datetime.now() - logger.start_time
    logger.info(f"任务完成，总耗时: {elapsed}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 输入文件: {INPUT_PATH}")
    print(f"📝 日志文件: {LOG_FILE}")
    print("-" * 60)
    
    main()