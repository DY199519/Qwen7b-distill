#!/usr/bin/env python
# coding: utf-8
"""
single_question_grade_combination.py
------------------------------------
Reads multi-combination answer JSON, automatically scores and continuously saves progress for specified combinations.
Enhanced features:
- Data quality checks
- Automatic re-grading of abnormal results
- Detailed error logging
- Scoring consistency verification
"""

import json, re, os, time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import httpx
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import statistics

# ========== Configuration Options =========================================================
# Support scoring for multiple field names
FIELDS_TO_GRADE = ["3+1_reply", "default_reply"]  # List of field names in priority order
SAVE_INTERVAL = 1  # Save progress every N questions

# Scoring quality thresholds
MIN_VALID_TRIALS = 2  # Minimum number of successful scoring attempts required
MAX_SCORE_VARIANCE = 5  # Maximum variance for multiple scores (used to detect inconsistencies)
SUSPICIOUS_SCORE_THRESHOLD = 10  # Scores below this are considered suspicious and require re-grading

# ========== OpenAI Initialization ====================================================
httpx_client = httpx.Client(verify=False)
os.environ["OPENAI_API_KEY"]  = "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa"
os.environ["OPENAI_BASE_URL"] = "https://vir.vimsai.com/v1"
client = OpenAI(http_client=httpx_client)

# ========== Path Settings =========================================================
INPUT_PATH = r"D:\project7\MM\result\3+1\deepseek_answers_without_summary3+1-9400-10000.json"
OUTPUT_DIR = r"D:\project7\MM\result\3+1"
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = Path(OUTPUT_DIR) / "grades-3+1-9400-10000.json"

# Log file
LOG_FILE = Path(OUTPUT_DIR) / f"grading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ========== Logger =======================================================
class Logger:
    """Simple logging utility"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = datetime.now()
        self._write(f"=== Grading started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    def _write(self, message: str):
        """Write to log"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    
    def info(self, message: str):
        """Log information"""
        self._write(f"INFO: {message}")
        print(f"📝 {message}")
    
    def error(self, message: str):
        """Log error"""
        self._write(f"ERROR: {message}")
        print(f"❌ {message}")
    
    def warning(self, message: str):
        """Log warning"""
        self._write(f"WARNING: {message}")
        print(f"⚠️ {message}")
    
    def success(self, message: str):
        """Log success"""
        self._write(f"SUCCESS: {message}")
        print(f"✅ {message}")

# ========== Score Validator ===================================================
class ScoreValidator:
    """Score quality validator"""
    
    @staticmethod
    def validate_single_score(scores: Dict[str, int]) -> Tuple[bool, List[str]]:
        """Validate the validity of a single score"""
        issues = []
        
        # Check score range
        if not (0 <= scores["total"] <= 50):
            issues.append(f"Abnormal total score: {scores['total']}")
        
        for key in ["logic", "depth", "innovation", "accuracy", "completeness"]:
            if key not in scores:
                issues.append(f"Missing {key} score")
            elif not (0 <= scores[key] <= 10):
                issues.append(f"Abnormal {key} score: {scores[key]}")
        
        # Check if total equals sum of components
        expected_total = sum(scores.get(k, 0) for k in ["logic", "depth", "innovation", "accuracy", "completeness"])
        if scores["total"] != expected_total:
            issues.append(f"Total score ({scores['total']}) does not match sum of components ({expected_total})")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_multiple_scores(all_scores: List[Dict[str, int]]) -> Tuple[bool, List[str]]:
        """Validate consistency of multiple scores"""
        issues = []
        
        if len(all_scores) < MIN_VALID_TRIALS:
            issues.append(f"Insufficient valid scoring attempts: {len(all_scores)} < {MIN_VALID_TRIALS}")
            return False, issues
        
        # Calculate variance of total scores
        totals = [s["total"] for s in all_scores]
        if len(totals) > 1:
            variance = statistics.variance(totals)
            if variance > MAX_SCORE_VARIANCE:
                issues.append(f"Poor scoring consistency, variance: {variance:.2f} > {MAX_SCORE_VARIANCE}")
        
        # Check for abnormally low scores
        avg_total = sum(totals) / len(totals)
        if avg_total < SUSPICIOUS_SCORE_THRESHOLD:
            issues.append(f"Average score too low: {avg_total:.2f} < {SUSPICIOUS_SCORE_THRESHOLD}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_grading_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate complete grading result"""
        issues = []
        
        # Check for required fields
        required_fields = ["question", "avg_scores", "avg_score_100", "num_valid_trials", "all_scores"]
        for field in required_fields:
            if field not in result:
                issues.append(f"Missing required field: {field}")
        
        # Check number of scoring attempts
        if result.get("num_valid_trials", 0) < MIN_VALID_TRIALS:
            issues.append(f"Insufficient valid scoring attempts")
        
        # Check average score calculation
        if "avg_score_100" in result and "avg_scores" in result:
            expected_100 = result["avg_scores"]["total"] * 2
            if abs(result["avg_score_100"] - expected_100) > 0.1:
                issues.append(f"Percentage score calculation error")
        
        return len(issues) == 0, issues

# ========== Prompt Template =====================================================
PROMPT_TMPL = """
You are a professional answer reviewer. Please score the following answer based on the following 5 dimensions:
1. Logic   2. Depth   3. Innovation   4. Accuracy   5. Completeness
Each dimension is scored out of 5, with a total score of 25.

Scoring format example (strictly copy the numbers and spaces):
15 3 3 3 3 3
(Follow this line with your scoring reasons in paragraphs)

### Question
{question}

### Answer
{answer}

### Output Requirements
- The first line should **only contain 6 numbers**, separated by spaces, in the following order: total score, logic, depth, innovation, accuracy, completeness
- Do not write any text, units, or punctuation
- Start writing detailed scoring reasons from the second line (at least 2 paragraphs)

1. Logic —— Whether the argument structure and causal chain are rigorous;  
2. Depth —— Whether academic concepts/data/multi-angle analysis are cited;  
3. Innovation —— Whether new viewpoints or insights that are not clichés are proposed;  
4. Accuracy —— Whether facts, data, and concepts are correct;  
5. Completeness —— Whether all key points of the question are fully addressed.

**Strict scoring rules** (must be followed, please be cautious with high scores):  
| Single dimension score | Evaluation criteria (examples) |  
|----------|-----------------|  
| 5 | Almost flawless, only minor details can be criticized |  
| 4  | 1–2 minor flaws |  
| 3  | Obvious flaws or missing key points |  
| 2  | Multiple flaws, more than 2 argument/fact errors |  
| 0–1  | Critical logic is invalid, or serious factual errors |
Strictly follow the format, now begin:
"""

# ---------------------------------------------------------------------------
def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Read JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return []

# ---------------------------------------------------------------------------
def load_existing_results(output_file: Path, logger: Logger) -> Tuple[Dict[str, Any] | None, set, List[str]]:
    """
    Load existing grading progress and check quality
    Returns: (complete data, set of valid completions, list of questions needing re-grading)
    """
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            detailed_results = data.get("detailed_results", [])
            valid_done = set()
            need_regrade = []
            
            logger.info(f"Checking existing grading quality...")
            
            for result in detailed_results:
                question = result["question"]
                is_valid, issues = ScoreValidator.validate_grading_result(result)
                
                if is_valid:
                    # Check consistency of multiple scores
                    if "all_scores" in result:
                        consistency_valid, consistency_issues = ScoreValidator.validate_multiple_scores(result["all_scores"])
                        if not consistency_valid:
                            is_valid = False
                            issues.extend(consistency_issues)
                
                if is_valid:
                    valid_done.add(question)
                else:
                    need_regrade.append(question)
                    logger.warning(f"Question '{question[:40]}...' needs re-grading: {', '.join(issues)}")
            
            logger.info(f"Grading quality check completed:")
            logger.info(f"  · Valid grades: {len(valid_done)}")
            logger.info(f"  · Need re-grading: {len(need_regrade)}")
            
            return data, valid_done, need_regrade
            
        except Exception as e:
            logger.error(f"Failed to read progress file: {e}")
    
    return None, set(), []

# ---------------------------------------------------------------------------
def save_progress(data: Dict[str, Any], output_file: Path, logger: Logger):
    """Save progress with backup mechanism"""
    try:
        # Create backup
        if output_file.exists():
            backup_file = output_file.with_suffix('.backup.json')
            output_file.rename(backup_file)
        
        # Save new file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.success(f"Progress saved to {output_file}")
        
        # Delete backup
        if output_file.with_suffix('.backup.json').exists():
            output_file.with_suffix('.backup.json').unlink()
            
    except Exception as e:
        logger.error(f"Saving failed: {e}")
        # Restore backup
        backup_file = output_file.with_suffix('.backup.json')
        if backup_file.exists():
            backup_file.rename(output_file)
            logger.info("Restored from backup")

# ---------------------------------------------------------------------------
def parse_response(raw: str, logger: Logger) -> Tuple[Dict[str, int], str]:
    """Parse GPT output with enhanced error handling"""
    keys = ["total", "logic", "depth", "innovation", "accuracy", "completeness"]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Find score line
    score_line = None
    for line in lines:
        # Try to find line containing 6 numbers
        numbers = re.findall(r'\b\d+\b', line)
        if len(numbers) >= 6:
            score_line = line
            break
    
    if not score_line:
        raise ValueError("Could not find complete score line")
    
    nums = list(map(int, re.findall(r'\b\d+\b', score_line)[:6]))
    
    # Validate scores
    scores = dict(zip(keys, nums))
    is_valid, issues = ScoreValidator.validate_single_score(scores)
    if not is_valid:
        logger.warning(f"Score validation failed: {', '.join(issues)}")
        raise ValueError(f"Score validation failed: {', '.join(issues)}")

    # Extract commentary
    score_line_idx = lines.index(score_line)
    commentary = "\n".join(lines[score_line_idx + 1:]).strip()
    if not commentary:
        raise ValueError("Missing scoring reasons")

    return scores, commentary

# ---------------------------------------------------------------------------
def ask_and_parse(prompt: str,
                  logger: Logger,
                  model: str = "gpt-4o",
                  max_attempts: int = 6,
                  backoff_base: int = 2):
    """Call API and parse results with enhanced error handling"""
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
            logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}, retrying in {wait}s")
            time.sleep(wait)
    return None

# ---------------------------------------------------------------------------
def grade_single(question: str, answer: str, logger: Logger, trials: int = 3, max_retries: int = 2):
    """
    Score a single question with retry mechanism
    max_retries: Maximum number of retry rounds if all trials fail
    """
    prompt = PROMPT_TMPL.format(question=question, answer=answer)
    
    for retry in range(max_retries + 1):
        if retry > 0:
            logger.info(f"Retry round {retry + 1}...")
            time.sleep(5 * retry)  # Increasing wait time
        
        all_scores, all_cmts, raws = [], [], []
        
        for t in range(trials):
            res = ask_and_parse(prompt, logger)
            if not res:
                logger.warning(f"  Trial {t+1} failed")
                continue
            
            score, cmt, raw = res
            all_scores.append(score)
            all_cmts.append(cmt)
            raws.append(raw)
            logger.info(f"  Trial {t+1} score: {score['total']}/50")
        
        # Check scoring consistency
        if len(all_scores) >= MIN_VALID_TRIALS:
            is_valid, issues = ScoreValidator.validate_multiple_scores(all_scores)
            if is_valid:
                # Calculate average scores
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
                logger.warning(f"Scoring consistency check failed: {', '.join(issues)}")
    
    logger.error(f"Failed to obtain valid scores after {max_retries + 1} retry rounds")
    return None

# ---------------------------------------------------------------------------
def get_field_value(record: Dict[str, Any], fields: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Get field value from record, trying different field names by priority
    Returns: (field value, field name used)
    """
    for field in fields:
        if field in record and record[field] and record[field].strip():
            return record[field], field
    return None, None

# ---------------------------------------------------------------------------
def grade_replies(records: List[Dict[str, Any]], logger: Logger):
    """Score reply fields, supporting multiple field names"""
    logger.info(f"===== Scoring {'/'.join(FIELDS_TO_GRADE)} fields =====")

    # Output file name
    prev, valid_done_set, need_regrade = load_existing_results(OUTPUT_FILE, logger)
    
    # Get existing valid results
    existing_results = []
    if prev:
        for result in prev.get("detailed_results", []):
            if result["question"] not in need_regrade:
                existing_results.append(result)
    
    # Filter valid data and record used fields
    items = []
    field_usage = {}  # Record which field was used for each question
    
    for record in records:
        value, field_used = get_field_value(record, FIELDS_TO_GRADE)
        if value:
            items.append(record)
            field_usage[record["question"]] = field_used
    
    if not items:
        logger.error(f"No valid data containing {'/'.join(FIELDS_TO_GRADE)} fields found")
        return
    
    # Count field usage
    field_counts = {}
    for field in field_usage.values():
        field_counts[field] = field_counts.get(field, 0) + 1
    
    logger.info("Field usage statistics:")
    for field, count in field_counts.items():
        logger.info(f"  · {field}: {count} questions")

    # Calculate pending items
    # 1. Brand new questions
    new_questions = [d for d in items if d["question"] not in valid_done_set and d["question"] not in need_regrade]
    # 2. Questions needing re-grading
    regrade_questions = [d for d in items if d["question"] in need_regrade]
    
    pending = new_questions + regrade_questions
    
    logger.info(f"Data statistics:")
    logger.info(f"  · Total questions: {len(items)}")
    logger.info(f"  · Successfully completed: {len(valid_done_set)}")
    logger.info(f"  · Need re-grading: {len(regrade_questions)}")
    logger.info(f"  · New questions: {len(new_questions)}")
    logger.info(f"  · Total pending: {len(pending)}")

    # Main loop
    results = existing_results.copy()
    all_totals = [r["avg_scores"]["total"] for r in results]
    all_totals100 = [r["avg_score_100"] for r in results]
    
    # Statistics by field
    field_stats = {field: {"count": 0, "total_score": 0} for field in FIELDS_TO_GRADE}
    
    failed_questions = []
    regraded_count = 0

    for idx, item in enumerate(pending, 1):
        q = item["question"]
        field_used = field_usage[q]
        a = item[field_used]
        
        is_regrade = q in need_regrade
        
        if is_regrade:
            logger.info(f"\n🔄 [{idx}/{len(pending)}] Re-grading ({field_used}): {q[:40]}...")
            regraded_count += 1
        else:
            logger.info(f"\n[{idx}/{len(pending)}] Grading ({field_used}): {q[:40]}...")
        
        res = grade_single(q, a, logger)
        
        if res:
            res["field_graded"] = field_used  # Record actual field used
            res["is_regraded"] = is_regrade
            res["grading_timestamp"] = datetime.now().isoformat()
            
            results.append(res)
            all_totals.append(res["avg_scores"]["total"])
            all_totals100.append(res["avg_score_100"])
            
            # Update field statistics
            field_stats[field_used]["count"] += 1
            field_stats[field_used]["total_score"] += res["avg_scores"]["total"]
            
            # Validate result again
            is_valid, issues = ScoreValidator.validate_grading_result(res)
            if not is_valid:
                logger.warning(f"Grading result validation failed: {', '.join(issues)}")
        else:
            failed_questions.append({
                "question": q,
                "field_used": field_used,
                "reason": "Could not obtain valid score",
                "timestamp": datetime.now().isoformat()
            })

        # Save periodically
        if idx % SAVE_INTERVAL == 0:
            # Calculate average scores by field
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

    # Final statistics and save
    if all_totals:
        # Calculate final average scores by field
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
        
        # Save failed records
        if failed_questions:
            final_data["failed_questions"] = failed_questions
            
            # Save failed records in separate file
            failed_file = Path(OUTPUT_DIR) / f"failed_grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_questions, f, ensure_ascii=False, indent=2)
            logger.warning(f"Failed records saved to: {failed_file}")
        
        save_progress(final_data, OUTPUT_FILE, logger)
        
        logger.success(f"\n📊 Grading completed!")
        logger.success(f"  · Total average: {stats['total_average']}/50 ({stats['total_average_100']} points)")
        logger.success(f"  · Valid grades: {stats['valid_grades']}")
        logger.success(f"  · Failed: {stats['failed_grades']}")
        logger.success(f"  · Re-graded: {stats['regraded_count']}")
        
        # Display field statistics
        logger.success(f"\n📈 Field statistics:")
        for field, field_stat in field_averages.items():
            logger.success(f"  · {field}: {field_stat['count']} questions, average {field_stat['average']}/50 ({field_stat['average_100']} points)")

# ---------------------------------------------------------------------------
def main():
    logger = Logger(LOG_FILE)
    logger.info("Starting grading task")
    
    data = read_json_file(INPUT_PATH)
    if not data:
        logger.error("Failed to read data file")
        return

    # Score reply fields (support multiple field names)
    grade_replies(data, logger)
    
    # Calculate total time elapsed
    elapsed = datetime.now() - logger.start_time
    logger.info(f"Task completed, total time elapsed: {elapsed}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 Input file: {INPUT_PATH}")
    print(f"📝 Log file: {LOG_FILE}")
    print("-" * 60)
    
    main()
