#!/usr/bin/env python
# coding: utf-8
"""
fusion_reply_grade.py
------------------------------------
读取融合答案 JSON，对 fusion_reply 自动打分并持续保存进度。
"""

import json, re, os, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import httpx
from openai import OpenAI
from tqdm import tqdm

# ========== 文件路径配置 (方便修改) ==========================================
INPUT_PATH = r"D:\project7\MM\result\2+1\doubao-pro-32k_answers_2+1-2-7800-8100.json"
OUTPUT_DIR = r"D:\project7\MM\result"
OUTPUT_FILENAME = "grades_doubao-pro-256k_answers_2+1-2-7800-8100.json"  # 自定义输出文件名

# ========== 其他配置选项 =====================================================
SAVE_INTERVAL = 1  # 每 N 题保存一次

# ========== OpenAI 初始化 ====================================================
httpx_client = httpx.Client(verify=False)
os.environ["OPENAI_API_KEY"]  = "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa"
os.environ["OPENAI_BASE_URL"] = "https://vir.vimsai.com/v1"
client = OpenAI(http_client=httpx_client)

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

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
def load_existing_results(output_file: Path) -> Tuple[Dict[str, Any] | None, set]:
    """加载已有评分进度"""
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            done = {r["question"] for r in data.get("detailed_results", [])}
            print(f"📂 已有进度：{len(done)} 题")
            return data, done
        except Exception as e:
            print(f"⚠️ 读取进度文件失败: {e}")
    return None, set()

# ---------------------------------------------------------------------------
def save_progress(data: Dict[str, Any], output_file: Path):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 进度已保存至 {output_file}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

# ---------------------------------------------------------------------------
def parse_response(raw: str) -> Tuple[Dict[str, int], str]:
    """解析 GPT 输出"""
    keys = ["total", "logic", "depth", "innovation", "accuracy", "completeness"]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # 找分数字串
    score_line = next((l for l in lines if len(re.findall(r"\d+", l)) >= 6), None)
    if not score_line:
        raise ValueError("找不到完整分数行")
    nums = list(map(int, re.findall(r"\d+", score_line)[:6]))

    commentary = "\n".join(lines[lines.index(score_line) + 1:]).strip()
    if not commentary:
        raise ValueError("缺少评分理由")

    return dict(zip(keys, nums)), commentary

# ---------------------------------------------------------------------------
def ask_and_parse(prompt: str,
                  model: str = "gpt-4o",
                  max_attempts: int = 6,
                  backoff_base: int = 2):
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
            scores, detail = parse_response(raw)
            return scores, detail, raw
        except Exception as e:
            wait = backoff_base ** attempt
            print(f"⚠️ 第 {attempt}/{max_attempts} 次失败: {e}，{wait}s 后重试")
            time.sleep(wait)
    return None

# ---------------------------------------------------------------------------
def grade_single(question: str, answer: str, trials: int = 3):
    prompt = PROMPT_TMPL.format(question=question, answer=answer)
    all_scores, all_cmts, raws = [], [], []

    for t in range(trials):
        res = ask_and_parse(prompt)
        if not res:
            print(f"  第 {t+1} 次评分失败")
            continue
        score, cmt, raw = res
        all_scores.append(score); all_cmts.append(cmt); raws.append(raw)
        print(f"  第 {t+1} 次得分：{score['total']}/50")

    if not all_scores:
        return None

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
        "all_gpt_raws": raws
    }

# ---------------------------------------------------------------------------
def grade_fusion_replies(records: List[Dict[str, Any]]):
    print(f"\n===== 评分 fusion_reply =====")
    
    # 使用配置的输出文件名
    output_file = Path(OUTPUT_DIR) / OUTPUT_FILENAME
    
    prev, done_set = load_existing_results(output_file)
    results = prev.get("detailed_results", []) if prev else []

    # 筛选有 fusion_reply 的记录
    items = [d for d in records if "fusion_reply" in d and d["fusion_reply"]]
    pending = [d for d in items if d["question"] not in done_set]
    
    print(f"共有 {len(items)} 题 | 待评分 {len(pending)} 题")

    # 主循环
    all_totals, all_totals100 = [], []

    # 补入旧成绩
    if prev:
        all_totals = [r["avg_scores"]["total"] for r in results]
        all_totals100 = [r["avg_score_100"] for r in results]

    for idx, item in enumerate(pending, 1):
        q = item["question"]
        a = item["fusion_reply"]
        
        print(f"\n[{idx}/{len(pending)}] {q[:40]}...")
        res = grade_single(q, a)
        
        if res:
            # 保存额外的元数据
            res["type"] = "fusion_reply"
            if "third_model" in item:
                res["third_model"] = item["third_model"]
            if "A1_third_answer" in item:
                res["has_third_answer"] = True
            if "A2_combination_reply" in item:
                res["has_combination_reply"] = True
                
            results.append(res)
            all_totals.append(res["avg_scores"]["total"])
            all_totals100.append(res["avg_score_100"])

        if idx % SAVE_INTERVAL == 0:
            stats = {
                "type": "fusion_reply",
                "input_file": INPUT_PATH,
                "total_questions": len(items),
                "valid_grades": len(all_totals),
                "total_average": round(sum(all_totals)/len(all_totals), 2),
                "total_average_100": round(sum(all_totals100)/len(all_totals100), 2)
            }
            save_progress({"statistics": stats, "detailed_results": results}, output_file)

    # 最终统计
    if all_totals:
        stats = {
            "type": "fusion_reply",
            "input_file": INPUT_PATH,
            "total_questions": len(items),
            "valid_grades": len(all_totals),
            "total_average": round(sum(all_totals)/len(all_totals), 2),
            "total_average_100": round(sum(all_totals100)/len(all_totals100), 2)
        }
        save_progress({"statistics": stats, "detailed_results": results}, output_file)
        print(f"\n📊 fusion_reply 平均 {stats['total_average']}/50 "
              f"(百分制 {stats['total_average_100']})")

# ---------------------------------------------------------------------------
def main():
    data = read_json_file(INPUT_PATH)
    if not data:
        return
    
    grade_fusion_replies(data)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()