#!/usr/bin/env python
# coding: utf-8
"""
multmm2_run2plus2_topic_first_modified_v2_simplified.py
----------------------------------------
简化版：只处理第一组回答组合
combination_1_A1A2_reply 作为 A1
combination_1_A1A3_reply 作为 A2
修改版：添加答案质量检查功能
"""

import json, time
from pathlib import Path
from openai import OpenAI

# ===== 0. 路径配置 ===========================================================
BASE_DIR = Path(r"D:\project7\MM\result")
BASE_DIR_1 = Path(r"D:\project7\prompt")

OUTPUT_DIR = Path(r"D:\project7\MM\result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件定义（移到最前面）
OUTPUT_FILE = OUTPUT_DIR / "answer-2+2-2.json"

# 读取包含各种回答的 JSON 文件
ANSWER_FILE = BASE_DIR / "answer-2+2-1.json"
PROMPT_FILE = BASE_DIR_1 / "prompt-2+2-2.txt"
SAVE_INTERVAL = 10  # 每 N 题保存一次

# 答案质量检查参数
MIN_ANSWER_LENGTH = 10  # 最小字数要求
VALID_END_PUNCTUATION = {'.', '。', '!', '！', '?', '？', ')', '）', '"', '"', "'", "'"}  # 有效的结尾标点

# ===== 1. 模型账户配置 =======================================================
MODEL_CFGS = [
    {
        "model_name": "deepseek-v3",
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1",
    }
]

# ===== 2. 读取提示词模板 ====================================================
def load_prompt_template(template_path: Path):
    """读取提示词模板文件"""
    try:
        with template_path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ 读取提示词模板失败: {e}")
        # 如果读取失败，使用默认模板
        return """阅读 其他两个模型的总结回答A2 并完善您之前的回答A1。如仍有空缺，可补充你的常识或公开资料，并用括号注明来源（常识／公开资料）。
问题：
{q}
A1：
{A1}
A2：
{A2}
【任务说明】  
不展示中间提取过程。前面不带任何铺垫性的语句
【输出要求】  
- 条理清晰，可使用编号或分段；  
- 避免赘述，保持简练。"""

# 加载提示词模板
PROMPT_TEMPLATE = load_prompt_template(PROMPT_FILE)

# ===== 3. 辅助函数 ==========================================================

def check_answer_quality(answer: str) -> tuple[bool, str]:
    """
    检查答案质量
    返回: (是否合格, 问题描述)
    """
    if not answer:
        return False, "答案为空"
    
    # 检查字数
    if len(answer.strip()) < MIN_ANSWER_LENGTH:
        return False, f"答案过短（少于{MIN_ANSWER_LENGTH}字）"
    
    # 检查结尾标点
    last_char = answer.strip()[-1] if answer.strip() else ''
    if last_char not in VALID_END_PUNCTUATION:
        return False, f"答案未以标点符号结尾（最后字符: '{last_char}'）"
    
    return True, "合格"

def ask(api: OpenAI, model: str, prompt: str, retry: int = 3, pause: int = 2):
    """调用模型API，并进行质量检查"""
    for i in range(retry):
        try:
            rsp = api.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            txt = rsp.choices[0].message.content.strip()
            
            # 检查答案质量
            is_valid, issue = check_answer_quality(txt)
            if not is_valid:
                print(f"  ⚠️ 答案质量问题: {issue}")
                if i < retry - 1:
                    print(f"  🔄 重试中...")
                    time.sleep(pause)
                    continue
                else:
                    print(f"  ❌ 多次尝试后仍有质量问题，使用当前结果")
            
            return txt
            
        except Exception as e:
            print(f"❌ {model} 第 {i+1} 次失败: {e}")
            time.sleep(pause)
    return ""

def load_progress(file: Path):
    """加载已处理的进度"""
    if not file.exists():
        return {}
    try:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {row["question"]: row for row in data}
    except Exception as e:
        print(f"⚠️ 读取进度失败: {e}")
        return {}

def save_progress(done_dict: dict, file: Path):
    """保存进度"""
    try:
        rows = list(done_dict.values())
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(file)
        print(f"💾 保存 {file.name} （{len(rows)} 条）")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def load_answer_data(answer_path: Path):
    """读取包含问题和回答的 JSON 文件"""
    data = {}
    
    if not answer_path.exists():
        print(f"❌ 文件不存在: {answer_path}")
        return data
        
    with answer_path.open("r", encoding="utf-8") as f:
        # 判断文件是 JSON 数组还是 JSON lines
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            entries = json.load(f)
        else:
            entries = [json.loads(line) for line in f if line.strip()]
    
    print(f"📖 读取到 {len(entries)} 条记录")
    
    for i, entry in enumerate(entries):
        try:
            q = entry["question"]
            
            # 从 answers 数组中提取回答
            answers = entry.get("answers", [])
            
            # 调试信息（只打印第一条）
            if i == 0:
                print(f"🔍 第一条记录的字段: {list(entry.keys())}")
                print(f"   ✓ answers 数量: {len(answers)}")
            
            # 确保有至少两个答案
            if len(answers) >= 2:
                # 提取前两个答案的 reply 作为 A1 和 A2
                a1 = answers[0].get("reply", "")
                a2 = answers[1].get("reply", "")
                
                if a1 and a2:
                    data[q] = {"A1": a1, "A2": a2}
                    
        except (KeyError, IndexError) as e:
            print(f"⚠️ 第 {i+1} 条记录处理出错: {e}")
            continue
    
    print(f"✅ 成功加载 {len(data)} 个问题的数据")
    return data

# ===== 4. 主处理函数 =========================================================
def process_questions(data, model_cfg):
    """处理所有问题"""
    model_name = model_cfg["model_name"]
    api = OpenAI(api_key=model_cfg["api_key"], base_url=model_cfg["base_url"])
    
    # 使用全局定义的输出文件
    done_dict = load_progress(OUTPUT_FILE)
    
    print(f"\n📊 {model_name} 已处理 {len(done_dict)} 题")
    
    processed = 0
    skipped = 0
    quality_issues = 0
    questions = sorted(data.keys())
    
    for qi, q in enumerate(questions, 1):
        print(f"\n📝 [{qi}/{len(questions)}] {q[:60]}...")
        
        # 已处理过则跳过
        if q in done_dict:
            print(f"  ⏭️ 已处理过，跳过")
            skipped += 1
            continue
        
        # 获取 A1 和 A2
        a1 = data[q]["A1"]
        a2 = data[q]["A2"]
        
        # 构建 prompt
        prompt = PROMPT_TEMPLATE.format(q=q, A1=a1, A2=a2)
        
        # 调用模型
        print(f"  🤖 调用 {model_name}")
        reply = ask(api, model_name, prompt)
        
        # 检查融合回答的质量
        is_valid, issue = check_answer_quality(reply)
        
        # 保存结果
        item = {
            "question": q,
            "A1": a1,
            "A2": a2,
            "fusion_prompt": prompt,
            "fusion_reply": reply,
        }
        
        if not is_valid:
            quality_issues += 1
            item["quality_issue"] = issue
        
        done_dict[q] = item
        processed += 1
        
        # 定期保存
        if processed > 0 and processed % SAVE_INTERVAL == 0:
            print(f"\n💾 达到保存间隔，保存进度...")
            save_progress(done_dict, OUTPUT_FILE)
    
    # 最终保存
    save_progress(done_dict, OUTPUT_FILE)
    
    return processed, skipped, quality_issues

# ===== 5. 脚本入口 ==========================================================
if __name__ == "__main__":
    print(f"📁 输出文件: {OUTPUT_FILE}")
    print(f"📏 答案质量要求: 最少{MIN_ANSWER_LENGTH}字，需以标点符号结尾")
    
    # 检查模板文件
    if not PROMPT_FILE.exists():
        print(f"⚠️ 提示词模板文件不存在: {PROMPT_FILE}")
        print("📝 请创建 prompt-2+2-2.txt 文件，包含 {q}、{A1} 和 {A2} 占位符")
    else:
        print(f"✅ 已加载提示词模板: {PROMPT_FILE}")
    
    # 读取答案数据
    data = load_answer_data(ANSWER_FILE)
    
    if not data:
        print("❌ 没有找到有效数据，退出")
        exit(1)
    
    print(f"📚 总题目数: {len(data)}")
    
    # 处理每个模型
    total_quality_issues = 0
    for cfg in MODEL_CFGS:
        print(f"\n{'='*60}")
        print(f"🤖 开始处理模型: {cfg['model_name']}")
        print(f"{'='*60}")
        
        processed, skipped, quality_issues = process_questions(data, cfg)
        total_quality_issues += quality_issues
        
        print(f"\n🏁 {cfg['model_name']} 处理完成！")
        print(f"   - 新处理: {processed} 题")
        print(f"   - 跳过: {skipped} 题")
        print(f"   - 质量问题: {quality_issues} 个")
        print(f"   - 输出文件: {OUTPUT_FILE}")
    
    # 质量问题汇总
    if total_quality_issues > 0:
        print(f"\n⚠️ 总共发现 {total_quality_issues} 个答案质量问题")
        print("  可在输出文件中查看具体问题详情（quality_issue 字段）")
    
    print("\n🎉 全部处理完毕！")