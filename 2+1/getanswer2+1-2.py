#!/usr/bin/env python
# coding: utf-8
"""
multmm2_run2plus2_topic_first_modified.py
----------------------------------------
修改版：从 JSON 格式中提取 third_answer 作为 A1，combination_1_reply 作为 A2
使用新的提示词模板
添加答案质量检查功能
"""

import csv, json, time, re, traceback
from pathlib import Path
from openai import OpenAI
from datetime import datetime

# ===== 0. 输出路径和文件名配置 ======================================================
BASE_DIR   = Path(r"D:\project7\MM\result\2+1")
BASE_DIR_1   = Path(r"D:\project7\prompt")
OUTPUT_DIR = Path(r"D:\project7\MM\result\2+1")  # 可以在这里轻松修改输出路径
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 输出文件名配置 - 在这里修改！ ==========
# 方式1: 使用固定的后缀
OUTPUT_SUFFIX = "answers_2+1-2-7800-8100"  # 修改这个值来改变输出文件名

# 方式2: 使用时间戳
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# OUTPUT_SUFFIX = f"answers_{timestamp}"

# 方式3: 使用完全自定义的名称
# OUTPUT_FILENAME_TEMPLATE = "{model_name}_融合结果_v2.json"  # {model_name} 会被替换为模型名

# 方式4: 为每个模型单独指定输出文件名（在 MODEL_CFGS 中添加 output_filename 字段）
# ================================================

# 读取包含 combination_1_reply 和 third_answer 的 JSON 文件
ANSWER_FILE = BASE_DIR / "gemini-2.5-flash_answers_2+1-1-7800-8100.json"  # 包含问题、combination_1_reply 和 third_answer 的文件
PROMPT_FILE = BASE_DIR_1 / "prompt-2+1-2.txt"  # 提示词模板文件
SAVE_INTERVAL = 1  # 每 N 题保存一次

# ===== 1. 模型账户配置 =======================================================
MODEL_CFGS = [
    {
        "model_name": "doubao-pro-32k",
        "api_key": "sk-TlCq2TfX7oLuXzZMD1A3681285A2460bA26b6f0cEa5517Aa",
        "base_url": "https://vir.vimsai.com/v1",
        # "output_filename": "doubao_自定义输出.json"  # 可选：为特定模型指定输出文件名
    }
]

# ===== 2. 答案质量检查配置 ===================================================
MIN_ANSWER_LENGTH = 10  # 最小答案长度
VALID_ENDINGS = ['。', '！', '？', '.', '!', '?', ')', '）', '"', '"', "'", "'"]  # 有效的结尾标点
MAX_RETRIES = 3  # 最大重试次数

def check_answer_quality(answer: str, question: str = ""):
    """
    检查答案质量
    返回: (is_valid, error_message)
    """
    if not answer or not answer.strip():
        return False, "答案为空"
    
    answer = answer.strip()
    
    # 检查长度
    if len(answer) < MIN_ANSWER_LENGTH:
        return False, f"答案过短（{len(answer)}字符，最少需要{MIN_ANSWER_LENGTH}字符）"
    
    # 检查是否以合适的标点符号结尾
    if not any(answer.endswith(ending) for ending in VALID_ENDINGS):
        return False, f"答案可能被截断，结尾字符: '{answer[-1] if answer else 'N/A'}'"
    
    # 检查是否包含明显的截断标志
    truncation_signs = ['...', '……', '[未完成]', '[截断]', '(未完', '（未完']
    if any(sign in answer for sign in truncation_signs):
        return False, "答案包含截断标志"
    
    # 检查答案是否过于重复（可能是生成异常）
    words = answer.split()
    if len(words) > 5:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        max_freq = max(word_freq.values())
        if max_freq > len(words) * 0.5:  # 如果某个词出现超过50%
            return False, "答案内容过于重复"
    
    return True, "质量检查通过"

# ===== 3. 读取提示词模板 ====================================================
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

# ===== 4. 辅助函数 ==========================================================

def get_output_filename(model_name: str, cfg: dict):
    """根据配置生成输出文件名"""
    # 优先使用模型配置中的自定义文件名
    if "output_filename" in cfg:
        return cfg["output_filename"]
    
    # 使用全局模板（如果定义了）
    if 'OUTPUT_FILENAME_TEMPLATE' in globals():
        return OUTPUT_FILENAME_TEMPLATE.format(model_name=model_name)
    
    # 默认使用后缀方式
    return f"{model_name}_{OUTPUT_SUFFIX}.json"

def ask(api: OpenAI, model: str, prompt: str, retry: int = 3, pause: int = 2, question: str = ""):
    """调用API并进行质量检查"""
    for i in range(retry):
        try:
            rsp = api.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            txt = rsp.choices[0].message.content.strip()
            
            # 进行质量检查
            is_valid, error_msg = check_answer_quality(txt, question)
            
            if is_valid:
                print(f"    ✅ 答案质量检查通过（长度: {len(txt)}字符）")
                return txt
            else:
                print(f"    ⚠️ 第 {i+1} 次尝试质量检查失败: {error_msg}")
                if i < retry - 1:  # 如果不是最后一次尝试
                    print(f"    🔄 将重试...")
                    time.sleep(pause)
                    continue
                else:
                    print(f"    ❌ 达到最大重试次数，仍返回当前答案")
                    return txt  # 即使质量不佳也返回，避免完全失败
                    
        except Exception as e:
            print(f"❌ {model} 第 {i+1} 次API调用失败: {e}")
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
        print(f"⚠️ 读取进度失败: {e}")
        return {}

def save_progress(done_dict: dict, file: Path):
    """保存进度，只保存 done 字典中的记录"""
    try:
        rows = list(done_dict.values())
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(file)
        print(f"💾 保存 {file.name} （{len(rows)} 条）")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def load_answer_data(answer_path: Path):
    """读取包含问题、combination_1_reply 和 third_answer 的 JSON 文件"""
    q2data = {}
    
    # 检查文件是否存在
    if not answer_path.exists():
        print(f"❌ 文件不存在: {answer_path}")
        return q2data
        
    with answer_path.open("r", encoding="utf-8") as f:
        # 判断文件是 JSON 数组还是 JSON lines
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON 数组
            entries = json.load(f)
        else:
            # JSON lines
            entries = [json.loads(line) for line in f if line.strip()]
    
    print(f"📖 读取到 {len(entries)} 条记录")
    
    for i, entry in enumerate(entries):
        try:
            q = entry["question"]
            combination_reply = entry.get("combination_1_reply", "")
            third_answer = entry.get("third_answer", "")
            third_model = entry.get("third_model", "")
            
            # 调试信息
            if i == 0:  # 只打印第一条记录的字段信息
                print(f"🔍 第一条记录的字段: {list(entry.keys())}")
                if "third_answer" in entry:
                    print(f"   ✓ 找到 third_answer (长度: {len(third_answer)})")
                else:
                    print(f"   ❌ 未找到 third_answer")
                if "combination_1_reply" in entry:
                    print(f"   ✓ 找到 combination_1_reply (长度: {len(combination_reply)})")
                else:
                    print(f"   ❌ 未找到 combination_1_reply")
            
            # 只有当两个答案都存在时才加入
            if combination_reply and third_answer:
                q2data[q] = {
                    "combination_reply": combination_reply,  # A2
                    "third_answer": third_answer,  # A1
                    "third_model": third_model
                }
            else:
                if i < 3:  # 只打印前几条的警告信息
                    print(f"⚠️ 第 {i+1} 条记录缺少必要字段: combination_reply={bool(combination_reply)}, third_answer={bool(third_answer)}")
                    
        except KeyError as e:
            print(f"⚠️ 第 {i+1} 条记录缺少字段 {e}: {list(entry.keys())}")
            continue
    
    print(f"✅ 成功加载 {len(q2data)} 个问题的数据")
    return q2data

# ===== 5. 脚本入口 ==========================================================
if __name__ == "__main__":
    # 0) 检查模板文件是否存在
    if not PROMPT_FILE.exists():
        print(f"⚠️ 提示词模板文件不存在: {PROMPT_FILE}")
        print("📝 请创建 prompt-2+1-2.txt 文件，包含 {q}、{A1} 和 {A2} 占位符")
    else:
        print(f"✅ 已加载提示词模板: {PROMPT_FILE}")
        print(f"📋 答案质量检查配置: 最小长度={MIN_ANSWER_LENGTH}, 最大重试={MAX_RETRIES}")
    
    # 1) 读取答案数据
    q2data = load_answer_data(ANSWER_FILE)
    all_questions = sorted(q2data.keys())
    print(f"📚 题目数: {len(all_questions)}")
    
    # 2) 为每个模型准备 API、进度文件、行缓存
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
        # 打印已有进度
        existing_count = len(model_env[name]["done"])
        if existing_count > 0:
            print(f"📊 {name} 已有进度: {existing_count} 题")
        print(f"📄 {name} 输出文件: {output_filename}")
    
    processed = 0
    skipped = 0
    quality_failures = 0
    
    # ------- 主循环：题目优先 -----------------
    for qi, q in enumerate(all_questions, 1):
        print(f"\n📝 [{qi}/{len(all_questions)}] {q[:60]}…")
        
        # 获取该问题的数据
        data = q2data[q]
        a1 = data["third_answer"]  # third_answer 作为 A1
        a2 = data["combination_reply"]  # combination_1_reply 作为 A2
        source_model = data["third_model"]
        
        question_processed = False
        
        for cfg in MODEL_CFGS:
            mname = cfg["model_name"]
            env = model_env[mname]
            
            # 已有则跳过
            if q in env["done"]:
                print(f"  ⏭️ {mname} 已处理过，跳过")
                skipped += 1
                continue
            
            api = env["api"]
            
            print(f"  🤖 调用 {mname}")
            
            # 使用新模板构建 prompt
            prompt = PROMPT_TEMPLATE.format(q=q, A1=a1, A2=a2)
            
            # 调用模型（已包含质量检查）
            reply = ask(api, mname, prompt, question=q)
            
            # 记录质量检查结果
            if reply:
                is_valid, quality_msg = check_answer_quality(reply, q)
                if not is_valid:
                    quality_failures += 1
                    print(f"    ⚠️ 最终答案质量问题: {quality_msg}")
            
            # 保存结果
            item = {
                "question": q,
                "third_model": source_model,
                "A1_third_answer": a1,
                "A2_combination_reply": a2,
                "fusion_prompt": prompt,
                "fusion_reply": reply,
                "quality_check": check_answer_quality(reply, q)[1] if reply else "生成失败"
            }
            
            # 直接加入 done 字典，不使用 rows
            env["done"][q] = item
            question_processed = True
        
        if question_processed:
            processed += 1
            
        # ---- SAVE_INTERVAL ----
        if processed > 0 and processed % SAVE_INTERVAL == 0:
            print(f"\n💾 达到保存间隔，保存进度...")
            for mname, env in model_env.items():
                save_progress(env["done"], env["out"])
    
    # 3) 全部完成后保存一次
    print(f"\n🏁 处理完成！")
    print(f"📊 统计信息:")
    print(f"   - 新处理: {processed} 题")
    print(f"   - 跳过: {skipped} 题") 
    print(f"   - 质量问题: {quality_failures} 题")
    
    for mname, env in model_env.items():
        save_progress(env["done"], env["out"])
        print(f"✅ {mname} 总计 {len(env['done'])} 条记录")
    
    print(f"\n🎉 按题目顺序全部处理完毕，文件保存在: {OUTPUT_DIR}")