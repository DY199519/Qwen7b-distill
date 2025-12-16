import csv, json, time
from pathlib import Path
from openai import OpenAI

# === 加载已有 basic answers 数据 ===
def load_existing_answers(grouped_json_path: Path) -> dict:
    if not grouped_json_path.exists():
        return {}
    with grouped_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for q, entry in data.items():
        for model, answer in entry.get("basic_answers", []):
            if q not in result:
                result[q] = {}
            result[q][model] = answer
    return result

def run_model_batch(
    model_name: str,
    api_key: str,
    base_url: str,
    csv_path: Path,
    output_json: Path,
    grouped_answers_path: Path
):
    # 初始化客户端
    api_client = OpenAI(api_key=api_key, base_url=base_url)
    existing_answers = load_existing_answers(grouped_answers_path)

    # === 读取 CSV → version→prompt ===
    groups, questions = {"top2": {}, "top3": {}, "top4": {}}, set()
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ver = row.get("version", "").lower()
            q   = row["question"]
            if ver in groups:
                groups[ver][q] = row["prompt"]
                questions.add(q)

    questions = sorted(list(questions))
    print(f"📌 模型 {model_name} - 处理问题总数: {len(questions)}")

    def ask_model(prompt: str, max_retry: int = 3, retry_pause: float = 2.0) -> str:
        for attempt in range(1, max_retry + 1):
            try:
                resp = api_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = resp.choices[0].message.content.strip()
                if answer:
                    return answer
                else:
                    print(f"⚠️ 第 {attempt} 次返回空内容，重试…")
            except Exception as e:
                print(f"❌ 第 {attempt} 次请求失败: {e}，重试…")
            time.sleep(retry_pause)
        return ""

    result_list = []
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] 处理问题 → {q[:60]}…")

        # 如果有缓存则直接用
        direct_prompt = q
        direct_reply = existing_answers.get(q, {}).get(model_name, "")
        if direct_reply:
            print(f"✅ 使用已有回答")
        else:
            direct_reply = ask_model(direct_prompt)
            time.sleep(1.2)

        def answer_with_tag(tag):
            ptxt = groups[tag].get(q, "")
            if not ptxt:
                return ptxt, ""
            print(f"   ↳ {tag} 回答…")
            ans = ask_model(ptxt)
            time.sleep(1.2)
            return ptxt, ans

        top2_prompt, top2_reply = answer_with_tag("top2")
        top3_prompt, top3_reply = answer_with_tag("top3")
        top4_prompt, top4_reply = answer_with_tag("top4")

        result_list.append({
            "question":      q,
            "direct_prompt": direct_prompt,
            "direct_reply":  direct_reply,
            "top2_prompt":   top2_prompt,
            "top2_reply":    top2_reply,
            "top3_prompt":   top3_prompt,
            "top3_reply":    top3_reply,
            "top4_prompt":   top4_prompt,
            "top4_reply":    top4_reply
        })

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 模型 {model_name} 已写入 {len(result_list)} 条数据到 {output_json}")


# === 配置 ===
csv_file = Path(r"D:\project\final_prompt_contexts.csv")
grouped_answers_file = Path("D:\project\grouped_answers.json")

model_configs = [
    {
        "model_name": "deepseek-v3",
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1",
        "output_json": Path(r"D:\project\deepseek_combined_answers.json")
    },
    {
        "model_name": "gemini-2.5-flash-preview-04-17-thinking",
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/",
        "output_json": Path(r"D:\project\gemini_combined_answers.json")
    },
    {
        "model_name": "qwen2.5-72b-instruct",
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1",
        "output_json": Path(r"D:\project\qwen_combined_answers.json")
    }
]

# === 执行每个模型任务 ===
for cfg in model_configs:
    run_model_batch(
        model_name=cfg["model_name"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        csv_path=csv_file,
        output_json=cfg["output_json"],
        grouped_answers_path=grouped_answers_file
    )
