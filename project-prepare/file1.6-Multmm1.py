import json, csv, pandas as pd
from pathlib import Path



#从表格中读取需要的top模型名称信息，对prompt进行整理
# === 文件路径 ===
xlsx_path  = Path(r"D:\project\model_comparison_separated.xlsx")
json_path  = Path(r"D:\project\grouped_answers.json")
output_csv = Path(r"D:\project\final_prompt_contexts.csv")

QUESTION_SHEET = "Question & Domains"
BASIC_SHEET    = "Basic Model Scores"

def _top_n(df, metric, n):                     # 取前 n 名模型
    return df.sort_values(metric, ascending=False)["Model"].head(n).tolist()

# ---------- 1. Excel ----------
wb   = pd.ExcelFile(xlsx_path)
q_df = (wb.parse(QUESTION_SHEET)
          .rename(columns=str.strip)
          .sort_values("ID", ignore_index=True))
questions = q_df["Question"].tolist()

basic_df = wb.parse(BASIC_SHEET)
metric   = "total_avg" if "total_avg" in basic_df.columns else "total"
top2_models = _top_n(basic_df, metric, 2)
top3_models = _top_n(basic_df, metric, 3)
top4_models = _top_n(basic_df, metric, 4)

# ---------- 2. json ----------
with open(json_path, encoding="utf-8") as f:
    answers_data = json.load(f)

def build_records(model_list, tag):
    records = []
    for q in questions:
        m2a = {m: a for m, a in answers_data.get(q, {}).get("basic_answers", [])}
        valid = [m2a[m] for m in model_list if m in m2a and m2a[m].strip()]
        if not valid:
            continue
        ctx = "\n".join(f"回答{i+1}：{ans}" for i, ans in enumerate(valid))
        prompt = f"请回答：\"{q}\"，请结合以下回答对你的答案进行完善：{ctx}。"
        records.append({
            "question": q,
            "prompt": prompt,
            "model": ",".join(model_list),
            "version": tag
        })
    return records

rows_top2 = build_records(top2_models, "top2")
rows_top3 = build_records(top3_models, "top3")
rows_top4 = build_records(top4_models, "top4")   # 新增 Top-4

# ---------- 3. 写出 CSV ----------
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "prompt", "model", "version"])
    writer.writeheader()
    writer.writerows(rows_top2 + rows_top3 + rows_top4)   # 按 2→3→4 顺序

print(
    f"✅ 已保存 {len(rows_top2)} 条 Top-2 + "
    f"{len(rows_top3)} 条 Top-3 + "
    f"{len(rows_top4)} 条 Top-4 到 {output_csv}"
)
