import json
import pandas as pd
from collections import defaultdict
from itertools import combinations

# === 指定评分 JSON 文件及对应模型标签 ===
json_files = {
    "Deepseek": "pairwise_grades_top234_deepseek.json",
    "Qwen": "pairwise_grades_top234_qwen.json",
    "gemini":"pairwise_grades_top234_gemini.json"
}

# === 加载题目与学科映射数据 ===
with open("generated_results_multi_model1.json", "r", encoding="utf-8") as f:
    domain_data_1 = json.load(f)

try:
    with open("generated_results_multi_model_updated.json", "r", encoding="utf-8") as f:
        content = f.read().strip()
        domain_data_2 = json.loads(content) if content else []
except (FileNotFoundError, json.JSONDecodeError):
    domain_data_2 = []

# === 提取题目 - 学科映射 ===
question_to_domains = {}
def extract_domains(data):
    for item in data:
        for r in item.get("results", []):
            q = r.get("core_question", [""])[0].strip()
            doms = r.get("core_question", ["", []])[1]
            question_to_domains[q] = [d.strip() for d in doms]

extract_domains(domain_data_1)
extract_domains(domain_data_2)

# === 提取评分数据 ===
def extract_pairs(data, key, tag):
    rows = []
    for q, content in data.items():
        for pair in content.get(key, []):
            if not all(k in pair for k in ["model_a", "model_b", "winner_order"]):
                continue
            rows.append({
                "Tag": tag,
                "question": q,
                "model_a": pair["model_a"],
                "scores_a": pair.get("scores_a", {}),
                "model_b": pair["model_b"],
                "scores_b": pair.get("scores_b", {}),
                "winner": pair["model_a"] if pair["winner_order"][0] == "A" else pair["model_b"]
            })
    return pd.DataFrame(rows)

# === 合并所有评分数据 ===
df_list = []
for tag, file in json_files.items():
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = extract_pairs(data, "top_pairs", tag)
    df_list.append(df)
df_all = pd.concat(df_list, ignore_index=True)

# === 所有问题与模型 ===
all_questions = list(dict.fromkeys(df_all["question"].tolist()))
all_models = sorted(set(df_all["model_a"]) | set(df_all["model_b"]))

# === 模型总得分统计 ===
def overall_scores(df):
    bucket = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        for m_key, s_key in [("model_a", "scores_a"), ("model_b", "scores_b")]:
            for dim, sc in row[s_key].items():
                bucket[(row["Tag"], row[m_key])][dim].append(sc)
    recs = []
    for (tag, m), dims in bucket.items():
        entry = {"Tag": tag, "Model": m}
        for dim in ["logic", "depth", "innovation", "accuracy", "completeness", "total"]:
            vals = dims.get(dim, [])
            entry[f"{dim}_avg"] = round(sum(vals)/len(vals), 2) if vals else 0
            entry[f"{dim}_sum"] = round(sum(vals), 2)
        recs.append(entry)
    return pd.DataFrame(recs)

scores_df = overall_scores(df_all)
scores_sorted = scores_df.sort_values(["Tag", "total_sum"], ascending=[True, False])

# === 总均分映射表 ===
def build_avg_map(df):
    return {(row["Tag"], row["Model"]): row["total_avg"] for _, row in df.iterrows()}

avg_map = build_avg_map(scores_sorted)

# === 学科统计表 ===
def domain_tables(df, avg_map):
    seen = set()
    d_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    d_counts = defaultdict(lambda: defaultdict(int))
    d_qcount = defaultdict(set)

    for _, row in df.iterrows():
        domains = question_to_domains.get(row["question"], [])
        for dom in domains:
            d_qcount[(row["Tag"], dom)].add(row["question"])
            for m_key, s_key in [("model_a", "scores_a"), ("model_b", "scores_b")]:
                model = row[m_key]
                key = (row["Tag"], dom, model, row["question"])
                if key in seen:
                    continue
                seen.add(key)
                for dim, sc in row[s_key].items():
                    d_totals[(row["Tag"], dom)][model][dim] += sc
                d_counts[(row["Tag"], dom)][model] += 1

    tables = {}
    for (tag, dom), mdict in d_totals.items():
        rows = []
        for model, dims in mdict.items():
            cnt = d_counts[(tag, dom)][model]
            row = {"Model": model}
            for dim in ["logic", "depth", "innovation", "accuracy", "completeness", "total"]:
                row[dim] = round(dims.get(dim, 0), 2)
            row["单学科均分"] = round(row["total"] / cnt, 2) if cnt else 0
            row["模型总均分"] = avg_map.get((tag, model), 0)
            row["Tag"] = tag
            rows.append(row)
        df_table = pd.DataFrame(rows).sort_values("total", ascending=False)
        df_table.loc[len(df_table.index)] = {
            "Model": f"共有 {len(d_qcount[(tag, dom)])} 道题目涉及“{dom}”学科"
        }
        tables[f"{tag}_{dom}"] = df_table
    return tables

def winrate_matrix(df, models):
    mat = pd.DataFrame(index=models, columns=models)
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                mat.loc[m1, m2] = "-"
                continue
            sub = df[((df["model_a"] == m1) & (df["model_b"] == m2)) |
                     ((df["model_a"] == m2) & (df["model_b"] == m1))]
            if sub.empty:
                mat.loc[m1, m2] = "N/A"
            else:
                wins = (sub["winner"] == m1).sum()
                mat.loc[m1, m2] = f"{wins/len(sub)*100:.1f}% ({wins})"
    return mat

# === 生成各类统计表 ===
domain_tables_all = domain_tables(df_all, avg_map)

# === 题目-学科表格 ===
question_domains_df = pd.DataFrame([{
    "ID": i+1,
    "Question": q,
    "Domains": ", ".join(question_to_domains.get(q, []))
} for i, q in enumerate(all_questions)])

# === 输出 TXT 文件（可选）===
with open("question_list_with_domains.txt", "w", encoding="utf-8") as f:
    f.write("【题目编号 - 问题 - 学科】\n")
    for i, q in enumerate(all_questions, 1):
        f.write(f"{i}. {q} —— 学科: {', '.join(question_to_domains.get(q, []))}\n")

# === 输出 Excel 文件 ===
with pd.ExcelWriter("model_comparison_all_tags.xlsx", engine="openpyxl") as w:
    scores_sorted.to_excel(w, sheet_name="Model Scores", index=False)
    question_domains_df.to_excel(w, sheet_name="Question & Domains", index=False)
    for name, df in domain_tables_all.items():
        df.to_excel(w, sheet_name=f"{name[:31]}", index=False)
    for tag in df_all["Tag"].unique():
        sub_df = df_all[df_all["Tag"] == tag]
        models = sorted(set(sub_df["model_a"]) | set(sub_df["model_b"]))
        matrix = winrate_matrix(sub_df, models)
        matrix.to_excel(w, sheet_name=f"{tag}_Winrate")

print("✅ 所有分析已完成，文件已写入 model_comparison_all_tags.xlsx")