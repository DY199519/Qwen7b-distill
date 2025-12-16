import json
import pandas as pd
from collections import defaultdict

# === 加载数据 ===
with open("pairwise_grades_retry_basic.json", "r", encoding="utf-8") as f:
    basic_data = json.load(f)

with open("pairwise_grades_retry_context.json", "r", encoding="utf-8") as f:
    context_data = json.load(f)

with open("generated_results_multi_model.json", "r", encoding="utf-8") as f:
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

# === 提取配对数据 ===
def extract_pairs(data, key):
    rows = []
    for q, content in data.items():
        for pair in content.get(key, []):
            if not all(k in pair for k in ["model_a", "model_b", "winner_order"]):
                continue
            rows.append(
                {"question": q,
                 "model_a": pair["model_a"], "scores_a": pair.get("scores_a", {}),
                 "model_b": pair["model_b"], "scores_b": pair.get("scores_b", {}),
                 "winner": pair["model_a"] if pair["winner_order"][0] == "A" else pair["model_b"]}
            )
    return pd.DataFrame(rows)

df_basic   = extract_pairs(basic_data,   "basic_pairs")
df_context = extract_pairs(context_data, "answer_pairs")

all_questions = list(dict.fromkeys(df_basic["question"].tolist() + df_context["question"].tolist()))
all_models = sorted(set(df_basic["model_a"]) | set(df_basic["model_b"]) |
                    set(df_context["model_a"]) | set(df_context["model_b"]))

# === 胜率矩阵 ===
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

matrix_basic   = winrate_matrix(df_basic,   all_models)
matrix_context = winrate_matrix(df_context, all_models)

# === 模型总得分 ===
def overall_scores(df):
    bucket = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        for m_key, s_key in [("model_a", "scores_a"), ("model_b", "scores_b")]:
            for dim, sc in row[s_key].items():
                bucket[row[m_key]][dim].append(sc)

    recs = []
    for m, dims in bucket.items():
        entry = {"Model": m}
        for dim in ["logic","depth","innovation","accuracy","completeness","total"]:
            vals = dims.get(dim, [])
            entry[f"{dim}_avg"] = round(sum(vals)/len(vals), 2) if vals else 0
            entry[f"{dim}_sum"] = round(sum(vals), 2)
        recs.append(entry)
    return pd.DataFrame(recs)

basic_scores_df   = overall_scores(df_basic)
context_scores_df = overall_scores(df_context)
basic_scores_sorted   = basic_scores_df.sort_values("total_sum",   ascending=False)
context_scores_sorted = context_scores_df.sort_values("total_sum", ascending=False)

# === 提取总均分映射 ===
def build_avg_map(df):
    return {row["Model"]: row["total_avg"] for _, row in df.iterrows()}

basic_avg_map   = build_avg_map(basic_scores_sorted)
context_avg_map = build_avg_map(context_scores_sorted)

# === 学科统计表（含总均分 & 涉及题目数） ===
def domain_tables(df, avg_map):
    seen = set()
    d_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    d_counts = defaultdict(lambda: defaultdict(int))
    d_qcount = defaultdict(set)

    for _, row in df.iterrows():
        domains = question_to_domains.get(row["question"], [])
        for dom in domains:
            d_qcount[dom].add(row["question"])
            for m_key, s_key in [("model_a","scores_a"), ("model_b","scores_b")]:
                model = row[m_key]
                key = (dom, model, row["question"])
                if key in seen:
                    continue
                seen.add(key)
                for dim, sc in row[s_key].items():
                    d_totals[dom][model][dim] += sc
                d_counts[dom][model] += 1

    tables = {}
    for dom, mdict in d_totals.items():
        rows = []
        for model, dims in mdict.items():
            cnt = d_counts[dom][model]
            row = {"Model": model}
            for dim in ["logic","depth","innovation","accuracy","completeness","total"]:
                row[dim] = round(dims.get(dim, 0), 2)
            row["单学科均分"] = round(row["total"] / cnt, 2) if cnt else 0
            row["模型总均分"] = avg_map.get(model, 0)
            rows.append(row)
        df_table = pd.DataFrame(rows).sort_values("total", ascending=False)
        # 添加一行：学科涉及题目数说明
        df_table.loc[len(df_table.index)] = {
            "Model": f"共有 {len(d_qcount[dom])} 道题目涉及“{dom}”学科"
        }
        tables[dom] = df_table
    return tables

basic_domain_tables   = domain_tables(df_basic,   basic_avg_map)
context_domain_tables = domain_tables(df_context, context_avg_map)

# === Question & Domains 表格 ===
question_domains_df = pd.DataFrame([
    {"ID": i+1, "Question": q, "Domains": ", ".join(question_to_domains.get(q, []))}
    for i, q in enumerate(all_questions)
])

# === 题目-学科 TXT 输出 ===
with open("question_list_with_domains.txt", "w", encoding="utf-8") as f:
    f.write("【题目编号 - 问题 - 学科】\n")
    for i, q in enumerate(all_questions, 1):
        f.write(f"{i}. {q} —— 学科: {', '.join(question_to_domains.get(q, []))}\n")

# === 写入 Excel 文件 ===
with pd.ExcelWriter("model_comparison_separated.xlsx", engine="openpyxl") as w:
    basic_scores_sorted.to_excel(w, sheet_name="Basic Model Scores", index=False)
    context_scores_sorted.to_excel(w, sheet_name="Context Model Scores", index=False)
    matrix_basic.to_excel(w,   sheet_name="Basic Winrate")
    matrix_context.to_excel(w, sheet_name="Context Winrate")
    question_domains_df.to_excel(w, sheet_name="Question & Domains", index=False)
    pd.DataFrame({"ID": range(1, len(all_questions)+1), "Question": all_questions}).to_excel(
        w, sheet_name="Question List", index=False)
    for dom, df in basic_domain_tables.items():
        df.to_excel(w, sheet_name=f"Basic_{dom[:25]}", index=False)
    for dom, df in context_domain_tables.items():
        df.to_excel(w, sheet_name=f"Context_{dom[:23]}", index=False)

print("✅ 所有工作表和学科页均已生成完成，包括总均分列与题目数量备注。")
