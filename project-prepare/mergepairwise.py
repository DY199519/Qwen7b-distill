import json

# 输入文件路径
file2 = 'pairwise_grades_retry_basic.json'
file1 = 'pairwise_grades_retry_basic2.json'
output_file = 'merged_output_context_all.json'

# 读取第一个 JSON 文件
with open(file1, 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

# 读取第二个 JSON 文件
with open(file2, 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# 合并数据，优先保留 data1 中的问题
merged_data = dict(data1)
added_count = 0

for key, value in data2.items():
    if key not in merged_data:
        merged_data[key] = value
        added_count += 1

# 总保留问题数
total_questions = len(merged_data)

# 保存合并后的数据
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(merged_data, out_f, indent=2, ensure_ascii=False)

# 输出统计信息
print(f"合并完成，输出文件为：{output_file}")
print(f"总共保留了 {total_questions} 个唯一问题。")
print(f"其中从第二个文件中新增了 {added_count} 个问题。")
