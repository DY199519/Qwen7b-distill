import json

# Input file paths
file2 = 'pairwise_grades_retry_basic.json'
file1 = 'pairwise_grades_retry_basic2.json'
output_file = 'merged_output_context_all.json'

# Read the first JSON file
with open(file1, 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

# Read the second JSON file
with open(file2, 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# Merge data, giving priority to retaining questions in data1
merged_data = dict(data1)
added_count = 0

for key, value in data2.items():
    if key not in merged_data:
        merged_data[key] = value
        added_count += 1

# Total number of retained questions
total_questions = len(merged_data)

# Save the merged data
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(merged_data, out_f, indent=2, ensure_ascii=False)

# Output statistical information
print(f"Merging completed, the output file is: {output_file}")
print(f"A total of {total_questions} unique questions were retained.")
print(f"Among them, {added_count} questions were added from the second file.")
