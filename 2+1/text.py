import json
import numpy as np

# 读取生成的数据
with open(r'D:\project7\alreadyfinish\2+1\alpaca_dataset_top700.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计长度
lengths = []
for item in data:
    total_len = len(item['instruction']) + len(item['input']) + len(item['output'])
    lengths.append(total_len)

# 转换为numpy数组以便统计
lengths = np.array(lengths)

print(f"样本总数: {len(lengths)}")
print(f"平均长度: {np.mean(lengths):.0f} 字符")
print(f"中位数长度: {np.median(lengths):.0f} 字符")
print(f"最短样本: {np.min(lengths)} 字符")
print(f"最长样本: {np.max(lengths)} 字符")
print(f"\n长度分布:")
print(f"<5000字符: {np.sum(lengths < 5000)} 个")
print(f"5000-10000字符: {np.sum((lengths >= 5000) & (lengths < 10000))} 个")
print(f"10000-15000字符: {np.sum((lengths >= 10000) & (lengths < 15000))} 个")
print(f"15000-20000字符: {np.sum((lengths >= 15000) & (lengths < 20000))} 个")
print(f">20000字符: {np.sum(lengths >= 20000)} 个")

# 估算token数（中文约1.5-2个字符/token）
estimated_tokens = lengths / 1.5
print(f"\n估算的token数:")
print(f"平均: {np.mean(estimated_tokens):.0f} tokens")
print(f"最大: {np.max(estimated_tokens):.0f} tokens")
print(f">14000 tokens的样本数: {np.sum(estimated_tokens > 14000)} 个")