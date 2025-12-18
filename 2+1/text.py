import json
import numpy as np

# Read JSON
with open(r'D:\project7\alreadyfinish\2+1\alpaca_dataset_top700.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Lengths
lengths = []
for item in data:
    total_len = len(item['instruction']) + len(item['input']) + len(item['output'])
    lengths.append(total_len)

# Transform to numpy
lengths = np.array(lengths)

print(f"Total number of samples: {len(lengths)}")
print(f"Average length: {np.mean(lengths):.0f} characters")
print(f"Median length: {np.median(lengths):.0f} characters")
print(f"Shortest sample: {np.min(lengths)} characters")
print(f"Longest sample: {np.max(lengths)} characters")
print(f"\nLength distribution:")
print(f"<5000 characters: {np.sum(lengths < 5000)}")
print(f"5000-10000 characters: {np.sum((lengths >= 5000) & (lengths < 10000))}")
print(f"10000-15000 characters: {np.sum((lengths >= 10000) & (lengths < 15000))}")
print(f"15000-20000 characters: {np.sum((lengths >= 15000) & (lengths < 20000))}")
print(f">20000 characters: {np.sum(lengths >= 20000)}")

# Estimate token count (Chinese is approximately 1.5-2 characters per token)
estimated_tokens = lengths / 1.5
print(f"\nEstimated number of tokens:")
print(f"Average: {np.mean(estimated_tokens):.0f} tokens")
print(f"Maximum: {np.max(estimated_tokens):.0f} tokens")

print(f"Number of samples with >14000 tokens: {np.sum(estimated_tokens > 14000)}")
