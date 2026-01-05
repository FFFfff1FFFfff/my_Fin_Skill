#!/usr/bin/env python3
"""Download TableBench dataset from HuggingFace and save as jsonl"""

from datasets import load_dataset

print("Downloading TableBench from HuggingFace...")
dataset = load_dataset("Multilingual-Multimodal-NLP/TableBench", split="test")

print(f"Downloaded {len(dataset)} samples")

# Save as jsonl
output_file = "tablebench.jsonl"
dataset.to_json(output_file)
print(f"Saved to {output_file}")

# Show sample distribution
from collections import Counter
qtypes = Counter(dataset["qtype"])
print("\nSample distribution by qtype:")
for qtype, count in qtypes.items():
    print(f"  {qtype}: {count}")
