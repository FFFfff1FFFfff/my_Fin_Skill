#!/usr/bin/env python3
"""Download SealQA dataset from HuggingFace"""

from datasets import load_dataset

print("Downloading SealQA from HuggingFace...")

# Load all splits
seal_0 = load_dataset("vtllms/sealqa", name="seal_0", split="test")
seal_hard = load_dataset("vtllms/sealqa", name="seal_hard", split="test")
longseal = load_dataset("vtllms/sealqa", name="longseal", split="test")

print(f"seal_0: {len(seal_0)} samples")
print(f"seal_hard: {len(seal_hard)} samples")
print(f"longseal: {len(longseal)} samples")

# Save as JSONL
seal_0.to_json("seal_0.jsonl")
seal_hard.to_json("seal_hard.jsonl")
longseal.to_json("longseal.jsonl")

print("\nSaved to: seal_0.jsonl, seal_hard.jsonl, longseal.jsonl")
