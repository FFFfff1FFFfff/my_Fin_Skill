"""SealQA data loader - supports local file or HuggingFace"""

import json
import os
from typing import Optional


def load_from_jsonl(file_path: str, limit: Optional[int] = None) -> list[dict]:
    """Load from local JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line.strip())
            samples.append({
                "id": item.get("id", i),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })
    return samples


def load_from_huggingface(name: str = "seal_0", limit: Optional[int] = None) -> list[dict]:
    """Load from HuggingFace datasets."""
    from datasets import load_dataset
    dataset = load_dataset("vtllms/sealqa", name=name, split="test")

    samples = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break
        samples.append({
            "id": item.get("id", i),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        })
    return samples


def load_sealqa(source: str = "seal_0", limit: Optional[int] = None) -> list[dict]:
    """
    Load SealQA dataset.

    Args:
        source: "seal_0", "seal_hard", "longseal" (HuggingFace) or path to local file
        limit: Optional limit on number of samples

    Returns:
        List of samples with keys: id, question, answer
    """
    if source in ["seal_0", "seal_hard", "longseal"]:
        return load_from_huggingface(name=source, limit=limit)
    elif os.path.isfile(source):
        return load_from_jsonl(source, limit)
    else:
        raise ValueError(f"Unknown source: {source}")


# Sample data for testing
SAMPLE_DATA = [
    {
        "id": "test_1",
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "id": "test_2",
        "question": "Who won the 2024 US Presidential Election?",
        "answer": "Donald Trump"
    },
    {
        "id": "test_3",
        "question": "What year was the iPhone first released?",
        "answer": "2007"
    },
]


def load_sample_data() -> list[dict]:
    """Load sample data for testing."""
    return SAMPLE_DATA.copy()


if __name__ == "__main__":
    print("Testing with sample data...")
    samples = load_sample_data()
    print(f"Loaded {len(samples)} samples")
    for s in samples:
        print(f"  Q: {s['question']} -> A: {s['answer']}")
