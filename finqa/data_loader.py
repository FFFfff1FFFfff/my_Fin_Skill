"""FinQA data loader - supports local JSON file"""

import json
import os
from typing import Optional


def load_from_json(file_path: str, limit: Optional[int] = None) -> list[dict]:
    """Load from local JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "data" in data:
        items = data["data"]
    else:
        items = [data]

    samples = []
    for i, item in enumerate(items):
        if limit and len(samples) >= limit:
            break
        samples.append({
            "id": item.get("id", i),
            "question": item.get("question", ""),
            "context": item.get("context", ""),
            "answer": item.get("answer", ""),
        })
    return samples


def load_finqa(source: str, limit: Optional[int] = None) -> list[dict]:
    """
    Load FinQA dataset.

    Args:
        source: Path to local JSON file
        limit: Optional limit on number of samples

    Returns:
        List of samples with keys: id, question, context, answer
    """
    if os.path.isfile(source):
        return load_from_json(source, limit)
    else:
        raise ValueError(f"File not found: {source}")


# Sample data for testing
SAMPLE_DATA = [
    {
        "id": "test_1",
        "question": "What is the percentage change in revenue from 2019 to 2020?",
        "context": "Revenue 2019: $5,735 million\nRevenue 2020: $5,829 million",
        "answer": "1.64"
    },
    {
        "id": "test_2",
        "question": "What percentage of total facilities are leased?",
        "context": "Leased facilities: 140\nTotal facilities: 1,000",
        "answer": "14"
    },
    {
        "id": "test_3",
        "question": "What is the total revenue for 2018, 2019, and 2020?",
        "context": "Revenue 2018: $5,500 million\nRevenue 2019: $5,735 million\nRevenue 2020: $5,829 million",
        "answer": "17064"
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
        print(f"  Q: {s['question'][:50]}... -> A: {s['answer']}")
