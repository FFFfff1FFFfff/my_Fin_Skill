"""FinQA data loader - supports official FinQA format"""

import json
import os
from typing import Optional


def format_table(table: list) -> str:
    """Format table as readable string."""
    if not table:
        return ""
    lines = []
    for row in table:
        lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)


def load_from_json(file_path: str, limit: Optional[int] = None) -> list[dict]:
    """Load from local JSON file (official FinQA format)."""
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

        # Official FinQA format
        if "qa" in item:
            qa = item["qa"]
            # Build context from pre_text, table, post_text
            context_parts = []
            if item.get("pre_text"):
                context_parts.append("\n".join(item["pre_text"]))
            if item.get("table"):
                context_parts.append(format_table(item["table"]))
            if item.get("post_text"):
                context_parts.append("\n".join(item["post_text"]))

            samples.append({
                "id": item.get("id", i),
                "question": qa.get("question", ""),
                "context": "\n\n".join(context_parts),
                "answer": str(qa.get("answer", "")),
            })
        # Simple format (question, context, answer at top level)
        else:
            samples.append({
                "id": item.get("id", i),
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "answer": str(item.get("answer", "")),
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
    import sys
    if len(sys.argv) > 1:
        samples = load_finqa(sys.argv[1], limit=3)
    else:
        samples = load_sample_data()
    print(f"Loaded {len(samples)} samples")
    for s in samples:
        print(f"\nQ: {s['question'][:80]}...")
        print(f"A: {s['answer']}")
        print(f"Context: {s['context'][:200]}...")
