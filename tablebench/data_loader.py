"""TableBench data loader - supports local file or HuggingFace"""

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
            if item.get("qtype") == "Visualization":
                continue
            samples.append({
                "id": item.get("id", i),
                "qtype": item.get("qtype", ""),
                "qsubtype": item.get("qsubtype", ""),
                "table": item.get("table", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })
    return samples


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
        if item.get("qtype") == "Visualization":
            continue
        samples.append({
            "id": item.get("id", i),
            "qtype": item.get("qtype", ""),
            "qsubtype": item.get("qsubtype", ""),
            "table": item.get("table", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        })
    return samples


def load_from_huggingface(limit: Optional[int] = None) -> list[dict]:
    """Load from HuggingFace datasets."""
    from datasets import load_dataset

    # Load TQA_test split specifically (Direct Prompting format)
    # The dataset has multiple configs with different columns, so we load just one
    try:
        dataset = load_dataset(
            "Multilingual-Multimodal-NLP/TableBench",
            data_files="TableBench_DP.jsonl",
            split="train"
        )
    except Exception:
        # Fallback: try loading with name parameter
        dataset = load_dataset(
            "Multilingual-Multimodal-NLP/TableBench",
            name="TQA",
            split="test"
        )

    samples = []
    for i, item in enumerate(dataset):
        if limit and len(samples) >= limit:
            break
        if item.get("qtype") == "Visualization":
            continue

        # Handle table format - can be dict or string
        table = item.get("table", "")
        if isinstance(table, dict):
            table = json.dumps(table)

        samples.append({
            "id": item.get("id", str(i)),
            "qtype": item.get("qtype", ""),
            "qsubtype": item.get("qsubtype", ""),
            "table": table,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        })
    return samples


def load_tablebench(source: str = "huggingface", limit: Optional[int] = None) -> list[dict]:
    """
    Load TableBench dataset.

    Args:
        source: "huggingface" or path to local file (.json or .jsonl)
        limit: Optional limit on number of samples

    Returns:
        List of samples with keys: id, qtype, qsubtype, table, question, answer
    """
    if source == "huggingface":
        return load_from_huggingface(limit)
    elif source.endswith('.jsonl'):
        return load_from_jsonl(source, limit)
    elif source.endswith('.json'):
        return load_from_json(source, limit)
    elif os.path.isfile(source):
        # Auto-detect format
        if source.endswith('.jsonl'):
            return load_from_jsonl(source, limit)
        return load_from_json(source, limit)
    else:
        raise ValueError(f"Unknown source: {source}")


def filter_by_qtype(samples: list[dict], qtypes: list[str]) -> list[dict]:
    """Filter samples by question type."""
    return [s for s in samples if s["qtype"] in qtypes]


def filter_by_qsubtype(samples: list[dict], qsubtypes: list[str]) -> list[dict]:
    """Filter samples by question subtype."""
    return [s for s in samples if s["qsubtype"] in qsubtypes]


# Sample data for testing when HuggingFace is unavailable
SAMPLE_DATA = [
    {
        "id": "test_1",
        "qtype": "NumericalReasoning",
        "qsubtype": "Aggregation",
        "table": '{"columns": ["Product", "Sales"], "data": [["A", 100], ["B", 200], ["C", 150]]}',
        "question": "What is the total sales?",
        "answer": "450"
    },
    {
        "id": "test_2",
        "qtype": "FactChecking",
        "qsubtype": "MatchBased",
        "table": '{"columns": ["Name", "Age"], "data": [["Alice", 30], ["Bob", 25]]}',
        "question": "Is Alice older than Bob?",
        "answer": "yes"
    },
    {
        "id": "test_3",
        "qtype": "NumericalReasoning",
        "qsubtype": "Counting",
        "table": '{"columns": ["City", "Population"], "data": [["NYC", 8000000], ["LA", 4000000], ["Chicago", 2700000]]}',
        "question": "How many cities have population over 3 million?",
        "answer": "2"
    },
    {
        "id": "test_4",
        "qtype": "DataAnalysis",
        "qsubtype": "Comparison",
        "table": '{"columns": ["Year", "Revenue"], "data": [[2020, 100], [2021, 110], [2022, 121]]}',
        "question": "What is the percentage increase in revenue from 2020 to 2022?",
        "answer": "21"
    },
]


def load_sample_data() -> list[dict]:
    """Load sample data for testing."""
    return SAMPLE_DATA.copy()


if __name__ == "__main__":
    # Test with sample data
    print("Testing with sample data...")
    samples = load_sample_data()
    print(f"Loaded {len(samples)} samples")
    for s in samples:
        print(f"  [{s['qtype']}] Q: {s['question']} -> A: {s['answer']}")
