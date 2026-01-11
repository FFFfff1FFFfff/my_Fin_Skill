"""ChartQAPro data loader - Chart Question Answering benchmark"""

import base64
import io
import os
from typing import Optional
from pathlib import Path

# Cache directory
CACHE_DIR = Path(__file__).parent / "cache"


def ensure_cache_dirs():
    """Create cache directories if they don't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def load_chartqapro(limit: Optional[int] = None,
                    question_types: Optional[list[str]] = None) -> list[dict]:
    """
    Load ChartQAPro dataset from HuggingFace.

    Args:
        limit: Maximum number of samples to load
        question_types: Filter by question types
                       (e.g., ["Reasoning", "Conversational", "Multi Choice",
                               "Hypothetical", "Fact Checking"])

    Returns:
        List of samples with keys: id, image_base64, questions, answers,
                                   question_type, year_flags, paragraph
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return []

    print("Loading ChartQAPro from HuggingFace...")
    dataset = load_dataset("ahmed-masry/ChartQAPro", split="test")

    samples = []
    for i, item in enumerate(dataset):
        # Filter by question type if specified
        q_type = item.get("Question Type", "")
        if question_types and q_type not in question_types:
            continue

        # Convert image to base64
        image = item.get("image")
        image_base64 = None
        if image is not None:
            try:
                # HuggingFace datasets often return PIL Image directly
                from PIL import Image as PILImage
                if isinstance(image, PILImage.Image):
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    image_base64 = base64.standard_b64encode(buffered.getvalue()).decode('utf-8')
                elif isinstance(image, bytes):
                    image_base64 = base64.standard_b64encode(image).decode('utf-8')
                elif isinstance(image, dict) and 'bytes' in image:
                    # Some HF datasets store as {'bytes': ..., 'path': ...}
                    image_base64 = base64.standard_b64encode(image['bytes']).decode('utf-8')
                elif hasattr(image, 'save'):
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    image_base64 = base64.standard_b64encode(buffered.getvalue()).decode('utf-8')
                else:
                    print(f"Unknown image type {i}: {type(image)}")
            except Exception as e:
                print(f"Failed to convert image {i}: {e}")

        # Get questions and answers (can be lists for conversational)
        questions = item.get("Question", [])
        answers = item.get("Answer", [])
        year_flags = item.get("Year", [])

        # Ensure they are lists
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]
        if isinstance(year_flags, str):
            year_flags = [year_flags]

        samples.append({
            "id": i,
            "image_base64": image_base64,
            "questions": questions,
            "answers": answers,
            "question_type": q_type,
            "year_flags": year_flags,
            "paragraph": item.get("Paragraph", ""),
        })

        if limit and len(samples) >= limit:
            break

    return samples


def get_question_type_stats(samples: list[dict]) -> dict:
    """Get statistics on question types in the dataset."""
    stats = {}
    for s in samples:
        q_type = s["question_type"]
        stats[q_type] = stats.get(q_type, 0) + 1
    return stats


# Sample data for testing without downloading
SAMPLE_DATA = [
    {
        "id": 0,
        "image_base64": None,  # Would be actual base64 in real data
        "questions": ["What is the highest value shown in the bar chart?"],
        "answers": ["45"],
        "question_type": "Reasoning",
        "year_flags": ["NO"],
        "paragraph": "",
    },
    {
        "id": 1,
        "image_base64": None,
        "questions": ["What color represents the category 'Sales'?"],
        "answers": ["blue"],
        "question_type": "Reasoning",
        "year_flags": ["NO"],
        "paragraph": "",
    },
    {
        "id": 2,
        "image_base64": None,
        "questions": [
            "What is the value for USA?",
            "And what about China?",
            "Which one is higher?"
        ],
        "answers": ["42", "35", "USA"],
        "question_type": "Conversational",
        "year_flags": ["NO", "NO", "NO"],
        "paragraph": "",
    },
    {
        "id": 3,
        "image_base64": None,
        "questions": ["The chart shows that sales increased every year. True or False?"],
        "answers": ["False"],
        "question_type": "Fact Checking",
        "year_flags": ["NO"],
        "paragraph": "The chart displays annual sales figures from 2018 to 2022.",
    },
]


def load_sample_data() -> list[dict]:
    """Load sample data for testing."""
    return SAMPLE_DATA.copy()


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Loading ChartQAPro (limit={limit})...")
    samples = load_chartqapro(limit=limit)

    print(f"Loaded {len(samples)} samples\n")

    # Show stats
    stats = get_question_type_stats(samples)
    print("Question Type Distribution:")
    for q_type, count in sorted(stats.items()):
        print(f"  {q_type}: {count}")
    print()

    # Show samples
    for s in samples[:3]:
        print(f"[{s['id']}] Type: {s['question_type']}")
        print(f"  Q: {s['questions'][0][:60]}...")
        print(f"  A: {s['answers'][0]}")
        print(f"  Has image: {s['image_base64'] is not None}")
        print()
