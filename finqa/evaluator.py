"""FinQA evaluator based on official evaluation logic"""

import re
from typing import Union


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    Based on FinQA official evaluate.py
    """
    if not answer:
        return ""

    s = str(answer).strip().lower()

    # Remove common formatting
    s = s.replace(',', '')  # Remove comma separators
    s = s.replace('$', '')  # Remove dollar signs
    s = s.replace(' ', '')  # Remove spaces

    # Handle percentage: remove % but keep the number as-is
    # (FinQA expects percentage as the number, e.g., "14" for 14%)
    s = s.replace('%', '')

    # Handle special const_ prefix (from FinQA DSL)
    if s.startswith('const_'):
        s = s.replace('const_', '')
        if s == 'm1':
            s = '-1'

    return s


def extract_number(text: str) -> Union[float, None]:
    """Extract numeric value from text."""
    if not text:
        return None

    # Normalize first
    text = normalize_answer(text)

    # Handle negative numbers
    text = text.replace('−', '-').replace('–', '-')

    # Try to parse as float
    try:
        return float(text)
    except ValueError:
        pass

    # Try to extract number with regex
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def finqa_equal(prediction: str, ground_truth: str, precision: int = 5) -> bool:
    """
    Compare prediction with ground truth using FinQA official logic.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        precision: Decimal precision for rounding (default: 5)

    Returns:
        True if answers are considered equal
    """
    # Normalize both
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # Direct string match
    if pred_norm == gt_norm:
        return True

    # Numeric comparison with rounding
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        # Round to specified precision (FinQA uses 5)
        pred_rounded = round(pred_num, precision)
        gt_rounded = round(gt_num, precision)
        return pred_rounded == gt_rounded

    return False


def evaluate_finqa(results: list[dict]) -> dict:
    """
    Evaluate a batch of FinQA results.

    Args:
        results: List of {prediction, ground_truth, ...}

    Returns:
        Dict with accuracy metrics
    """
    total = len(results)
    correct = 0

    for r in results:
        if finqa_equal(r["prediction"], r["ground_truth"]):
            correct += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


if __name__ == "__main__":
    # Test cases based on FinQA format
    print("Testing FinQA evaluator...")

    # Basic tests
    assert finqa_equal("14", "14") == True
    assert finqa_equal("14.0", "14") == True
    assert finqa_equal("14%", "14") == True
    assert finqa_equal("14.00000", "14") == True
    assert finqa_equal("-1", "const_m1") == True
    assert finqa_equal("1,234", "1234") == True
    assert finqa_equal("$100", "100") == True

    # Precision tests (5 decimal places)
    assert finqa_equal("0.12345", "0.12345") == True
    assert finqa_equal("0.123456", "0.123457") == True  # Rounds to same
    assert finqa_equal("0.12345", "0.12346") == False  # Different at 5th decimal

    # Edge cases
    assert finqa_equal("15", "14") == False
    assert finqa_equal("yes", "no") == False

    print("All tests passed!")
