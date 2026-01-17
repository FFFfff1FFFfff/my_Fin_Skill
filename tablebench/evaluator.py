"""Evaluation metrics for TableBench"""

import re
from typing import Union


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    if not answer:
        return ""
    # Lowercase, strip whitespace
    s = str(answer).lower().strip()
    # Remove common punctuation
    s = re.sub(r'[,\$%]', '', s)
    # Normalize whitespace
    s = ' '.join(s.split())
    return s


def extract_number(text: str) -> Union[float, None]:
    """Extract numeric value from text."""
    if not text:
        return None
    # Remove common formatting
    text = str(text).replace(',', '').replace('$', '').replace('%', '').strip()
    # Handle negative numbers
    text = text.replace('−', '-').replace('–', '-')
    # Try to extract number
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Exact match evaluation.
    Used for: Fact Checking, Numerical Reasoning
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # Direct string match
    if pred_norm == gt_norm:
        return True

    # Numeric comparison (handles format differences like "14" vs "14.0")
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < 1e-6

    return False


def em_with_error(prediction: str, ground_truth: str, tolerance: float = 0.1) -> bool:
    """
    Exact match with error tolerance.
    Used for: Data Analysis (allows ±10% error for numerical answers)

    Args:
        tolerance: Relative error tolerance (default: 0.1 = 10%)
    """
    # Try exact match first
    if exact_match(prediction, ground_truth):
        return True

    # Numeric comparison with tolerance
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return abs(pred_num) < 1e-6
        relative_error = abs(pred_num - gt_num) / abs(gt_num)
        return relative_error <= tolerance

    return False


def evaluate_sample(prediction: str, ground_truth: str, qtype: str) -> bool:
    """
    Evaluate a single sample based on question type.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        qtype: Question type (FactChecking, NumericalReasoning, DataAnalysis)
    """
    if qtype == "DataAnalysis":
        return em_with_error(prediction, ground_truth, tolerance=0.1)
    else:
        # FactChecking, NumericalReasoning use exact match
        return exact_match(prediction, ground_truth)


def evaluate_batch(results: list[dict]) -> dict:
    """
    Evaluate a batch of results.

    Args:
        results: List of {prediction, ground_truth, qtype, ...}

    Returns:
        Dict with overall and per-type accuracy
    """
    total = len(results)
    correct = 0
    by_type = {}

    for r in results:
        qtype = r.get("qtype", "Unknown")
        is_correct = evaluate_sample(r["prediction"], r["ground_truth"], qtype)

        if is_correct:
            correct += 1

        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if is_correct:
            by_type[qtype]["correct"] += 1

    # Calculate accuracies
    overall_acc = correct / total if total > 0 else 0
    type_acc = {}
    for qtype, stats in by_type.items():
        type_acc[qtype] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "overall": {"accuracy": overall_acc, "correct": correct, "total": total},
        "by_type": type_acc,
        "by_type_detail": by_type
    }


if __name__ == "__main__":
    # Test cases
    print("Testing evaluator...")

    # Exact match tests
    assert exact_match("14", "14") == True
    assert exact_match("14.0", "14") == True
    assert exact_match("$14", "14") == True
    assert exact_match("yes", "YES") == True
    assert exact_match("15", "14") == False

    # EM with error tests
    assert em_with_error("110", "100", 0.1) == True  # 10% error OK
    assert em_with_error("115", "100", 0.1) == False  # 15% error NOT OK
    assert em_with_error("95", "100", 0.1) == True  # -5% error OK

    print("All tests passed!")
