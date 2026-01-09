"""MMLongBench-Doc evaluator - based on official evaluation logic"""

import re
import ast
from math import isclose
from typing import Union


def get_clean_string(s: str) -> str:
    """
    Clean string for comparison (matching official MMLongBench-Doc evaluation).

    Removes quotes, parentheses, currency symbols, percentages, and common units.
    """
    if not s:
        return ""

    s = str(s).strip()

    # Remove quotes
    s = s.strip('"\'')

    # Remove parentheses content if it's just a wrapper
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]

    # Remove leading currency symbols
    s = re.sub(r'^[\$€£¥]', '', s)

    # Remove trailing percentage
    s = re.sub(r'%$', '', s)

    # Remove trailing units (common ones from the paper)
    units_pattern = r'\s*(miles|million|billion|thousand|percent|dollars|USD|EUR|GBP|kg|km|m|cm|mm|lb|oz|year|years|month|months|day|days)$'
    s = re.sub(units_pattern, '', s, flags=re.IGNORECASE)

    return s.strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5, clean: bool = True) -> float:
    """
    Calculate ANLS (Average Normalized Levenshtein Similarity) score.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        threshold: Minimum similarity threshold (default: 0.5)
        clean: Whether to apply string cleaning (default: True)

    Returns:
        Score between 0 and 1
    """
    if clean:
        prediction = get_clean_string(prediction).lower()
        ground_truth = get_clean_string(ground_truth).lower()
    else:
        prediction = str(prediction).strip().lower()
        ground_truth = str(ground_truth).strip().lower()

    if not ground_truth:
        return 1.0 if not prediction else 0.0

    if not prediction:
        return 0.0

    distance = levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    similarity = 1 - (distance / max_len)

    return similarity if similarity >= threshold else 0.0


def is_float_equal(pred: str, gt: str, precision: int = 2) -> bool:
    """
    Compare float values with tolerance.

    Handles percentages and various number formats.
    """
    def clean_number(s: str) -> float:
        s = str(s).strip().lower()
        # Remove common formatting
        s = s.replace(',', '').replace('$', '').replace(' ', '')
        # Handle percentage
        if s.endswith('%'):
            s = s[:-1]
        # Handle parentheses for negative numbers
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        return float(s)

    try:
        pred_num = clean_number(pred)
        gt_num = clean_number(gt)

        # Round to precision
        pred_rounded = round(pred_num, precision)
        gt_rounded = round(gt_num, precision)

        if pred_rounded == gt_rounded:
            return True

        # Use relative tolerance
        return isclose(pred_num, gt_num, rel_tol=0.01)
    except (ValueError, TypeError):
        return False


def is_int_equal(pred: str, gt: str) -> bool:
    """Compare integer values."""
    def clean_int(s: str) -> int:
        s = str(s).strip()
        s = s.replace(',', '').replace(' ', '')
        # Handle float strings like "10.0"
        if '.' in s:
            return int(float(s))
        return int(s)

    try:
        return clean_int(pred) == clean_int(gt)
    except (ValueError, TypeError):
        return False


def parse_list(s: str) -> list:
    """Parse a string representation of a list."""
    if isinstance(s, list):
        return s
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else [result]
    except:
        return [s]


def eval_score(prediction: str, ground_truth: str, answer_format: str) -> float:
    """
    Evaluate a single prediction against ground truth.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        answer_format: Expected format (Str, Int, Float, List, None)

    Returns:
        Score between 0 and 1
    """
    # Handle "Not answerable" cases
    if answer_format == "None" or ground_truth == "Not answerable":
        pred_lower = str(prediction).strip().lower()
        if "not answerable" in pred_lower or "unanswerable" in pred_lower:
            return 1.0
        return 0.0

    # Handle empty prediction
    if not prediction or str(prediction).strip() == "":
        return 0.0

    # Evaluate based on answer format
    if answer_format == "Int":
        return 1.0 if is_int_equal(prediction, ground_truth) else 0.0

    elif answer_format == "Float":
        # Try exact float comparison first
        if is_float_equal(prediction, ground_truth):
            return 1.0
        # Fallback to ANLS for string comparison
        return anls_score(prediction, ground_truth)

    elif answer_format == "List":
        pred_list = parse_list(prediction)
        gt_list = parse_list(ground_truth)

        if len(pred_list) != len(gt_list):
            # Partial credit based on overlap
            pred_set = set(str(x).lower().strip() for x in pred_list)
            gt_set = set(str(x).lower().strip() for x in gt_list)
            intersection = pred_set & gt_set
            union = pred_set | gt_set
            return len(intersection) / len(union) if union else 0.0

        # Element-wise comparison
        scores = []
        pred_sorted = sorted([str(x).lower().strip() for x in pred_list])
        gt_sorted = sorted([str(x).lower().strip() for x in gt_list])
        for p, g in zip(pred_sorted, gt_sorted):
            scores.append(anls_score(p, g))
        return sum(scores) / len(scores) if scores else 0.0

    else:  # Str or default
        # Check for special formats (exact match)
        gt_lower = str(ground_truth).lower()
        if any(x in gt_lower for x in ['http://', 'https://', '@', '.com', '.org']):
            return 1.0 if str(prediction).strip().lower() == gt_lower else 0.0

        # General string comparison using ANLS
        return anls_score(prediction, ground_truth)


def evaluate_batch(results: list[dict]) -> dict:
    """
    Evaluate a batch of results.

    Args:
        results: List of {prediction, ground_truth, answer_format, ...}

    Returns:
        Dict with accuracy and per-format breakdown
    """
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "total": 0}

    total_score = 0.0
    by_format = {}

    for r in results:
        score = eval_score(
            r.get("prediction", ""),
            r.get("ground_truth", ""),
            r.get("answer_format", "Str")
        )
        total_score += score

        # Track by format
        fmt = r.get("answer_format", "Str")
        if fmt not in by_format:
            by_format[fmt] = {"total_score": 0.0, "count": 0}
        by_format[fmt]["total_score"] += score
        by_format[fmt]["count"] += 1

    # Calculate averages
    accuracy = total_score / total

    format_accuracy = {}
    for fmt, stats in by_format.items():
        format_accuracy[fmt] = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0.0

    return {
        "accuracy": accuracy,
        "total": total,
        "total_score": total_score,
        "by_format": format_accuracy,
        "by_format_detail": by_format
    }


if __name__ == "__main__":
    print("Testing MMLongBench-Doc evaluator...")

    # Test ANLS
    assert anls_score("hello", "hello") == 1.0
    assert anls_score("hello", "hallo") > 0.5
    assert anls_score("abc", "xyz") == 0.0  # Below threshold

    # Test Int
    assert is_int_equal("42", "42") == True
    assert is_int_equal("42.0", "42") == True
    assert is_int_equal("1,000", "1000") == True

    # Test Float
    assert is_float_equal("3.14", "3.14") == True
    assert is_float_equal("3.14159", "3.14") == True  # Rounded
    assert is_float_equal("50%", "50") == True

    # Test eval_score
    assert eval_score("42", "42", "Int") == 1.0
    assert eval_score("hello world", "hello world", "Str") == 1.0
    assert eval_score("Not answerable", "Not answerable", "None") == 1.0
    assert eval_score("['a', 'b']", "['a', 'b']", "List") == 1.0

    print("All tests passed!")
