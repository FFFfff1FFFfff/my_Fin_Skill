"""ChartQAPro evaluator - based on official evaluation logic from
https://github.com/vis-nlp/ChartQAPro/blob/main/evaluate_predictions.py
"""

import ast
import re
from typing import Union


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


def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Calculate ANLS (Average Normalized Levenshtein Similarity) score.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        threshold: Minimum similarity threshold (default: 0.5)

    Returns:
        Score between 0 and 1
    """
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


def to_float(text: str) -> float:
    """Convert text to float, stripping percent signs."""
    text = str(text).strip()
    text = text.replace('%', '').replace(',', '').strip()
    return float(text)


def fix_list_format(text: str) -> str:
    """Standardize string representations of lists."""
    text = str(text).strip()
    if not text.startswith('['):
        return text

    # Try to fix unquoted list elements
    try:
        # Replace unquoted elements with quoted ones
        fixed = re.sub(r'\[([^\]]+)\]', lambda m: '[' + ', '.join(
            f'"{x.strip()}"' if not (x.strip().startswith('"') or x.strip().startswith("'"))
            else x.strip()
            for x in m.group(1).split(',')
        ) + ']', text)
        return fixed
    except:
        return text


def parse_to_list(text: str) -> list:
    """Parse text to list."""
    if isinstance(text, list):
        return text

    text = str(text).strip()
    if not text:
        return []

    # Try to parse as Python literal
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return result
        return [result]
    except:
        pass

    # Try with fixed format
    try:
        fixed = fix_list_format(text)
        result = ast.literal_eval(fixed)
        if isinstance(result, list):
            return result
        return [result]
    except:
        pass

    return [text]


def evaluate_single_answer(target: str, prediction: str, year_flag: str = "NO",
                           tolerance: float = 0.05) -> float:
    """
    Evaluate a single answer against target.

    Args:
        target: Ground truth answer
        prediction: Model prediction
        year_flag: Whether this is a year-based answer ("YES" or "NO")
        tolerance: Relative tolerance for numeric comparison (default: 5%)

    Returns:
        Score between 0 and 1
    """
    target = str(target).strip()
    prediction = str(prediction).strip()

    if not target or not prediction:
        return 0.0

    # Try numeric comparison first
    try:
        pred_float = to_float(prediction)
        target_float = to_float(target)

        # Handle zero case
        if target_float == 0:
            return 1.0 if pred_float == 0 else 0.0

        # Check if within tolerance
        relative_error = abs(pred_float - target_float) / abs(target_float)
        return 1.0 if relative_error <= tolerance else 0.0

    except (ValueError, TypeError):
        pass

    # Fall back to ANLS for text comparison
    return anls_score(prediction, target, threshold=0.5)


def relaxed_correctness(targets: list, predictions: list, year_flags: list,
                        question_type: str) -> float:
    """
    Calculate relaxed correctness for ChartQAPro.

    Args:
        targets: List of ground truth answers
        predictions: List of model predictions
        year_flags: List of year flags for each answer
        question_type: Type of question

    Returns:
        Average score across all answer pairs
    """
    if not targets or not predictions:
        return 0.0

    # For conversational questions, only evaluate the final answer
    if question_type == "Conversational":
        targets = targets[-1:]
        predictions = predictions[-1:] if predictions else [""]
        year_flags = year_flags[-1:] if year_flags else ["NO"]

    # Ensure predictions match targets length
    while len(predictions) < len(targets):
        predictions.append("")
    predictions = predictions[:len(targets)]

    # Ensure year_flags match targets length
    while len(year_flags) < len(targets):
        year_flags.append("NO")
    year_flags = year_flags[:len(targets)]

    # For Fact Checking and Multi Choice, use exact match
    if question_type in ["Fact Checking", "Multi Choice"]:
        scores = []
        for t, p in zip(targets, predictions):
            t_clean = str(t).strip().lower()
            p_clean = str(p).strip().lower()
            scores.append(1.0 if t_clean == p_clean else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    # For other types, evaluate each answer
    scores = []
    for target, prediction, year_flag in zip(targets, predictions, year_flags):
        # Handle list answers
        target_list = parse_to_list(target)
        pred_list = parse_to_list(prediction)

        if len(target_list) > 1:
            # List comparison
            if len(pred_list) != len(target_list):
                scores.append(0.0)
            else:
                # Sort and compare element-wise
                target_sorted = sorted([str(x).lower() for x in target_list])
                pred_sorted = sorted([str(x).lower() for x in pred_list])
                elem_scores = [
                    evaluate_single_answer(t, p, year_flag)
                    for t, p in zip(target_sorted, pred_sorted)
                ]
                scores.append(sum(elem_scores) / len(elem_scores) if elem_scores else 0.0)
        else:
            # Single answer comparison
            scores.append(evaluate_single_answer(target, prediction, year_flag))

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_batch(results: list[dict]) -> dict:
    """
    Evaluate a batch of results.

    Args:
        results: List of {predictions, answers, question_type, year_flags, ...}

    Returns:
        Dict with accuracy and per-type breakdown
    """
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "total": 0}

    total_score = 0.0
    by_type = {}

    for r in results:
        predictions = r.get("predictions", [])
        answers = r.get("answers", [])
        question_type = r.get("question_type", "")
        year_flags = r.get("year_flags", [])

        score = relaxed_correctness(answers, predictions, year_flags, question_type)
        total_score += score

        # Track by question type
        if question_type not in by_type:
            by_type[question_type] = {"total_score": 0.0, "count": 0}
        by_type[question_type]["total_score"] += score
        by_type[question_type]["count"] += 1

    # Calculate averages
    accuracy = total_score / total

    type_accuracy = {}
    for q_type, stats in by_type.items():
        type_accuracy[q_type] = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0.0

    return {
        "accuracy": accuracy,
        "total": total,
        "total_score": total_score,
        "by_type": type_accuracy,
        "by_type_detail": by_type
    }


if __name__ == "__main__":
    print("Testing ChartQAPro evaluator...")

    # Test numeric comparison
    assert evaluate_single_answer("100", "100") == 1.0
    assert evaluate_single_answer("100", "102") == 1.0  # Within 5%
    assert evaluate_single_answer("100", "110") == 0.0  # Over 5%
    assert evaluate_single_answer("50%", "50") == 1.0

    # Test text comparison
    assert evaluate_single_answer("blue", "blue") == 1.0
    assert evaluate_single_answer("Blue", "blue") == 1.0  # Case insensitive via ANLS
    assert anls_score("hello", "hello") == 1.0

    # Test relaxed_correctness
    assert relaxed_correctness(["42"], ["42"], ["NO"], "Reasoning") == 1.0
    assert relaxed_correctness(["True"], ["true"], ["NO"], "Fact Checking") == 1.0
    assert relaxed_correctness(["False"], ["True"], ["NO"], "Fact Checking") == 0.0

    # Test conversational (only last answer matters)
    assert relaxed_correctness(
        ["10", "20", "30"],
        ["wrong", "wrong", "30"],
        ["NO", "NO", "NO"],
        "Conversational"
    ) == 1.0

    print("All tests passed!")
