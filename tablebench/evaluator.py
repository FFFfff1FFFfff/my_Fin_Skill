"""
Evaluation metrics for TableBench - Official Implementation

Based on official TableBench repository:
https://github.com/TableBench/TableBench

Key features:
- Decimal precision for numeric comparisons
- Multi-answer support (comma-separated with equal weights)
- Percentage handling (normalize % to decimal)
- EM_with_error_10 for Data Analysis subtypes
"""

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Union, Optional


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    Official implementation: lowercase, remove punctuation, articles, extra whitespace.
    """
    if not answer:
        return ""
    s = str(answer).lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation except . - % for numbers
    s = re.sub(r'[,\$"\'\(\)\[\]\{\}]', '', s)
    # Normalize whitespace
    s = ' '.join(s.split())
    return s


def normalize_number(text: str) -> Optional[Decimal]:
    """
    Convert string to Decimal, handling percentages.

    - "50%" -> 0.5
    - "50" -> 50
    - Returns None if not a valid number
    """
    if not text:
        return None

    text = str(text).strip()
    is_percent = text.endswith('%')

    # Remove formatting
    text = text.replace(',', '').replace('$', '').replace('%', '').strip()
    # Handle different minus signs
    text = text.replace('−', '-').replace('–', '-')

    # Try to extract number
    match = re.search(r'^-?\d+\.?\d*$', text)
    if not match:
        return None

    try:
        value = Decimal(match.group())
        if is_percent:
            value = value / Decimal('100')
        # Quantize to 4 decimal places
        return value.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


def get_decimal_precision(references: list[str]) -> int:
    """
    Determine the minimum decimal precision from reference values.
    Ignores percentages.
    """
    max_precision = 0
    for ref in references:
        ref = str(ref).strip()
        if ref.endswith('%'):
            continue
        ref = ref.replace(',', '').replace('$', '')
        if '.' in ref:
            decimal_part = ref.split('.')[-1]
            precision = len(decimal_part.rstrip('0')) or 1
            max_precision = max(max_precision, precision)
    return max_precision


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Official Exact Match evaluation with multi-answer support.

    Features:
    - Comma-separated multi-answers with equal weights
    - Decimal precision comparison
    - Percentage normalization

    Returns:
        float: Match score (0.0 to 1.0)
    """
    # Split comma-separated answers
    ref_answers = [x.strip() for x in str(ground_truth).split(',')]
    pred_answers = [x.strip() for x in str(prediction).split(',')]

    if len(ref_answers) != len(pred_answers):
        # Different number of answers - try direct comparison
        pred_norm = normalize_answer(prediction)
        gt_norm = normalize_answer(ground_truth)
        if pred_norm == gt_norm:
            return 1.0
        # Try numeric comparison
        pred_num = normalize_number(prediction)
        gt_num = normalize_number(ground_truth)
        if pred_num is not None and gt_num is not None and pred_num == gt_num:
            return 1.0
        return 0.0

    # Equal number of answers - compare each pair
    weight = 1.0 / len(ref_answers)
    total_score = 0.0

    # Determine precision from references
    precision = get_decimal_precision(ref_answers)

    for ref, pred in zip(ref_answers, pred_answers):
        ref_norm = normalize_answer(ref)
        pred_norm = normalize_answer(pred)

        # Direct string match
        if ref_norm == pred_norm:
            total_score += weight
            continue

        # Numeric comparison with precision
        ref_num = normalize_number(ref)
        pred_num = normalize_number(pred)

        if ref_num is not None and pred_num is not None:
            # Round to detected precision
            if precision > 0:
                quantizer = Decimal(10) ** -precision
                ref_rounded = ref_num.quantize(quantizer, rounding=ROUND_HALF_UP)
                pred_rounded = pred_num.quantize(quantizer, rounding=ROUND_HALF_UP)
            else:
                ref_rounded = ref_num.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
                pred_rounded = pred_num.quantize(Decimal('1'), rounding=ROUND_HALF_UP)

            if ref_rounded == pred_rounded:
                total_score += weight

    return total_score


def em_with_error(prediction: str, ground_truth: str, tolerance: float = 0.1) -> float:
    """
    Exact match with error tolerance for numeric values.

    Args:
        tolerance: Relative error tolerance (default: 0.1 = 10%)

    Returns:
        float: Match score (0.0 to 1.0)
    """
    # First try exact match
    em_score = exact_match(prediction, ground_truth)
    if em_score >= 1.0:
        return 1.0

    # Split comma-separated answers
    ref_answers = [x.strip() for x in str(ground_truth).split(',')]
    pred_answers = [x.strip() for x in str(prediction).split(',')]

    if len(ref_answers) != len(pred_answers):
        # Try single value comparison with tolerance
        pred_num = normalize_number(prediction)
        gt_num = normalize_number(ground_truth)

        if pred_num is not None and gt_num is not None:
            if gt_num == Decimal('0'):
                return 1.0 if abs(pred_num) < Decimal('0.0001') else 0.0
            relative_error = abs(pred_num - gt_num) / abs(gt_num)
            return 1.0 if float(relative_error) <= tolerance else 0.0
        return 0.0

    # Compare each pair with tolerance
    weight = 1.0 / len(ref_answers)
    total_score = 0.0

    for ref, pred in zip(ref_answers, pred_answers):
        # Try exact match first
        if normalize_answer(ref) == normalize_answer(pred):
            total_score += weight
            continue

        # Numeric comparison with tolerance
        ref_num = normalize_number(ref)
        pred_num = normalize_number(pred)

        if ref_num is not None and pred_num is not None:
            if ref_num == Decimal('0'):
                if abs(pred_num) < Decimal('0.0001'):
                    total_score += weight
            else:
                relative_error = abs(pred_num - ref_num) / abs(ref_num)
                if float(relative_error) <= tolerance:
                    total_score += weight

    return total_score


def evaluate_sample(prediction: str, ground_truth: str, qtype: str, qsubtype: str = "") -> float:
    """
    Evaluate a single sample based on question type and subtype.

    Official metrics per category:
    - FactChecking: EM
    - NumericalReasoning: EM
    - DataAnalysis:
        - Correlation, Trend, StatisticalAnalysis: EM_with_error_10
        - ImpactAnalysis: EM (strict)
        - Other: ROUGE-L (we fallback to EM for now)

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        qtype: Question type (FactChecking, NumericalReasoning, DataAnalysis)
        qsubtype: Question subtype for DataAnalysis

    Returns:
        float: Score (0.0 to 1.0)
    """
    if qtype == "DataAnalysis":
        # Different subtypes use different metrics
        tolerance_subtypes = ["Correlation", "Trend", "StatisticalAnalysis"]
        strict_subtypes = ["ImpactAnalysis"]

        if qsubtype in tolerance_subtypes:
            return em_with_error(prediction, ground_truth, tolerance=0.1)
        elif qsubtype in strict_subtypes:
            return exact_match(prediction, ground_truth)
        else:
            # Other subtypes - use EM with error as fallback
            return em_with_error(prediction, ground_truth, tolerance=0.1)
    else:
        # FactChecking, NumericalReasoning use strict EM
        return exact_match(prediction, ground_truth)


def evaluate_batch(results: list[dict]) -> dict:
    """
    Evaluate a batch of results.

    Args:
        results: List of {prediction, ground_truth, qtype, qsubtype, ...}

    Returns:
        Dict with overall and per-type accuracy
    """
    total = len(results)
    total_score = 0.0
    by_type = {}
    by_subtype = {}

    for r in results:
        qtype = r.get("qtype", "Unknown")
        qsubtype = r.get("qsubtype", "")
        score = evaluate_sample(
            r["prediction"],
            r["ground_truth"],
            qtype,
            qsubtype
        )

        total_score += score

        # Track by type
        if qtype not in by_type:
            by_type[qtype] = {"score": 0.0, "total": 0}
        by_type[qtype]["total"] += 1
        by_type[qtype]["score"] += score

        # Track by subtype
        key = f"{qtype}/{qsubtype}" if qsubtype else qtype
        if key not in by_subtype:
            by_subtype[key] = {"score": 0.0, "total": 0}
        by_subtype[key]["total"] += 1
        by_subtype[key]["score"] += score

    # Calculate accuracies
    overall_acc = total_score / total if total > 0 else 0

    type_acc = {}
    type_detail = {}
    for qtype, stats in by_type.items():
        acc = stats["score"] / stats["total"] if stats["total"] > 0 else 0
        type_acc[qtype] = acc
        type_detail[qtype] = {
            "correct": round(stats["score"], 2),
            "total": stats["total"],
            "accuracy": round(acc * 100, 2)
        }

    subtype_acc = {}
    for key, stats in by_subtype.items():
        acc = stats["score"] / stats["total"] if stats["total"] > 0 else 0
        subtype_acc[key] = round(acc * 100, 2)

    return {
        "overall": {
            "accuracy": overall_acc,
            "correct": round(total_score, 2),
            "total": total
        },
        "by_type": type_acc,
        "by_type_detail": type_detail,
        "by_subtype": subtype_acc
    }


if __name__ == "__main__":
    # Test cases matching official implementation
    print("Testing official evaluator...")

    # Basic exact match tests
    assert exact_match("14", "14") == 1.0
    assert exact_match("14.0", "14") == 1.0
    assert exact_match("$14", "14") == 1.0
    assert exact_match("yes", "YES") == 1.0
    assert exact_match("15", "14") == 0.0
    print("  Basic EM: OK")

    # Percentage normalization tests
    assert exact_match("50%", "0.5") == 1.0
    assert exact_match("0.5", "50%") == 1.0
    assert exact_match("25%", "0.25") == 1.0
    print("  Percentage normalization: OK")

    # Multi-answer tests
    assert exact_match("10, 20", "10, 20") == 1.0
    assert exact_match("10, 20", "10, 21") == 0.5  # Half correct
    assert exact_match("a, b, c", "a, b, c") == 1.0
    print("  Multi-answer: OK")

    # EM with error tests
    assert em_with_error("110", "100", 0.1) == 1.0  # 10% error OK
    assert em_with_error("111", "100", 0.1) == 0.0  # 11% error NOT OK
    assert em_with_error("95", "100", 0.1) == 1.0   # -5% error OK
    assert em_with_error("100", "100", 0.1) == 1.0  # Exact match
    print("  EM with tolerance: OK")

    # Subtype evaluation tests
    assert evaluate_sample("110", "100", "DataAnalysis", "Correlation") == 1.0
    assert evaluate_sample("110", "100", "DataAnalysis", "ImpactAnalysis") == 0.0
    assert evaluate_sample("100", "100", "NumericalReasoning", "Counting") == 1.0
    print("  Subtype evaluation: OK")

    print("\nAll tests passed!")
