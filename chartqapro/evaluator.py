"""ChartQAPro evaluator - aligned with official evaluation script from
https://github.com/vis-nlp/ChartQAPro/blob/main/evaluate_predictions.py
"""

import ast
import re
from typing import List, Optional, Any

# Try to use official anls package, fallback to custom implementation
try:
    from anls import anls_score as _anls_score
    def anls_score(prediction: str, gold_labels: List[str], threshold: float = 0.5) -> float:
        return _anls_score(prediction=prediction, gold_labels=gold_labels, threshold=threshold)
except ImportError:
    def levenshtein_distance(s1: str, s2: str) -> int:
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

    def anls_score(prediction: str, gold_labels: List[str], threshold: float = 0.5) -> float:
        """ANLS score with multiple gold labels."""
        if not gold_labels:
            return 0.0
        scores = []
        for gold in gold_labels:
            if not gold:
                continue
            dist = levenshtein_distance(prediction, gold)
            max_len = max(len(prediction), len(gold))
            sim = 1 - (dist / max_len) if max_len > 0 else 1.0
            scores.append(sim if sim >= threshold else 0.0)
        return max(scores) if scores else 0.0


def fix_list_format(item: str) -> Any:
    """Standardize string representations of lists (from official script)."""
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def parse_to_list(text: str) -> Optional[List[str]]:
    """Parses text to a list of strings (from official script)."""
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def to_float(text: str) -> Optional[float]:
    """Converts text to float, stripping percent signs (from official script)."""
    try:
        return float(text.strip().strip('%'))
    except ValueError:
        return None


def evaluate_single_answer(target: str, prediction: str, max_relative_change: float = 0.05) -> float:
    """Evaluate single target-prediction pair (from official script)."""
    t = target.strip().strip('%').strip()
    p = prediction.strip().strip('%').strip()

    # Attempt numeric comparison
    t_f = to_float(t)
    p_f = to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0

    # Fallback to ANLS for text
    return anls_score(prediction=p.lower(), gold_labels=[t.lower()], threshold=0.5)


def relaxed_correctness(targets: list, predictions: list, year_flags: list,
                        question_type: str, max_relative_change: float = 0.05) -> float:
    """
    Calculate relaxed correctness (aligned with official script).
    """
    # Handle list format
    fixed_t = fix_list_format(str(targets[-1]) if targets else "")
    t_list = parse_to_list(str(fixed_t)) or [str(targets[-1]) if targets else ""]

    fixed_p = fix_list_format(str(predictions[-1]) if predictions else "")
    p_list = parse_to_list(str(fixed_p)) or [str(predictions[-1]) if predictions else ""]

    n = len(t_list)

    # For conversational, only use last year_flag
    if question_type == "Conversational":
        year_flags = year_flags[-1:] if year_flags else ["NO"]

    # Expand year_flags if needed
    if year_flags and len(year_flags) < n:
        year_flags = year_flags * n

    # Determine if we should use exact match
    always_use_exact_match = question_type in ["Fact Checking", "Multi Choice"]

    # Evaluate elements
    scores = []
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue

        t_item = t_list[idx]
        p_item = p_list[idx]
        flag = year_flags[idx] if year_flags and idx < len(year_flags) else "NO"
        flag_cond = str(flag).upper() == 'YES'

        if flag_cond or always_use_exact_match:
            # Exact match for years, fact checking, or multichoice
            scores.append(1.0 if t_item.strip().lower() == p_item.strip().lower() else 0.0)
        else:
            scores.append(evaluate_single_answer(t_item, p_item, max_relative_change))

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_batch(results: list) -> dict:
    """Evaluate a batch of results."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "total": 0, "total_score": 0.0, "by_type": {}, "by_type_detail": {}}

    total_score = 0.0
    by_type = {}

    for r in results:
        score = r.get("score", 0.0)
        total_score += score
        question_type = r.get("question_type", "")

        if question_type not in by_type:
            by_type[question_type] = {"total_score": 0.0, "count": 0}
        by_type[question_type]["total_score"] += score
        by_type[question_type]["count"] += 1

    accuracy = total_score / total
    type_accuracy = {q: stats["total_score"] / stats["count"] for q, stats in by_type.items() if stats["count"] > 0}

    return {
        "accuracy": accuracy,
        "total": total,
        "total_score": total_score,
        "by_type": type_accuracy,
        "by_type_detail": by_type
    }
