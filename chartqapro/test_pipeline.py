#!/usr/bin/env python3
"""Test ChartQAPro pipeline with a generated chart image."""

import base64
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generate a simple bar chart
def create_test_chart():
    """Create a simple bar chart for testing."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping chart generation")
        return None

    # Create data
    categories = ['Apple', 'Banana', 'Orange', 'Grape']
    values = [45, 32, 28, 19]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, values, color=['red', 'yellow', 'orange', 'purple'])

    # Add labels
    ax.set_xlabel('Fruit')
    ax.set_ylabel('Quantity')
    ax.set_title('Fruit Inventory')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom')

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return base64.standard_b64encode(buf.getvalue()).decode('utf-8')


def test_pipeline():
    """Test the full pipeline with a generated chart."""
    from runner import ask_baseline, ask_with_skill
    from evaluator import relaxed_correctness

    # Create test chart
    print("Creating test chart...")
    image_base64 = create_test_chart()
    if image_base64 is None:
        print("Could not create test chart")
        return

    print(f"Chart created (base64 length: {len(image_base64)})")

    # Test cases
    test_cases = [
        {
            "questions": ["What is the value for Apple?"],
            "answers": ["45"],
            "question_type": "Reasoning",
            "year_flags": ["NO"],
        },
        {
            "questions": ["Which fruit has the highest quantity?"],
            "answers": ["Apple"],
            "question_type": "Reasoning",
            "year_flags": ["NO"],
        },
        {
            "questions": ["What is the total quantity of all fruits?"],
            "answers": ["124"],  # 45+32+28+19=124
            "question_type": "Reasoning",
            "year_flags": ["NO"],
        },
        {
            "questions": ["Apple has more quantity than Banana. True or False?"],
            "answers": ["True"],
            "question_type": "Fact Checking",
            "year_flags": ["NO"],
        },
    ]

    print("\n" + "="*60)
    print("Testing ChartQAPro Pipeline")
    print("="*60)

    results_baseline = []
    results_skill = []

    for i, tc in enumerate(test_cases):
        questions = tc["questions"]
        answers = tc["answers"]
        question_type = tc["question_type"]
        year_flags = tc["year_flags"]

        print(f"\n[{i+1}] Q: {questions[0]}")
        print(f"    Expected: {answers[0]}")

        # Baseline
        try:
            pred_baseline = ask_baseline(image_base64, questions, question_type)
            score_baseline = relaxed_correctness(answers, pred_baseline, year_flags, question_type)
            print(f"    Baseline: {pred_baseline[0]} -> {score_baseline:.2f}")
            results_baseline.append(score_baseline)
        except Exception as e:
            print(f"    Baseline: ERROR - {e}")
            results_baseline.append(0.0)

        # With skill
        try:
            pred_skill, extracted = ask_with_skill(image_base64, questions, question_type)
            score_skill = relaxed_correctness(answers, pred_skill, year_flags, question_type)
            print(f"    Skill:    {pred_skill[0]} -> {score_skill:.2f}")
            results_skill.append(score_skill)

            # Show extracted data for first case
            if i == 0:
                print("\n    [Extracted Data Preview]")
                for line in extracted.split('\n')[:10]:
                    print(f"    {line}")
        except Exception as e:
            print(f"    Skill:    ERROR - {e}")
            results_skill.append(0.0)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    avg_baseline = sum(results_baseline) / len(results_baseline)
    avg_skill = sum(results_skill) / len(results_skill)
    print(f"Baseline: {avg_baseline:.1%}")
    print(f"Skill:    {avg_skill:.1%}")
    print(f"Improvement: {avg_skill - avg_baseline:+.1%}")


if __name__ == "__main__":
    test_pipeline()
