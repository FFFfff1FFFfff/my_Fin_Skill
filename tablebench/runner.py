#!/usr/bin/env python3
"""
TableBench benchmark runner: compare baseline vs with-skill performance
"""

import json
import os
import sys
import re
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_tablebench, load_sample_data
from evaluator import evaluate_sample, evaluate_batch
from skill_system import SkillManager

client = Anthropic()


def extract_answer(text: str) -> str:
    """Extract final answer from response."""
    # Look for "Answer:" pattern
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


def ask_baseline(question: str, table: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    """Baseline: direct question without skill enhancement."""
    prompt = f"""Answer this question about the table below. Give ONLY the final answer.

Table:
{table}

Question: {question}

Answer:"""

    response = client.messages.create(
        model=model,
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def ask_with_skill(question: str, table: str, skill_prompt: str,
                   model: str = "claude-sonnet-4-5-20250929") -> str:
    """With skill: question with skill-enhanced system prompt."""
    user_prompt = f"""Table:
{table}

Question: {question}

Follow the reasoning framework above. End with "Answer: [your answer]"."""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=skill_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    full_response = response.content[0].text.strip()
    return extract_answer(full_response)


def run_benchmark(source: str = "sample", limit: int = 50, model: str = "claude-sonnet-4-5-20250929"):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.

    Args:
        source: "sample", "huggingface", or path to local file
        limit: Number of samples to test
        model: Model to use
    """
    print("=" * 70)
    print("TableBench Skill Benchmark")
    print("=" * 70)

    # Load data
    print(f"\nLoading data (source={source}, limit={limit})...")
    if source == "sample":
        samples = load_sample_data()
    else:
        samples = load_tablebench(source=source, limit=limit)
    print(f"Loaded {len(samples)} samples (excluding Visualization)")

    # Load skill
    skill_manager = SkillManager()
    skill_prompt = skill_manager.build_system_prompt(['table-reasoning'])
    print(f"Loaded skill: table-reasoning")

    # Results storage
    results_baseline = []
    results_skill = []

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        qid = sample["id"]
        qtype = sample["qtype"]
        qsubtype = sample["qsubtype"]
        question = sample["question"]
        table = sample["table"]
        ground_truth = sample["answer"]

        print(f"\n[{i+1}/{len(samples)}] {qtype}/{qsubtype}")
        print(f"Q: {question[:60]}...")
        print(f"GT: {ground_truth}")

        # Baseline
        try:
            pred_baseline = ask_baseline(question, table, model)
            correct_baseline = evaluate_sample(pred_baseline, ground_truth, qtype)
            print(f"Baseline: {pred_baseline} {'✓' if correct_baseline else '✗'}")
        except Exception as e:
            pred_baseline = ""
            correct_baseline = False
            print(f"Baseline: ERROR - {e}")

        results_baseline.append({
            "id": qid,
            "qtype": qtype,
            "qsubtype": qsubtype,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred_baseline,
            "correct": correct_baseline
        })

        # With skill
        try:
            pred_skill = ask_with_skill(question, table, skill_prompt, model)
            correct_skill = evaluate_sample(pred_skill, ground_truth, qtype)
            print(f"Skill:    {pred_skill} {'✓' if correct_skill else '✗'}")
        except Exception as e:
            pred_skill = ""
            correct_skill = False
            print(f"Skill:    ERROR - {e}")

        results_skill.append({
            "id": qid,
            "qtype": qtype,
            "qsubtype": qsubtype,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred_skill,
            "correct": correct_skill
        })

    # Evaluate
    eval_baseline = evaluate_batch(results_baseline)
    eval_skill = evaluate_batch(results_skill)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    acc_base = eval_baseline["overall"]["accuracy"]
    acc_skill = eval_skill["overall"]["accuracy"]
    improvement = acc_skill - acc_base

    print(f"\nOverall Accuracy:")
    print(f"  Baseline:   {eval_baseline['overall']['correct']}/{eval_baseline['overall']['total']} ({acc_base:.1%})")
    print(f"  With Skill: {eval_skill['overall']['correct']}/{eval_skill['overall']['total']} ({acc_skill:.1%})")
    print(f"  Improvement: {improvement:+.1%}")

    print(f"\nBy Question Type:")
    all_types = set(eval_baseline["by_type"].keys()) | set(eval_skill["by_type"].keys())
    for qtype in sorted(all_types):
        base_acc = eval_baseline["by_type"].get(qtype, 0)
        skill_acc = eval_skill["by_type"].get(qtype, 0)
        base_detail = eval_baseline["by_type_detail"].get(qtype, {"correct": 0, "total": 0})
        skill_detail = eval_skill["by_type_detail"].get(qtype, {"correct": 0, "total": 0})
        diff = skill_acc - base_acc
        print(f"  {qtype}:")
        print(f"    Baseline:   {base_detail['correct']}/{base_detail['total']} ({base_acc:.1%})")
        print(f"    With Skill: {skill_detail['correct']}/{skill_detail['total']} ({skill_acc:.1%})")
        print(f"    Improvement: {diff:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "source": source,
        "num_samples": len(samples),
        "baseline": {
            "accuracy": acc_base,
            "eval": eval_baseline,
            "results": results_baseline
        },
        "skill": {
            "accuracy": acc_skill,
            "eval": eval_skill,
            "results": results_skill
        },
        "improvement": improvement
    }

    output_file = f"tablebench_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TableBench Skill Benchmark")
    parser.add_argument("--source", type=str, default="sample",
                        help="Data source: 'sample', 'huggingface', or path to local file")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929", help="Model to use")

    args = parser.parse_args()
    run_benchmark(source=args.source, limit=args.limit, model=args.model)
