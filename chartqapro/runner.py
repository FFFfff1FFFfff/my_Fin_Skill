#!/usr/bin/env python3
"""
ChartQAPro benchmark runner: compare baseline vs with-skill performance
Chart Question Answering with visual and logical reasoning

Skill approach: Chain-of-Thought (CoT) with structured output
Based on paper findings: CoT > PoT > Direct for closed-source models
"""

import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_chartqapro, load_sample_data, get_question_type_stats
from evaluator import relaxed_correctness, evaluate_batch

client = Anthropic()


def create_image_message(image_base64: str, text: str) -> list:
    """Create a message with image content for Claude API."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
            {"type": "text", "text": text}
        ]
    }]


def extract_final_answer(text: str) -> str:
    """Extract answer from 'Final Answer: xxx' format."""
    # Try to find "Final Answer:" pattern
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
    else:
        # Fallback: take last non-empty line
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        answer = lines[-1] if lines else text.strip()

    # Clean up
    for prefix in ["Answer:", "The answer is", "A:", "**"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    return answer.rstrip('*').strip()


# =============================================================================
# BASELINE: Direct question answering (no CoT)
# =============================================================================

def ask_baseline(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929") -> list:
    """Baseline: direct questions without CoT."""
    format_hints = {
        "Fact Checking": "Output ONLY 'True' or 'False'.",
        "Multi Choice": "Output ONLY the letter (A, B, C, or D).",
        "Reasoning": "Output ONLY the number or value.",
        "Hypothetical": "Output ONLY the short answer.",
        "Conversational": "Output ONLY the short answer.",
    }

    answers = []
    conversation_history = []

    for question in questions:
        hint = format_hints.get(question_type, "Output ONLY the answer.")
        context = ""
        if question_type == "Conversational" and conversation_history:
            context = "Previous:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in conversation_history) + "\n\n"

        prompt = f"""{context}Question: {question}

{hint} No explanation.

Answer:"""

        response = client.messages.create(
            model=model, max_tokens=50, temperature=0,
            messages=create_image_message(image_base64, prompt)
        )

        answer = response.content[0].text.strip().split('\n')[0].strip()
        for prefix in ["Answer:", "The answer is", "A:", "**"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        answer = answer.rstrip('*').strip()

        answers.append(answer)
        conversation_history.append((question, answer))

    return answers


# =============================================================================
# SKILL: Chain-of-Thought (CoT)
# =============================================================================

def ask_with_cot(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929") -> list:
    """
    Skill: Chain-of-Thought reasoning.

    Paper finding: CoT significantly outperforms direct answering for closed-source models.
    Claude Sonnet 3.5 achieved highest accuracy (55.81%) with CoT.
    """
    format_rules = {
        "Fact Checking": "Your final answer must be exactly 'True' or 'False'.",
        "Multi Choice": "Your final answer must be exactly one letter: A, B, C, or D.",
        "Reasoning": "Your final answer must be a specific number or value.",
        "Hypothetical": "Your final answer must be a concise answer.",
        "Conversational": "Your final answer must be a concise answer.",
    }

    answers = []
    conversation_history = []

    for question in questions:
        rule = format_rules.get(question_type, "Your final answer must be concise.")
        context = ""
        if question_type == "Conversational" and conversation_history:
            context = "Previous Q&A:\n" + "\n".join(f"Q: {q} → A: {a}" for q, a in conversation_history) + "\n\n"

        prompt = f"""{context}Question: {question}

Think step by step:
1. What specific data do I need to read from the chart?
2. Read those values carefully from the chart
3. Apply reasoning/calculation if needed
4. Verify the answer makes sense

{rule}
If the chart does not contain enough information, answer "Cannot determine".

End your response with:
Final Answer: [your answer]"""

        response = client.messages.create(
            model=model, max_tokens=500, temperature=0,
            messages=create_image_message(image_base64, prompt)
        )

        answer = extract_final_answer(response.content[0].text)
        answers.append(answer)
        conversation_history.append((question, answer))

    return answers


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(limit: int = None,
                  question_types: list = None,
                  model: str = "claude-sonnet-4-5-20250929",
                  use_sample: bool = False,
                  use_cot: bool = True):
    """
    Run benchmark comparing baseline (direct) vs CoT performance.
    """
    print("=" * 70)
    print("ChartQAPro Skill Benchmark")
    print("=" * 70)
    print("\nSkill: Chain-of-Thought (CoT) reasoning")

    # Load data
    print(f"\nLoading data (limit={limit})...")
    if use_sample:
        samples = load_sample_data()
    else:
        samples = load_chartqapro(limit=limit, question_types=question_types)
    print(f"Loaded {len(samples)} samples")

    # Show question type distribution
    stats = get_question_type_stats(samples)
    print("\nQuestion Type Distribution:")
    for q_type, count in sorted(stats.items()):
        print(f"  {q_type}: {count}")

    # Filter out samples without images
    samples = [s for s in samples if s["image_base64"] is not None]
    print(f"\nSamples with images: {len(samples)}")

    if not samples:
        print("No samples to process. Exiting.")
        return None

    results_baseline = []
    results_cot = []

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        questions = sample["questions"]
        answers = sample["answers"]
        question_type = sample["question_type"]
        year_flags = sample["year_flags"]
        image_base64 = sample["image_base64"]

        print(f"\n[{i+1}/{len(samples)}] Type: {question_type}")
        print(f"Q: {questions[0][:60]}..." if len(questions[0]) > 60 else f"Q: {questions[0]}")
        print(f"A: {answers[-1]}")

        # Baseline (direct)
        try:
            pred_baseline = ask_baseline(image_base64, questions, question_type, model)
            score_baseline = relaxed_correctness(answers, pred_baseline, year_flags, question_type)
            print(f"Direct: {pred_baseline[-1][:40]} -> {score_baseline:.2f}")
        except Exception as e:
            pred_baseline = [""] * len(questions)
            score_baseline = 0.0
            print(f"Direct: ERROR - {e}")

        results_baseline.append({
            "id": sample["id"],
            "questions": questions,
            "answers": answers,
            "question_type": question_type,
            "year_flags": year_flags,
            "predictions": pred_baseline,
            "score": score_baseline
        })

        # With CoT
        if use_cot:
            try:
                pred_cot = ask_with_cot(image_base64, questions, question_type, model)
                score_cot = relaxed_correctness(answers, pred_cot, year_flags, question_type)
                print(f"CoT:    {pred_cot[-1][:40]} -> {score_cot:.2f}")
            except Exception as e:
                pred_cot = [""] * len(questions)
                score_cot = 0.0
                print(f"CoT:    ERROR - {e}")

            results_cot.append({
                "id": sample["id"],
                "questions": questions,
                "answers": answers,
                "question_type": question_type,
                "year_flags": year_flags,
                "predictions": pred_cot,
                "score": score_cot
            })

    # Evaluate
    eval_baseline = evaluate_batch(results_baseline)
    eval_cot = evaluate_batch(results_cot) if results_cot else None

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nDirect (no CoT):")
    print(f"  Accuracy: {eval_baseline['accuracy']:.1%} ({eval_baseline['total_score']:.1f}/{eval_baseline['total']})")
    print(f"  By Question Type:")
    for q_type, acc in sorted(eval_baseline['by_type'].items()):
        detail = eval_baseline['by_type_detail'][q_type]
        print(f"    {q_type}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

    if eval_cot:
        print(f"\nWith CoT:")
        print(f"  Accuracy: {eval_cot['accuracy']:.1%} ({eval_cot['total_score']:.1f}/{eval_cot['total']})")
        print(f"  By Question Type:")
        for q_type, acc in sorted(eval_cot['by_type'].items()):
            detail = eval_cot['by_type_detail'][q_type]
            print(f"    {q_type}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

        improvement = eval_cot['accuracy'] - eval_baseline['accuracy']
        print(f"\nImprovement: {improvement:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "num_samples": len(samples),
        "direct": {
            "accuracy": eval_baseline['accuracy'],
            "by_type": eval_baseline['by_type'],
            "results": results_baseline
        },
    }

    if eval_cot:
        output["cot"] = {
            "accuracy": eval_cot['accuracy'],
            "by_type": eval_cot['by_type'],
            "results": results_cot
        }
        output["improvement"] = eval_cot['accuracy'] - eval_baseline['accuracy']

    output_file = f"chartqapro_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChartQAPro Skill Benchmark")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of samples (default: all)")
    parser.add_argument("--type", type=str, action="append", dest="types",
                        help="Filter by question type (can specify multiple)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample data for testing")
    parser.add_argument("--no-cot", action="store_true",
                        help="Run direct only, skip CoT")

    args = parser.parse_args()
    run_benchmark(
        limit=args.limit,
        question_types=args.types,
        model=args.model,
        use_sample=args.sample,
        use_cot=not args.no_cot
    )
