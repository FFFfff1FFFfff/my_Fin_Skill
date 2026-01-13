#!/usr/bin/env python3
"""
ChartQAPro benchmark runner: compare baseline vs with-skill performance
Chart Question Answering with visual and logical reasoning

Skill approach: Careful reading with verification
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_chartqapro, load_sample_data, get_question_type_stats
from evaluator import relaxed_correctness, evaluate_batch

client = Anthropic()


def create_image_message(image_base64: str, text: str) -> dict:
    """Create a message with image content for Claude API."""
    return {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                {"type": "text", "text": text}
            ]
        }]
    }


def extract_answer(text: str) -> str:
    """Extract clean answer from response."""
    answer = text.strip().split('\n')[0].strip()
    for prefix in ["Answer:", "The answer is", "A:", "answer:", "**", "Final answer:"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    return answer.rstrip('*').strip()


# =============================================================================
# BASELINE: Direct question answering
# =============================================================================

def ask_baseline(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929") -> list:
    """Baseline: direct questions without skills."""
    format_hints = {
        "Fact Checking": "Output ONLY 'True' or 'False'. No explanation.",
        "Multi Choice": "Output ONLY the letter (A, B, C, or D). No explanation.",
        "Reasoning": "Output ONLY the number or value. No explanation.",
        "Hypothetical": "Output ONLY the short answer. No explanation.",
        "Conversational": "Output ONLY the short answer. No explanation.",
    }

    answers = []
    conversation_history = []

    for question in questions:
        hint = format_hints.get(question_type, "Output ONLY the answer. No explanation.")
        context = ""
        if question_type == "Conversational" and conversation_history:
            context = "Previous:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in conversation_history) + "\n\n"

        prompt = f"""{context}Question: {question}

IMPORTANT: {hint}
Do NOT explain. Do NOT describe what you see. Just output the answer directly.

Answer:"""

        response = client.messages.create(
            model=model, max_tokens=50, temperature=0,
            messages=create_image_message(image_base64, prompt)["messages"]
        )
        answer = extract_answer(response.content[0].text)
        answers.append(answer)
        conversation_history.append((question, answer))

    return answers


# =============================================================================
# SKILL: Careful reading with verification
# =============================================================================

def ask_with_skill(image_base64: str, questions: list, question_type: str,
                   model: str = "claude-sonnet-4-5-20250929") -> tuple[list, str]:
    """
    Skill: Read carefully, verify, then answer.
    Single approach - no classification needed.
    """
    answers = []
    conversation_history = []

    for question in questions:
        context = ""
        if question_type == "Conversational" and conversation_history:
            context = "Previous:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in conversation_history) + "\n\n"

        prompt = f"""{context}Question: {question}

Instructions:
1. Read the chart carefully
2. Find the specific data points needed
3. Verify your reading is correct

Output ONLY the final answer. No explanation.

Answer:"""

        response = client.messages.create(
            model=model, max_tokens=50, temperature=0,
            messages=create_image_message(image_base64, prompt)["messages"]
        )

        answer = extract_answer(response.content[0].text)
        answers.append(answer)
        conversation_history.append((question, answer))

    return answers, "careful"


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(limit: int = None,
                  question_types: list = None,
                  model: str = "claude-sonnet-4-5-20250929",
                  use_sample: bool = False,
                  use_skill: bool = True):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.
    """
    print("=" * 70)
    print("ChartQAPro Skill Benchmark")
    print("=" * 70)
    print("\nSkill: Careful reading with verification")

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
    results_skill = []

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

        # Baseline
        try:
            pred_baseline = ask_baseline(image_base64, questions, question_type, model)
            score_baseline = relaxed_correctness(answers, pred_baseline, year_flags, question_type)
            print(f"Baseline: {pred_baseline[-1][:40]} -> {score_baseline:.2f}")
        except Exception as e:
            pred_baseline = [""] * len(questions)
            score_baseline = 0.0
            print(f"Baseline: ERROR - {e}")

        results_baseline.append({
            "id": sample["id"],
            "questions": questions,
            "answers": answers,
            "question_type": question_type,
            "year_flags": year_flags,
            "predictions": pred_baseline,
            "score": score_baseline
        })

        # With skill
        if use_skill:
            try:
                pred_skill, method = ask_with_skill(
                    image_base64, questions, question_type, model
                )
                score_skill = relaxed_correctness(answers, pred_skill, year_flags, question_type)
                print(f"Skill({method[:4]}): {pred_skill[-1][:35]} -> {score_skill:.2f}")
            except Exception as e:
                pred_skill = [""] * len(questions)
                method = ""
                score_skill = 0.0
                print(f"Skill:    ERROR - {e}")

            results_skill.append({
                "id": sample["id"],
                "questions": questions,
                "answers": answers,
                "question_type": question_type,
                "year_flags": year_flags,
                "predictions": pred_skill,
                "method": method,
                "score": score_skill
            })

    # Evaluate
    eval_baseline = evaluate_batch(results_baseline)
    eval_skill = evaluate_batch(results_skill) if results_skill else None

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline (direct):")
    print(f"  Accuracy: {eval_baseline['accuracy']:.1%} ({eval_baseline['total_score']:.1f}/{eval_baseline['total']})")
    print(f"  By Question Type:")
    for q_type, acc in sorted(eval_baseline['by_type'].items()):
        detail = eval_baseline['by_type_detail'][q_type]
        print(f"    {q_type}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

    if eval_skill:
        print(f"\nWith Skill (Careful):")
        print(f"  Accuracy: {eval_skill['accuracy']:.1%} ({eval_skill['total_score']:.1f}/{eval_skill['total']})")
        print(f"  By Question Type:")
        for q_type, acc in sorted(eval_skill['by_type'].items()):
            detail = eval_skill['by_type_detail'][q_type]
            print(f"    {q_type}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

        improvement = eval_skill['accuracy'] - eval_baseline['accuracy']
        print(f"\nImprovement: {improvement:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "num_samples": len(samples),
        "baseline": {
            "accuracy": eval_baseline['accuracy'],
            "by_type": eval_baseline['by_type'],
            "results": results_baseline
        },
    }

    if eval_skill:
        output["skill"] = {
            "accuracy": eval_skill['accuracy'],
            "by_type": eval_skill['by_type'],
            "results": results_skill
        }
        output["improvement"] = eval_skill['accuracy'] - eval_baseline['accuracy']

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
    parser.add_argument("--no-skill", action="store_true",
                        help="Run baseline only, skip skill")

    args = parser.parse_args()
    run_benchmark(
        limit=args.limit,
        question_types=args.types,
        model=args.model,
        use_sample=args.sample,
        use_skill=not args.no_skill
    )
