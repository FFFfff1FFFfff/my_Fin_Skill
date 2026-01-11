#!/usr/bin/env python3
"""
ChartQAPro benchmark runner: compare baseline vs with-skill performance
Chart Question Answering with visual and logical reasoning
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_chartqapro, load_sample_data, get_question_type_stats
from evaluator import relaxed_correctness, evaluate_batch
from skill_system import SkillManager

client = Anthropic()


def create_image_message(image_base64: str, text: str, system_prompt: str = None) -> dict:
    """Create a message with image content for Claude API."""
    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_base64
            }
        },
        {
            "type": "text",
            "text": text
        }
    ]

    return {
        "messages": [{"role": "user", "content": user_content}],
        "system": system_prompt
    }


def ask_single_question(image_base64: str, question: str, question_type: str,
                        model: str = "claude-sonnet-4-5-20250929",
                        system_prompt: str = None,
                        conversation_history: list = None) -> str:
    """
    Ask a single question about a chart image.

    Args:
        image_base64: Base64 encoded chart image
        question: The question to ask
        question_type: Type of question for format hints
        model: Model to use
        system_prompt: Optional system prompt
        conversation_history: Previous Q&A pairs for conversational questions

    Returns:
        Model's answer
    """
    # Build prompt with format hints based on question type
    format_hints = {
        "Fact Checking": "Answer with only 'True' or 'False'.",
        "Multi Choice": "Answer with only the letter of the correct option (e.g., 'A', 'B', 'C', or 'D').",
        "Reasoning": "Provide a concise answer. For numbers, give just the number.",
        "Hypothetical": "Provide a concise answer based on the chart data.",
        "Conversational": "Answer concisely based on the chart and previous context.",
    }

    hint = format_hints.get(question_type, "Answer concisely.")

    # Build conversation context if available
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for prev_q, prev_a in conversation_history:
            context += f"Q: {prev_q}\nA: {prev_a}\n"
        context += "\n"

    prompt = f"""{context}Question: {question}

{hint}

Answer:"""

    msg_data = create_image_message(image_base64, prompt, system_prompt)

    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        system=msg_data["system"] if msg_data["system"] else [],
        messages=msg_data["messages"]
    )

    answer = response.content[0].text.strip()

    # Clean up answer - take first line only for most types
    lines = answer.split('\n')
    answer = lines[0].strip()

    # Remove common prefixes
    for prefix in ["Answer:", "The answer is", "A:"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    return answer


def ask_baseline(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929") -> list:
    """
    Baseline: direct questions without skills.

    For conversational questions, maintains context between turns.

    Returns:
        List of answers
    """
    answers = []
    conversation_history = []

    for question in questions:
        answer = ask_single_question(
            image_base64, question, question_type, model,
            conversation_history=conversation_history if question_type == "Conversational" else None
        )
        answers.append(answer)
        conversation_history.append((question, answer))

    return answers


def ask_with_skill(image_base64: str, questions: list, question_type: str,
                   skill_prompt: str, model: str = "claude-sonnet-4-5-20250929") -> list:
    """
    Answer with skill enhancement.

    Returns:
        List of answers
    """
    answers = []
    conversation_history = []

    for question in questions:
        answer = ask_single_question(
            image_base64, question, question_type, model,
            system_prompt=skill_prompt,
            conversation_history=conversation_history if question_type == "Conversational" else None
        )
        answers.append(answer)
        conversation_history.append((question, answer))

    return answers


def run_benchmark(limit: int = None,
                  question_types: list = None,
                  model: str = "claude-sonnet-4-5-20250929",
                  use_sample: bool = False):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.

    Args:
        limit: Number of samples to test
        question_types: Filter by question types
        model: Model to use
        use_sample: Use sample data instead of downloading
    """
    print("=" * 70)
    print("ChartQAPro Skill Benchmark")
    print("=" * 70)

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

    # Load skills (for future implementation)
    skill_manager = SkillManager()
    skill_names = ['chart_data_extractor']  # Future skill

    # Check if skills exist
    available_skills = skill_manager.list_skills()
    active_skills = [s for s in skill_names if s in available_skills]

    if active_skills:
        skill_prompt = skill_manager.build_system_prompt(active_skills)
        print(f"\nLoaded skills: {active_skills}")
    else:
        skill_prompt = None
        print("\nNo chart skills found. Running baseline only.")

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
        print(f"A: {answers[-1]}")  # Show final answer for conversational

        # Baseline
        try:
            pred_baseline = ask_baseline(image_base64, questions, question_type, model)
            score_baseline = relaxed_correctness(answers, pred_baseline, year_flags, question_type)
            print(f"Baseline: {pred_baseline[-1][:50]}... -> {score_baseline:.2f}")
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

        # With skill (if available)
        if skill_prompt:
            try:
                pred_skill = ask_with_skill(
                    image_base64, questions, question_type, skill_prompt, model
                )
                score_skill = relaxed_correctness(answers, pred_skill, year_flags, question_type)
                print(f"Skill:    {pred_skill[-1][:50]}... -> {score_skill:.2f}")
            except Exception as e:
                pred_skill = [""] * len(questions)
                score_skill = 0.0
                print(f"Skill:    ERROR - {e}")

            results_skill.append({
                "id": sample["id"],
                "questions": questions,
                "answers": answers,
                "question_type": question_type,
                "year_flags": year_flags,
                "predictions": pred_skill,
                "score": score_skill
            })

    # Evaluate
    eval_baseline = evaluate_batch(results_baseline)
    eval_skill = evaluate_batch(results_skill) if results_skill else None

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline:")
    print(f"  Accuracy: {eval_baseline['accuracy']:.1%} ({eval_baseline['total_score']:.1f}/{eval_baseline['total']})")
    print(f"  By Question Type:")
    for q_type, acc in sorted(eval_baseline['by_type'].items()):
        detail = eval_baseline['by_type_detail'][q_type]
        print(f"    {q_type}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

    if eval_skill:
        print(f"\nWith Skill:")
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

    args = parser.parse_args()
    run_benchmark(
        limit=args.limit,
        question_types=args.types,
        model=args.model,
        use_sample=args.sample
    )
