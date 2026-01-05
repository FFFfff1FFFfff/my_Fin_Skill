#!/usr/bin/env python3
"""
SealQA benchmark runner: compare baseline vs with-skill performance
Supports web search via built-in WebSearch or external APIs (Tavily/Serper)
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_sealqa, load_sample_data
from evaluator import grade_answer, evaluate_batch
from skill_system import SkillManager

client = Anthropic()


def ask_baseline(question: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    """Baseline: direct question without search or skills."""
    prompt = f"""Answer this question directly and concisely.

Question: {question}

Answer:"""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def ask_with_skill(question: str, skill_prompt: str, use_search: bool = True,
                   search_backend: str = "builtin", model: str = "claude-sonnet-4-5-20250929") -> str:
    """
    Answer with skill enhancement and optional web search.

    Args:
        question: The question to answer
        skill_prompt: System prompt with skill framework
        use_search: Whether to enable web search
        search_backend: "builtin", "tavily", or "serper"
        model: Model to use
    """
    if use_search and search_backend == "builtin":
        # Use Claude's native web search via tool
        user_prompt = f"""Question: {question}

Use web search to find current information if needed.
Follow the reasoning framework in the system prompt.
Give a direct, concise answer."""

        # Enable web search tool (correct format with name field)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            system=skill_prompt,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3
            }],
            messages=[{"role": "user", "content": user_prompt}]
        )

        # Extract text from response (may have tool use blocks)
        answer_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                answer_parts.append(block.text)
        return " ".join(answer_parts).strip() if answer_parts else "No answer generated"

    elif use_search and search_backend in ["tavily", "serper"]:
        # Use external search API
        from skills.web_search_tool.search_tools import search

        # First, do a search
        search_results = search(question, backend=search_backend)

        user_prompt = f"""Question: {question}

Search results:
{search_results}

Based on the search results above, answer the question.
Follow the reasoning framework in the system prompt.
Give a direct, concise answer."""

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=skill_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text.strip()

    else:
        # No search, just skill prompt
        user_prompt = f"""Question: {question}

Follow the reasoning framework in the system prompt.
Give a direct, concise answer."""

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=skill_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text.strip()


def run_benchmark(source: str = "sample", limit: int = 50,
                  use_search: bool = True, search_backend: str = "builtin",
                  model: str = "claude-sonnet-4-5-20250929",
                  grading_model: str = "claude-sonnet-4-5-20250929"):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.

    Args:
        source: "sample", "seal_0", "seal_hard", "longseal", or path to file
        limit: Number of samples to test
        use_search: Whether to use web search for skill mode
        search_backend: "builtin", "tavily", or "serper"
        model: Model for answering questions
        grading_model: Model for grading answers
    """
    print("=" * 70)
    print("SealQA Skill Benchmark")
    print("=" * 70)

    # Load data
    print(f"\nLoading data (source={source}, limit={limit})...")
    if source == "sample":
        samples = load_sample_data()
    else:
        samples = load_sealqa(source=source, limit=limit)
    print(f"Loaded {len(samples)} samples")

    # Load skills
    skill_manager = SkillManager()
    skill_names = ['web_search_tool', 'conflicting_info_reasoner']
    skill_prompt = skill_manager.build_system_prompt(skill_names)
    print(f"Loaded skills: {skill_names}")
    print(f"Search enabled: {use_search} (backend: {search_backend})")

    results_baseline = []
    results_skill = []

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        qid = sample["id"]
        question = sample["question"]
        gold_answer = sample["answer"]

        print(f"\n[{i+1}/{len(samples)}] {question[:60]}...")
        print(f"Gold: {gold_answer[:50]}...")

        # Baseline (no search, no skills)
        try:
            pred_baseline = ask_baseline(question, model)
            grade_baseline = grade_answer(question, gold_answer, pred_baseline, grading_model)
            grade_str = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[grade_baseline]
            print(f"Baseline: {pred_baseline[:50]}... -> {grade_str}")
        except Exception as e:
            pred_baseline = ""
            grade_baseline = "C"
            print(f"Baseline: ERROR - {e}")

        results_baseline.append({
            "id": qid,
            "question": question,
            "answer": gold_answer,
            "prediction": pred_baseline,
            "grade": grade_baseline
        })

        # With skill (and optionally search)
        try:
            pred_skill = ask_with_skill(question, skill_prompt, use_search, search_backend, model)
            grade_skill = grade_answer(question, gold_answer, pred_skill, grading_model)
            grade_str = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[grade_skill]
            print(f"Skill:    {pred_skill[:50]}... -> {grade_str}")
        except Exception as e:
            pred_skill = ""
            grade_skill = "C"
            print(f"Skill:    ERROR - {e}")

        results_skill.append({
            "id": qid,
            "question": question,
            "answer": gold_answer,
            "prediction": pred_skill,
            "grade": grade_skill
        })

    # Calculate metrics
    def calc_metrics(results):
        grades = [r["grade"] for r in results]
        total = len(grades)
        return {
            "correct": sum(g == 'A' for g in grades) / total if total > 0 else 0,
            "incorrect": sum(g == 'B' for g in grades) / total if total > 0 else 0,
            "not_attempted": sum(g == 'C' for g in grades) / total if total > 0 else 0,
            "total": total
        }

    metrics_baseline = calc_metrics(results_baseline)
    metrics_skill = calc_metrics(results_skill)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline (no search, no skills):")
    print(f"  Correct:       {metrics_baseline['correct']:.1%}")
    print(f"  Incorrect:     {metrics_baseline['incorrect']:.1%}")
    print(f"  Not Attempted: {metrics_baseline['not_attempted']:.1%}")

    print(f"\nWith Skill (search={use_search}, backend={search_backend}):")
    print(f"  Correct:       {metrics_skill['correct']:.1%}")
    print(f"  Incorrect:     {metrics_skill['incorrect']:.1%}")
    print(f"  Not Attempted: {metrics_skill['not_attempted']:.1%}")

    improvement = metrics_skill['correct'] - metrics_baseline['correct']
    print(f"\nImprovement: {improvement:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "source": source,
        "num_samples": len(samples),
        "search_enabled": use_search,
        "search_backend": search_backend,
        "baseline": {"metrics": metrics_baseline, "results": results_baseline},
        "skill": {"metrics": metrics_skill, "results": results_skill},
        "improvement": improvement
    }

    output_file = f"sealqa_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SealQA Skill Benchmark")
    parser.add_argument("--source", type=str, default="sample",
                        help="Data source: 'sample', 'seal_0', 'seal_hard', 'longseal', or path")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples")
    parser.add_argument("--search", action="store_true", default=True, help="Enable web search")
    parser.add_argument("--no-search", dest="search", action="store_false", help="Disable web search")
    parser.add_argument("--backend", type=str, default="builtin",
                        choices=["builtin", "tavily", "serper"], help="Search backend")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")

    args = parser.parse_args()
    run_benchmark(
        source=args.source,
        limit=args.limit,
        use_search=args.search,
        search_backend=args.backend,
        model=args.model
    )
