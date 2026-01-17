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
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_chartqapro, load_sample_data, get_question_type_stats
from evaluator import relaxed_correctness, evaluate_batch

client = Anthropic()


# =============================================================================
# STAGE PARSING: Extract structured reasoning stages from CoT response
# =============================================================================

def parse_cot_stages(response_text: str) -> dict:
    """
    Parse CoT response to extract structured reasoning stages.

    Expected format in response:
    [DATA]: ...
    [READ]: ...
    [CALC]: ...
    [VERIFY]: ...
    Final Answer: ...
    """
    stages = {
        "data": None,      # What data is needed
        "read": None,      # Values read from chart
        "calc": None,      # Calculation/reasoning
        "verify": None,    # Verification
        "final": None,     # Final answer
        "raw": response_text,  # Full response for debugging
    }

    # Extract each stage using regex
    patterns = {
        "data": r'\[DATA\]:\s*(.+?)(?=\[READ\]|\[CALC\]|\[VERIFY\]|Final Answer:|$)',
        "read": r'\[READ\]:\s*(.+?)(?=\[CALC\]|\[VERIFY\]|Final Answer:|$)',
        "calc": r'\[CALC\]:\s*(.+?)(?=\[VERIFY\]|Final Answer:|$)',
        "verify": r'\[VERIFY\]:\s*(.+?)(?=Final Answer:|$)',
        "final": r'Final Answer:\s*(.+?)(?:\n|$)',
    }

    for stage, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Filter out cases where LLM repeated the prompt question
            if content and not content.startswith("What specific data") and \
               not content.startswith("List the actual") and \
               not content.startswith("Show any calculation") and \
               not content.startswith("Quick sanity"):
                stages[stage] = content

    # More robust Final Answer extraction
    if not stages["final"]:
        # Try alternative patterns
        alt_patterns = [
            r'Final Answer:\s*\*?\*?(.+?)\*?\*?\s*$',  # Handle **answer**
            r'(?:The answer is|Answer:)\s*(.+?)(?:\n|$)',
            r'\n([A-D])\s*$',  # Single letter at end for multi-choice
            r'\n(True|False)\s*$',  # True/False at end
            r'\n(\d+(?:\.\d+)?)\s*$',  # Number at end
        ]
        for pat in alt_patterns:
            match = re.search(pat, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                stages["final"] = match.group(1).strip().strip('*').strip()
                break

    return stages


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
    """Extract answer from 'Final Answer: xxx' format and clean it."""
    # Try to find "Final Answer:" pattern
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
    else:
        # Fallback: take last non-empty line
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        answer = lines[-1] if lines else text.strip()

    # Clean up common prefixes
    for prefix in ["Answer:", "The answer is", "A:", "**", "answer:"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    # Remove trailing markers
    answer = answer.rstrip('*').rstrip('.').strip()

    # Strip common unit suffixes and parenthetical notes
    # e.g., "40 players" -> "40", "9070.001 PKR in Million" -> "9070.001"
    answer = re.sub(r'\s*\([^)]*\)\s*$', '', answer)  # Remove trailing (...)
    answer = re.sub(r'\s+(players?|years?|times?|points?|percentage points?|percent|%|dollars?|million|billion|PKR|USD|EUR|GBP|in Million|in Billion)\s*$', '', answer, flags=re.IGNORECASE)

    # If answer starts with a description, try to extract just the value
    # e.g., "Rest of Ontario has the highest..." -> "Rest of Ontario"
    if ' has ' in answer.lower() or ' is ' in answer.lower() or ' shows ' in answer.lower():
        # Try to get text before "has/is/shows"
        for sep in [' has ', ' is ', ' shows ', ' had ', ' was ']:
            if sep in answer.lower():
                idx = answer.lower().find(sep)
                potential = answer[:idx].strip()
                if potential:
                    answer = potential
                    break

    return answer.strip()


# =============================================================================
# BASELINE: Direct question answering (no CoT)
# =============================================================================

def ask_baseline(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929",
                 verbose: bool = True) -> tuple:
    """
    Baseline: direct questions without CoT.

    Returns:
        tuple: (answers_list, trace_list) where trace contains call info
    """
    format_hints = {
        "Fact Checking": "Output ONLY 'True' or 'False'.",
        "Multi Choice": "Output ONLY the letter (A, B, C, or D).",
        "Reasoning": "Output ONLY the number or value.",
        "Hypothetical": "Output ONLY the short answer.",
        "Conversational": "Output ONLY the short answer.",
    }

    answers = []
    traces = []
    conversation_history = []

    for q_idx, question in enumerate(questions):
        hint = format_hints.get(question_type, "Output ONLY the answer.")
        context = ""
        if question_type == "Conversational" and conversation_history:
            context = "Previous:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in conversation_history) + "\n\n"

        prompt = f"""{context}Question: {question}

{hint} No explanation.

Answer:"""

        if verbose:
            print(f"      [Direct Q{q_idx+1}] LLM...", end="", flush=True)

        start_time = time.time()
        response = client.messages.create(
            model=model, max_tokens=50, temperature=0,
            messages=create_image_message(image_base64, prompt)
        )
        duration_ms = int((time.time() - start_time) * 1000)

        response_text = response.content[0].text
        answer = response_text.strip().split('\n')[0].strip()
        for prefix in ["Answer:", "The answer is", "A:", "**"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        answer = answer.rstrip('*').strip()

        if verbose:
            print(f" {len(response_text)}c {duration_ms}ms -> \"{answer}\"")

        # Build trace
        trace = {
            "question_idx": q_idx,
            "question": question,
            "prompt": prompt,
            "response": response_text,
            "extracted_answer": answer,
            "duration_ms": duration_ms,
            "response_length": len(response_text),
        }
        traces.append(trace)

        answers.append(answer)
        conversation_history.append((question, answer))

    return answers, traces


# =============================================================================
# SKILL: Chain-of-Thought (CoT) with Structured Output
# =============================================================================

def ask_with_cot(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929",
                 verbose: bool = True) -> tuple:
    """
    Skill: Chain-of-Thought reasoning with structured output format.

    Paper finding: CoT significantly outperforms direct answering for closed-source models.
    Claude Sonnet 3.5 achieved highest accuracy (55.81%) with CoT.

    Returns:
        tuple: (answers_list, trace_list) where trace contains detailed call info
    """
    format_rules = {
        "Fact Checking": "EXACTLY 'True' or 'False'",
        "Multi Choice": "EXACTLY one letter: A, B, C, or D",
        "Reasoning": "ONLY the number/value (no units)",
        "Hypothetical": "ONLY the short answer",
        "Conversational": "ONLY the short answer",
    }

    answers = []
    traces = []
    conversation_history = []

    for q_idx, question in enumerate(questions):
        rule = format_rules.get(question_type, "concise answer only")
        context = ""
        if question_type == "Conversational" and conversation_history:
            context = "Previous Q&A:\n" + "\n".join(f"Q: {q} → A: {a}" for q, a in conversation_history) + "\n\n"

        # Structured CoT prompt for easy parsing
        prompt = f"""{context}Question: {question}

You MUST follow this EXACT output format (do NOT repeat the questions, just fill in your analysis):

[DATA]: <describe what data you need to find>
[READ]: <list the actual values you read from the chart>
[CALC]: <show your calculation or reasoning>
[VERIFY]: <brief sanity check>

Final Answer: <your answer>

EXAMPLE of correct format:
[DATA]: I need the sales values for 2020 and 2021
[READ]: 2020 = 150, 2021 = 200
[CALC]: Difference = 200 - 150 = 50
[VERIFY]: Positive growth makes sense given the upward trend
Final Answer: 50

RULES:
- {rule}
- NO units in final answer (not "50 million", just "50")
- NO explanation after Final Answer
- If unsure, answer "Cannot determine"

Now analyze the chart:"""

        if verbose:
            print(f"      [CoT Q{q_idx+1}] LLM...", end="", flush=True)

        start_time = time.time()
        response = client.messages.create(
            model=model, max_tokens=600, temperature=0,
            messages=create_image_message(image_base64, prompt)
        )
        duration_ms = int((time.time() - start_time) * 1000)

        response_text = response.content[0].text
        stages = parse_cot_stages(response_text)
        answer = stages["final"] if stages["final"] else extract_final_answer(response_text)

        if verbose:
            print(f" {len(response_text)}c {duration_ms}ms")
            if stages["data"]:
                print(f"        → Data: {stages['data'][:60]}...")
            if stages["read"]:
                print(f"        → Read: {stages['read'][:60]}...")
            if stages["calc"]:
                print(f"        → Calc: {stages['calc'][:60]}...")
            if stages["final"]:
                print(f"        → Final: {stages['final']}")

        # Build trace for this call
        trace = {
            "question_idx": q_idx,
            "question": question,
            "prompt": prompt,
            "response": response_text,
            "stages": {
                "data": stages["data"],
                "read": stages["read"],
                "calc": stages["calc"],
                "verify": stages["verify"],
            },
            "extracted_answer": answer,
            "duration_ms": duration_ms,
            "response_length": len(response_text),
        }
        traces.append(trace)

        answers.append(answer)
        conversation_history.append((question, answer))

    return answers, traces


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

        print(f"\n[{i+1}/{len(samples)}] Type: {question_type} | ID: {sample['id']}")
        print(f"    Q: {questions[0][:70]}..." if len(questions[0]) > 70 else f"    Q: {questions[0]}")
        print(f"    A: {answers[-1]}")

        # Baseline (direct)
        baseline_trace = None
        try:
            pred_baseline, baseline_trace = ask_baseline(image_base64, questions, question_type, model)
            score_baseline = relaxed_correctness(answers, pred_baseline, year_flags, question_type)
            status = "✓" if score_baseline >= 1.0 else "✗"
            print(f"    [Direct Result] {pred_baseline[-1][:30]} -> {score_baseline:.2f} {status}")
        except Exception as e:
            pred_baseline = [""] * len(questions)
            score_baseline = 0.0
            print(f"    [Direct] ERROR - {e}")

        results_baseline.append({
            "id": sample["id"],
            "questions": questions,
            "ground_truth": answers,
            "question_type": question_type,
            "year_flags": year_flags,
            "predictions": pred_baseline,
            "score": score_baseline,
            "correct": score_baseline >= 1.0,
            "trace": baseline_trace,  # Full conversation trace
        })

        # With CoT
        if use_cot:
            cot_trace = None
            try:
                pred_cot, cot_trace = ask_with_cot(image_base64, questions, question_type, model)
                score_cot = relaxed_correctness(answers, pred_cot, year_flags, question_type)
                status = "✓" if score_cot >= 1.0 else "✗"
                print(f"    [CoT Result] {pred_cot[-1][:30]} -> {score_cot:.2f} {status}")
            except Exception as e:
                pred_cot = [""] * len(questions)
                score_cot = 0.0
                print(f"    [CoT] ERROR - {e}")

            results_cot.append({
                "id": sample["id"],
                "questions": questions,
                "ground_truth": answers,
                "question_type": question_type,
                "year_flags": year_flags,
                "predictions": pred_cot,
                "score": score_cot,
                "correct": score_cot >= 1.0,
                "trace": cot_trace,  # Full conversation trace with stages
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

    # Save results with full traces
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "meta": {
            "timestamp": timestamp,
            "model": model,
            "num_samples": len(samples),
            "question_types": list(stats.keys()),
        },
        "summary": {
            "direct": {
                "accuracy": eval_baseline['accuracy'],
                "by_type": eval_baseline['by_type'],
                "total_correct": sum(1 for r in results_baseline if r.get("correct")),
                "total": len(results_baseline),
            },
        },
        "traces": {
            "direct": results_baseline,
        },
    }

    if eval_cot:
        output["summary"]["cot"] = {
            "accuracy": eval_cot['accuracy'],
            "by_type": eval_cot['by_type'],
            "total_correct": sum(1 for r in results_cot if r.get("correct")),
            "total": len(results_cot),
        }
        output["summary"]["improvement"] = eval_cot['accuracy'] - eval_baseline['accuracy']
        output["traces"]["cot"] = results_cot

        # Add comparison trace (side-by-side)
        comparison = []
        for b, c in zip(results_baseline, results_cot):
            comparison.append({
                "id": b["id"],
                "question_type": b["question_type"],
                "question": b["questions"][0] if b["questions"] else "",
                "ground_truth": b["ground_truth"][-1] if b["ground_truth"] else "",
                "direct_pred": b["predictions"][-1] if b["predictions"] else "",
                "direct_correct": b.get("correct", False),
                "cot_pred": c["predictions"][-1] if c["predictions"] else "",
                "cot_correct": c.get("correct", False),
                "cot_improved": c.get("correct", False) and not b.get("correct", False),
                "cot_regressed": b.get("correct", False) and not c.get("correct", False),
            })
        output["comparison"] = comparison

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
