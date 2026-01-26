#!/usr/bin/env python3
"""
ChartQAPro benchmark runner: compare baseline vs with-skill performance
Chart Question Answering with visual and logical reasoning

Uses official ChartQAPro paper prompt templates (Table 6 & 7)
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
# OFFICIAL PROMPT TEMPLATES FROM ChartQAPro PAPER (Table 6 & 7)
# =============================================================================

# Direct setup prompts (Table 6)
DIRECT_PROMPTS = {
    "Factoid": """You are given a factoid question that you need to answer based on the provided image.
Your answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!

STRICT FORMAT RULES:
- Output ONLY the answer, nothing else
- For years: use exact format shown in chart (e.g., "2020" not "2020年")
- For numbers: no units unless shown in chart (e.g., "50" not "50 million")
- For percentages: match chart format (e.g., "25" or "25%")

Question: {question}""",

    "Multi Choice": """You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.
Your answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!

STRICT FORMAT RULES:
- Output ONLY a single lowercase letter: a, b, c, or d
- No periods, no explanations, no "Option" prefix
- Example correct output: a
- Example wrong output: "a.", "Option a", "The answer is a"

Question: {question}""",

    "Hypothetical": """You are given a hypothetical question that you need to answer based on the provided image.
Your answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!

STRICT FORMAT RULES:
- Output ONLY the answer, nothing else
- For numbers: no units unless shown in chart
- Be precise and concise

Question: {question}""",

    "Fact Checking": """You are given a fact statement that you need to assess based on the provided image.
Your answer should be either true or false (without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!

STRICT FORMAT RULES:
- Output ONLY: true OR false (lowercase)
- No periods, no explanations
- Example correct: true
- Example wrong: "True", "TRUE", "true.", "The statement is true"

Question: {question}""",

    "Conversational": """You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.
Your answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!

STRICT FORMAT RULES:
- Output ONLY the answer, nothing else
- For numbers: no units unless shown in chart
- Be precise and concise

{conversation}
Question: {question}""",
}

# Chain of Thought prompts (Table 7)
COT_PROMPTS = {
    "Factoid": """You are given a factoid question that you need to answer based on the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What is the question asking?
2. [LOCATE]: Where in the chart is the relevant data?
3. [READ]: What are the exact values from the chart?
4. [CALCULATE]: Any calculations needed? (show work)
5. [VERIFY]: Does the answer make sense?

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X should be ONLY the answer (no units unless in chart)
- For years: exact format from chart
- For numbers: no extra units

Question: {question}""",

    "Multi Choice": """You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.
You need to think step-by-step, but your final answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What is the question asking?
2. [LOCATE]: Where in the chart is the relevant data?
3. [READ]: What are the exact values?
4. [EVALUATE]: Check each option against the data
5. [VERIFY]: Confirm the selected option

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X must be ONLY a single lowercase letter: a, b, c, or d
- Example: "The answer is b"

Question: {question}""",

    "Hypothetical": """You are given a hypothetical question that you need to answer based on the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What hypothetical scenario is being asked?
2. [LOCATE]: Where is the relevant baseline data?
3. [READ]: What are the current values?
4. [CALCULATE]: Apply the hypothetical change
5. [VERIFY]: Is the result reasonable?

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X should be ONLY the answer

Question: {question}""",

    "Fact Checking": """You are given a fact statement that you need to assess based on the information in the provided image.
You need to think step-by-step, but your final answer should be either true or false (without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What claim is being made?
2. [LOCATE]: Where is the relevant data in the chart?
3. [READ]: What are the actual values?
4. [COMPARE]: Does the data support or refute the claim?
5. [VERIFY]: Double-check the conclusion

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X must be ONLY: true OR false (lowercase)
- Example: "The answer is true"

Question: {question}""",

    "Conversational": """You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [CONTEXT]: What was discussed in previous turns?
2. [UNDERSTAND]: What is the current question asking?
3. [LOCATE]: Where is the relevant data?
4. [READ]: What are the exact values?
5. [VERIFY]: Does this follow logically from the conversation?

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X should be ONLY the answer

{conversation}
Question: {question}""",
}


# =============================================================================
# STAGE PARSING: Extract structured reasoning stages from CoT response
# =============================================================================

def parse_cot_stages(response_text: str) -> dict:
    """
    Parse CoT response to extract structured reasoning stages.

    Expected format in response (using official paper + stage markers):
    [UNDERSTAND]: ...
    [LOCATE]: ...
    [READ]: ...
    [CALCULATE]/[EVALUATE]/[COMPARE]: ...
    [VERIFY]: ...
    The answer is ...
    """
    stages = {
        "understand": None,   # What is the question asking
        "locate": None,       # Where in chart is data
        "read": None,         # Exact values read
        "calculate": None,    # Calculation/evaluation/comparison
        "verify": None,       # Verification
        "final": None,        # Final answer
        "raw": response_text, # Full response for debugging
    }

    # Extract each stage using regex
    patterns = {
        "understand": r'\[UNDERSTAND\]:\s*(.+?)(?=\[LOCATE\]|\[READ\]|\[CALC|\[EVAL|\[COMP|\[VERIFY\]|\[CONTEXT\]|The answer is|$)',
        "locate": r'\[LOCATE\]:\s*(.+?)(?=\[READ\]|\[CALC|\[EVAL|\[COMP|\[VERIFY\]|The answer is|$)',
        "read": r'\[READ\]:\s*(.+?)(?=\[CALC|\[EVAL|\[COMP|\[VERIFY\]|The answer is|$)',
        "calculate": r'\[(?:CALCULATE|EVALUATE|COMPARE)\]:\s*(.+?)(?=\[VERIFY\]|The answer is|$)',
        "verify": r'\[VERIFY\]:\s*(.+?)(?=The answer is|$)',
        "final": r'The answer is\s+(.+?)(?:\.|$)',
    }

    for stage, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content:
                stages[stage] = content

    # Also try to capture context stage for conversational
    context_match = re.search(r'\[CONTEXT\]:\s*(.+?)(?=\[UNDERSTAND\]|$)', response_text, re.DOTALL | re.IGNORECASE)
    if context_match:
        stages["context"] = context_match.group(1).strip()

    # More robust Final Answer extraction if not found
    if not stages["final"]:
        alt_patterns = [
            r'The answer is\s*[:\s]*(.+?)(?:\.|,|\n|$)',
            r'(?:^|\n)\s*([a-d])\s*$',  # Single letter at end for multi-choice
            r'(?:^|\n)\s*(true|false)\s*$',  # True/False at end
            r"(?:^|\n)\s*\[?'?([^'[\]]+)'?\]?\s*$",  # Last line as answer
        ]
        for pat in alt_patterns:
            match = re.search(pat, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                stages["final"] = match.group(1).strip().strip("'\"").strip()
                break

    return stages


def analyze_stage_metrics(stages: dict) -> dict:
    """
    Analyze stage completion metrics for a sample.

    Returns:
        dict with metrics:
        - stages_completed: list of completed stages
        - stages_in_order: bool (whether stages followed expected order)
        - total_stages: int
        - stage_completion_rate: float
    """
    # Updated stage order matching new CoT format
    expected_order = ["understand", "locate", "read", "calculate", "verify", "final"]
    first_appearance = {}

    for i, stage in enumerate(expected_order):
        if stages.get(stage):
            first_appearance[stage] = i

    completed_stages = list(first_appearance.keys())

    # Check if stages are in expected order
    stages_in_order = True
    prev_idx = -1
    for stage in completed_stages:
        curr_idx = expected_order.index(stage)
        if curr_idx < prev_idx:
            stages_in_order = False
            break
        prev_idx = curr_idx

    return {
        "stages_completed": completed_stages,
        "stages_in_order": stages_in_order,
        "total_stages": len(completed_stages),
        "stage_completion_rate": len(completed_stages) / len(expected_order),
        "expected_stages": expected_order,
    }


def create_image_message(image_base64: str, text: str) -> list:
    """Create a message with image content for Claude API."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
            {"type": "text", "text": text}
        ]
    }]


def extract_final_answer(text: str, question_type: str = None) -> str:
    """
    Extract answer from 'The answer is X' format (official paper format) and clean it.

    Args:
        text: Response text from LLM
        question_type: Type of question for format-specific extraction
    """
    answer = None

    # Primary: Official paper format "The answer is X"
    match = re.search(r'The answer is\s+(.+?)(?:\.|,|\n|$)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()

    # Fallback patterns
    if not answer:
        # Try "Final Answer:" pattern
        match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()

    if not answer:
        # Fallback: take last non-empty line
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        answer = lines[-1] if lines else text.strip()

    # Clean up common prefixes
    for prefix in ["Answer:", "The answer is", "A:", "**", "answer:", "="]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    # Remove trailing markers
    answer = answer.rstrip('*').rstrip('.').rstrip(',').strip()

    # Question-type specific cleaning
    if question_type == "Multi Choice":
        # Extract just the letter for multi-choice
        letter_match = re.search(r'^([a-dA-D])(?:\s|$|\)|\.|,)', answer)
        if letter_match:
            answer = letter_match.group(1).lower()
        elif answer.lower() in ['a', 'b', 'c', 'd']:
            answer = answer.lower()

    elif question_type == "Fact Checking":
        # Normalize true/false
        if answer.lower() in ['true', 'false']:
            answer = answer.lower()
        elif 'true' in answer.lower():
            answer = 'true'
        elif 'false' in answer.lower():
            answer = 'false'

    # Strip common unit suffixes and parenthetical notes (for other types)
    if question_type not in ["Multi Choice", "Fact Checking"]:
        answer = re.sub(r'\s*\([^)]*\)\s*$', '', answer)  # Remove trailing (...)
        answer = re.sub(r'\s+(players?|years?|times?|points?|percentage points?|percent|%|dollars?|million|billion|PKR|USD|EUR|GBP|in Million|in Billion)\s*$', '', answer, flags=re.IGNORECASE)

        # If answer starts with a description, try to extract just the value
        if ' has ' in answer.lower() or ' is ' in answer.lower() or ' shows ' in answer.lower():
            for sep in [' has ', ' is ', ' shows ', ' had ', ' was ']:
                if sep in answer.lower():
                    idx = answer.lower().find(sep)
                    potential = answer[:idx].strip()
                    if potential:
                        answer = potential
                        break

    return answer.strip()


# =============================================================================
# BASELINE: Direct question answering (no CoT) - Using Official Paper Templates
# =============================================================================

def ask_baseline(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929",
                 verbose: bool = True) -> tuple:
    """
    Baseline: direct questions without CoT using official paper prompts (Table 6).

    Returns:
        tuple: (answers_list, trace_list) where trace contains call info
    """
    answers = []
    traces = []
    conversation_history = []

    for q_idx, question in enumerate(questions):
        # Get official prompt template
        template_key = question_type if question_type in DIRECT_PROMPTS else "Factoid"
        template = DIRECT_PROMPTS[template_key]

        # Build conversation context for conversational questions
        conversation_str = ""
        if question_type == "Conversational" and conversation_history:
            conversation_str = "Conversation:\n" + "\n".join(
                f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(conversation_history)
            ) + "\n"

        # Format the prompt
        if question_type == "Conversational":
            prompt = template.format(conversation=conversation_str, question=question)
        else:
            prompt = template.format(question=question)

        if verbose:
            print(f"      [Direct Q{q_idx+1}] LLM...", end="", flush=True)

        start_time = time.time()
        response = client.messages.create(
            model=model, max_tokens=100, temperature=0,
            messages=create_image_message(image_base64, prompt)
        )
        duration_ms = int((time.time() - start_time) * 1000)

        response_text = response.content[0].text
        answer = extract_final_answer(response_text, question_type)

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
# SKILL: Chain-of-Thought (CoT) with Official Paper Templates + Stage Monitor
# =============================================================================

def ask_with_cot(image_base64: str, questions: list, question_type: str,
                 model: str = "claude-sonnet-4-5-20250929",
                 verbose: bool = True) -> tuple:
    """
    Skill: Chain-of-Thought reasoning using official paper prompts (Table 7).

    Paper finding: CoT significantly outperforms direct answering for closed-source models.
    Claude Sonnet 3.5 achieved highest accuracy (55.81%) with CoT.

    Stage markers added for monitoring:
    [UNDERSTAND] -> [LOCATE] -> [READ] -> [CALCULATE/EVALUATE/COMPARE] -> [VERIFY]
    Final format: "The answer is X"

    Returns:
        tuple: (answers_list, trace_list) where trace contains detailed call info with stage metrics
    """
    answers = []
    traces = []
    conversation_history = []

    for q_idx, question in enumerate(questions):
        # Get official prompt template
        template_key = question_type if question_type in COT_PROMPTS else "Factoid"
        template = COT_PROMPTS[template_key]

        # Build conversation context for conversational questions
        conversation_str = ""
        if question_type == "Conversational" and conversation_history:
            conversation_str = "Conversation:\n" + "\n".join(
                f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(conversation_history)
            ) + "\n"

        # Format the prompt
        if question_type == "Conversational":
            prompt = template.format(conversation=conversation_str, question=question)
        else:
            prompt = template.format(question=question)

        if verbose:
            print(f"      [CoT Q{q_idx+1}] LLM...", end="", flush=True)

        start_time = time.time()
        response = client.messages.create(
            model=model, max_tokens=800, temperature=0,
            messages=create_image_message(image_base64, prompt)
        )
        duration_ms = int((time.time() - start_time) * 1000)

        response_text = response.content[0].text
        stages = parse_cot_stages(response_text)
        answer = stages["final"] if stages["final"] else extract_final_answer(response_text, question_type)

        if verbose:
            print(f" {len(response_text)}c {duration_ms}ms")
            if stages.get("understand"):
                print(f"        → Understand: {stages['understand'][:60]}...")
            if stages.get("locate"):
                print(f"        → Locate: {stages['locate'][:60]}...")
            if stages.get("read"):
                print(f"        → Read: {stages['read'][:60]}...")
            if stages.get("calculate"):
                print(f"        → Calculate: {stages['calculate'][:60]}...")
            if stages.get("verify"):
                print(f"        → Verify: {stages['verify'][:60]}...")
            print(f"        → Final: {answer}")

        # Calculate stage metrics for monitoring
        stage_metrics = analyze_stage_metrics(stages)

        # Build trace for this call
        trace = {
            "question_idx": q_idx,
            "question": question,
            "question_type": question_type,
            "prompt": prompt,
            "response": response_text,
            "stages": {
                "understand": stages.get("understand"),
                "locate": stages.get("locate"),
                "read": stages.get("read"),
                "calculate": stages.get("calculate"),
                "verify": stages.get("verify"),
                "final": stages.get("final"),
            },
            "stage_metrics": stage_metrics,
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
                  offset: int = 0,
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
    load_limit = (offset + limit) if limit else None
    print(f"\nLoading data (offset={offset}, limit={limit})...")
    if use_sample:
        samples = load_sample_data()
    else:
        samples = load_chartqapro(limit=load_limit, question_types=question_types)

    # Apply offset
    if offset > 0:
        samples = samples[offset:]
        print(f"Skipped first {offset} samples")

    # Apply limit after offset
    if limit and len(samples) > limit:
        samples = samples[:limit]

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
    stage_metrics_list = []  # Collect stage metrics for aggregation

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

                # Collect stage metrics from each question's trace
                if cot_trace:
                    for trace_item in cot_trace:
                        if trace_item.get("stage_metrics"):
                            stage_metrics_list.append(trace_item["stage_metrics"])

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

    # Stage Monitor Summary
    print("\n" + "-" * 40)
    print("STAGE MONITOR SUMMARY")
    aggregated_stage_metrics = {}
    if stage_metrics_list:
        total = len(stage_metrics_list)
        in_order_count = sum(1 for s in stage_metrics_list if s.get("stages_in_order", False))
        avg_completion = sum(s.get("stage_completion_rate", 0) for s in stage_metrics_list) / total

        # Count each stage (updated to match new CoT format)
        expected_stages = ["understand", "locate", "read", "calculate", "verify", "final"]
        stage_counts = {stage: 0 for stage in expected_stages}
        for s in stage_metrics_list:
            for stage in s.get("stages_completed", []):
                if stage in stage_counts:
                    stage_counts[stage] += 1

        aggregated_stage_metrics = {
            "total_samples": total,
            "stages_in_order_count": in_order_count,
            "stages_in_order_rate": in_order_count / total,
            "avg_completion_rate": avg_completion,
            "stage_counts": stage_counts,
            "stage_rates": {k: v / total for k, v in stage_counts.items()},
        }

        print(f"\n[CoT Mode]")
        print(f"  Stages in order: {in_order_count}/{total} ({in_order_count/total:.1%})")
        print(f"  Avg completion rate: {avg_completion:.1%}")
        print(f"  Stage breakdown:")
        for stage, count in stage_counts.items():
            print(f"    {stage}: {count}/{total} ({count/total:.1%})")
    else:
        print("\n  No stage metrics collected (no successful CoT responses)")

    # Save results with full traces
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "meta": {
            "timestamp": timestamp,
            "model": model,
            "num_samples": len(samples),
            "question_types": list(stats.keys()),
            "skill": "chain_of_thought",
        },
        "summary": {
            "direct": {
                "accuracy": eval_baseline['accuracy'],
                "by_type": eval_baseline['by_type'],
                "total_correct": sum(1 for r in results_baseline if r.get("correct")),
                "total": len(results_baseline),
            },
        },
        "stage_monitor": aggregated_stage_metrics,
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
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N samples")
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
        offset=args.offset,
        question_types=args.types,
        model=args.model,
        use_sample=args.sample,
        use_cot=not args.no_cot
    )
