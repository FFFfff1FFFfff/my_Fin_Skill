#!/usr/bin/env python3
"""
TableBench benchmark runner: compare baseline vs with-skill performance

POT (Program of Thought) and visualization tools are loaded from:
  skills/tablebench_pot/pot_tools.py
"""

import json
import os
import sys
import re
import time
import traceback
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_tablebench, load_sample_data
from evaluator import evaluate_sample, evaluate_batch
from skill_system import SkillManager

# Import POT tools from skill
from skills.tablebench_pot.pot_tools import (
    extract_python_code,
    execute_code_safely,
    extract_answer_from_output,
    extract_answer_from_response,
    extract_chart_data,
    execute_chart_code,
    compare_chart_data,
    parse_viz_ground_truth,
    build_viz_prompt,
)

client = Anthropic()


# =============================================================================
# VISUALIZATION: LLM call for chart generation (uses skill tools for execution)
# =============================================================================

def ask_visualization(question: str, table: str, model: str = "claude-sonnet-4-5-20250929",
                      verbose: bool = True) -> tuple:
    """
    Generate matplotlib code for visualization task.

    Returns:
        tuple: (code: str, trace_dict)
    """
    # Use prompt from skill
    prompt = build_viz_prompt(question, table)

    if verbose:
        print(f"      [VIZ] LLM...", end="", flush=True)

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    duration_ms = int((time.time() - start_time) * 1000)

    full_response = response.content[0].text.strip()
    code = extract_python_code(full_response)

    if verbose:
        print(f" {len(full_response)}c {duration_ms}ms")
        if code:
            print(f"        → Code: {code.split(chr(10))[0][:50]}...")

    trace = {
        "prompt": prompt[:500] + "...",
        "response": full_response,
        "code": code,
        "duration_ms": duration_ms,
    }

    return code, trace


# =============================================================================
# STAGE PARSING: Extract key reasoning stages from TCoT response
# =============================================================================

KEY_STAGES = ["extract", "calculate", "answer"]


def parse_tcot_stages(response_text: str) -> dict:
    """Parse TCoT response to extract key stages: extract, calculate, answer."""
    stages = {"extract": None, "calculate": None, "answer": None, "raw": response_text}

    # EXTRACT
    match = re.search(r'(?:STEP\s*3|Extract|Data)[:\s]*(.+?)(?=STEP\s*4|Calculate|Final|$)',
                      response_text, re.DOTALL | re.IGNORECASE)
    if match:
        stages["extract"] = match.group(1).strip()[:100]

    # CALCULATE
    match = re.search(r'(?:STEP\s*4|Calculate|Calculation)[:\s]*(.+?)(?=STEP\s*5|Final|Answer|$)',
                      response_text, re.DOTALL | re.IGNORECASE)
    if match:
        stages["calculate"] = match.group(1).strip()[:100]

    # ANSWER
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if match:
        stages["answer"] = match.group(1).strip()[:100]

    return stages


def analyze_stage_metrics(stages: dict) -> dict:
    """Analyze stage completion metrics."""
    completed = [s for s in KEY_STAGES if stages.get(s)]
    return {"stages_completed": completed, "total_stages": len(completed)}


# Use extract_answer_from_response from skill (imported above as extract_answer_from_response)
def extract_answer(text: str) -> str:
    """Extract final answer from response. Delegates to skill tool."""
    return extract_answer_from_response(text)


def ask_baseline(question: str, table: str, instruction: str = "",
                 model: str = "claude-sonnet-4-5-20250929",
                 verbose: bool = True) -> tuple:
    """
    Baseline: Direct Prompting (DP) using official TableBench instruction.

    Args:
        question: The question (for logging only, already in instruction)
        table: The table (for logging only, already in instruction)
        instruction: Official instruction from TableBench_DP.jsonl
        model: Model to use
        verbose: Whether to print progress

    Returns:
        tuple: (extracted_answer, trace_dict)
    """
    # Use official instruction if provided, otherwise build fallback prompt
    if instruction:
        prompt = instruction
    else:
        # Fallback prompt matching official TableBench DP format
        # DP = Direct Prompting, expects direct answer without reasoning
        prompt = f"""You are a table analyst. Your task is to answer questions based on the table content.

The answer should follow the format below as the last line of your response:
Final Answer: AnswerName1, AnswerName2...

Ensure the final answer format is the last line. The answer should be a number or entity names, as short as possible.

Read the table below in JSON format:
{table}

Question: {question}

Analyze the table and give the final answer."""

    if verbose:
        print(f"      [Baseline] LLM...", end="", flush=True)

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=256,  # Keep small for DP (direct answer, no reasoning)
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    duration_ms = int((time.time() - start_time) * 1000)

    raw_answer = response.content[0].text.strip()
    extracted = extract_answer(raw_answer)

    if verbose:
        print(f" {len(raw_answer)}c {duration_ms}ms -> \"{extracted}\"")

    trace = {
        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        "response": raw_answer,
        "extracted_answer": extracted,
        "duration_ms": duration_ms,
        "response_length": len(raw_answer),
    }

    return extracted, trace


def ask_with_skill(question: str, table: str, skill_prompt: str,
                   model: str = "claude-sonnet-4-5-20250929",
                   verbose: bool = True) -> tuple:
    """
    TCoT (Textual Chain-of-Thought) - step by step reasoning without code execution.

    Returns:
        tuple: (extracted_answer, trace_dict)
    """
    # Official TCoT prompt template (enriched version)
    user_prompt = f"""You are a table analyst. Your task is to answer questions based on the table content.

The answer should follow the format below:
[Answer Format]
Final Answer: AnswerName1, AnswerName2...

**CRITICAL - ANSWER FORMAT RULES:**
- The Final Answer line MUST contain ONLY the answer values
- NO explanations, NO descriptions, NO reasoning in the Final Answer
- For "impact" questions: answer MUST be exactly "Positive impact", "Negative impact", or "No clear impact"
- For "which factors" questions: answer MUST be ONLY the column names, comma-separated
- Numbers and entity names should be as SHORT as possible

**Examples of CORRECT Final Answers:**
- Final Answer: Positive impact
- Final Answer: lost, points for, points against
- Final Answer: 450
- Final Answer: candidates

**Examples of WRONG Final Answers (DO NOT DO THIS):**
- Final Answer: There is a positive correlation between... ❌
- Final Answer: candidates, because they have the highest impact... ❌

Let's think step by step and then give the final answer to the question.

Read the table below in JSON format:
[TABLE]
{table}

Let's get start!
Question: {question}"""

    if verbose:
        print(f"      [Skill/TCoT] LLM...", end="", flush=True)

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        temperature=0,
        system=skill_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    duration_ms = int((time.time() - start_time) * 1000)

    full_response = response.content[0].text.strip()
    extracted = extract_answer(full_response)

    # Parse stages and calculate metrics
    stages = parse_tcot_stages(full_response)
    stage_metrics = analyze_stage_metrics(stages)

    if verbose:
        # Print stage info inline
        stages_str = "/".join([s.upper()[:3] for s in stage_metrics.get("stages_completed", [])])
        if stages_str:
            print(f" {len(full_response)}c {duration_ms}ms [{stages_str}]")
        else:
            print(f" {len(full_response)}c {duration_ms}ms")
        # Show brief reasoning trace
        lines = full_response.split('\n')
        for line in lines[:3]:
            if line.strip() and not line.startswith('Final'):
                print(f"        → {line[:60]}...")
                break
        print(f"        → Final: {extracted}")

    trace = {
        "prompt": user_prompt,
        "system_prompt": skill_prompt[:500] + "..." if len(skill_prompt) > 500 else skill_prompt,
        "response": full_response,
        "extracted_answer": extracted,
        "duration_ms": duration_ms,
        "response_length": len(full_response),
        "stages": {k: v for k, v in stages.items() if k != "raw"},
        "stage_metrics": stage_metrics,
    }

    return extracted, trace


def run_benchmark(source: str = "sample", limit: int = None, offset: int = 0,
                  model: str = "claude-sonnet-4-5-20250929", qtype: str = None):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.

    Args:
        source: "sample", "huggingface", or path to local file
        limit: Number of samples to test
        offset: Skip first N samples
        model: Model to use
        qtype: Filter by question type (FC/NR/DA/VIZ or full names)
    """
    # Question type mapping (short -> full)
    QTYPE_MAP = {
        "FC": "FactChecking",
        "NR": "NumericalReasoning",
        "DA": "DataAnalysis",
        "VIZ": "Visualization",
        # Also accept full names
        "FactChecking": "FactChecking",
        "NumericalReasoning": "NumericalReasoning",
        "DataAnalysis": "DataAnalysis",
        "Visualization": "Visualization",
    }

    # Special values for qtype
    SKIP_VIZ = ["NOVIZ", "ALL", "SKIPVIZ"]  # These skip VIZ and test others

    print("=" * 70)
    print("TableBench Skill Benchmark")
    print("=" * 70)

    # Load data
    load_limit = None  # Load all first, then filter
    print(f"\nLoading data (source={source}, offset={offset}, limit={limit}, qtype={qtype})...")

    # Determine if we need Visualization samples
    include_viz = False
    skip_viz_mode = False
    if qtype:
        qtype_upper = qtype.upper() if len(qtype) <= 5 else qtype
        if qtype_upper in ("VIZ", "Visualization"):
            include_viz = True
        elif qtype_upper in SKIP_VIZ:
            skip_viz_mode = True
            include_viz = False  # Don't load VIZ

    if source == "sample":
        samples = load_sample_data()
    else:
        samples = load_tablebench(source=source, limit=None, include_viz=include_viz)

    # Filter by question type if specified
    if qtype and not skip_viz_mode:
        qtype_full = QTYPE_MAP.get(qtype.upper() if len(qtype) <= 3 else qtype, qtype)
        samples = [s for s in samples if s["qtype"] == qtype_full]
        print(f"Filtered to qtype={qtype_full}: {len(samples)} samples")
    elif skip_viz_mode:
        # VIZ already excluded by include_viz=False, just print info
        print(f"Skip VIZ mode: testing FC, NR, DA ({len(samples)} samples)")

    # Apply offset
    if offset > 0:
        samples = samples[offset:]
        print(f"Skipped first {offset} samples")

    # Apply limit after offset
    if limit and len(samples) > limit:
        samples = samples[:limit]

    print(f"Loaded {len(samples)} samples")

    # Load skill
    skill_manager = SkillManager()
    skill_prompt = skill_manager.build_system_prompt(['table_reasoning'])
    print(f"Loaded skill: table_reasoning")

    # Results storage
    results_baseline = []
    results_skill = []
    stage_metrics_list = []  # Collect stage metrics for aggregation

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        qid = sample["id"]
        sample_qtype = sample["qtype"]
        qsubtype = sample["qsubtype"]
        question = sample["question"]
        table = sample["table"]
        ground_truth = sample["answer"]
        instruction = sample.get("instruction", "")  # Official DP instruction

        print(f"\n[{i+1}/{len(samples)}] {sample_qtype}/{qsubtype} | ID: {qid}")
        print(f"    Q: {question[:70]}..." if len(question) > 70 else f"    Q: {question}")
        print(f"    GT: {ground_truth}")

        # Handle Visualization separately
        if sample_qtype == "Visualization":
            # For Visualization: generate code, execute, and compare chart data
            baseline_trace = None
            skill_trace = None
            pred_data = []
            ref_data = []
            code = ""

            try:
                # Generate visualization code
                code, viz_trace = ask_visualization(question, table, model=model, verbose=True)

                if code:
                    # Execute code and extract chart data
                    success, pred_data, error = execute_chart_code(code)

                    # Parse reference data from ground_truth using skill tool
                    ref_data = parse_viz_ground_truth(ground_truth)
                    if not ref_data:
                        print(f"        [Warning] Failed to parse ref_data from ground_truth")

                    if success and pred_data and ref_data:
                        # Compare chart data (Pass@1)
                        is_correct = compare_chart_data(pred_data, ref_data)
                        correct_baseline = 1.0 if is_correct else 0.0
                        status = "✓" if is_correct else "✗"
                        print(f"    [VIZ Execute] Success - Pred: {pred_data[:3]}... Ref: {ref_data[:3]}... {status}")
                    elif success:
                        correct_baseline = 0.0
                        print(f"    [VIZ Execute] Code ran but data extraction failed")
                        print(f"        Pred data: {pred_data}")
                        print(f"        Ref data: {ref_data}")
                    else:
                        correct_baseline = 0.0
                        print(f"    [VIZ Execute] Failed - {error}")

                    baseline_trace = {
                        "code": code,
                        "success": success,
                        "pred_data": pred_data,
                        "ref_data": ref_data,
                        "error": error,
                        **viz_trace
                    }
                else:
                    correct_baseline = 0.0
                    print(f"    [VIZ] No code generated")
                    baseline_trace = viz_trace

            except Exception as e:
                correct_baseline = 0.0
                print(f"    [VIZ] ERROR - {e}")
                traceback.print_exc()
                baseline_trace = {"error": str(e)}

            # For Visualization, skill uses the same approach (no separate skill method for now)
            correct_skill = correct_baseline
            skill_trace = baseline_trace

            results_baseline.append({
                "id": qid,
                "qtype": sample_qtype,
                "qsubtype": qsubtype,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": str(pred_data) if pred_data else "",
                "score": correct_baseline,
                "trace": baseline_trace,
            })

            results_skill.append({
                "id": qid,
                "qtype": sample_qtype,
                "qsubtype": qsubtype,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": str(pred_data) if pred_data else "",
                "score": correct_skill,
                "trace": skill_trace,
            })

        else:
            # Non-Visualization: standard QA approach
            # Baseline - use official instruction
            baseline_trace = None
            try:
                pred_baseline, baseline_trace = ask_baseline(
                    question, table, instruction=instruction, model=model
                )
                correct_baseline = evaluate_sample(pred_baseline, ground_truth, sample_qtype, qsubtype)
                status = "✓" if correct_baseline >= 1.0 else "✗"
                print(f"    [Baseline Result] {pred_baseline} {status}")
            except Exception as e:
                pred_baseline = ""
                correct_baseline = 0.0
                print(f"    [Baseline] ERROR - {e}")

            results_baseline.append({
                "id": qid,
                "qtype": sample_qtype,
                "qsubtype": qsubtype,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": pred_baseline,
                "score": correct_baseline,  # Float score (0.0 to 1.0)
                "trace": baseline_trace,
            })

            # With skill
            skill_trace = None
            try:
                pred_skill, skill_trace = ask_with_skill(question, table, skill_prompt, model)
                correct_skill = evaluate_sample(pred_skill, ground_truth, sample_qtype, qsubtype)
                status = "✓" if correct_skill >= 1.0 else "✗"
                print(f"    [Skill Result] {pred_skill} {status}")

                # Collect stage metrics
                if skill_trace and skill_trace.get("stage_metrics"):
                    stage_metrics_list.append(skill_trace["stage_metrics"])
            except Exception as e:
                pred_skill = ""
                correct_skill = 0.0
                print(f"    [Skill] ERROR - {e}")

            results_skill.append({
                "id": qid,
                "qtype": sample_qtype,
                "qsubtype": qsubtype,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": pred_skill,
                "score": correct_skill,  # Float score (0.0 to 1.0)
                "trace": skill_trace,
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

    # Stage Monitor Summary
    print("\n" + "-" * 40)
    print("STAGE MONITOR")
    aggregated_stage_metrics = {}
    if stage_metrics_list:
        total = len(stage_metrics_list)
        stage_counts = {stage: 0 for stage in KEY_STAGES}
        for s in stage_metrics_list:
            for stage in s.get("stages_completed", []):
                if stage in stage_counts:
                    stage_counts[stage] += 1
        aggregated_stage_metrics = {"total": total, "stages": stage_counts}
        stage_str = ", ".join([f"{k}:{v}" for k, v in stage_counts.items()])
        print(f"  Stages: {stage_str}")

    # Save results with full traces
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "meta": {
            "timestamp": timestamp,
            "model": model,
            "source": source,
            "offset": offset,
            "num_samples": len(samples),
            "skill": "table_reasoning",
        },
        "summary": {
            "baseline": {
                "accuracy": acc_base,
                "total_correct": eval_baseline["overall"]["correct"],
                "total": eval_baseline["overall"]["total"],
                "by_type": eval_baseline["by_type"],
                "by_subtype": eval_baseline.get("by_subtype", {}),
            },
            "skill": {
                "accuracy": acc_skill,
                "total_correct": eval_skill["overall"]["correct"],
                "total": eval_skill["overall"]["total"],
                "by_type": eval_skill["by_type"],
                "by_subtype": eval_skill.get("by_subtype", {}),
            },
            "improvement": improvement,
        },
        "stage_monitor": aggregated_stage_metrics,
        "traces": {
            "baseline": results_baseline,
            "skill": results_skill,
        },
    }

    # Add comparison trace (side-by-side)
    comparison = []
    for b, s in zip(results_baseline, results_skill):
        b_score = b.get("score", 0.0)
        s_score = s.get("score", 0.0)
        comparison.append({
            "id": b["id"],
            "qtype": b["qtype"],
            "qsubtype": b["qsubtype"],
            "question": b["question"][:100] + "..." if len(b["question"]) > 100 else b["question"],
            "ground_truth": b["ground_truth"],
            "baseline_pred": b["prediction"],
            "baseline_score": b_score,
            "skill_pred": s["prediction"],
            "skill_score": s_score,
            "skill_improved": s_score > b_score,  # Skill scored higher
            "skill_regressed": s_score < b_score,  # Skill scored lower
        })
    output["comparison"] = comparison

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
    parser.add_argument("--limit", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929", help="Model to use")
    parser.add_argument("--qtype", type=str, default=None,
                        help="Filter by question type: FC, NR, DA, VIZ, or NOVIZ (skip VIZ, test others)")

    args = parser.parse_args()
    run_benchmark(source=args.source, limit=args.limit, offset=args.offset,
                  model=args.model, qtype=args.qtype)
