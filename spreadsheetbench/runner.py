#!/usr/bin/env python3
"""
SpreadsheetBench Runner - PoT (Program of Thought) Style

Aligned with official inference scripts:
https://github.com/RUCKBReasoning/SpreadsheetBench/tree/main/inference

Settings:
- row_react_exec: Data preview + Multi-round (optimized with task routing)
- pure_react_exec: No preview + Multi-round
- react_exec: Data preview + Single-round (baseline)
- compare: Run both baseline and multi-round, then compare

Usage:
    python runner.py --limit 20 --setting row_react_exec --max-turns 5
    python runner.py --limit 20 --setting compare
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_spreadsheetbench, load_sample_data
from evaluator import evaluate_instruction, calculate_metrics
from skills.spreadsheet_pot.pot_tools import (
    extract_code,
    execute_code,
    format_exec_result,
    check_output_exists,
    build_prompt,
)


client = anthropic.Anthropic()
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def call_llm(messages: list, model: str = DEFAULT_MODEL) -> str:
    """Call Claude API."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=messages,
    )
    return response.content[0].text


def run_pot(sample: dict, setting: str, max_turns: int, model: str,
            test_input: str, test_output: str) -> dict:
    """
    Run PoT inference and return detailed trace.

    Returns dict with: code, turns, rounds (list of round details)
    """
    prompt = build_prompt(sample, setting=setting, max_turn_num=max_turns, output_path=test_output or "output.xlsx")
    messages = [{"role": "user", "content": prompt}]

    trace = {"rounds": [], "final_code": None, "total_turns": 0}

    # Single-round mode
    if setting == "react_exec":
        print(f"      [R1] LLM call...", end="", flush=True)
        response = call_llm(messages, model=model)
        code = extract_code(response)
        print(f" {len(code)} chars code")

        trace["rounds"].append({
            "round": 1,
            "response": response,
            "code": code,
            "exec": None,
            "feedback": None,
        })
        trace["final_code"] = code
        trace["total_turns"] = 1
        return trace

    # Multi-round mode
    for turn in range(max_turns):
        round_num = turn + 1
        print(f"      [R{round_num}/{max_turns}] LLM...", end="", flush=True)

        response = call_llm(messages, model=model)
        messages.append({"role": "assistant", "content": response})
        code = extract_code(response)
        print(f" {len(code)}c", end="", flush=True)

        round_data = {
            "round": round_num,
            "response": response,
            "code": code,
            "exec": None,
            "feedback": None,
            "output_created": False,
        }

        if not test_input or not test_output:
            trace["rounds"].append(round_data)
            trace["final_code"] = code
            trace["total_turns"] = round_num
            print(" (no test file)")
            return trace

        # Remove old output
        if os.path.exists(test_output):
            os.remove(test_output)

        # Execute
        result = execute_code(code, test_input, test_output)
        feedback = format_exec_result(result, test_output)
        messages.append({"role": "user", "content": feedback})

        output_created = check_output_exists(test_output)
        status = "OK" if result["success"] else "ERR"
        file_status = "✓" if output_created else "✗"
        print(f" exec:{status} file:{file_status}")

        round_data["exec"] = {
            "success": result["success"],
            "stdout": result["output"][:500] if result["output"] else "",
            "error": result["error"][:500] if result["error"] else "",
        }
        round_data["feedback"] = feedback[:500]
        round_data["output_created"] = output_created
        trace["rounds"].append(round_data)
        trace["final_code"] = code
        trace["total_turns"] = round_num

        if output_created:
            return trace

    return trace


def run_benchmark(
    limit: int = None,
    offset: int = 0,
    model: str = DEFAULT_MODEL,
    setting: str = "row_react_exec",
    max_turns: int = 5,
    use_sample: bool = False,
    data_dir: str = None,
    dataset_type: str = "sample_200",
    output_file: str = None,
    instruction_types: list = None,
):
    """Run SpreadsheetBench benchmark. Output: single JSON file."""

    settings_to_run = ["react_exec", "row_react_exec"] if setting == "compare" else [setting]

    print("=" * 60)
    print("SpreadsheetBench - PoT Runner")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Settings: {', '.join(settings_to_run)}")
    if any(s != "react_exec" for s in settings_to_run):
        print(f"Max turns: {max_turns}")

    # Load data
    if use_sample:
        samples = load_sample_data()
        print("Using sample data (no evaluation)")
    else:
        # Load enough samples to cover offset + limit
        load_limit = (offset + limit) if limit else None
        samples = load_spreadsheetbench(
            data_dir=data_dir,
            dataset_type=dataset_type,
            limit=load_limit,
            instruction_types=instruction_types,
        )
        # Apply offset
        if offset > 0:
            samples = samples[offset:]
            print(f"Skipped first {offset} samples")
        # Apply limit after offset
        if limit and len(samples) > limit:
            samples = samples[:limit]

    if not samples:
        print("No samples. Exiting.")
        return

    # Output file
    if output_file is None:
        output_file = f"spreadsheetbench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"Output: {output_file}")

    # Temp dir for execution (will be cleaned up)
    temp_dir = tempfile.mkdtemp(prefix="ssbench_")

    # Collect all results
    all_traces = []
    metrics_by_setting = {s: [] for s in settings_to_run}
    total_turns = {s: 0 for s in settings_to_run}

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] ID: {sample['id']} | {sample['instruction_type']}")
        print(f"    Task: {sample['instruction'][:70]}...")

        test_input = sample['test_cases'][0]['input_file'] if sample.get('test_cases') else None

        sample_trace = {
            "id": sample["id"],
            "instruction": sample["instruction"],
            "instruction_type": sample["instruction_type"],
            "answer_position": sample["answer_position"],
            "settings": {},
        }

        for run_setting in settings_to_run:
            print(f"    [{run_setting}]")

            # Temp output file (not saved permanently)
            test_output = os.path.join(temp_dir, f"{sample['id']}_{run_setting}.xlsx")

            try:
                trace = run_pot(
                    sample, run_setting, max_turns, model, test_input, test_output
                )
                total_turns[run_setting] += trace["total_turns"]

                # Evaluate
                eval_result = None
                if sample.get('test_cases'):
                    eval_result = evaluate_instruction(
                        code=trace["final_code"],
                        test_cases=sample['test_cases'],
                        answer_position=sample['answer_position'],
                        output_dir=temp_dir,
                    )
                    metrics_by_setting[run_setting].append({
                        "id": sample["id"],
                        "instruction_type": sample["instruction_type"],
                        "soft_restriction": eval_result["soft_restriction"],
                        "hard_restriction": eval_result["hard_restriction"],
                        "turns": trace["total_turns"],
                    })

                    status = "PASS" if eval_result["hard_restriction"] == 1 else "FAIL"
                    print(f"      → {status} Soft:{eval_result['soft_restriction']:.0%} Turns:{trace['total_turns']}")
                else:
                    print(f"      → (no test) Turns:{trace['total_turns']}")

                sample_trace["settings"][run_setting] = {
                    "turns": trace["total_turns"],
                    "rounds": trace["rounds"],
                    "final_code": trace["final_code"],
                    "evaluation": eval_result,
                }

            except Exception as e:
                print(f"      → ERROR: {e}")
                sample_trace["settings"][run_setting] = {"error": str(e)}
                metrics_by_setting[run_setting].append({
                    "id": sample["id"],
                    "instruction_type": sample["instruction_type"],
                    "soft_restriction": 0.0,
                    "hard_restriction": 0,
                    "error": str(e),
                })

            # Clean up temp xlsx
            if os.path.exists(test_output):
                os.remove(test_output)

        all_traces.append(sample_trace)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    final_metrics = {}
    for run_setting in settings_to_run:
        results = metrics_by_setting[run_setting]
        if results:
            metrics = calculate_metrics(results)
            final_metrics[run_setting] = metrics
            label = "baseline" if run_setting == "react_exec" else run_setting
            print(f"\n[{label}]")
            print(f"  Soft: {metrics['soft_restriction_avg']:.1%}")
            print(f"  Hard: {metrics['hard_restriction_avg']:.1%}")
            print(f"  Avg Turns: {total_turns[run_setting] / len(samples):.1f}")
            if metrics.get('by_type'):
                for t, d in metrics['by_type'].items():
                    print(f"    {t}: S={d['soft_restriction_avg']:.1%} H={d['hard_restriction_avg']:.1%}")

    # Comparison
    if len(settings_to_run) > 1 and all(s in final_metrics for s in settings_to_run):
        print("\n" + "-" * 40)
        print("COMPARISON (multi-round vs baseline)")
        base = final_metrics["react_exec"]
        multi = final_metrics["row_react_exec"]
        print(f"  Soft: {multi['soft_restriction_avg'] - base['soft_restriction_avg']:+.1%}")
        print(f"  Hard: {multi['hard_restriction_avg'] - base['hard_restriction_avg']:+.1%}")

    # Save single JSON
    output_data = {
        "meta": {
            "model": model,
            "settings": settings_to_run,
            "max_turns": max_turns,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(samples),
        },
        "metrics": final_metrics,
        "traces": all_traces,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved to: {output_file}")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpreadsheetBench PoT Runner")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--setting", type=str, default="row_react_exec",
                        choices=["row_react_exec", "pure_react_exec", "react_exec", "compare"])
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="sample_200",
                        choices=["sample_200", "full_912", "verified_400"],
                        help="Dataset: sample_200, full_912, verified_400")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--cell-level", action="store_true")
    parser.add_argument("--sheet-level", action="store_true")

    args = parser.parse_args()

    instruction_types = None
    if args.cell_level:
        instruction_types = ["Cell-Level Manipulation"]
    elif args.sheet_level:
        instruction_types = ["Sheet-Level Manipulation"]

    run_benchmark(
        limit=args.limit,
        offset=args.offset,
        model=args.model,
        setting=args.setting,
        max_turns=args.max_turns,
        use_sample=args.sample,
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        output_file=args.output,
        instruction_types=instruction_types,
    )
