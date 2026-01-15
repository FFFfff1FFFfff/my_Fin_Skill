#!/usr/bin/env python3
"""
SpreadsheetBench Runner - PoT (Program of Thought) Style

Aligned with official inference scripts:
https://github.com/RUCKBReasoning/SpreadsheetBench/tree/main/inference

Settings:
- row_react_exec: Data preview + Multi-round (default, best performance)
- pure_react_exec: No preview + Multi-round (model explores on its own)
- react_exec: Data preview + Single-round (baseline)
- compare: Run both baseline and multi-round, then compare

Usage:
    python runner.py --limit 20 --setting row_react_exec --max-turns 5
    python runner.py --limit 20 --setting react_exec
    python runner.py --limit 20 --setting compare  # Compare baseline vs multi-round
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

import anthropic

# Add parent directory to path for imports
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


def run_pot(
    sample: dict,
    setting: str = "row_react_exec",
    max_turns: int = 5,
    model: str = DEFAULT_MODEL,
    test_input: str = None,
    test_output: str = None,
) -> tuple:
    """
    Run PoT (Program of Thought) inference.

    Args:
        sample: Sample data dict
        setting: "row_react_exec", "pure_react_exec", or "react_exec"
        max_turns: Maximum interaction rounds
        model: Model to use
        test_input: Input file path for execution
        test_output: Output file path for execution

    Returns:
        Tuple of (final_code, turns_used)
    """
    # Build initial prompt with output_path
    prompt = build_prompt(sample, setting=setting, max_turn_num=max_turns, output_path=test_output or "output.xlsx")
    messages = [{"role": "user", "content": prompt}]

    # Single-round mode (react_exec)
    if setting == "react_exec":
        response = call_llm(messages, model=model)
        return extract_code(response), 1

    # Multi-round mode (row_react_exec, pure_react_exec)
    final_code = None
    for turn in range(max_turns):
        # Get LLM response
        response = call_llm(messages, model=model)
        messages.append({"role": "assistant", "content": response})

        # Extract code
        code = extract_code(response)
        final_code = code

        # No test files - return after first response
        if not test_input or not test_output:
            return final_code, turn + 1

        # Remove old output before execution
        if os.path.exists(test_output):
            os.remove(test_output)

        # Execute code
        result = execute_code(code, test_input, test_output)

        # Format feedback
        feedback = format_exec_result(result, test_output)
        messages.append({"role": "user", "content": feedback})

        # Check termination: output file created
        if check_output_exists(test_output):
            return final_code, turn + 1

    return final_code, max_turns


def run_benchmark(
    limit: int = None,
    model: str = DEFAULT_MODEL,
    setting: str = "row_react_exec",
    max_turns: int = 5,
    use_sample: bool = False,
    data_dir: str = None,
    output_dir: str = None,
    instruction_types: list = None,
):
    """Run SpreadsheetBench benchmark."""

    # Determine settings to run
    if setting == "compare":
        settings_to_run = ["react_exec", "row_react_exec"]
    else:
        settings_to_run = [setting]

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
        samples = load_spreadsheetbench(
            data_dir=data_dir,
            limit=limit,
            instruction_types=instruction_types,
        )

    if not samples:
        print("No samples. Exiting.")
        return

    # Output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="spreadsheetbench_")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    # Results per setting
    all_results = {s: [] for s in settings_to_run}
    total_turns = {s: 0 for s in settings_to_run}

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] ID: {sample['id']}")
        print(f"  Type: {sample['instruction_type']}")
        print(f"  Task: {sample['instruction'][:80]}...")

        sample_dir = os.path.join(output_dir, str(sample['id']))
        os.makedirs(sample_dir, exist_ok=True)

        # Get test file for execution feedback
        test_input = None
        if sample['test_cases']:
            test_input = sample['test_cases'][0]['input_file']

        for run_setting in settings_to_run:
            test_output = os.path.join(sample_dir, f"{run_setting}_output.xlsx")

            try:
                code, turns = run_pot(
                    sample,
                    setting=run_setting,
                    max_turns=max_turns,
                    model=model,
                    test_input=test_input,
                    test_output=test_output,
                )
                total_turns[run_setting] += turns

                # Save code
                with open(os.path.join(sample_dir, f"{run_setting}_code.py"), "w") as f:
                    f.write(code)

                # Evaluate
                if sample['test_cases']:
                    eval_result = evaluate_instruction(
                        code=code,
                        test_cases=sample['test_cases'],
                        answer_position=sample['answer_position'],
                        output_dir=sample_dir,
                    )
                    eval_result["id"] = sample["id"]
                    eval_result["instruction_type"] = sample["instruction_type"]
                    eval_result["turns"] = turns
                    all_results[run_setting].append(eval_result)

                    status = "PASS" if eval_result["hard_restriction"] == 1 else "FAIL"
                    print(f"  [{run_setting}] {status} Soft:{eval_result['soft_restriction']:.0%} Turns:{turns}")
                else:
                    print(f"  [{run_setting}] (No test cases) Turns: {turns}")

            except Exception as e:
                print(f"  [{run_setting}] ERROR: {e}")
                if sample['test_cases']:
                    all_results[run_setting].append({
                        "id": sample["id"],
                        "instruction_type": sample["instruction_type"],
                        "soft_restriction": 0.0,
                        "hard_restriction": 0,
                        "error": str(e),
                    })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    metrics_by_setting = {}
    for run_setting in settings_to_run:
        results = all_results[run_setting]
        if results:
            metrics = calculate_metrics(results)
            metrics_by_setting[run_setting] = metrics
            label = "baseline" if run_setting == "react_exec" else run_setting
            print(f"\n[{label}]")
            print(f"  Soft Restriction: {metrics['soft_restriction_avg']:.1%}")
            print(f"  Hard Restriction: {metrics['hard_restriction_avg']:.1%}")
            print(f"  Avg Turns: {total_turns[run_setting] / len(samples):.1f}")

            if metrics['by_type']:
                for t, d in metrics['by_type'].items():
                    print(f"    {t}: Soft={d['soft_restriction_avg']:.1%}, Hard={d['hard_restriction_avg']:.1%}")

    # Comparison
    if len(settings_to_run) > 1 and all(s in metrics_by_setting for s in settings_to_run):
        print("\n" + "-" * 40)
        print("COMPARISON (multi-round vs baseline)")
        print("-" * 40)
        base = metrics_by_setting["react_exec"]
        multi = metrics_by_setting["row_react_exec"]
        soft_diff = multi['soft_restriction_avg'] - base['soft_restriction_avg']
        hard_diff = multi['hard_restriction_avg'] - base['hard_restriction_avg']
        print(f"  Soft: {soft_diff:+.1%}")
        print(f"  Hard: {hard_diff:+.1%}")

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "model": model,
            "settings": settings_to_run,
            "max_turns": max_turns,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics_by_setting,
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpreadsheetBench PoT Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--setting", type=str, default="row_react_exec",
                        choices=["row_react_exec", "pure_react_exec", "react_exec", "compare"],
                        help="Inference setting (compare: run baseline + multi-round)")
    parser.add_argument("--max-turns", type=int, default=5, help="Max interaction rounds")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--cell-level", action="store_true", help="Only Cell-Level tasks")
    parser.add_argument("--sheet-level", action="store_true", help="Only Sheet-Level tasks")

    args = parser.parse_args()

    instruction_types = None
    if args.cell_level:
        instruction_types = ["Cell-Level Manipulation"]
    elif args.sheet_level:
        instruction_types = ["Sheet-Level Manipulation"]

    run_benchmark(
        limit=args.limit,
        model=args.model,
        setting=args.setting,
        max_turns=args.max_turns,
        use_sample=args.sample,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        instruction_types=instruction_types,
    )
