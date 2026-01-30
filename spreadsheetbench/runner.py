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
import re
import sys
import tempfile
import time
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


# =============================================================================
# STAGE MONITOR: Track reasoning stages in PoT multi-round flow
# =============================================================================

def parse_pot_stages(response_text: str, round_num: int, is_final_round: bool = False,
                     exec_success: bool = False) -> dict:
    """
    Parse PoT response to extract structured reasoning stages.

    Expected stages for SpreadsheetBench PoT:
    - Round 1-2: EXPLORE (list sheets, tables, columns - structure discovery)
    - Round 2+: IMPLEMENT (actual solution code with data transformation)
    - All rounds: EXECUTE (successful code execution)
    - Final round only: VERIFY (self-check after successful execution)

    Args:
        response_text: The LLM response text
        round_num: Current round number (1-indexed)
        is_final_round: Whether this is the final round
        exec_success: Whether code execution was successful this round
    """
    stages = {
        "explore": None,      # Structure exploration (Round 1-2)
        "implement": None,    # Solution implementation
        "execute": None,      # Code execution (set externally based on exec result)
        "verify": None,       # Self-check / verification (final round only)
        "raw": response_text,
    }

    # Extract code block for analysis
    code_match = re.search(r'```python\s*(.*?)```', response_text, re.DOTALL)
    code_block = code_match.group(1).strip() if code_match else ""

    # EXPLORE: Only in early rounds (1-2), must have actual structure discovery code
    if round_num <= 2 and code_block:
        explore_code_patterns = [
            r'\.sheet_names',                    # Getting sheet names
            r'\.sheetnames',                     # openpyxl sheetnames
            r'pd\.ExcelFile',                    # Excel file object
            r'\.keys\(\)',                       # Dict keys (sheet names)
            r'print\(.*sheet',                   # Printing sheet info
            r'print\(.*column',                  # Printing column info
            r'\.columns\.tolist\(\)',            # Getting column names
            r'\.head\(\)',                       # Previewing data
            r'\.info\(\)',                       # DataFrame info
            r'\.dtypes',                         # Column types
        ]
        for pattern in explore_code_patterns:
            if re.search(pattern, code_block, re.IGNORECASE):
                stages["explore"] = code_block[:300]
                break

    # IMPLEMENT: Must have actual data transformation/computation code
    if code_block:
        impl_code_patterns = [
            r'\.to_excel\(',                     # Writing to Excel
            r'\.save\(',                         # Saving workbook
            r'df\[.+\]\s*=',                     # DataFrame assignment
            r'ws\[.+\]\s*=',                     # Worksheet cell assignment
            r'\.loc\[.*\]\s*=',                  # DataFrame loc assignment
            r'\.iloc\[.*\]\s*=',                 # DataFrame iloc assignment
            r'\.apply\(',                        # Apply function
            r'\.merge\(',                        # Merge DataFrames
            r'\.concat\(',                       # Concatenate
            r'\.groupby\(',                      # Group by operations
            r'\.pivot',                          # Pivot operations
            r'\.sort_values\(',                  # Sorting
            r'\.drop\(',                         # Drop rows/columns
            r'\.fillna\(',                       # Fill NA values
            r'for\s+.*\s+in\s+.*:.*=',          # Loop with assignment
        ]
        for pattern in impl_code_patterns:
            if re.search(pattern, code_block, re.IGNORECASE | re.DOTALL):
                stages["implement"] = code_block[:300]
                break

    # VERIFY: Only count in final round OR when execution succeeded with explicit check
    if is_final_round or (exec_success and round_num >= 2):
        verify_code_patterns = [
            r'print\(.*(?:result|output|final|answer|done|complete)',  # Print results
            r'assert\s+',                        # Assertions
            r'==\s*expected',                    # Comparison with expected
            r'\.equals\(',                       # DataFrame equals check
        ]
        verify_text_patterns = [
            r'(?:verify|check|confirm|validate)\s+(?:the\s+)?(?:result|output|answer)',
            r'(?:successfully|completed|finished)',
            r'output\s+(?:file|spreadsheet)\s+(?:has been|is)\s+(?:created|saved|updated)',
        ]

        for pattern in verify_code_patterns:
            if re.search(pattern, code_block, re.IGNORECASE):
                stages["verify"] = True
                break

        if not stages["verify"]:
            for pattern in verify_text_patterns:
                if re.search(pattern, response_text, re.IGNORECASE):
                    stages["verify"] = True
                    break

    return stages


def analyze_stage_metrics(trace: dict) -> dict:
    """
    Analyze stage completion metrics for a sample.

    Strict order check: Stages must appear in sequence explore → implement → execute → verify.
    A stage is considered "first appeared" at the round where it was first detected.

    Returns:
        dict with metrics:
        - stages_completed: list of completed stages
        - stages_in_order: bool (whether stages followed strict expected order)
        - total_stages: int
        - stage_completion_rate: float
        - first_appearance: dict mapping stage to round number
    """
    expected_order = ["explore", "implement", "execute", "verify"]
    first_appearance = {}  # stage -> round_num when first detected

    rounds = trace.get("rounds", [])
    total_rounds = len(rounds)

    for i, round_data in enumerate(rounds):
        round_num = round_data.get("round", i + 1)
        response = round_data.get("response", "")
        is_final = (i == total_rounds - 1)
        exec_success = round_data.get("exec", {}).get("success", False)

        stages = parse_pot_stages(response, round_num, is_final, exec_success)

        # Record first appearance of each stage
        if stages["explore"] and "explore" not in first_appearance:
            first_appearance["explore"] = round_num
        if stages["implement"] and "implement" not in first_appearance:
            first_appearance["implement"] = round_num
        if exec_success and "execute" not in first_appearance:
            first_appearance["execute"] = round_num
        if stages["verify"] and "verify" not in first_appearance:
            first_appearance["verify"] = round_num

    # Build completed stages list in order of first appearance
    completed_stages = sorted(first_appearance.keys(), key=lambda s: first_appearance[s])

    # Strict order check: stages must appear in expected_order sequence
    # e.g., if we have [explore, implement, execute], check that explore < implement < execute
    stages_in_order = True
    prev_expected_idx = -1
    for stage in completed_stages:
        if stage in expected_order:
            curr_idx = expected_order.index(stage)
            if curr_idx < prev_expected_idx:
                # Stage appeared out of order
                stages_in_order = False
                break
            prev_expected_idx = curr_idx

    return {
        "stages_completed": completed_stages,
        "stages_in_order": stages_in_order,
        "total_stages": len(completed_stages),
        "stage_completion_rate": len(completed_stages) / len(expected_order),
        "expected_stages": expected_order,
        "first_appearance": first_appearance,
    }


client = anthropic.Anthropic()
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def call_llm(messages: list, model: str = DEFAULT_MODEL) -> tuple:
    """Call Claude API. Returns (response_text, duration_ms)."""
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=messages,
    )
    duration_ms = int((time.time() - start_time) * 1000)
    return response.content[0].text, duration_ms


def run_pot(sample: dict, setting: str, max_turns: int, model: str,
            test_input: str, test_output: str) -> dict:
    """
    Run PoT inference and return detailed trace with stage monitoring.

    Returns dict with: code, turns, rounds (list of round details), stage_metrics
    """
    prompt = build_prompt(sample, setting=setting, max_turn_num=max_turns, output_path=test_output or "output.xlsx")
    messages = [{"role": "user", "content": prompt}]

    trace = {"rounds": [], "final_code": None, "total_turns": 0, "stage_metrics": None}

    # Single-round mode
    if setting == "react_exec":
        print(f"      [R1] LLM call...", end="", flush=True)
        response, duration_ms = call_llm(messages, model=model)
        code = extract_code(response)
        # Single-round is always the final round
        stages = parse_pot_stages(response, round_num=1, is_final_round=True, exec_success=False)
        print(f" {len(code)}c {duration_ms}ms")

        # Print stage info
        if stages["explore"]:
            print(f"        → Explore: {stages['explore'][:50]}...")
        if stages["implement"]:
            print(f"        → Implement: {stages['implement'][:50]}...")

        trace["rounds"].append({
            "round": 1,
            "response": response,
            "code": code,
            "exec": None,
            "feedback": None,
            "duration_ms": duration_ms,
            "stages": {
                "explore": stages["explore"],
                "implement": stages["implement"],
                "verify": stages["verify"],
            },
        })
        trace["final_code"] = code
        trace["total_turns"] = 1
        trace["stage_metrics"] = analyze_stage_metrics(trace)
        return trace

    # Multi-round mode
    for turn in range(max_turns):
        round_num = turn + 1
        is_potentially_final = (round_num == max_turns)  # May end early if output created
        print(f"      [R{round_num}/{max_turns}] LLM...", end="", flush=True)

        response, duration_ms = call_llm(messages, model=model)
        messages.append({"role": "assistant", "content": response})
        code = extract_code(response)
        # For display, we don't know exec_success yet; analyze_stage_metrics does final analysis
        stages = parse_pot_stages(response, round_num, is_final_round=is_potentially_final, exec_success=False)
        print(f" {len(code)}c {duration_ms}ms", end="", flush=True)

        round_data = {
            "round": round_num,
            "response": response,
            "code": code,
            "exec": None,
            "feedback": None,
            "output_created": False,
            "duration_ms": duration_ms,
            "stages": {
                "explore": stages["explore"],
                "implement": stages["implement"],
                "verify": stages["verify"],
            },
        }

        # Print stage info inline
        stage_tags = []
        if stages["explore"]:
            stage_tags.append("EXPLORE")
        if stages["implement"]:
            stage_tags.append("IMPL")
        if stages["verify"]:
            stage_tags.append("VERIFY")
        if stage_tags:
            print(f" [{'/'.join(stage_tags)}]", end="")

        if not test_input or not test_output:
            trace["rounds"].append(round_data)
            trace["final_code"] = code
            trace["total_turns"] = round_num
            print(" (no test file)")
            trace["stage_metrics"] = analyze_stage_metrics(trace)
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
            trace["stage_metrics"] = analyze_stage_metrics(trace)
            return trace

    trace["stage_metrics"] = analyze_stage_metrics(trace)
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
    stage_metrics_by_setting = {s: [] for s in settings_to_run}  # Stage monitor aggregation

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

                stage_metrics = trace.get("stage_metrics")
                sample_trace["settings"][run_setting] = {
                    "turns": trace["total_turns"],
                    "rounds": trace["rounds"],
                    "final_code": trace["final_code"],
                    "evaluation": eval_result,
                    "stage_metrics": stage_metrics,
                }
                # Collect stage metrics for aggregation
                if stage_metrics:
                    stage_metrics_by_setting[run_setting].append(stage_metrics)

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

    # Stage Monitor Summary
    print("\n" + "-" * 40)
    print("STAGE MONITOR SUMMARY")
    aggregated_stage_metrics = {}
    for run_setting in settings_to_run:
        stage_list = stage_metrics_by_setting[run_setting]
        if stage_list:
            total = len(stage_list)
            in_order_count = sum(1 for s in stage_list if s.get("stages_in_order", False))
            avg_completion = sum(s.get("stage_completion_rate", 0) for s in stage_list) / total

            # Count each stage
            stage_counts = {"explore": 0, "implement": 0, "execute": 0, "verify": 0}
            for s in stage_list:
                for stage in s.get("stages_completed", []):
                    if stage in stage_counts:
                        stage_counts[stage] += 1

            aggregated = {
                "total_samples": total,
                "stages_in_order_count": in_order_count,
                "stages_in_order_rate": in_order_count / total,
                "avg_completion_rate": avg_completion,
                "stage_counts": stage_counts,
                "stage_rates": {k: v / total for k, v in stage_counts.items()},
            }
            aggregated_stage_metrics[run_setting] = aggregated

            label = "baseline" if run_setting == "react_exec" else run_setting
            print(f"\n[{label}]")
            print(f"  Stages in order: {in_order_count}/{total} ({in_order_count/total:.1%})")
            print(f"  Avg completion rate: {avg_completion:.1%}")
            print(f"  Stage breakdown:")
            for stage, count in stage_counts.items():
                print(f"    {stage}: {count}/{total} ({count/total:.1%})")

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
        "stage_monitor": aggregated_stage_metrics,  # Aggregated stage metrics
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
    parser.add_argument("--setting", type=str, default="compare",
                        choices=["row_react_exec", "pure_react_exec", "react_exec", "compare"])
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="verified_400",
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
