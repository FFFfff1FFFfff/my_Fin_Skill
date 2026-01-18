#!/usr/bin/env python3
"""
TableBench benchmark runner: compare baseline vs with-skill performance
"""

import json
import os
import sys
import re
import time
import io
import signal
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_tablebench, load_sample_data
from evaluator import evaluate_sample, evaluate_batch
from skill_system import SkillManager

client = Anthropic()


# =============================================================================
# POT (Program of Thought): Code extraction and execution
# =============================================================================

def extract_python_code(text: str) -> str:
    """Extract Python code block from LLM response."""
    # Try to find ```python ... ``` block
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find ``` ... ``` block
    match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Check if it looks like Python code
        if 'import' in code or 'print' in code or 'def ' in code:
            return code

    return ""


def execute_code_safely(code: str, timeout_seconds: int = 15) -> tuple:
    """
    Execute Python code safely with timeout and capture output.

    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    if not code:
        return False, "", "No code to execute"

    # Capture stdout
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    try:
        # Set timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        # Execute code with captured output
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': None,
            'json': json,
        }

        # Import pandas if available
        try:
            import pandas as pd
            exec_globals['pd'] = pd
        except ImportError:
            pass

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)

        # Cancel timeout
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()

        return True, output, error

    except TimeoutError as e:
        signal.alarm(0)
        return False, "", str(e)
    except Exception as e:
        signal.alarm(0)
        return False, "", f"{type(e).__name__}: {str(e)}"


def extract_answer_from_output(output: str) -> str:
    """Extract Final Answer from code execution output."""
    # Look for "Final Answer:" in output
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: return last non-empty line
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


# =============================================================================
# VISUALIZATION: Chart generation and evaluation
# =============================================================================

def extract_chart_data(plt_module) -> list:
    """Extract y-data from matplotlib figure for comparison."""
    try:
        ax = plt_module.gca()
        data = []

        # Try line plots
        for line in ax.get_lines():
            ydata = line.get_ydata()
            if len(ydata) > 0:
                data.extend([float(y) for y in ydata if not (isinstance(y, float) and y != y)])

        # Try bar plots
        for patch in ax.patches:
            height = patch.get_height()
            if height and not (isinstance(height, float) and height != height):
                data.append(float(height))

        # Try pie charts (from wedges)
        for child in ax.get_children():
            if hasattr(child, 'theta2') and hasattr(child, 'theta1'):
                # Pie wedge - calculate proportion
                angle = child.theta2 - child.theta1
                data.append(round(angle / 360.0, 4))

        return data
    except Exception:
        return []


def execute_chart_code(code: str, timeout_seconds: int = 15) -> tuple:
    """
    Execute matplotlib code and extract chart data.

    Returns:
        tuple: (success: bool, chart_data: list, error: str)
    """
    if not code:
        return False, [], "No code to execute"

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Prepare execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'plt': plt,
            'pd': None,
            'json': json,
        }

        try:
            import pandas as pd
            exec_globals['pd'] = pd
        except ImportError:
            pass

        # Clear any existing figures
        plt.clf()
        plt.close('all')

        # Set timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        # Execute code
        exec(code, exec_globals)

        # Cancel timeout
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        # Extract chart data
        chart_data = extract_chart_data(plt)

        # Clean up
        plt.close('all')

        return True, chart_data, ""

    except TimeoutError as e:
        signal.alarm(0)
        return False, [], str(e)
    except Exception as e:
        signal.alarm(0)
        return False, [], f"{type(e).__name__}: {str(e)}"


def compare_chart_data(pred_data: list, ref_data: list, tolerance: float = 0.02) -> bool:
    """Compare predicted chart data with reference data."""
    if not pred_data or not ref_data:
        return False

    # Sort both lists for comparison
    pred_sorted = sorted([round(x, 2) for x in pred_data])
    ref_sorted = sorted([round(x, 2) for x in ref_data])

    if len(pred_sorted) != len(ref_sorted):
        return False

    # Compare with tolerance
    for p, r in zip(pred_sorted, ref_sorted):
        if abs(p - r) > tolerance * max(abs(r), 1):
            return False

    return True


def ask_visualization(question: str, table: str, model: str = "claude-sonnet-4-5-20250929",
                      verbose: bool = True) -> tuple:
    """
    Generate matplotlib code for visualization task.

    Returns:
        tuple: (code: str, trace_dict)
    """
    # Official Visualization prompt format
    prompt = f"""You are a data visualization expert. Generate Python matplotlib code to create the requested chart.

**MANDATORY CODE FORMAT:**
Your code must start with these exact three lines:
```python
import matplotlib.pyplot as plt
import pandas as pd
import json
```

Then:
1. Parse the table data from JSON
2. Create the visualization using matplotlib
3. Do NOT call plt.show() - just create the figure

**Table (JSON format):**
{table}

**Task:** {question}

Generate the complete Python code inside ```python``` block:"""

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
# STAGE PARSING: Extract structured reasoning stages from TCoT response
# =============================================================================

def parse_tcot_stages(response_text: str) -> dict:
    """
    Parse TCoT (Table Chain-of-Thought) response to extract reasoning stages.

    Expected stages based on skill:
    - STEP 1: Parse the Table
    - STEP 2: Understand the Question
    - STEP 3: Extract Data
    - STEP 4: Calculate
    - STEP 5: Format Answer / Final Answer
    """
    stages = {
        "parse_table": None,
        "understand_question": None,
        "extract_data": None,
        "calculate": None,
        "final_answer": None,
        "raw": response_text,
    }

    # Try to extract numbered steps
    step_patterns = {
        "parse_table": r'(?:STEP\s*1|Step\s*1|1\.|Parse|Table Structure)[:\s]*(.+?)(?=STEP\s*2|Step\s*2|2\.|Understand|Question|$)',
        "understand_question": r'(?:STEP\s*2|Step\s*2|2\.|Understand|Question Type)[:\s]*(.+?)(?=STEP\s*3|Step\s*3|3\.|Extract|Data|$)',
        "extract_data": r'(?:STEP\s*3|Step\s*3|3\.|Extract|Data Point)[:\s]*(.+?)(?=STEP\s*4|Step\s*4|4\.|Calculate|Calculation|$)',
        "calculate": r'(?:STEP\s*4|Step\s*4|4\.|Calculate|Calculation)[:\s]*(.+?)(?=STEP\s*5|Step\s*5|5\.|Final|Answer|Format|$)',
    }

    for stage, pattern in step_patterns.items():
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content and len(content) > 5:  # Filter out very short matches
                stages[stage] = content[:200]  # Limit length for display

    # Extract final answer
    answer_patterns = [
        r'Final Answer:\s*(.+?)(?:\n|$)',
        r'Answer:\s*(.+?)(?:\n|$)',
        r'\*\*(.+?)\*\*\s*$',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            stages["final_answer"] = match.group(1).strip()
            break

    return stages


def extract_answer(text: str) -> str:
    """Extract final answer from response."""
    # Look for "Final Answer:" pattern (official format)
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for "Answer:" pattern
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for bold answer like **1251** or **1251
    match = re.search(r'\*\*(-?\d+\.?\d*)\*?\*?', text)
    if match:
        return match.group(1)

    # Look for "= number" at end of calculation
    match = re.search(r'=\s*(-?\d+\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        return match.group(1)

    # Try to find the last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


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
        max_tokens=1024,  # Increased for reasoning
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

    if verbose:
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

    print("=" * 70)
    print("TableBench Skill Benchmark")
    print("=" * 70)

    # Load data
    load_limit = None  # Load all first, then filter
    print(f"\nLoading data (source={source}, offset={offset}, limit={limit}, qtype={qtype})...")

    # Determine if we need Visualization samples
    include_viz = False
    if qtype:
        qtype_upper = qtype.upper() if len(qtype) <= 3 else qtype
        if qtype_upper in ("VIZ", "Visualization"):
            include_viz = True

    if source == "sample":
        samples = load_sample_data()
    else:
        samples = load_tablebench(source=source, limit=None, include_viz=include_viz)

    # Filter by question type if specified
    if qtype:
        qtype_full = QTYPE_MAP.get(qtype.upper() if len(qtype) <= 3 else qtype, qtype)
        samples = [s for s in samples if s["qtype"] == qtype_full]
        print(f"Filtered to qtype={qtype_full}: {len(samples)} samples")

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

                    # Parse reference data from ground_truth (expected to be JSON list or chart values)
                    try:
                        if ground_truth.startswith('['):
                            ref_data = json.loads(ground_truth)
                        else:
                            # Try to parse as comma-separated numbers
                            ref_data = [float(x.strip()) for x in ground_truth.split(',') if x.strip()]
                    except (json.JSONDecodeError, ValueError):
                        ref_data = []

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

    # Save results with full traces
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "meta": {
            "timestamp": timestamp,
            "model": model,
            "source": source,
            "offset": offset,
            "num_samples": len(samples),
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
                        help="Filter by question type: FC (FactChecking), NR (NumericalReasoning), DA (DataAnalysis), VIZ (Visualization)")

    args = parser.parse_args()
    run_benchmark(source=args.source, limit=args.limit, offset=args.offset,
                  model=args.model, qtype=args.qtype)
