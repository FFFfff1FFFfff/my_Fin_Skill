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
    With skill: PoT (Program of Thought) - generate and execute Python code.
    Falls back to baseline if code execution fails.

    Returns:
        tuple: (extracted_answer, trace_dict)
    """
    # PoT prompt - clear template with data type conversion emphasis
    user_prompt = f"""Answer the question by writing Python code.

**IMPORTANT RULES:**
1. Your code MUST be inside a ```python``` code block
2. MUST end with: print(f"Final Answer: {{result}}")
3. Convert columns to numeric: df['col'] = pd.to_numeric(df['col'], errors='coerce')
4. Do NOT round - keep full precision

**Code Template:**
```python
import pandas as pd
import json

table_data = json.loads('''{table}''')
df = pd.DataFrame(table_data['data'], columns=table_data['columns'])

# Convert numeric columns (IMPORTANT - avoid string concatenation)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Your calculation here
result = ...

print(f"Final Answer: {{result}}")
```

Question: {question}

Write the Python code inside ```python``` block:"""

    if verbose:
        print(f"      [Skill/PoT] LLM...", end="", flush=True)

    # Step 1: Get code from LLM
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=skill_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    llm_duration_ms = int((time.time() - start_time) * 1000)

    full_response = response.content[0].text.strip()

    # Step 2: Extract Python code
    code = extract_python_code(full_response)

    # Step 3: Execute code
    exec_start = time.time()
    success, output, error = execute_code_safely(code)
    exec_duration_ms = int((time.time() - exec_start) * 1000)

    # Step 4: Extract answer
    extracted = ""
    used_fallback = False

    if success and output:
        extracted = extract_answer_from_output(output)
        # Validate: check if extracted looks reasonable
        if not extracted or extracted == "nan" or "Error" in extracted:
            success = False

    if not success or not extracted:
        # FALLBACK: Use baseline direct prompting
        used_fallback = True
        if verbose:
            print(f" {len(full_response)}c {llm_duration_ms}ms")
            print(f"        → Code: {'Found' if code else 'NOT FOUND'}")
            print(f"        → Exec: FAILED - {error[:40] if error else 'No output'}")
            print(f"        → Fallback to baseline...", end="", flush=True)

        # Call baseline
        fallback_start = time.time()
        fallback_prompt = f"""You are a table analyst. Answer this question directly.

Read the table in JSON format:
{table}

Question: {question}

Think step by step, then give your answer as:
Final Answer: [your answer]

The answer should be a number or short text, with full precision (no rounding)."""

        fallback_response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": fallback_prompt}]
        )
        fallback_duration_ms = int((time.time() - fallback_start) * 1000)
        fallback_text = fallback_response.content[0].text.strip()
        extracted = extract_answer(fallback_text)

        if verbose:
            print(f" {fallback_duration_ms}ms -> \"{extracted}\"")

        total_duration_ms = llm_duration_ms + exec_duration_ms + fallback_duration_ms
    else:
        total_duration_ms = llm_duration_ms + exec_duration_ms
        if verbose:
            print(f" {len(full_response)}c {total_duration_ms}ms")
            if code:
                code_preview = code.split('\n')[0][:50] + "..." if len(code) > 50 else code.split('\n')[0]
                print(f"        → Code: {code_preview}")
            print(f"        → Exec: OK ({exec_duration_ms}ms)")
            print(f"        → Final: {extracted}")

    trace = {
        "prompt": user_prompt[:500] + "...",
        "system_prompt": skill_prompt[:300] + "..." if len(skill_prompt) > 300 else skill_prompt,
        "response": full_response,
        "code": code,
        "execution": {
            "success": success,
            "output": output,
            "error": error,
            "duration_ms": exec_duration_ms,
        },
        "used_fallback": used_fallback,
        "extracted_answer": extracted,
        "llm_duration_ms": llm_duration_ms,
        "total_duration_ms": total_duration_ms,
        "response_length": len(full_response),
    }

    return extracted, trace


def run_benchmark(source: str = "sample", limit: int = None, offset: int = 0,
                  model: str = "claude-sonnet-4-5-20250929"):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.

    Args:
        source: "sample", "huggingface", or path to local file
        limit: Number of samples to test
        offset: Skip first N samples
        model: Model to use
    """
    print("=" * 70)
    print("TableBench Skill Benchmark")
    print("=" * 70)

    # Load data
    load_limit = (offset + limit) if limit else None
    print(f"\nLoading data (source={source}, offset={offset}, limit={limit})...")
    if source == "sample":
        samples = load_sample_data()
    else:
        samples = load_tablebench(source=source, limit=load_limit)

    # Apply offset
    if offset > 0:
        samples = samples[offset:]
        print(f"Skipped first {offset} samples")

    # Apply limit after offset
    if limit and len(samples) > limit:
        samples = samples[:limit]

    print(f"Loaded {len(samples)} samples (excluding Visualization)")

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
        qtype = sample["qtype"]
        qsubtype = sample["qsubtype"]
        question = sample["question"]
        table = sample["table"]
        ground_truth = sample["answer"]
        instruction = sample.get("instruction", "")  # Official DP instruction

        print(f"\n[{i+1}/{len(samples)}] {qtype}/{qsubtype} | ID: {qid}")
        print(f"    Q: {question[:70]}..." if len(question) > 70 else f"    Q: {question}")
        print(f"    GT: {ground_truth}")

        # Baseline - use official instruction
        baseline_trace = None
        try:
            pred_baseline, baseline_trace = ask_baseline(
                question, table, instruction=instruction, model=model
            )
            correct_baseline = evaluate_sample(pred_baseline, ground_truth, qtype, qsubtype)
            status = "✓" if correct_baseline >= 1.0 else "✗"
            print(f"    [Baseline Result] {pred_baseline} {status}")
        except Exception as e:
            pred_baseline = ""
            correct_baseline = 0.0
            print(f"    [Baseline] ERROR - {e}")

        results_baseline.append({
            "id": qid,
            "qtype": qtype,
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
            correct_skill = evaluate_sample(pred_skill, ground_truth, qtype, qsubtype)
            status = "✓" if correct_skill >= 1.0 else "✗"
            print(f"    [Skill Result] {pred_skill} {status}")
        except Exception as e:
            pred_skill = ""
            correct_skill = 0.0
            print(f"    [Skill] ERROR - {e}")

        results_skill.append({
            "id": qid,
            "qtype": qtype,
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

    args = parser.parse_args()
    run_benchmark(source=args.source, limit=args.limit, offset=args.offset, model=args.model)
