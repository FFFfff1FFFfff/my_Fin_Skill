#!/usr/bin/env python3
"""
Runner for SpreadsheetBench benchmark.

Implements five approaches:
1. Baseline (Single-round): Direct code generation matching original paper
2. ReAct (Multi-round): Execute → Error Feedback → Fix → Repeat (original paper's approach)
3. Explore (Code-based): NO static preview → Model explores via code → Sees real output → Generates solution
4. Schema-First: Static text analysis → Generate robust code
5. Combined: Schema + ReAct refinement

Key insight from paper: Static text preview has limited effect. Letting the model
explore the real Excel structure via code execution is more effective.

Usage:
    python runner.py --limit 10
    python runner.py --limit 50 --mode react --max-turns 3
    python runner.py --limit 50 --mode explore  # Paper's recommended approach
    python runner.py --sample
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import anthropic

from data_loader import load_spreadsheetbench, load_sample_data
from evaluator import evaluate_instruction, calculate_metrics


# Initialize Anthropic client
client = anthropic.Anthropic()
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    patterns = [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    # Fallback: extract code-like content
    lines = response.split('\n')
    code_lines = []
    in_code = False

    for line in lines:
        if (line.strip().startswith(('import ', 'from ', 'def ', 'class ', '#')) or
            ('=' in line and not line.strip().startswith(('#', '//', '<!--'))) or
            line.strip().startswith(('for ', 'if ', 'while ', 'with ', 'try:', 'except'))):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    return response.strip()


def call_claude(messages: list, model: str = DEFAULT_MODEL, max_tokens: int = 4096) -> str:
    """Call Claude API with messages."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.content[0].text


def execute_code(code: str, input_path: str, output_path: str, timeout: int = 30) -> dict:
    """
    Execute Python code and return result.

    Returns dict with:
    - success: bool
    - output: stdout if success
    - error: error message/traceback if failed
    """
    # Copy input to output path
    try:
        shutil.copy(input_path, output_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to copy input file: {e}"}

    # Create wrapper script
    wrapper_code = f'''
import sys
sys.path.insert(0, '.')

# Variables the code expects
file_path = r"{output_path}"
input_file = r"{input_path}"
output_file = r"{output_path}"

{code}
'''

    # Write and execute
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            wrapper_path = f.name

        proc = subprocess.run(
            ['python', wrapper_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(input_path) or '.',
        )

        if proc.returncode != 0:
            # Extract relevant error info
            error_lines = proc.stderr.strip().split('\n')
            # Get last few lines which usually contain the actual error
            relevant_error = '\n'.join(error_lines[-10:]) if len(error_lines) > 10 else proc.stderr
            return {"success": False, "error": relevant_error}

        return {"success": True, "output": proc.stdout}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if 'wrapper_path' in locals():
            try:
                os.remove(wrapper_path)
            except:
                pass


# =============================================================================
# BASELINE: Single-round (matching original paper's prompt format)
# =============================================================================

PROMPT_FORMAT_SINGLE = """You are a spreadsheet expert who can manipulate spreadsheets through Python code using openpyxl library.

Given the following information:
1. Instruction: {instruction}
2. Spreadsheet path: {spreadsheet_path}
3. Spreadsheet content (first few rows):
{spreadsheet_content}
4. Instruction type: {instruction_type}
5. Answer position: {answer_position}
6. Output path: {output_path}

Please generate Python code for the final solution of the question.

Requirements:
- Use openpyxl library to manipulate the spreadsheet
- Load the workbook from the spreadsheet path
- Write results to the specified answer position
- Save the modified workbook to the output path
- Handle edge cases (empty cells, varying data lengths)

Generate only the Python code:

```python
"""


def generate_baseline_code(sample: dict, model: str = DEFAULT_MODEL) -> str:
    """Generate code using baseline single-round approach (matching original paper)."""
    # Use first test case path if available, otherwise use placeholder
    spreadsheet_path = "file_path"  # Will be set by execution wrapper
    output_path = "file_path"  # Same file for in-place modification

    prompt = PROMPT_FORMAT_SINGLE.format(
        instruction=sample["instruction"],
        spreadsheet_path=spreadsheet_path,
        spreadsheet_content=sample["preview"],
        instruction_type=sample["instruction_type"],
        answer_position=sample["answer_position"],
        output_path=output_path,
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_claude(messages, model=model)
    return extract_code(response)


# =============================================================================
# REACT: Multi-round with error feedback (original paper's key approach)
# =============================================================================

PROMPT_REACT_INITIAL = """You are a spreadsheet expert who can manipulate spreadsheets through Python code using openpyxl library.

Given the following information:
1. Instruction: {instruction}
2. Spreadsheet path: file_path (variable already set)
3. Spreadsheet content (first few rows):
{spreadsheet_content}
4. Instruction type: {instruction_type}
5. Answer position: {answer_position}
6. Output path: file_path (same as input, modify in place)

The solution can be generated through {max_turn_num} rounds of interaction. You can:
1. Generate Python code to explore the spreadsheet structure (e.g., check sheet names, row counts, column headers)
2. Generate Python code for the final solution

After each code execution, you will receive the output or error traceback to help you refine your solution.

Start by generating Python code. If you need to explore the spreadsheet first, do that. Otherwise, generate the solution directly.

```python
"""

PROMPT_REACT_CONTINUE = """The previous code execution returned:

{execution_result}

{feedback}

Please analyze the result and generate the next Python code. If the execution was successful and the task is complete, you can indicate that. If there was an error, fix the code based on the traceback.

```python
"""


def generate_react_code(
    sample: dict,
    model: str = DEFAULT_MODEL,
    max_turns: int = 3,
    test_input_path: str = None,
    test_output_path: str = None,
) -> tuple:
    """
    Generate code using ReAct multi-round approach.

    Returns (final_code, conversation_history)
    """
    messages = []
    conversation = []
    final_code = None

    # Initial prompt
    initial_prompt = PROMPT_REACT_INITIAL.format(
        instruction=sample["instruction"],
        spreadsheet_content=sample["preview"],
        instruction_type=sample["instruction_type"],
        answer_position=sample["answer_position"],
        max_turn_num=max_turns,
    )

    messages.append({"role": "user", "content": initial_prompt})
    conversation.append({"role": "user", "content": initial_prompt})

    for turn in range(max_turns):
        # Get model response
        response = call_claude(messages, model=model)
        code = extract_code(response)

        messages.append({"role": "assistant", "content": response})
        conversation.append({"role": "assistant", "content": response, "code": code})

        final_code = code  # Update final code

        # If we have test paths, execute and get feedback
        if test_input_path and test_output_path:
            exec_result = execute_code(code, test_input_path, test_output_path)

            if exec_result["success"]:
                execution_result = exec_result.get("output", "Code executed successfully.")
                feedback = "The code executed without errors. If the task is complete, this is the final solution."

                # Check if output file was modified (task likely complete)
                if os.path.exists(test_output_path):
                    conversation.append({"role": "execution", "success": True, "output": execution_result})
                    break  # Task complete
            else:
                execution_result = exec_result["error"]
                feedback = "There was an error. Please fix the code based on the traceback above."
                conversation.append({"role": "execution", "success": False, "error": execution_result})

            # Continue conversation with feedback
            continue_prompt = PROMPT_REACT_CONTINUE.format(
                execution_result=execution_result,
                feedback=feedback,
            )
            messages.append({"role": "user", "content": continue_prompt})
            conversation.append({"role": "user", "content": continue_prompt})
        else:
            # No execution available, just return the first code
            break

    return final_code, conversation


# =============================================================================
# EXPLORE: Code-based exploration (paper's key insight - NO static preview)
# =============================================================================

PROMPT_EXPLORE_INITIAL = """You are a spreadsheet expert who manipulates spreadsheets through Python code using openpyxl.

## Task Information
1. Instruction: {instruction}
2. Instruction type: {instruction_type}
3. Answer position: {answer_position}
4. File path: file_path (variable already set)

**IMPORTANT**: You have NOT been given a preview of the spreadsheet content.
You should first generate Python code to EXPLORE the spreadsheet structure:
- Check sheet names
- Check dimensions (max_row, max_column)
- Print first few rows to understand the data layout
- Identify headers and data types
- Look for any special structures (merged cells, multiple tables, etc.)

Generate exploration code that prints useful information about the spreadsheet:

```python
"""

PROMPT_EXPLORE_SOLVE = """Based on your exploration, here is what you discovered about the spreadsheet:

{exploration_output}

Now generate the FINAL SOLUTION code to accomplish the task:
- Instruction: {instruction}
- Answer position: {answer_position}

Requirements:
- Use dynamic finding (no hardcoded row/column numbers)
- Handle varying data lengths (use max_row, not fixed ranges)
- The same code will run on 3 different test cases with different data

Generate the solution code:

```python
"""


def generate_explore_code(
    sample: dict,
    model: str = DEFAULT_MODEL,
    max_turns: int = 3,
    test_input_path: str = None,
    test_output_path: str = None,
) -> tuple:
    """
    Explore-then-solve approach (paper's key insight).

    Stage 1: Model generates exploration code (no static preview given)
    Stage 2: Execute exploration, show real output
    Stage 3: Model generates solution based on real exploration
    Stage 4: ReAct refinement if needed

    Returns (final_code, {"exploration_output": ..., "conversation": ...})
    """
    messages = []
    conversation = []

    # Stage 1: Exploration prompt (NO static preview)
    explore_prompt = PROMPT_EXPLORE_INITIAL.format(
        instruction=sample["instruction"],
        instruction_type=sample["instruction_type"],
        answer_position=sample["answer_position"],
    )

    messages.append({"role": "user", "content": explore_prompt})
    conversation.append({"role": "user", "content": explore_prompt})

    # Get exploration code
    response = call_claude(messages, model=model)
    explore_code = extract_code(response)

    messages.append({"role": "assistant", "content": response})
    conversation.append({"role": "assistant", "content": response, "code": explore_code})

    # Execute exploration code
    exploration_output = ""
    if test_input_path and test_output_path:
        # Create a modified exploration code that prints but doesn't modify
        safe_explore_code = f'''
from openpyxl import load_workbook

wb = load_workbook(file_path)
print("=== SPREADSHEET EXPLORATION ===")
print(f"Sheet names: {{wb.sheetnames}}")

for sheet_name in wb.sheetnames[:3]:  # Limit to first 3 sheets
    ws = wb[sheet_name]
    print(f"\\n=== Sheet: {{sheet_name}} ===")
    print(f"Dimensions: {{ws.max_row}} rows x {{ws.max_column}} columns")

    # Print first 10 rows
    print("\\nFirst 10 rows:")
    for row in range(1, min(11, ws.max_row + 1)):
        row_data = []
        for col in range(1, min(ws.max_column + 1, 20)):  # Limit columns too
            val = ws.cell(row, col).value
            row_data.append(str(val) if val is not None else "")
        print(f"  Row {{row}}: {{row_data}}")

wb.close()
'''
        # Also try to run the model's exploration code
        exec_result = execute_code(explore_code, test_input_path, test_output_path)
        if exec_result["success"]:
            exploration_output = exec_result.get("output", "")
        else:
            # If model's code failed, use our safe exploration
            exec_result2 = execute_code(safe_explore_code, test_input_path, test_output_path)
            if exec_result2["success"]:
                exploration_output = exec_result2.get("output", "")
            else:
                exploration_output = f"Exploration failed: {exec_result['error']}"

        conversation.append({"role": "execution", "output": exploration_output})
    else:
        exploration_output = "(No execution available - using static preview)\n" + sample.get("preview", "")

    # Stage 2: Generate solution based on exploration
    solve_prompt = PROMPT_EXPLORE_SOLVE.format(
        exploration_output=exploration_output[:3000],  # Limit size
        instruction=sample["instruction"],
        answer_position=sample["answer_position"],
    )

    messages.append({"role": "user", "content": solve_prompt})
    conversation.append({"role": "user", "content": solve_prompt})

    # Get solution and refine with ReAct
    final_code = None

    for turn in range(max_turns):
        response = call_claude(messages, model=model)
        code = extract_code(response)

        messages.append({"role": "assistant", "content": response})
        conversation.append({"role": "assistant", "content": response, "code": code})

        final_code = code

        # Execute and get feedback
        if test_input_path and test_output_path:
            exec_result = execute_code(code, test_input_path, test_output_path)

            if exec_result["success"]:
                conversation.append({"role": "execution", "success": True})
                break
            else:
                error = exec_result["error"]
                conversation.append({"role": "execution", "success": False, "error": error})

                continue_prompt = f"""The code execution failed with error:

{error}

Please fix the code. Remember what you learned from exploring the spreadsheet:
{exploration_output[:1000]}

Generate fixed code:

```python
"""
                messages.append({"role": "user", "content": continue_prompt})
                conversation.append({"role": "user", "content": continue_prompt})
        else:
            break

    return final_code, {"exploration_output": exploration_output, "conversation": conversation}


# =============================================================================
# SCHEMA-FIRST: Analyze structure then generate robust code (our skill)
# =============================================================================

SCHEMA_ANALYSIS_PROMPT = """You are a spreadsheet analysis expert. Analyze this spreadsheet structure carefully.

## Spreadsheet Content (first rows):
{spreadsheet_content}

## Task:
{instruction}

Analyze and provide:

1. **Data Structure**:
   - Number of tables/data regions
   - Header row location (if any)
   - Data types per column
   - Any merged cells or irregular structures

2. **Key Patterns**:
   - How to identify the data region dynamically (not by hardcoded row numbers)
   - Variable-length data considerations
   - Edge cases to handle

3. **Robust Approach**:
   - How to find headers by content, not position
   - How to handle varying row counts
   - Validation checks needed

Be concise but thorough. This analysis will be used to generate robust code.
"""

ROBUST_CODE_PROMPT = """You are a spreadsheet expert. Generate ROBUST Python code that works across multiple test cases with varying data.

## Task Information
- Instruction: {instruction}
- Instruction type: {instruction_type}
- Answer position: {answer_position}
- File path: file_path (variable already set)

## Schema Analysis:
{schema_analysis}

## Spreadsheet Preview:
{spreadsheet_content}

## CRITICAL: Generate Robust Code

Your code MUST:
1. **Find data dynamically** - NEVER hardcode row/column numbers like `ws['A5']` or `range(2, 100)`
2. **Use max_row/max_column** - Loop based on actual data extent
3. **Search for headers** - Find headers by content, not position
4. **Handle None values** - Check before processing
5. **Work on different data** - The same code runs on 3 test cases with different values

Example patterns:
```python
# Find header row dynamically
header_row = None
for row in range(1, ws.max_row + 1):
    if ws.cell(row, 1).value and "expected_text" in str(ws.cell(row, 1).value).lower():
        header_row = row
        break

# Process all data rows
for row in range(data_start_row, ws.max_row + 1):
    value = ws.cell(row, col).value
    if value is not None:
        # process
```

Generate only the Python code:

```python
"""


def generate_schema_code(sample: dict, model: str = DEFAULT_MODEL) -> tuple:
    """
    Generate code using schema-first approach.

    Stage 1: Analyze spreadsheet structure
    Stage 2: Generate robust code based on analysis

    Returns (code, schema_analysis)
    """
    # Stage 1: Schema Analysis
    analysis_prompt = SCHEMA_ANALYSIS_PROMPT.format(
        spreadsheet_content=sample["preview"],
        instruction=sample["instruction"],
    )

    messages = [{"role": "user", "content": analysis_prompt}]
    schema_analysis = call_claude(messages, model=model)

    # Stage 2: Robust Code Generation
    code_prompt = ROBUST_CODE_PROMPT.format(
        instruction=sample["instruction"],
        instruction_type=sample["instruction_type"],
        answer_position=sample["answer_position"],
        schema_analysis=schema_analysis,
        spreadsheet_content=sample["preview"],
    )

    messages = [{"role": "user", "content": code_prompt}]
    response = call_claude(messages, model=model)
    code = extract_code(response)

    return code, schema_analysis


# =============================================================================
# COMBINED: Schema Analysis + ReAct (best of both)
# =============================================================================

def generate_combined_code(
    sample: dict,
    model: str = DEFAULT_MODEL,
    max_turns: int = 3,
    test_input_path: str = None,
    test_output_path: str = None,
) -> tuple:
    """
    Combined approach: Schema analysis first, then ReAct refinement.

    Returns (final_code, {"schema_analysis": ..., "conversation": ...})
    """
    # Stage 1: Schema Analysis
    analysis_prompt = SCHEMA_ANALYSIS_PROMPT.format(
        spreadsheet_content=sample["preview"],
        instruction=sample["instruction"],
    )

    messages = [{"role": "user", "content": analysis_prompt}]
    schema_analysis = call_claude(messages, model=model)

    # Stage 2: Initial robust code with schema context
    initial_prompt = f"""You are a spreadsheet expert. Generate ROBUST Python code based on the analysis below.

## Schema Analysis:
{schema_analysis}

## Task Information:
- Instruction: {sample["instruction"]}
- Instruction type: {sample["instruction_type"]}
- Answer position: {sample["answer_position"]}
- File path: file_path (variable already set)

## Spreadsheet Preview:
{sample["preview"]}

## Requirements:
- Find data dynamically (no hardcoded positions)
- Handle varying data lengths
- Work across multiple test cases

You have {max_turns} rounds to refine if needed. Generate Python code:

```python
"""

    messages = [{"role": "user", "content": initial_prompt}]
    conversation = [{"role": "user", "content": initial_prompt}]

    final_code = None

    for turn in range(max_turns):
        response = call_claude(messages, model=model)
        code = extract_code(response)

        messages.append({"role": "assistant", "content": response})
        conversation.append({"role": "assistant", "content": response, "code": code})

        final_code = code

        # Execute and get feedback if paths available
        if test_input_path and test_output_path:
            exec_result = execute_code(code, test_input_path, test_output_path)

            if exec_result["success"]:
                conversation.append({"role": "execution", "success": True})
                break
            else:
                error = exec_result["error"]
                conversation.append({"role": "execution", "success": False, "error": error})

                continue_prompt = f"""The code execution failed with error:

{error}

Please fix the code based on the traceback. Remember to:
- Use dynamic data finding (no hardcoded rows)
- Handle edge cases
- Check for None values

```python
"""
                messages.append({"role": "user", "content": continue_prompt})
                conversation.append({"role": "user", "content": continue_prompt})
        else:
            break

    return final_code, {"schema_analysis": schema_analysis, "conversation": conversation}


# =============================================================================
# Main Runner
# =============================================================================

def run_benchmark(
    limit: int = None,
    model: str = DEFAULT_MODEL,
    mode: str = "all",  # "baseline", "react", "explore", "schema", "combined", "all"
    max_turns: int = 3,
    use_sample: bool = False,
    data_dir: str = None,
    output_dir: str = None,
    instruction_types: list = None,
):
    """Run the SpreadsheetBench benchmark."""

    print("=" * 70)
    print("SpreadsheetBench Skill Benchmark")
    print("=" * 70)
    print(f"\nModel: {model}")
    print(f"Mode: {mode}")
    if mode in ["react", "explore", "combined", "all"]:
        print(f"Max turns: {max_turns}")

    # Load data
    print(f"\nLoading data (limit={limit})...")
    if use_sample:
        samples = load_sample_data()
        print("Using built-in sample data (no real evaluation possible)")
    else:
        samples = load_spreadsheetbench(
            data_dir=data_dir,
            limit=limit,
            instruction_types=instruction_types,
        )

    if not samples:
        print("No samples to process. Exiting.")
        return

    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="spreadsheetbench_")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Results storage
    results = {m: [] for m in ["baseline", "react", "explore", "schema", "combined"]}
    modes_to_run = ["baseline", "react", "explore", "schema", "combined"] if mode == "all" else [mode]

    # Process each sample
    for i, sample in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(samples)}] ID: {sample['id']}")
        print(f"Type: {sample['instruction_type']}")
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Answer Position: {sample['answer_position']}")
        print(f"Test Cases: {len(sample['test_cases'])}")

        sample_output_dir = os.path.join(output_dir, sample['id'])
        os.makedirs(sample_output_dir, exist_ok=True)

        # Get first test case for ReAct execution feedback
        test_input = None
        test_output = None
        if sample['test_cases']:
            tc = sample['test_cases'][0]
            test_input = tc['input_file']
            test_output = os.path.join(sample_output_dir, "react_test_output.xlsx")

        for run_mode in modes_to_run:
            print(f"\n  [{run_mode.upper()}]")

            try:
                if run_mode == "baseline":
                    code = generate_baseline_code(sample, model=model)
                    extra_info = None

                elif run_mode == "react":
                    code, conv = generate_react_code(
                        sample, model=model, max_turns=max_turns,
                        test_input_path=test_input, test_output_path=test_output,
                    )
                    extra_info = {"conversation_turns": len([c for c in conv if c["role"] == "assistant"])}

                elif run_mode == "explore":
                    code, info = generate_explore_code(
                        sample, model=model, max_turns=max_turns,
                        test_input_path=test_input, test_output_path=test_output,
                    )
                    extra_info = {
                        "exploration_output": info["exploration_output"][:500],
                        "turns": len([c for c in info["conversation"] if c["role"] == "assistant"]),
                    }

                elif run_mode == "schema":
                    code, schema = generate_schema_code(sample, model=model)
                    extra_info = {"schema_analysis": schema[:500]}

                elif run_mode == "combined":
                    code, info = generate_combined_code(
                        sample, model=model, max_turns=max_turns,
                        test_input_path=test_input, test_output_path=test_output,
                    )
                    extra_info = {
                        "schema_analysis": info["schema_analysis"][:500],
                        "turns": len([c for c in info["conversation"] if c["role"] == "assistant"]),
                    }

                print(f"  Generated {len(code)} chars of code")

                # Save code
                with open(os.path.join(sample_output_dir, f"{run_mode}_code.py"), "w") as f:
                    f.write(code)

                # Evaluate if test cases available
                if sample['test_cases']:
                    eval_result = evaluate_instruction(
                        code=code,
                        test_cases=sample['test_cases'],
                        answer_position=sample['answer_position'],
                        output_dir=sample_output_dir,
                    )
                    eval_result["id"] = sample["id"]
                    eval_result["instruction_type"] = sample["instruction_type"]
                    if extra_info:
                        eval_result.update(extra_info)
                    results[run_mode].append(eval_result)

                    print(f"  Soft: {eval_result['soft_restriction']:.1%}, Hard: {eval_result['hard_restriction']}")
                else:
                    print("  (No test cases - skipping evaluation)")

            except Exception as e:
                print(f"  ERROR: {e}")
                if sample['test_cases']:
                    results[run_mode].append({
                        "id": sample["id"],
                        "instruction_type": sample["instruction_type"],
                        "soft_restriction": 0.0,
                        "hard_restriction": 0,
                        "error": str(e),
                    })

    # Final Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    metrics_summary = {}
    for run_mode in modes_to_run:
        if results[run_mode]:
            metrics = calculate_metrics(results[run_mode])
            metrics_summary[run_mode] = metrics
            print(f"\n[{run_mode.upper()}]")
            print(f"  Soft Restriction: {metrics['soft_restriction_avg']:.1%}")
            print(f"  Hard Restriction: {metrics['hard_restriction_avg']:.1%}")
            if metrics['by_type']:
                print("  By Type:")
                for itype, data in metrics['by_type'].items():
                    print(f"    {itype}: Soft={data['soft_restriction_avg']:.1%}, Hard={data['hard_restriction_avg']:.1%}")

    # Comparison
    if "baseline" in metrics_summary and len(metrics_summary) > 1:
        baseline_soft = metrics_summary["baseline"]["soft_restriction_avg"]
        baseline_hard = metrics_summary["baseline"]["hard_restriction_avg"]
        print(f"\n[IMPROVEMENT vs BASELINE]")
        for run_mode, metrics in metrics_summary.items():
            if run_mode != "baseline":
                soft_diff = metrics["soft_restriction_avg"] - baseline_soft
                hard_diff = metrics["hard_restriction_avg"] - baseline_hard
                print(f"  {run_mode}: Soft {soft_diff:+.1%}, Hard {hard_diff:+.1%}")

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": model,
            "mode": mode,
            "max_turns": max_turns,
            "timestamp": datetime.now().isoformat(),
            "results": {k: v for k, v in results.items() if v},
            "metrics": metrics_summary,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpreadsheetBench benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["baseline", "react", "explore", "schema", "combined", "all"],
                        help="Evaluation mode")
    parser.add_argument("--max-turns", type=int, default=3, help="Max turns for ReAct")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--cell-level", action="store_true", help="Only Cell-Level tasks")
    parser.add_argument("--sheet-level", action="store_true", help="Only Sheet-Level tasks")

    args = parser.parse_args()

    instruction_types = None
    if args.cell_level:
        instruction_types = ["Cell-Level"]
    elif args.sheet_level:
        instruction_types = ["Sheet-Level"]

    run_benchmark(
        limit=args.limit,
        model=args.model,
        mode=args.mode,
        max_turns=args.max_turns,
        use_sample=args.sample,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        instruction_types=instruction_types,
    )
