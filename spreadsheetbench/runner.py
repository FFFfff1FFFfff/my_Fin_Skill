#!/usr/bin/env python3
"""
Runner for SpreadsheetBench benchmark.

Compares baseline (direct code generation) with skill-enhanced approach
(Schema Analysis → Robust Code Generation).

Usage:
    python runner.py --limit 10
    python runner.py --limit 50 --skill-only
    python runner.py --sample  # Use built-in sample data
"""

import argparse
import json
import os
import re
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
    # Try to find code in markdown code blocks
    patterns = [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the longest code block
            return max(matches, key=len).strip()

    # If no code block, try to extract code-like content
    lines = response.split('\n')
    code_lines = []
    in_code = False

    for line in lines:
        # Heuristic: lines starting with import, from, def, class, or containing = are likely code
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


# =============================================================================
# Baseline: Direct Code Generation
# =============================================================================

BASELINE_PROMPT = """You are a spreadsheet expert. Generate Python code using openpyxl to solve the following spreadsheet manipulation task.

## Task Information
- **Instruction**: {instruction}
- **Instruction Type**: {instruction_type}
- **Answer Position**: {answer_position} (where to write the result)
- **Input File**: The spreadsheet is already loaded at `file_path`

## Spreadsheet Preview (first rows):
{preview}

## Spreadsheet Schema:
{schema}

## Requirements:
1. Use openpyxl library
2. The workbook is already loaded - assume `wb = load_workbook(file_path)`
3. Write results to the answer position specified
4. Save the workbook at the end with `wb.save(file_path)`
5. Handle edge cases (empty cells, different data lengths)

Generate ONLY the Python code, no explanations. The code should:
- Load the workbook from `file_path`
- Perform the manipulation
- Save back to `file_path`

```python
"""


def generate_baseline_code(sample: dict, model: str = DEFAULT_MODEL) -> str:
    """Generate code using baseline (direct) approach."""
    prompt = BASELINE_PROMPT.format(
        instruction=sample["instruction"],
        instruction_type=sample["instruction_type"],
        answer_position=sample["answer_position"],
        preview=sample["preview"],
        schema=json.dumps(sample["schema"], indent=2),
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_claude(messages, model=model)
    return extract_code(response)


# =============================================================================
# Skill: Schema Analysis → Robust Code Generation
# =============================================================================

SCHEMA_ANALYSIS_PROMPT = """You are a spreadsheet analysis expert. Analyze this spreadsheet and provide a detailed schema description.

## Spreadsheet Preview:
{preview}

## Basic Schema Info:
{schema}

## Task Context:
The user wants to: {instruction}

Analyze the spreadsheet and provide:

1. **Table Structure**:
   - How many tables are in the sheet?
   - Where does each table start and end (row numbers)?
   - Are there nested headers, merged cells, or irregular structures?

2. **Column Analysis**:
   - What are the column headers (if any)?
   - What data type is each column (text, number, date, etc.)?
   - Are there any special patterns in the data?

3. **Key Observations**:
   - Any potential edge cases to handle?
   - Variable-length data considerations?
   - Special formatting or non-standard elements?

4. **Recommended Approach**:
   - How to dynamically find the relevant data (not hardcoded positions)?
   - What validation checks should be included?

Provide a structured analysis that will help generate robust, reusable code.
"""

ROBUST_CODE_PROMPT = """You are a spreadsheet expert. Generate ROBUST Python code that will work across multiple test cases with different data.

## Task Information
- **Instruction**: {instruction}
- **Instruction Type**: {instruction_type}
- **Answer Position**: {answer_position}
- **Input File**: `file_path`

## Spreadsheet Schema Analysis:
{schema_analysis}

## Original Preview:
{preview}

## CRITICAL Requirements for Robust Code:

1. **NEVER hardcode row/column numbers** - Always find data dynamically:
   - Search for headers by content, not position
   - Use `ws.max_row` and `ws.max_column` for boundaries
   - Loop until empty cells, don't assume fixed ranges

2. **Use dynamic finding patterns**:
   ```python
   # Good: Find header row dynamically
   header_row = None
   for row in range(1, ws.max_row + 1):
       if ws.cell(row, 1).value == "Expected Header":
           header_row = row
           break

   # Good: Process all data rows
   for row in range(header_row + 1, ws.max_row + 1):
       if ws.cell(row, col).value is not None:
           # process
   ```

3. **Validate assumptions**:
   - Check if expected columns/headers exist
   - Handle None values gracefully
   - Don't assume specific number of data rows

4. **Code structure**:
   - Load: `wb = load_workbook(file_path)`
   - Process: Manipulate the data
   - Save: `wb.save(file_path)`

Generate ONLY the Python code. Make it robust enough to work with different spreadsheet data following the same structure.

```python
"""


def analyze_schema(sample: dict, model: str = DEFAULT_MODEL) -> str:
    """Stage 1: Analyze spreadsheet schema."""
    prompt = SCHEMA_ANALYSIS_PROMPT.format(
        preview=sample["preview"],
        schema=json.dumps(sample["schema"], indent=2),
        instruction=sample["instruction"],
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_claude(messages, model=model)
    return response


def generate_robust_code(sample: dict, schema_analysis: str, model: str = DEFAULT_MODEL) -> str:
    """Stage 2: Generate robust code based on schema analysis."""
    prompt = ROBUST_CODE_PROMPT.format(
        instruction=sample["instruction"],
        instruction_type=sample["instruction_type"],
        answer_position=sample["answer_position"],
        schema_analysis=schema_analysis,
        preview=sample["preview"],
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_claude(messages, model=model)
    return extract_code(response)


def generate_skill_code(sample: dict, model: str = DEFAULT_MODEL) -> tuple:
    """Generate code using skill-enhanced approach (two-stage)."""
    # Stage 1: Schema Analysis
    schema_analysis = analyze_schema(sample, model=model)

    # Stage 2: Robust Code Generation
    code = generate_robust_code(sample, schema_analysis, model=model)

    return code, schema_analysis


# =============================================================================
# Main Runner
# =============================================================================

def run_benchmark(
    limit: int = None,
    model: str = DEFAULT_MODEL,
    baseline_only: bool = False,
    skill_only: bool = False,
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
    print(f"Mode: {'Baseline only' if baseline_only else 'Skill only' if skill_only else 'Both'}")

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
    baseline_results = []
    skill_results = []

    # Process each sample
    for i, sample in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(samples)}] ID: {sample['id']}")
        print(f"Type: {sample['instruction_type']}")
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Answer Position: {sample['answer_position']}")
        print(f"Test Cases: {len(sample['test_cases'])}")

        # Create sample output directory
        sample_output_dir = os.path.join(output_dir, sample['id'])
        os.makedirs(sample_output_dir, exist_ok=True)

        # Skip if no test cases (sample data)
        if not sample['test_cases']:
            print("  No test cases available - showing generated code only")

            if not skill_only:
                print("\n  [Baseline Code Generation]")
                try:
                    baseline_code = generate_baseline_code(sample, model=model)
                    print(f"  Generated {len(baseline_code)} chars of code")
                    # Save code
                    with open(os.path.join(sample_output_dir, "baseline_code.py"), "w") as f:
                        f.write(baseline_code)
                except Exception as e:
                    print(f"  ERROR: {e}")

            if not baseline_only:
                print("\n  [Skill Code Generation]")
                try:
                    skill_code, schema_analysis = generate_skill_code(sample, model=model)
                    print(f"  Schema Analysis: {len(schema_analysis)} chars")
                    print(f"  Generated {len(skill_code)} chars of code")
                    # Save
                    with open(os.path.join(sample_output_dir, "schema_analysis.txt"), "w") as f:
                        f.write(schema_analysis)
                    with open(os.path.join(sample_output_dir, "skill_code.py"), "w") as f:
                        f.write(skill_code)
                except Exception as e:
                    print(f"  ERROR: {e}")

            continue

        # Run baseline
        if not skill_only:
            print("\n  [Baseline]")
            try:
                baseline_code = generate_baseline_code(sample, model=model)
                print(f"  Generated {len(baseline_code)} chars of code")

                # Save code
                with open(os.path.join(sample_output_dir, "baseline_code.py"), "w") as f:
                    f.write(baseline_code)

                # Evaluate
                baseline_eval = evaluate_instruction(
                    code=baseline_code,
                    test_cases=sample['test_cases'],
                    answer_position=sample['answer_position'],
                    output_dir=sample_output_dir,
                )
                baseline_eval["id"] = sample["id"]
                baseline_eval["instruction_type"] = sample["instruction_type"]
                baseline_results.append(baseline_eval)

                print(f"  Soft: {baseline_eval['soft_restriction']:.1%}, Hard: {baseline_eval['hard_restriction']}")

            except Exception as e:
                print(f"  ERROR: {e}")
                baseline_results.append({
                    "id": sample["id"],
                    "instruction_type": sample["instruction_type"],
                    "soft_restriction": 0.0,
                    "hard_restriction": 0,
                    "error": str(e),
                })

        # Run skill
        if not baseline_only:
            print("\n  [Skill: Schema → Robust Code]")
            try:
                skill_code, schema_analysis = generate_skill_code(sample, model=model)
                print(f"  Schema Analysis: {len(schema_analysis)} chars")
                print(f"  Generated {len(skill_code)} chars of code")

                # Save
                with open(os.path.join(sample_output_dir, "schema_analysis.txt"), "w") as f:
                    f.write(schema_analysis)
                with open(os.path.join(sample_output_dir, "skill_code.py"), "w") as f:
                    f.write(skill_code)

                # Evaluate
                skill_eval = evaluate_instruction(
                    code=skill_code,
                    test_cases=sample['test_cases'],
                    answer_position=sample['answer_position'],
                    output_dir=sample_output_dir,
                )
                skill_eval["id"] = sample["id"]
                skill_eval["instruction_type"] = sample["instruction_type"]
                skill_eval["schema_analysis"] = schema_analysis[:500]  # Truncate for storage
                skill_results.append(skill_eval)

                print(f"  Soft: {skill_eval['soft_restriction']:.1%}, Hard: {skill_eval['hard_restriction']}")

            except Exception as e:
                print(f"  ERROR: {e}")
                skill_results.append({
                    "id": sample["id"],
                    "instruction_type": sample["instruction_type"],
                    "soft_restriction": 0.0,
                    "hard_restriction": 0,
                    "error": str(e),
                })

    # Calculate and display final metrics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if baseline_results:
        baseline_metrics = calculate_metrics(baseline_results)
        print(f"\n[Baseline]")
        print(f"  Soft Restriction: {baseline_metrics['soft_restriction_avg']:.1%}")
        print(f"  Hard Restriction: {baseline_metrics['hard_restriction_avg']:.1%}")
        if baseline_metrics['by_type']:
            print("  By Type:")
            for itype, data in baseline_metrics['by_type'].items():
                print(f"    {itype}: Soft={data['soft_restriction_avg']:.1%}, Hard={data['hard_restriction_avg']:.1%} (n={data['count']})")

    if skill_results:
        skill_metrics = calculate_metrics(skill_results)
        print(f"\n[Skill: Schema → Robust Code]")
        print(f"  Soft Restriction: {skill_metrics['soft_restriction_avg']:.1%}")
        print(f"  Hard Restriction: {skill_metrics['hard_restriction_avg']:.1%}")
        if skill_metrics['by_type']:
            print("  By Type:")
            for itype, data in skill_metrics['by_type'].items():
                print(f"    {itype}: Soft={data['soft_restriction_avg']:.1%}, Hard={data['hard_restriction_avg']:.1%} (n={data['count']})")

    if baseline_results and skill_results:
        baseline_metrics = calculate_metrics(baseline_results)
        skill_metrics = calculate_metrics(skill_results)
        soft_diff = skill_metrics['soft_restriction_avg'] - baseline_metrics['soft_restriction_avg']
        hard_diff = skill_metrics['hard_restriction_avg'] - baseline_metrics['hard_restriction_avg']
        print(f"\n[Improvement]")
        print(f"  Soft Restriction: {soft_diff:+.1%}")
        print(f"  Hard Restriction: {hard_diff:+.1%}")

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "baseline_results": baseline_results,
            "skill_results": skill_results,
            "baseline_metrics": calculate_metrics(baseline_results) if baseline_results else None,
            "skill_metrics": calculate_metrics(skill_results) if skill_results else None,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpreadsheetBench benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    parser.add_argument("--skill-only", action="store_true", help="Run skill only")
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
        baseline_only=args.baseline_only,
        skill_only=args.skill_only,
        use_sample=args.sample,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        instruction_types=instruction_types,
    )
