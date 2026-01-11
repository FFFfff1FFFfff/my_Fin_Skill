# SpreadsheetBench Skills

SpreadsheetBench requires generating Python code that works across multiple test cases with different data. The benchmark uses OJ-style evaluation where the same code must pass all 3 test cases.

## Available Modes

### 1. Baseline (Single-round)
Direct code generation matching the original paper's approach.
- Single prompt → Single code output
- No iteration or refinement

### 2. ReAct (Multi-round with Error Feedback)
The original paper's key improvement over baseline.
```
Generate Code → Execute → Get Error → Fix → Repeat (up to N times)
```
- Provides traceback feedback to the model
- Allows iterative refinement
- Proven to improve performance in the original paper

### 3. Schema-First (Our Skill)
Analyze spreadsheet structure before generating code.
```
Stage 1: Analyze schema (tables, headers, data types, patterns)
Stage 2: Generate robust code using dynamic finding
```
Key principle: Understand the data structure to avoid hardcoded positions.

### 4. Combined (Schema + ReAct)
Best of both approaches:
```
Schema Analysis → Generate Robust Code → Execute → Error Feedback → Fix
```

## Why Hardcoded Code Fails

SpreadsheetBench test cases have the same instruction but different data:
- Test 1: 50 rows of data
- Test 2: 100 rows of data
- Test 3: 25 rows of data

Hardcoded code like this fails:
```python
# ❌ FAILS: Assumes fixed row count
for row in range(2, 52):
    ws.cell(row, 4).value = ws.cell(row, 1).value * 2
```

Robust code works across all:
```python
# ✅ WORKS: Dynamic row detection
for row in range(2, ws.max_row + 1):
    if ws.cell(row, 1).value is not None:
        ws.cell(row, 4).value = ws.cell(row, 1).value * 2
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Soft Restriction** | % of test cases that pass (0-100%) |
| **Hard Restriction** | 1 if ALL test cases pass, 0 otherwise |

Goal: Maximize Hard Restriction by generating code that handles data variations.

## Usage

```bash
# Run all modes and compare
python spreadsheetbench/runner.py --limit 50 --mode all

# Run specific mode
python spreadsheetbench/runner.py --limit 50 --mode combined --max-turns 3
```
