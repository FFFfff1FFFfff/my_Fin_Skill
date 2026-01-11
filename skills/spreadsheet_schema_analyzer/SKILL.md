# Spreadsheet Schema Analyzer Skill

## Purpose
Analyzes spreadsheet structure before generating manipulation code to produce robust, reusable solutions that work across multiple test cases with different data.

## Two-Stage Approach

### Stage 1: Schema Analysis
Analyze the spreadsheet to understand:
- Table structure (boundaries, headers, data regions)
- Column types and patterns
- Nested headers, merged cells, irregular structures
- Potential edge cases

### Stage 2: Robust Code Generation
Generate Python code that:
- Finds data dynamically (search for headers, not hardcoded positions)
- Uses relative references (row index from header, not absolute)
- Handles variable-length data (loop until empty, not fixed ranges)
- Validates assumptions (check columns exist before processing)

## Why This Approach?

SpreadsheetBench uses OJ-style evaluation where the same code must work on multiple test cases with different data. Naive approaches fail because they:
1. Hardcode row/column numbers (e.g., `ws['A5']`)
2. Assume fixed data lengths (e.g., `for row in range(2, 100)`)
3. Don't validate that expected columns exist

By separating schema understanding from code generation, we produce more robust solutions.

## Key Principles

```python
# ❌ BAD: Hardcoded, brittle
for row in range(2, 100):
    ws['D' + str(row)] = ws['B' + str(row)].value.upper()

# ✅ GOOD: Dynamic, robust
header_row = find_header_row(ws, "Name")
name_col = find_column(ws, header_row, "Name")
output_col = find_column(ws, header_row, "Output") or ws.max_column + 1

for row in range(header_row + 1, ws.max_row + 1):
    cell_value = ws.cell(row, name_col).value
    if cell_value is not None:
        ws.cell(row, output_col).value = str(cell_value).upper()
```

## Evaluation Metrics

- **Soft Restriction**: % of test cases that pass (0-100%)
- **Hard Restriction**: Binary (1 if ALL test cases pass, 0 otherwise)

The goal is to maximize Hard Restriction by generating code that handles data variations.
