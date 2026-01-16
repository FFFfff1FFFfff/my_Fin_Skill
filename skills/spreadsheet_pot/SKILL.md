name	spreadsheet_pot
description	Program-of-Thought style multi-round React framework for spreadsheet manipulation tasks

# Spreadsheet PoT (Program of Thought)

Multi-round code generation with execution feedback for Excel/spreadsheet manipulation.

---

## Optimized Multi-Round Strategy

### Key Improvements (vs basic React)

| Problem | Solution |
|---------|----------|
| Wrong output type (改文件 vs 写公式) | **Task routing**: First determine if task needs formula/VBA/data modification |
| State not preserved between rounds | Remind model to reload workbook each round |
| Guess structure then fix | **Structure probe first**: Always explore sheets/tables/columns in Round 1 |
| No verification | **Self-check**: Verify target cell before saving |

### Multi-Round Flow

```
Round 1: Structure Exploration
├── List all sheets
├── List tables/named ranges
├── Print first few rows
└── Identify column types

Round 2+: Solution Implementation
├── Based on discovered structure
├── No guessing
└── Handle errors if any

Final: Self-Check
├── Verify answer_position has expected value
└── Save to output_path
```

---

## Settings

| Setting | Data Preview | Multi-round | Description |
|---------|-------------|-------------|-------------|
| `row_react_exec` | ✅ | ✅ | Default, with optimized prompt |
| `pure_react_exec` | ❌ | ✅ | Model explores on its own |
| `react_exec` | ✅ | ❌ | Single-round baseline |
| `compare` | - | - | Run baseline + multi-round |

---

## Usage

```bash
# Single JSON output, no intermediate files
python spreadsheetbench/runner.py --limit 20 --setting compare -o results.json

# Multi-round only
python spreadsheetbench/runner.py --limit 50 --setting row_react_exec --max-turns 5

# Baseline only
python spreadsheetbench/runner.py --limit 50 --setting react_exec
```

---

## Output Format

Single JSON file containing:
```json
{
  "meta": {"model", "settings", "max_turns", "timestamp"},
  "metrics": {
    "react_exec": {"soft_restriction_avg", "hard_restriction_avg", "by_type"},
    "row_react_exec": {...}
  },
  "traces": [
    {
      "id": "sample_id",
      "instruction": "...",
      "settings": {
        "react_exec": {
          "turns": 1,
          "rounds": [{round details}],
          "final_code": "...",
          "evaluation": {soft, hard, test_results}
        },
        "row_react_exec": {...}
      }
    }
  ]
}
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Soft Restriction** | % of test cases passed (0-100%) |
| **Hard Restriction** | 1 if ALL test cases pass, 0 otherwise |

Paper reference: GPT-4o ~18% Hard, Human ~71% Hard
