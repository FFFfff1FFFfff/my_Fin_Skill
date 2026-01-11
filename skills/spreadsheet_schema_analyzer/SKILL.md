# SpreadsheetBench Skills

Based on insights from the original paper, this benchmark tests different approaches for spreadsheet manipulation.

## Paper's Key Findings

| Finding | Implication |
|---------|-------------|
| ✅ Multi-round + execution feedback | Most effective improvement method |
| ✅ Strong code models > general LLMs | DeepseekCoder outperforms GPT-3.5 |
| ❌ Static text preview | Limited effect (行数 5→10 几乎不变) |
| ❌ TableQA methods | Completely fail (0%) - this is "改表" not "读表" |
| ❌ Single-round prompt | High risk of overfitting to one test case |

## Available Modes

### 1. Baseline (Single-round)
```
Prompt with preview → Generate code → Done
```
- Matches original paper's baseline
- Problem: Easy to overfit to the sample data

### 2. ReAct (Multi-round with Error Feedback)
```
Generate → Execute → Error? → Fix → Repeat
```
- Original paper's key improvement
- **Proven to double performance** for most models
- GPT-4o: minimal improvement (already strong first round)

### 3. Explore (Code-based Exploration) ⭐ NEW
```
NO static preview given
↓
Model generates exploration code
↓
Execute → See real spreadsheet structure
↓
Generate solution based on real output
↓
ReAct refinement if needed
```
- **Paper's key insight**: Let model discover structure via code
- More accurate than static text preview
- Model sees real dimensions, headers, data patterns

### 4. Schema-First (Static Analysis)
```
Static text preview → Analyze structure → Generate robust code
```
- Our original approach
- Limitation: Text preview may miss important details

### 5. Combined (Schema + ReAct)
```
Static analysis → Generate code → ReAct refinement
```
- Combines static analysis with error feedback

## Why Explore Mode Matters

The paper shows that increasing preview rows (5→10→20) barely helps:

> "行数从 5 → 10, 性能几乎不变"

Because:
1. Static preview can't capture: merged cells, multiple tables, non-standard headers
2. Real structure is only discoverable by actually reading the Excel file
3. Code execution provides ground truth

The **Explore** mode addresses this by:
1. NOT giving static preview (avoids false assumptions)
2. Model writes code to inspect the real file
3. Model sees actual output (dimensions, structure, sample data)
4. Then generates solution based on real information

## Recommended Approach

Based on paper findings:

```bash
# Best for most cases
python spreadsheetbench/runner.py --limit 50 --mode explore --max-turns 3

# If explore doesn't have execution environment
python spreadsheetbench/runner.py --limit 50 --mode react --max-turns 3
```

## Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Soft Restriction** | % of test cases passed | Partial success |
| **Hard Restriction** | 1 if ALL pass, 0 otherwise | Production-ready |

Paper emphasizes: **Hard restriction reflects real-world usability**.

A solution that works on 2/3 test cases is still broken for production.
