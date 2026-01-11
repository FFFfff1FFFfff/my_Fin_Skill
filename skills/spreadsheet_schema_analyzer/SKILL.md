# SpreadsheetBench Skills

Based on paper insights, simplified to two modes:

## Available Modes

### 1. Baseline (Single-round)
```
Simple prompt → Generate code → Done
```
- Matches original paper
- ~500-1500 chars prompt

### 2. ReAct (Multi-round with Error Feedback)
```
Generate → Execute → Error? → Fix → Repeat (up to N times)
```
- Paper's key improvement
- **Proven to ~2x performance** for most models

## Paper's Key Findings

| What Works | What Doesn't |
|------------|--------------|
| ✅ Multi-round + execution feedback | ❌ More preview rows (5→10→20) |
| ✅ Strong code models | ❌ Complex prompts |
| ✅ Simple, direct prompts | ❌ TableQA methods (0%) |

## Why Simplified?

Previous "schema analysis" and "robust code" prompts:
- Generated 6000-9000 char code
- More complex = more bugs
- Extra tokens didn't help

New approach:
- Simple prompts (~500 chars)
- Let model write natural code
- Fix via execution feedback

## Usage

```bash
# Baseline
python spreadsheetbench/runner.py --limit 20 --mode baseline

# ReAct (recommended)
python spreadsheetbench/runner.py --limit 20 --mode react --max-turns 5
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Soft Restriction** | % of test cases passed |
| **Hard Restriction** | 1 if ALL 3 test cases pass |

Paper's best result: ~18% Hard Restriction (GPT-4o)
Human performance: 71%

This is a very hard benchmark.
