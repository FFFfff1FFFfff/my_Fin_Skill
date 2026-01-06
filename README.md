# Skill Benchmark

Benchmark comparing Claude's baseline vs skill-augmented performance on multiple QA tasks.

## Benchmarks

| Benchmark | Task | Evaluation | Skills |
|-----------|------|------------|--------|
| **FinQA** | Financial QA | Exact match (5 decimals) | finqa_reasoning, formula_code_assistant |
| **TableBench** | Table QA | EM / EM±10% | table_reasoning |
| **SealQA** | Search-augmented QA | LLM grading | web_search_tool, conflicting_info_reasoner |

## Usage

```bash
# FinQA - test with built-in samples
python finqa/runner.py --source sample --limit 10

# FinQA - test with dataset file
python finqa/runner.py --source finqa_test.json --limit 50

# TableBench - test with built-in samples
python tablebench/runner.py --source sample --limit 10

# TableBench - load from HuggingFace
python tablebench/runner.py --source hf --limit 50

# SealQA - with web search enabled
python sealqa/runner.py --source sample --limit 10 --search --backend builtin

# SealQA - without web search
python sealqa/runner.py --source sample --limit 10 --no-search
```

## Project Structure

```
├── finqa/              # FinQA benchmark
├── tablebench/         # TableBench benchmark
├── sealqa/             # SealQA benchmark
├── skills/
│   ├── finqa_reasoning/
│   ├── formula_code_assistant/
│   ├── table_reasoning/
│   ├── web_search_tool/
│   └── conflicting_info_reasoner/
└── skill_system.py     # Skill loader
```
