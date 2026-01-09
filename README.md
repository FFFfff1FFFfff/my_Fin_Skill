# Skill Benchmark

Benchmark comparing Claude's baseline vs skill-augmented performance on multiple QA tasks.

## Benchmarks

| Benchmark | Task | Evaluation | Skills |
|-----------|------|------------|--------|
| **FinQA** | Financial QA | Exact match (5 decimals) | finqa_reasoning, formula_code_assistant |
| **TableBench** | Table QA | EM / EM±10% | table_reasoning |
| **SealQA** | Search-augmented QA | LLM grading | web_search_tool, conflicting_info_reasoner |
| **MMLongBench-Doc** | PDF Document QA | ANLS / F1 | pdf_document_qa |

## Usage

```bash
# FinQA - test with dataset file
python finqa/runner.py --source test.json --limit 50

# TableBench - load from HuggingFace
python tablebench/runner.py --source hf --limit 50

# SealQA - with web search enabled
python sealqa/runner.py --source sample --limit 10 --search --backend builtin

# MMLongBench-Doc - PDF document understanding
python mmlongbench/runner.py --limit 10

# MMLongBench-Doc - skip unanswerable questions
python mmlongbench/runner.py --limit 50 --skip-unanswerable
```

## Project Structure

```
├── finqa/              # FinQA benchmark
├── tablebench/         # TableBench benchmark
├── sealqa/             # SealQA benchmark
├── mmlongbench/        # MMLongBench-Doc benchmark
├── skills/
│   ├── finqa_reasoning/
│   ├── formula_code_assistant/
│   ├── table_reasoning/
│   ├── web_search_tool/
│   ├── conflicting_info_reasoner/
│   └── pdf_document_qa/
└── skill_system.py     # Skill loader
```
