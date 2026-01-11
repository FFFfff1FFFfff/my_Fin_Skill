# Skill Benchmark

Benchmark comparing Claude's baseline vs skill-augmented performance on multiple QA tasks.

## Benchmarks

| Benchmark | Task | Evaluation | Skills |
|-----------|------|------------|--------|
| **FinQA** | Financial QA | Exact match (5 decimals) | finqa_reasoning, formula_code_assistant |
| **TableBench** | Table QA | EM / EM±10% | table_reasoning |
| **SealQA** | Search-augmented QA | LLM grading | web_search_tool, conflicting_info_reasoner |
| **MMLongBench-Doc** | PDF Document QA | ANLS / F1 | pdf_document_qa, pdf_text_extractor |
| **ChartQAPro** | Chart QA | Relaxed Accuracy (5% tolerance) | chart_data_extractor |
| **SpreadsheetBench** | Spreadsheet Manipulation | OJ-style (Soft/Hard Restriction) | spreadsheet_schema_analyzer |

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

# ChartQAPro - chart question answering
python chartqapro/runner.py --limit 50

# ChartQAPro - filter by question type
python chartqapro/runner.py --limit 50 --type Reasoning --type "Fact Checking"

# SpreadsheetBench - spreadsheet manipulation (all modes)
python spreadsheetbench/runner.py --limit 10

# SpreadsheetBench - specific modes
python spreadsheetbench/runner.py --limit 50 --mode baseline    # Single-round (original paper)
python spreadsheetbench/runner.py --limit 50 --mode react       # Multi-round with error feedback
python spreadsheetbench/runner.py --limit 50 --mode explore     # Code-based exploration (paper's key insight)
python spreadsheetbench/runner.py --limit 50 --mode schema      # Schema-first analysis
python spreadsheetbench/runner.py --limit 50 --mode combined    # Schema + ReAct

# SpreadsheetBench - filter by instruction type
python spreadsheetbench/runner.py --limit 50 --cell-level
python spreadsheetbench/runner.py --limit 50 --sheet-level
```

## Project Structure

```
├── finqa/              # FinQA benchmark
├── tablebench/         # TableBench benchmark
├── sealqa/             # SealQA benchmark
├── mmlongbench/        # MMLongBench-Doc benchmark
├── chartqapro/         # ChartQAPro benchmark
├── spreadsheetbench/   # SpreadsheetBench benchmark
├── skills/
│   ├── finqa_reasoning/
│   ├── formula_code_assistant/
│   ├── table_reasoning/
│   ├── web_search_tool/
│   ├── conflicting_info_reasoner/
│   ├── pdf_document_qa/
│   ├── pdf_text_extractor/
│   ├── chart_data_extractor/
│   └── spreadsheet_schema_analyzer/
└── skill_system.py     # Skill loader
```
