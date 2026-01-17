# Skill Benchmark

Benchmark comparing Claude's baseline vs skill-augmented performance on QA tasks.

## Benchmarks & Commands

| Benchmark | Task | Metric | Command |
|-----------|------|--------|---------|
| **FinQA** | Financial QA | Exact Match | `python finqa/runner.py --limit 50` |
| **TableBench** | Table QA | EM / EM±10% | `python tablebench/runner.py --source hf --limit 50` |
| **SealQA** | Search QA | LLM Grading | `python sealqa/runner.py --search --limit 10` |
| **MMLongBench** | PDF QA | ANLS / F1 | `python mmlongbench/runner.py --limit 50` |
| **ChartQAPro** | Chart QA | Relaxed Acc | `python chartqapro/runner.py --limit 50` |
| **SpreadsheetBench** | Excel Code Gen | Hard/Soft | `python spreadsheetbench/runner.py --mode all --limit 20` |

## Skills

| Skill | Purpose | Benchmark |
|-------|---------|-----------|
| `finqa_reasoning` | Step-by-step financial calculation | FinQA |
| `formula_code_assistant` | Generate Python for numeric computation | FinQA |
| `table_reasoning` | Structured table analysis | TableBench |
| `web_search_tool` | Web search integration | SealQA |
| `conflicting_info_reasoner` | Resolve contradictory sources | SealQA |
| `pdf_document_qa` | PDF comprehension strategies | MMLongBench |
| `pdf_text_extractor` | Extract text/tables from PDF | MMLongBench |
| `chart_data_extractor` | Extract data from chart images | ChartQAPro |
| `spreadsheet_schema_analyzer` | Excel structure analysis | SpreadsheetBench |

## Key Options

```bash
# FinQA - specify dataset file
python finqa/runner.py --source test.json --limit 50

# SealQA - enable web search
python sealqa/runner.py --search --backend builtin --limit 10

# MMLongBench - reports both all & answerable-only metrics
python mmlongbench/runner.py --official --limit 50

# ChartQAPro - filter by type
python chartqapro/runner.py --type Reasoning --limit 50

# SpreadsheetBench - compare baseline vs react
python spreadsheetbench/runner.py --mode all --limit 20

# SpreadsheetBench - filter by instruction type
python spreadsheetbench/runner.py --cell-level --limit 50
```

## Project Structure

```
├── finqa/              # Financial QA
├── tablebench/         # Table QA
├── sealqa/             # Search-augmented QA
├── mmlongbench/        # PDF Document QA
├── chartqapro/         # Chart QA
├── spreadsheetbench/   # Spreadsheet manipulation
├── skills/             # Skill prompts (SKILL.md files)
└── skill_system.py     # Skill loader
```
