# Skill Benchmark

Benchmark comparing Claude's baseline vs skill-augmented performance on QA tasks.

## Benchmarks & Commands

| Benchmark | Task | Metric | Command |
|-----------|------|--------|---------|
| **FinQA** | Financial QA | Exact Match | `python finqa/runner.py --limit 50` |
| **TableBench** | Table QA | EM / EM±10% | `python tablebench/runner.py --limit 50` |
| **SealQA** | Search QA | LLM Grading | `python sealqa/runner.py --limit 50` |
| **MMLongBench** | PDF QA | ANLS / F1 | `python mmlongbench/runner.py --limit 50` |
| **ChartQAPro** | Chart QA | Relaxed Acc | `python chartqapro/runner.py --limit 50` |
| **SpreadsheetBench** | Excel Code Gen | Hard/Soft | `python spreadsheetbench/runner.py --limit 50` |

## Skills

| Skill | Purpose | Benchmark |
|-------|---------|-----------|
| `chartqa_cot` | 5-stage CoT prompting for chart QA | ChartQAPro |
| `finqa_reasoning` | Step-by-step financial calculation | FinQA |
| `formula_code_assistant` | Generate Python for numeric computation | FinQA |
| `table_reasoning` | Structured table analysis (TCoT) | TableBench |
| `tablebench_pot` | Program-of-Thought code execution | TableBench |
| `web_search_tool` | Web search integration | SealQA |
| `conflicting_info_reasoner` | Resolve contradictory sources | SealQA |
| `pdf_document_qa` | PDF comprehension strategies | MMLongBench |
| `pdf_text_extractor` | Extract text from PDF pages | MMLongBench |
| `pdf_retriever` | Semantic page retrieval via Voyage AI | MMLongBench |
| `spreadsheet_pot` | Multi-round ReAct for Excel manipulation | SpreadsheetBench |

## Key Options

```bash
# All benchmarks use sensible defaults, just add --limit
python finqa/runner.py --limit 50
python tablebench/runner.py --limit 50      # default: huggingface, NOVIZ
python sealqa/runner.py --limit 50          # default: seal_hard, builtin search
python mmlongbench/runner.py --limit 50
python chartqapro/runner.py --limit 50
python spreadsheetbench/runner.py --limit 50  # default: verified_400, compare

# Optional overrides
python tablebench/runner.py --qtype FC --limit 50      # specific question type
python sealqa/runner.py --source seal_0 --limit 50     # different dataset
python spreadsheetbench/runner.py --setting row_react_exec --limit 50  # single setting
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
