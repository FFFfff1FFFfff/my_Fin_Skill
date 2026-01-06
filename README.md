# Skill Benchmark

Benchmark comparing Claude's baseline vs skill-augmented performance on multiple QA tasks.

## Benchmarks

| Benchmark | Task | Evaluation | Skills |
|-----------|------|------------|--------|
| **FinQA** | Financial QA | Exact match (5 decimals) | finqa_reasoning, formula_code_assistant |
| **TableBench** | Table QA | EM / EM±10% | table_reasoning |
| **SealQA** | Search-augmented QA | LLM grading | web_search_tool, conflicting_info_reasoner |

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key"

# FinQA
python finqa/runner.py --source sample

# TableBench
python tablebench/runner.py --source sample

# SealQA (with web search)
python sealqa/runner.py --source sample --search --backend builtin
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

## Usage

Each benchmark supports:
- `--source`: Data source (`sample`, `hf`, or file path)
- `--limit`: Number of samples
- `--model`: Model to use (default: claude-sonnet-4-5-20250929)

SealQA also supports:
- `--search/--no-search`: Enable/disable web search
- `--backend`: Search backend (`builtin`, `tavily`, `serper`)
