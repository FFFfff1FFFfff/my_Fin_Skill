name	spreadsheet_pot
description	Program-of-Thought style multi-round React framework for spreadsheet manipulation tasks

# Spreadsheet PoT (Program of Thought)

Multi-round code generation with execution feedback for Excel/spreadsheet manipulation.

---

## Prompt Templates

### PROMPT_DF_RCT_FORMAT (Data + React)

Primary template for multi-round interaction with data preview.

```
You are a spreadsheet manipulation agent. I will provide you with the following information:

Instruction: {instruction}
Spreadsheet file path: {spreadsheet_path}

The spreadsheet has the following content (first rows of each sheet):
{content}

Instruction type: {type}
Answer position: {answer_position}

The solution can be generated through {max_turn_num} rounds of interaction. You can take one of the two actions:
1. Spreadsheet information acquisition: If the information I provide is not enough to solve the problem, you can write python code to load the spreadsheet and access more information.
2. Question solution: Provide the final python code. If the code you write has an error, I will provide the error message, and you can fix the code.

Please generate Python code using openpyxl library.
- Load from: input_file (variable provided)
- Save to: output_file (variable provided)

```python
from openpyxl import load_workbook

wb = load_workbook(input_file)
# Your code here
wb.save(output_file)
```
```

### PROMPT_NO_DF_RCT_FORMAT (Pure React)

Template for multi-round without data preview (model explores on its own).

```
You are a spreadsheet manipulation agent. I will provide you with the following information:

Instruction: {instruction}
Spreadsheet file path: {spreadsheet_path}
Instruction type: {type}
Answer position: {answer_position}

The solution can be generated through {max_turn_num} rounds of interaction. You can take one of the two actions:
1. Spreadsheet information acquisition: Write python code to load the spreadsheet and explore its structure and content.
2. Question solution: Provide the final python code. If the code you write has an error, I will provide the error message, and you can fix the code.

Please generate Python code using openpyxl library.
- Load from: input_file (variable provided)
- Save to: output_file (variable provided)

```python
from openpyxl import load_workbook

wb = load_workbook(input_file)
# Your code here
wb.save(output_file)
```
```

### PROMPT_FORMAT_SINGLE (Baseline)

Single-round template without interaction.

```
You are a spreadsheet manipulation agent. I will provide you with the following information:

Instruction: {instruction}
Spreadsheet file path: {spreadsheet_path}

The spreadsheet has the following content (first rows of each sheet):
{content}

Instruction type: {type}
Answer position: {answer_position}

Please generate Python code using openpyxl library.
- The spreadsheet is available via the `file_path` variable
- Save to the same `file_path` after modification

```python
from openpyxl import load_workbook

wb = load_workbook(file_path)
# Your code here
wb.save(file_path)
```
```

---

## PoT Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Round 1: Initial Prompt                                    │
│  - Provide instruction, data preview, constraints           │
│  - Model generates Python code                              │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Execute Code                                               │
│  - Run generated code with input_file/output_file           │
│  - Capture stdout, stderr, exceptions                       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
              ┌────────────┴────────────┐
              │  Output file created?   │
              └────────────┬────────────┘
                    Yes    │    No
              ┌────────────┴────────────┐
              ↓                         ↓
        ┌─────────┐            ┌────────────────────┐
        │  Done   │            │  Round 2..N:       │
        └─────────┘            │  - Send exec result│
                               │  - Model fixes code│
                               │  - Execute again   │
                               └────────────────────┘
```

---

## Settings

| Setting | Data Preview | Multi-round | Use Case |
|---------|-------------|-------------|----------|
| `row_react_exec` | Yes | Yes | Default, best performance |
| `pure_react_exec` | No | Yes | Test model exploration |
| `react_exec` | Yes | No | Baseline comparison |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Soft Restriction** | Percentage of test cases passed (0-100%) |
| **Hard Restriction** | 1 if ALL test cases pass, 0 otherwise |

Paper best: ~18% Hard (GPT-4o), Human: 71%
