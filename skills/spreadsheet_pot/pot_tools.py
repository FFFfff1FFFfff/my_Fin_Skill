"""
PoT (Program of Thought) tools for spreadsheet manipulation.
Provides code execution and extraction utilities.
"""

import os
import re
import subprocess
import tempfile


def extract_code(response: str) -> str:
    """
    Extract Python code from LLM response.

    Args:
        response: The full LLM response text

    Returns:
        Extracted Python code, or original response if no code block found
    """
    patterns = [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    return response.strip()


def execute_code(code: str, input_file: str, output_file: str, timeout: int = 30) -> dict:
    """
    Execute Python code for spreadsheet manipulation.

    Args:
        code: Python code to execute
        input_file: Path to input spreadsheet
        output_file: Path to save output spreadsheet
        timeout: Execution timeout in seconds

    Returns:
        Dict with keys: success (bool), output (str), error (str)
    """
    wrapper = f'''
import sys
sys.path.insert(0, '.')
input_file = r"{input_file}"
output_file = r"{output_file}"
file_path = input_file

{code}
'''
    wrapper_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper)
            wrapper_path = f.name

        proc = subprocess.run(
            ['python', wrapper_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(input_file) or '.',
        )

        if proc.returncode != 0:
            error_lines = proc.stderr.strip().split('\n')[-10:]
            return {"success": False, "output": "", "error": '\n'.join(error_lines)}

        return {"success": True, "output": proc.stdout, "error": ""}

    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}
    finally:
        if wrapper_path and os.path.exists(wrapper_path):
            os.remove(wrapper_path)


def format_exec_result(result: dict, output_file: str) -> str:
    """
    Format execution result as feedback message for LLM.

    Args:
        result: Dict from execute_code()
        output_file: Path to check for output file existence

    Returns:
        Formatted string to send back to LLM
    """
    if result["success"]:
        output = result["output"].strip()
        if output:
            return output
        if os.path.exists(output_file):
            return "Code executed successfully. Output file created."
        return "Code executed successfully."
    else:
        return f"Error occurred when running code.\n{result['error']}"


def check_output_exists(output_file: str) -> bool:
    """
    Check if output file was created.

    Args:
        output_file: Path to output file

    Returns:
        True if file exists
    """
    return os.path.exists(output_file)


# Prompt templates as constants
PROMPT_DF_RCT_FORMAT = """You are a spreadsheet manipulation agent. I will provide you with the following information:

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
```"""

PROMPT_NO_DF_RCT_FORMAT = """You are a spreadsheet manipulation agent. I will provide you with the following information:

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
```"""

PROMPT_FORMAT_SINGLE = """You are a spreadsheet manipulation agent. I will provide you with the following information:

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
```"""


def build_prompt(sample: dict, setting: str = "row_react_exec", max_turn_num: int = 5) -> str:
    """
    Build prompt from sample data.

    Args:
        sample: Dict with instruction, preview, instruction_type, answer_position, etc.
        setting: One of "row_react_exec", "pure_react_exec", "react_exec"
        max_turn_num: Maximum interaction rounds

    Returns:
        Formatted prompt string
    """
    # Get spreadsheet path from first test case if available
    spreadsheet_path = "spreadsheet.xlsx"
    if sample.get("test_cases"):
        spreadsheet_path = sample["test_cases"][0].get("input_file", spreadsheet_path)

    params = {
        "instruction": sample["instruction"],
        "spreadsheet_path": spreadsheet_path,
        "content": sample.get("preview", "")[:3000],
        "type": sample["instruction_type"],
        "answer_position": sample["answer_position"],
        "max_turn_num": max_turn_num,
    }

    if setting == "row_react_exec":
        return PROMPT_DF_RCT_FORMAT.format(**params)
    elif setting == "pure_react_exec":
        return PROMPT_NO_DF_RCT_FORMAT.format(**params)
    else:  # react_exec (single round)
        return PROMPT_FORMAT_SINGLE.format(**params)
