"""
PoT (Program of Thought) tools for spreadsheet manipulation.
Aligned with official SpreadsheetBench inference scripts.
"""

import os
import re
import subprocess
import tempfile


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
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

    The code should use spreadsheet_path (input) and output_path (output) variables,
    matching the official prompt format.
    """
    wrapper = f'''
import sys
import os
sys.path.insert(0, '.')
os.chdir(r"{os.path.dirname(input_file) or '.'}")

# Set file paths as variables the code expects
spreadsheet_path = r"{input_file}"
output_path = r"{output_file}"

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
        )

        if proc.returncode != 0:
            error_lines = proc.stderr.strip().split('\n')[-10:]
            return {"success": False, "output": proc.stdout, "error": '\n'.join(error_lines)}

        return {"success": True, "output": proc.stdout, "error": ""}

    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}
    finally:
        if wrapper_path and os.path.exists(wrapper_path):
            os.remove(wrapper_path)


def format_exec_result(result: dict, output_file: str) -> str:
    """Format execution result as feedback message for LLM."""
    if result["success"]:
        output = result["output"].strip()
        if output:
            return output
        if os.path.exists(output_file):
            return "Code executed successfully. Output file created."
        return "Code executed successfully. Note: output file was NOT created yet."
    else:
        return f"Error occurred:\n{result['error']}"


def check_output_exists(output_file: str) -> bool:
    """Check if output file was created."""
    return os.path.exists(output_file)


# =============================================================================
# Optimized Prompt Templates (with task routing + structure probe + self-check)
# =============================================================================

PROMPT_FORMAT_SINGLE = """You are a spreadsheet expert who can manipulate spreadsheets through Python code.

You need to solve the given spreadsheet manipulation question, which contains six types of information:
- instruction: The question about spreadsheet manipulation.
- spreadsheet_path: The path of the spreadsheet file you need to manipulate.
- spreadsheet_content: The first few rows of the content of speadsheet file.
- instruction_type: There are two values (Cell-Level Manipulation, Sheet-Level Manipulation) used to indicate whether the answer to this question applies only to specific cells or to the entire worksheet.
- answer_position: The position need to be modified or filled. For Cell-Level Manipulation questions, this field is filled with the cell position; for Sheet-Level Manipulation, it is the maximum range of cells you need to modify. You only need to modify or fill in values within the cell range specified by answer_position.
- output_path: You need to generate the modified spreadsheet file in this new path.

Below is the spreadsheet manipulation question you need to solve:
### instruction
{instruction}

### spreadsheet_path
{spreadsheet_path}

### spreadsheet_content
{spreadsheet_content}

### instruction_type
{instruction_type}

### answer_position
{answer_position}

### output_path
{output_path}

You should generate Python code for the final solution of the question.
"""

PROMPT_NO_DF_RCT_FORMAT = """You are a spreadsheet expert who can manipulate spreadsheets through Python code.

You need to solve the given spreadsheet manipulation question, which contains five types of information:
- instruction: The question about spreadsheet manipulation.
- spreadsheet_path: The path of the spreadsheet file you need to manipulate.
- instruction_type: There are two values (Cell-Level Manipulation, Sheet-Level Manipulation) used to indicate whether the answer to this question applies only to specific cells or to the entire worksheet.
- answer_position: The position need to be modified or filled. For Cell-Level Manipulation questions, this field is filled with the cell position; for Sheet-Level Manipulation, it is the maximum range of cells you need to modify. You only need to modify or fill in values within the cell range specified by answer_position.
- output_path: You need to generate the modified spreadsheet file in this new path.

Below is the spreadsheet manipulation question you need to solve:
### instruction
{instruction}

### spreadsheet_path
{spreadsheet_path}

### instruction_type
{instruction_type}

### answer_position
{answer_position}

### output_path
{output_path}

The solution of the question can be generate through {max_turn_num} rounds of interaction and you can do two types of actions.
1. Spreadsheet information acquisition: You can generate Python code to obtain the information in the spreadsheet file. In the next turn, the execution result of you Python code will provide to you.
2. Question solution generation: You can generate Python code for the final solution of the question. If error occur when executing code, the error traceback will provide to you for code refinement.
"""

# Optimized multi-round prompt with task routing, structure probe, and self-check
PROMPT_DF_RCT_FORMAT = """You are a spreadsheet expert who can manipulate spreadsheets through Python code.

You need to solve the given spreadsheet manipulation question, which contains six types of information:
- instruction: The question about spreadsheet manipulation.
- spreadsheet_path: The path of the spreadsheet file you need to manipulate.
- spreadsheet_content: The first few rows of the content of speadsheet file.
- instruction_type: There are two values (Cell-Level Manipulation, Sheet-Level Manipulation) used to indicate whether the answer to this question applies only to specific cells or to the entire worksheet.
- answer_position: The position need to be modified or filled. For Cell-Level Manipulation questions, this field is filled with the cell position; for Sheet-Level Manipulation, it is the maximum range of cells you need to modify. You only need to modify or fill in values within the cell range specified by answer_position.
- output_path: You need to generate the modified spreadsheet file in this new path.

Below is the spreadsheet manipulation question you need to solve:
### instruction
{instruction}

### spreadsheet_path
{spreadsheet_path}

### spreadsheet_content
{spreadsheet_content}

### instruction_type
{instruction_type}

### answer_position
{answer_position}

### output_path
{output_path}

The solution of the question can be generate through {max_turn_num} rounds of interaction and you can do two types of actions.
1. Spreadsheet information acquisition: You can generate Python code to obtain the information in the spreadsheet file. In the next turn, the execution result of you Python code will provide to you.
2. Question solution generation: You can generate Python code for the final solution of the question. If error occur when executing code, the error traceback will provide to you for code refinement.
"""


def build_prompt(sample: dict, setting: str, max_turn_num: int, output_path: str) -> str:
    """
    Build prompt from sample data.

    Args:
        sample: Dict with instruction, preview, instruction_type, answer_position, etc.
        setting: One of "row_react_exec", "pure_react_exec", "react_exec"
        max_turn_num: Maximum interaction rounds
        output_path: Path where output file should be saved
    """
    # Get spreadsheet path from first test case
    spreadsheet_path = sample["test_cases"][0]["input_file"] if sample.get("test_cases") else "spreadsheet.xlsx"

    params = {
        "instruction": sample["instruction"],
        "spreadsheet_path": spreadsheet_path,
        "spreadsheet_content": sample.get("preview", "")[:5000],
        "instruction_type": sample["instruction_type"],
        "answer_position": sample["answer_position"],
        "output_path": output_path,
        "max_turn_num": max_turn_num,
    }

    if setting == "row_react_exec":
        return PROMPT_DF_RCT_FORMAT.format(**params)
    elif setting == "pure_react_exec":
        return PROMPT_NO_DF_RCT_FORMAT.format(**params)
    else:  # react_exec (single round baseline)
        return PROMPT_FORMAT_SINGLE.format(**params)
