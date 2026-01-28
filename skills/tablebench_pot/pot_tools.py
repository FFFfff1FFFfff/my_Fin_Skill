"""
TableBench POT (Program of Thought) Tools

Provides code extraction, safe execution, and visualization utilities
for table-based question answering.
"""

import io
import json
import re
import signal
from contextlib import redirect_stdout, redirect_stderr


# =============================================================================
# Code Extraction
# =============================================================================

def extract_python_code(text: str) -> str:
    """
    Extract Python code block from LLM response.

    Args:
        text: LLM response text containing code block

    Returns:
        Extracted Python code, or empty string if not found
    """
    # Try to find ```python ... ``` block
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find ``` ... ``` block
    match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Check if it looks like Python code
        if 'import' in code or 'print' in code or 'def ' in code:
            return code

    return ""


# =============================================================================
# Safe Code Execution
# =============================================================================

def execute_code_safely(code: str, timeout_seconds: int = 15) -> tuple:
    """
    Execute Python code safely with timeout and capture output.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time (default: 15)

    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    if not code:
        return False, "", "No code to execute"

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    try:
        # Set timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        # Execute code with captured output
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': None,
            'json': json,
        }

        # Import pandas if available
        try:
            import pandas as pd
            exec_globals['pd'] = pd
        except ImportError:
            pass

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)

        # Cancel timeout
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()

        return True, output, error

    except TimeoutError as e:
        signal.alarm(0)
        return False, "", str(e)
    except Exception as e:
        signal.alarm(0)
        return False, "", f"{type(e).__name__}: {str(e)}"


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer_from_output(output: str) -> str:
    """
    Extract Final Answer from code execution output.

    Args:
        output: stdout from code execution

    Returns:
        Extracted answer string
    """
    # Look for "Final Answer:" in output
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: return last non-empty line
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def extract_answer_from_response(text: str) -> str:
    """
    Extract final answer from LLM response text.

    Args:
        text: Full LLM response

    Returns:
        Extracted answer string
    """
    # Look for "Final Answer:" pattern (official format)
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for "Answer:" pattern
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for bold answer like **1251** or **1251
    match = re.search(r'\*\*(-?\d+\.?\d*)\*?\*?', text)
    if match:
        return match.group(1)

    # Look for "= number" at end of calculation
    match = re.search(r'=\s*(-?\d+\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        return match.group(1)

    # Try to find the last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


# =============================================================================
# Visualization Tools
# =============================================================================

def extract_chart_data(plt_module) -> list:
    """
    Extract y-data from matplotlib figure for comparison.

    Args:
        plt_module: matplotlib.pyplot module with active figure

    Returns:
        List of numeric data points from the chart
    """
    try:
        ax = plt_module.gca()
        data = []

        # Try line plots
        for line in ax.get_lines():
            ydata = line.get_ydata()
            if len(ydata) > 0:
                data.extend([float(y) for y in ydata if not (isinstance(y, float) and y != y)])

        # Try bar plots
        for patch in ax.patches:
            height = patch.get_height()
            if height and not (isinstance(height, float) and height != height):
                data.append(float(height))

        # Try pie charts (from wedges)
        for child in ax.get_children():
            if hasattr(child, 'theta2') and hasattr(child, 'theta1'):
                # Pie wedge - calculate proportion
                angle = child.theta2 - child.theta1
                data.append(round(angle / 360.0, 4))

        return data
    except Exception:
        return []


def execute_chart_code(code: str, timeout_seconds: int = 15) -> tuple:
    """
    Execute matplotlib code and extract chart data.

    Args:
        code: Python code that generates matplotlib chart
        timeout_seconds: Maximum execution time (default: 15)

    Returns:
        tuple: (success: bool, chart_data: list, error: str)
    """
    if not code:
        return False, [], "No code to execute"

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Prepare execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'plt': plt,
            'pd': None,
            'json': json,
        }

        try:
            import pandas as pd
            exec_globals['pd'] = pd
        except ImportError:
            pass

        # Clear any existing figures
        plt.clf()
        plt.close('all')

        # Set timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        # Execute code
        exec(code, exec_globals)

        # Cancel timeout
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        # Extract chart data
        chart_data = extract_chart_data(plt)

        # Clean up
        plt.close('all')

        return True, chart_data, ""

    except TimeoutError as e:
        signal.alarm(0)
        return False, [], str(e)
    except Exception as e:
        signal.alarm(0)
        return False, [], f"{type(e).__name__}: {str(e)}"


def compare_chart_data(pred_data: list, ref_data: list, tolerance: float = 0.02) -> bool:
    """
    Compare predicted chart data with reference data.

    Args:
        pred_data: Predicted data points from generated chart
        ref_data: Reference data points from ground truth
        tolerance: Relative tolerance for comparison (default: 0.02 = 2%)

    Returns:
        True if data matches within tolerance
    """
    if not pred_data or not ref_data:
        return False

    # Sort both lists for comparison
    pred_sorted = sorted([round(x, 2) for x in pred_data])
    ref_sorted = sorted([round(x, 2) for x in ref_data])

    if len(pred_sorted) != len(ref_sorted):
        return False

    # Compare with tolerance
    for p, r in zip(pred_sorted, ref_sorted):
        if abs(p - r) > tolerance * max(abs(r), 1):
            return False

    return True


def parse_viz_ground_truth(ground_truth: str) -> list:
    """
    Parse visualization ground truth to extract reference data.

    Args:
        ground_truth: Ground truth string, may contain y_references format

    Returns:
        List of reference data points
    """
    try:
        ref_data = []
        if 'y_references' in ground_truth:
            # Extract the list part after '='
            match = re.search(r'y_references\s*=\s*(\[.+\])', ground_truth)
            if match:
                # Use ast.literal_eval to safely parse Python list
                import ast
                nested_list = ast.literal_eval(match.group(1))
                # Flatten nested list: [[1,2], [3,4]] -> [1,2,3,4]
                for sublist in nested_list:
                    if isinstance(sublist, list):
                        ref_data.extend([float(x) for x in sublist])
                    else:
                        ref_data.append(float(sublist))
        elif ground_truth.startswith('['):
            ref_data = json.loads(ground_truth)
        else:
            # Try to parse as comma-separated numbers
            ref_data = [float(x.strip()) for x in ground_truth.split(',') if x.strip()]
        return ref_data
    except (json.JSONDecodeError, ValueError, SyntaxError):
        return []


# =============================================================================
# Prompt Building
# =============================================================================

def build_pot_prompt(question: str, table: str) -> str:
    """
    Build POT (Program of Thought) prompt for code generation.

    Args:
        question: The question to answer
        table: Table data in JSON format

    Returns:
        Formatted prompt string
    """
    return f"""You are a table analyst. Generate Python code to answer the question based on the table data.

**Requirements:**
1. Parse the table from JSON
2. Perform necessary calculations
3. Print the final answer in this EXACT format: `print(f"Final Answer: {{result}}")`

**Table (JSON format):**
{table}

**Question:** {question}

Generate Python code inside ```python``` block:"""


def build_viz_prompt(question: str, table: str) -> str:
    """
    Build visualization prompt for matplotlib code generation.

    Args:
        question: The visualization task description
        table: Table data in JSON format

    Returns:
        Formatted prompt string
    """
    return f"""You are a data visualization expert. Generate Python matplotlib code to create the requested chart.

**MANDATORY CODE FORMAT:**
Your code must start with these exact three lines:
```python
import matplotlib.pyplot as plt
import pandas as pd
import json
```

Then:
1. Parse the table data from JSON
2. Create the visualization using matplotlib
3. Do NOT call plt.show() - just create the figure

**Table (JSON format):**
{table}

**Task:** {question}

Generate the complete Python code inside ```python``` block:"""
