"""
Calculation tools for formula generation and code execution
Provides utilities to assist with complex financial calculations
"""

import json
import re
from typing import Dict, Any, Union


def generate_calculation_formula(question: str, data_points: str) -> str:
    """
    Generate a step-by-step mathematical formula for the calculation
    
    Args:
        question: The financial question being asked
        data_points: JSON string or dict with extracted values, e.g., '{"old": 500, "new": 550}'
    
    Returns:
        String with step-by-step formula breakdown
    """
    # Parse data_points if string
    if isinstance(data_points, str):
        try:
            data = json.loads(data_points)
        except json.JSONDecodeError:
            return f"Error: Invalid data_points format. Expected JSON, got: {data_points}"
    else:
        data = data_points
    
    q_lower = question.lower()
    
    # Detect question type and generate appropriate formula
    if "percentage change" in q_lower or "% change" in q_lower:
        if "old" in data and "new" in data:
            old_val = data["old"]
            new_val = data["new"]
            change = new_val - old_val
            percentage = (change / old_val) * 100 if old_val != 0 else 0
            
            return f"""Formula: Percentage Change = ((New - Old) / Old) × 100

Step 1: Calculate change
   Change = {new_val} - {old_val} = {change}

Step 2: Calculate percentage
   Percentage = ({change} / {old_val}) × 100 = {percentage:.2f}%

Result: {percentage:.2f}%"""
    
    elif "percentage" in q_lower or "percent" in q_lower:
        if "part" in data and "total" in data:
            part = data["part"]
            total = data["total"]
            percentage = (part / total) * 100 if total != 0 else 0
            
            return f"""Formula: Percentage = (Part / Total) × 100

Step 1: Identify values
   Part = {part}
   Total = {total}

Step 2: Calculate percentage
   Percentage = ({part} / {total}) × 100 = {percentage:.2f}%

Result: {percentage:.2f}%"""
    
    elif "change" in q_lower or "difference" in q_lower:
        if "old" in data and "new" in data:
            old_val = data["old"]
            new_val = data["new"]
            change = new_val - old_val
            
            return f"""Formula: Change = New - Old

Calculation:
   Change = {new_val} - {old_val} = {change}

Result: {change}"""
    
    elif "average" in q_lower or "mean" in q_lower:
        if "values" in data:
            values = data["values"]
            if isinstance(values, list):
                total = sum(values)
                count = len(values)
                average = total / count if count > 0 else 0
                
                values_str = " + ".join(str(v) for v in values)
                return f"""Formula: Average = Sum of Values / Count

Step 1: Sum values
   Sum = {values_str} = {total}

Step 2: Count values
   Count = {count}

Step 3: Calculate average
   Average = {total} / {count} = {average:.2f}

Result: {average:.2f}"""
    
    elif "total" in q_lower or "sum" in q_lower:
        if "values" in data:
            values = data["values"]
            if isinstance(values, list):
                total = sum(values)
                values_str = " + ".join(str(v) for v in values)
                
                return f"""Formula: Total = Sum of Values

Calculation:
   Total = {values_str} = {total}

Result: {total}"""
    
    # Generic calculation for provided values
    if len(data) == 2 and all(isinstance(v, (int, float)) for v in data.values()):
        keys = list(data.keys())
        vals = list(data.values())
        
        return f"""Data Points:
   {keys[0]} = {vals[0]}
   {keys[1]} = {vals[1]}

Common operations:
   Addition: {vals[0]} + {vals[1]} = {vals[0] + vals[1]}
   Subtraction: {vals[0]} - {vals[1]} = {vals[0] - vals[1]}
   Ratio: {vals[0]} / {vals[1]} = {vals[0] / vals[1]:.4f}
   Percentage: ({vals[0]} / {vals[1]}) × 100 = {(vals[0] / vals[1]) * 100:.2f}%"""
    
    return f"""Unable to auto-generate formula. 
Data provided: {json.dumps(data, indent=2)}
Please specify the operation type or provide more context."""


def generate_python_code(question: str, data_points: str, operation_type: str = "auto") -> str:
    """
    Generate executable Python code for the calculation
    
    Args:
        question: The financial question
        data_points: JSON string with values
        operation_type: Type of operation - "sum", "average", "percentage", "change", "auto"
    
    Returns:
        Python code as string
    """
    # Parse data_points
    if isinstance(data_points, str):
        try:
            data = json.loads(data_points)
        except json.JSONDecodeError:
            return f"# Error: Invalid JSON\n# {data_points}"
    else:
        data = data_points
    
    # Auto-detect operation type if not specified
    if operation_type == "auto":
        q_lower = question.lower()
        if "percentage change" in q_lower:
            operation_type = "percentage_change"
        elif "percentage" in q_lower or "percent" in q_lower:
            operation_type = "percentage"
        elif "change" in q_lower or "difference" in q_lower:
            operation_type = "change"
        elif "total" in q_lower or "sum" in q_lower:
            operation_type = "sum"
        elif "average" in q_lower or "mean" in q_lower:
            operation_type = "average"
        else:
            operation_type = "generic"
    
    # Generate code based on operation type
    if operation_type == "percentage_change":
        if "old" in data and "new" in data:
            return f"""# Percentage Change Calculation
old_value = {data['old']}
new_value = {data['new']}

change = new_value - old_value
percentage_change = (change / old_value) * 100

print(f"Change: {{change}}")
print(f"Percentage Change: {{percentage_change:.2f}}%")

# Result: {{percentage_change:.2f}}%
"""
    
    elif operation_type == "percentage":
        if "part" in data and "total" in data:
            return f"""# Percentage Calculation
part = {data['part']}
total = {data['total']}

percentage = (part / total) * 100

print(f"Percentage: {{percentage:.2f}}%")

# Result: {{percentage:.2f}}%
"""
    
    elif operation_type == "change":
        if "old" in data and "new" in data:
            return f"""# Change Calculation
old_value = {data['old']}
new_value = {data['new']}

change = new_value - old_value

print(f"Change: {{change}}")

# Result: {{change}}
"""
    
    elif operation_type == "sum":
        if "values" in data and isinstance(data["values"], list):
            values_str = str(data["values"])
            return f"""# Sum Calculation
values = {values_str}

total = sum(values)

print(f"Total: {{total}}")

# Result: {{total}}
"""
    
    elif operation_type == "average":
        if "values" in data and isinstance(data["values"], list):
            values_str = str(data["values"])
            return f"""# Average Calculation
values = {values_str}

total = sum(values)
count = len(values)
average = total / count

print(f"Average: {{average:.2f}}")

# Result: {{average:.2f}}
"""
    
    # Generic code generation
    data_assignments = "\n".join([f"{k} = {v}" for k, v in data.items()])
    return f"""# Calculation for: {question}

# Data
{data_assignments}

# Perform calculation
# (Modify as needed based on question)
result = None  # TODO: Add calculation logic

print(f"Result: {{result}}")
"""


def execute_calculation(expression: str) -> str:
    """
    Execute a Python expression and return the result
    
    Args:
        expression: Python expression or simple code to execute
    
    Returns:
        String with calculation result
    """
    # Safety check - only allow safe operations
    allowed_chars = set("0123456789+-*/(). ,[]")
    if not all(c in allowed_chars or c.isspace() for c in expression):
        # Check if it's using allowed functions
        allowed_functions = ["sum", "abs", "round", "min", "max", "len"]
        safe = any(func in expression for func in allowed_functions)
        if not safe:
            return f"Error: Expression contains potentially unsafe characters. Allowed: numbers, +, -, *, /, (), [], and functions: {', '.join(allowed_functions)}"
    
    try:
        # Create safe execution environment
        safe_dict = {
            "__builtins__": {
                "sum": sum,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "len": len,
            }
        }
        
        # Execute expression
        result = eval(expression, safe_dict, {})
        
        # Format result
        if isinstance(result, float):
            return f"""Result: {result}
Rounded (2 decimals): {round(result, 2)}
Rounded (1 decimal): {round(result, 1)}"""
        else:
            return f"Result: {result}"
    
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError as e:
        return f"Error: Invalid syntax - {str(e)}"
    except Exception as e:
        return f"Error: {type(e).__name__} - {str(e)}"

