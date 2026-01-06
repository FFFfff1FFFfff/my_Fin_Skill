name	formula_code_assistant
description	Generates formulas and executable code for complex financial calculations and verification

# Formula & Code Assistant

This skill provides tools for generating mathematical formulas and Python code to handle complex calculations that are difficult to verify through mental arithmetic.

---

## When to Use This Skill

**Use these tools when you encounter:**

1. **Complex multi-step calculations** with multiple intermediate values
2. **Need for calculation verification** to ensure accuracy
3. **Nested formulas** that are hard to track mentally
4. **Data processing** that requires iteration or conditional logic
5. **Ambiguity in calculation approach** and need to test different methods

**DO NOT use for simple calculations** like:
- Basic percentages (e.g., 14/100 × 100 = 14%)
- Simple subtraction/addition
- Direct table lookups

---

## Available Tools

### 1. `generate_calculation_formula()`

Generates a clear, step-by-step mathematical formula for the calculation.

**Use when:**
- You need to clarify the calculation logic
- Multiple steps are involved
- You want to verify formula correctness

**Example input:**
```
question: "What is the percentage change from 2019 to 2020?"
old_value: 5735
new_value: 5829
```

**Example output:**
```
Formula: Percentage Change = ((New - Old) / Old) × 100
Step 1: Change = 5829 - 5735 = 94
Step 2: Percentage = (94 / 5735) × 100 = 1.64%
Result: 1.64%
```

### 2. `generate_python_code()`

Generates executable Python code for the calculation.

**Use when:**
- Need to handle multiple data points programmatically
- Want to verify calculation through execution
- Complex data extraction or processing needed

**Example input:**
```
question: "What is the total revenue across 2015, 2016, and 2017?"
data: {"2015": 4200, "2016": 4500, "2017": 4800}
```

**Example output:**
```python
# Calculate total revenue across multiple years
revenues = {
    '2015': 4200,
    '2016': 4500,
    '2017': 4800
}

total = sum(revenues.values())
print(f"Total revenue: {total}")
# Result: 13500
```

### 3. `execute_calculation()`

Executes a Python expression or code snippet and returns the result.

**Use when:**
- Need to perform calculation that's hard to do mentally
- Want to verify a complex formula
- Need precise floating-point arithmetic

**Example input:**
```
expression: "((5829 - 5735) / 5735) * 100"
```

**Example output:**
```
Result: 1.6390243902439024
Rounded: 1.64
```

---

## Usage Guidelines

### ✅ Good Use Cases

1. **Verification of complex calculations**
```
You calculated: 1.64%
Use: execute_calculation("((5829 - 5735) / 5735) * 100")
Verify: Confirms 1.64%
```

2. **Multi-step formulas**
```
Question: "What would fair value be without corporate stocks?"
Total: 51,052
Corporate stocks: 8,432
Use: generate_calculation_formula() to clarify: 51,052 - 8,432 = 42,620
```

3. **Summing multiple values**
```
Question: "Total expenses from Q1 to Q4?"
Use: generate_python_code() to sum quarterly values
```

### ❌ Poor Use Cases

1. **Simple arithmetic**
```
Question: "What is 100 + 200?"
Don't use tools - just calculate: 300
```

2. **Direct lookups**
```
Question: "What is revenue in 2020?"
Don't use tools - just read from table: $5,829M
```

3. **Already confident calculations**
```
You've verified the data and formula - don't use tools just for confirmation
```

---

## Best Practices

1. **Try manual calculation first** - Only use tools when genuinely needed

2. **Specify clear inputs** - Provide extracted values and question context

3. **Verify tool outputs** - Make sure generated formula/code makes sense

4. **Use for complex scenarios** - Save tools for when they add real value

5. **Combine with reasoning framework** - Use finqa-reasoning skill first, then tools if needed

---

## Example Workflow

**Question:** "What is the average percentage change across 3 years?"

**Step 1: Manual reasoning** (using finqa-reasoning)
```
- Year 1→2: ((520-500)/500) × 100 = 4%
- Year 2→3: ((550-520)/520) × 100 = 5.77%
- Year 3→4: ((540-550)/550) × 100 = -1.82%
```

**Step 2: Use tool for verification**
```
Call: generate_python_code()
Purpose: Verify average calculation
Output: (4 + 5.77 + (-1.82)) / 3 = 2.65%
```

**Step 3: Final answer**
```
2.65%
```

---

## Tool Parameters

### `generate_calculation_formula(question, data_points)`

- `question` (string): The original question
- `data_points` (dict): Extracted values, e.g., `{"old": 500, "new": 550}`
- Returns: Step-by-step formula breakdown

### `generate_python_code(question, data_points, operation_type)`

- `question` (string): The original question
- `data_points` (dict): Values to use in calculation
- `operation_type` (string, optional): "sum", "average", "percentage", "change", etc.
- Returns: Executable Python code as string

### `execute_calculation(expression)`

- `expression` (string): Python expression or code to execute
- Returns: Calculation result (number or string)

---

## Notes

- **Safety**: Code execution is sandboxed - only basic arithmetic and Python built-ins
- **Precision**: Uses Python's float arithmetic - results are precise to ~15 decimal places
- **Error handling**: If calculation fails, error message is returned with explanation

