name	tablebench_pot
description	Program-of-Thought (POT) and visualization code execution for TableBench question answering

# TableBench POT - Program of Thought

Code generation and execution framework for table-based question answering and visualization tasks.

---

## Overview

This skill provides two approaches for TableBench:

1. **TCoT (Textual Chain-of-Thought)**: Step-by-step reasoning without code execution
2. **POT (Program of Thought)**: Generate Python code, execute it, and extract answer from output

---

## TCoT Reasoning Framework

For non-visualization questions, use structured step-by-step reasoning:

### Step 1: Parse the Table
- Identify column names and their meanings
- Note data types (numbers, text, dates)
- Check for any special formatting (percentages, currencies)

### Step 2: Understand the Question
- What is being asked? (sum, average, count, filter, comparison)
- Which columns are relevant?
- What conditions/filters apply?

### Step 3: Extract Data
- Locate the specific rows and columns needed
- Apply any filter conditions carefully
- Double-check column names match exactly

### Step 4: Calculate
- Perform the required calculation
- Keep full precision (don't round)
- Verify the calculation is correct

### Step 5: Format Answer
- Present answer in required format
- Numbers: keep original precision
- Text: use exact values from table

---

## POT Code Execution

For complex calculations or when code verification is needed:

### Code Format Requirements

```python
import pandas as pd
import json

# Parse the table
data = json.loads('''TABLE_JSON_HERE''')
df = pd.DataFrame(data)

# Your calculation logic here
result = ...

# MUST print final answer in this format
print(f"Final Answer: {result}")
```

### Execution Environment

Available modules:
- `pandas` (as `pd`)
- `json`
- Python built-ins

Constraints:
- Timeout: 15 seconds
- No file I/O
- No network access

---

## Visualization (POT-VIZ)

For visualization tasks, generate matplotlib code:

### Code Format Requirements

```python
import matplotlib.pyplot as plt
import pandas as pd
import json

# Parse the table
data = json.loads('''TABLE_JSON_HERE''')
df = pd.DataFrame(data)

# Create visualization
plt.figure(figsize=(10, 6))
# Your chart code here (bar, line, pie, etc.)

# Do NOT call plt.show()
```

### Chart Data Extraction

The system extracts y-data from generated charts for comparison:
- Line plots: y-values from each line
- Bar charts: heights of each bar
- Pie charts: proportions (angle/360)

---

## Answer Format

The answer MUST follow this exact format as the last line:

```
Final Answer: AnswerName1, AnswerName2...
```

### Format Rules

- **Numbers**: Keep original precision, no rounding
- **Entity names**: Use exact values from table
- **Multiple answers**: Comma-separated
- **Impact questions**: Use exactly "Positive impact", "Negative impact", or "No clear impact"
- **Factor questions**: List column names only, comma-separated

### Examples

**CORRECT:**
```
Final Answer: 450
Final Answer: Positive impact
Final Answer: lost, points for, points against
Final Answer: candidates
```

**WRONG:**
```
Final Answer: The answer is 450  (extra text)
Final Answer: There is a positive correlation...  (explanation)
Final Answer: candidates, because they...  (reasoning in answer)
```

---

## Question Type Guidelines

### NumericalReasoning (NR)
- Extract exact numbers from table
- Perform arithmetic operations
- Keep full decimal precision

### FactChecking (FC)
- Verify claims against table data
- Answer with exact values that confirm/deny

### DataAnalysis (DA)
- For "impact" questions: determine direction (positive/negative/no clear)
- For "which factors" questions: list relevant column names only
- NO explanations in final answer

### Visualization (VIZ)
- Generate matplotlib code
- Parse table data from JSON
- Create the requested chart type
- Do NOT call plt.show()

---

## Common Pitfalls

1. **Wrong column**: Double-check column names match exactly
2. **Missing filter**: Ensure all conditions in the question are applied
3. **Rounding errors**: Keep full decimal precision
4. **String vs Number**: "100" and 100 may look same but behave differently
5. **Explanation in answer**: Final Answer line must contain ONLY the answer value
