name	finqa-reasoning
description	Step-by-step reasoning framework for financial question answering with systematic data extraction and calculation strategies

# FinQA Reasoning Framework

A systematic approach to answering financial questions through structured thinking and careful data handling.

---

## Core Philosophy

**Most errors come from wrong data extraction, not calculation mistakes.**

Focus on:
1. 📖 Understanding what the question asks
2. 🎯 Locating the correct data points
3. ✓ Verifying data before calculating
4. 🧮 Applying the right formula
5. 📝 Formatting the answer correctly

---

## Step-by-Step Reasoning Template

For EVERY question, follow this structured approach:

### STEP 1: Understand the Question Type

Identify what's being asked:

- **Direct Lookup**: "What is the revenue in 2020?"
- **Percentage**: "What percentage of X is Y?"
- **Change**: "What is the change from 2019 to 2020?"
- **Percentage Change**: "What is the percentage change from 2019 to 2020?"
- **Comparison**: "Did X increase?" / "Is A greater than B?"
- **Multi-step**: "What would X be without Y?"

### STEP 2: Locate the Data

**Be systematic:**

```
1. Read the question keywords carefully
2. Scan for table structure (rows/columns)
3. Find EXACT matches for keywords
4. Note the units (millions, thousands, %)
5. Check the time period (year, quarter)
```

**Common mistakes to avoid:**
- ❌ Grabbing data from wrong year
- ❌ Using wrong row (e.g., "Net income" vs "Operating income")
- ❌ Missing unit conversions (millions vs thousands)

### STEP 3: Extract and Verify

**Write down explicitly:**

```
Data Point 1: [Label] = [Value] [Unit] (from [Location])
Data Point 2: [Label] = [Value] [Unit] (from [Location])

Verification:
- Years/periods correct? ✓/✗
- Units consistent? ✓/✗
- Numbers reasonable? ✓/✗
```

**Example:**
```
Question: "What is the percentage change from 2019 to 2020?"

Data Point 1: Revenue 2019 = $5,735 million (from "Revenue" row, "2019" column)
Data Point 2: Revenue 2020 = $5,829 million (from "Revenue" row, "2020" column)

Verification:
- 2019 comes before 2020? ✓
- Both in millions? ✓
- Values in reasonable range? ✓
```

### STEP 4: Choose the Right Formula

Based on question type:

| Question Type | Formula | Example |
|--------------|---------|---------|
| **Percentage** | `(Part / Total) × 100` | "14% of facilities are leased" → (140 / 1000) × 100 = 14% |
| **Change** | `New - Old` | "Change from 5735 to 5829" → 5829 - 5735 = 94 |
| **Percentage Change** | `((New - Old) / Old) × 100` | "% change 500→550" → ((550-500)/500)×100 = 10% |
| **Ratio** | `Value A / Value B` | "Ratio of debt to equity" → 1000 / 500 = 2.0 |
| **Subtraction** | `Total - Exclusion` | "X without Y" → Total_X - Y_component |

### STEP 5: Calculate

**Show your work:**

```
Formula: [write it explicitly]
Substitution: [plug in numbers]
Calculation: [show intermediate steps if complex]
Result: [final number]
```

### STEP 6: Format the Answer

**Match the expected format:**

- Question mentions "percentage" → Add % symbol: `"14%"`
- Question asks for "change in dollars" → Plain number: `"94"`
- Question asks for "ratio" → Decimal: `"2.0"`
- Yes/no question → `"yes"` or `"no"`

**Pro tip:** Look at how the question is phrased to determine format.

---

## Common Question Patterns & Pitfalls

### Pattern 1: Percentage vs Percentage Change

❌ **Wrong:**
- Question: "What percentage changed?"
- Mistake: Calculate `(New / Old) × 100`

✓ **Correct:**
- Use: `((New - Old) / Old) × 100`

### Pattern 2: "Without" / "Excluding"

Questions like: *"What would the fair value be without corporate stocks?"*

**Approach:**
1. Find total value (X)
2. Find excluded component (Y)
3. Calculate: X - Y
4. ⚠️ Make sure to find the excluded component in the data!

### Pattern 3: Multi-period Aggregation

Questions like: *"What is the total during 2015, 2016, and 2017?"*

**Approach:**
1. Extract value for each period separately
2. Verify all periods found
3. Sum them: Value₂₀₁₅ + Value₂₀₁₆ + Value₂₀₁₇

### Pattern 4: Wrong Row Selection

⚠️ Tables often have similar-sounding rows:
- "Revenue" vs "Net Revenue"
- "Operating Income" vs "Net Income"
- "Assets" vs "Current Assets"

**Solution:** Match the EXACT terminology from the question.

---

## Self-Check Before Finalizing

Before submitting your answer, verify:

- [ ] Did I use data from the correct time period?
- [ ] Are all units consistent across data points?
- [ ] Did I apply the correct formula for this question type?
- [ ] Does my answer format match the question's expectation?
- [ ] Is the magnitude reasonable? (not 1000% or 0.0001%)
- [ ] Did I double-check row/column labels in the table?

---

## When to Ask for Help

If you encounter:

1. **Complex nested calculations** → Consider requesting formula generation assistance
2. **Ambiguous table structure** → May need data extraction tools
3. **Need to verify calculation** → Can generate verification code
4. **Multiple conditional steps** → Might benefit from programmatic approach

But try to solve it yourself first using the framework above!

---

## Examples

### Example 1: Simple Percentage

**Question:** "What percentage of total facilities are leased?"

**Process:**
```
STEP 1: Question Type = Percentage calculation
STEP 2: Look for "leased" and "total facilities" in table
STEP 3: 
  - Leased facilities = 140 (from "Leased" row)
  - Total facilities = 1,000 (from "Total" row)
  - Verification: Both in same units ✓
STEP 4: Formula = (Part / Total) × 100
STEP 5: Calculate = (140 / 1000) × 100 = 14.0%
STEP 6: Format = "14%"
```

**Answer:** 14%

### Example 2: Absolute Change

**Question:** "What is the change in revenue from 2019 to 2020?"

**Process:**
```
STEP 1: Question Type = Absolute change
STEP 2: Look for "revenue" row, "2019" and "2020" columns
STEP 3:
  - Revenue 2019 = 5,735 million
  - Revenue 2020 = 5,829 million
  - Verification: Same units (millions) ✓
STEP 4: Formula = New - Old
STEP 5: Calculate = 5,829 - 5,735 = 94
STEP 6: Format = "94" (million is implied from context)
```

**Answer:** 94

### Example 3: Percentage Change

**Question:** "What is the percentage change in net income from 2018 to 2019?"

**Process:**
```
STEP 1: Question Type = Percentage change (note: "percentage" + "change")
STEP 2: Look for "net income" row
STEP 3:
  - Net Income 2018 = 500 million
  - Net Income 2019 = 550 million
  - Verification: 2018 < 2019 ✓, same units ✓
STEP 4: Formula = ((New - Old) / Old) × 100
STEP 5: Calculate = ((550 - 500) / 500) × 100 = 10.0%
STEP 6: Format = "10%" or "10.0%"
```

**Answer:** 10%

---

## Key Insights

🎯 **Accuracy > Speed**: Take time to verify data before calculating

🎯 **Context Matters**: Always check units, time periods, and row labels

🎯 **Formula Selection**: "Percentage change" ≠ "Percentage" ≠ "Change"

🎯 **Format Appropriately**: Match your answer format to the question

🎯 **Trust Your Process**: Follow the steps systematically, don't skip verification

