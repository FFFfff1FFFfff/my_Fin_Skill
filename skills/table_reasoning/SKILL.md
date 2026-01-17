name	table_reasoning
description	Structured reasoning framework for table question answering with systematic data extraction and calculation strategies

# Table Reasoning Framework

A systematic approach to answering questions about tabular data.

---

## Task Types

| Type | Description | Strategy |
|------|-------------|----------|
| **Fact Checking** | Verify statements against table data | Locate → Verify → yes/no |
| **Numerical Reasoning** | Calculate values (sum, avg, count, etc.) | Extract → Formula → Calculate |
| **Data Analysis** | Analyze trends, correlations, impacts | Multi-step reasoning → Interpret |

---

## Step-by-Step Process

### STEP 1: Parse the Table

Before answering, understand the table structure:

```
1. Identify columns and their data types (text, number, date)
2. Identify row structure (header row? hierarchical?)
3. Note units (millions, %, currency)
```

### STEP 2: Understand the Question

Classify the question:

- **Lookup**: "What is X in row Y?" → Direct extraction
- **Aggregation**: "Total/Sum/Average of X?" → SUM/AVG/COUNT
- **Filtering**: "X where condition?" → Filter then calculate
- **Comparison**: "Is A > B?" or "Which is largest?" → Compare values
- **Multi-hop**: Requires multiple lookups/calculations

### STEP 3: Extract Data

**Be precise:**

```
Data Point 1: [Column] = [Value] (Row: [row identifier])
Data Point 2: [Column] = [Value] (Row: [row identifier])
```

**Common errors to avoid:**
- Wrong row (similar names exist)
- Wrong column (check header carefully)
- Missing unit conversion

### STEP 4: Calculate (if needed)

| Operation | Formula |
|-----------|---------|
| Sum | `value1 + value2 + ...` |
| Average | `sum / count` |
| Percentage | `(part / total) * 100` |
| Change | `new - old` |
| % Change | `((new - old) / old) * 100` |
| Count | Count rows matching condition |

### STEP 5: Format Answer

Match the expected format:
- Numeric question → number (e.g., `42`, `3.14`)
- Yes/No question → `yes` or `no`
- Percentage → include % if question asks for percentage

---

## Common Patterns

### Pattern: Aggregation with Filter

Q: "What is the total sales for product X?"

```
1. Filter: rows where Product = X
2. Aggregate: SUM(Sales) for filtered rows
3. Return: number
```

### Pattern: Multi-hop Reasoning

Q: "What is the revenue of the company with the highest profit?"

```
1. Find: row with MAX(Profit)
2. Extract: Revenue from that row
3. Return: revenue value
```

### Pattern: Comparison

Q: "Did sales increase from 2020 to 2021?"

```
1. Extract: Sales_2020, Sales_2021
2. Compare: Sales_2021 > Sales_2020
3. Return: yes/no
```

---

## Checklist Before Answering

- [ ] Correct row(s) identified?
- [ ] Correct column(s) used?
- [ ] Units consistent?
- [ ] Calculation formula correct?
- [ ] Answer format matches question?

---

## Output Format

Always end your response with:

```
Answer: [your final answer]
```

The answer should be concise: a number, yes/no, or short phrase.
