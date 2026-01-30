name	chartqa_cot
description	Chain-of-Thought reasoning framework for chart question answering

# ChartQA CoT - Chain of Thought Reasoning

Structured reasoning framework for answering questions about charts and visualizations.

Based on ChartQAPro paper findings: CoT significantly outperforms direct answering for closed-source models.

---

## Question Types

- Factoid
- Multi Choice
- Hypothetical
- Fact Checking
- Conversational

Prompt templates for each type are defined in `cot_prompts.py`.

---

## Stage Markers

All CoT responses follow these reasoning stages:

1. **[UNDERSTAND]**: What is the question asking?
2. **[LOCATE]**: Where in the chart is the relevant data?
3. **[READ]**: What are the exact values from the chart?
4. **[CALCULATE/EVALUATE/COMPARE]**: Any calculations or comparisons needed?
5. **[VERIFY]**: Does the answer make sense?

Final answer format: `The answer is X`

---

## Answer Format Rules

| Type | Format | Example |
|------|--------|---------|
| Factoid | Number, word, or phrase | `The answer is 2020` |
| Multi Choice | Single lowercase letter | `The answer is b` |
| Hypothetical | Number, word, or phrase | `The answer is 150` |
| Fact Checking | true or false (lowercase) | `The answer is true` |
| Conversational | Number, word, or phrase | `The answer is 45%` |

---

## Common Pitfalls

1. Adding units when not in chart
2. Wrong format for multi-choice (must be single lowercase letter)
3. Wrong case for fact checking (must be lowercase true/false)
4. Extra text in final answer line
