name	chartqa_cot
description	Chain-of-Thought (CoT) reasoning framework for chart question answering with structured stage markers

# ChartQA CoT - Chain of Thought Reasoning

Structured reasoning framework for answering questions about charts and visualizations.

Based on ChartQAPro paper findings: CoT significantly outperforms direct answering for closed-source models.

---

## Overview

This skill provides Chain-of-Thought prompts for 5 question types:
- Factoid
- Multi Choice
- Hypothetical
- Fact Checking
- Conversational

Each prompt uses structured stage markers for monitoring and debugging.

---

## Stage Markers

All CoT responses should follow these reasoning stages:

1. **[UNDERSTAND]**: What is the question asking?
2. **[LOCATE]**: Where in the chart is the relevant data?
3. **[READ]**: What are the exact values from the chart?
4. **[CALCULATE/EVALUATE/COMPARE]**: Any calculations or comparisons needed?
5. **[VERIFY]**: Does the answer make sense?

Final answer format: `The answer is X`

---

## CoT Prompt Templates

### Factoid Questions

```
You are given a factoid question that you need to answer based on the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What is the question asking?
2. [LOCATE]: Where in the chart is the relevant data?
3. [READ]: What are the exact values from the chart?
4. [CALCULATE]: Any calculations needed? (show work)
5. [VERIFY]: Does the answer make sense?

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X should be ONLY the answer (no units unless in chart)
- For years: exact format from chart
- For numbers: no extra units

Question: {question}
```

### Multi Choice Questions

```
You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.
You need to think step-by-step, but your final answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What is the question asking?
2. [LOCATE]: Where in the chart is the relevant data?
3. [READ]: What are the exact values?
4. [EVALUATE]: Check each option against the data
5. [VERIFY]: Confirm the selected option

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X must be ONLY a single lowercase letter: a, b, c, or d
- Example: "The answer is b"

Question: {question}
```

### Hypothetical Questions

```
You are given a hypothetical question that you need to answer based on the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What hypothetical scenario is being asked?
2. [LOCATE]: Where is the relevant baseline data?
3. [READ]: What are the current values?
4. [CALCULATE]: Apply the hypothetical change
5. [VERIFY]: Is the result reasonable?

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X should be ONLY the answer

Question: {question}
```

### Fact Checking Questions

```
You are given a fact statement that you need to assess based on the information in the provided image.
You need to think step-by-step, but your final answer should be either true or false (without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [UNDERSTAND]: What claim is being made?
2. [LOCATE]: Where is the relevant data in the chart?
3. [READ]: What are the actual values?
4. [COMPARE]: Does the data support or refute the claim?
5. [VERIFY]: Double-check the conclusion

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X must be ONLY: true OR false (lowercase)
- Example: "The answer is true"

Question: {question}
```

### Conversational Questions

```
You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].

STRICT REASONING FORMAT:
Think step-by-step using these stages:
1. [CONTEXT]: What was discussed in previous turns?
2. [UNDERSTAND]: What is the current question asking?
3. [LOCATE]: Where is the relevant data?
4. [READ]: What are the exact values?
5. [VERIFY]: Does this follow logically from the conversation?

STRICT ANSWER FORMAT:
- End with exactly: "The answer is X"
- X should be ONLY the answer

{conversation}
Question: {question}
```

---

## Answer Format Rules

### General Rules
- End with exactly: `The answer is X`
- X should be ONLY the answer value
- No explanations in the final answer
- No units unless shown in chart

### Type-Specific Rules

| Type | Answer Format | Example |
|------|---------------|---------|
| Factoid | Number, word, or phrase | `The answer is 2020` |
| Multi Choice | Single lowercase letter | `The answer is b` |
| Hypothetical | Number, word, or phrase | `The answer is 150` |
| Fact Checking | true or false (lowercase) | `The answer is true` |
| Conversational | Number, word, or phrase | `The answer is 45%` |

---

## Common Pitfalls

1. **Adding units**: Don't add units unless they appear in the chart
2. **Wrong format for multi-choice**: Must be single lowercase letter (a/b/c/d)
3. **Wrong case for fact checking**: Must be lowercase (true/false)
4. **Extra text in answer**: Final answer line should contain ONLY the value
5. **Missing stage markers**: Use [UNDERSTAND], [LOCATE], etc. for better tracking
