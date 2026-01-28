"""
ChartQA Chain-of-Thought (CoT) Prompt Templates

Based on ChartQAPro paper Table 7 - CoT setup prompts.
Paper finding: CoT significantly outperforms direct answering for closed-source models.
"""

# Chain of Thought prompts (Table 7 from ChartQAPro paper)
COT_PROMPTS = {
    "Factoid": """You are given a factoid question that you need to answer based on the provided image.
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

Question: {question}""",

    "Multi Choice": """You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.
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

Question: {question}""",

    "Hypothetical": """You are given a hypothetical question that you need to answer based on the provided image.
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

Question: {question}""",

    "Fact Checking": """You are given a fact statement that you need to assess based on the information in the provided image.
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

Question: {question}""",

    "Conversational": """You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.
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
Question: {question}""",
}


def get_cot_prompt(question_type: str, question: str, conversation: str = "") -> str:
    """
    Get the formatted CoT prompt for a given question type.

    Args:
        question_type: One of "Factoid", "Multi Choice", "Hypothetical", "Fact Checking", "Conversational"
        question: The question text
        conversation: Previous conversation history (for Conversational type)

    Returns:
        Formatted prompt string
    """
    template_key = question_type if question_type in COT_PROMPTS else "Factoid"
    template = COT_PROMPTS[template_key]

    if question_type == "Conversational":
        return template.format(conversation=conversation, question=question)
    else:
        return template.format(question=question)


# Key stages to track (simplified)
KEY_STAGES = ["read", "calculate", "answer"]
