#!/usr/bin/env python3
"""
FinQA benchmark runner: compare baseline vs with-skill performance
Uses official FinQA evaluation (exact match with 5 decimal precision)
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_finqa, load_sample_data
from evaluator import finqa_equal, evaluate_finqa
from skill_system import SkillManager

client = Anthropic()


def ask_baseline(question: str, context: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    """Baseline: direct question without skills. Outputs number only."""
    prompt = f"""Answer this financial question.

Data:
{context}

Question: {question}

Reply with ONLY the final number (no text, no units, no explanation).
- For percentages: round to whole number or 1 decimal (e.g., "14" or "9.9")
- For yes/no: reply "yes" or "no"
- No symbols like % or $

Answer:"""

    response = client.messages.create(
        model=model,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def extract_final_answer(text: str) -> str:
    """Extract the final answer from text after 'Answer:' marker."""
    import re

    # Look for "Answer:" followed by the answer
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: return last line or full text
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else text.strip()


def ask_with_skill(question: str, context: str, skill_manager: SkillManager,
                   skill_names: list[str], model: str = "claude-sonnet-4-5-20250929",
                   max_turns: int = 5) -> tuple[str, str]:
    """
    Answer with skill enhancement and optional tool execution.

    Returns:
        tuple: (full_response, final_answer)
    """
    # Get tool definitions
    tools = skill_manager.get_tools_for_anthropic(skill_names)

    # Build system prompt with skill documentation
    skill_context = skill_manager.build_system_prompt(skill_names)

    prompt = f"""{skill_context}

---

**IMPORTANT**: You have access to calculation tools, but use them ONLY when necessary.

**Reasoning Strategy**:
1. First, try to solve the question using the step-by-step reasoning framework above
2. Extract data carefully and apply the appropriate formula
3. Only use tools if you encounter:
   - Complex multi-step calculations that are hard to verify
   - Need for programmatic data processing
   - Ambiguity that code/formula generation would clarify

**Most questions can be solved through careful reasoning without tools.**

---

Data:
{context}

Question: {question}

You can show your reasoning process, but you MUST end with:
Answer: [your final answer here]

**Answer Format Requirements** (CRITICAL for correct evaluation):
- Output ONLY the numeric value, no units or symbols
- For percentages: round to whole number or 1 decimal place (e.g., "14" or "9.9", NOT "14.46429")
- For currency: output the number only (e.g., "1234" not "$1,234")
- No commas in numbers
- For yes/no questions: reply "yes" or "no"
- For negative numbers: use minus sign (e.g., "-5")

**IMPORTANT**: Match the precision level typical in financial reporting. Don't over-precise."""

    messages = [{"role": "user", "content": prompt}]

    # Collect full reasoning process for display
    full_reasoning = []

    for turn in range(max_turns):
        # Call Claude with tools
        if tools:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                tools=tools,
                messages=messages
            )
        else:
            response = client.messages.create(
                model=model,
                max_tokens=50,
                temperature=0,
                messages=messages
            )

        # Check if we got a final answer
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    full_text = block.text.strip()
                    full_reasoning.append(full_text)
                    final_answer = extract_final_answer(full_text)
                    complete_response = "\n".join(full_reasoning)
                    return complete_response, final_answer

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Collect reasoning text before tool calls
            for block in response.content:
                if block.type == "text":
                    full_reasoning.append(block.text.strip())

            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    print(f"  [Tool Call] {tool_name}({tool_input})")

                    try:
                        result = skill_manager.call_function(tool_name, **tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })
                        print(f"  [Tool Result] {result}")
                        full_reasoning.append(f"[Tool: {tool_name}] → {result}")
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })
                        print(f"  [Tool Error] {e}")
                        full_reasoning.append(f"[Tool Error: {tool_name}] {e}")

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    # If we exhausted turns, extract any text response
    final_text = ""
    for block in response.content:
        if block.type == "text":
            final_text = block.text.strip()
            full_reasoning.append(final_text)

    complete_response = "\n".join(full_reasoning)
    final_answer = extract_final_answer(final_text) if final_text else ""
    return complete_response, final_answer


def run_benchmark(source: str = "sample", limit: int = 50,
                  model: str = "claude-sonnet-4-5-20250929"):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.

    Args:
        source: "sample" or path to JSON file
        limit: Number of samples to test
        model: Model for answering questions
    """
    print("=" * 70)
    print("FinQA Skill Benchmark")
    print("=" * 70)

    # Load data
    print(f"\nLoading data (source={source}, limit={limit})...")
    if source == "sample":
        samples = load_sample_data()
    else:
        samples = load_finqa(source=source, limit=limit)
    print(f"Loaded {len(samples)} samples")

    # Load skills
    skill_manager = SkillManager()
    skill_names = ['finqa_reasoning', 'formula_code_assistant']
    skill_prompt = skill_manager.build_system_prompt(skill_names)
    print(f"Loaded skills: {skill_names}")

    # Show registered tools
    tools = skill_manager.get_tools_for_anthropic(skill_names)
    print(f"Registered tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool['name']}")

    results_baseline = []
    results_skill = []

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        qid = sample["id"]
        question = sample["question"]
        context = sample["context"]
        gold_answer = sample["answer"]

        print(f"\n[{i+1}/{len(samples)}] {question[:60]}...")
        print(f"Gold: {gold_answer}")

        # Baseline (no skills)
        try:
            pred_baseline = ask_baseline(question, context, model)
            is_correct = finqa_equal(pred_baseline, gold_answer)
            status = "✓" if is_correct else "✗"
            print(f"Baseline: {pred_baseline} -> {status}")
        except Exception as e:
            pred_baseline = ""
            is_correct = False
            print(f"Baseline: ERROR - {e}")

        results_baseline.append({
            "id": qid,
            "question": question,
            "prediction": pred_baseline,
            "ground_truth": gold_answer,
            "correct": is_correct
        })

        # With skill
        try:
            full_response, pred_skill = ask_with_skill(
                question, context, skill_manager, skill_names, model
            )
            is_correct = finqa_equal(pred_skill, gold_answer)
            status = "✓" if is_correct else "✗"
            print(f"Skill:    {pred_skill} -> {status}")
        except Exception as e:
            pred_skill = ""
            full_response = ""
            is_correct = False
            print(f"Skill:    ERROR - {e}")

        results_skill.append({
            "id": qid,
            "question": question,
            "full_response": full_response,
            "prediction": pred_skill,
            "ground_truth": gold_answer,
            "correct": is_correct
        })

    # Calculate metrics
    def calc_metrics(results):
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total
        }

    metrics_baseline = calc_metrics(results_baseline)
    metrics_skill = calc_metrics(results_skill)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline (no skills):")
    print(f"  Accuracy: {metrics_baseline['correct']}/{metrics_baseline['total']} ({metrics_baseline['accuracy']:.1%})")

    print(f"\nWith Skill:")
    print(f"  Accuracy: {metrics_skill['correct']}/{metrics_skill['total']} ({metrics_skill['accuracy']:.1%})")

    improvement = metrics_skill['accuracy'] - metrics_baseline['accuracy']
    print(f"\nImprovement: {improvement:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "source": source,
        "num_samples": len(samples),
        "baseline": {"metrics": metrics_baseline, "results": results_baseline},
        "skill": {"metrics": metrics_skill, "results": results_skill},
        "improvement": improvement
    }

    output_file = f"finqa_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinQA Skill Benchmark")
    parser.add_argument("--source", type=str, default="sample",
                        help="Data source: 'sample' or path to JSON file")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")

    args = parser.parse_args()
    run_benchmark(
        source=args.source,
        limit=args.limit,
        model=args.model
    )
