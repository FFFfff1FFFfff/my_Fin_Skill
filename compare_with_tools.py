#!/usr/bin/env python3
"""
Compare baseline vs skills with ACTUAL tool execution
Uses Anthropic tool calling API to invoke skill functions
"""

import json
import os
from anthropic import Anthropic
from skill_system import SkillManager

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def ask_baseline(question, context, model="claude-sonnet-4-5-20250929"):
    """Baseline without skills"""
    prompt = f"""Answer this financial question using the data below. Give ONLY the number.

Examples:
- "What percentage are leased?" → "14%"
- "What is the change?" → "94"

Data:
{context}

Question: {question}

Answer (number only):"""

    message = client.messages.create(
        model=model,
        max_tokens=50,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text.strip()


def extract_final_answer(text):
    """
    Extract the final answer from text after 'Answer:' marker
    If no marker found, return the full text
    """
    import re
    
    # Look for "Answer:" followed by the answer
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: return last line or full text
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else text.strip()


def ask_with_tools(question, context, skill_manager, skill_names, model="claude-sonnet-4-5-20250929", max_turns=5):
    """
    Ask with skills using tool calling API
    Allows Claude to actually execute skill functions
    
    Returns:
        tuple: (full_response, final_answer)
        - full_response: Complete reasoning process with all tool calls
        - final_answer: Extracted answer after "Answer:" marker
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
Answer: [your final answer here]"""

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
            # Extract text response
            for block in response.content:
                if block.type == "text":
                    full_text = block.text.strip()
                    full_reasoning.append(full_text)
                    # Return: (full response with reasoning, extracted answer)
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
                        # Execute the function
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
            # Unexpected stop reason
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


def llm_judge(question, context, prediction, ground_truth, model="claude-sonnet-4-5-20250929"):
    """
    Pure LLM-based judge for answer evaluation
    Simple and direct to avoid over-guidance
    """
    
    judge_prompt = f"""Compare these two answers to the same question:

Question: {question}

Ground Truth: {ground_truth}
Predicted: {prediction}

Are they equivalent? Consider:
- Format differences OK (14% = 14 = 0.14 if percentage context)
- Minor rounding OK (±0.5 absolute or ±3% relative)
- Different values = WRONG

Answer ONLY "CORRECT" or "INCORRECT":"""

    message = client.messages.create(
        model=model,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    response = message.content[0].text.strip().upper()
    return "CORRECT" in response


def run_comparison_with_tools(dataset_path, start_idx=0, end_idx=10):
    """
    Run comparison with actual tool execution
    
    Args:
        dataset_path: Path to dataset JSON file
        start_idx: Start index (0-based, inclusive)
        end_idx: End index (0-based, exclusive)
    """

    # Load skill manager
    skill_manager = SkillManager()
    skill_names = ['finqa-reasoning', 'formula-code-assistant']

    print("=" * 80)
    print(f"Loaded Skills: {skill_manager.list_skills()}")
    print(f"Testing Skills: {skill_names}")

    # Show registered functions
    tools = skill_manager.get_tools_for_anthropic(skill_names)
    print(f"\nRegistered Tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    print("=" * 80)

    # Load dataset
    with open(dataset_path) as f:
        full_data = json.load(f)['data']
        data = full_data[start_idx:end_idx]
    
    print(f"\nTesting samples {start_idx+1} to {end_idx} (total: {len(data)} samples)")
    print("=" * 80)

    results_baseline = []
    results_skills = []

    correct_baseline = 0
    correct_skills = 0

    for i, sample in enumerate(data, start_idx+1):
        question = sample['question']
        context = sample['context']
        ground_truth = sample['answer']

        print(f"\n【Sample {i}/{end_idx}】")
        print(f"Q: {question[:70]}...")
        print(f"Ground Truth: '{ground_truth}'")

        # Test baseline
        try:
            pred_baseline = ask_baseline(question, context)
            is_correct_baseline = llm_judge(question, context, pred_baseline, ground_truth)

            print(f"Baseline: '{pred_baseline}' → {'✓' if is_correct_baseline else '✗'}")

            if is_correct_baseline:
                correct_baseline += 1

            results_baseline.append({
                "question": question,
                "prediction": pred_baseline,
                "ground_truth": ground_truth,
                "correct": is_correct_baseline
            })
        except Exception as e:
            print(f"Baseline Error: {e}")
            results_baseline.append({
                "question": question,
                "prediction": "",
                "ground_truth": ground_truth,
                "correct": False,
                "error": str(e)
            })

        # Test with skills + tools
        try:
            full_response, final_answer = ask_with_tools(question, context, skill_manager, skill_names)
            is_correct_skills = llm_judge(question, context, final_answer, ground_truth)

            # Print full reasoning process
            print(f"With Tools (Reasoning):")
            print(f"  {full_response[:200]}..." if len(full_response) > 200 else f"  {full_response}")
            print(f"With Tools (Answer): '{final_answer}' → {'✓' if is_correct_skills else '✗'}")

            if is_correct_skills:
                correct_skills += 1

            results_skills.append({
                "question": question,
                "full_response": full_response,  # Save full reasoning
                "prediction": final_answer,       # Save extracted answer
                "ground_truth": ground_truth,
                "correct": is_correct_skills
            })
        except Exception as e:
            print(f"Skills Error: {e}")
            results_skills.append({
                "question": question,
                "full_response": "",
                "prediction": "",
                "ground_truth": ground_truth,
                "correct": False,
                "error": str(e)
            })

        print("-" * 80)

    # Summary
    acc_baseline = correct_baseline / len(data) if data else 0
    acc_skills = correct_skills / len(data) if data else 0
    improvement = acc_skills - acc_baseline

    print(f"\n{'=' * 80}")
    print(f"RESULTS:")
    print(f"  Baseline:    {correct_baseline}/{len(data)} ({acc_baseline:.1%})")
    print(f"  With Tools:  {correct_skills}/{len(data)} ({acc_skills:.1%})")
    print(f"  Improvement: {improvement:+.1%} ({'+' if improvement > 0 else ''}{correct_skills - correct_baseline})")
    print(f"{'=' * 80}")

    # Save results to JSON
    output = {
        "baseline": {
            "accuracy": acc_baseline,
            "correct": correct_baseline,
            "total": len(data),
            "results": results_baseline
        },
        "skills": {
            "accuracy": acc_skills,
            "correct": correct_skills,
            "total": len(data),
            "results": results_skills
        },
        "improvement": {
            "absolute": improvement,
            "delta_correct": correct_skills - correct_baseline
        }
    }

    # Generate filename suffix based on range
    range_suffix = f"{start_idx+1}_{end_idx}"
    
    json_filename = f"results_with_tools_{range_suffix}.json"
    baseline_txt = f"results_baseline_{range_suffix}.txt"
    skills_txt = f"results_with_skills_{range_suffix}.txt"
    
    with open(json_filename, "w", encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_filename}")

    # Save results to TXT files
    # Baseline results
    with open(baseline_txt, "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"BASELINE RESULTS (Samples {start_idx+1}-{end_idx})\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, result in enumerate(results_baseline, start_idx+1):
            f.write(f"【Sample {idx}】\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Correct: {'✓' if result['correct'] else '✗'}\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            f.write("-" * 80 + "\n\n")
        
        f.write(f"\nSummary:\n")
        f.write(f"Accuracy: {correct_baseline}/{len(data)} ({acc_baseline:.1%})\n")

    # Skills results
    with open(skills_txt, "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RESULTS WITH SKILLS (Samples {start_idx+1}-{end_idx})\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, result in enumerate(results_skills, start_idx+1):
            f.write(f"【Sample {idx}】\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"\n--- Full Reasoning Process ---\n")
            f.write(f"{result.get('full_response', 'N/A')}\n")
            f.write(f"\n--- Extracted Answer ---\n")
            f.write(f"Answer: {result['prediction']}\n")
            f.write(f"Correct: {'✓' if result['correct'] else '✗'}\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            f.write("=" * 80 + "\n\n")
        
        f.write(f"\nSummary:\n")
        f.write(f"Accuracy: {correct_skills}/{len(data)} ({acc_skills:.1%})\n")
        f.write(f"Improvement over baseline: {improvement:+.1%}\n")

    print(f"Text results saved to {baseline_txt} and {skills_txt}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare baseline vs skills with tool execution')
    parser.add_argument('--dataset', type=str, default='finqa_test.json',
                        help='Path to dataset')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index (0-based, default: 0)')
    parser.add_argument('--end', type=int, default=10,
                        help='End index (0-based, exclusive, default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Number of samples to test (alternative to --end)')

    args = parser.parse_args()

    # If --limit is provided, use it to calculate end_idx
    if args.limit is not None:
        end_idx = args.start + args.limit
    else:
        end_idx = args.end

    run_comparison_with_tools(args.dataset, args.start, end_idx)
