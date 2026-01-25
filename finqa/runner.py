#!/usr/bin/env python3
"""
FinQA benchmark runner: compare baseline vs with-skill performance
Uses official FinQA evaluation (exact match with 5 decimal precision)

Includes Stage Monitor for tracking reasoning stages:
- UNDERSTAND: Question type identification
- LOCATE: Data location in table/text
- EXTRACT: Data extraction and verification
- FORMULA: Formula selection
- CALCULATE: Calculation execution
- FORMAT: Answer formatting
"""

import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_finqa, load_sample_data
from evaluator import finqa_equal, evaluate_finqa
from skill_system import SkillManager

client = Anthropic()


# =============================================================================
# STAGE MONITOR: Track reasoning stages in FinQA skill flow
# =============================================================================

def parse_finqa_stages(response_text: str) -> dict:
    """
    Parse FinQA response to extract structured reasoning stages.

    Expected stages based on finqa_reasoning skill:
    - UNDERSTAND: Question type identification (STEP 1)
    - LOCATE: Data location (STEP 2)
    - EXTRACT: Data extraction and verification (STEP 3)
    - FORMULA: Formula selection (STEP 4)
    - CALCULATE: Calculation (STEP 5)
    - FORMAT: Answer formatting (STEP 6)
    """
    stages = {
        "understand": None,    # Question type identification
        "locate": None,        # Data location
        "extract": None,       # Data extraction
        "formula": None,       # Formula selection
        "calculate": None,     # Calculation
        "format": None,        # Answer formatting
        "raw": response_text,
    }

    # Pattern for STEP markers
    step_patterns = {
        "understand": [
            r'(?:STEP\s*1|Step\s*1)[:\s]*(.+?)(?=STEP\s*2|Step\s*2|$)',
            r'(?:Question Type|Type of question)[:\s]*(.+?)(?=\n\n|STEP|$)',
            r'(?:being asked|question asks|identify)[:\s]*(.+?)(?=\n\n|$)',
        ],
        "locate": [
            r'(?:STEP\s*2|Step\s*2)[:\s]*(.+?)(?=STEP\s*3|Step\s*3|$)',
            r'(?:Locate|Find|Look for)[:\s]*(.+?)(?=\n\n|STEP|$)',
            r'(?:from the table|in the table|from table)[:\s]*(.+?)(?=\n\n|$)',
        ],
        "extract": [
            r'(?:STEP\s*3|Step\s*3)[:\s]*(.+?)(?=STEP\s*4|Step\s*4|$)',
            r'(?:Data Point|Extract)[:\s]*(.+?)(?=\n\n|STEP|$)',
            r'(?:Value|=)\s*[\$]?[\d,]+(?:\.\d+)?',
        ],
        "formula": [
            r'(?:STEP\s*4|Step\s*4)[:\s]*(.+?)(?=STEP\s*5|Step\s*5|$)',
            r'(?:Formula|Using formula)[:\s]*(.+?)(?=\n\n|STEP|$)',
            r'(?:percentage change|change|ratio|sum)[:\s]*(.+?)(?=\n|$)',
        ],
        "calculate": [
            r'(?:STEP\s*5|Step\s*5)[:\s]*(.+?)(?=STEP\s*6|Step\s*6|$)',
            r'(?:Calculate|Calculation|Computing)[:\s]*(.+?)(?=\n\n|STEP|$)',
            r'(?:=\s*[\d\.\-\+\*/\(\)]+\s*=\s*[\d\.]+)',
        ],
        "format": [
            r'(?:STEP\s*6|Step\s*6)[:\s]*(.+?)(?=Answer|$)',
            r'(?:Format|Final Answer)[:\s]*(.+?)(?=\n|$)',
        ],
    }

    for stage, patterns in step_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if content and len(content) > 3:
                    stages[stage] = content[:200]
                    break

    # Also detect tool usage
    if "[Tool:" in response_text or "[Tool Call]" in response_text:
        stages["tool_used"] = True

    return stages


def analyze_stage_metrics(stages: dict) -> dict:
    """
    Analyze stage completion metrics for a sample.

    Returns:
        dict with metrics:
        - stages_completed: list of completed stages
        - stages_in_order: bool (whether stages followed expected order)
        - total_stages: int
        - stage_completion_rate: float
    """
    expected_order = ["understand", "locate", "extract", "formula", "calculate", "format"]
    first_appearance = {}

    # Since FinQA has single response, we check stage presence
    for i, stage in enumerate(expected_order):
        if stages.get(stage):
            first_appearance[stage] = i  # Use order index as "appearance order"

    completed_stages = list(first_appearance.keys())

    # Check if stages are in expected order
    stages_in_order = True
    prev_idx = -1
    for stage in completed_stages:
        curr_idx = expected_order.index(stage)
        if curr_idx < prev_idx:
            stages_in_order = False
            break
        prev_idx = curr_idx

    return {
        "stages_completed": completed_stages,
        "stages_in_order": stages_in_order,
        "total_stages": len(completed_stages),
        "stage_completion_rate": len(completed_stages) / len(expected_order),
        "expected_stages": expected_order,
        "tool_used": stages.get("tool_used", False),
    }


def ask_baseline(question: str, context: str, model: str = "claude-sonnet-4-5-20250929") -> tuple:
    """
    Baseline: direct question without skills. Outputs number only.

    Returns:
        tuple: (answer, trace_dict)
    """
    prompt = f"""Answer this financial question.

Data:
{context}

Question: {question}

Reply with ONLY the final number (no text, no units, no explanation).
- For percentages: round to whole number or 1 decimal (e.g., "14" or "9.9")
- For yes/no: reply "yes" or "no"
- No symbols like % or $

Answer:"""

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    duration_ms = int((time.time() - start_time) * 1000)
    answer = response.content[0].text.strip()

    trace = {
        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        "response": answer,
        "duration_ms": duration_ms,
    }

    return answer, trace


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
                   max_turns: int = 5) -> tuple[str, str, dict]:
    """
    Answer with skill enhancement and optional tool execution.

    Returns:
        tuple: (full_response, final_answer, trace_dict)
    """
    start_time = time.time()
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
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Parse stages and calculate metrics
                    stages = parse_finqa_stages(complete_response)
                    stage_metrics = analyze_stage_metrics(stages)

                    trace = {
                        "prompt": prompt[:500] + "...",
                        "response": complete_response,
                        "duration_ms": duration_ms,
                        "stages": {k: v for k, v in stages.items() if k != "raw"},
                        "stage_metrics": stage_metrics,
                    }
                    return complete_response, final_answer, trace

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
    duration_ms = int((time.time() - start_time) * 1000)

    # Parse stages and calculate metrics
    stages = parse_finqa_stages(complete_response)
    stage_metrics = analyze_stage_metrics(stages)

    trace = {
        "prompt": prompt[:500] + "...",
        "response": complete_response,
        "duration_ms": duration_ms,
        "stages": {k: v for k, v in stages.items() if k != "raw"},
        "stage_metrics": stage_metrics,
    }
    return complete_response, final_answer, trace


def run_benchmark(source: str = "sample", limit: int = None,
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
    stage_metrics_list = []  # Collect stage metrics for aggregation

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        qid = sample["id"]
        question = sample["question"]
        context = sample["context"]
        gold_answer = sample["answer"]

        print(f"\n[{i+1}/{len(samples)}] {question[:60]}...")
        print(f"Gold: {gold_answer}")

        # Baseline (no skills)
        baseline_trace = None
        try:
            pred_baseline, baseline_trace = ask_baseline(question, context, model)
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
            "correct": is_correct,
            "trace": baseline_trace,
        })

        # With skill
        skill_trace = None
        try:
            full_response, pred_skill, skill_trace = ask_with_skill(
                question, context, skill_manager, skill_names, model
            )
            is_correct = finqa_equal(pred_skill, gold_answer)
            status = "✓" if is_correct else "✗"

            # Print stage info
            if skill_trace and skill_trace.get("stage_metrics"):
                metrics = skill_trace["stage_metrics"]
                stages_str = "/".join([s.upper()[:3] for s in metrics.get("stages_completed", [])])
                if stages_str:
                    print(f"Skill:    {pred_skill} -> {status} [{stages_str}]")
                else:
                    print(f"Skill:    {pred_skill} -> {status}")
                stage_metrics_list.append(metrics)
            else:
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
            "correct": is_correct,
            "trace": skill_trace,
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

    # Stage Monitor Summary
    print("\n" + "-" * 40)
    print("STAGE MONITOR SUMMARY")
    aggregated_stage_metrics = {}
    if stage_metrics_list:
        total = len(stage_metrics_list)
        in_order_count = sum(1 for s in stage_metrics_list if s.get("stages_in_order", False))
        avg_completion = sum(s.get("stage_completion_rate", 0) for s in stage_metrics_list) / total
        tool_used_count = sum(1 for s in stage_metrics_list if s.get("tool_used", False))

        # Count each stage
        expected_stages = ["understand", "locate", "extract", "formula", "calculate", "format"]
        stage_counts = {stage: 0 for stage in expected_stages}
        for s in stage_metrics_list:
            for stage in s.get("stages_completed", []):
                if stage in stage_counts:
                    stage_counts[stage] += 1

        aggregated_stage_metrics = {
            "total_samples": total,
            "stages_in_order_count": in_order_count,
            "stages_in_order_rate": in_order_count / total,
            "avg_completion_rate": avg_completion,
            "tool_used_count": tool_used_count,
            "tool_used_rate": tool_used_count / total,
            "stage_counts": stage_counts,
            "stage_rates": {k: v / total for k, v in stage_counts.items()},
        }

        print(f"\n[Skill Mode]")
        print(f"  Stages in order: {in_order_count}/{total} ({in_order_count/total:.1%})")
        print(f"  Avg completion rate: {avg_completion:.1%}")
        print(f"  Tool usage: {tool_used_count}/{total} ({tool_used_count/total:.1%})")
        print(f"  Stage breakdown:")
        for stage, count in stage_counts.items():
            print(f"    {stage}: {count}/{total} ({count/total:.1%})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "meta": {
            "timestamp": timestamp,
            "model": model,
            "source": source,
            "num_samples": len(samples),
            "skills": skill_names,
        },
        "metrics": {
            "baseline": metrics_baseline,
            "skill": metrics_skill,
            "improvement": improvement,
        },
        "stage_monitor": aggregated_stage_metrics,
        "traces": {
            "baseline": results_baseline,
            "skill": results_skill,
        },
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
    parser.add_argument("--limit", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")

    args = parser.parse_args()
    run_benchmark(
        source=args.source,
        limit=args.limit,
        model=args.model
    )
