#!/usr/bin/env python3
"""
SealQA benchmark runner: compare baseline vs with-skill performance
Supports web search via built-in WebSearch or external APIs (Tavily/Serper)

Includes Stage Monitor for tracking reasoning stages:
- SEARCH: Multiple query search
- CATEGORIZE: Rate source reliability
- DETECT: Detect conflicts
- RESOLVE: Resolve conflicts
- FINAL: Formulate answer
"""

import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_sealqa, load_sample_data
from evaluator import grade_answer, evaluate_batch
from skill_system import SkillManager

client = Anthropic()


# =============================================================================
# STAGE MONITOR: Track reasoning stages in conflicting_info_reasoner flow
# =============================================================================

def parse_skill_stages(response_text: str, used_web_search: bool = False) -> dict:
    """
    Parse skill response to extract structured reasoning stages.

    Expected stages based on conflicting_info_reasoner skill:
    - SEARCH: Web search was used or search results mentioned
    - CATEGORIZE: Source reliability assessment (explicit rating)
    - DETECT: Conflict detection between sources (explicit mention)
    - RESOLVE: Reasoning to resolve conflicts or reach conclusion
    - FINAL: Final answer is provided

    Args:
        response_text: The LLM response text
        used_web_search: Whether web search tool was actually invoked
    """
    stages = {
        "search": None,       # Search findings
        "categorize": None,   # Source reliability rating
        "detect": None,       # Conflict detection
        "resolve": None,      # Conflict resolution
        "final": None,        # Final answer
        "raw": response_text,
    }

    # SEARCH: Web search was used OR explicit mention of search/sources
    if used_web_search:
        stages["search"] = "web_search_invoked"
    else:
        search_patterns = [
            r'(?:search results?|web search|I searched)',
            r'(?:according to|based on)\s+(?:my\s+)?search',
            r'(?:found|retrieved)\s+(?:information|results)',
            r'sources?\s+(?:indicate|show|report|state)',
        ]
        for pattern in search_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 10)
                end = min(len(response_text), match.end() + 80)
                stages["search"] = response_text[start:end].strip()[:200]
                break

    # CATEGORIZE: Explicit source reliability assessment
    # Must have explicit reliability/credibility language
    categorize_patterns = [
        r'(?:source|website|site)\s*(?:is|are)\s*(?:reliable|unreliable|credible|trustworthy)',
        r'(?:reliability|credibility|trustworthiness)\s*[:=]?\s*(?:high|medium|low|good|poor)',
        r'(?:official|authoritative|reputable)\s+(?:source|website)',
        r'(?:primary|secondary|tertiary)\s+source',
    ]
    for pattern in categorize_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 10)
            end = min(len(response_text), match.end() + 60)
            stages["categorize"] = response_text[start:end].strip()[:150]
            break

    # DETECT: Explicit conflict/contradiction detection
    # Must explicitly mention conflicts or contradictions
    detect_patterns = [
        r'(?:conflict|contradiction|discrepancy)\s+(?:between|in|found)',
        r'(?:conflicting|contradictory)\s+(?:information|sources|data|reports)',
        r'sources?\s+(?:disagree|differ|conflict)',
        r'(?:inconsistent|contradicting)\s+(?:information|claims|statements)',
        r'found\s+(?:conflicting|different)\s+(?:information|answers)',
    ]
    for pattern in detect_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 10)
            end = min(len(response_text), match.end() + 80)
            stages["detect"] = response_text[start:end].strip()[:150]
            break

    # RESOLVE: Explicit reasoning/resolution step
    # Must show reasoning process, not just conclusions
    resolve_patterns = [
        r'(?:resolving|to resolve)\s+(?:this|the)\s+(?:conflict|contradiction)',
        r'(?:weighing|considering|comparing)\s+(?:the\s+)?(?:sources|evidence)',
        r'(?:more reliable|most credible|best supported)',
        r'(?:given|considering)\s+(?:the\s+)?(?:evidence|sources|reliability)',
        r'(?:therefore|thus|consequently),?\s+(?:I|we|the)',
    ]
    for pattern in resolve_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 10)
            end = min(len(response_text), match.end() + 80)
            stages["resolve"] = response_text[start:end].strip()[:150]
            break

    # FINAL: Final answer is provided (always check)
    # Look for explicit answer markers or the substantive answer
    final_patterns = [
        r'\*\*(?:answer|final answer)\*\*[:\s]*(.+?)(?:\n|$)',  # **Answer**: ...
        r'(?:the answer is|answer:|final answer:)\s*(.+?)(?:\.|$)',
        r'(?:in conclusion|to summarize|in summary)[,:]?\s*(.+?)(?:\.|$)',
    ]
    for pattern in final_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            content = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if len(content) > 5:  # Avoid empty matches
                stages["final"] = content[:200]
                break

    # If no explicit final answer marker, check if response has substantive content
    if not stages["final"] and len(response_text.strip()) > 20:
        # Use first substantive line as the answer indicator
        lines = [l.strip() for l in response_text.split('\n') if l.strip() and len(l.strip()) > 10]
        if lines:
            stages["final"] = lines[0][:200]

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
    expected_order = ["search", "categorize", "detect", "resolve", "final"]
    completed_stages = []

    for stage in expected_order:
        if stages.get(stage):
            completed_stages.append(stage)

    # Check if stages are in expected order (strict)
    stages_in_order = True
    last_idx = -1
    for stage in completed_stages:
        if stage in expected_order:
            idx = expected_order.index(stage)
            if idx < last_idx:
                stages_in_order = False
                break
            last_idx = idx

    return {
        "stages_completed": completed_stages,
        "stages_in_order": stages_in_order,
        "total_stages": len(completed_stages),
        "stage_completion_rate": len(completed_stages) / len(expected_order),
        "expected_stages": expected_order,
    }


# =============================================================================
# BASELINE AND SKILL FUNCTIONS
# =============================================================================

def ask_baseline(question: str, model: str = "claude-sonnet-4-5-20250929") -> tuple:
    """
    Baseline: direct question without search or skills.

    Returns: (answer, trace_dict)
    """
    prompt = f"""Answer this question directly and concisely.

Question: {question}

Answer:"""

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    duration_ms = int((time.time() - start_time) * 1000)
    answer = response.content[0].text.strip()

    trace = {
        "prompt": prompt,
        "response": answer,
        "duration_ms": duration_ms,
    }

    return answer, trace


def ask_with_skill(question: str, skill_prompt: str, use_search: bool = True,
                   search_backend: str = "builtin", model: str = "claude-sonnet-4-5-20250929") -> tuple:
    """
    Answer with skill enhancement and optional web search.

    Returns: (answer, trace_dict with stages)
    """
    start_time = time.time()
    full_response_text = ""
    search_results = None

    if use_search and search_backend == "builtin":
        # Use Claude's native web search via tool
        user_prompt = f"""Question: {question}

Use web search to find current information if needed.
Follow the reasoning framework in the system prompt.
Give a direct, concise answer."""

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            system=skill_prompt,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3
            }],
            messages=[{"role": "user", "content": user_prompt}]
        )

        # Extract text from response (may have tool use blocks)
        answer_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                answer_parts.append(block.text)
        full_response_text = " ".join(answer_parts).strip()
        answer = full_response_text if full_response_text else "No answer generated"

    elif use_search and search_backend in ["tavily", "serper"]:
        # Use external search API
        from skills.web_search_tool.search_tools import search

        # First, do a search
        search_results = search(question, backend=search_backend)

        user_prompt = f"""Question: {question}

Search results:
{search_results}

Based on the search results above, answer the question.
Follow the reasoning framework in the system prompt.
Give a direct, concise answer."""

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=skill_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        full_response_text = response.content[0].text.strip()
        answer = full_response_text

    else:
        # No search, just skill prompt
        user_prompt = f"""Question: {question}

Follow the reasoning framework in the system prompt.
Give a direct, concise answer."""

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=skill_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        full_response_text = response.content[0].text.strip()
        answer = full_response_text

    duration_ms = int((time.time() - start_time) * 1000)

    # Determine if web search was actually used
    used_web_search = use_search and (search_backend == "builtin" or search_results is not None)

    # Parse stages from response
    stages = parse_skill_stages(full_response_text, used_web_search=used_web_search)
    stage_metrics = analyze_stage_metrics(stages)

    trace = {
        "prompt": user_prompt if 'user_prompt' in dir() else question,
        "response": full_response_text,
        "duration_ms": duration_ms,
        "search_results": search_results,
        "stages": {
            "search": stages["search"],
            "categorize": stages["categorize"],
            "detect": stages["detect"],
            "resolve": stages["resolve"],
            "final": stages["final"],
        },
        "stage_metrics": stage_metrics,
    }

    return answer, trace


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(source: str = "sample", limit: int = None,
                  use_search: bool = True, search_backend: str = "builtin",
                  model: str = "claude-sonnet-4-5-20250929",
                  grading_model: str = "claude-sonnet-4-5-20250929"):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.
    """
    print("=" * 70)
    print("SealQA Skill Benchmark")
    print("=" * 70)

    # Load data
    print(f"\nLoading data (source={source}, limit={limit})...")
    if source == "sample":
        samples = load_sample_data()
    else:
        samples = load_sealqa(source=source, limit=limit)
    print(f"Loaded {len(samples)} samples")

    # Load skills
    skill_manager = SkillManager()
    skill_names = ['web_search_tool', 'conflicting_info_reasoner']
    skill_prompt = skill_manager.build_system_prompt(skill_names)
    print(f"Loaded skills: {skill_names}")
    print(f"Search enabled: {use_search} (backend: {search_backend})")

    results_baseline = []
    results_skill = []
    stage_metrics_list = []  # Collect stage metrics for aggregation

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        qid = sample["id"]
        question = sample["question"]
        gold_answer = sample["answer"]

        print(f"\n[{i+1}/{len(samples)}] {question[:60]}...")
        print(f"Gold: {gold_answer[:50]}...")

        # Baseline (no search, no skills)
        baseline_trace = None
        try:
            pred_baseline, baseline_trace = ask_baseline(question, model)
            grade_baseline = grade_answer(question, gold_answer, pred_baseline, grading_model)
            grade_str = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[grade_baseline]
            print(f"  [Baseline] {pred_baseline[:50]}... -> {grade_str} ({baseline_trace['duration_ms']}ms)")
        except Exception as e:
            pred_baseline = ""
            grade_baseline = "C"
            print(f"  [Baseline] ERROR - {e}")

        results_baseline.append({
            "id": qid,
            "question": question,
            "answer": gold_answer,
            "prediction": pred_baseline,
            "grade": grade_baseline,
            "trace": baseline_trace,
        })

        # With skill (and optionally search)
        skill_trace = None
        try:
            pred_skill, skill_trace = ask_with_skill(question, skill_prompt, use_search, search_backend, model)
            grade_skill = grade_answer(question, gold_answer, pred_skill, grading_model)
            grade_str = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[grade_skill]

            # Print stage info
            stages = skill_trace.get("stages", {})
            stage_tags = []
            if stages.get("search"):
                stage_tags.append("SEARCH")
            if stages.get("categorize"):
                stage_tags.append("CAT")
            if stages.get("detect"):
                stage_tags.append("DETECT")
            if stages.get("resolve"):
                stage_tags.append("RESOLVE")
            stage_info = f"[{'/'.join(stage_tags)}]" if stage_tags else ""

            print(f"  [Skill] {pred_skill[:50]}... -> {grade_str} ({skill_trace['duration_ms']}ms) {stage_info}")

            # Collect stage metrics
            if skill_trace.get("stage_metrics"):
                stage_metrics_list.append(skill_trace["stage_metrics"])

        except Exception as e:
            pred_skill = ""
            grade_skill = "C"
            print(f"  [Skill] ERROR - {e}")

        results_skill.append({
            "id": qid,
            "question": question,
            "answer": gold_answer,
            "prediction": pred_skill,
            "grade": grade_skill,
            "trace": skill_trace,
        })

    # Calculate metrics
    def calc_metrics(results):
        grades = [r["grade"] for r in results]
        total = len(grades)
        return {
            "correct": sum(g == 'A' for g in grades) / total if total > 0 else 0,
            "incorrect": sum(g == 'B' for g in grades) / total if total > 0 else 0,
            "not_attempted": sum(g == 'C' for g in grades) / total if total > 0 else 0,
            "total": total
        }

    metrics_baseline = calc_metrics(results_baseline)
    metrics_skill = calc_metrics(results_skill)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline (no search, no skills):")
    print(f"  Correct:       {metrics_baseline['correct']:.1%}")
    print(f"  Incorrect:     {metrics_baseline['incorrect']:.1%}")
    print(f"  Not Attempted: {metrics_baseline['not_attempted']:.1%}")

    print(f"\nWith Skill (search={use_search}, backend={search_backend}):")
    print(f"  Correct:       {metrics_skill['correct']:.1%}")
    print(f"  Incorrect:     {metrics_skill['incorrect']:.1%}")
    print(f"  Not Attempted: {metrics_skill['not_attempted']:.1%}")

    improvement = metrics_skill['correct'] - metrics_baseline['correct']
    print(f"\nImprovement: {improvement:+.1%}")

    # Stage Monitor Summary
    print("\n" + "-" * 40)
    print("STAGE MONITOR SUMMARY")
    aggregated_stage_metrics = {}
    if stage_metrics_list:
        total = len(stage_metrics_list)
        in_order_count = sum(1 for s in stage_metrics_list if s.get("stages_in_order", False))
        avg_completion = sum(s.get("stage_completion_rate", 0) for s in stage_metrics_list) / total

        # Count each stage
        stage_counts = {"search": 0, "categorize": 0, "detect": 0, "resolve": 0, "final": 0}
        for s in stage_metrics_list:
            for stage in s.get("stages_completed", []):
                if stage in stage_counts:
                    stage_counts[stage] += 1

        aggregated_stage_metrics = {
            "total_samples": total,
            "stages_in_order_count": in_order_count,
            "stages_in_order_rate": in_order_count / total,
            "avg_completion_rate": avg_completion,
            "stage_counts": stage_counts,
            "stage_rates": {k: v / total for k, v in stage_counts.items()},
        }

        print(f"\n[Skill Mode]")
        print(f"  Stages in order: {in_order_count}/{total} ({in_order_count/total:.1%})")
        print(f"  Avg completion rate: {avg_completion:.1%}")
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
            "search_enabled": use_search,
            "search_backend": search_backend,
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

    output_file = f"sealqa_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SealQA Skill Benchmark")
    parser.add_argument("--source", type=str, default="sample",
                        help="Data source: 'sample', 'seal_0', 'seal_hard', 'longseal', or path")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--search", action="store_true", default=True, help="Enable web search")
    parser.add_argument("--no-search", dest="search", action="store_false", help="Disable web search")
    parser.add_argument("--backend", type=str, default="builtin",
                        choices=["builtin", "tavily", "serper"], help="Search backend")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")

    args = parser.parse_args()
    run_benchmark(
        source=args.source,
        limit=args.limit,
        use_search=args.search,
        search_backend=args.backend,
        model=args.model
    )
