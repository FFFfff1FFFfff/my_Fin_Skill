#!/usr/bin/env python3
"""
MMLongBench-Doc benchmark runner: compare baseline vs with-skill performance
PDF document understanding and QA benchmark
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from data_loader import load_mmlongbench, load_sample_data, download_pdf, pdf_to_base64
from evaluator import eval_score, evaluate_batch
from skill_system import SkillManager

client = Anthropic()


def create_pdf_message(pdf_base64: str, question: str, system_prompt: str = None) -> dict:
    """Create a message with PDF content for Claude API."""
    user_content = [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_base64
            }
        },
        {
            "type": "text",
            "text": question
        }
    ]

    return {
        "messages": [{"role": "user", "content": user_content}],
        "system": system_prompt
    }


def ask_baseline(pdf_base64: str, question: str, answer_format: str,
                 model: str = "claude-sonnet-4-5-20250929") -> str:
    """Baseline: direct question without skills."""
    format_hint = {
        "Int": "Reply with only the integer number.",
        "Float": "Reply with only the number (can include decimals).",
        "Str": "Reply with a concise answer.",
        "List": "Reply with a list in format: ['item1', 'item2', ...]",
        "None": "If the question cannot be answered from the document, reply 'Not answerable'."
    }.get(answer_format, "Reply concisely.")

    prompt = f"""Based on the PDF document above, answer the following question.

Question: {question}

{format_hint}

Answer:"""

    msg_data = create_pdf_message(pdf_base64, prompt)

    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        messages=msg_data["messages"]
    )

    return response.content[0].text.strip()


def ask_with_skill(pdf_base64: str, question: str, answer_format: str,
                   skill_prompt: str, model: str = "claude-sonnet-4-5-20250929") -> tuple[str, str]:
    """
    Answer with skill enhancement.

    Returns:
        tuple: (full_response, extracted_answer)
    """
    format_hint = {
        "Int": "The answer should be an integer number.",
        "Float": "The answer should be a number (can include decimals).",
        "Str": "The answer should be a concise text string.",
        "List": "The answer should be a list in format: ['item1', 'item2', ...]",
        "None": "If the question cannot be answered, state 'Not answerable'."
    }.get(answer_format, "")

    prompt = f"""Based on the PDF document above, answer the following question.

Question: {question}

{format_hint}

Use the document analysis framework to:
1. Identify relevant sections/pages in the document
2. Extract the specific information needed
3. Formulate your answer

End your response with:
Final Answer: [your answer]"""

    msg_data = create_pdf_message(pdf_base64, prompt, system_prompt=skill_prompt)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=msg_data["system"] if msg_data["system"] else [],
        messages=msg_data["messages"]
    )

    full_response = response.content[0].text.strip()

    # Extract final answer
    import re
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', full_response, re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        # Fallback: use last line
        lines = [l.strip() for l in full_response.split('\n') if l.strip()]
        extracted = lines[-1] if lines else full_response

    return full_response, extracted


def run_benchmark(limit: int = None,
                  skip_unanswerable: bool = False,
                  model: str = "claude-sonnet-4-5-20250929",
                  use_sample: bool = False):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.

    Args:
        limit: Number of samples to test
        skip_unanswerable: Skip "Not answerable" questions
        model: Model to use
        use_sample: Use sample data instead of downloading
    """
    print("=" * 70)
    print("MMLongBench-Doc Skill Benchmark")
    print("=" * 70)

    # Load data
    print(f"\nLoading data (limit={limit}, skip_unanswerable={skip_unanswerable})...")
    if use_sample:
        samples = load_sample_data()
    else:
        samples = load_mmlongbench(limit=limit, skip_unanswerable=skip_unanswerable)
    print(f"Loaded {len(samples)} samples")

    # Load skills
    skill_manager = SkillManager()
    skill_names = ['pdf_document_qa']

    # Check if skills exist
    available_skills = skill_manager.list_skills()
    active_skills = [s for s in skill_names if s in available_skills]

    if active_skills:
        skill_prompt = skill_manager.build_system_prompt(active_skills)
        print(f"Loaded skills: {active_skills}")
    else:
        skill_prompt = None
        print("No PDF skills found. Running baseline only.")

    # Track unique PDFs to download
    unique_pdfs = set(s["doc_id"] for s in samples)
    print(f"Unique PDFs to process: {len(unique_pdfs)}")

    # Download PDFs and cache
    print("\nDownloading PDFs...")
    pdf_cache = {}
    for doc_id in unique_pdfs:
        pdf_path = download_pdf(doc_id)
        if pdf_path:
            pdf_base64 = pdf_to_base64(pdf_path)
            if pdf_base64:
                pdf_cache[doc_id] = pdf_base64
                print(f"  ✓ {doc_id}")
            else:
                print(f"  ✗ {doc_id} (failed to read)")
        else:
            print(f"  ✗ {doc_id} (download failed)")

    # Filter samples to only those with available PDFs
    samples = [s for s in samples if s["doc_id"] in pdf_cache]
    print(f"\nSamples with available PDFs: {len(samples)}")

    if not samples:
        print("No samples to process. Exiting.")
        return None

    results_baseline = []
    results_skill = []

    print("\n" + "-" * 70)

    for i, sample in enumerate(samples):
        doc_id = sample["doc_id"]
        question = sample["question"]
        answer = sample["answer"]
        answer_format = sample["answer_format"]

        print(f"\n[{i+1}/{len(samples)}] {doc_id}")
        print(f"Q: {question[:60]}...")
        print(f"A: {answer} ({answer_format})")

        pdf_base64 = pdf_cache[doc_id]

        # Baseline
        try:
            pred_baseline = ask_baseline(pdf_base64, question, answer_format, model)
            score_baseline = eval_score(pred_baseline, answer, answer_format)
            print(f"Baseline: {pred_baseline[:50]}... -> {score_baseline:.2f}")
        except Exception as e:
            pred_baseline = ""
            score_baseline = 0.0
            print(f"Baseline: ERROR - {e}")

        results_baseline.append({
            "id": sample["id"],
            "doc_id": doc_id,
            "question": question,
            "ground_truth": answer,
            "answer_format": answer_format,
            "prediction": pred_baseline,
            "score": score_baseline
        })

        # With skill (if available)
        if skill_prompt:
            try:
                full_response, pred_skill = ask_with_skill(
                    pdf_base64, question, answer_format, skill_prompt, model
                )
                score_skill = eval_score(pred_skill, answer, answer_format)
                print(f"Skill:    {pred_skill[:50]}... -> {score_skill:.2f}")
            except Exception as e:
                pred_skill = ""
                full_response = ""
                score_skill = 0.0
                print(f"Skill:    ERROR - {e}")

            results_skill.append({
                "id": sample["id"],
                "doc_id": doc_id,
                "question": question,
                "ground_truth": answer,
                "answer_format": answer_format,
                "full_response": full_response,
                "prediction": pred_skill,
                "score": score_skill
            })

    # Evaluate
    eval_baseline = evaluate_batch(results_baseline)
    eval_skill = evaluate_batch(results_skill) if results_skill else None

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline:")
    print(f"  Accuracy: {eval_baseline['accuracy']:.1%} ({eval_baseline['total_score']:.1f}/{eval_baseline['total']})")
    print(f"  By Format:")
    for fmt, acc in eval_baseline['by_format'].items():
        detail = eval_baseline['by_format_detail'][fmt]
        print(f"    {fmt}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

    if eval_skill:
        print(f"\nWith Skill:")
        print(f"  Accuracy: {eval_skill['accuracy']:.1%} ({eval_skill['total_score']:.1f}/{eval_skill['total']})")
        print(f"  By Format:")
        for fmt, acc in eval_skill['by_format'].items():
            detail = eval_skill['by_format_detail'][fmt]
            print(f"    {fmt}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

        improvement = eval_skill['accuracy'] - eval_baseline['accuracy']
        print(f"\nImprovement: {improvement:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "num_samples": len(samples),
        "baseline": {
            "accuracy": eval_baseline['accuracy'],
            "results": results_baseline
        },
    }

    if eval_skill:
        output["skill"] = {
            "accuracy": eval_skill['accuracy'],
            "results": results_skill
        }
        output["improvement"] = eval_skill['accuracy'] - eval_baseline['accuracy']

    output_file = f"mmlongbench_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMLongBench-Doc Skill Benchmark")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of samples (default: all)")
    parser.add_argument("--skip-unanswerable", action="store_true",
                        help="Skip 'Not answerable' questions")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample data for testing")

    args = parser.parse_args()
    run_benchmark(
        limit=args.limit,
        skip_unanswerable=args.skip_unanswerable,
        model=args.model,
        use_sample=args.sample
    )
