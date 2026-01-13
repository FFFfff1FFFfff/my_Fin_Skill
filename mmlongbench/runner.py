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
from data_loader import load_mmlongbench, load_sample_data, download_pdf, pdf_to_base64, pdf_to_images, PDF_CACHE_DIR
from evaluator import eval_score, evaluate_batch
from skill_system import SkillManager

# Import PDF text extraction tools
try:
    from skills.pdf_text_extractor.pdf_tools import extract_pdf_text, search_in_pdf
    HAS_PDF_TOOLS = True
except ImportError:
    HAS_PDF_TOOLS = False

# Import OpenAI for answer extraction (optional)
try:
    from openai import OpenAI
    openai_client = OpenAI()
    HAS_OPENAI = True
except ImportError:
    openai_client = None
    HAS_OPENAI = False

client = Anthropic()

# Answer extraction prompt (from official MMLongBench-Doc)
ANSWER_EXTRACTION_PROMPT = """Your task is to extract the answer from the given analysis.
The answer should be in one of these formats: Integer, Float, String, or List.

Rules:
- If the analysis indicates the question cannot be answered from the document, output "Not answerable"
- If the analysis indicates failure to read/understand the document, output "Fail to answer"
- For Integer: extract just the number (e.g., "25")
- For Float: extract the number with decimals (e.g., "3.14")
- For String: extract the concise answer text
- For List: format as ['item1', 'item2', ...]

Format your response exactly as:
Extracted answer: [answer]
Answer format: [format]"""


def extract_answer_with_gpt(question: str, analysis: str) -> str:
    """Use GPT-4o to extract structured answer from free-form analysis."""
    if not HAS_OPENAI:
        return None

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": ANSWER_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nAnalysis:\n{analysis}"}
            ],
            temperature=0,
            max_tokens=256
        )
        result = response.choices[0].message.content.strip()

        # Parse "Extracted answer: ..."
        import re
        match = re.search(r'Extracted answer:\s*(.+?)(?:\n|Answer format:|$)', result, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return result
    except Exception as e:
        print(f"GPT extraction failed: {e}")
        return None


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


def create_images_message(page_images: list[dict], question: str, system_prompt: str = None) -> dict:
    """Create a message with page images for Claude API (matching original paper approach)."""
    user_content = []

    # Add each page as an image
    for img in page_images:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["image_base64"]
            }
        })

    # Add question text
    user_content.append({
        "type": "text",
        "text": question
    })

    return {
        "messages": [{"role": "user", "content": user_content}],
        "system": system_prompt
    }


def ask_baseline(doc_content: any, question: str, answer_format: str,
                 model: str = "claude-sonnet-4-5-20250929",
                 use_images: bool = False,
                 use_gpt_extraction: bool = False) -> tuple[str, str]:
    """
    Baseline: direct question without skills.

    Args:
        doc_content: Either pdf_base64 (str) or page_images (list)
        question: The question to answer
        answer_format: Expected answer format (Int, Float, Str, List, None)
        model: Model to use
        use_images: If True, doc_content is list of page images
        use_gpt_extraction: If True, use GPT-4o to extract answer from response

    Returns:
        tuple: (raw_response, extracted_answer)
    """
    # Allow free-form response when using GPT extraction
    if use_gpt_extraction:
        prompt = f"""Based on the document above, answer the following question.

Question: {question}

Analyze the document carefully and provide your answer. If the information is not available in the document, state that it cannot be answered."""
        max_tokens = 1024
    else:
        format_instruction = {
            "Int": "Answer with ONLY the integer number, nothing else.",
            "Float": "Answer with ONLY the number, nothing else.",
            "Str": "Answer as concisely as possible - just the answer, no explanation.",
            "List": "Answer with ONLY a list like: ['item1', 'item2']",
            "None": "If not answerable from the document, reply ONLY: Not answerable"
        }.get(answer_format, "Answer concisely.")

        prompt = f"""Question: {question}

{format_instruction}

Final Answer:"""
        max_tokens = 100

    if use_images:
        msg_data = create_images_message(doc_content, prompt)
    else:
        msg_data = create_pdf_message(doc_content, prompt)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=msg_data["messages"]
    )

    raw = response.content[0].text.strip()

    # Extract answer
    if use_gpt_extraction and HAS_OPENAI:
        extracted = extract_answer_with_gpt(question, raw)
        if extracted:
            return raw, extracted

    # Fallback: simple extraction
    lines = raw.split('\n')
    return raw, lines[0].strip()


def ask_with_skill(doc_content: any, question: str, answer_format: str,
                   skill_prompt: str, model: str = "claude-sonnet-4-5-20250929",
                   extracted_text: dict = None,
                   use_images: bool = False,
                   use_gpt_extraction: bool = False) -> tuple[str, str]:
    """
    Answer with skill enhancement, optionally including extracted text.

    Args:
        doc_content: Either pdf_base64 (str) or page_images (list)
        use_images: If True, doc_content is list of page images
        use_gpt_extraction: If True, use GPT-4o to extract answer

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

    # Build SMART text context using keyword search
    text_context = ""
    if extracted_text and "pages" in extracted_text and HAS_PDF_TOOLS:
        # Extract keywords from question (simple approach: nouns and numbers)
        import re
        words = re.findall(r'\b[A-Za-z]{3,}\b|\b\d+\.?\d*\b', question)
        keywords = [w.lower() for w in words if w.lower() not in
                   {'what', 'which', 'where', 'when', 'how', 'many', 'much',
                    'the', 'and', 'for', 'are', 'this', 'that', 'from', 'with',
                    'does', 'did', 'was', 'were', 'have', 'has', 'been', 'being'}]

        if keywords:
            # Search for keywords in document
            matches = search_in_pdf(extracted_text, ' '.join(keywords[:5]))

            if matches:
                # Only include pages with matches, with full context
                relevant_pages = []
                seen_pages = set()
                for match in matches[:10]:  # Top 10 matches
                    page_num = match.get("page", 0)
                    if page_num not in seen_pages:
                        seen_pages.add(page_num)
                        page_key = page_num if page_num in extracted_text["pages"] else str(page_num)
                        if page_key in extracted_text["pages"]:
                            page_text = extracted_text["pages"][page_key]
                            # Include more text per page (up to 1500 chars)
                            relevant_pages.append(f"[Page {page_num}]:\n{page_text[:1500]}")

                if relevant_pages:
                    text_context = f"""

--- RELEVANT TEXT SECTIONS (based on keyword search) ---
{chr(10).join(relevant_pages[:5])}
--- END RELEVANT SECTIONS ---
"""

    # Simpler, more focused prompt
    prompt = f"""Based on the document above, answer this question:

Question: {question}

{format_hint}
{text_context}
Provide your answer. End with:
Final Answer: [your answer]"""

    # Use a simpler system prompt instead of the full skill framework
    simple_skill_prompt = """You are a document analysis expert. Extract information accurately from the provided document.
For questions about data: look for exact numbers, dates, and names.
For list questions: identify all items that match the criteria.
If information is not in the document, state "Not answerable"."""

    if use_images:
        msg_data = create_images_message(doc_content, prompt, system_prompt=simple_skill_prompt)
    else:
        msg_data = create_pdf_message(doc_content, prompt, system_prompt=simple_skill_prompt)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=msg_data["system"] if msg_data["system"] else [],
        messages=msg_data["messages"]
    )

    full_response = response.content[0].text.strip()

    # Extract answer
    if use_gpt_extraction and HAS_OPENAI:
        extracted = extract_answer_with_gpt(question, full_response)
        if extracted:
            return full_response, extracted

    # Fallback: regex extraction
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
                  model: str = "claude-sonnet-4-5-20250929",
                  use_sample: bool = False,
                  use_images: bool = False,
                  use_gpt_extraction: bool = False):
    """
    Run benchmark comparing baseline vs skill-enhanced performance.
    Reports both all-questions and answerable-only metrics.

    Args:
        limit: Number of samples to test
        model: Model to use
        use_sample: Use sample data instead of downloading
        use_images: Use page images instead of PDF (matches original paper)
        use_gpt_extraction: Use GPT-4o to extract answers (matches original paper)
    """
    print("=" * 70)
    print("MMLongBench-Doc Skill Benchmark")
    print("=" * 70)

    if use_images:
        print("Mode: Page images (matching original paper)")
    else:
        print("Mode: PDF base64")

    if use_gpt_extraction:
        if HAS_OPENAI:
            print("Answer extraction: GPT-4o (matching original paper)")
        else:
            print("Warning: OpenAI not available, using regex extraction")
            use_gpt_extraction = False
    else:
        print("Answer extraction: Regex")

    # Load data (always load all, report both metrics)
    print(f"\nLoading data (limit={limit})...")
    if use_sample:
        samples = load_sample_data()
    else:
        samples = load_mmlongbench(limit=limit, skip_unanswerable=False)
    print(f"Loaded {len(samples)} samples")

    # Load skills
    skill_manager = SkillManager()
    skill_names = ['pdf_document_qa', 'pdf_text_extractor']

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
    print("\nDownloading and processing PDFs...")
    doc_cache = {}  # Either pdf_base64 or page_images
    text_cache = {}
    for doc_id in unique_pdfs:
        pdf_path = download_pdf(doc_id)
        if pdf_path:
            if use_images:
                # Convert to page images (matching original paper)
                images = pdf_to_images(pdf_path)
                if images:
                    doc_cache[doc_id] = images
                    print(f"  ✓ {doc_id} ({len(images)} pages)")
                else:
                    print(f"  ✗ {doc_id} (failed to convert to images)")
            else:
                # Use PDF base64
                pdf_base64 = pdf_to_base64(pdf_path)
                if pdf_base64:
                    doc_cache[doc_id] = pdf_base64
                    print(f"  ✓ {doc_id}")
                else:
                    print(f"  ✗ {doc_id} (failed to read)")

            # Also extract text if tools available (for skill mode)
            if HAS_PDF_TOOLS and doc_id in doc_cache:
                extracted = extract_pdf_text(str(pdf_path))
                if "error" not in extracted:
                    text_cache[doc_id] = extracted
        else:
            print(f"  ✗ {doc_id} (download failed)")

    if HAS_PDF_TOOLS and text_cache:
        print(f"Extracted text from {len(text_cache)} PDFs")

    # Filter samples to only those with available documents
    samples = [s for s in samples if s["doc_id"] in doc_cache]
    print(f"\nSamples with available documents: {len(samples)}")

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

        doc_content = doc_cache[doc_id]

        # Baseline
        try:
            raw_response, pred_baseline = ask_baseline(
                doc_content, question, answer_format, model,
                use_images=use_images, use_gpt_extraction=use_gpt_extraction
            )
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
                extracted_text = text_cache.get(doc_id)
                full_response, pred_skill = ask_with_skill(
                    doc_content, question, answer_format, skill_prompt, model,
                    extracted_text=extracted_text,
                    use_images=use_images, use_gpt_extraction=use_gpt_extraction
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

    # Separate results into all vs answerable-only
    results_baseline_answerable = [r for r in results_baseline if r["ground_truth"] != "Not answerable"]
    results_skill_answerable = [r for r in results_skill if r["ground_truth"] != "Not answerable"] if results_skill else []

    # Evaluate all
    eval_baseline_all = evaluate_batch(results_baseline)
    eval_skill_all = evaluate_batch(results_skill) if results_skill else None

    # Evaluate answerable only
    eval_baseline_ans = evaluate_batch(results_baseline_answerable) if results_baseline_answerable else None
    eval_skill_ans = evaluate_batch(results_skill_answerable) if results_skill_answerable else None

    num_unanswerable = len(results_baseline) - len(results_baseline_answerable)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal: {len(results_baseline)} samples ({num_unanswerable} unanswerable)")

    print(f"\n{'='*35} ALL QUESTIONS {'='*35}")
    print(f"\nBaseline:")
    print(f"  Accuracy: {eval_baseline_all['accuracy']:.1%} ({eval_baseline_all['total_score']:.1f}/{eval_baseline_all['total']})")
    print(f"  By Format:")
    for fmt, acc in eval_baseline_all['by_format'].items():
        detail = eval_baseline_all['by_format_detail'][fmt]
        print(f"    {fmt}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

    if eval_skill_all:
        print(f"\nWith Skill:")
        print(f"  Accuracy: {eval_skill_all['accuracy']:.1%} ({eval_skill_all['total_score']:.1f}/{eval_skill_all['total']})")
        print(f"  By Format:")
        for fmt, acc in eval_skill_all['by_format'].items():
            detail = eval_skill_all['by_format_detail'][fmt]
            print(f"    {fmt}: {acc:.1%} ({detail['total_score']:.1f}/{detail['count']})")

        improvement = eval_skill_all['accuracy'] - eval_baseline_all['accuracy']
        print(f"\n  Improvement: {improvement:+.1%}")

    if eval_baseline_ans:
        print(f"\n{'='*30} ANSWERABLE ONLY ({len(results_baseline_answerable)}) {'='*30}")
        print(f"\nBaseline:")
        print(f"  Accuracy: {eval_baseline_ans['accuracy']:.1%} ({eval_baseline_ans['total_score']:.1f}/{eval_baseline_ans['total']})")

        if eval_skill_ans:
            print(f"\nWith Skill:")
            print(f"  Accuracy: {eval_skill_ans['accuracy']:.1%} ({eval_skill_ans['total_score']:.1f}/{eval_skill_ans['total']})")

            improvement_ans = eval_skill_ans['accuracy'] - eval_baseline_ans['accuracy']
            print(f"\n  Improvement: {improvement_ans:+.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "num_samples": len(samples),
        "num_unanswerable": num_unanswerable,
        "all_questions": {
            "baseline": {
                "accuracy": eval_baseline_all['accuracy'],
                "results": results_baseline
            }
        },
        "answerable_only": {
            "num_samples": len(results_baseline_answerable),
            "baseline": {
                "accuracy": eval_baseline_ans['accuracy'] if eval_baseline_ans else 0
            }
        }
    }

    if eval_skill_all:
        output["all_questions"]["skill"] = {
            "accuracy": eval_skill_all['accuracy'],
            "results": results_skill
        }
        output["all_questions"]["improvement"] = eval_skill_all['accuracy'] - eval_baseline_all['accuracy']

    if eval_skill_ans:
        output["answerable_only"]["skill"] = {
            "accuracy": eval_skill_ans['accuracy']
        }
        output["answerable_only"]["improvement"] = eval_skill_ans['accuracy'] - eval_baseline_ans['accuracy']

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
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample data for testing")
    parser.add_argument("--use-images", action="store_true",
                        help="Use page images instead of PDF (matches original paper)")
    parser.add_argument("--use-gpt-extraction", action="store_true",
                        help="Use GPT-4o to extract answers (matches original paper)")
    parser.add_argument("--official", action="store_true",
                        help="Use official paper settings (--use-images + --use-gpt-extraction)")

    args = parser.parse_args()

    # --official enables both official paper settings
    use_images = args.use_images or args.official
    use_gpt_extraction = args.use_gpt_extraction or args.official

    run_benchmark(
        limit=args.limit,
        model=args.model,
        use_sample=args.sample,
        use_images=use_images,
        use_gpt_extraction=use_gpt_extraction
    )
