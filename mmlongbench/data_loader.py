"""MMLongBench-Doc data loader - PDF document understanding benchmark"""

import json
import os
import ast
import base64
import requests
from typing import Optional
from pathlib import Path

# URLs for data
SAMPLES_URL = "https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/main/data/samples.json"
PDF_BASE_URL = "https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/main/data/documents"

# Local cache directory
CACHE_DIR = Path(__file__).parent / "cache"
PDF_CACHE_DIR = CACHE_DIR / "pdfs"


def ensure_cache_dirs():
    """Create cache directories if they don't exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    PDF_CACHE_DIR.mkdir(exist_ok=True)


def download_samples() -> list[dict]:
    """Download samples.json from GitHub."""
    cache_file = CACHE_DIR / "samples.json"

    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    ensure_cache_dirs()
    print("Downloading samples.json...")
    response = requests.get(SAMPLES_URL)
    response.raise_for_status()
    data = response.json()

    # Cache locally
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def download_pdf(doc_id: str) -> Optional[Path]:
    """Download a PDF file from GitHub."""
    ensure_cache_dirs()

    # URL encode the filename
    pdf_url = f"{PDF_BASE_URL}/{requests.utils.quote(doc_id)}"
    local_path = PDF_CACHE_DIR / doc_id

    if local_path.exists():
        return local_path

    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

        return local_path
    except Exception as e:
        print(f"Failed to download {doc_id}: {e}")
        return None


def pdf_to_base64(pdf_path: Path) -> Optional[str]:
    """Convert PDF file to base64 string for Claude API."""
    try:
        with open(pdf_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to read PDF {pdf_path}: {e}")
        return None


def pdf_to_images(pdf_path: Path, dpi: int = 144, max_pages: int = None) -> list[dict]:
    """
    Convert PDF pages to base64-encoded PNG images.

    This matches the original MMLongBench-Doc evaluation approach which uses
    page images instead of PDF documents.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering (default 144 as in original paper)
        max_pages: Maximum number of pages to convert (None for all)

    Returns:
        List of dicts with {"page": int, "image_base64": str, "media_type": "image/png"}
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Warning: PyMuPDF (fitz) not installed. Run: pip install pymupdf")
        return []

    images = []
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        if max_pages:
            total_pages = min(total_pages, max_pages)

        zoom = dpi / 72  # 72 is PDF default DPI
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)

            # Convert to PNG bytes
            png_bytes = pix.tobytes("png")
            image_base64 = base64.standard_b64encode(png_bytes).decode('utf-8')

            images.append({
                "page": page_num + 1,
                "image_base64": image_base64,
                "media_type": "image/png"
            })

        doc.close()
        return images

    except Exception as e:
        print(f"Failed to convert PDF to images {pdf_path}: {e}")
        return []


def parse_field(field_str: str):
    """Parse string representation of list/value."""
    if not field_str or field_str == "[]":
        return []
    try:
        return ast.literal_eval(field_str)
    except:
        return field_str


def load_mmlongbench(limit: Optional[int] = None,
                      doc_types: Optional[list[str]] = None,
                      answer_formats: Optional[list[str]] = None,
                      skip_unanswerable: bool = False) -> list[dict]:
    """
    Load MMLongBench-Doc dataset.

    Args:
        limit: Maximum number of samples to load
        doc_types: Filter by document types (e.g., ["Academic paper", "Research report"])
        answer_formats: Filter by answer format (e.g., ["Str", "Int", "Float"])
        skip_unanswerable: Skip "Not answerable" questions

    Returns:
        List of samples with keys: id, doc_id, doc_type, question, answer,
                                   answer_format, evidence_pages, evidence_sources
    """
    raw_data = download_samples()

    samples = []
    for i, item in enumerate(raw_data):
        # Parse fields
        answer = item.get("answer", "")
        answer_format = item.get("answer_format", "Str")

        # Skip unanswerable if requested
        if skip_unanswerable and (answer == "Not answerable" or answer_format == "None"):
            continue

        # Filter by doc_type
        doc_type = item.get("doc_type", "")
        if doc_types and not any(dt in doc_type for dt in doc_types):
            continue

        # Filter by answer_format
        if answer_formats and answer_format not in answer_formats:
            continue

        samples.append({
            "id": i,
            "doc_id": item.get("doc_id", ""),
            "doc_type": doc_type,
            "question": item.get("question", ""),
            "answer": answer,
            "answer_format": answer_format,
            "evidence_pages": parse_field(item.get("evidence_pages", "[]")),
            "evidence_sources": parse_field(item.get("evidence_sources", "[]")),
        })

        if limit and len(samples) >= limit:
            break

    return samples


# Sample data for testing without downloading
SAMPLE_DATA = [
    {
        "id": 0,
        "doc_id": "sample.pdf",
        "doc_type": "Research report",
        "question": "What is the main finding of this report?",
        "answer": "Economic growth increased by 5%",
        "answer_format": "Str",
        "evidence_pages": [1],
        "evidence_sources": ["Pure-text"],
    },
    {
        "id": 1,
        "doc_id": "sample.pdf",
        "doc_type": "Research report",
        "question": "How many countries were surveyed?",
        "answer": "25",
        "answer_format": "Int",
        "evidence_pages": [3],
        "evidence_sources": ["Table"],
    },
]


def load_sample_data() -> list[dict]:
    """Load sample data for testing."""
    return SAMPLE_DATA.copy()


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Loading MMLongBench-Doc (limit={limit})...")
    samples = load_mmlongbench(limit=limit)

    print(f"Loaded {len(samples)} samples\n")

    for s in samples:
        print(f"[{s['id']}] {s['doc_type']}")
        print(f"  Doc: {s['doc_id']}")
        print(f"  Q: {s['question'][:80]}...")
        print(f"  A: {s['answer']} ({s['answer_format']})")
        print(f"  Evidence: pages {s['evidence_pages']}, sources {s['evidence_sources']}")
        print()
