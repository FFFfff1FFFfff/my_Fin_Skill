"""
PDF Retriever Tools - Semantic page retrieval using Voyage AI embeddings.

Finds the most relevant pages for a given question, reducing token cost
and improving accuracy for long PDF documents.

Requirements:
    pip install voyageai numpy
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Cache directory for embeddings
EMBEDDING_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "pdf_embeddings"


def _get_voyage_client():
    """Get Voyage AI client, lazy initialization."""
    try:
        import voyageai
        return voyageai.Client()
    except ImportError:
        raise ImportError(
            "voyageai package not installed. Install with: pip install voyageai"
        )
    except Exception as e:
        if "VOYAGE_API_KEY" in str(e):
            raise ValueError(
                "VOYAGE_API_KEY environment variable not set. "
                "Get your API key from https://www.voyageai.com/"
            )
        raise


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    if HAS_NUMPY:
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    else:
        # Pure Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0


def _get_cache_path(pdf_path: str) -> Path:
    """Get cache file path for a PDF."""
    # Use hash of absolute path as cache key
    path_hash = hashlib.md5(os.path.abspath(pdf_path).encode()).hexdigest()[:16]
    return EMBEDDING_CACHE_DIR / f"{path_hash}.pkl"


def embed_pdf_pages(extracted_text: dict, model: str = "voyage-3") -> dict:
    """
    Create embeddings for all pages of a PDF.

    Args:
        extracted_text: Output from extract_pdf_text(), containing:
            - pages: dict mapping page_num to text content
            - total_pages: int
        model: Voyage AI model to use (default: voyage-3)

    Returns:
        dict with:
            - page_nums: list of page numbers
            - embeddings: list of embedding vectors
            - model: model used
    """
    if "pages" not in extracted_text:
        raise ValueError("extracted_text must contain 'pages' dict")

    pages = extracted_text["pages"]
    if not pages:
        return {"page_nums": [], "embeddings": [], "model": model}

    # Sort pages by number
    page_nums = sorted(pages.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    texts = []

    for page_num in page_nums:
        text = pages[page_num]
        # Truncate very long pages to avoid token limits
        if len(text) > 8000:
            text = text[:8000]
        # Skip empty pages
        if not text.strip():
            text = f"[Page {page_num} - No text content]"
        texts.append(text)

    # Get Voyage client and embed
    client = _get_voyage_client()

    # Batch embed (Voyage handles batching internally)
    try:
        result = client.embed(
            texts,
            model=model,
            input_type="document"
        )
        embeddings = result.embeddings
    except Exception as e:
        raise RuntimeError(f"Voyage AI embedding failed: {e}")

    return {
        "page_nums": page_nums,
        "embeddings": embeddings,
        "model": model,
        "total_pages": len(page_nums)
    }


def find_relevant_pages(
    question: str,
    page_embeddings: dict,
    top_k: int = 5,
    min_score: float = 0.3
) -> list:
    """
    Find the most relevant pages for a question using semantic similarity.

    Args:
        question: The question to find relevant pages for
        page_embeddings: Output from embed_pdf_pages()
        top_k: Number of top pages to return
        min_score: Minimum similarity score to include (0-1)

    Returns:
        List of dicts with:
            - page: page number
            - score: similarity score (0-1)
    """
    if not page_embeddings.get("embeddings"):
        return []

    # Embed the question
    client = _get_voyage_client()
    model = page_embeddings.get("model", "voyage-3")

    try:
        q_result = client.embed(
            [question],
            model=model,
            input_type="query"
        )
        q_embedding = q_result.embeddings[0]
    except Exception as e:
        raise RuntimeError(f"Failed to embed question: {e}")

    # Calculate similarities
    scores = []
    for i, emb in enumerate(page_embeddings["embeddings"]):
        score = _cosine_similarity(q_embedding, emb)
        page_num = page_embeddings["page_nums"][i]
        if score >= min_score:
            scores.append({"page": page_num, "score": round(score, 4)})

    # Sort by score descending
    scores.sort(key=lambda x: -x["score"])

    return scores[:top_k]


def get_or_create_embeddings(
    pdf_path: str,
    extracted_text: dict,
    model: str = "voyage-3",
    force_refresh: bool = False
) -> dict:
    """
    Get cached embeddings or create new ones.

    Args:
        pdf_path: Path to the PDF file (used as cache key)
        extracted_text: Output from extract_pdf_text()
        model: Voyage AI model to use
        force_refresh: If True, ignore cache and recompute

    Returns:
        Embeddings dict (same as embed_pdf_pages output)
    """
    # Ensure cache directory exists
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = _get_cache_path(pdf_path)

    # Try to load from cache
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            # Verify cache is valid
            if (cached.get("model") == model and
                cached.get("total_pages") == len(extracted_text.get("pages", {}))):
                return cached
        except Exception:
            pass  # Cache corrupted, will recompute

    # Compute embeddings
    embeddings = embed_pdf_pages(extracted_text, model=model)

    # Save to cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
    except Exception:
        pass  # Cache write failed, continue without caching

    return embeddings


def get_relevant_page_texts(
    question: str,
    extracted_text: dict,
    page_embeddings: dict,
    top_k: int = 5,
    max_chars_per_page: int = 2000
) -> tuple:
    """
    Convenience function: find relevant pages and return their text content.

    Args:
        question: The question
        extracted_text: Output from extract_pdf_text()
        page_embeddings: Output from get_or_create_embeddings()
        top_k: Number of pages to return
        max_chars_per_page: Max characters per page to include

    Returns:
        tuple: (relevant_pages_info, combined_text)
            - relevant_pages_info: list of {"page": N, "score": S}
            - combined_text: string with all relevant page texts
    """
    relevant = find_relevant_pages(question, page_embeddings, top_k=top_k)

    if not relevant:
        return [], ""

    pages = extracted_text.get("pages", {})
    text_parts = []

    for item in relevant:
        page_num = item["page"]
        # Handle both int and str keys
        page_key = page_num if page_num in pages else str(page_num)
        if page_key in pages:
            text = pages[page_key][:max_chars_per_page]
            text_parts.append(f"[Page {page_num} (relevance: {item['score']:.2f})]:\n{text}")

    combined_text = "\n\n".join(text_parts)
    return relevant, combined_text


# =============================================================================
# Hybrid retrieval: combine keyword search with semantic search
# =============================================================================

def hybrid_retrieve(
    question: str,
    extracted_text: dict,
    page_embeddings: Optional[dict] = None,
    pdf_path: Optional[str] = None,
    top_k: int = 5,
    keyword_threshold: int = 2
) -> tuple:
    """
    Hybrid retrieval: try keyword search first, fall back to semantic.

    Args:
        question: The question
        extracted_text: Output from extract_pdf_text()
        page_embeddings: Pre-computed embeddings (optional)
        pdf_path: Path to PDF for caching (required if embeddings not provided)
        top_k: Number of pages to return
        keyword_threshold: Min keyword matches to trust keyword search

    Returns:
        tuple: (relevant_pages, combined_text, method_used)
    """
    # Try keyword search first (free, fast)
    try:
        from skills.pdf_text_extractor.pdf_tools import search_in_pdf

        # Extract keywords from question
        import re
        stopwords = {
            'what', 'which', 'where', 'when', 'how', 'many', 'much',
            'the', 'and', 'for', 'are', 'this', 'that', 'from', 'with',
            'does', 'did', 'was', 'were', 'have', 'has', 'been', 'being',
            'is', 'in', 'on', 'at', 'to', 'of', 'a', 'an'
        }
        words = re.findall(r'\b[A-Za-z]{3,}\b|\b\d+\.?\d*\b', question)
        keywords = [w for w in words if w.lower() not in stopwords]

        if keywords:
            # Search with multiple keywords
            matches = search_in_pdf(extracted_text, ' '.join(keywords[:5]))

            # Check if we have good keyword matches
            if matches and matches[0].get("count", 0) >= keyword_threshold:
                # Use keyword results
                pages = extracted_text.get("pages", {})
                text_parts = []
                relevant = []

                for match in matches[:top_k]:
                    page_num = match["page"]
                    page_key = page_num if page_num in pages else str(page_num)
                    if page_key in pages:
                        text = pages[page_key][:2000]
                        text_parts.append(f"[Page {page_num} (keyword match)]:\n{text}")
                        relevant.append({"page": page_num, "score": 1.0, "method": "keyword"})

                if text_parts:
                    return relevant, "\n\n".join(text_parts), "keyword"

    except ImportError:
        pass  # pdf_text_extractor not available

    # Fall back to semantic search
    if page_embeddings is None:
        if pdf_path is None:
            raise ValueError("Either page_embeddings or pdf_path must be provided")
        page_embeddings = get_or_create_embeddings(pdf_path, extracted_text)

    relevant, combined_text = get_relevant_page_texts(
        question, extracted_text, page_embeddings, top_k=top_k
    )

    # Add method info
    for item in relevant:
        item["method"] = "semantic"

    return relevant, combined_text, "semantic"
