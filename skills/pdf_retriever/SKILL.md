name	pdf_retriever
description	Semantic page retrieval for PDF documents using Voyage AI embeddings

# PDF Retriever

Semantic search over PDF pages using vector embeddings. Finds the most relevant pages for a given question, reducing token cost and improving accuracy.

---

## Overview

For long PDF documents (50-200+ pages), sending the entire document to Claude is:
- Expensive (100K+ tokens)
- Slow
- Noisy (irrelevant pages dilute attention)

This skill solves this by:
1. Embedding each page with Voyage AI
2. Finding top-K relevant pages via semantic similarity
3. Sending only relevant pages to Claude

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF Document                             │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              │
│  │Page 1│ │Page 2│ │Page 3│ │ ...  │ │Page N│              │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘              │
└─────┼────────┼────────┼────────┼────────┼──────────────────┘
      │        │        │        │        │
      ▼        ▼        ▼        ▼        ▼
┌─────────────────────────────────────────────────────────────┐
│              Voyage AI Embedding (voyage-3)                  │
│     [vec1]   [vec2]   [vec3]   [...]   [vecN]               │
│                    (cached per PDF)                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           │  Question: "What was the revenue in 2023?"
                           │       │
                           │       ▼
                           │  [query_vec]
                           │       │
                           ▼       ▼
┌─────────────────────────────────────────────────────────────┐
│              Cosine Similarity Search                        │
│     Page 3: 0.89  ←── highest                               │
│     Page 7: 0.85                                            │
│     Page 1: 0.72                                            │
│     Page 5: 0.68                                            │
│     Page 12: 0.65                                           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Return pages [3, 7, 1, 5, 12]
```

---

## Cost Comparison

| Approach | Tokens (100-page PDF) | Cost (Claude) | Cost (Voyage) |
|----------|----------------------|---------------|---------------|
| Full PDF | ~100K | ~$0.30 | $0 |
| Top-5 pages | ~5K | ~$0.015 | ~$0.003 |
| **Savings** | **95%** | **95%** | - |

---

## Available Tools

### `embed_pdf_pages(extracted_text: dict) -> dict`

Create embeddings for all pages of a PDF.

**Input:** Output from `extract_pdf_text()`
**Output:**
```python
{
    "page_nums": [1, 2, 3, ...],
    "embeddings": [[0.1, 0.2, ...], [...], ...],
    "model": "voyage-3"
}
```

### `find_relevant_pages(question: str, page_embeddings: dict, top_k: int = 5) -> list`

Find the most relevant pages for a question.

**Output:**
```python
[
    {"page": 3, "score": 0.89},
    {"page": 7, "score": 0.85},
    {"page": 1, "score": 0.72},
    ...
]
```

### `get_or_create_embeddings(pdf_path: str, extracted_text: dict) -> dict`

Get cached embeddings or create new ones.

---

## Caching

Embeddings are cached in `.cache/pdf_embeddings/` using PDF path hash as key.

- First request: ~1-2s (compute embeddings)
- Subsequent requests: ~100ms (load from cache + similarity search)

---

## Configuration

Environment variable required:
```bash
export VOYAGE_API_KEY="your-api-key"
```

Get API key from: https://www.voyageai.com/

---

## Usage Example

```python
from skills.pdf_retriever.retriever_tools import (
    get_or_create_embeddings,
    find_relevant_pages,
)
from skills.pdf_text_extractor.pdf_tools import extract_pdf_text

# Extract text from PDF
extracted_text = extract_pdf_text("document.pdf")

# Get or create embeddings (cached)
embeddings = get_or_create_embeddings("document.pdf", extracted_text)

# Find relevant pages for a question
question = "What was the company's revenue in 2023?"
relevant = find_relevant_pages(question, embeddings, top_k=5)

# Result: [{"page": 15, "score": 0.91}, {"page": 3, "score": 0.87}, ...]
```

---

## Hybrid Retrieval Strategy

For best results, combine with keyword search:

```python
# 1. Try keyword search first (free, fast)
keyword_matches = search_in_pdf(extracted_text, "revenue 2023")

# 2. If no good matches, use semantic search
if not keyword_matches or keyword_matches[0]["count"] < 2:
    relevant = find_relevant_pages(question, embeddings, top_k=5)
```

---

## Limitations

- Requires `VOYAGE_API_KEY` environment variable
- Text-based retrieval (won't help for pure image/chart questions)
- Best for documents with meaningful text content
