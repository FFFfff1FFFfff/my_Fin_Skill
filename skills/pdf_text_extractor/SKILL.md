name	pdf_text_extractor
description	Extract and search text from PDF documents to assist with text-based questions and cross-page evidence finding

# PDF Text Extractor

A tool for extracting and searching text content from PDF documents. Particularly useful for:
- **Pure-text questions** where OCR/text extraction is more reliable than visual understanding
- **Cross-page evidence finding** by searching across all pages
- **Locating relevant pages** before detailed visual analysis

---

## When to Use This Skill

### ✅ Use for:
- Questions asking about specific text, names, dates, or numbers mentioned in the document
- Finding which pages contain relevant information
- Cross-page reasoning where evidence is scattered
- Questions about document structure (sections, references, etc.)

### ❌ Don't use for:
- Questions about charts, graphs, or visual diagrams
- Questions requiring interpretation of images or figures
- Layout-dependent questions (e.g., "what's in the top-right corner")

---

## Available Tools

### 1. `extract_pdf_text(pdf_path: str) -> dict`

Extract all text content from a PDF file.

**Parameters:**
- `pdf_path`: Path to the PDF file

**Returns:**
```python
{
    "total_pages": 10,
    "pages": {
        1: "Text content from page 1...",
        2: "Text content from page 2...",
        ...
    },
    "full_text": "All text concatenated..."
}
```

**Example:**
```python
result = extract_pdf_text("/path/to/document.pdf")
print(f"Document has {result['total_pages']} pages")
print(f"Page 1 content: {result['pages'][1][:200]}...")
```

### 2. `search_in_pdf(extracted_text: dict, query: str) -> list`

Search for a query string across all pages of extracted text.

**Parameters:**
- `extracted_text`: Output from `extract_pdf_text()`
- `query`: Search string (case-insensitive)

**Returns:**
```python
[
    {"page": 1, "snippet": "...context around the match...", "count": 2},
    {"page": 5, "snippet": "...another match...", "count": 1},
]
```

**Example:**
```python
matches = search_in_pdf(extracted_text, "revenue growth")
for match in matches:
    print(f"Page {match['page']}: {match['snippet']}")
```

### 3. `get_page_text(extracted_text: dict, page_num: int) -> str`

Get the full text content of a specific page.

**Parameters:**
- `extracted_text`: Output from `extract_pdf_text()`
- `page_num`: Page number (1-indexed)

**Returns:** Text content of the specified page

---

## Workflow Example

**Question:** "What year was the company founded according to the report?"

**Step 1: Extract text**
```python
doc = extract_pdf_text("annual_report.pdf")
```

**Step 2: Search for relevant terms**
```python
matches = search_in_pdf(doc, "founded")
# Returns: [{"page": 3, "snippet": "The company was founded in 1985..."}]
```

**Step 3: Get full page context if needed**
```python
page_text = get_page_text(doc, 3)
# Read full context to confirm answer
```

**Step 4: Answer**
```
Final Answer: 1985
```

---

## Integration with Visual Analysis

For questions that may need both text and visual understanding:

1. **First**: Use text extraction to locate relevant pages
2. **Then**: Use visual understanding for charts/figures on those pages
3. **Finally**: Combine evidence from both sources

This hybrid approach is especially effective for:
- Tables (text extraction for data, visual for structure)
- Reports with mixed content (text + charts)
- Cross-page questions requiring multiple evidence types

---

## Notes

- Text extraction works best on native PDFs (not scanned images)
- For scanned documents, OCR quality may vary
- Page numbers are 1-indexed (matching PDF page numbers)
- Search is case-insensitive
