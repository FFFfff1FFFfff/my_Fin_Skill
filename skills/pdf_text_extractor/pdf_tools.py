"""PDF text extraction and search tools for document QA."""

import re
from typing import Optional

# Try to import PDF libraries
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


def extract_pdf_text(pdf_path: str) -> dict:
    """
    Extract all text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        dict with keys:
        - total_pages: Number of pages
        - pages: Dict mapping page number (1-indexed) to text content
        - full_text: All text concatenated
    """
    pages = {}

    # Try pdfplumber first (better table handling)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    pages[i] = text
        except Exception as e:
            return {"error": f"pdfplumber failed: {str(e)}"}

    # Fallback to pypdf
    elif HAS_PYPDF:
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    pages[i] = text
        except Exception as e:
            return {"error": f"pypdf failed: {str(e)}"}

    else:
        return {"error": "No PDF library available. Install pypdf or pdfplumber."}

    full_text = "\n\n".join(f"[Page {i}]\n{text}" for i, text in sorted(pages.items()))

    return {
        "total_pages": len(pages),
        "pages": pages,
        "full_text": full_text
    }


def search_in_pdf(extracted_text: dict, query: str, context_chars: int = 100) -> list:
    """
    Search for a query string across all pages of extracted text.

    Args:
        extracted_text: Output from extract_pdf_text()
        query: Search string (case-insensitive)
        context_chars: Number of characters to show around each match

    Returns:
        List of dicts with keys: page, snippet, count
    """
    if "error" in extracted_text:
        return [{"error": extracted_text["error"]}]

    results = []
    query_lower = query.lower()

    for page_num, text in extracted_text.get("pages", {}).items():
        text_lower = text.lower()

        # Count occurrences
        count = text_lower.count(query_lower)

        if count > 0:
            # Find first match and extract context
            idx = text_lower.find(query_lower)
            start = max(0, idx - context_chars)
            end = min(len(text), idx + len(query) + context_chars)

            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."

            results.append({
                "page": page_num,
                "snippet": snippet.strip(),
                "count": count
            })

    # Sort by count (most matches first)
    results.sort(key=lambda x: x["count"], reverse=True)

    return results


def get_page_text(extracted_text: dict, page_num: int) -> str:
    """
    Get the full text content of a specific page.

    Args:
        extracted_text: Output from extract_pdf_text()
        page_num: Page number (1-indexed)

    Returns:
        Text content of the specified page, or error message
    """
    if "error" in extracted_text:
        return f"Error: {extracted_text['error']}"

    pages = extracted_text.get("pages", {})

    if page_num not in pages:
        return f"Error: Page {page_num} not found. Document has {len(pages)} pages."

    return pages[page_num]


def find_tables_in_page(extracted_text: dict, page_num: int) -> list:
    """
    Attempt to identify table-like structures in a page.

    Args:
        extracted_text: Output from extract_pdf_text()
        page_num: Page number (1-indexed)

    Returns:
        List of potential table sections found
    """
    text = get_page_text(extracted_text, page_num)

    if text.startswith("Error:"):
        return []

    # Simple heuristic: lines with multiple tab/space-separated values
    tables = []
    lines = text.split('\n')

    table_lines = []
    for line in lines:
        # Check if line looks like a table row (multiple space-separated values)
        parts = re.split(r'\s{2,}|\t', line.strip())
        if len(parts) >= 3:
            table_lines.append(line)
        else:
            if len(table_lines) >= 2:
                tables.append('\n'.join(table_lines))
            table_lines = []

    if len(table_lines) >= 2:
        tables.append('\n'.join(table_lines))

    return tables


def summarize_document(extracted_text: dict, max_chars_per_page: int = 500) -> str:
    """
    Create a summary of the document structure.

    Args:
        extracted_text: Output from extract_pdf_text()
        max_chars_per_page: Max characters to show per page preview

    Returns:
        Document summary string
    """
    if "error" in extracted_text:
        return f"Error: {extracted_text['error']}"

    pages = extracted_text.get("pages", {})
    total = extracted_text.get("total_pages", 0)

    summary_parts = [f"Document Summary ({total} pages):", "=" * 40]

    for page_num in sorted(pages.keys()):
        text = pages[page_num]
        preview = text[:max_chars_per_page].strip()
        if len(text) > max_chars_per_page:
            preview += "..."

        summary_parts.append(f"\n[Page {page_num}]")
        summary_parts.append(preview)

    return '\n'.join(summary_parts)


# Export functions for skill system
__all__ = [
    'extract_pdf_text',
    'search_in_pdf',
    'get_page_text',
    'find_tables_in_page',
    'summarize_document'
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_tools.py <pdf_path> [search_query]")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"Extracting text from: {pdf_path}")
    result = extract_pdf_text(pdf_path)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"Total pages: {result['total_pages']}")
    print("\n" + summarize_document(result, 200))

    if len(sys.argv) >= 3:
        query = sys.argv[2]
        print(f"\n\nSearching for: '{query}'")
        matches = search_in_pdf(result, query)
        for match in matches:
            print(f"\nPage {match['page']} ({match['count']} matches):")
            print(f"  {match['snippet']}")
