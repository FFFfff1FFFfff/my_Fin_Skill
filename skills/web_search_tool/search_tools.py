"""Web search tools for SealQA - supports multiple search backends"""

import os
import json
from typing import Optional


def web_search_builtin(query: str) -> str:
    """
    Search using Claude's built-in web search (via WebSearch tool).
    This is a placeholder - actual implementation uses Claude's native capability.

    Args:
        query: Search query string

    Returns:
        Search results as formatted string
    """
    # This function is called by the skill system but actual search
    # is handled by Claude's native WebSearch tool
    return f"[Search query: {query}] - Use Claude's WebSearch tool for actual results"


def tavily_search(query: str, api_key: Optional[str] = None) -> str:
    """
    Search using Tavily API.

    Args:
        query: Search query string
        api_key: Tavily API key (or from TAVILY_API_KEY env var)

    Returns:
        Search results as formatted string
    """
    api_key = api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not configured"

    try:
        import requests
    except ImportError:
        return "Error: requests library not installed"

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 5
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Format results
        results = []
        if data.get("answer"):
            results.append(f"Summary: {data['answer']}\n")

        for r in data.get("results", [])[:5]:
            results.append(f"- {r.get('title', 'N/A')}")
            results.append(f"  {r.get('content', '')[:200]}...")
            results.append(f"  URL: {r.get('url', '')}\n")

        return "\n".join(results) if results else "No results found"

    except Exception as e:
        return f"Search error: {str(e)}"


def serper_search(query: str, api_key: Optional[str] = None) -> str:
    """
    Search using Serper API (Google Search).

    Args:
        query: Search query string
        api_key: Serper API key (or from SERPER_API_KEY env var)

    Returns:
        Search results as formatted string
    """
    api_key = api_key or os.environ.get("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not configured"

    try:
        import requests
    except ImportError:
        return "Error: requests library not installed"

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Format results
        results = []

        # Answer box if available
        if data.get("answerBox"):
            ab = data["answerBox"]
            if ab.get("answer"):
                results.append(f"Answer: {ab['answer']}\n")
            elif ab.get("snippet"):
                results.append(f"Snippet: {ab['snippet']}\n")

        # Organic results
        for r in data.get("organic", [])[:5]:
            results.append(f"- {r.get('title', 'N/A')}")
            results.append(f"  {r.get('snippet', '')[:200]}")
            results.append(f"  URL: {r.get('link', '')}\n")

        return "\n".join(results) if results else "No results found"

    except Exception as e:
        return f"Search error: {str(e)}"


def search(query: str, backend: str = "tavily") -> str:
    """
    Unified search interface.

    Args:
        query: Search query string
        backend: "tavily", "serper", or "builtin"

    Returns:
        Search results as formatted string
    """
    if backend == "tavily":
        return tavily_search(query)
    elif backend == "serper":
        return serper_search(query)
    else:
        return web_search_builtin(query)


if __name__ == "__main__":
    # Test search
    print("Testing search tools...")
    print(search("OpenAI CEO 2025", backend="builtin"))
