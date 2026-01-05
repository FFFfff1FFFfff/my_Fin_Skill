name	web-search-tool
description	Web search capability for retrieving up-to-date information from the internet

# Web Search Tool

This skill provides web search capabilities to retrieve current information.

---

## When to Use

Use web search when:
- Question asks about recent events (after your knowledge cutoff)
- Question requires current/real-time data (prices, scores, weather)
- Question asks "who currently...", "what is the latest...", etc.
- You're uncertain about facts that may have changed

---

## Search Strategy

### Step 1: Identify Search Need

Determine if the question requires:
- **Current facts**: Use search
- **Historical facts**: May not need search
- **Time-sensitive info**: Definitely use search

### Step 2: Formulate Queries

Create effective search queries:
- Use specific keywords
- Include relevant dates/years if applicable
- Try multiple query variations if needed

**Examples:**
| Question | Good Query |
|----------|-----------|
| "Who is the current CEO of OpenAI?" | `OpenAI CEO 2025` |
| "What was the score of yesterday's Lakers game?" | `Lakers game score January 2025` |

### Step 3: Evaluate Results

When reviewing search results:
1. Check source reliability (official sites > blogs)
2. Check date of information
3. Look for consensus across multiple sources
4. Note any conflicting information

---

## Available Tools

### `web_search(query: str) -> str`

Searches the web and returns relevant results.

**Parameters:**
- `query`: Search query string

**Returns:**
- Search results with titles, snippets, and URLs

**Usage:**
```
Query: "OpenAI CEO 2025"
Returns: [relevant search results]
```

### `tavily_search(query: str) -> str` (if API key configured)

Alternative search using Tavily API for more detailed results.

---

## Best Practices

1. **Search first, answer second** - Get current info before answering
2. **Multiple queries** - Try different phrasings if first search is unhelpful
3. **Verify sources** - Cross-check important facts
4. **Cite sources** - Mention where you found information
5. **Acknowledge uncertainty** - If sources conflict, say so

---

## Output Format

After searching, structure your answer as:

```
Based on my search results:

[Your answer here]

Sources:
- [Source 1]
- [Source 2]
```
