name	conflicting_info_reasoner
description	Framework for handling conflicting, noisy, or uncertain information from multiple sources

# Conflicting Information Reasoner

A systematic approach to handling contradictory or uncertain information.

---

## Core Principle

**Search results often conflict. Your job is to find truth through careful reasoning, not just report what you find.**

---

## Step-by-Step Process

### STEP 1: Search with Multiple Queries

Don't rely on a single search. Try variations:

```
Original: "Who is CEO of OpenAI?"
Variations:
- "OpenAI CEO 2025"
- "OpenAI leadership current"
- "Sam Altman OpenAI"
```

### STEP 2: Categorize Sources

Rate source reliability:

| Source Type | Reliability | Examples |
|-------------|-------------|----------|
| **Official** | Highest | Company websites, government sites |
| **News (Major)** | High | Reuters, AP, major newspapers |
| **News (Other)** | Medium | Blogs, smaller outlets |
| **Social Media** | Low | Twitter, Reddit |
| **Outdated** | Very Low | Articles > 6 months old for current events |

### STEP 3: Detect Conflicts

Look for:
- **Direct contradictions**: Source A says X, Source B says Y
- **Temporal conflicts**: Old info vs new info
- **Partial information**: Sources have different pieces

**Example:**
```
Source 1 (2024): "CEO is Sam Altman"
Source 2 (2023): "CEO is Sam Altman" [briefly fired then returned]
Source 3 (2024): "Sam Altman leads OpenAI"
→ Consensus: Sam Altman is CEO
```

### STEP 4: Resolve Conflicts

Use this decision framework:

```
IF official source available:
    → Trust official source
ELSE IF recent sources agree:
    → Trust consensus of recent sources
ELSE IF sources conflict on facts:
    → Report the conflict, give best judgment
ELSE IF information is genuinely uncertain:
    → Say "I cannot determine with certainty"
```

### STEP 5: Formulate Answer

**If confident:**
```
Based on [sources], the answer is [X].
```

**If partially confident:**
```
The most likely answer is [X], based on [sources].
However, [caveat if any].
```

**If genuinely uncertain:**
```
I cannot determine with certainty. Sources indicate [X] or [Y].
The most reliable source suggests [X].
```

---

## Handling Common Scenarios

### Scenario 1: Outdated vs Current Info

```
Question: "Who is the current CEO of X?"

Old source (2022): "John Smith"
New source (2024): "Jane Doe"

→ Trust newer source, answer "Jane Doe"
→ Verify with official company page if possible
```

### Scenario 2: Multiple Conflicting Sources

```
Question: "What is X's population?"

Source A: 1.2 million
Source B: 1.3 million
Source C: 1.25 million

→ Sources roughly agree (~1.2-1.3M)
→ Answer: "approximately 1.2-1.3 million"
```

### Scenario 3: No Reliable Information

```
Question: "What is X?" (obscure topic)

Sources: None reliable or all outdated

→ Admit uncertainty
→ Answer: "I could not find reliable current information about X"
```

---

## Red Flags to Watch For

1. **Single source** - Never trust just one source for important claims
2. **No dates** - Information without dates may be outdated
3. **Circular sources** - Multiple articles citing the same original source
4. **Contradictions within source** - Internal inconsistency = unreliable
5. **Too-specific numbers from unreliable sources** - May be fabricated

---

## Output Format

Structure your reasoning:

```
Search findings:
- [Source 1]: [Key info] (reliability: high/medium/low)
- [Source 2]: [Key info] (reliability: high/medium/low)

Analysis:
- [Note any conflicts or consensus]

Conclusion:
[Your answer with appropriate confidence level]
```

---

## Key Reminders

- **It's OK to say "I don't know"** - Better than being wrong
- **Hedge appropriately** - "likely", "appears to be", "sources suggest"
- **Cite your reasoning** - Show why you trust certain sources
- **Update if contradicted** - Don't stick to wrong answers
