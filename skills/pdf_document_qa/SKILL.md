name	pdf_document_qa
description	Comprehensive framework for answering questions from PDF documents with systematic evidence location and format-aware extraction

# PDF Document QA Framework

A systematic approach to answering questions from PDF documents, handling various document types, evidence sources, and answer formats.

---

## Document Types & Reading Strategies

| Document Type | Key Sections | Reading Focus |
|--------------|--------------|---------------|
| **Academic Paper** | Abstract, Methods, Results, Figures | Focus on figures/tables for data |
| **Research Report** | Executive Summary, Charts, Key Findings | Charts often contain answers |
| **10-K / Financial Report** | Tables, Notes, MD&A | Precise numbers in tables |
| **Manual / Guide** | TOC, Sections, Steps | Follow structure for specific info |
| **Brochure** | Headlines, Callouts, Infographics | Visual elements are key |

---

## Step-by-Step QA Process

### STEP 1: Understand the Question

Identify:
- **What** is being asked (specific value, name, list, yes/no)
- **Where** to look (which section, page, element)
- **Format** expected (number, text, list)

### STEP 2: Locate Evidence

Scan for relevant content:
```
1. Check document structure (TOC, headers, sections)
2. Look for keywords from the question
3. Identify relevant pages/sections
4. Note evidence type: text, table, chart, or figure
```

### STEP 3: Extract Information

Based on evidence source:

**From Pure Text:**
- Find exact quotes or paraphrased information
- Note the page number for verification

**From Tables:**
- Identify correct row and column
- Check headers and units
- Handle merged cells carefully

**From Charts/Figures:**
- Read axis labels and legends
- Extract data points or trends
- Note if values are approximate

### STEP 4: Formulate Answer

Match the expected format:

| Format | How to Answer |
|--------|---------------|
| **Int** | Exact integer, no decimals (e.g., `42`) |
| **Float** | Number with decimals if needed (e.g., `3.14`) |
| **Str** | Concise text, as short as possible |
| **List** | Format as `['item1', 'item2', 'item3']` |
| **None** | Reply `Not answerable` if info not in document |

---

## Handling "Not Answerable" Cases

A question is **Not answerable** when:

1. ❌ The information is simply not in the document
2. ❌ The question asks about a time period not covered
3. ❌ The question requires external knowledge not in the PDF
4. ❌ The document discusses the topic but lacks the specific detail asked

**DO NOT guess** - if you cannot find explicit evidence, answer "Not answerable"

**Warning signs:**
- You're making assumptions not supported by text
- You're extrapolating beyond what's stated
- You find related but not exact information

---

## Visual Element Extraction

### Reading Charts

```
1. Identify chart type (bar, line, pie, etc.)
2. Read axis labels and units
3. Check legend for data series
4. Extract specific values asked about
5. For approximations, give best estimate
```

### Reading Tables

```
1. Understand header row/column structure
2. Locate intersection of relevant row and column
3. Check for footnotes or special notations
4. Handle merged cells by tracing to headers
5. Note units (millions, thousands, %)
```

### Reading Figures/Diagrams

```
1. Read figure caption for context
2. Identify labeled elements
3. Follow arrows/connections if present
4. Extract specific requested information
```

---

## Answer Format Examples

**Integer (Int):**
- Q: "How many countries were surveyed?"
- A: `25`

**Float:**
- Q: "What is the percentage of users?"
- A: `73.5`

**String (Str):**
- Q: "What is the main conclusion?"
- A: `Economic growth increased due to policy changes`

**List:**
- Q: "What colors appear in the diagram?"
- A: `['red', 'blue', 'green']`

**Not Answerable:**
- Q: "What was the 2025 revenue?" (doc only covers 2020-2023)
- A: `Not answerable`

---

## Common Pitfalls to Avoid

1. **Wrong page/section** - Verify you're looking at the right part
2. **Misreading charts** - Double-check axis scales and units
3. **Incomplete lists** - Make sure you captured all items
4. **Guessing** - If uncertain, say "Not answerable"
5. **Wrong format** - Int vs Float vs Str matters for evaluation

---

## Self-Check Before Answering

- [ ] Did I find explicit evidence in the document?
- [ ] Is my answer in the correct format?
- [ ] Did I extract the exact value (not approximate when exact is available)?
- [ ] For lists, did I include all items?
- [ ] Am I confident, or should I say "Not answerable"?

---

## Output Format

Always end your response with:

```
Final Answer: [your answer]
```

Keep the answer as concise as possible - just the value, name, or list.
