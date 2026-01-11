name	chart_data_extractor
description	Extract structured data from chart images for accurate reasoning

# Chart Data Extractor

Extract data from charts into structured tables before answering questions.

## Process

1. **Extract**: Identify all data points visible in the chart
2. **Table**: Format as structured data (JSON or markdown table)
3. **Reason**: Use the extracted data to answer the question
4. **Answer**: Output only the final answer in required format
