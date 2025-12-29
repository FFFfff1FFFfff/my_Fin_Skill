# FinQA Skills Evaluation

Benchmark comparing Claude's baseline vs skill-augmented performance on financial question answering.



## Results


Using the first 100 samples from the FinQA test dataset:
 

- **Baseline**: 63% accuracy

- **With Skills**: 72% accuracy

 

## Dataset


- **Source**: FinQA test set (first 100 samples)

- **Tasks**: Financial QA requiring percentage calculations, changes, multi-step arithmetic, and table data extraction

 

## Skills


### 1. finqa-reasoning


6-step systematic reasoning framework for financial questions:

- Question type identification

- Data location and extraction

- Data verification (units, time periods)

- Formula selection

- Calculation

- Answer formatting

 

### 2. formula-code-assistant


Three calculation tools for complex computations:

- `generate_calculation_formula()` - Step-by-step formula breakdowns

- `generate_python_code()` - Executable Python code generation

- `execute_calculation()` - Python expression execution

 


## Methodology


1. **Baseline**: Direct prompting with minimal guidance

2. **With Skills**: Claude receives skill documentation + access to calculation tools via Anthropic API

3. **Evaluation**: LLM judge compares predictions to ground truth

 

**LLM Judge Caveat**: Automated judge struggles with decimal precision, percentage formatting, and rounding variations. Results were manually verified for accuracy.

 

## Installation

 

```bash

pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-api-key"

```

 

## Usage

 

```bash

# Test samples 1-10 (default)

python compare_with_tools.py



# Test 50 samples

python compare_with_tools.py --start 0 --limit 50

```

 

**Options**:

- `--dataset`: Dataset path (default: `finqa_test.json`)

- `--start`: Start index, 0-based (default: `0`)

- `--end`: End index, exclusive (default: `10`)

- `--limit`: Number of samples (alternative to `--end`)

 

## Output Files

 

- `results_baseline_{range}.txt` - Baseline predictions

- `results_with_skills_{range}.txt` - Skills predictions with reasoning traces

 

## Project Structure

 

```

my_Fin_Skill/

├── skills/

│   ├── finqa-reasoning/SKILL.md

│   └── formula-code-assistant/

│       ├── SKILL.md

│       └── calculation_tools.py

├── compare_with_tools.py        # Main evaluation script

├── skill_system.py              # Skill loader and manager

└── finqa_test.json              # Dataset
