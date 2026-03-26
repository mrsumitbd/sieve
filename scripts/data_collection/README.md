# Data Collection Scripts

Scripts for generating AI-labeled code datasets to train the SIEVE `llm_score` classifier.

## Workflow

### 1. Extract human code via SIEVE

Run SIEVE to collect human-written code from GitHub repos, then export to JSONL:

```bash
sieve run \
  --language Java \
  --start-date 2024-06-01 \
  --end-date 2025-01-01 \
  --min-stars 50 \
  --output-dir ./sieve_output/java \
  --export-format jsonl
```

### 2. Convert SIEVE output to generation-ready CSV

SIEVE's JSONL output maps directly to the CSV schema expected by the generation script:

| SIEVE field   | CSV column (function)      | CSV column (class)    |
|---------------|----------------------------|-----------------------|
| `repo`        | `repository_name`          | `repository_name`     |
| `file_path`   | `func_path_in_repository`  | `file_path`           |
| `func_name`   | `func_name`                | —                     |
| `class_name`  | —                          | `class_name`          |
| `source_code` | `complete_extracted_code`  | `human_written_code`  |
| `signature`   | `func_signature`           | —                     |
| `skeleton`    | —                          | `class_skeleton`      |

### 3. Generate LLM code

```bash
pip install -r scripts/data_collection/requirements.txt

# Function level
python scripts/data_collection/generate_llm_code.py \
  --input   sieve_output/java_functions.csv \
  --output  data/generated/deepseek_java_func.csv \
  --model   deepseek-ai/DeepSeek-V3 \
  --language Java \
  --granularity function

# Class level
python scripts/data_collection/generate_llm_code.py \
  --input   sieve_output/java_classes.csv \
  --output  data/generated/deepseek_java_class.csv \
  --model   deepseek-ai/DeepSeek-V3 \
  --language Java \
  --granularity class
```

**Supported models (Together.ai):**

| Model | ID |
|---|---|
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` |
| Qwen2.5-Coder 7B | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| GPT-OSS 20B | `gpt-oss-20b` |

The `--mode` flag controls batch vs. sequential:
- `auto` (default): batch if >500 rows, else sequential
- `batch`: always use Together.ai batch API (cheaper, async)
- `sequential`: one request at a time (easier to debug)

### 4. Clean generated code

```bash
python scripts/data_collection/clean_generated_code.py \
  --input      data/generated/deepseek_java_func.csv \
  --language   Java \
  --granularity function
```

This adds `generated_code_cleaned` in-place — the first parseable function/class
block extracted from the raw LLM output, with markdown fences and prose stripped.

## Environment

Set `TOGETHER_API_KEY` in a `.env` file at the project root:

```
TOGETHER_API_KEY=your_key_here
```

## Output schema

Each generated CSV matches the schema of the existing Python training data:

**Function level:**
```
repository_name, func_path_in_repository, func_name,
complete_extracted_code, func_signature,
generated_code, generated_code_cleaned
```

**Class level:**
```
repository_name, file_path, class_name,
human_written_code, class_skeleton,
generated_code, generated_code_cleaned
```