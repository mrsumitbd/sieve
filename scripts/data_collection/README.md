# Data Collection Scripts

Scripts for generating AI-labeled code datasets to train the SIEVE `llm_score` classifier.

## Workflow

### 1. Extract human code via SIEVE

Run SIEVE to collect human-written code from GitHub repos, then export to JSONL. The command below is what was run for the dataset, once per language (Java, JavaScript and C++; the Python data came from a separate pipeline). Set `GITHUB_TOKEN` in the environment first.

```bash
sieve run \
  --language Java \
  --start-date 2024-01-01 \
  --end-date 2025-01-01 \
  --min-stars 500 \
  --max-repos 100 \
  --deduplicate \
  --dedup-threshold 0.8 \
  --output-dir ./sieve_output/java \
  --export-format jsonl
```

For the other languages change `--language` (`JavaScript`, `C++`) and `--output-dir` (`./sieve_output/javascript`, `./sieve_output/cpp`). Everything else was left at its default, so no function or class caps were set; the per-repository and total caps are applied afterwards by `balance_and_convert.py`.

`--deduplicate` and `--dedup-threshold` are passed explicitly: the SIEVE builds that produced this data (v0.1.x and the development builds before it) deduplicated by default, with threshold 0.8. The default is now off. Each run's `manifest.json` records `deduplicate: true` and `dedup_threshold: 0.8`.

**SIEVE version and reproducibility.** The data was produced with a pre-release build, not a tagged release. Each run's `manifest.json` records `"sieve_version": "0.1.0"` and `generated_at` on 2026-03-26, but that version string was hard-coded in the code up to and including v0.1.1, so it does not identify a commit. The v0.1.0 tag was created later and differs from the code of that date, so the exact commit is not recorded; the repository history around 2026-03-26 is the closest reference. Re-running the command with v0.2.0 will not give the same files:

- GitHub has no snapshot, so star counts, repositories and branch contents have changed since the runs.
- v0.2.0 counts contributors exactly and excludes bots, reports what discovery could not cover, and extracts files in a fixed order while skipping vendored and generated directories (`vendor`, `node_modules`, `build`, `dist`, ...).
- If you add `--max-functions` or `--max-classes`, v0.2.0 can allocate the per-repository caps differently, because the Pass 1 counts are now exact (they equal what Pass 2 extracts) and discovery completeness changed.

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