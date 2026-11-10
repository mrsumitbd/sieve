---
title: SIEVE
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🔬 SIEVE
**Software Ingestion & Extraction for Verifiable Evaluation**

[![Coverage](https://codecov.io/gh/mrsumitbd/sieve/graph/badge.svg?token=19aecee5-afb0-4d67-9c0d-bdab723ce8d3)](https://codecov.io/gh/mrsumitbd/sieve)
[![Tests](https://github.com/mrsumitbd/sieve/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mrsumitbd/sieve/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Languages](https://img.shields.io/badge/languages-Python%20%7C%20Java%20%7C%20JavaScript%20%7C%20C%2B%2B-orange?style=flat-square)

SIEVE is a parameterized GitHub corpus builder for software engineering research. It lets you curate high-quality code datasets from public repositories with full control over language, recency, repository quality, and extraction granularity — for LLM evaluation, dependency analysis, structural studies of AI-generated code, or general-purpose corpus building.

**Live demo:** https://mrahman2025-sieve.hf.space

**Demo video:** https://youtu.be/1Imgn7kpkNg — recorded with v0.1.1; the web UI has since changed (granularity checkboxes, a single **Export Full AST** toggle, and caps where `0` means no cap), so to reproduce the video exactly, install tag `v0.1.1`:

```bash
git clone --branch v0.1.1 https://github.com/mrsumitbd/sieve.git && cd sieve
pip install -r requirements.txt
streamlit run src/sieve/ui/Home.py
```

> **Note:** the public Hugging Face Space has 8 GB of RAM. Do not run large corpora there; [run the web UI locally](#run-the-web-ui-locally) or use the CLI instead.

---

## Why SIEVE?

Researchers building code corpora today either write one-off scraping scripts per study or reuse static, pre-built datasets like CodeSearchNet and The Stack, which offer no control over composition and can overlap with the training data of the LLMs being evaluated. SIEVE is reusable infrastructure instead: point it at GitHub with the parameters your study needs (language, granularity, repository quality, creation date) and it builds a richly-annotated corpus on demand, with the creation-date window under your control.

One application is **contamination-aware LLM evaluation**: restricting discovery to repositories created after a chosen date (for example, a model's release date) makes the corpus temporally bounded relative to that model. This bounds repository age only. SIEVE does not verify the origin of individual snippets, and a new repository can contain code copied from older sources, so this reduces, but does not rule out, overlap with training data. The same pipeline also supports dependency-management research (every record's package manifest is parsed automatically), structural analysis of LLM-generated code (via the built-in AI-generation classifier and AST export), and general-purpose corpus building for any language-specific study.

---

## Installation

### Prerequisites

**Python 3.11+** is required. Check your version with `python --version`.

**cloc** (Count Lines of Code) is optional. The current pipeline does not need it, because the engineered filter does not count lines of code. Install commands are listed below for completeness.

| Platform | Command |
|---|---|
| macOS | `brew install cloc` |
| Ubuntu / Debian | `sudo apt install cloc` |
| Windows (Chocolatey) | `choco install cloc` |
| Windows (winget) | `winget install AlDanial.Cloc` |
| Manual | [github.com/AlDanial/cloc/releases](https://github.com/AlDanial/cloc/releases) |

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/mrsumitbd/sieve.git
cd sieve
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**3. Install SIEVE and its dependencies**

Runtime only (the first line of `requirements.txt` is `-e .`, so this also installs the `sieve` command; run it from the repository root):
```bash
pip install -r requirements.txt
```

For development and testing:
```bash
pip install -r requirements-dev.txt
```

Check the install:
```bash
sieve --help
```
If `sieve: command not found`, activate the virtual environment, or run `python -m sieve.cli --help`.

On an Intel Mac (x86_64), PyTorch has no wheels after 2.2.2, which needs extra pins. See *Install troubleshooting* on the Documentation page.

**4. Set up your GitHub token**

Generate a classic token at [github.com/settings/tokens](https://github.com/settings/tokens) with the `public_repo` scope. For the CLI, either export it or put it in `.env`:
```bash
export GITHUB_TOKEN=<your token>       # or: cp .env.example .env and fill in GITHUB_TOKEN
```
The CLI also reads `HF_TOKEN` (optional, only for `--annotate-llm-score`). The web UI reads neither variable: paste the GitHub token into its sidebar.

---

## Usage

### Web Interface

#### Run the web UI locally

Repository processing (cloning and extraction) is sequential, so a large run takes a long time and a lot of memory. Run large jobs on your own machine, not on the 8 GB public Space. The full guide is on the Documentation page.

```bash
git clone https://github.com/mrsumitbd/sieve.git && cd sieve   # Python 3.11+
python -m venv .venv && source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/sieve/ui/Home.py
```

Open `http://localhost:8501`, paste a GitHub token into the sidebar (recommended), configure the parameters and click **▶ Run SIEVE**. Stop the server with `Ctrl+C` in the terminal.

### CLI

```bash
sieve run --language Python --start-date 2024-01-01 --end-date 2025-01-01 \
          --min-stars 50 --min-contributors 5 --granularity function --deduplicate

# From config file
sieve run --config my_config.json

# Validate config without running
sieve validate-config my_config.json
```

By default a GitHub rate limit that will not clear within about a minute stops the run with an error. Add `--wait-for-rate-limit` to sleep until it resets and carry on (the web UI never waits).

### Python API

`run_pipeline(config)` takes a `SIEVEConfig` and returns a summary dictionary (counts, `output_paths`, `failed_repos`, per-repository metadata). It takes an optional `progress_callback(message, current, total)`. The CLI and the web UI call nothing else. The next section shows a complete listing.

---

## Worked Example

A researcher is building a corpus of engineered Python repositories created between 2024-06-01 and 2025-06-01, with at least 50 stars and 5 contributors. The corpus is limited to 20 repositories, 30 functions and 30 classes, and each record gets an AI-generated-code score. The three forms below build the same configuration. Deduplication is off unless you ask for it (`--deduplicate`), so it is off here; the other options keep their defaults (both granularities, JSONL output).

**1. CLI**

```bash
export GITHUB_TOKEN=<your token>       # or put it in .env
sieve run --language Python \
          --start-date 2024-06-01 --end-date 2025-06-01 \
          --min-stars 50 --min-contributors 5 \
          --max-repos 20 --max-functions 30 --max-classes 30 \
          --engineered-only --annotate-llm-score \
          --output-dir ./my_corpus
```

**2. JSON config for `--config`**

`my_config.json`:
```json
{
  "language": "Python",
  "start_date": "2024-06-01",
  "end_date": "2025-06-01",
  "min_stars": 50,
  "min_contributors": 5,
  "max_repos": 20,
  "max_functions": 30,
  "max_classes": 30,
  "engineered_only": true,
  "annotate_llm_score": true,
  "output_dir": "./my_corpus"
}
```
```bash
sieve validate-config my_config.json   # checks the file and prints the full configuration
sieve run --config my_config.json      # with --config, every other flag is ignored
```
`end_date` is required in a config file (the CLI defaults it to 30 days before today). `GITHUB_TOKEN` is read from the environment, so the token does not have to be written in the file.

**3. Python API**

```python
import os
from datetime import date

from sieve.config import SIEVEConfig
from sieve.pipeline import run_pipeline

config = SIEVEConfig(
    language="Python",
    start_date=date(2024, 6, 1),   # repository creation lower bound
    end_date=date(2025, 6, 1),     # repository creation upper bound
    min_stars=50,
    min_contributors=5,
    max_repos=20,
    max_functions=30,
    max_classes=30,
    engineered_only=True,
    annotate_llm_score=True,
    output_dir="./my_corpus",
    github_token=os.environ.get("GITHUB_TOKEN"),   # the API does not read the environment by itself
)

summary = run_pipeline(config)
print(summary["total_functions"])
print(summary["output_paths"])
```

The run writes `functions.jsonl`, `classes.jsonl` and `manifest.json` to `./my_corpus`; `output_paths` maps each written file to its path. The engineered filter runs after the repository cap, so fewer than 20 repositories may remain. The classifier weights are downloaded from the Hugging Face Hub on the first scored run (see below).

---

## AI-Generated-Code Score

`annotate_llm_score` adds `llm_score` to each record: the estimated probability that the snippet is AI-generated, from a fine-tuned [CodeBERT](https://huggingface.co/microsoft/codebert-base) classifier. The classifier weights are hosted on the Hugging Face Hub at **[mrahman2025/sieve-llm-classifier](https://huggingface.co/mrahman2025/sieve-llm-classifier)** (`best_model.pt`). They are downloaded on first use and cached; if `src/sieve/models/artifacts/best_model.pt` exists it is used instead.

`llm_score` is a probabilistic signal, not evidence of authorship. The classifier was trained on code from nine LLMs and evaluated on a held-out set that shares its generators with the training data, so that evaluation does not show how the score behaves on code from other LLMs. See the **About the Classifier** page and the Documentation page (*AI-Generated-Code Score*) for how scoring runs and its limits.

---

## Parameters

"CLI" is `sieve run`; "config" is a JSON file passed to `--config` or a `SIEVEConfig` object. Defaults differ between interfaces where noted. The Documentation page has a table that maps each parameter to its CLI flag, config key and web UI control.

| Config key | CLI flag | Default | Description |
|---|---|---|---|
| `language` | `--language` | required; web UI: `Python` | `Python`, `Java`, `JavaScript`, `C++` |
| `start_date` | `--start-date` | required; web UI: 2024-01-01 | Earliest repository **creation** date (`YYYY-MM-DD`). Bounds repository age; it does not verify the origin of each snippet |
| `end_date` | `--end-date` | required in config; CLI and web UI: 30 days before today | Latest repository creation date; on or after `start_date` |
| `min_stars` | `--min-stars` | CLI/config: 10; web UI: 50 | Minimum GitHub stars |
| `min_contributors` | `--min-contributors` | CLI/config: 1; web UI: 5 | Minimum contributors, at least 1. Bots are not counted |
| `max_repos` | `--max-repos` | none (no cap); web UI: 20 | Cap on repositories to process. Omit it (`null` in a config file) for no cap, otherwise it must be at least 1. Web UI: 0 means no cap |
| `max_functions` | `--max-functions` | none; web UI: 30 | Cap on total extracted functions, stratified by repository. Same rules as `max_repos` |
| `max_classes` | `--max-classes` | none; web UI: 30 | Cap on total extracted classes, stratified by repository. Same rules as `max_repos` |
| `granularity` | `--granularity function\|class\|both` | both (a list in the config file) | What to extract. Web UI: the **Extract functions** and **Extract classes** checkboxes. At least one is required |
| `engineered_only` | `--engineered-only` / `--no-engineered-only` | off | Engineered project filter (Xiao et al., 2025; in the spirit of Reaper, Munaiah et al., 2017). See the notes below |
| `annotate_llm_score` | `--annotate-llm-score` / `--no-annotate-llm-score` | off | Score each record with P(AI-generated) using a CodeBERT classifier. Weights are downloaded from the Hugging Face Hub on first use |
| `export_ast` | `--export-ast` / `--no-export-ast` | off | Include the full AST as nested JSON in every record (large). `ast_depth`, `ast_num_nodes` and `ast_node_types` are always exported. Web UI: **Export Full AST** |
| `deduplicate` | `--deduplicate` / `--no-deduplicate` | off in every interface | MinHash near-duplicate removal |
| `dedup_threshold` | `--dedup-threshold` | 0.8 | Jaccard similarity threshold, 0 to 1 (web UI slider: 0.5 to 1.0). Used only when deduplication is on |
| `output_dir` | `--output-dir` | `./sieve_output` | Output directory. The web UI writes to a temporary directory and offers downloads |
| `export_format` | `--export-format` | `jsonl` | `jsonl`, `parquet` or `both` |
| `github_token` | `--github-token`, or the `GITHUB_TOKEN` environment variable | none | GitHub token. Without one GitHub allows 60 requests per hour |
| `hf_token` | none; the `HF_TOKEN` environment variable | none | Hugging Face token, optional, only used with `annotate_llm_score` |
| `wait_for_rate_limit` | `--wait-for-rate-limit` / `--no-wait-for-rate-limit` | off | Sleep until a GitHub rate limit resets and carry on, instead of stopping. Not available in the web UI |

Notes:

- With `--config`, the JSON file is the only source of settings and every other flag, including `--github-token`, is ignored. Fields missing from the file take the config defaults above. Unknown keys are an error.
- Tokens: `GITHUB_TOKEN` and `HF_TOKEN` are read from the environment by the CLI, with flags and with `--config`; they fill only a token that is still unset, so a token written in the config file takes precedence. There is no `--hf-token` flag. Tokens are never written to `manifest.json` or to the config printed at the start of a run. A config file is plain text, so prefer the environment variable. The web UI reads neither variable.
- Caps: `0` and negative numbers are rejected by the CLI and config. To skip functions or classes, drop that level from `granularity`.
- Discovery: GitHub search returns at most 1,000 results per query, so a larger result set is collected by splitting the creation-date range into windows; what was and was not covered is recorded in the `discovery` block of `manifest.json`.
- When `engineered_only` is on, the filter applies a license check (repos with no license or an excluded license are dropped), a no-release exclusion, a minimum of 2 contributors, and bottom-Q1 exclusion on pull-request count and issue count. The effective contributor threshold is `max(min_contributors, 2)`, because discovery applies `min_contributors` first. The Q1 (25th percentile) thresholds are computed over the candidates that pass the license, release, and contributor checks within the discovered pool (after `min_stars`, `min_contributors`, and `max_repos` have been applied), not over all of GitHub. The LOC and code-ratio criteria are implemented in `core/quality.py` but are not applied by the pipeline, because filtering runs before repositories are cloned.

---

## Output Schema

The record examples below are abbreviated. The *Record Schema* section of the Documentation page lists every field of both record types with its type and meaning.

### Function/Method records (`functions.jsonl`) — CodeSearchNet-compatible

```json
{
  "repo": "owner/repo",
  "file_path": "src/utils.py",
  "language": "Python",
  "func_name": "compute_metrics",
  "parameters": ["predictions", "labels"],
  "return_annotation": "dict",
  "docstring": "Compute precision, recall, and F1.",
  "source_code": "def compute_metrics(...): ...",
  "signature": "def compute_metrics(predictions, labels) -> dict: ...",
  "used_imports": ["from sklearn.metrics import f1_score"],
  "decorators": [],
  "start_line": 42,
  "end_line": 61,
  "is_method": false,
  "parent_class": null,
  "commit_sha": "a1b2c3d4...",
  "llm_score": 0.12,
  "llm_score_truncated": false,
  "ast_depth": 8,
  "ast_num_nodes": 54,
  "ast_node_types": {"function_definition": 1, "return_statement": 1},
  "cyclomatic_complexity": 3,
  "loc": 12,
  "sloc": 9,
  "volume": 87.5,
  "maintainability_index": 74.2
}
```

### Class records (`classes.jsonl`) — OpenClassEval/OpenClassGen-compatible

```json
{
  "repo": "owner/repo",
  "file_path": "src/model.py",
  "language": "Python",
  "class_name": "TransformerBlock",
  "parent_classes": ["nn.Module"],
  "docstring": "Single transformer block.",
  "source_code": "class TransformerBlock(...): ...",
  "skeleton": "class TransformerBlock(nn.Module):\n    def forward(self, x): pass",
  "method_names": ["__init__", "forward"],
  "method_count": 2,
  "has_constructor": true,
  "decorators": [],
  "used_imports": ["import torch.nn as nn"],
  "start_line": 10,
  "end_line": 45,
  "commit_sha": "a1b2c3d4...",
  "llm_score": 0.31,
  "llm_score_truncated": false,
  "cyclomatic_complexity": 5,
  "maintainability_index": 68.1
}
```

### Manifest (`manifest.json`)

A JSON file capturing the config (never any token), per-repo metadata and summary statistics. Share it so others can see how the corpus was built (a rerun can give a different corpus, see *Known Limitations*). The full layout, with the fields that can be `null`, is under *manifest.json Layout* on the Documentation page. It records:

- `sieve_version`, `generated_at`, `config`, `repos` (one entry per processed repository, with the star count and other metadata as collected) and `summary` (`total_repos`, `total_functions`, `total_classes`, `total_llm_scored`, `total_llm_score_truncated`).
- `discovery`: whether the repository search covered everything that matched, the windows it used, single days that held more than 1,000 repositories, repositories dropped because their contributors could not be fetched, and how long the run waited on rate limits.
- `extraction`: the directory names extraction skips and how many source files that skipped.

`llm_score_truncated` is `true` for a record whose score reflects only its first 512 tokens, and `null` when the record was not scored.

---

## Dependency Graph

SIEVE automatically parses package dependency manifests from each cloned repository and stores them in the corpus manifest. The web UI shows an interactive force-directed dependency graph in the Dataset Statistics panel, with packages colored by kind (main, dev, optional).

Supported manifest formats:

| Language | Manifest Files |
|---|---|
| Python | `requirements*.txt`, `requirements/*.txt`, `pyproject.toml`, `setup.cfg` |
| JavaScript | `package.json` (dependencies, devDependencies, peerDependencies) |
| Java | `pom.xml` (Maven) |
| C++ | `conanfile.txt`, `vcpkg.json`, `CMakeLists.txt` (find_package) |

---

## Code Metrics

SIEVE computes 22 structural metrics for every extracted record using tree-sitter — no external tools required, consistent across all four languages:

| Category | Metrics |
|---|---|
| **Raw** | `loc`, `sloc`, `lloc`, `comments`, `multi`, `blank`, `comment_ratio` |
| **Complexity** | `cyclomatic_complexity`, `max_nesting_depth` |
| **Halstead** | `h1`, `h2`, `N1`, `N2`, `vocabulary`, `halstead_length`, `calculated_length`, `volume`, `difficulty`, `effort`, `time`, `bugs` |
| **Composite** | `maintainability_index` |

---

## Testing

```bash
pytest
# With coverage
pytest --cov=sieve --cov-report=term-missing
```

The suite covers unit, integration, and end-to-end levels. GitHub and git are mocked — no network access required.

---

## Known Limitations

The full list is on the Documentation page. In short:

- **Creation date is not code age.** The date filter uses the repository's GitHub creation date. A repository with imported history can contain commits that predate it.
- **No snapshot date.** Results depend on when the run happens, and the default branch HEAD at that moment is what is cloned. The manifest records `generated_at`, a `collected_at` timestamp per repository and the cloned `commit_sha` on each record, not a fixed snapshot.
- **Skipped directories.** `vendor`, `vendored`, `third_party`, `third-party`, `site-packages`, `node_modules`, `dist`, `build`, `venv`, `.venv`, `.tox` and `.eggs` are skipped, at any depth. `external` and `env` are not skipped.
- **Test-file rules are not language-specific.** A file name that starts with `Test` is dropped in every language (so `Testimonials.js` too), and `test/`, `tests/`, `__tests__/`, `spec/` and `specs/` directories are skipped. `*Tests.java`, `*IT.java`, `*.test.jsx` and `foo_test.h` are not treated as tests.
- **A narrow Pass 1 / Pass 2 gap.** Pass 1 counts exactly what Pass 2 extracts, except that a file which parses but fails while its records are built is counted in Pass 1 and dropped in Pass 2.
- **The AI-generation score is an estimate.** It is a probabilistic signal, not evidence of authorship, and its evaluation does not cover LLMs outside the training data.
- **Sequential processing.** Repositories are processed one after another, not in parallel, so large runs are slow.
- **C++ parsing is slower than other languages.** C++ extraction takes 2–5× longer than Python, Java or JavaScript repositories of comparable size. This is a known limitation of `tree-sitter-cpp`: C++ has syntactic ambiguities (e.g. `a * b` can be a multiplication or a pointer declaration) that require Generalized LR (GLR) parsing, which is more expensive than the deterministic LR(1) parsing used for the other languages.

---

## Documentation

The web UI has a **Documentation** page and an **About the Classifier** page in its sidebar (on the [live demo](https://mrahman2025-sieve.hf.space), or locally with `streamlit run src/sieve/ui/Home.py`). The Documentation page covers every parameter, the record schemas, the 22 metrics, the engineered-project filter, deduplication, the `manifest.json` layout, dependency extraction, the architecture and how to extend SIEVE. The **FAQ** page answers common questions.

---

## License

SIEVE is released under the MIT License. See [`LICENSE`](LICENSE).

---

## Citation

If you use SIEVE in your research, please cite:

```bibtex
@article{rahman2026sieve,
  title={SIEVE: A Parameterized GitHub Corpus Builder for Software Engineering Research},
  author={Rahman, Musfiqur and Begoug, Mahi and Shihab, Emad},
  journal={arXiv preprint arXiv:2610.22544},
  year={2026}
}
```

GitHub's **Cite this repository** button gives the same citation, generated from [`CITATION.cff`](CITATION.cff).
