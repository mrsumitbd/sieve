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

Runtime only:
```bash
pip install -r requirements.txt
pip install -e .
```

For development and testing:
```bash
pip install -r requirements-dev.txt
pip install -e .
```

**4. Set up your GitHub token**
```bash
cp .env.example .env
```
Open `.env` and add the line `GITHUB_TOKEN=<your GitHub Personal Access Token>`. Generate one at [github.com/settings/tokens](https://github.com/settings/tokens); only the `public_repo` scope is needed. The CLI reads this file. The web UI does not, so paste the token into the sidebar there.

---

## Usage

### Web Interface

```bash
streamlit run src/sieve/ui/Home.py
```

Open `http://localhost:8501`, configure parameters in the sidebar and click **▶ Run SIEVE** to build a corpus. A GitHub token is optional but recommended for full runs.

### CLI

```bash
sieve run --language Python --start-date 2024-01-01 --end-date 2025-01-01 \
          --min-stars 50 --min-contributors 5

# From config file
sieve run --config my_config.json

# Validate config without running
sieve validate-config my_config.json
```

### Python API

```python
from datetime import date
from sieve.config import SIEVEConfig
from sieve.pipeline import run_pipeline

config = SIEVEConfig(
    language="Python",
    start_date=date(2024, 6, 1),   # repo creation lower bound (e.g. a model's release date)
    end_date=date(2025, 6, 1),     # repo creation upper bound
    min_stars=50,
    min_contributors=5,
    max_repos=20,
    max_functions=30,
    max_classes=30,
    engineered_only=True,
    annotate_llm_score=True,
    output_dir="./my_corpus",
    export_format="both",          # "jsonl", "parquet", or "both"
)

summary = run_pipeline(config)
print(summary)
```

---

## Parameters

Defaults differ between interfaces where noted. "CLI/API" means the `sieve run` command and `SIEVEConfig`.

| Option | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `language` | `--language` | str | required in API and CLI; web UI: `Python` | Target language: `Python`, `Java`, `JavaScript`, `C++` |
| `start_date` | `--start-date` | date (`YYYY-MM-DD` on the CLI) | required in API and CLI (unless `--config`); web UI: 2024-01-01 | Earliest repository creation date (YYYY-MM-DD). Bounds repository age (for example for contamination-aware evaluation); it does not verify snippet origin. |
| `end_date` | `--end-date` | date (`YYYY-MM-DD` on the CLI) | required in API/config; CLI: 30 days before today if omitted; web UI: 30 days before today (rolling, editable) | Only include repos **created** on or before this date. Must be on or after `start_date` |
| `min_stars` | `--min-stars` | int | CLI/API: 10; web UI: 50 | Minimum GitHub stars |
| `min_contributors` | `--min-contributors` | int (at least 1) | CLI/API: 1; web UI: 5 | Minimum unique contributors |
| `max_repos` | `--max-repos` | int \| None | CLI/API: None (no cap); web UI: 20 | Cap on repos to process, applied during discovery (repos are taken in descending star order). In the CLI/API leave it unset for no cap, because 0 yields no repos. In the web UI, 0 means no cap |
| `max_functions` | `--max-functions` | int \| None | CLI/API: None (no cap); web UI: 30 | Cap on total extracted functions (stratified by repo). In the CLI/API leave it unset for no cap, because 0 extracts no functions, and do not pass -1 (use `granularity` to skip a level). In the web UI, 0 means no cap and -1 skips function extraction |
| `max_classes` | `--max-classes` | int \| None | CLI/API: None (no cap); web UI: 30 | Cap on total extracted classes (stratified by repo). Same rules as `max_functions`; in the web UI, -1 skips class extraction |
| `granularity` | config file / API only | list | `["function", "class"]` | `function`, `class`, or both. The CLI always extracts both; the web UI derives it from the -1 values of `max_functions` and `max_classes` |
| `engineered_only` | `--engineered-only` / `--no-engineered-only` | bool | False | Apply engineered project filter (Xiao et al., 2025; in the spirit of Reaper, Munaiah et al., 2017). See the notes below |
| `annotate_llm_score` | config file / API only | bool | False | Score each record with P(AI-generated) via CodeBERT classifier. The classifier weights are downloaded from HuggingFace Hub on first use. Also available as a web UI toggle |
| `export_ast` | config file / API only | bool | False | Include the full AST as nested JSON in every record (increases file size significantly). The summary fields `ast_depth`, `ast_num_nodes`, and `ast_node_types` are always present. In the web UI this is the "Include Full AST in Export" toggle |
| `deduplicate` | `--deduplicate` / `--no-deduplicate` | bool | CLI: True; API/config: False; web UI: off | Apply MinHash near-duplicate removal |
| `dedup_threshold` | `--dedup-threshold` | float (0 to 1) | 0.8 | Jaccard similarity threshold for deduplication (web UI slider: 0.5 to 1.0) |
| `output_dir` | `--output-dir` | str | `./sieve_output` | Local path for output files. The web UI writes to a temporary directory and offers downloads instead |
| `export_format` | `--export-format` | str | `jsonl` | `jsonl`, `parquet`, or `both` |
| `github_token` | `--github-token` | str | None | GitHub Personal Access Token for higher rate limits |
| `hf_token` | config file / API only | str | None | HuggingFace token for downloading classifier weights (only used with `annotate_llm_score`) |

Notes:

- `--config` takes precedence over individual flags: when it is given, the JSON file is the only source of settings and every other flag is ignored. Fields missing from the file take the API/config defaults above (for example `deduplicate` is False). This includes the token, so with `--config` set `github_token` in the file.
- The GitHub token is read from `--github-token` or the `GITHUB_TOKEN` environment variable in the CLI (without `--config`), from the `github_token` field in the API, and only from the text box in the web UI (no environment fallback). Without a token GitHub allows 60 requests per hour.
- When `engineered_only` is on, the filter applies a license check (repos with no license or an excluded license are dropped), a no-release exclusion, a minimum of 2 contributors, and bottom-Q1 exclusion on pull-request count and issue count. The effective contributor threshold is `max(min_contributors, 2)`, because discovery applies `min_contributors` first. The Q1 (25th percentile) thresholds are computed over the candidates that pass the license, release, and contributor checks within the discovered pool (after `min_stars`, `min_contributors`, and `max_repos` have been applied), not over all of GitHub. The LOC and code-ratio criteria are implemented in `core/quality.py` but are not applied by the pipeline, because filtering runs before repositories are cloned.

---

## Output Schema

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
  "cyclomatic_complexity": 5,
  "maintainability_index": 68.1
}
```

### Manifest (`manifest.json`)

A JSON file capturing the full config, per-repo metadata, and summary statistics. Share it to let others reproduce the same corpus.

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

**C++ parsing is slower than other languages.** C++ extraction takes 2–5× longer than Python, Java, or JavaScript repositories of comparable size. This is a known limitation of `tree-sitter-cpp` — C++ has fundamental syntactic ambiguities (e.g. `a * b` can be a multiplication or a pointer declaration) that require Generalized LR (GLR) parsing to resolve at runtime, which is more expensive than the deterministic LR(1) parsing used for the other languages. See the [Documentation page](https://mrahman2025-sieve.hf.space) for details.

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
