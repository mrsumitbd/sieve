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
![Tests](https://img.shields.io/badge/tests-361%20passed-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Languages](https://img.shields.io/badge/languages-Python%20%7C%20Java%20%7C%20JavaScript%20%7C%20C%2B%2B-orange?style=flat-square)
![Venue](https://img.shields.io/badge/venue-MSR%202027%20Dublin-blueviolet?style=flat-square)

SIEVE is a parameterized GitHub corpus builder for software engineering research. It lets you curate contamination-aware, high-quality code datasets from public repositories with full control over language, recency, repository quality, and extraction granularity.

**Live demo:** https://mrahman2025-sieve.hf.space

---

## Why SIEVE?

Static benchmarks like HumanEval and CodeSearchNet have well-known contamination and saturation problems. SIEVE lets you build fresh corpora from post-cutoff repositories, ensuring your evaluation data was not part of any model's training set.

---

## Installation

### Prerequisites

**Python 3.11+** is required. Check your version with `python --version`.

**cloc** (Count Lines of Code) is recommended. SIEVE uses it for accurate LOC and comment line counting when the Engineered Projects filter is enabled. Without it, SIEVE falls back to an AST-based counter.

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
Open `.env` and replace `your_github_pat_here` with your GitHub Personal Access Token. Generate one at [github.com/settings/tokens](https://github.com/settings/tokens) — only the `public_repo` scope is needed.

---

## Usage

### Web Interface

```bash
streamlit run src/sieve/ui/Home.py
```

Open `http://localhost:8501`. Click **📦 Load Example Dataset** to explore without a GitHub token, or configure parameters in the sidebar and click **▶ Run SIEVE** to build a real corpus.

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
    start_date=date(2024, 1, 1),
    end_date=date(2025, 1, 1),
    min_stars=50,
    min_contributors=5,
    output_dir="./my_corpus",
    export_format="jsonl",
)

summary = run_pipeline(config)
print(summary)
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `language` | str | Target language: `Python`, `Java`, `JavaScript`, `C++` |
| `start_date` | date | Only include repos pushed on or after this date |
| `end_date` | date | Only include repos pushed on or before this date |
| `min_stars` | int | Minimum GitHub stars |
| `min_contributors` | int | Minimum unique contributors |
| `max_repos` | int \| None | Cap on repos to process |
| `max_functions` | int \| None | Cap on extracted functions (stratified by repo) |
| `max_classes` | int \| None | Cap on extracted classes (stratified by repo) |
| `granularity` | list | `function`, `class`, or both |
| `engineered_only` | bool | Apply Munaiah et al. (2017) engineered project filter |
| `annotate_llm_score` | bool | Score each record with P(AI-generated) via CodeBERT classifier |
| `export_ast` | bool | Add AST features to every record |
| `deduplicate` | bool | Apply MinHash near-duplicate removal |
| `dedup_threshold` | float | Jaccard similarity threshold (default 0.8) |
| `output_dir` | str | Local path for output files |
| `export_format` | str | `jsonl`, `parquet`, or `both` |

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

## Code Metrics

SIEVE computes 23 structural metrics for every extracted record using tree-sitter — no external tools required, consistent across all four languages:

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

The suite covers 361 tests across unit, integration, and end-to-end levels. GitHub and git are mocked — no network access required.

---

## Citation

If you use SIEVE in your research, please cite:

```bibtex
TBA
```