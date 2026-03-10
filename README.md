# 🔬 SIEVE
**Software Ingestion & Extraction for Verifiable Evaluation**

[![Coverage](https://codecov.io/gh/mrsumitbd/sieve/graph/badge.svg?token=19aecee5-afb0-4d67-9c0d-bdab723ce8d3)](https://codecov.io/gh/mrsumitbd/sieve)
![Tests](https://img.shields.io/badge/tests-214%20passed-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Languages](https://img.shields.io/badge/languages-Python%20%7C%20Java%20%7C%20JavaScript-orange?style=flat-square)
![Venue](https://img.shields.io/badge/venue-ICSME%202025-blueviolet?style=flat-square)

SIEVE is a parameterized GitHub corpus builder for software engineering research. It lets you curate contamination-aware, high-quality code datasets from public repositories with full control over language, recency, repository quality, and test suite presence.

---

## Why SIEVE?

Static benchmarks like HumanEval and CodeSearchNet have well-known contamination and saturation problems. SIEVE lets you build fresh corpora from post-cutoff repositories, ensuring your evaluation data was not part of any model's training set.

---

## Installation

### Prerequisites

**Python 3.11+** is required. Check your version with `python --version`.

**cloc** (Count Lines of Code) is highly recommended. SIEVE uses it for accurate LOC and comment line counting when the Engineered Projects filter is enabled. Without it, SIEVE falls back to an AST-based counter (accurate, but slower on large repos).

| Platform | Command |
|---|---|
| macOS | `brew install cloc` |
| Ubuntu / Debian | `sudo apt install cloc` |
| Windows (Chocolatey) | `choco install cloc` |
| Windows (winget) | `winget install AlDanial.Cloc` |
| pip (cross-platform) | `pip install cloc` |
| Manual | [github.com/AlDanial/cloc/releases](https://github.com/AlDanial/cloc/releases) |

Verify installation: `cloc --version`

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-org/sieve.git
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

You should see `(.venv)` at the start of your terminal prompt confirming the environment is active. All subsequent commands run inside this isolated environment — your system Python is untouched.

**3. Install SIEVE and its dependencies**

Runtime only:
```bash
pip install -r requirements.txt
pip install -e .
```

For development and testing (includes pytest):
```bash
pip install -r requirements-dev.txt
pip install -e .
```

Or equivalently using the package extras:
```bash
pip install -e ".[dev]"
```

**4. Set up your GitHub token**
```bash
cp .env.example .env
```
Open `.env` and replace `your_github_pat_here` with your GitHub Personal Access Token. Generate one at [github.com/settings/tokens](https://github.com/settings/tokens) — only the `public_repo` scope is needed. Without a token, the GitHub API rate limit is 60 requests/hour which is too low for any meaningful run.

---

## Usage

> **Note:** Make sure your virtual environment is activated (`source .venv/bin/activate`) before running any of the commands below.

### Web Interface (Streamlit)

```bash
streamlit run src/sieve/ui/app.py
```

Open `http://localhost:8501` in your browser. To explore SIEVE without a GitHub token, click **📦 Load Example Dataset** in the sidebar — this loads a pre-built corpus of 54 functions and 8 classes across three synthetic Python repositories, with no API calls required. To build a real corpus, configure parameters in the sidebar and click **▶ Run SIEVE**.

### CLI

```bash
# Inline parameters
sieve run --language Python --start-date 2024-01-01 --end-date 2025-01-01 --min-stars 50 --min-contributors 5

# Config file
sieve run --config my_config.json

# Validate a config without running
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
    require_tests=True,
    output_dir="./my_corpus",
    export_format="jsonl",
)

summary = run_pipeline(config)
print(summary)
```

### Deactivating the virtual environment

When you are done, deactivate the virtual environment with:
```bash
deactivate
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `language` | str | Target language: `Python`, `Java`, `JavaScript` |
| `start_date` | date | Only include repos pushed on or after this date |
| `end_date` | date | Only include repos pushed on or before this date |
| `min_stars` | int | Minimum GitHub stars |
| `min_contributors` | int | Minimum unique contributors |
| `max_repos` | int \| None | Cap on repos to process |
| `max_functions` | int \| None | Cap on extracted functions (stratified sample drawn if exceeded) |
| `max_classes` | int \| None | Cap on extracted classes (stratified sample drawn if exceeded) |
| `granularity` | list | `function`, `class` |
| `require_tests` | bool | Only include repos with a detected test suite |
| `engineered_only` | bool | Apply Xiao et al. / Munaiah et al. engineered project filter |
| `deduplicate` | bool | Apply MinHash near-duplicate removal |
| `dedup_threshold` | float | Jaccard similarity threshold (default 0.8) |
| `output_dir` | str | Local path for output files |
| `export_format` | str | `jsonl`, `parquet`, or `both` |

---

## Output Schema

### Function-level (`functions.jsonl`) — CodeSearchNet-compatible
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
  "llm_score": null
}
```

### Class-level (`classes.jsonl`) — OpenClassEval-compatible
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
  "llm_score": null
}
```

> `llm_score` is a placeholder for the CodeProbe classifier (P(LLM-generated) per record). It will be populated in a future release.

### Manifest (`manifest.json`)
A JSON file capturing the full config, per-repo metadata (stars, contributors, license, test suite report), and summary statistics. Enables reproducibility — share the manifest to let others regenerate the same corpus.

---

## Testing

Run the full test suite with:
```bash
pytest
```

Or with coverage:
```bash
pytest --cov=sieve --cov-report=term-missing
```

The suite covers 179 tests across unit, integration, and end-to-end levels. GitHub and git are mocked in all tests — no network access is required.

---

## Roadmap

- [x] Java and JavaScript extraction
- [ ] File-level granularity
- [ ] `llm_score` population via CodeProbe classifier
- [ ] Execution-based test coverage (Docker sandboxed)
- [ ] SWE-bench style issue-to-fix task extraction
- [ ] HuggingFace Datasets integration for direct push

---

## Citation

If you use SIEVE in your research, please cite:

```bibtex
@inproceedings{sieve2025,
  title     = {SIEVE: A Parameterized Corpus Builder for Contamination-Aware Software Engineering Research},
  author    = {Rahman, Musfiqur and Shihab, Emad},
  booktitle = {Proceedings of the International Conference on Software Maintenance and Evolution (ICSME)},
  year      = {2025}
}
```
