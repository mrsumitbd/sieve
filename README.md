# 🔬 SIEVE
**Software Ingestion & Extraction for Verifiable Evaluation**

SIEVE is a parameterized GitHub corpus builder for software engineering research. It lets you curate contamination-aware, high-quality code datasets from public repositories with full control over language, recency, repository quality, and test suite presence.

---

## Why SIEVE?

Static benchmarks like HumanEval and CodeSearchNet have well-known contamination and saturation problems. SIEVE lets you build fresh corpora from post-cutoff repositories, ensuring your evaluation data was not part of any model's training set.

---

## Installation

### Prerequisites

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

### SIEVE

```bash
git clone https://github.com/your-org/sieve.git
cd sieve
pip install -e .
```

A GitHub Personal Access Token is strongly recommended (set as `GITHUB_TOKEN` or enter it in the UI). Without it, the GitHub API rate limit is 10 requests/minute.

---

## Usage

### Web Interface (Streamlit)

```bash
streamlit run sieve/ui/app.py
```

Open `http://localhost:8501` in your browser, configure parameters in the sidebar, and click **Run SIEVE**.

### CLI

```bash
# Inline parameters
sieve run --language Python --cutoff-date 2024-01-01 --min-stars 50 --min-contributors 5

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
    cutoff_date=date(2024, 1, 1),
    min_stars=50,
    min_contributors=5,
    require_tests=True,
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
| `language` | str | Target language: `Python`, `Java`, `JavaScript` |
| `cutoff_date` | date | Repos pushed after this date only |
| `min_stars` | int | Minimum GitHub stars |
| `min_contributors` | int | Minimum unique contributors |
| `max_repos` | int \| None | Cap on repos to process |
| `granularity` | list | `function`, `class`, `file` |
| `require_tests` | bool | Only include repos with detected test suite |
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
  "start_line": 42,
  "end_line": 61,
  "is_method": false,
  "parent_class": null,
  "decorators": []
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
  "start_line": 10,
  "end_line": 45
}
```

### Manifest (`manifest.json`)
A JSON file capturing the full config, per-repo metadata (stars, contributors, license, test suite report), and summary statistics. Enables reproducibility — share the manifest to let others regenerate the same corpus.

---

## Roadmap

- [ ] Java and JavaScript extraction
- [ ] File-level granularity
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
