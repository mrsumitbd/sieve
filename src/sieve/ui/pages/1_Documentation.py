"""
pages/1_📖_Documentation.py

Documentation and FAQ page for SIEVE.
Automatically discovered by Streamlit's multipage feature when placed in pages/.
"""

import streamlit as st

st.set_page_config(
    page_title="SIEVE — Documentation",
    page_icon="📖",
    layout="wide",
)

st.title("📖 SIEVE Documentation")
st.caption("**S**oftware **I**ngestion & **E**xtraction for **V**erifiable **E**valuation")
st.divider()

# ─── Overview ─────────────────────────────────────────────────────────────────

st.header("Contamination-Aware Evaluation")
st.markdown("""
### The Problem

Large language models (LLMs) are pre-trained on massive corpora scraped from GitHub,
Stack Overflow, and other public sources. The training data typically has a **cutoff
date** — a point after which new data was not ingested. Any code that existed publicly
before that cutoff may have been seen during pre-training.

This creates a contamination problem for evaluation:

> If you evaluate an LLM on code it has already seen during training, the model may
> reproduce memorized solutions rather than demonstrate genuine generalization. The
> result is an inflated, unreliable performance estimate.

This is not hypothetical. Research has shown:

1. **For code generation:** LLMs achieve 84–89% correctness on widely-used synthetic
   benchmarks (e.g. HumanEval, MBPP) but only **25–34%** on real-world class-level
   code generation tasks drawn from open-source repositories
   *(Rahman et al., "Beyond Synthetic Benchmarks", arXiv:2510.26130, under review at EMSE)*.
   The synthetic benchmarks are closed, well-known, and likely represented in training
   data — producing optimistic results that do not reflect real-world capability.

2. **For LLM-generated code detection:** A classifier trained to distinguish
   human-written from AI-generated code shows **significant performance degradation**
   when evaluated on post-cutoff data compared to pre-cutoff data
   *(Rahman et al., arXiv:2409.01382)*.
   Contamination in the evaluation set makes detection appear more accurate than it is
   — a critical issue given that reviewers now routinely require contamination checks.

### How SIEVE Solves This

SIEVE filters repositories by **creation date**, not last push date. A repository
created after an LLM's training cutoff **cannot have existed** when the model was
trained — making every function and class extracted from it contamination-free by
construction.

### How to Choose Your Cutoff Date

Set **Start Date** to the training cutoff of the LLM(s) you are studying. Some
reference points:

| Model | Approximate Training Cutoff |
|---|---|
| GPT-3.5 (ChatGPT) | September 2021 |
| GPT-4 | April 2023 |
| GPT-4o | October 2023 |
| Claude 3 (Haiku/Sonnet/Opus) | August 2023 |
| Claude 3.5 Sonnet | April 2024 |
| Gemini 1.5 Pro | November 2023 |
| Llama 3 (8B/70B) | December 2023 |
| CodeLlama | January 2023 |

If studying multiple models simultaneously, use the **latest** cutoff among them
to guarantee contamination-free data for all.

### Reviewer Checklist

When submitting papers that use SIEVE-generated corpora, you can state:

> *"Our evaluation corpus was built using SIEVE, collecting only from GitHub
> repositories created after [DATE]. By construction, no extracted code could have
> appeared in the pre-training data of any model evaluated, as the repositories did
> not exist at the time of training."*

This satisfies the contamination check now required by most SE venues.
""")

st.divider()

# ─── What is SIEVE? ───────────────────────────────────────────────────────────
st.markdown("""
SIEVE is a parameterized GitHub corpus builder for software engineering research.
It addresses a core validity threat in LLM-based code studies: **benchmark contamination**.
When training data and evaluation data overlap, benchmark scores are inflated and results
are not reproducible. SIEVE mitigates this by letting you construct evaluation corpora
**on demand** from repositories whose last activity falls within a user-defined date window
— making it straightforward to target a period after a model's training cutoff.

SIEVE extracts two types of program units:
- **Functions** — CodeSearchNet-style records with full source, signature, parameters, return type, and docstring.
- **Classes** — OpenClassEval-style records with full source, skeleton, method list, and inheritance.

Both record types include the import statements used within the extracted unit, enabling
self-contained code snippets for prompting or evaluation.
""")

st.divider()

# ─── Parameters ───────────────────────────────────────────────────────────────

st.header("Parameter Reference")

st.subheader("Date Range")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Start Date** | Only include repos **created** on or after this date. Set this to the LLM training cutoff date — any repo created after this date is guaranteed contamination-free, since it did not exist when the model was trained. |
| **End Date** | Only include repos **created** on or before this date. Default: first of last month. Filters out very new repos with little code. |
| **Min Last Activity** | Only include repos pushed on or after this date. Must be ≥ End Date. Default: same as End Date. Ensures repos are actively maintained rather than abandoned after creation. |

The GitHub search query uses `created:START..END pushed:>=ACTIVITY`, filtering on
**repository creation date** — not last push date. This is the correct approach for
contamination-aware corpus building: a repo created after the cutoff cannot have
existed in any LLM training dataset.

**Guarantee:** If Start Date = Jan 1 2024, every extracted function and class is
from code that was written after Jan 1 2024.
""")

st.subheader("Repository Filters")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Min Stars** | Minimum GitHub star count. Higher values bias toward popular, well-maintained projects but reduce diversity. |
| **Min Contributors** | Minimum unique contributor count. Helps exclude personal or toy projects. Note: bots are excluded from the count. |
| **Max Repos** | Hard cap on the number of repos to process (0 = no cap). Applies during discovery, before cloning. |
| **Max Functions** | Cap on total extracted functions *after deduplication*. When the corpus exceeds this, stratified sampling is applied (see below). |
| **Max Classes** | Cap on total extracted classes *after deduplication*. Same stratified sampling applies. |
""")

st.subheader("Content Filters")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Granularity** | Choose to extract functions, classes, or both. |
| **Engineered Projects Only** | Applies the Munaiah et al. (2017) filter — see section below. Includes test suite presence as one of its signals. |
""")

st.subheader("Deduplication")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Deduplicate** | Enable MinHash LSH near-duplicate removal across all extracted records. |
| **Similarity Threshold** | Jaccard similarity threshold for MinHash deduplication (0.0–1.0). Higher values are more conservative — only near-identical code is removed. Default: 0.8. |

Deduplication tokenizes source code (stripping string literals and normalizing whitespace) before
hashing, so superficial differences like renamed variables or changed comments do not prevent
duplicate detection.
""")

st.divider()

# ─── Stratified Sampling ──────────────────────────────────────────────────────

st.header("Corpus Size Caps & Stratified Sampling")
st.markdown("""
When **Max Functions** or **Max Classes** is set and the extracted corpus exceeds that cap,
SIEVE does not draw a simple random sample. Instead it uses **proportional stratified sampling**
by repository:

1. Records are grouped by source repository.
2. Each repo's allocation is `cap × (repo_count / total_count)`.
3. Integer parts are taken first; remainder slots are filled by the repos with the largest
   fractional allocations (standard largest-remainder method).
4. Within each repo's allocation, records are drawn uniformly at random without replacement.

**Example:** cap = 20, corpus = TheAlgorithms (276 classes), system-design-primer (52 classes).
Proportional allocation: TheAlgorithms gets 17, system-design-primer gets 3.

**Important caveat:** Proportional allocation preserves the *natural* distribution of the corpus.
If your corpus is dominated by one repository (e.g. a large algorithm collection), the sample
will reflect that. If you need balanced representation across repos, use the **Engineered Projects**
filter or set **Max Repos** to a small number of carefully chosen repositories.
""")

st.divider()

# ─── Engineered Projects ──────────────────────────────────────────────────────

st.header("Engineered Projects Filter")
st.markdown("""
Enabling **Engineered Projects Only** applies a three-stage quality filter based on
[Xiao et al. (2025)](https://arxiv.org/abs/2507.10422) and the REAPER methodology
([Munaiah et al., 2017](https://doi.org/10.1007/s10664-017-9512-6)):

**Stage 1 — License exclusion.** Repos with non-software licenses are excluded:
`CC-BY-4.0`, `CC0-1.0`, `CC-BY-SA-4.0`, `OFL-1.1` (SIL Open Font License),
and repos with no license at all.
These are typically documentation sites, datasets, or font projects — not software.

**Which licenses are kept?** All standard software licenses pass Stage 1, including
MIT, Apache-2.0, GPL-2.0, GPL-3.0, BSD-2-Clause, BSD-3-Clause, LGPL, MPL-2.0,
ISC, AGPL-3.0, and **Unlicense**. The Unlicense is a public domain dedication for
software (equivalent to "no restrictions") and is treated as a valid software license.
Its presence does not indicate a non-software repo — many legitimate, production-quality
software projects use it.

**Stage 2 — Hard thresholds.** Repos with zero releases or fewer than 2 contributors
are excluded unconditionally.

**Stage 3 — Population-level Q1 filter.** Across all discovered repos, SIEVE computes
the first quartile (Q1) for pull request count, issue count, and LOC, plus a 97% confidence
interval on code ratio. Repos falling below Q1 on any dimension are excluded.

This filter is the most reliable way to avoid documentation repos, "awesome lists", dataset
repos, and other non-software projects that can otherwise dominate high-star searches.
It requires a **two-pass discovery** (collect all candidates first, compute population
statistics, then filter), which is slower but produces a significantly higher-quality corpus.

**When to enable it:** Any time corpus quality matters more than speed. Strongly recommended
for corpora intended for publication.
""")

with st.expander("Why do documentation repos appear in high-star searches?"):
    st.markdown("""
    GitHub's star count reflects popularity, not software content. Repos like
    `awesome-python`, `free-programming-books`, and `public-apis` routinely rank in the
    top 10 most-starred Python repos despite containing little or no executable code.
    Without the engineered filter, these repos will be discovered, cloned, and processed —
    yielding zero or near-zero extracted records while consuming API quota and clone time.

    If you do not want to use the full engineered filter (it's slower due to the two-pass
    discovery and per-repo metrics collection), a lighter alternative is to set a higher
    **Min Stars** threshold in combination with a meaningful **Min Contributors** count,
    then inspect the per-repo breakdown in the Dataset Statistics panel after a run.
    Repos with 0 extracted functions *and* 0 extracted classes are a reliable signal of
    non-software content.
    """)

st.divider()

# ─── Record Types ─────────────────────────────────────────────────────────────

st.header("Record Schema")

tab_func, tab_class = st.tabs(["Function Record", "Class Record"])

with tab_func:
    st.markdown("""
    | Field | Type | Description |
    |-------|------|-------------|
    | `repo` | str | Full repo name, e.g. `owner/repo` |
    | `file_path` | str | Path relative to repo root |
    | `language` | str | Programming language |
    | `func_name` | str | Function name |
    | `parameters` | list[str] | Parameter list as strings (includes type annotations) |
    | `return_annotation` | str \\| null | Return type annotation if present |
    | `docstring` | str \\| null | Docstring content, stripped of delimiters |
    | `source_code` | str | Full function source, indentation-normalized |
    | `signature` | str | Function header + docstring + `pass` — no implementation |
    | `used_imports` | list[str] | Import statements whose names appear in the function body |
    | `start_line` | int | Start line in the original file |
    | `end_line` | int | End line in the original file |
    | `is_method` | bool | True if defined inside a class |
    | `parent_class` | str \\| null | Enclosing class name if `is_method` is true |
    | `decorators` | list[str] | Decorator/annotation strings (e.g. `@staticmethod`, `@Override`) |
    | `commit_sha` | str \\| null | Git commit SHA at extraction time — use to build GitHub permalink |
    | `llm_score` | float \\| null | P(LLM-generated) from the built-in CodeBERT classifier |
    | `ast_depth` | int \\| null | Maximum depth of the parse tree |
    | `ast_num_nodes` | int \\| null | Total number of AST nodes |
    | `ast_node_types` | dict \\| null | Node type → count mapping |
    | `ast` | dict \\| null | Full AST as nested JSON (only when Export AST is enabled) |
    | `loc` | int \\| null | Total lines of code |
    | `sloc` | int \\| null | Source lines (non-blank, non-comment) |
    | `lloc` | int \\| null | Logical lines of code (statements) |
    | `comments` | int \\| null | Comment lines |
    | `multi` | int \\| null | Multi-line string/comment lines |
    | `blank` | int \\| null | Blank lines |
    | `comment_ratio` | float \\| null | Comment lines / LOC |
    | `cyclomatic_complexity` | int \\| null | McCabe cyclomatic complexity |
    | `max_nesting_depth` | int \\| null | Maximum control flow nesting depth |
    | `h1` | int \\| null | Halstead: distinct operators |
    | `h2` | int \\| null | Halstead: distinct operands |
    | `N1` | int \\| null | Halstead: total operators |
    | `N2` | int \\| null | Halstead: total operands |
    | `vocabulary` | int \\| null | Halstead: h1 + h2 |
    | `halstead_length` | int \\| null | Halstead: N1 + N2 |
    | `calculated_length` | float \\| null | Halstead: calculated length |
    | `volume` | float \\| null | Halstead: volume |
    | `difficulty` | float \\| null | Halstead: difficulty |
    | `effort` | float \\| null | Halstead: effort |
    | `time` | float \\| null | Halstead: estimated programming time (seconds) |
    | `bugs` | float \\| null | Halstead: estimated number of bugs |
    | `maintainability_index` | float \\| null | Maintainability Index (0–100) |
    """)

with tab_class:
    st.markdown("""
    | Field | Type | Description |
    |-------|------|-------------|
    | `repo` | str | Full repo name, e.g. `owner/repo` |
    | `file_path` | str | Path relative to repo root |
    | `language` | str | Programming language |
    | `class_name` | str | Class name |
    | `parent_classes` | list[str] | Base class names |
    | `docstring` | str \\| null | Class-level docstring |
    | `source_code` | str | Full class source, indentation-normalized |
    | `skeleton` | str | Class signature + method signatures + docstrings + `pass` |
    | `used_imports` | list[str] | Import statements whose names appear in the class body |
    | `method_names` | list[str] | Names of all methods defined in the class |
    | `method_count` | int | Number of methods |
    | `has_constructor` | bool | True if a constructor is defined |
    | `decorators` | list[str] | Decorator/annotation strings on the class |
    | `start_line` | int | Start line in the original file |
    | `end_line` | int | End line in the original file |
    | `commit_sha` | str \\| null | Git commit SHA at extraction time — use to build GitHub permalink |
    | `llm_score` | float \\| null | P(LLM-generated) from the built-in CodeBERT classifier |
    | `ast_depth` | int \\| null | Maximum depth of the parse tree |
    | `ast_num_nodes` | int \\| null | Total number of AST nodes |
    | `ast_node_types` | dict \\| null | Node type → count mapping |
    | `ast` | dict \\| null | Full AST as nested JSON (only when Export AST is enabled) |
    | `loc` | int \\| null | Total lines of code |
    | `sloc` | int \\| null | Source lines (non-blank, non-comment) |
    | `lloc` | int \\| null | Logical lines of code (statements) |
    | `comments` | int \\| null | Comment lines |
    | `multi` | int \\| null | Multi-line string/comment lines |
    | `blank` | int \\| null | Blank lines |
    | `comment_ratio` | float \\| null | Comment lines / LOC |
    | `cyclomatic_complexity` | int \\| null | McCabe cyclomatic complexity |
    | `max_nesting_depth` | int \\| null | Maximum control flow nesting depth |
    | `h1` | int \\| null | Halstead: distinct operators |
    | `h2` | int \\| null | Halstead: distinct operands |
    | `N1` | int \\| null | Halstead: total operators |
    | `N2` | int \\| null | Halstead: total operands |
    | `vocabulary` | int \\| null | Halstead: h1 + h2 |
    | `halstead_length` | int \\| null | Halstead: N1 + N2 |
    | `calculated_length` | float \\| null | Halstead: calculated length |
    | `volume` | float \\| null | Halstead: volume |
    | `difficulty` | float \\| null | Halstead: difficulty |
    | `effort` | float \\| null | Halstead: effort |
    | `time` | float \\| null | Halstead: estimated programming time (seconds) |
    | `bugs` | float \\| null | Halstead: estimated number of bugs |
    | `maintainability_index` | float \\| null | Maintainability Index (0–100) |
    """)

st.divider()

# ─── Code Structure Metrics ───────────────────────────────────────────────────

st.header("Code Structure Metrics")
st.markdown("""
SIEVE computes **23 structural metrics** for every extracted record using its own
tree-sitter-based metrics engine (`sieve.core.metrics`). All metrics are computed
from isolated code snippets — no external tools required — and are consistent
across all four supported languages.

#### Raw Metrics

| Metric | Description |
|--------|-------------|
| `loc` | Total lines of code |
| `sloc` | Source lines (non-blank, non-comment) |
| `lloc` | Logical lines of code — number of statement nodes |
| `comments` | Number of single-line comment lines |
| `multi` | Number of multi-line comment/docstring lines |
| `blank` | Number of blank lines |
| `comment_ratio` | Comment lines / LOC |

#### Complexity Metrics

| Metric | Description |
|--------|-------------|
| `cyclomatic_complexity` | McCabe complexity: 1 + number of branching statements (if/elif, for, while, except, logical &&/\\|\\|, comprehensions) |
| `max_nesting_depth` | Maximum depth of nested control flow structures (if, for, while, try, with) |

#### Halstead Metrics

Halstead metrics are derived from the count of distinct and total operators
and operands in the source code.

| Metric | Description |
|--------|-------------|
| `h1` | Distinct operators |
| `h2` | Distinct operands |
| `N1` | Total operator occurrences |
| `N2` | Total operand occurrences |
| `vocabulary` | h1 + h2 |
| `halstead_length` | N1 + N2 |
| `calculated_length` | h1·log₂(h1) + h2·log₂(h2) |
| `volume` | (N1+N2)·log₂(h1+h2) — information content in bits |
| `difficulty` | (h1/2)·(N2/h2) — difficulty to understand |
| `effort` | difficulty × volume |
| `time` | effort / 18 — estimated programming time in seconds |
| `bugs` | volume / 3000 — estimated number of delivered bugs |

#### Composite Metric

| Metric | Description |
|--------|-------------|
| `maintainability_index` | Score from 0 to 100 — higher is more maintainable. Derived from Halstead volume, cyclomatic complexity, and SLOC using the Radon/SEI formula. |

All metrics are computed entirely from the tree-sitter AST — no lizard, radon,
or other external tools are required at runtime.
""")

st.divider()

# ─── Import Detection ─────────────────────────────────────────────────────────

st.header("Import Detection")
st.markdown("""
SIEVE populates `used_imports` for every extracted record by statically analyzing the
file's top-level import statements. The detection logic:

1. All `import X`, `import X as Y`, `from X import A`, and `from X import A as B`
   statements at module level are collected.
2. For each import, the **names introduced into the module namespace** are identified
   (e.g. `import numpy as np` → `np`; `from pathlib import Path` → `Path`).
3. Each name is searched in the function or class source using a word-boundary regex
   (`\\bname\\b`), preventing false positives like `os` matching inside `cosmos`.
4. `from X import *` wildcard imports are always included.

**Enabling imports in the Sample Viewer** prepends the detected import block above the
source code or skeleton — useful for verifying that the record is self-contained.

**Note:** This is static name-reference detection, not full data-flow analysis. It may
include imports that are referenced in type annotations but not at runtime, and will miss
imports used only via `getattr` or dynamic dispatch. For the purposes of LLM prompting
or evaluation, this is generally sufficient.
""")

st.divider()

# ─── Output Files ─────────────────────────────────────────────────────────────

st.header("Output Files")
st.markdown("""
All output is written to the directory specified in **Output Directory**. Files produced:

| File | Description |
|------|-------------|
| `functions.jsonl` | One JSON object per line, one per function record |
| `classes.jsonl` | One JSON object per line, one per class record |
| `functions.parquet` | Columnar format — list fields serialized as JSON strings |
| `classes.parquet` | Columnar format — list fields serialized as JSON strings |
| `manifest.json` | Run metadata: config snapshot, per-repo stats, summary counts, timestamps |

JSONL and Parquet are written based on the **Export Format** setting. The manifest is always written.

**Loading the data in Python:**
```python
import json
from pathlib import Path

functions = [json.loads(l) for l in Path("sieve_output/functions.jsonl").read_text().splitlines() if l]
classes   = [json.loads(l) for l in Path("sieve_output/classes.jsonl").read_text().splitlines() if l]
```

Or with pandas / polars:
```python
import pandas as pd
df = pd.read_json("sieve_output/functions.jsonl", lines=True)
```
""")

st.divider()

# ─── Technical Notes ──────────────────────────────────────────────────────────

st.header("Technical Notes")

with st.expander("Why do some repos yield 0 functions and 0 classes?"):
    st.markdown("""
    These are almost always non-software repositories — documentation sites, curated lists
    (e.g. `awesome-*`), dataset repos, or book projects. They may contain Python scripts
    but often have no `.py` files with extractable function or class definitions.

    **How to avoid this:** Enable the **Engineered Projects Only** filter, which uses
    license type, release count, contributor count, pull request activity, and code ratio
    to exclude non-software repos before cloning. This is the most robust solution but adds
    processing time. A lighter alternative is to inspect the Dataset Statistics panel after
    a run and adjust your filters accordingly.
    """)

with st.expander("Why are most of my records from one repository?"):
    st.markdown("""
    Without caps, the corpus reflects the natural size distribution of the repos — a large
    algorithm collection like `TheAlgorithms/Python` can contribute thousands of records
    while a smaller repo contributes tens.

    When you set **Max Functions** or **Max Classes**, SIEVE applies stratified sampling
    to preserve proportional representation. However, if the corpus is dominated by one
    repo, even proportional sampling will allocate most slots to it — this is mathematically
    correct but may not suit all research designs.

    For balanced, equal-per-repo sampling, set a small **Max Repos** value and choose your
    repos carefully, or use the **Engineered Projects** filter which tends to produce a more
    diverse corpus by excluding non-software repos.
    """)

with st.expander("What does the skeleton look like for an Enum or dataclass with no methods?"):
    st.markdown("""
    Classes with no method definitions (e.g. `Enum` subclasses, simple dataclasses, constants
    containers) get a skeleton with `pass` as the body:

    ```python
    class State(Enum):
        pass
    ```

    This is valid Python and makes the skeleton usable as a prompt without modification.
    The full source code in `source_code` still contains all the assignments and enum values.
    """)

with st.expander("What does 'indentation-normalized' mean?"):
    st.markdown("""
    Tree-sitter returns node text starting at the first token. For a top-level function,
    this means 4-space body indentation as expected. For a method inside a class, the raw
    text has 8-space body indentation (the absolute position in the file).

    SIEVE normalizes this by measuring the minimum indentation of body lines and stripping
    the excess, so all extracted code — whether top-level functions or deeply nested methods —
    has a consistent 4-space body indent. Relative indentation within the body (nested loops,
    conditionals, etc.) is preserved.
    """)

with st.expander("Why does the line number say 'body lines, excl. imports'?"):
    st.markdown("""
    `start_line` and `end_line` record the position of the function or class definition
    within its source file. They do **not** account for import statements, which are at the
    top of the file and may be prepended to the record in the Sample Viewer for display
    purposes.

    These line numbers are useful for locating the original code in the repository for
    manual verification — navigate to `file_path` in the repo at `github.com/{repo}` and
    go to line `start_line`.
    """)

with st.expander("What GitHub API rate limits should I expect?"):
    st.markdown("""
    Without a token: 10 requests/minute (search) and 60 requests/hour (REST).
    This will be exhausted quickly for any non-trivial run.

    With a Personal Access Token (PAT): 30 requests/minute (search) and 5,000 requests/hour
    (REST). A PAT with no scopes (public repo access only) is sufficient — SIEVE only reads
    public repository data.

    SIEVE sleeps 0.5 seconds between contributor count fetches and 0.2 seconds between
    per-repo processing to stay within limits. For large runs (50+ repos), expect the
    discovery phase to take several minutes.

    To create a PAT: GitHub → Settings → Developer settings → Personal access tokens →
    Tokens (classic) → Generate new token → no scopes required → copy into the token field.
    """)

with st.expander("Which languages are supported?"):
    st.markdown("""
    **Python, Java, JavaScript, and C++** are all fully supported for function and class
    extraction. All four use tree-sitter grammars under the hood.

    **Python** extracts `function_definition` and `class_definition` nodes, including
    async functions, decorators (`@staticmethod`, `@dataclass`, etc.), type annotations,
    and default parameter values.

    **Java** extracts `method_declaration`, `constructor_declaration`, and `class_declaration`
    nodes, including annotations (`@Override`, `@SuppressWarnings`), Javadoc comments,
    generic type parameters, and `extends`/`implements` relationships.

    **JavaScript** extracts `function_declaration`, `method_definition`, `class_declaration`,
    and arrow functions assigned to `const`/`let`/`var`. Handles default imports, named
    imports, and namespace imports (`import * as`). Minified and bundled files are
    automatically skipped. TypeScript (`.ts`, `.tsx`) files are processed using the
    JavaScript grammar — type annotations are preserved in parameter strings but not
    separately parsed.

    **C++** extracts free functions and class methods, including pointer/reference return
    types, template functions, namespace-qualified names, and qualified method names
    (e.g. `Log::LogMessage::valid`). Uses `#include` detection for used imports.

    Discovery and repository filtering support all four languages at the GitHub search level.
    """)

with st.expander("How is 'used imports' different from all imports in the file?"):
    st.markdown("""
    A file may import 20 modules, but a given function might only use 3 of them. SIEVE
    filters the file-level imports down to only those whose introduced names appear as
    whole tokens in the function or class source code.

    For example, if a file has `import os`, `import re`, and `from pathlib import Path`,
    but a function only uses `Path`, only `from pathlib import Path` will appear in
    `used_imports` for that function.

    This keeps snippets minimal and avoids cluttering prompts with irrelevant imports.
    """)

with st.expander("What is the LLM Score field?"):
    st.markdown("""
    `llm_score` is P(AI-generated) for the extracted snippet — a probability in [0, 1]
    produced by a fine-tuned [CodeBERT](https://huggingface.co/microsoft/codebert-base)
    classifier integrated directly into SIEVE.

    Enable it via the **Annotate LLM Score** toggle before running. The classifier weights
    (~500MB) are downloaded automatically from HuggingFace Hub on first use.

    See the **About the Classifier** page for full details on training data, architecture,
    and performance metrics (F1=0.9478, AUROC=0.9902 on a held-out test set of 11,419
    samples across Python, Java, JavaScript, and C++).
    """)

st.divider()

# ─── Dependency Graph ─────────────────────────────────────────────────────────

st.header("Dependency Graph")
st.markdown("""
After a pipeline run, the **Dataset Statistics** panel includes an interactive
dependency graph for each processed repository. The graph shows direct package
dependencies parsed from the repository's manifest files, colored by kind:

| Color | Kind | Description |
|---|---|---|
| 🔵 Blue | Main | Runtime dependencies |
| 🟣 Purple | Dev | Development/test dependencies |
| 🟠 Orange | Optional | Peer or optional dependencies |

The center node (red) represents the repository itself. Drag nodes to explore
the graph; use scroll to zoom.

**Supported manifest formats:**

| Language | Files Parsed |
|---|---|
| Python | `requirements*.txt`, `requirements/*.txt`, `pyproject.toml` (PEP 508 + Poetry), `setup.cfg` |
| JavaScript | `package.json` — `dependencies`, `devDependencies`, `peerDependencies` |
| Java | `pom.xml` (Maven) — `<dependency>` blocks, scope-aware |
| C++ | `conanfile.txt` (Conan `[requires]`), `vcpkg.json`, `CMakeLists.txt` (`find_package`) |

Only **direct** dependencies are shown — transitive dependencies require
network access to package registries and are not computed.

Dependencies are stored in the corpus `manifest.json` under each repo's
`dependencies` key, so they can be used for downstream analysis even without
the UI.
""")

st.divider()

# ─── C++ Parsing Performance ──────────────────────────────────────────────────

st.header("C++ Parsing Performance")
st.markdown("""
C++ extraction is significantly slower than Python, Java, and JavaScript. This is
a known limitation of tree-sitter's C++ grammar, not a SIEVE-specific issue.
Three factors contribute:

**1. Inherent grammar ambiguity.** The C++ language contains fundamental syntactic
ambiguities that cannot be resolved without type information. For example,
`a * b` is syntactically ambiguous between a multiplication expression and a
pointer-type declaration — the correct parse depends on whether `a` is a variable
or a type name, which may be defined in an included header file not available
at parse time. The `tree-sitter-cpp` grammar explicitly documents this class of
ambiguity ([issue #74](https://github.com/tree-sitter/tree-sitter-cpp/issues/74)).

**2. GLR parsing overhead.** Tree-sitter resolves C++ ambiguities at runtime using
the Generalized LR (GLR) algorithm, which explores multiple parse trees in parallel
and merges them when they reconverge. This parallel exploration is more expensive
than the deterministic LR(1) parsing used for Python, Java, and JavaScript.
Tree-sitter's documentation notes that performance is best when grammars are close
to the LR(1) class — C++ is far from it.

**3. Grammar size.** The `tree-sitter-cpp` grammar is substantially larger and more
complex than the grammars for the other three supported languages. C++ is widely
recognized as one of the most difficult languages to parse due to its size and
context sensitivity.

**Practical impact on SIEVE:** C++ repositories take 2–5× longer to extract than
Python or JavaScript repositories of comparable size. This affects the two-pass
extraction (both Pass 1 counting and Pass 2 extraction), and is most noticeable
on large repos like `godotengine/godot` or `opencv/opencv`. Setting a reasonable
`Max Repos` cap (≤ 20) and using per-repo function/class caps mitigates the
impact.

**References:**
- tree-sitter-cpp issue tracker: [C++ grammar is ambiguous](https://github.com/tree-sitter/tree-sitter-cpp/issues/74)
- Semgrep (2024): [Modernizing Static Analysis for C/C++](https://semgrep.dev/blog/2024/modernizing-static-analysis-for-c/) — describes GLR overhead in C++ parsing
- Mathew (2025): [Designing Effective Tree-sitter Grammars](https://medium.com/@linz07m/designing-effective-tree-sitter-grammars-84411ebdf830) — LR(1) vs GLR performance tradeoffs
""")

st.divider()

# ─── How Detection Works ──────────────────────────────────────────────────────

st.header("How Test Suite Detection Works")
st.markdown("""
SIEVE automatically detects whether a repository has a test suite using a
**four-signal heuristic**. Each signal that fires contributes one point; a
repo is considered to have a test suite if **≥ 2 signals** fire.

| Signal | Examples |
|---|---|
| **Test directory** | `test/`, `tests/`, `__tests__/`, `spec/`, `specs/` at any depth |
| **Test file naming** | `test_*.py`, `*_test.py`, `*_test.cpp`, `*.test.js`, `*.spec.ts`, `Test*.java` |
| **Test runner config** | `pytest.ini`, `setup.cfg` with `[tool:pytest]`, `jest.config.js`, `karma.conf.js`, `pom.xml` with Surefire, `CMakeLists.txt` with `enable_testing()` |
| **CI workflow** | `.github/workflows/*.yml` or `.circleci/config.yml` containing `pytest`, `jest`, `mvn test`, `ctest`, `cargo test`, or similar keywords |

The two-signal threshold avoids false positives from repos that have a single
test file but no real test infrastructure, while still catching repos that use
non-standard directory layouts.

Test suite presence is also one of the signals used by the **Engineered Projects**
filter (Munaiah et al., 2017) — repos without any test infrastructure are more
likely to be personal or experimental projects rather than production software.
""")

st.divider()

# ─── License ─────────────────────────────────────────────────────────────────

st.header("License")
st.markdown("""
SIEVE is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Musfiqur Rahman, Emad Shihab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The full license is also available in the
[GitHub repository](https://github.com/mrsumitbd/sieve/blob/main/LICENSE).
""")

st.divider()
st.caption("SIEVE v0.1.0 · Built for SE research · [GitHub](https://github.com/mrsumitbd/sieve)")