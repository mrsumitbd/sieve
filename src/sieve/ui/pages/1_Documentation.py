"""
pages/1_📖_Documentation.py

Documentation and FAQ page for SIEVE.
Automatically discovered by Streamlit's multipage feature when placed in pages/.
"""

import streamlit as st

from sieve import __version__

st.set_page_config(
    page_title="SIEVE — Documentation",
    page_icon="📖",
    layout="wide",
)

st.title("📖 SIEVE Documentation")
st.caption("**S**oftware **I**ngestion & **E**xtraction for **V**erifiable **E**valuation")
st.info(
    "**The public Hugging Face Space has 8 GB of RAM.** Do not run large corpora there. "
    "Run the web UI on your own machine (see **Run the Web UI Locally** below) or use the CLI."
)
st.divider()

# ─── What is SIEVE? ───────────────────────────────────────────────────────────

st.header("What is SIEVE?")
st.markdown("""
SIEVE is a **parameterized GitHub corpus builder** for software engineering
research. Rather than a static, pre-built dataset, SIEVE is a *tool* you run
with your own settings — programming language, granularity (functions/methods
vs. classes), repository quality, and repository creation date — to build the
corpus a given study actually needs, built from GitHub on each run.

SIEVE extracts two types of program units:
- **Functions** — CodeSearchNet-style records with full source, signature, parameters, return type, and docstring.
- **Classes** — OpenClassEval-style records with full source, skeleton, method list, and inheritance.

Both record types include the import statements used within the extracted unit,
enabling self-contained code snippets for prompting or evaluation. Every record
is also annotated with 22 structural code metrics, an optional AI-generation
likelihood score, parsed dependency information, and (optionally) full AST
features — supporting research well beyond any single use case.
""")

st.divider()

# ─── Use Cases ─────────────────────────────────────────────────────────────────

st.header("Use Cases")
st.markdown("""
The same underlying pipeline supports several kinds of SE research, depending
on which parameters you configure:

- **Contamination-aware LLM evaluation.** Filter repositories by *creation date* (not last push date) so that every sampled repository is newer than a chosen date such as a model's release date; see **Contamination-Aware Evaluation** below for the rationale, its limits, and a cutoff-date reference table.
- **Dependency-management research.** Every processed repository's package manifest is parsed automatically; the corpus manifest and the **Dependency Graph** panel expose these for supply-chain and library-usage studies.
- **Structural analysis of LLM-generated code.** Enable **Annotate LLM Score** and **Export Full AST** to get a P(AI-generated) probability and the full parse tree on every record (the summary AST features are always exported) — useful for studying how AI-authored and human-authored code differ structurally.
- **General-purpose corpus building.** The creation-date filter is equally useful for bounding a corpus to a specific development era for longitudinal studies, independent of any LLM, or simply source a richly-annotated, deduplicated corpus for any language-specific study.
""")

st.divider()

# ─── Contamination-Aware Evaluation ────────────────────────────────────────────

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
   Contamination in the evaluation set makes detection appear more accurate than it is.
   This is a growing concern for evaluating LLMs on code, and reviewers increasingly
   ask authors to address it.

### How SIEVE Solves This

SIEVE filters repositories by **creation date**, not last push date. A repository
created after an LLM's training cutoff did not exist when the model was trained, so
restricting discovery this way makes the corpus temporally bounded relative to that
model and reduces the most direct route for contamination. This bounds the age of
the *repository*; SIEVE does not verify the origin of individual snippets, and a
new repository can still contain code copied or migrated from older sources.

### How to Choose Your Cutoff Date

**Use the model's public release date, not its knowledge cutoff, when you need
the most conservative bound.** These are often confused but are different: the
knowledge cutoff is the last date of data used for *pre-training*, while the
release date is when the model became publicly available. Between the two,
models are typically fine-tuned on additional data (instruction-following
examples, RLHF preference data, sometimes supervised fine-tuning on code) —
data that postdates the pre-training cutoff but predates the release. Setting
**Repo Creation Lower Bound** to the release date restricts your corpus to repositories created
after both the pre-training *and* fine-tuning phases; setting it to the knowledge
cutoff only covers the former.

If studying multiple models simultaneously, use whichever date (cutoff or
release, per the column you're using) is **latest** among them, so that the
sampled repositories were created after all of the models' training phases.

**Reference table.** Verified against official provider documentation and
model cards; "—" means the provider has not publicly disclosed a knowledge
cutoff for that model. Coverage is necessarily a snapshot — always check the
provider's own model card for anything released after this table was last
updated, or for a model not listed here.

**OpenAI**

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| GPT-3.5 Turbo | Nov 2022 | Sep 2021 |
| GPT-4 | Mar 2023 | Sep 2021 |
| GPT-4 Turbo | Nov 2023 | Apr 2023 |
| GPT-4o | May 2024 | Oct 2023 |
| GPT-4.5 | Feb 2025 | Oct 2023 |
| GPT-4.1 | Apr 2025 | Jun 2024 |
| o1 | Dec 2024 | Oct 2023 |
| o3 | Apr 2025 | Jun 2024 |
| GPT-5 | Aug 2025 | Sep 2024 |
| GPT-5.1 | Nov 2025 | Sep 2024 |

**Anthropic** (knowledge cutoff shown is the "reliable" cutoff for Claude 4+ models, per Anthropic's own distinction between reliable and broader training-data cutoffs)

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| Claude 3 (Haiku/Sonnet/Opus) | Mar 2024 | Aug 2023 |
| Claude 3.5 Sonnet | Jun 2024 | Apr 2024 |
| Claude 3.5 Haiku | Nov 2024 | Jul 2024 |
| Claude 3.7 Sonnet | Feb 2025 | Oct 2024 |
| Claude Opus 4 / Sonnet 4 | May 2025 | Mar 2025 |
| Claude Sonnet 4.5 | Sep 2025 | Jan 2025 |
| Claude Haiku 4.5 | Oct 2025 | Feb 2025 |
| Claude Opus 4.5 | Nov 2025 | May 2025 |
| Claude Sonnet 5 | Jun 2026 | Jan 2026 |
| Claude Opus 5 | Jul 2026 | May 2026 |

**Google**

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| Gemini 1.0 Pro | Dec 2023 | Feb 2023 |
| Gemini 1.5 Pro | Feb 2024 | May 2024 |
| Gemini 1.5 Flash | May 2024 | May 2024 |
| Gemini 2.0 Flash | Dec 2024 | Aug 2024 |
| Gemini 2.5 Pro | Mar 2025 | Jan 2025 |
| Gemini 2.5 Flash | Apr 2025 | Jan 2025 |

**Meta**

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| Llama 2 | Jul 2023 | Sep 2022 |
| Code Llama | Aug 2023 | Sep 2022 |
| Llama 3 (8B / 70B) | Apr 2024 | Mar 2023 / Dec 2023 |
| Llama 3.1 | Jul 2024 | Dec 2023 |
| Llama 3.2 | Sep 2024 | Dec 2023 |
| Llama 3.3 | Dec 2024 | Dec 2023 |
| Llama 4 (Scout / Maverick) | Apr 2025 | Aug 2024 |

**Mistral AI** (Mistral does not publicly disclose knowledge cutoffs for most models)

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| Mistral 7B | Sep 2023 | — |
| Mixtral 8x7B | Dec 2023 | — |
| Mixtral 8x22B | Apr 2024 | — |
| Codestral | May 2024 | — |
| Mistral Large 2 | Jul 2024 | — |

**DeepSeek**

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| DeepSeek-Coder | Nov 2023 | Mar 2023 |
| DeepSeek-Coder-V2 | Jun 2024 | Nov 2023 |
| DeepSeek-V3 | Dec 2024 | Jul 2024 |
| DeepSeek-R1 | Jan 2025 | Jul 2024 |

**Qwen (Alibaba)**

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| Qwen2.5 | Sep 2024 | Late 2023 |
| Qwen2.5-Coder | Sep 2024 | Late 2023 |

**xAI** (does not publicly disclose knowledge cutoffs for Grok-1)

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| Grok-1 | Nov 2023 | — |
| Grok-3 | Feb 2025 | Nov 2024 |
| Grok-4 | Jul 2025 | Nov 2024 |

**BigCode** (does not publicly disclose a single knowledge cutoff date)

| Model | Release Date | Knowledge Cutoff |
|---|---|---|
| StarCoder | May 2023 | — |
| StarCoder2 | Feb 2024 | — |

### Reviewer Checklist

When submitting papers that use SIEVE-generated corpora, you can state:

> *"Our evaluation corpus was built using SIEVE, restricted to GitHub
> repositories created after [DATE] (the release date of [Model]). This bounds
> repository age relative to the evaluated models; it does not verify the origin
> of individual snippets, which we discuss as a limitation."*

Stating the bound and its limits explicitly lets reviewers judge the contamination risk themselves.
""")

st.divider()

# ─── Run locally ──────────────────────────────────────────────────────────────

st.header("Run the Web UI Locally")
st.warning(
    "The public Hugging Face Space has **8 GB of RAM**. Large runs (many repositories, C++, "
    "full ASTs, LLM scoring) can exhaust it. Run them on your own machine."
)
st.markdown("""
Repositories are processed one after another, not in parallel, so a large run is slow
whatever the hardware.

**Prerequisites**

- **Python 3.11 or newer** (`requires-python = ">=3.11"` in `pyproject.toml`). Check with `python --version`.
- **git**, because SIEVE clones each repository with `git clone --depth 1`.
- A **GitHub token** is optional but recommended. Without one GitHub allows 60 requests per hour.

**Steps**

1. Get the code and enter the repository root:
   ```bash
   git clone https://github.com/mrsumitbd/sieve.git
   cd sieve
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
   ```
3. Install the dependencies. Run this from the repository root: the first line of
   `requirements.txt` is `-e .`, which installs SIEVE itself and the `sieve` command.
   The file also lists PyTorch and Transformers (used only for LLM scoring), which are large downloads.
   ```bash
   pip install -r requirements.txt
   ```
4. Create a classic GitHub token with the `public_repo` scope at
   [github.com/settings/tokens](https://github.com/settings/tokens). **The web UI does not read
   `GITHUB_TOKEN`**: paste the token into the **GitHub Token (PAT)** field in the sidebar after the app
   opens. (`GITHUB_TOKEN` and `HF_TOKEN` are read by the CLI only.)
5. Start the app from the repository root:
   ```bash
   streamlit run src/sieve/ui/Home.py
   ```
6. Open **http://localhost:8501** in a browser (Streamlit's default port; the terminal prints the
   address, and Streamlit may open it for you). To use another port, add `--server.port 8600`.
7. Set the parameters in the sidebar and click **▶ Run SIEVE**. The web UI writes its output to a
   temporary directory and offers it through the **Download Corpus** buttons. To write straight to a
   directory of your choice, use the CLI with `--output-dir`.
8. Stop the app with `Ctrl+C` in the terminal that is running it, then leave the virtual environment
   with `deactivate`.
""")

st.divider()

# ─── Install troubleshooting ──────────────────────────────────────────────────

st.header("Install Troubleshooting")
st.markdown("""
SIEVE needs Python 3.11 or newer. The repository pins no upper versions: `pyproject.toml` and
`requirements.txt` give lower bounds only (for example `numpy>=1.26.0`, `torch>=2.1.0`,
`transformers>=4.38.0`). On Windows, Linux and Apple-silicon Macs a plain
`pip install -r requirements.txt` needs nothing more.

**What is installed where.** `pyproject.toml` lists the core dependencies. `requirements.txt` adds
`torch`, `transformers`, `huggingface_hub` and `sentencepiece`, which only **Annotate LLM Score** uses.
If they are missing, the scoring step reports an error and the run continues without scores.

**Intel Macs (x86_64).** PyPI has no PyTorch wheels for macOS x86_64 after **2.2.2**, so pip installs
torch 2.2.2 there. With the other requirements left unpinned, pip also picks NumPy 2 and Transformers 5,
and neither works with torch 2.2.2:

- **NumPy 2:** torch 2.2.2 was built against NumPy 1.x. It prints
  `Failed to initialize NumPy: _ARRAY_API not found` and its NumPy interop does not work.
- **Transformers 5:** version 5.22.2 refuses to load the model with
  `RobertaModel requires PyTorch >= 2.6`.

Install the compatible versions explicitly:

```bash
pip install "torch==2.2.2" "numpy<2" "transformers<5"
```

With torch 2.2.2, `numpy<2` and `transformers<5` (4.57.6 was used), SIEVE's classifier and export
tests pass. This was checked by running the tests with those versions on an Apple-silicon Mac;
it was not run on Intel hardware. If you do not use **Annotate LLM Score**, none of these pins
matter for the rest of SIEVE.

**`sieve: command not found`.** Activate the virtual environment, or run
`python -m sieve.cli run ...` instead.
""")

st.divider()

# ─── Parameters ───────────────────────────────────────────────────────────────

st.header("Parameter Reference")

st.subheader("Date Range")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Start Date** | Only include repos **created** on or after this date (web UI field: **Repo Creation Lower Bound**), e.g. an LLM's release date for contamination-aware evaluation, or any reference date to bound a corpus to a development era for other studies. |
| **End Date** | Only include repos **created** on or before this date (web UI field: **Repo Creation Upper Bound**). Default: 30 days before today. Filters out very new repos with little code. |

The GitHub search query combines `created:START..END` with `stars:>=MIN_STARS`,
`fork:false`, and `archived:false`, and sorts results by stars in descending order.
It filters on **repository creation date**, not last push date. This distinction matters
whenever you need to know when a repository came into existence, not just when
it was last touched: for contamination-aware corpus building, a repo created
after an LLM's cutoff did not exist when it was trained (though SIEVE does not
check whether individual snippets were copied from older sources); for longitudinal
studies, the same filter lets you bound a corpus to a specific development era
regardless of any LLM.

**What this establishes:** If Repo Creation Lower Bound = Jan 1 2024, every extracted function and
class comes from a repository created on or after Jan 1 2024. SIEVE does not verify
the origin of individual snippets, so code in a new repository may still have been
copied or migrated from older sources.
""")

st.subheader("Repository Filters")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Min Stars** | Minimum GitHub star count. Higher values bias toward popular, well-maintained projects but reduce diversity. |
| **Min Contributors** | Minimum number of human contributors, at least 1. Helps exclude personal or toy projects. Bots are not counted, and the count is exact (see **Repository Discovery**). |
| **Max Repos** | Cap on the number of repos to process. Applies during discovery, before cloning. Web UI: `0` = no cap (default 20). CLI and config file: omit it (`null`) for no cap, otherwise it must be at least 1. |
| **Max Functions** | Cap on total extracted functions *after deduplication*. Web UI: `0` = no cap (default 30). CLI and config file: omit it for no cap, otherwise at least 1. When the corpus exceeds the cap, stratified sampling is applied (see below). |
| **Max Classes** | Cap on total extracted classes *after deduplication*. Same rules and stratified sampling as **Max Functions**. |

The `0` = no cap convention exists in the web UI only. The CLI and config reject `0` and negative
numbers. To skip functions or classes, do not extract that level (see **Granularity** below).
""")

st.subheader("Content Filters")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Granularity** | Which program units to extract: functions, classes, or both (default both). CLI: `--granularity function`, `class` or `both`. Config file: `granularity`, a list such as `["function"]`. Web UI: the **Extract functions** and **Extract classes** checkboxes (at least one must stay checked; the cap of an unchecked level is disabled). |
| **Engineered Projects Only** | Applies the engineered project filter of Xiao et al. (2025), in the spirit of Reaper (Munaiah et al., 2017). See section below. |
""")

st.subheader("Deduplication")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Deduplicate** | Enable MinHash LSH near-duplicate removal across all extracted records. **Off by default in every interface** (CLI `--deduplicate`, config `deduplicate`, web UI **Deduplicate** toggle). |
| **Similarity Threshold** | Jaccard similarity threshold for MinHash deduplication (0.0–1.0; the web UI slider covers 0.5–1.0). Higher values are more conservative — only near-identical code is removed. Default: 0.8. Used only when deduplication is on. |

Deduplication tokenizes source code (stripping string literals and normalizing whitespace) before
hashing, so superficial differences like renamed variables or changed comments do not prevent
duplicate detection.
""")

st.subheader("Annotation and Output")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **Annotate LLM Score** | Score each record with P(AI-generated). CLI `--annotate-llm-score`, config `annotate_llm_score`. Default off. Downloads the classifier weights on first use. |
| **Export Full AST** | Include the full AST as nested JSON in each record's `ast` field. CLI `--export-ast`, config `export_ast`, web UI **Export Full AST** toggle. Default off; it increases file size significantly. |
| **Export Format** | `jsonl` (default), `parquet` or `both`. |
| **Output Directory** | CLI `--output-dir`, config `output_dir` (default `./sieve_output`). The web UI has no such field: it writes to a temporary directory and offers downloads. |

The AST summary features `ast_depth`, `ast_num_nodes` and `ast_node_types` are **always** exported.
Only the full AST JSON (`ast`) depends on **Export Full AST**.
""")

st.subheader("Tokens and Rate-Limit Waiting")
st.markdown("""
| Parameter | Description |
|-----------|-------------|
| **GitHub token** | CLI: `--github-token`, or the `GITHUB_TOKEN` environment variable. Config file: `github_token`. Web UI: the **GitHub Token (PAT)** field only. Without a token GitHub allows 60 requests per hour. |
| **Hugging Face token** | Optional; only used to download the classifier weights. Config file: `hf_token`. CLI: the `HF_TOKEN` environment variable (there is **no** `--hf-token` flag). The web UI has no field for it. |
| **Wait for rate limit** | CLI `--wait-for-rate-limit`, config `wait_for_rate_limit`. Default off. Not available in the web UI, which never waits. See **GitHub Rate Limits**. |

How the CLI resolves tokens:

- `GITHUB_TOKEN` and `HF_TOKEN` are read from the environment with individual flags **and** with
  `--config`. When SIEVE is installed from the repository as described above, a `.env` file in the
  repository root is loaded into the environment too (`.env.example` shows the format).
- The environment fills a token only where nothing else set it. A token written in the config file
  (or given with `--github-token`) therefore takes precedence over the variable. With `--config`, the
  `--github-token` flag is ignored along with every other flag.
- Tokens are never written to `manifest.json`, and never appear in the config printed at the start of
  a run or by `sieve validate-config`, even when the config file contains one. A config file is plain
  text on disk, so prefer the environment variable.
- The web UI reads neither variable; paste the GitHub token into the sidebar.
""")

st.subheader("Parameter Parity Across Interfaces")
st.markdown("""
Every setting, with its CLI flag, config-file key and web UI control. Defaults are given as
CLI and config / web UI. "none" means no cap.

| Parameter | CLI flag | Config key | Web UI control | Default (CLI and config / web UI) |
|-----------|----------|------------|----------------|-----------------------------------|
| Language | `--language` | `language` | Programming Language | required / Python |
| Start date | `--start-date` | `start_date` | Repo Creation Lower Bound | required / 2024-01-01 |
| End date | `--end-date` | `end_date` | Repo Creation Upper Bound | CLI: 30 days before today; config: required / 30 days before today |
| Minimum stars | `--min-stars` | `min_stars` | Minimum Stars | 10 / 50 |
| Minimum contributors | `--min-contributors` | `min_contributors` | Minimum Contributors | 1 / 5 |
| Max repos | `--max-repos` | `max_repos` | Max Repos (0 = no cap) | none / 20 |
| Max functions | `--max-functions` | `max_functions` | Max Functions (0 = no cap) | none / 30 |
| Max classes | `--max-classes` | `max_classes` | Max Classes (0 = no cap) | none / 30 |
| Granularity | `--granularity function`, `class` or `both` | `granularity` (list) | Extract functions, Extract classes (checkboxes) | both / both |
| Engineered projects only | `--engineered-only` / `--no-engineered-only` | `engineered_only` | Engineered Projects Only | off / off |
| Annotate LLM score | `--annotate-llm-score` / `--no-annotate-llm-score` | `annotate_llm_score` | Annotate LLM Score | off / off |
| Export full AST | `--export-ast` / `--no-export-ast` | `export_ast` | Export Full AST | off / off |
| Deduplicate | `--deduplicate` / `--no-deduplicate` | `deduplicate` | Deduplicate | off / off |
| Dedup threshold | `--dedup-threshold` | `dedup_threshold` | Dedup Similarity Threshold (0.5 to 1.0) | 0.8 / 0.8 |
| Export format | `--export-format` | `export_format` | Export Format | jsonl / jsonl |
| Output directory | `--output-dir` | `output_dir` | none (temporary directory, Download Corpus buttons) | ./sieve_output / not applicable |
| GitHub token | `--github-token`, `GITHUB_TOKEN` | `github_token` | GitHub Token (PAT) | none / none |
| Hugging Face token | `HF_TOKEN` (no flag) | `hf_token` | none | none / none |
| Wait for rate limit | `--wait-for-rate-limit` / `--no-wait-for-rate-limit` | `wait_for_rate_limit` | none (never waits) | off / off |
| Config file | `--config` | not applicable | none | none |

`sieve validate-config FILE` checks a config file without running anything. The config file rejects
unknown keys, so a misspelled key is an error rather than being ignored.
""")

st.divider()

# ─── Discovery ────────────────────────────────────────────────────────────────

st.header("Repository Discovery")
st.markdown("""
SIEVE finds repositories with GitHub's repository search, using the qualifiers `language`,
`created:START..END`, `stars:>=MIN_STARS`, `fork:false` and `archived:false`, sorted by stars,
highest first. GitHub returns at most **1,000 results per search query**, however many repositories
match.

**How results beyond the 1,000-result limit are collected**

- **Max Repos is 1,000 or less:** SIEVE runs one query. The first 1,000 results are the exact top
  repositories by stars, and repositories that fail the contributor filter are skipped as they come.
  If all 1,000 results were scanned and fewer than Max Repos passed the filters, SIEVE logs a warning
  and marks the discovery incomplete, because more matches may exist beyond the limit.
- **No cap, or Max Repos above 1,000:** SIEVE splits the creation-date range at its midpoint day,
  recursively, into whole-day windows with no gap and no overlap, until each window holds at most
  1,000 results. It uses the true total that GitHub reports for each window.
  - With a cap above 1,000, candidates from every window are collected and ranked by stars (ties by
    repository name); the first Max Repos that pass the contributor filter are kept, which is the exact
    top by stars.
  - With no cap, repositories are taken window by window, **oldest window first**. Stars order holds
    only within a window.
- **A single day with more than 1,000 matching repositories** cannot be split further. SIEVE collects
  that day's 1,000 most-starred repositories, records the day as **truncated**, and marks the
  discovery incomplete. Nothing is dropped silently.

**Contributor counts**

- The count is exact: SIEVE lists the repository's contributors (100 per page, so
  `ceil(n / 100)` requests) and does not stop early. Bots (a login ending in `[bot]`, or account type
  `Bot`) and anonymous email-only contributors are not counted.
- When GitHub refuses to list the contributors because the list is too large, the count is `null`.
  The repository is kept (it passes any **Min Contributors** threshold) and its name is listed in
  `discovery.contributors_unknown`.
- If the contributors cannot be fetched at all, even after retrying server and connection errors up to
  three times, the repository is skipped and listed in `discovery.excluded` with the reason. Such a run
  is marked incomplete.

**The `discovery` block of `manifest.json`**

| Field | Meaning |
|-------|---------|
| `complete` | `true` when every matching repository was reachable: no single day exceeded 1,000 results, GitHub did not flag a count as incomplete, no repository was excluded, and no error or rate limit stopped discovery. A **Max Repos** cap does not make a run incomplete. |
| `total_matching` | Repositories GitHub reports for the filters. For a single-query run (Max Repos 1,000 or less) GitHub is not asked for a total, and this is the number of search results scanned. |
| `slices` | Number of search queries (windows) run. |
| `truncated_windows` | Single days that held more than 1,000 repositories, each as `{start, end, total_count}`. Only the 1,000 most-starred of each were collected. |
| `incomplete_results` | `true` if GitHub flagged a result count as incomplete (its search timed out, so results may be missing). |
| `excluded` | Repositories dropped because their contributors could not be fetched, each as `{repo, reason}`. |
| `contributors_unknown` | Repositories kept although GitHub would not list their contributors. |
| `rate_limit_waits` | `{count, seconds}`: how often the run waited out a rate limit and for how long. Zero unless **Wait for rate limit** is on. |

If a rate limit stops discovery after some repositories were found, the run continues with those,
`complete` is set to `false`, and the run summary reports `rate_limited` (the CLI prints a warning
and the web UI shows one).
""")

st.divider()

# ─── Rate limits ──────────────────────────────────────────────────────────────

st.header("GitHub Rate Limits")
st.markdown("""
**What the limits are.** Without a token GitHub allows 60 REST requests per hour and 10 search requests
per minute; with a token, 5,000 per hour and 30 per minute. Search requests (including the result counts
used to split date windows) count against the search limit, everything else against the REST limit.
Per accepted repository discovery spends about `ceil(n / 100)` requests listing its `n` contributors
plus one for the latest commit, and every scanned repository that is rejected still costs its
contributor pages. SIEVE plans on about 5 requests per accepted repository, so a run without a token
covers roughly 12 repositories. Create a classic token with the `public_repo` scope, as SIEVE's own
error messages ask.

**Default: fail fast.**

- Before any work SIEVE reads the quota (this read costs nothing). If fewer than 10 REST requests
  remain, the run stops at once with an error giving the reset time.
- A primary limit that resets within about a minute (the usual case for the search limit) is waited
  out automatically. One that would take longer raises an error immediately, with the reset time,
  instead of sleeping for up to an hour. Secondary limits (too many requests too quickly) are retried a
  few times by the GitHub client, 30 seconds apart, before the run stops.
- If the limit is hit after some repositories were found, the run continues with them and the results
  are marked partial. If none were found yet, the run ends with an error (CLI exit code 1).

**`--wait-for-rate-limit`.** The CLI flag (config key `wait_for_rate_limit`) makes the run sleep and
resume. On a primary limit it sleeps until the reset time plus a 5-second margin, then retries the same
request. On a secondary limit it sleeps for the `Retry-After` time GitHub sends (60 seconds if absent).
Paginated listings resume where they stopped, with nothing repeated or skipped. A request gives up
after 5 waits in a row. Progress lines show the time left. A single wait can last up to an hour,
`Ctrl+C` stops the run, and nothing is written until the run ends, so stopping loses the run. The
number of waits and the seconds slept are recorded in `discovery.rate_limit_waits`.

**The web UI never waits.** It always runs with waiting off, because an hour-long sleep would hang the
hosted Space. A rate limit there ends the run with an error or a partial-results warning.

**Request pacing.** Requests go out back to back while plenty of quota is left. Once the remaining
quota falls to a tenth of the window (6 of 60, 500 of 5,000), discovery pauses 1 second after each
repository. The figure comes from the headers of the last response, so it costs no request. There are
also fixed short pauses: 0.3 seconds between the per-repository lookups of the engineered filter and
0.2 seconds between repositories during extraction.

**Repositories are processed one after another, not in parallel.**
""")

st.divider()

# ─── Stratified Sampling ──────────────────────────────────────────────────────

st.header("Corpus Size Caps & Stratified Sampling")
st.markdown("""
When **Max Functions** or **Max Classes** is set, SIEVE does not draw a simple random sample.
Before extracting (Pass 1), it counts the available records in each repository. The count is
**exact**: it covers the same files and the same function and class definitions that extraction
(Pass 2) will find, so with no caps Pass 1 and Pass 2 give the same numbers (the one gap is listed
under **Known Limitations**). SIEVE then splits the cap across repositories as follows:

1. Every repository that has at least one record first gets **1 slot**, so small
   repositories are not rounded down to nothing.
2. The remaining slots (cap minus the number of non-empty repositories) are divided in
   proportion to each repository's **remaining records** (its count minus the one slot
   already given). Whole shares are assigned first; any leftover slots go to the
   repositories with the largest fractional remainders (largest-remainder method, using
   integer arithmetic).
3. Ties are broken deterministically: the repository with more records wins, then the
   repository name in alphabetical order. The result does not depend on the order in which
   repositories were processed.
4. If the cap is **smaller than the number of non-empty repositories**, the minimum of one
   cannot be met. The largest repositories get 1 slot each (ties by name) and the rest get 0.

No repository is ever allocated more records than it has, and the allocations add up to the
cap (or to the total available, if that is smaller). Within a repository, the allocated
records are the first ones found while walking its files in a fixed order (see **What Extraction
Skips**), not a random draw, so the same repository state gives the same records. Random sampling
without replacement is used only in a final safety-net step, which applies if the corpus still
exceeds the cap after deduplication.

**Shortfall (Pass 3).** Pass 2 extracts more than each repository's allocation (twice the cap, at most
50 more) and keeps the surplus. If a repository yields less than its allocation, the missing records
are filled first from other repositories' surplus and then, if still short, by cloning repositories
that have records left. Pass 1 runs only when **Max Functions** or **Max Classes** is set.

**Example:** cap = 20, corpus = TheAlgorithms (276 classes), system-design-primer (52 classes).
Each repository first gets 1 slot, leaving 18. These are split in proportion to 275 and 51
remaining classes: TheAlgorithms gets 15 more and system-design-primer gets 3 more, so the
final allocation is TheAlgorithms 16, system-design-primer 4.

**Important caveat:** Apart from the one-slot minimum, allocation is proportional, so it largely preserves the *natural* distribution of the corpus.
If your corpus is dominated by one repository (e.g. a large algorithm collection), the sample
will reflect that. If you need balanced representation across repos, use the **Engineered Projects**
filter or set **Max Repos** to a small number of carefully chosen repositories.
""")

st.divider()

# ─── Extraction rules ─────────────────────────────────────────────────────────

st.header("What Extraction Skips")
st.markdown("""
Extraction walks each cloned repository (its default branch HEAD, cloned with `--depth 1`) and
parses the source files of the selected language.

**File extensions processed**

| Language | Extensions |
|----------|------------|
| Python | `.py` |
| Java | `.java` |
| JavaScript | `.js`, `.jsx` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.hxx` |

TypeScript (`.ts`, `.tsx`) and the JavaScript module extensions `.mjs` and `.cjs` are **not**
processed.

**Directories that are skipped.** These names are matched exactly (case-sensitive) on directories at
any depth below the repository root, and nothing inside them is read:
`vendor`, `vendored`, `third_party`, `third-party`, `site-packages`, `node_modules`, `dist`, `build`,
`venv`, `.venv`, `.tox`, `.eggs`. `external` and `env` are **not** skipped, because they are ordinary
names for real source, and other hidden directories are not skipped as a class. Test directories are
skipped the same way: `test`, `tests`, `__tests__`, `spec`, `specs`.

**Test files that are skipped.** The rules look at the file name only (never its directories),
are case-sensitive, and apply to **every language**:

| Rule | Skipped file names |
|------|--------------------|
| Name starts with | `test_` or `Test` |
| Name ends with | `_test.py`, `Test.java`, `.test.js`, `.spec.js`, `.test.ts`, `.spec.ts`, `_test.cpp`, `_test.cc` |

So `test_utils.py`, `TestApp.py`, `Testimonials.js`, `TestX.cpp`, `UserTest.java` and `foo_test.cc` are
skipped, while `test.py`, `testing.py`, `conftest.py`, `FooTests.java`, `FooIT.java`, `Foo.test.jsx`,
`foo_test.h` and `foo_test.cxx` are not. This is a name rule, not a detector of test code.

**JavaScript only.** Files ending in `.min.js` and any path containing `node_modules` are skipped by
name. A file is also skipped as minified or bundled when any line is longer than 1,000 characters
or the average line is longer than 300 characters.

**Order and counts.** Files are visited in a deterministic order: the files of a directory sorted by
name, then its subdirectories sorted by name, depth first. Pass 1 (counting) and Pass 2 (extraction) select
files through the same code and find definitions with the same walk, so the Pass 1 count equals
what Pass 2 extracts. Forward declarations (a C++ `class X;` with no body) are not counted or
extracted.

**What the manifest records.** The `extraction` block of `manifest.json` holds `skip_dirs` (the
sorted list of skipped directory names above) and `files_skipped` (how many source files of the
selected language sat inside those directories across all processed repositories). Each repository
entry has its own `extraction_files_skipped`. Files skipped as tests or as minified JavaScript are not counted
there.
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

**Stage 3 — Population-level Q1 filter.** Across the discovered candidates that pass
Stages 1 and 2, SIEVE computes the first quartile (Q1) for pull request count and issue
count. Repos falling below Q1 on either are excluded. The LOC and code-ratio criteria are
implemented in SIEVE's quality module but are not applied by the pipeline, because
filtering runs before repositories are cloned.

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
    | `llm_score` | float \\| null | P(LLM-generated) from the built-in CodeBERT classifier; null if scoring is off, failed, or the snippet is blank |
    | `llm_score_truncated` | bool \\| null | True if the score reflects only the first 512 tokens of the snippet; null if not scored |
    | `ast_depth` | int \\| null | Maximum depth of the parse tree (always exported) |
    | `ast_num_nodes` | int \\| null | Total number of AST nodes |
    | `ast_node_types` | dict \\| null | Node type → count mapping |
    | `ast` | dict \\| null | Full AST as nested JSON (only when Export Full AST is enabled) |
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
    | `llm_score` | float \\| null | P(LLM-generated) from the built-in CodeBERT classifier; null if scoring is off, failed, or the snippet is blank |
    | `llm_score_truncated` | bool \\| null | True if the score reflects only the first 512 tokens of the snippet; null if not scored |
    | `ast_depth` | int \\| null | Maximum depth of the parse tree (always exported) |
    | `ast_num_nodes` | int \\| null | Total number of AST nodes |
    | `ast_node_types` | dict \\| null | Node type → count mapping |
    | `ast` | dict \\| null | Full AST as nested JSON (only when Export Full AST is enabled) |
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
SIEVE computes **22 structural metrics** for every extracted record using its own
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
| `cyclomatic_complexity` | McCabe complexity: 1 + number of decision points. Which constructs count depends on the language; see the table below. |
| `max_nesting_depth` | Maximum depth of nested control flow structures (if, for, while, try, with) |

**Decision points counted in `cyclomatic_complexity` (+1 each):**

| Language | Counted |
|----------|---------|
| Python | `if`, `elif`, `for`, `while`, `except`, `with`, `assert`, each `and` / `or`, conditional expressions (`x if c else y`), each list / set / dict comprehension and generator expression |
| Java | `if` (an `else if` counts as its own `if`), `for`, enhanced `for` (for-each), `while`, `do-while`, each `catch`, ternary `?:`, each `&&` and `\\|\\|`, each non-default `case` label |
| JavaScript | `if` (including `else if`), `for`, `for-in`, `for-of`, `while`, `do-while`, each `catch`, ternary `?:`, each `&&` and `\\|\\|`, each non-default `case` |
| C++ | `if` (including `else if`), `for`, range-based `for`, `while`, `do-while`, each `catch`, ternary `?:`, each `&&` / `and` and `\\|\\|` / `or`, each non-default `case` label |

For Java, JavaScript and C++: `&&` and `||` are counted per operator, so
`a && b && c` adds 2. Every `case` label counts, including empty fall-through
labels (`case 1: case 2:` adds 2), while `default` and the `switch` statement
itself add 0. A Java `case 2, 3 ->` is a single label and adds 1, and
`case null, default` is treated as the default label. Not counted in these
three languages: `else`, `default`, `switch`, `try`, `finally`, `return`,
`instanceof`, `await`, `??`, `&&=`, `||=`, and all other binary operators.
Python `match` / `case` is not counted.

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
Output is written to the CLI's `--output-dir` (config key `output_dir`, default `./sieve_output`).
The web UI writes to a temporary directory and offers the files as downloads. Files produced:

| File | Description |
|------|-------------|
| `functions.jsonl` | One JSON object per line, one per function record |
| `classes.jsonl` | One JSON object per line, one per class record |
| `functions.parquet` | Columnar format — list fields serialized as JSON strings |
| `classes.parquet` | Columnar format — list fields serialized as JSON strings |
| `manifest.json` | Run metadata, described below |

JSONL and Parquet are written based on the **Export Format** setting. A data file is written only when
it has records. The manifest is always written.

**`manifest.json`** contains:

| Key | Content |
|-----|---------|
| `sieve_version`, `generated_at` | SIEVE version and the UTC time the manifest was written |
| `config` | The run's settings. **Tokens are never included.** |
| `summary` | `total_repos`, `total_functions`, `total_classes`, `total_llm_scored` (records that received an `llm_score`) and `total_llm_score_truncated` (those scored on only their first 512 tokens) |
| `repos` | One entry per processed repository: stars, contributors (`null` if GitHub would not count them), last commit date, default branch, license, topics, `collected_at`, `test_suite`, `dependencies`, `extraction_files_skipped`, and `quality` when the engineered filter ran |
| `discovery` | What the repository search covered; see **Repository Discovery** |
| `extraction` | The skipped directory names and the number of files skipped; see **What Extraction Skips** |

Each record's `commit_sha` is the commit that was cloned (`null` if it could not be read).

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
    license type, release count, contributor count, and pull request and issue activity
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
    Without a token: 10 search requests per minute and 60 REST requests per hour, which is enough
    for roughly 12 repositories. With a token: 30 per minute and 5,000 per hour. Create a classic
    token with the `public_repo` scope.

    By default a limit that will not clear within about a minute stops the run with an error; the
    CLI's `--wait-for-rate-limit` makes it wait and resume, and the web UI never waits. Requests are
    paced down once the remaining quota is low. See **GitHub Rate Limits** above for the details.
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
    imports, and namespace imports (`import * as`). Only `.js` and `.jsx` files are
    processed; minified and bundled files are skipped. **TypeScript is not supported**:
    `.ts` and `.tsx` files are not processed, because SIEVE ships only the JavaScript grammar
    and it mis-parses typed parameters.

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

    Enable it via the **Annotate LLM Score** toggle (CLI `--annotate-llm-score`) before running.
    The classifier weights (~500MB) are downloaded automatically from HuggingFace Hub on first use.

    **Truncation.** The model reads at most 512 tokens (counting the start and end tokens). A longer
    snippet is scored on its first 512 tokens only, and its `llm_score_truncated` field is `true`.
    The run summary reports `total_llm_scored` and `total_llm_score_truncated`, and the web UI notes how
    many scored records were truncated.

    **Batching.** Snippets are sorted by token length (longest first) and scored in batches of up to 64,
    each padded only to its own longest member instead of to 512. Padding is masked, so this changes
    speed, not the scores. Results are returned in the original order.

    **Blank snippets and failures.** A blank snippet gets `llm_score` null. If the classifier cannot be
    loaded or fails, the run continues and the scores stay null.

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
repo is considered to have a test suite if **≥ 2 signals** fire. The confidence is
`high` for 3 or 4 signals, `medium` for 2, `low` for 1 and `none` for 0.

This detection is separate from the test-file rules that extraction uses to leave test files out of
the corpus (see **What Extraction Skips**). It walks the repository itself, so its rules and its list
of ignored directories differ.

| Signal | What SIEVE checks |
|---|---|
| **Test directory** | A directory (case-insensitive name) among `test`, `tests`, `test_suite`, `testing`, `__tests__`, `__test__`, `spec`, `specs`, `unit_tests`, `integration_tests`, `functional_tests` and `e2e`, **that contains at least one test file** (at any depth below it). An empty folder, or one holding only data or fixtures, does not count. |
| **Test file naming** | Patterns for the selected language. Python: `test_*.py`, `*_test.py`. Java: `*Test.java`, `*Tests.java`, `*TestCase.java`, `*IT.java`. JavaScript: `*.test.js`, `*.spec.js`, `*.test.ts`, `*.spec.ts`. C++: `test_*`, `*_test`, `*Test` or `*test`, and `*Spec` or `*spec`, with a `.cpp`, `.cc` or `.cxx` extension. In addition, any source file inside a `__tests__` or `__test__` directory counts as a test file (Jest keeps plain `foo.js` files there). |
| **Test runner config** | Config files for the selected language. Python: `conftest.py` (by presence), and `pyproject.toml`, `pytest.ini`, `.pytest.ini`, `setup.cfg`, `tox.ini` only when they hold real pytest configuration: a `[tool.pytest]` or `[tool.pytest.ini_options]` table in `pyproject.toml`, a `[pytest]` or `[tool:pytest]` section in the others. A bare mention of `pytest`, for example as a dev dependency, does not count. Java: `pom.xml` or `build.gradle` whose text contains `pytest`, `unittest`, `testng`, `junit` or `jest` (substring match), and `build.gradle.kts` by presence. JavaScript: e.g. `jest.config.js`, `karma.conf.js`, `.mocharc.js`, by presence. C++: `CMakeLists.txt`, `CTestTestfile.cmake`, `googletest.cmake`, `Makefile`, `meson.build`, by presence, so a `CMakeLists.txt` or `Makefile` triggers this signal even if it defines no tests. |
| **CI workflow** | `.yml` or `.yaml` files under `.github` whose text (case-insensitive) contains a test-runner keyword, e.g. `pytest`, `unittest`, `tox`, `jest`, `mocha`, `mvn test`, `gradle test`, `npm test`, `ctest`, `gtest` or `run tests`. The keyword list is the same for all languages. Other CI systems, such as `.circleci`, are not scanned. |

Detection does not enter these directories: `.git`, `venv`, `.venv`, `env`, `node_modules`,
`__pycache__`, `site-packages`, `vendor`, `vendored`, `third_party`, `third-party`, `external`, `build`,
`dist`, `.tox`, `.eggs`, `.mypy_cache`, `.pytest_cache`, nor any other hidden directory except `.github`.
Tests that live in vendored, generated or virtualenv code belong to someone else's project and
do not count.

The two-signal threshold avoids false positives from repos that have a single
test file but no real test infrastructure, while still catching repos that use
non-standard directory layouts.

The result is reported as repository metadata (the `test_suite` field). It is not
used by the **Engineered Projects** filter.
""")

st.divider()

# ─── Known limitations ───────────────────────────────────────────────────────

st.header("Known Limitations")
st.markdown("""
**Creation date is not code age.** The date filter uses the repository's GitHub creation date
(`created:`). A repository whose history was imported, for example pushed from an older repository, can
contain commits that predate its GitHub creation date. A creation-date bound therefore says when the
repository appeared on GitHub, not how old its code is. SIEVE does not inspect commit dates or the
origin of individual snippets.

**There is no snapshot date.** Results depend on when the run happens: SIEVE searches GitHub live and
clones the default branch HEAD at that moment. Star counts, contributor counts and the branch content
change over time, so two runs with the same settings on different days can differ. A repository is
cloned separately for Pass 1, Pass 2 and any shortfall top-up, so a repository pushed to between those
clones can change under the run. The manifest records `generated_at`, a `collected_at` timestamp and the
star count per repository, and the cloned `commit_sha` on each record. It does not pin a snapshot you
could re-fetch later.

**Skipped directories.** `build` and `dist` are skipped along with the other names listed under
**What Extraction Skips**; `external` and `env` are not.

**The test-name rules are not language-specific.** A file name that starts with `Test` is dropped in
every language, so `Testimonials.js` and `TestApp.py` are dropped along with real tests. File names such
as `FooTests.java`, `FooIT.java`, `Foo.test.jsx` and `foo_test.h` are not treated as tests and stay in
the corpus. Test directories are matched by exact name (`test`, `tests`, `__tests__`, `spec`, `specs`).

**One narrow Pass 1 / Pass 2 gap.** Pass 1 counts exactly what Pass 2 extracts, with one exception: a
file that parses but then fails while its records are built is counted in Pass 1 and dropped in Pass 2.
The same applies if the repository changed between the two clones.

**Capped corpora can differ between versions.** Per-repository caps come from the Pass 1 counts, so
with caps set, versions whose counts or discovery differ can allocate the caps differently.

**Processing is sequential.** Repositories are cloned and extracted one after another, so large runs
are slow. The public Space has 8 GB of RAM; run large corpora locally.

**C++ is slower.** See **C++ Parsing Performance**.

**Languages.** Python, Java, JavaScript (`.js`, `.jsx`) and C++ only. TypeScript is not processed.
""")

st.divider()

# ─── License ─────────────────────────────────────────────────────────────────

st.header("License")
st.markdown("""
SIEVE is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Musfiqur Rahman, Mahi Begoug, Emad Shihab

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
st.caption(f"SIEVE v{__version__} · Built for SE research · [GitHub](https://github.com/mrsumitbd/sieve)")