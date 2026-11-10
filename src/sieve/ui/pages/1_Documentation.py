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
DEMO_VIDEO_URL = "https://youtu.be/1Imgn7kpkNg"
st.video(DEMO_VIDEO_URL)
st.markdown(f"""
[Watch the demo video on YouTube]({DEMO_VIDEO_URL}). It was recorded with v0.1.1; the web UI has since
changed (granularity checkboxes, a single **Export Full AST** toggle, and caps where `0` means no cap),
so to reproduce the video exactly, install tag `v0.1.1` (`git clone --branch v0.1.1
https://github.com/mrsumitbd/sieve.git`).
""")
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

st.subheader("For researchers")
st.markdown("""
**What you can do.** Build a corpus of functions, classes or both for one language, bounded by repository
creation date, stars, contributors and, optionally, the engineered-project filter. Compare records on the
22 structural metrics, the AST summary features and, if enabled, `llm_score`. Read each repository's
license, topics, test-suite signal and declared dependencies from `manifest.json`. The manifest also
stores the configuration, the SIEVE version and the time each repository was collected, and every record
carries the commit that was cloned, so a run can be described and repeated. A repeat is not guaranteed to
give the same corpus (see **Known Limitations**).

**Caveat.** A creation-date filter bounds the age of the *repository*, not of the code in it. A repository
created after a model's training cutoff can contain code copied from older sources, so the filter does not
guarantee that the code was unseen by an LLM, or that it is original. SIEVE does not check where individual
snippets came from. State the bound and this limit when you report results.
""")

st.subheader("For practitioners")
st.markdown("""
**What you can do.** Survey a set of repositories in one language: for example, list the functions with the
highest `cyclomatic_complexity` or `max_nesting_depth`, or the lowest `maintainability_index`, to see where
the structurally heavy code sits. Read `dependencies` to see which libraries and version constraints the
selected repositories declare. Use `signature`, `docstring` and `used_imports` to assemble examples for a
code-search index, a prompt set or an internal benchmark from repositories you choose.

**Caveat.** The metrics are structural proxies computed on an isolated snippet. On their own they are not
measures of difficulty or quality: a high complexity value does not mean the code is wrong, and a high
maintainability index does not mean it is good. The repositories carry many different licenses
(`license_spdx` is recorded per repository in the manifest, and SIEVE does not check whether a particular
use is allowed), so check licenses before reusing code.
""")

st.subheader("For educators")
st.markdown("""
**What you can do.** Select examples by size and structure, for instance functions in a `sloc` or
`cyclomatic_complexity` range, to build graded material. A function's `signature` (header and docstring,
no implementation) or a class's `skeleton` can serve as an exercise prompt, with `source_code` as the
reference solution. `used_imports` makes a snippet closer to self-contained, and `commit_sha`, `file_path`
and `start_line` link a record back to its place in the original repository.

**Caveat.** The metrics are structural proxies, not measures of how hard an exercise is for a student, and
they are not a grade of the code's quality. Records are real-world code, not curated exercises: SIEVE does
not run it or attach tests to it, and it can depend on code elsewhere in its repository. The creation date
does not tell you that learners, or the LLMs they use, have not seen the code.
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
| **Max Functions** | Upper bound on the total number of function records in the output. Web UI: `0` = no cap (default 30). CLI and config file: omit it for no cap, otherwise at least 1. The cap is divided across repositories before extraction (see **Corpus Size Caps & Stratified Sampling**). Deduplication runs afterwards and its removals are not refilled, so the final count can be below the cap. |
| **Max Classes** | Upper bound on the total number of class records in the output. Same rules as **Max Functions**. |

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
| **Deduplicate** | Enable MinHash LSH near-duplicate removal across the records of each granularity (functions and classes separately). **Off by default in every interface** (CLI `--deduplicate`, config `deduplicate`, web UI **Deduplicate** toggle). |
| **Similarity Threshold** | Jaccard similarity threshold for MinHash deduplication (0.0–1.0; the web UI slider covers 0.5–1.0). Higher values are more conservative — only near-identical code is removed. Default: 0.8. Used only when deduplication is on. |

Deduplication compares sets of tokens after replacing string literals and collapsing whitespace.
Identifiers and comments are not normalized, so renamed variables lower the similarity. See
**Deduplication** below for the tokenization, the index and which record of a cluster is kept.
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
Skips**), not a random draw, so the same repository state gives the same records. A random draw,
proportional to each repository's share of the records, is used only in a final safety-net step, which
applies if the corpus still exceeds the cap after deduplication.

**Surplus buffer and shortfall (Pass 3).** Pass 2 extracts more than each repository's allocation: up to
the smaller of twice the allocation and the allocation plus 50. The first `allocation` records are kept
and the rest stay in memory as a surplus. If the kept records fall short of the cap (for example because
a repository yielded less than its allocation, or failed), the missing records are taken first from the
surplus of the other repositories, in the order the repositories were processed, and then, if still short,
by cloning again the repositories that have records left, the ones with the most remaining records first.
A repository that returned nothing despite having an allocation is not cloned again. Pass 1 runs only
when **Max Functions** or **Max Classes** is set. A repository whose clone failed in Pass 1 has no
allocation and is extracted without a per-repository limit in Pass 2; the final cap still applies. See
**Architecture Overview**.

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

# ─── Deduplication ────────────────────────────────────────────────────────────

st.header("Deduplication")
st.markdown("""
Deduplication is **off by default** (CLI `--deduplicate`, config `deduplicate`, web UI **Deduplicate**
toggle). When it is on, it removes near-duplicate records using MinHash with locality-sensitive hashing
(the `datasketch` library). This is what the code in `sieve/core/deduplication.py` does.

**Where it runs.** After extraction and the shortfall refill, before the final cap and before LLM
scoring. Records that deduplication removes are not replaced, so with caps set the final corpus can be
smaller than the cap.

**Per granularity, across repositories.** Function records and class records are deduplicated separately,
each with its own index, so a function is never compared with a class. Within one granularity, all
repositories are compared with each other, not only records of the same repository.

**Tokenization.** For each record the `source_code` is turned into a set of tokens:

1. Runs of whitespace are collapsed into one space.
2. String literals are replaced by the placeholder `STR`: first triple-quoted strings (three double quotes or three
   single quotes), then double-quoted and single-quoted strings. The quote characters stay as tokens.
3. Tokens are identifiers (a letter or underscore followed by letters, digits or underscores) and each
   remaining single punctuation character.

What this means in practice:

- Identifiers are kept as written. Renaming a variable or a parameter changes the token set and lowers the
  similarity.
- Comments are not removed. The words in a comment, and its `#` or `//` characters, are tokens.
- Number literals produce no tokens, because digits are matched neither as identifiers nor as punctuation.
  Only letters inside a number, such as the `xFF` of `0xFF`, are kept. Letters outside ASCII are dropped too.
- The comparison is between *sets* of distinct tokens: the order of the tokens and how often a token
  occurs do not matter.

As an illustration, for a two-line Python function, renaming its two parameters lowered the estimated
similarity to about 0.68, which is under the default threshold, while adding one comment line left it at
about 0.82.

**MinHash and threshold.** Each token set gets a MinHash signature with **128 permutations** (fixed; it is
not a setting of `SIEVEConfig`). The index is a `MinHashLSH` with the threshold `dedup_threshold` (default
**0.8**; a value from 0 to 1, where higher removes only closer duplicates). The similarity is an estimate of
the Jaccard similarity of the token sets, and LSH is probabilistic: pairs close to the threshold may or may
not be matched.

**Which record is kept.** Records are visited in list order. Each record is looked up in the index of the
records kept so far: if the index already holds a match, the record is dropped, otherwise it is kept and
added to the index. So the first record of a cluster in list order is kept and the later ones are dropped.
List order is the order in which repositories were processed, then the order of files within a repository
(each directory's files sorted by name, then its subdirectories), then any records added by the shortfall
refill. A dropped record is never added to the index. Records with an empty `source_code` are always kept.

The number of records removed is written to the log (`Deduplication: N → M items`). It is not stored in
`manifest.json`, which records only `deduplicate` and `dedup_threshold` in its `config`.
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
Enabling **Engineered Projects Only** applies a three-stage filter to the discovered repositories before
any of them is cloned. The code in `sieve/core/quality.py` cites two references: Xiao et al. (2025),
"Self-Admitted GenAI Usage in Open-Source Software"
([arXiv](https://arxiv.org/abs/2507.10422)), and Munaiah et al. (2017), "Curating GitHub for engineered
software projects" ([Reaper](https://doi.org/10.1007/s10664-017-9512-6)). The criteria below are what
the code applies.

**Stage 1: license.** A repository is excluded if its SPDX license identifier is `CC-BY-4.0`, `CC0-1.0`,
`CC-BY-SA-4.0` or `OFL-1.1` (SIL Open Font License), or if it has no license. A license GitHub reports as
`NOASSERTION` is treated as no license. Every other identifier passes, including MIT, Apache-2.0,
GPL-2.0, GPL-3.0, BSD-2-Clause, BSD-3-Clause, LGPL, MPL-2.0, ISC, AGPL-3.0 and the **Unlicense** (a
public-domain dedication for software, so it counts as a software license).

**Stage 2: hard thresholds.** A repository is excluded if it has **no releases** (the count of GitHub
releases) or **fewer than 2 contributors**. A repository whose contributor count
is unknown (`null`, because GitHub would not list that many) passes the contributor check. Archived
repositories were already left out by the search query. Because discovery applies **Min Contributors**
first, the effective contributor minimum is the larger of **Min Contributors** and 2.

**Stage 3: pull-request and issue activity.** Among the repositories that passed Stages 1 and 2, SIEVE
computes the first quartile (Q1, the 25th percentile with linear interpolation) of two counts:

- the number of pull requests, of any state (open, closed or merged);
- the number of issues, of any state, **excluding** pull requests (GitHub's issue list also contains pull
  requests, so the pull-request count is subtracted, with a floor of 0).

A repository is excluded if its pull-request count is below the Q1 of pull requests, or its issue count is
below the Q1 of issues. The comparison is strict, so a repository exactly at Q1 stays. The quartiles
are computed over this discovered pool only, which has already been cut by **Min Stars**, **Min
Contributors** and **Max Repos**, not over all of GitHub. The filter therefore runs *after* the **Max Repos**
cap and can leave fewer repositories than the cap. If no repository passes Stages 1 and 2, none passes
the filter.

**Lines-of-code and code-ratio criteria are not applied.** `apply_filters` also contains the Q1 of lines of
code and a range for the ratio of code to comment lines, but the pipeline runs the filter before cloning,
so it passes no clone and every repository has `loc` 0 and `code_ratio` 0. With all values equal to 0 these
two checks exclude nothing. The `quality` block in `manifest.json` therefore shows `loc`, `comment_lines`
and `code_ratio` as 0.

**Data used.** Each candidate costs GitHub requests for its release, pull-request and issue counts. If
GitHub returns an error other than a rate limit or bad credentials for these requests, the counts that
were not fetched stay 0, and a repository with 0 releases fails Stage 2. If a rate limit stops the filter
part-way, only the repositories checked so far are kept and the run summary reports `rate_limited`. `test_suite`
detection is separate and is not used by this filter.

This filter is the most reliable way to avoid documentation repositories, "awesome lists", dataset
repositories and other non-software projects that can otherwise dominate high-star searches. It needs
the whole candidate pool for the quartiles and three more GitHub requests per candidate, so it is slower.
""")

with st.expander("Why do documentation repos appear in high-star searches?"):
    st.markdown("""
    GitHub's star count reflects popularity, not software content. Repos like
    `awesome-python`, `free-programming-books`, and `public-apis` routinely rank in the
    top 10 most-starred Python repos despite containing little or no executable code.
    Without the engineered filter, these repos will be discovered, cloned, and processed —
    yielding zero or near-zero extracted records while consuming API quota and clone time.

    If you do not want to use the full engineered filter (it's slower because of the per-repo
    GitHub lookups it needs), a lighter alternative is to set a higher
    **Min Stars** threshold in combination with a meaningful **Min Contributors** count,
    then inspect the per-repo breakdown in the Dataset Statistics panel after a run.
    Repos with 0 extracted functions *and* 0 extracted classes are a reliable signal of
    non-software content.
    """)

st.divider()

# ─── Record Types ─────────────────────────────────────────────────────────────

st.header("Record Schema")
st.markdown("""
Every record is one JSON object per line in JSONL, or one row in Parquet. The two tables below list
every field of the function record (`FunctionRecord`) and the class record (`ClassRecord`) defined in
`sieve/core/extraction.py`, in the order the fields are written. `X | null` means the field can be `null`.

- **Function records** cover free functions and methods (`is_method` tells them apart). Constructors
  are methods and are included.
- **Class records** cover classes. The methods of a class also appear as their own function records.
- `llm_score` and `llm_score_truncated` are `null` unless **Annotate LLM Score** is on and the record was
  scored. `ast` is `null` unless **Export Full AST** is on. `commit_sha` is filled by the pipeline.
- The AST fields and the 22 metrics are computed on the record's `source_code`, parsed as a standalone
  snippet. If that parse fails they are `null`.
- In Parquet, list fields (such as `parameters`) are stored as JSON strings. See **Output Files and Export**.
""")

tab_func, tab_class = st.tabs(["Function Record", "Class Record"])

with tab_func:
    st.markdown("""
    | Field | Type | Meaning |
    |-------|------|---------|
    | `repo` | str | Repository full name, `owner/repo` |
    | `file_path` | str | Path of the source file relative to the repository root |
    | `language` | str | `Python`, `Java`, `JavaScript` or `C++` |
    | `func_name` | str | Function or method name |
    | `parameters` | list[str] | One string per parameter, as written in the source (Python: annotations and defaults included) |
    | `return_annotation` | str \\| null | Return type as written, if there is one |
    | `docstring` | str \\| null | Python: the docstring without its delimiters. Java, JavaScript, C++: the comment block directly before the definition. `null` if none (always `null` for JavaScript arrow functions) |
    | `source_code` | str | Full source of the function, with indentation normalized |
    | `signature` | str | Header plus docstring with the implementation left out (Python: header, docstring and `pass`) |
    | `used_imports` | list[str] | Import statements of the file whose imported names appear in `source_code` (see **Import Detection**) |
    | `start_line` | int | First line of the definition in the original file (1-based) |
    | `end_line` | int | Last line of the definition in the original file (1-based) |
    | `is_method` | bool | `true` if defined inside a class |
    | `parent_class` | str \\| null | Name of the enclosing class when `is_method` is `true`, otherwise `null` |
    | `decorators` | list[str] | Decorators or annotations written on the function (for example `@staticmethod`, `@Override`) |
    | `llm_score` | float \\| null | P(AI-generated) from the built-in classifier, rounded to 4 decimals. `null` when scoring is off, failed, or the snippet is blank |
    | `llm_score_truncated` | bool \\| null | `true` if the score reflects only the first 512 tokens of the snippet; `null` if the record was not scored |
    | `ast_depth` | int \\| null | Depth of the deepest node of the snippet's parse tree (the root is depth 0). Always exported |
    | `ast_num_nodes` | int \\| null | Number of nodes in the parse tree, anonymous tokens included. Always exported |
    | `ast_node_types` | dict \\| null | Node type to count mapping. Always exported |
    | `ast` | dict \\| null | Full parse tree as nested JSON, only with Export Full AST (see **AST Representation**); otherwise `null` |
    | `commit_sha` | str \\| null | Commit that was cloned (`git rev-parse HEAD`), for building a GitHub permalink with `file_path` and `start_line`. `null` if it could not be read |
    | `loc` | int \\| null | Total lines of the snippet |
    | `sloc` | int \\| null | Non-blank lines that contain no comment (a line of code with a trailing comment is counted in `comments` instead) |
    | `lloc` | int \\| null | Logical lines: number of statement nodes |
    | `comments` | int \\| null | Single-line comment lines |
    | `multi` | int \\| null | Lines spanned by multi-line comments |
    | `blank` | int \\| null | Blank lines |
    | `comment_ratio` | float \\| null | (`comments` + `multi`) / `loc`, rounded to 4 decimals |
    | `cyclomatic_complexity` | int \\| null | McCabe cyclomatic complexity |
    | `max_nesting_depth` | int \\| null | Maximum nesting depth of control-flow structures |
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
    | `time` | float \\| null | Halstead: estimated programming time in seconds |
    | `bugs` | float \\| null | Halstead: estimated number of delivered bugs |
    | `maintainability_index` | float \\| null | Maintainability Index (see **Code Structure Metrics**) |
    """)

with tab_class:
    st.markdown("""
    | Field | Type | Meaning |
    |-------|------|---------|
    | `repo` | str | Repository full name, `owner/repo` |
    | `file_path` | str | Path of the source file relative to the repository root |
    | `language` | str | `Python`, `Java`, `JavaScript` or `C++` |
    | `class_name` | str | Class name |
    | `parent_classes` | list[str] | Names of the base classes / extended and implemented types (Python keyword arguments such as `metaclass=` are not included) |
    | `docstring` | str \\| null | Python: the class docstring without delimiters. Java, JavaScript, C++: the comment block directly before the class. `null` if none |
    | `source_code` | str | Full source of the class, with indentation normalized |
    | `skeleton` | str | Class header plus the signatures and docstrings of its methods with implementations left out (Python: bodies replaced by `pass`) |
    | `used_imports` | list[str] | Import statements of the file whose imported names appear in `source_code` (see **Import Detection**) |
    | `method_names` | list[str] | Names of the methods defined directly in the class |
    | `method_count` | int | Number of entries in `method_names` |
    | `has_constructor` | bool | `true` if a constructor is defined (Python: `__init__`, or a `dataclass` decorator) |
    | `decorators` | list[str] | Decorators or annotations written on the class |
    | `start_line` | int | First line of the definition in the original file (1-based) |
    | `end_line` | int | Last line of the definition in the original file (1-based) |
    | `llm_score` | float \\| null | P(AI-generated) from the built-in classifier, rounded to 4 decimals. `null` when scoring is off, failed, or the snippet is blank |
    | `llm_score_truncated` | bool \\| null | `true` if the score reflects only the first 512 tokens of the snippet; `null` if the record was not scored |
    | `ast_depth` | int \\| null | Depth of the deepest node of the snippet's parse tree (the root is depth 0). Always exported |
    | `ast_num_nodes` | int \\| null | Number of nodes in the parse tree, anonymous tokens included. Always exported |
    | `ast_node_types` | dict \\| null | Node type to count mapping. Always exported |
    | `ast` | dict \\| null | Full parse tree as nested JSON, only with Export Full AST (see **AST Representation**); otherwise `null` |
    | `commit_sha` | str \\| null | Commit that was cloned (`git rev-parse HEAD`), for building a GitHub permalink with `file_path` and `start_line`. `null` if it could not be read |
    | `loc` | int \\| null | Total lines of the snippet |
    | `sloc` | int \\| null | Non-blank lines that contain no comment (a line of code with a trailing comment is counted in `comments` instead) |
    | `lloc` | int \\| null | Logical lines: number of statement nodes |
    | `comments` | int \\| null | Single-line comment lines |
    | `multi` | int \\| null | Lines spanned by multi-line comments |
    | `blank` | int \\| null | Blank lines |
    | `comment_ratio` | float \\| null | (`comments` + `multi`) / `loc`, rounded to 4 decimals |
    | `cyclomatic_complexity` | int \\| null | McCabe cyclomatic complexity |
    | `max_nesting_depth` | int \\| null | Maximum nesting depth of control-flow structures |
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
    | `time` | float \\| null | Halstead: estimated programming time in seconds |
    | `bugs` | float \\| null | Halstead: estimated number of delivered bugs |
    | `maintainability_index` | float \\| null | Maintainability Index (see **Code Structure Metrics**) |
    """)

st.divider()

# ─── Code Structure Metrics ───────────────────────────────────────────────────

st.header("Code Structure Metrics")
st.markdown("""
SIEVE computes **22 structural metrics** for every function and class record: 7 raw, 2 complexity,
12 Halstead and 1 composite. They come from its own tree-sitter-based engine (`sieve.core.metrics`), so
no external tool is needed, and the definitions are the same for all four languages. Each metric is
computed on the record's own `source_code`, parsed as a standalone snippet, so it describes that
snippet and not the file or repository around it. Fan-in and fan-out need a call graph of the whole
repository and are not computed. If a snippet cannot be parsed, all 22 metrics are `null`.

What counts as an operator, an operand, a branch, a comment or a statement is decided by per-language
tables of tree-sitter node types in `sieve/core/metrics.py`.

#### Raw Metrics

| Metric | Definition |
|--------|------------|
| `loc` | Number of lines in the snippet |
| `sloc` | Non-blank lines that contain no comment. A line of code with a trailing comment is counted in `comments`, not here. A Python docstring is a string, not a comment, so its lines count here |
| `lloc` | Logical lines: the number of statement nodes (for example expression, return, throw, break, continue statements, and declarations or assignments, depending on the language) |
| `comments` | Lines of comments that start and end on the same line |
| `multi` | Lines covered by comments that span more than one line (block comments in Java, JavaScript and C++). Python has no multi-line comment node, so `multi` is 0 for Python |
| `blank` | Lines that are empty or only whitespace |
| `comment_ratio` | (`comments` + `multi`) / `loc`, rounded to 4 decimals; 0.0 when `loc` is 0 |

#### Complexity Metrics

| Metric | Definition |
|--------|------------|
| `cyclomatic_complexity` | McCabe complexity: 1 + the number of decision points. Which constructs count depends on the language; see the table below |
| `max_nesting_depth` | The largest number of control-flow nodes on one path from the root, counting the node types listed below |

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

**Node types counted in `max_nesting_depth` (+1 per level):**

| Language | Counted |
|----------|---------|
| Python | `if`, `for`, `while`, `with`, `try`, `except`, `match` |
| Java | `if`, `for`, `while`, `do`, `try`, `catch`, `switch` (the for-each loop is not counted) |
| JavaScript | `if`, `for`, `for-in` / `for-of`, `while`, `do`, `try`, `catch`, `switch` |
| C++ | `if`, `for`, `while`, `do`, `try`, `catch`, `switch` (the range-based `for` is not counted) |

In Java, JavaScript and C++ an `else if` is an `if` nested inside the `else`, so a chain of `else if`
branches deepens the nesting by one level per branch. Python `elif` is not counted, so the same chain
has depth 1.

#### Halstead Metrics

Halstead metrics are derived from the operators and operands found in the syntax tree. Operator and
operand node types are listed per language in `sieve/core/metrics.py`; operands are identifiers and
literals (and type names in Java and C++). With `h1`, `h2`, `N1`, `N2` as below:

| Metric | Definition |
|--------|------------|
| `h1` | Distinct operators |
| `h2` | Distinct operands |
| `N1` | Total operator occurrences |
| `N2` | Total operand occurrences |
| `vocabulary` | h1 + h2 |
| `halstead_length` | N1 + N2 |
| `calculated_length` | h1·log₂(h1) + h2·log₂(h2) (a term is 0 when its count is 0) |
| `volume` | (N1 + N2)·log₂(h1 + h2); 0 when the vocabulary is 1 or less |
| `difficulty` | (h1 / 2)·(N2 / h2); 0 when h2 is 0 |
| `effort` | difficulty × volume |
| `time` | effort / 18, in seconds |
| `bugs` | volume / 3000 |

The count metrics are integers. `calculated_length`, `volume`, `difficulty`, `effort` and `time` are
rounded to 2 decimals and `bugs` to 4.

#### Composite Metric

| Metric | Definition |
|--------|------------|
| `maintainability_index` | MI = max(0, 100 · (171 − 5.2·ln(V) − 0.23·G − 16.2·ln(L) + 50·sin(√(2.4·C))) / 171), rounded to 2 decimals. V is the Halstead `volume`, G is `cyclomatic_complexity`, L is `sloc` and C is `comment_ratio` as a fraction. It is 100.0 when `sloc` or `volume` is 0 |

The formula has the form of the Radon variant of the Maintainability Index, scaled to 100. Its comment term
uses `comment_ratio` as a fraction between 0 and 1, and `comment_ratio` includes multi-line comment lines, so
values need not match those of other tools. The result is floored at 0 but not capped above, so a very
small snippet with a high comment ratio can score above 100. Higher means easier to maintain *by this
formula*.

These are structural proxies. They describe the shape and size of a snippet; on their own they do not
measure how hard the code is to understand or how good it is.
""")

st.divider()

# ─── AST representation ───────────────────────────────────────────────────────

st.header("AST Representation")
st.markdown("""
SIEVE parses each record's `source_code` as a standalone snippet with the tree-sitter grammar of the
selected language. Two things come from that parse.

**Summary fields, always exported.** These three are in every record, whatever the settings:

| Field | Content |
|-------|---------|
| `ast_depth` | Depth of the deepest node. The root node has depth 0 |
| `ast_num_nodes` | Number of nodes in the tree. Anonymous nodes (keywords and punctuation such as `def` or `(`) are counted |
| `ast_node_types` | Dictionary from node type to the number of nodes of that type |

**Full AST, opt-in.** Only when **Export Full AST** is on (CLI `--export-ast`, config `export_ast`, web
UI toggle; default off) is the whole tree written to the record's `ast` field. Otherwise `ast` is `null`.
Each node is an object with three keys:

| Key | Content |
|-----|---------|
| `type` | The tree-sitter node type, for example `function_definition` |
| `text` | For a leaf node (a node without children): its source text. Text longer than 80 characters is cut to its first 77 characters followed by `...`. `null` for every node that has children |
| `children` | List of child nodes in source order; empty for a leaf |

```json
{"type": "module", "text": null, "children": [
  {"type": "function_definition", "text": null, "children": [
    {"type": "def", "text": "def", "children": []},
    {"type": "identifier", "text": "add", "children": []}
  ]}
]}
```

The example is shortened; a real tree continues with the parameters, the colon and the body. There is no
practical depth limit (the converter allows 999 levels). The root type depends on the grammar
(`module` for Python, `program` for Java and JavaScript, `translation_unit` for C++). The full AST makes
files much larger, which is why it is off by default. If parsing raises an error, all four AST fields are
`null`.
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

st.header("Output Files and Export")
st.markdown("""
Output is written to the CLI's `--output-dir` (config key `output_dir`, default `./sieve_output`).
The web UI writes to a temporary directory and offers the files as downloads.

**Formats.** **Export Format** (CLI `--export-format`, config `export_format`) is `jsonl` (default),
`parquet` or `both`.

**Files per granularity.** Functions and classes are written to separate files. A data file is written only
when it has at least one record, so a function-only run produces no `classes.*` file. The manifest is
always written.

| File | Written when | Content |
|------|--------------|---------|
| `functions.jsonl` | format is `jsonl` or `both`, and there are function records | One JSON object per line, one per function record, UTF-8 (non-ASCII text is written as is), keys in the order of the **Record Schema** tables |
| `classes.jsonl` | format is `jsonl` or `both`, and there are class records | The same, for class records |
| `functions.parquet` | format is `parquet` or `both`, and there are function records | Columnar, written with `polars` |
| `classes.parquet` | format is `parquet` or `both`, and there are class records | The same, for class records |
| `manifest.json` | always | Run metadata, described below |

**JSONL vs Parquet.** The two hold the same fields. In Parquet, list-valued fields (`parameters`,
`used_imports`, `decorators`, `method_names`, `parent_classes`) are stored as JSON strings, so decode them
with `json.loads`. The dictionary fields `ast_node_types` and `ast` become nested struct columns whose
fields are inferred from the first records written, so a node type that first appears later in the file
can be missing from the Parquet column. JSONL always holds the complete dictionaries. If you need every
node type or the full tree, use JSONL.

**Web UI bundle.** After a run the **Download Corpus** panel shows one button per file that was written and
a **Download All (ZIP)** button. The ZIP is named `sieve_corpus.zip` and contains every one of those files,
including `manifest.json`, at the top level without folders.

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

st.subheader("manifest.json Layout")
st.markdown("""
`manifest.json` is one JSON object with seven top-level keys, written in this order. It never contains a
token.

```json
{
  "sieve_version": "...",
  "generated_at": "...",
  "config": {},
  "summary": {},
  "repos": [],
  "discovery": {},
  "extraction": {}
}
```

The tables list every field. "Null?" says whether the value can be `null`; "no" means the key is always
present with a value of the stated type (a list or count can be empty or 0).

**Top level**

| Key | Type | Null? | Content |
|-----|------|-------|---------|
| `sieve_version` | str | no | Version of SIEVE that wrote the file |
| `generated_at` | str | no | UTC time the manifest was written, ISO 8601 |
| `config` | object | no | The run's settings, below |
| `summary` | object | no | Totals, below |
| `repos` | list | no | One entry per processed repository, below |
| `discovery` | object | no | What the repository search covered, below. The pipeline always writes it |
| `extraction` | object | no | What extraction skipped, below. The pipeline always writes it |

**`config`** holds the `SIEVEConfig` fields except `github_token` and `hf_token`:

| Key | Type | Null? |
|-----|------|-------|
| `language` | str | no |
| `start_date`, `end_date` | str (`YYYY-MM-DD`) | no |
| `min_stars`, `min_contributors` | int | no |
| `max_repos`, `max_functions`, `max_classes` | int | **yes** (`null` = no cap) |
| `granularity` | list[str] (`function`, `class`) | no |
| `engineered_only`, `annotate_llm_score`, `export_ast`, `deduplicate`, `wait_for_rate_limit` | bool | no |
| `dedup_threshold` | float | no |
| `output_dir` | str | no |
| `export_format` | str (`jsonl`, `parquet`, `both`) | no |

**`summary`**

| Key | Type | Null? | Content |
|-----|------|-------|---------|
| `total_repos` | int | no | Number of entries in `repos` |
| `total_functions`, `total_classes` | int | no | Records written |
| `total_llm_scored` | int | no | Records that received an `llm_score` (0 if scoring was off or failed) |
| `total_llm_score_truncated` | int | no | Of those, records scored on only their first 512 tokens |

**Each entry of `repos`** (repository metadata at collection time):

| Key | Type | Null? | Content |
|-----|------|-------|---------|
| `full_name` | str | no | `owner/repo` |
| `url` | str | no | The repository's GitHub page |
| `stars` | int | no | Star count when discovered |
| `contributors` | int | **yes** | Exact contributor count without bots; `null` when GitHub would not list them (too many) |
| `last_commit_date` | str | no | ISO 8601 time of the latest commit. If that lookup fails it falls back to the repository's last push time, and to the text `unknown` if there is none |
| `default_branch` | str | no | The branch that was cloned |
| `license_spdx` | str | **yes** | SPDX identifier of the license; `null` if GitHub reports none (GitHub may also report `NOASSERTION`) |
| `language` | str | no | The language of the run |
| `collected_at` | str | no | UTC time the metadata was collected, ISO 8601 |
| `topics` | list[str] | no | GitHub topics (may be empty) |
| `test_suite` | object | no | Result of the test-suite heuristic: `test_suite_present` (bool), `confidence` (`high`, `medium`, `low` or `none`), `has_test_directory`, `has_test_files`, `has_test_config`, `has_ci_test_invocation` (bool each), `test_file_count` (int), `test_directories_found` and `test_config_files_found` (lists of str). See **How Test Suite Detection Works** |
| `dependencies` | list[object] | no | Direct dependencies, each `{name: str, version: str or null, kind: str}`; see **Dependency Graph** |
| `extraction_files_skipped` | int | no | Source files of the language inside skipped directories |
| `quality` | object | only present when **Engineered Projects Only** ran | `pull_request_count`, `issue_count`, `release_count`, `loc`, `comment_lines`, `code_ratio` and `passes_engineered_filter`. `loc`, `comment_lines` and `code_ratio` are 0 in the pipeline and `passes_engineered_filter` is always `true`, because only repositories that passed are processed |

**`discovery`** (fields explained under **Repository Discovery**; none is `null`):
`complete` (bool), `total_matching` (int), `slices` (int), `truncated_windows` (list of
`{start, end, total_count}`), `incomplete_results` (bool), `excluded` (list of `{repo, reason}`),
`contributors_unknown` (list[str]) and `rate_limit_waits` (`{count: int, seconds: number}`).

**`extraction`**: `skip_dirs` (list[str], sorted) and `files_skipped` (int, summed over all processed
repositories).

Repositories that fail to clone or process are not listed in `repos`. They appear in the run summary that
the CLI and web UI show (`failed_repos`), and repositories dropped during discovery are in
`discovery.excluded`.
""")

st.divider()

# ─── AI-generated-code score ──────────────────────────────────────────────────

st.header("AI-Generated-Code Score")
st.markdown("""
`llm_score` is the probability that a snippet is AI-generated, from a CodeBERT classifier that SIEVE
loads itself. It is computed only when **Annotate LLM Score** is on (CLI `--annotate-llm-score`, config
`annotate_llm_score`; default off). The page **About the Classifier** describes the training data and
reports the evaluation; this section describes how scoring runs and how to read the result.

**How scoring runs.**

- It is the last step before export (Phase 5 of `run_pipeline`), after deduplication and the final cap, so
  only records that are exported are scored. It scores the `source_code` of every function and class record.
- Each snippet is tokenized with the `microsoft/codebert-base` tokenizer. The model reads at most **512
  tokens**, counting the start and end tokens. A longer snippet is scored on its first 511 tokens plus the
  end token, and its `llm_score_truncated` is `true`.
- Snippets are sorted by token length (longest first) and scored in batches of up to **64**. Each batch is
  padded only to its own longest member. The model reads only the first position's hidden state and
  padding is masked, so this changes speed and not the scores. Results come back in the original order.
- The device is chosen automatically: CUDA if available, then Apple MPS, then CPU.
- A blank snippet gets `llm_score` and `llm_score_truncated` of `null`. Scores are rounded to 4 decimals.
- **Failure handling.** If the classifier cannot be loaded or scoring raises an error (for example PyTorch
  or Transformers is not installed, the weights cannot be downloaded, or memory runs out), the error is
  reported through the progress callback and the run continues. Scores are assigned only after every batch
  has finished, so a failed step leaves all `llm_score` values `null`; it never leaves some records scored.
  The manifest then shows `total_llm_scored` of 0.

**Where the weights come from.** The weights (`best_model.pt`) are hosted on the Hugging Face Hub in the
repository
[`mrahman2025/sieve-llm-classifier`](https://huggingface.co/mrahman2025/sieve-llm-classifier)
(`HF_REPO_ID` in `sieve/models/classifier.py`). They are downloaded on first use with `huggingface_hub`
and cached. If a file `src/sieve/models/artifacts/best_model.pt` exists, it is used instead of
downloading. The tokenizer and the encoder architecture come from
[`microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base). A Hugging Face token is
optional (config `hf_token`, or the `HF_TOKEN` environment variable in the CLI).

**Interpretation and limits.**

- `llm_score` is a **probabilistic signal, not evidence of authorship**. A high score means the snippet
  resembles the AI-generated examples the classifier was trained on. It does not show that a model wrote
  the code, and a low score does not show that a person did.
- The classifier was **trained on code from nine LLMs** (and on human code from repositories created after
  January 2024).
- It was **evaluated on a held-out set drawn from the same dataset** by a random split, stratified by
  language, granularity and label. The AI-generated examples in that set therefore come from the same
  generators as those in the training data. The evaluation does not show how the score behaves on code from
  LLMs that were not in the training data, such as newer models, and the About page notes that accuracy may
  drop for them.
- Length matters. Only the first 512 tokens of a long snippet are read, and very short snippets can get
  unreliable scores.
- Use the score to screen, stratify or compare groups of records, and report it as an estimate with these
  limits.
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

    `llm_score` is a probabilistic signal, not evidence of authorship, and the held-out evaluation does
    not cover LLMs outside the training data. See **AI-Generated-Code Score** above for how scoring runs
    and its limits, and the **About the Classifier** page for the training data, architecture, and
    performance metrics (F1=0.9478, AUROC=0.9902 on a held-out test set of 11,419
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

Dependencies are stored in the corpus `manifest.json` under each repo's
`dependencies` key, so they can be used for downstream analysis even without
the UI.
""")

st.subheader("Dependency Extraction")
st.markdown("""
Dependencies are read from the manifest files of the **selected language only**, in the cloned
repository, by `sieve/core/dependencies.py`. Nothing is installed or resolved and no package registry is
contacted. Files are looked for in the repository root unless noted.

**What is stored.** A list of `{name, version, kind}` per repository: `name` (str), `version` (the
constraint exactly as written, or `null`) and `kind` (`main`, `dev` or `optional`). Entries follow the
order in which the files are read. If two files list the same package, it appears twice.

| Language | Files read | How they are parsed | Kind |
|---|---|---|---|
| Python | `requirements*.txt` and `requirements/*.txt` | Line by line with a regular expression. Blank lines, comments and lines starting with `-` (such as `-r other.txt`, `-e .` and index options) are skipped. A trailing comment and `[extras]` are cut. `name` is the leading name; `version` is the operator-and-number text that follows (for example `>=2.0,<3`) | `main` for every line, whatever the file is called (`requirements-dev.txt` included) |
| Python | `pyproject.toml` | Parsed as TOML. `[project].dependencies`; every group of `[project.optional-dependencies]`; `[tool.poetry.dependencies]` (the `python` entry is skipped) and `[tool.poetry.dev-dependencies]`, where the version is the string or the `version` key of a table. If the TOML cannot be parsed, a fallback reads only the `dependencies = [...]` list | `main`, `optional`, `main`, `dev` respectively |
| Python | `setup.cfg` | The lines under `install_requires` | `main` |
| JavaScript | `package.json` | JSON. `dependencies`, `devDependencies`, `peerDependencies`; the version is the range string from the file | `main`, `dev`, `optional` |
| Java | `pom.xml` in the root, or if there is none, the first `*/pom.xml` one directory down | A regular expression over every `<dependency>` block (not an XML parser, so blocks under `dependencyManagement` or inside plugins are read too). `name` is `groupId:artifactId`; `version` is the `<version>` text as written (it may be a `${property}` placeholder) or `null` when absent | `dev` if the scope is `test` or `provided`, otherwise `main` |
| C++ | `conanfile.txt` | Lines of the `[requires]` section of the form `name/version` | `main` |
| C++ | `vcpkg.json` | JSON `dependencies`: a string, or an object with `name` and `version-semver` or `version` | `main` |
| C++ | `CMakeLists.txt` | `find_package(Name ...)` calls, case-insensitive. `CMake`, `Threads` and `PkgConfig` are ignored; `version` is `null` | `main` |

**Not handled.**

- Transitive dependencies, lock files and version resolution: the constraint is stored as written.
- Other manifests: `setup.py`, `Pipfile`, `environment.yml`, Gradle or Ant builds, `conanfile.py` (the
  module's docstring names it, but it is not read), Conan sections other than `[requires]`, `optionalDependencies`
  in `package.json`, Poetry dependency groups (`[tool.poetry.group.*]`), `[dependency-groups]` and
  `[build-system].requires` in `pyproject.toml`.
- Manifests in subdirectories, for example the packages of a monorepo (the one-level `pom.xml` search is the
  only exception).
- Requirement lines that are URLs or VCS references and environment markers. For example, a line starting
  `git+https://` is stored under the name `git`, and a marker such as `; python_version < "3.9"` is
  dropped, not recorded.
- A file that cannot be read or parsed is skipped without a message; `dependencies` is `[]` when no
  supported file is found.
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

# ─── Architecture ─────────────────────────────────────────────────────────────

st.header("Architecture Overview")
st.markdown("""
**One input.** The CLI, the web UI and the Python API all build one `SIEVEConfig`
(`sieve/config.py`), a pydantic model. It validates the settings (for example caps are `null` or at least
1, `end_date` is not before `start_date`, at least one granularity), rejects unknown keys, supplies the
defaults and keeps the two tokens out of every dump.

**One entry point.** `run_pipeline(config, progress_callback=None)` in `sieve/pipeline.py` is a facade
over the stages and returns a summary dictionary (counts, `output_paths`, `failed_repos`, per-repository
metadata and a `rate_limited` flag). The CLI and the web UI call nothing else. Each stage is also a
module you can import on its own:

| Stage | Module |
|-------|--------|
| Discovery | `sieve/core/discovery.py` (`discover_repos`), with `sieve/core/github_limits.py` for rate limits |
| Engineered filter | `sieve/core/quality.py` |
| Extraction | `sieve/core/extraction.py`, with `metrics.py`, `dependencies.py` and `detection.py` |
| Deduplication | `sieve/core/deduplication.py` |
| Annotation | `sieve/models/classifier.py` |
| Export | `sieve/core/export.py` |

**Order of the stages.**

```
SIEVEConfig
   ↓
run_pipeline:  quota check → discovery → [engineered filter]
               → Pass 1 count → Pass 2 extract → Pass 3 refill
               → [deduplication] → final cap → [LLM score] → export
```

Steps in brackets run only when their setting is on. Pass 1 runs only when **Max Functions** or **Max
Classes** is set. Discovery is a generator, but the pipeline reads it to the end before extraction starts,
so the full repository list is known before the first clone.

**Progress callback.** `progress_callback(message, current, total)` is called with each progress line.
`current` and `total` are 0 for lines that are not part of a counted loop. The CLI prints the lines and
the web UI streams them into a status box and keeps them as a log. The same lines go to the Python logger.

**Two-pass extraction with per-repository allocation.**

1. *Pass 1* clones each repository and counts its extractable functions and classes, with the same file
   selection and the same walk that extraction uses (no records are built).
2. The caps are split across repositories: every repository with at least one record gets one slot, and
   the rest are shared in proportion to the records beyond the first (largest-remainder method; see
   **Corpus Size Caps & Stratified Sampling**).
3. *Pass 2* clones again and extracts each repository with a per-repository limit larger than its
   allocation: the smaller of twice the allocation and the allocation plus 50. The first `allocation`
   records are kept and the extra ones are held in memory as a **surplus buffer**.
4. *Pass 3* fills any **shortfall** (kept records fewer than the cap). It takes records from the surplus
   buffers in repository order and then, if still short, clones again the repositories that have records
   left, those with the most remaining records first, skipping repositories that returned nothing despite
   an allocation. Records are matched by repository, file path and start line so none is added twice.

Without caps there is no Pass 1 and no allocation: every record is extracted in a single pass.

**How failures are isolated.**

- *A repository.* A failed or timed-out clone (300 seconds), or an unexpected error while processing it,
  drops that repository only. It is listed in `failed_repos` (and counted in `total_repos_failed`) and the
  run continues. During Pass 1 and Pass 3 such a repository is skipped silently.
- *A file.* A file that cannot be read, that hits the recursion limit, or that raises while its records
  are built is logged as a warning and skipped. It is not counted in the summary or the manifest.
- *Small parts of a repository.* A failure to read the commit gives `commit_sha` of `null`; a manifest that
  cannot be parsed gives no dependencies for that file.
- *GitHub limits and errors.* A rate limit that stops discovery or the engineered filter after some
  repositories were found lets the run continue with those and sets `rate_limited`; with none found it
  ends with an error. A repository whose contributors cannot be fetched is skipped and recorded in
  `discovery.excluded`.
- *Annotation.* A failure to load or run the classifier is reported and the run continues without scores.
- *Not isolated.* Invalid settings, a rejected GitHub token, an exhausted quota before discovery starts,
  and a failure while writing the output end the run with an error.

Repositories are processed one after another.
""")

st.divider()

# ─── Extending SIEVE ──────────────────────────────────────────────────────────

st.header("Extending SIEVE")
st.markdown("""
SIEVE has no plug-in registry. Languages, export formats and the classifier are all selected by code
inside the package, so extending it means editing the source. What each change touches:

#### Adding a language

| Part | Where | What to do |
|------|-------|------------|
| Name | `Language` enum in `sieve/config.py` | Add a member. The web UI's language list is built from this enum, and the config and CLI validate against it |
| File extensions | `LANGUAGE_EXTENSIONS` in `sieve/core/extraction.py` | Add the set of extensions to read |
| Grammar | `pyproject.toml` and `requirements.txt`; `_get_parser` in `extraction.py`; `_get_tree` in `sieve/core/metrics.py` | Add the tree-sitter grammar package and import it in **both** loaders. The sample viewer's tree builder, `build_ast_json` in `sieve/ui/ast_viz.py`, has a third copy |
| Definitions | `_WALKS` in `extraction.py` | Add a walk that decides which nodes are functions and classes. Pass 1 counting and Pass 2 extraction both use it |
| Records | `extraction.py` | Write an extractor that turns each definition into a `FunctionRecord` or `ClassRecord` (parameters, docstring, signature or skeleton, imports), and add its branch in `extract_from_file` |
| Metrics | `sieve/core/metrics.py` | Add the language to the node-type tables: `_OPERATORS`, `_OPERAND_TYPES`, `_BRANCH_NODES`, `_BRANCH_OPERATORS` and `_CASE_LABEL_NODES` (where the grammar uses a shared binary-expression node or case-label node), `_CONTROL_NODES`, `_COMMENT_TYPES`, and the statement types inside `_raw_metrics`. Without a grammar branch in `_get_tree` the metrics are all `null` |
| Dependencies | `parse_dependencies` in `sieve/core/dependencies.py` | Add a parser for the language's manifest files; otherwise `dependencies` is empty |
| Test-suite signal | tables in `sieve/core/detection.py` | Add file-name patterns, source extensions and runner config files; otherwise the language has no test patterns |
| Search | `LANGUAGE_QUERY_MAP` in `sieve/core/discovery.py` | Needed only if GitHub's name for the language is not its lowercased name |

The file-name rule for test files in `extraction.py` and the skipped-directory lists are shared by all
languages and may need new entries.

#### Adding an export format

- Add a member to `ExportFormat` in `sieve/config.py`. The CLI and the config validate against it.
- In `export_dataset` (`sieve/core/export.py`), add a branch for the format and a writer function that
  takes the record list and an output path, and add the new files to the returned `output_paths`. The
  manifest is written separately and needs no change.
- The web UI's **Export Format** list is built from the enum but its labels come from a fixed dictionary
  in `sieve/ui/Home.py`, which needs an entry for the new value. The download buttons and the ZIP bundle
  pick up every file in `output_paths` on their own.

#### Replacing the classifier

The pipeline uses the classifier through two calls (Phase 5 of `run_pipeline`):

- `LLMCodeClassifier.load(model_dir=<path>, hf_token=<token or None>)` returns the loaded object.
- `clf.score_batch_detailed(snippets, batch_size=64)` takes a list of `str` and returns a list of the
  same length and order. Each entry is `None` (the snippet was blank, so no score) or an object with
  `.score` (float in [0, 1]) and `.truncated` (bool). `ScoreResult` in `sieve/models/classifier.py` is
  such an object.

The class also offers `score` and `score_batch`, which the pipeline does not use.

**Not yet pluggable.** `run_pipeline` imports `LLMCodeClassifier` directly, and no setting selects another
classifier, so a replacement class means editing that import in `sieve/pipeline.py`. Two lighter changes
need no code change: new weights with the same architecture (a CodeBERT encoder, a dropout and one linear
output on the first-position state) can be placed at `src/sieve/models/artifacts/best_model.pt`, which is
used before the Hugging Face download; and `HF_REPO_ID` in `classifier.py` names the Hub repository. The
tokenizer is always `microsoft/codebert-base`. Likewise, there is no way to add a language, an export
format or a dependency parser without editing the modules listed above.
""")

st.divider()

# ─── Deployment and CI ────────────────────────────────────────────────────────

st.header("Deployment and CI")
st.markdown("""
**Docker image.** The `Dockerfile` in the repository root builds the image that the hosted Space runs:

- Base image `python:3.11-slim`; the system packages `git`, `curl` and `cloc` are installed with `apt`.
- The repository is copied in and `pip install -r requirements.txt` installs SIEVE (the file starts with
  `-e .`) and its dependencies, including PyTorch and Transformers.
- A user with id 1000 is created, as Hugging Face Spaces requires, with a writable Hugging Face cache and
  a writable `src/sieve/models/artifacts` directory.
- The environment sets the Streamlit port to 7860, address `0.0.0.0`, headless mode and no usage
  statistics. The command is `streamlit run src/sieve/ui/Home.py --server.port=7860 --server.address=0.0.0.0`.
- The classifier weights are **not** part of the image. They are downloaded on first use from the Hugging
  Face Hub repository `mrahman2025/sieve-llm-classifier` into the cache (see **AI-Generated-Code Score**).

The front matter of `README.md` declares `sdk: docker` and `app_port: 7860` for the Space.

**Deployment to the Space.** The workflow `.github/workflows/sync_to_hf.yml` runs when a GitHub release is
**published** or when it is started **manually** (workflow dispatch). It checks out the full history and
force-pushes `HEAD` to the `main` branch of the Space repository `mrahman2025/sieve` on Hugging Face,
using the `HF_TOKEN` secret. The workflow has no test step and does not wait for the CI workflow.

**Continuous integration.** The workflow `.github/workflows/ci.yml` runs on pushes to `main` and `dev`,
on pull requests to `main`, and on manual dispatch. It runs on Ubuntu with Python 3.11 and 3.12:

1. install `requirements-dev.txt`, then `pip install -e .`;
2. run `pytest` with coverage reporting;
3. upload the coverage report to Codecov (a failed upload does not fail the job).
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

**Deduplication is approximate and literal.** It compares token sets with MinHash, keeps identifiers and
comments as written and ignores numbers, so it finds near-copies, not renamed or reworded code. See
**Deduplication**.

**Parquet dictionary columns.** `ast_node_types` and `ast` are written as struct columns inferred from the
first records, so a node type first seen later can be missing from the Parquet file. JSONL has the full
dictionaries. See **Output Files and Export**.

**Failed repositories are not in the manifest.** A repository that fails to clone or process appears in the
run summary (`failed_repos`) but not in `manifest.json`, and the manifest does not record how many records
deduplication removed.

**The AI-generation score is an estimate.** See **AI-Generated-Code Score** for what it can and cannot show.

**Metrics are structural proxies.** They describe the shape and size of a snippet, not its difficulty or
quality. The Maintainability Index is not capped at 100. See **Code Structure Metrics**.

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