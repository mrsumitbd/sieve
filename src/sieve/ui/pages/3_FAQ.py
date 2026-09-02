"""
pages/3_FAQ.py

Frequently Asked Questions about SIEVE.
"""

import streamlit as st

st.set_page_config(
    page_title="SIEVE — FAQ",
    page_icon="🔬",
    layout="wide",
)

st.title("❓ Frequently Asked Questions")
st.divider()

# ── General ───────────────────────────────────────────────────────────────────

st.subheader("General")

with st.expander("What is data contamination and why does it matter?"):
    st.markdown("""
    **Data contamination** occurs when code used to evaluate an LLM was part of
    its pre-training data. Because LLMs are trained on massive scrapes of public
    GitHub repositories, any code that existed publicly before the model's training
    cutoff date may have been memorized — and a model that reproduces memorized
    code will appear more capable than it actually is.

    This has been empirically documented:

    - **For code generation:** LLMs achieve 84–89% correctness on widely-used
      synthetic benchmarks (e.g. HumanEval), but only **25–34%** on real-world
      class-level tasks drawn from open-source repositories
      *(Rahman et al., arXiv:2510.26130, under review at EMSE)*.
      The gap exists because synthetic benchmarks are fixed, well-known, and
      likely seen during training.

    - **For LLM-generated code detection:** A detection classifier shows
      **significant performance degradation** when evaluated on post-cutoff
      (unseen) data compared to pre-cutoff (potentially seen) data
      *(Rahman et al., arXiv:2409.01382)*. Contamination in the evaluation set
      produces overly optimistic detection results.

    Reviewers at major SE venues (EMSE, TSE, ICSE, FSE, ASE) now routinely
    require authors to demonstrate that their evaluation data is contamination-free.
    """)

with st.expander("How does SIEVE guarantee contamination-free data?"):
    st.markdown("""
    SIEVE filters repositories by **creation date** using the GitHub `created:`
    search qualifier. A repository created after an LLM's training cutoff date
    **cannot have existed** when the model was trained — making every function
    and class extracted from it contamination-free by construction.

    This is stronger than filtering by last push date (`pushed:`), which would
    still include repositories created before the cutoff that were merely updated
    after it. Old code in those repos could still be in the training data.

    **How to use it:** Set **Start Date** to the training cutoff of the LLM(s)
    you are evaluating. All extracted code will be from repositories that
    post-date that cutoff.
    """)

with st.expander("What training cutoff dates should I use?"):
    st.markdown("""
    Use the published or estimated training cutoff of the model(s) you are studying:

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

    If evaluating multiple models, use the **latest** cutoff among them to
    guarantee contamination-free data for all models simultaneously.

    When submitting papers, you can state:
    > *"Our evaluation corpus was built using SIEVE, collecting only from
    > repositories created after [DATE]. By construction, no extracted code
    > could have appeared in the pre-training data of any evaluated model."*
    """)
    st.markdown("""
    **SIEVE** (Software Ingestion & Extraction for Verifiable Evaluation) is a
    parameterized GitHub corpus builder for software engineering research. It
    discovers repositories, extracts function and class-level code snippets,
    deduplicates them, and optionally annotates each snippet with a
    contamination indicator (`llm_score`).

    The key design goal is **contamination-awareness**: by restricting discovery
    to repositories created or last updated after a configurable cutoff date,
    SIEVE ensures the corpus is unlikely to have been seen during the pre-training
    of large language models — making it suitable for rigorous LLM evaluation.
    """)

with st.expander("What does 'contamination-free' mean?"):
    st.markdown("""
    A dataset is *contaminated* if its contents appeared in the training data of
    the model being evaluated. This invalidates benchmarks — a model may appear
    to perform well simply because it memorized the answers.

    SIEVE addresses this by:
    1. **Temporal filtering** — only collecting repositories whose last commit
       falls after the training cutoff of major LLMs (configurable via Start/End
       Date).
    2. **LLM Score annotation** — scoring each extracted snippet with
       P(AI-generated), so researchers can further filter or stratify by origin.
    """)

with st.expander("What languages does SIEVE support?"):
    st.markdown("""
    Currently: **Python, Java, JavaScript, and C++**.

    Extraction uses [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
    parsers for all four languages, so the extraction is syntax-aware rather than
    regex-based.
    """)

with st.expander("What granularities does SIEVE extract?"):
    st.markdown("""
    - **Function level** — individual function/method bodies, extracted with their
      signature, docstring, parameters, return annotation, and a list of imports
      actually used in the function body.
    - **Class level** — full class bodies, extracted with a skeleton
      (signatures + docstrings, no implementation) alongside the full source.

    You can extract both granularities in a single run.
    """)

# ── GitHub Token ──────────────────────────────────────────────────────────────

st.subheader("GitHub Token")

with st.expander("Do I need a GitHub token?"):
    st.markdown("""
    A token is **strongly recommended**. Without one, the GitHub API limits you
    to 60 requests per hour, which is enough for only a handful of repositories.

    With a token, the limit increases to 5,000 requests per hour — sufficient
    for collecting hundreds of repositories.
    """)

with st.expander("How do I get a GitHub Personal Access Token (PAT)?"):
    st.markdown("""
    1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
    2. Click **Generate new token (classic)**
    3. Give it a name (e.g. `sieve-token`)
    4. Select the **`public_repo`** scope (read-only access to public repos)
    5. Click **Generate token** and copy it
    6. Paste it into the **GitHub Token** field in the sidebar

    Your token is never stored — it is only used for the duration of the current
    session.
    """)

# ── LLM Score ─────────────────────────────────────────────────────────────────

st.subheader("LLM Score")

with st.expander("What is the LLM Score?"):
    st.markdown("""
    Each extracted snippet can be annotated with `llm_score` — a probability
    in [0, 1] indicating how likely the snippet is to have been AI-generated
    rather than human-written.

    - **Score near 0** → likely human-written
    - **Score near 1** → likely AI-generated

    This is produced by a fine-tuned
    [CodeBERT](https://huggingface.co/microsoft/codebert-base) classifier trained
    on 114,186 labeled samples across Python, Java, JavaScript, and C++.
    """)

with st.expander("How accurate is the LLM Score?"):
    st.markdown("""
    On a held-out test set of 11,419 samples:

    | Metric | Value |
    |---|---|
    | Accuracy | 94.42% |
    | F1 | 94.78% |
    | Precision | 92.57% |
    | Recall | 97.09% |
    | AUROC | 99.02% |

    Performance is consistent across all four languages (F1 range: 0.94–0.95).
    See the **About the Classifier** page for full details.
    """)

with st.expander("Why does the first run with LLM Score take longer?"):
    st.markdown("""
    The classifier weights (~500MB) are hosted on HuggingFace Hub and downloaded
    automatically on first use. Subsequent runs use the cached weights and start
    immediately.
    """)

# ── Output ────────────────────────────────────────────────────────────────────

st.subheader("Output")

with st.expander("What format is the output?"):
    st.markdown("""
    SIEVE outputs:
    - **`functions.jsonl`** — one JSON object per line, each representing a
      function/method record
    - **`classes.jsonl`** — one JSON object per line, each representing a class
      record
    - **`manifest.json`** — metadata about the run: config, repo list, summary
      statistics

    Optionally, Parquet format is also available for larger-scale workflows.

    Each record includes: `repo`, `file_path`, `language`, `source_code`,
    `signature`/`skeleton`, `used_imports`, `docstring`, line numbers, and
    `llm_score` (if enabled).
    """)

with st.expander("How do I download the corpus?"):
    st.markdown("""
    After a successful run, download buttons appear in the **Download Corpus**
    section of the results page. You can download individual files or a ZIP
    bundle containing all output files.
    """)

with st.expander("Can I run SIEVE locally via the command line?"):
    st.markdown("""
    Yes. Install the package and run:

    ```bash
    pip install -e .
    sieve run --language Python \\
               --start-date 2024-01-01 \\
               --end-date 2025-01-01 \\
               --min-stars 500 \\
               --max-repos 100 \\
               --output-dir ./my_corpus
    ```

    See the [GitHub repository](https://github.com/mrsumitbd/sieve) for full
    CLI documentation.
    """)

with st.expander("Why do I see repos with the Unlicense even with Engineered Projects enabled?"):
    st.markdown("""
    The Engineered Projects filter's license check (Stage 1) only excludes
    **non-software licenses** — Creative Commons variants (CC-BY, CC0, CC-BY-SA)
    and font licenses (OFL-1.1). These are associated with documentation sites,
    datasets, and font projects rather than software.

    The **Unlicense** is a valid software license — a public domain dedication
    equivalent to "no restrictions on use." It is commonly used by legitimate,
    production-quality software projects. Its presence does not indicate a non-software
    repository, so SIEVE correctly allows it through Stage 1.

    If you want to exclude Unlicense repos, post-filter the manifest using the
    `license_spdx` field:

    ```python
    import json

    ALLOWED = {"MIT", "Apache-2.0", "GPL-2.0-only", "GPL-3.0-only",
               "BSD-2-Clause", "BSD-3-Clause", "LGPL-2.1-only", "ISC"}

    with open("manifest.json") as f:
        manifest = json.load(f)

    filtered = [r for r in manifest["repos"]
                if r.get("license_spdx") in ALLOWED]
    ```
    """)

with st.expander("Why does a repository show 0 functions/methods or 0 classes?"):
    st.markdown("""
    This is expected behavior — not a bug. There are several reasons a repository
    may contribute 0 records for a given granularity:

    **1. The repository has very few real definitions after filtering.**
    SIEVE filters out forward declarations, test files, and other non-substantive
    code. A repository may appear to have classes or functions at the file level
    but yield 0 after filtering.

    For example, `ocornut/imgui` is a single-header C++ library. Pass 1 counted
    9 classes, but most are forward declarations (`class ImGuiContext;`) which
    SIEVE correctly excludes. After filtering, 0 real class definitions remain.

    **2. Stratified cap allocation rounds a small repo to 0.**
    When **Max Functions/Methods** or **Max Classes** is set, SIEVE allocates
    slots proportionally across repos. A very small repo competing with much
    larger ones may receive a 0 allocation after rounding.

    For example, with Max Classes = 30 across 10 repos where `godot` has 8,168
    classes and `imgui` has 9, imgui's proportional share is less than 1 and may
    round to 0. SIEVE uses a minimum-of-1 allocation to avoid this, but if the
    repo then returns 0 real classes after filtering (see point 1), nothing can
    be done.

    **3. All source files are test files.**
    SIEVE skips files in `test/`, `tests/`, `spec/` directories and files named
    `test_*.py`, `*_test.cpp`, etc. A repo whose source is predominantly tests
    may yield 0 records.

    **4. No matching granularity.**
    If you select only **Class** granularity, a repository that only contains
    standalone functions will show 0 classes. Similarly for **Function/Method**
    in a class-only repo.

    **5. Minified or generated files (JavaScript only).**
    SIEVE skips `.min.js` files and files with very long lines (webpack bundles).
    A JavaScript repo that ships only minified builds may yield 0 records.

    **What to do:** SIEVE's Pass 3 automatically redistributes unmet slots to
    other repos with remaining capacity. If you still fall short, try increasing
    **Max Functions/Methods** and **Max Classes** or reducing **Max Repos** so
    each repo has a larger proportional allocation.
    """)

# ── Citation ──────────────────────────────────────────────────────────────────

st.subheader("Citation")

with st.expander("The tool crashes or becomes unresponsive during a large run. What should I do?"):
    st.markdown("""
    SIEVE is hosted on HuggingFace Spaces with **16 GB RAM**. For very large runs,
    the process can exhaust available memory and crash. This is expected behavior —
    not a bug.

    **Common causes:**
    - Discovering thousands of repos (large date windows + low star thresholds)
    - Processing very large repos like `godotengine/godot`, `tensorflow/tensorflow`,
      or `home-assistant/core` which can contain 100,000+ extractable functions
    - Running with C++ which is significantly slower and more memory-intensive due
      to tree-sitter's GLR parser

    **Recommended approach — process in batches:**

    Instead of one large run, split your corpus into smaller batches:

    ```python
    # Instead of one run for all of 2024:
    # start_date=2024-01-01, end_date=2024-12-31

    # Run in quarterly batches:
    # Batch 1: start_date=2024-01-01, end_date=2024-03-31
    # Batch 2: start_date=2024-04-01, end_date=2024-06-30
    # Batch 3: start_date=2024-07-01, end_date=2024-09-30
    # Batch 4: start_date=2024-10-01, end_date=2024-12-31
    ```

    **Other tips:**
    - Set **Max Repos** to 20–50 per run
    - Set **Max Functions** and **Max Classes** caps — these trigger SIEVE's
      two-pass extraction which prevents loading entire repos into memory
    - Avoid combining large date windows with low star thresholds (e.g. min_stars=5)
      which can return thousands of repos
    - For C++, use smaller batches than for other languages

    The manifest files from each batch can be combined for downstream analysis.
    """)

with st.expander("How do I cite SIEVE?"):
    st.markdown("""
    If you use SIEVE in your research, please cite:

    ```bibtex
    @inproceedings{rahman2027sieve,
      title     = {{SIEVE}: A Contamination-Aware GitHub Corpus Builder for
                   Software Engineering Research},
      author    = {Rahman, Musfiqur and Shihab, Emad},
      booktitle = {Proceedings of the 24th International Conference on
                   Mining Software Repositories (MSR)},
      year      = {2027}
    }
    ```
    *(Citation will be updated upon publication.)*
    """)