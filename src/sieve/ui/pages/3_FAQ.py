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

with st.expander("What is SIEVE?"):
    st.markdown("""
    **SIEVE** (Software Ingestion & Extraction for Verifiable Evaluation) is a
    parameterized GitHub corpus builder for software engineering research. It
    discovers repositories, extracts function- and class-level code snippets,
    deduplicates them, and annotates each with structural code metrics, parsed
    dependency information, and — optionally — an AI-generation likelihood
    score (`llm_score`) and full AST features.

    Rather than a static, pre-built dataset, SIEVE is a *tool* you run with
    your own settings to build the corpus a given study needs: **contamination-aware
    LLM evaluation** (by restricting discovery to repositories created after a
    configurable cutoff date), **dependency-management research**, **structural
    analysis of LLM-generated code**, or general-purpose corpus building for
    any language-specific study.
    """)

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
      data compared to pre-cutoff (potentially seen) data
      *(Rahman et al., arXiv:2409.01382)*. Contamination in the evaluation set
      produces overly optimistic detection results.

    Contamination is a growing concern for evaluating LLMs on code, and
    reviewers increasingly ask authors to address it in their evaluation data.
    """)

with st.expander("How does SIEVE support contamination-aware evaluation?"):
    st.markdown("""
    SIEVE filters repositories by **creation date** using the GitHub `created:`
    search qualifier. A repository created after an LLM's training cutoff date
    did not exist when the model was trained, so every function and class is
    extracted from a repository that post-dates the cutoff. This makes the corpus
    temporally bounded relative to the model.

    The filter bounds a repository's **age**. It does not verify where each
    snippet came from: a new repository can contain code copied or migrated from
    older sources, and SIEVE does not check for this.

    This is stronger than filtering by last push date (`pushed:`), which would
    still include repositories created before the cutoff that were merely updated
    after it. Old code in those repos could still be in the training data.

    **How to use it:** Set **Repo Creation Lower Bound** to the release date of
    the LLM(s) you are evaluating (see the next entry for why the release date
    rather than the knowledge cutoff). All extracted code comes from
    repositories created after that date.
    """)

with st.expander("What training cutoff dates should I use?"):
    st.markdown("""
    **Use the model's public release date, not its knowledge cutoff date.**

    These two dates are often confused but are different:

    - **Knowledge cutoff** — the last date of data used for *pre-training*. For
      example, GPT-4's knowledge cutoff is April 2023.
    - **Release date** — when the model became publicly available. GPT-4 was
      released in March 2023.

    **Why the release date is safer:** Between the knowledge cutoff and the
    release date, models are typically fine-tuned on additional data —
    instruction-following examples, RLHF preference data, and sometimes
    supervised fine-tuning on code. This fine-tuning data postdates the
    pre-training cutoff but predates the release. By setting
    **Repo Creation Lower Bound** to the release date, you restrict
    the collection to repositories created after both the pre-training and
    fine-tuning phases of the model.

    Using the knowledge cutoff instead of the release date leaves a window
    (sometimes weeks, sometimes months) where fine-tuning data could
    contaminate your evaluation set.

    For release dates and knowledge cutoffs across about 50 entries covering models from OpenAI,
    Anthropic, Google, Meta, Mistral, DeepSeek, Qwen, xAI, and BigCode, see
    the reference table on the **Documentation** page (Contamination-Aware
    Evaluation → How to Choose Your Cutoff Date).

    If evaluating multiple models simultaneously, use the **latest** release
    date among them, so that every sampled repository was created after all of
    the models' training phases.

    When submitting papers, you can state:
    > *"Our evaluation corpus was built using SIEVE, restricted to GitHub
    > repositories created after [DATE] (the release date of [Model]). This
    > bounds repository age relative to the evaluated models; it does not
    > verify the origin of individual snippets, which we discuss as a
    > limitation."*
    """)

with st.expander("What does 'contamination-aware' mean?"):
    st.markdown("""
    A dataset is *contaminated* if its contents appeared in the training data of
    the model being evaluated. This invalidates benchmarks — a model may appear
    to perform well simply because it memorized the answers.

    SIEVE supports contamination-aware evaluation by:
    1. **Temporal filtering** — only collecting repositories whose *creation date*
       falls after the training cutoff or release date of the LLMs under study
       (configurable via Start/End Date).
    2. **LLM Score annotation** — scoring each extracted snippet with
       P(AI-generated), so researchers can further filter or stratify by origin.

    **What this does and does not establish:** the creation-date filter shows that
    a repository is newer than a given date. It does not prove that each snippet
    in it was first written after that date, because a new repository can contain
    code copied or migrated from older sources. SIEVE does not verify the origin
    of individual snippets, so "contamination-aware" means the corpus is
    temporally bounded, not that it is proven free of contamination.
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

    You can extract both granularities in a single run, which is the default. To extract only
    one, use the **Extract functions** and **Extract classes** checkboxes in the web UI, or
    `--granularity function`, `--granularity class` or `--granularity both` on the CLI (`granularity`
    in a config file). At least one level is required. A cap of `0` means "no cap" in the web UI
    only; the CLI and config file take no value for no cap, or a number of at least 1.
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

    In the web UI the token is not stored — it is only used for the duration of the current
    session. In the CLI, set the `GITHUB_TOKEN` environment variable (or pass `--github-token`).
    The optional Hugging Face token for LLM scoring is read from `HF_TOKEN` in the CLI; there is no
    `--hf-token` flag. Tokens are never written to `manifest.json`. The web UI does not read either
    environment variable.
    """)

with st.expander("What happens when I hit the GitHub rate limit?"):
    st.markdown("""
    By default the run fails fast: a rate limit that will not clear within about a minute stops the
    run with a message that gives the reset time. If some repositories had already been found, the run
    continues with them and the results are marked partial.

    On the CLI, `--wait-for-rate-limit` makes the run sleep until the limit resets and carry on. A wait
    can last up to an hour, and nothing is written until the run ends. **The web UI never waits**, so
    use a token there and keep runs small. See **GitHub Rate Limits** on the Documentation page.
    """)

with st.expander("How do I know the repository list is complete?"):
    st.markdown("""
    Read the `discovery` block of `manifest.json`. `complete` is `true` when every matching
    repository was reachable. GitHub search returns at most 1,000 results per query, so SIEVE
    splits the creation-date range into windows to collect more. If one day alone holds more than 1,000
    matching repositories, only its 1,000 most-starred are collected and the day is listed in
    `truncated_windows`. Repositories whose contributors could not be fetched are listed in `excluded`.
    Contributor counts are exact and exclude bots; `null` means GitHub would not list the contributors.
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

with st.expander("The tool crashes or becomes unresponsive during a large run. What should I do?"):
    st.markdown("""
    The public SIEVE Space on HuggingFace has **8 GB of RAM**, so do not run large corpora there.
    For very large runs, the process can exhaust available memory and crash. This is expected
    behavior — not a bug. Run large jobs on your own machine (see **Run the Web UI Locally** on the
    Documentation page) or with the CLI.

    Repositories are processed one after another, not in parallel. A large run is therefore slow,
    not just big.

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

with st.expander("Can I run SIEVE locally via the command line?"):
    st.markdown("""
    Yes. From the repository root, install the dependencies (this also installs the `sieve` command)
    and run:

    ```bash
    pip install -r requirements.txt
    export GITHUB_TOKEN=<your token>
    sieve run --language Python \\
               --start-date 2024-01-01 \\
               --end-date 2025-01-01 \\
               --min-stars 500 \\
               --max-repos 100 \\
               --output-dir ./my_corpus
    ```

    Deduplication is off unless you add `--deduplicate`. See the
    [GitHub repository](https://github.com/mrsumitbd/sieve) for the full CLI documentation and the
    **Parameter Parity Across Interfaces** table on the Documentation page for every flag.
    """)

with st.expander("Can I run the web UI on my own machine?"):
    st.markdown("""
    Yes, and you should for large runs, since the public Space has 8 GB of RAM. You need Python 3.11
    or newer and git. In short:

    ```bash
    git clone https://github.com/mrsumitbd/sieve.git && cd sieve
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run src/sieve/ui/Home.py
    ```

    Then open `http://localhost:8501`, paste a GitHub token into the sidebar, and stop the server with
    `Ctrl+C`. The step-by-step guide is under **Run the Web UI Locally** on the Documentation page.
    """)

with st.expander("`pip install` fails or LLM scoring breaks on an Intel Mac. What do I do?"):
    st.markdown("""
    PyTorch has no macOS x86_64 wheels after 2.2.2, and that version needs older NumPy and
    Transformers. Run `pip install "torch==2.2.2" "numpy<2" "transformers<5"`. See **Install
    Troubleshooting** on the Documentation page.
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

    **1. The repository has no extractable definitions of that kind.**
    SIEVE counts and extracts only the definitions its parsers recognise, in the files it visits.
    Forward declarations (for example a C++ `class Foo;` with no body) are not counted or extracted,
    so a header that only forward-declares its classes contributes 0 classes. Pass 1 counts exactly
    what Pass 2 extracts, so a Pass 1 count of 0 means there was nothing for Pass 2 to extract.

    **2. The cap is smaller than the number of repositories.**
    When **Max Functions/Methods** or **Max Classes** is set, SIEVE gives every
    repository that has records at least 1 slot and splits the rest in proportion
    to repository size. If the cap is smaller than the number of repositories
    with records, that minimum cannot be met: the largest repositories get 1 slot
    each and the others get 0.

    For example, with Max Classes = 30 across 10 repos where one repo has thousands of classes
    and another has 9, the smaller repo still receives at least 1 slot rather than being rounded
    down to 0.

    **3. Its source is skipped.**
    SIEVE leaves out directories such as `vendor`, `node_modules`, `build`, `dist` and `venv`, the
    test directories `test/`, `tests/`, `__tests__/`, `spec/` and `specs/`, and test files by name:
    names starting with `test_` or `Test`, and names ending in `_test.py`, `Test.java`, `.test.js`,
    `.spec.js`, `_test.cpp` or `_test.cc`. These rules apply to every language, so a file such as
    `Testimonials.js` is dropped too. A repo whose source sits mostly in those places may yield 0
    records. The `extraction` block of `manifest.json` lists the skipped directory names and counts
    the files they held. The full rules are under **What Extraction Skips** on the Documentation page.

    **4. No matching granularity.**
    If you select only **Class** granularity, a repository that only contains
    standalone functions will show 0 classes. Similarly for **Function/Method**
    in a class-only repo.

    **5. Unsupported or minified files.**
    Only `.py` (Python), `.java` (Java), `.js` and `.jsx` (JavaScript) and the C++ extensions
    `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.hxx` are processed; TypeScript is not. For JavaScript,
    `.min.js` files and files with very long lines (webpack bundles) are skipped, so a repo that
    ships only minified builds may yield 0 records.

    **What to do:** SIEVE's Pass 3 automatically redistributes unmet slots to
    other repos with remaining capacity. If you still fall short, try increasing
    **Max Functions/Methods** and **Max Classes** or reducing **Max Repos** so
    each repo has a larger proportional allocation.
    """)

with st.expander("Is the Pass 1 count exact?"):
    st.markdown("""
    Yes, with one narrow exception. Pass 1 (counting, used to split the caps across repositories)
    selects files with the same code as Pass 2 (extraction) and finds functions and classes with the
    same walk, so with no caps both give the same numbers. The exception is a file that parses but
    then fails while its records are built: Pass 1 counts it and Pass 2 drops it. A repository that
    changed between the Pass 1 clone and the Pass 2 clone can also differ.
    """)

with st.expander("Are the results a fixed snapshot of GitHub?"):
    st.markdown("""
    No. SIEVE searches GitHub when the run happens and clones each repository's default branch HEAD
    at that moment, so a rerun on another day can give a different corpus. The manifest records
    `generated_at`, a `collected_at` timestamp and the star count per repository, and each record
    carries the `commit_sha` that was cloned, but it does not pin a snapshot you could re-fetch.
    Also, the creation-date filter uses the repository's GitHub creation date: a repository with
    imported history can contain commits older than that date. See **Known Limitations** on the
    Documentation page.
    """)

# ── Citation ──────────────────────────────────────────────────────────────────

st.subheader("Citation")

with st.expander("How do I cite SIEVE?"):
    st.markdown("""
    If you use SIEVE in your research, please cite:

    ```bibtex
    @article{rahman2026sieve,
      title={SIEVE: A Parameterized GitHub Corpus Builder for Software Engineering Research},
      author={Rahman, Musfiqur and Begoug, Mahi and Shihab, Emad},
      journal={arXiv preprint arXiv:2610.22544},
      year={2026}
    }
    ```
    The same citation is also available in the repository's `CITATION.cff` file.
    """)