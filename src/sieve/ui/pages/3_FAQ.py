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

# ── Citation ──────────────────────────────────────────────────────────────────

st.subheader("Citation")

with st.expander("How do I cite SIEVE?"):
    st.markdown("""
    If you use SIEVE in your research, please cite:

    ```bibtex
    @inproceedings{rahman2026sieve,
      title     = {{SIEVE}: A Contamination-Aware GitHub Corpus Builder for
                   Software Engineering Research},
      author    = {Rahman, Musfiqur and Shihab, Emad},
      booktitle = {Proceedings of the 42nd International Conference on
                   Software Maintenance and Evolution (ICSME)},
      year      = {2026}
    }
    ```
    *(Citation will be updated upon publication.)*
    """)