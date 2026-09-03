"""
ui/app.py

Streamlit web interface for SIEVE.
Runs the pipeline synchronously in the main thread.
Progress is streamed live using st.status().

Run with: streamlit run src/sieve/ui/app.py
"""

import json
import os
import random
import tempfile
import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from sieve.config import SIEVEConfig, Language, Granularity, ExportFormat
from sieve.pipeline import run_pipeline
from sieve.core.quality import check_cloc, CLOC_INSTALL_INSTRUCTIONS

# ─── Demo dataset helpers ─────────────────────────────────────────────────────

_DEMO_DIR = Path(__file__).parent / "demo_data"

# Language → Streamlit syntax highlighting identifier
_LANG_HIGHLIGHT = {
    "Python":     "python",
    "Java":       "java",
    "JavaScript": "javascript",
    "C++":        "cpp",
}


def load_demo_dataset() -> dict:
    fn_path  = _DEMO_DIR / "functions.jsonl"
    cl_path  = _DEMO_DIR / "classes.jsonl"
    manifest = json.loads((_DEMO_DIR / "manifest.json").read_text())

    return {
        "total_repos_discovered":          3,
        "total_repos_after_quality_filter": 3,
        "total_repos_processed":           3,
        "total_repos_failed":              0,
        "total_functions":                 manifest["summary"]["total_functions"],
        "total_classes":                   manifest["summary"]["total_classes"],
        "failed_repos":                    [],
        "output_paths": {
            "functions_jsonl": str(fn_path),
            "classes_jsonl":   str(cl_path),
            "manifest":        str(_DEMO_DIR / "manifest.json"),
        },
        "output_dir":    str(_DEMO_DIR),
        "repo_stats":    manifest["repo_stats"],
        "repo_metadata": manifest.get("repo_metadata", []),
        "_is_demo":      True,
    }


def _build_zip(output_paths: dict) -> bytes:
    """Bundle all output files into a ZIP for download."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, path in output_paths.items():
            p = Path(path)
            if p.exists():
                zf.write(p, arcname=p.name)
    return buf.getvalue()


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SIEVE",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Defaults ───────────────────────────────────────────────────

for key in ("summary", "error", "sample", "output_dir_path", "pipeline_log"):
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🔬 SIEVE")
st.caption("**S**oftware **I**ngestion & **E**xtraction for **V**erifiable **E**valuation")
st.markdown("Curate contamination-aware code corpora from GitHub for SE research.")

st.info(
    "**Why contamination-free data matters** — "
    "LLMs trained on public GitHub code may \"remember\" any repository that existed "
    "before their training cutoff. Evaluating on that code produces inflated, unreliable "
    "results. SIEVE collects only from repositories **created after your specified cutoff**, "
    "guaranteeing no overlap with training data. "
    "See the **Documentation** page for the full rationale and how to choose your cutoff date.",
    icon="🛡️",
)

st.divider()

# ─── Sidebar: Parameter Form ─────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Parameters")

    language = st.selectbox(
        "Programming Language",
        options=[l.value for l in Language],
        index=0,
    )

    # Default end_date = 30 days before today
    _default_end = date.today() - __import__("datetime").timedelta(days=30)

    start_date = st.date_input(
        "Repo Creation Lower Bound",
        value=date(2024, 1, 1),
        help=(
            "Only repos created on or after this date are included. "
            "For contamination-free evaluation, set this to the **release date** "
            "of the LLM(s) you are studying — data used for both pre-training "
            "and fine-tuning predates the release, so any repo created after "
            "the release date is guaranteed unseen by the model."
        ),
    )
    end_date = st.date_input(
        "Repo Creation Upper Bound",
        value=_default_end,
        help=(
            "Only repos created on or before this date are included. "
            "Default: one month before today, allowing time for initial "
            "commits and activity to accumulate in recently created repos."
        ),
    )
    if end_date < start_date:
        st.error("Repo Creation Upper Bound must be on or after Repo Creation Lower Bound.")

    st.subheader("Repository Filters")
    min_stars        = st.number_input("Minimum Stars",        min_value=0, value=50,  step=10)
    min_contributors = st.number_input("Minimum Contributors", min_value=1, value=5,   step=1)
    max_repos        = st.number_input("Max Repos (0 = no cap)",      min_value=0, value=100, step=10)
    max_functions    = st.number_input(
        "Max Functions (-1 = skip, 0 = no cap)", min_value=-1, value=0, step=500,
        help="Cap on total extracted functions. Set to -1 to skip function extraction entirely.",
    )
    max_classes = st.number_input(
        "Max Classes (-1 = skip, 0 = no cap)", min_value=-1, value=0, step=100,
        help="Cap on total extracted classes. Set to -1 to skip class extraction entirely.",
    )

    st.subheader("Content Filters")
    engineered_only = st.toggle(
        "Engineered Projects Only", value=False,
        help=(
            "Apply engineered project filter (Munaiah et al., 2017). "
            "Excludes non-software licenses, repos without releases, "
            "and bottom-Q1 repos. Slower but higher-quality corpus."
        ),
    )

    if engineered_only and not check_cloc():
        st.warning(
            "**cloc not found.** SIEVE will fall back to an AST-based LOC "
            "counter. For best results, install cloc before running with this filter.",
            icon="⚠️",
        )
    annotate_llm_score = st.toggle(
        "Annotate LLM Score", value=False,
        help=(
            "Score each snippet with P(AI-generated) using a fine-tuned "
            "CodeBERT classifier. Weights (~500MB) are downloaded from "
            "HuggingFace Hub automatically on first use."
        ),
    )
    export_ast = st.toggle(
        "Export AST-Derived Features", value=False,
        help=(
            "Include AST-derived structural features — parse tree depth, "
            "node count, and node type distribution — in every exported record."
        ),
    )
    export_full_ast = st.toggle(
        "Include Full AST in Export", value=False,
        disabled=not export_ast,
        help=(
            "Include the complete parse tree as nested JSON in each record. "
            "Only available when 'Export AST-Derived Features' is enabled. "
            "Significantly increases file size — use with small corpora."
        ),
    ) if export_ast else False

    st.subheader("Processing")
    deduplicate_flag = st.toggle("Deduplicate", value=False)
    dedup_threshold  = st.slider(
        "Dedup Similarity Threshold",
        min_value=0.5, max_value=1.0, value=0.8, step=0.05,
        disabled=not deduplicate_flag,
    )

    st.subheader("Output")
    export_format = st.selectbox(
        "Export Format",
        options=[f.value for f in ExportFormat],
        format_func=lambda x: {"jsonl": "JSONL", "parquet": "Parquet", "both": "Both (JSONL + Parquet)"}[x],
        index=0,
        help="JSONL is human-readable and line-oriented. Parquet is columnar and efficient for large-scale analysis.",
    )

    st.subheader("API Tokens")
    github_token = st.text_input(
        "GitHub Token (PAT)", type="password",
        help="Recommended. Without it, rate limit is 60 requests/hour.",
    )

    import os
    _survey_token_available = bool(os.environ.get("SIEVE_SURVEY_TOKEN"))
    if _survey_token_available:
        use_survey_token = st.toggle(
            "🎓 Use our API token (survey participants)",
            value=False,
            help=(
                "Toggle this on if you don't have a GitHub token. "
                "Provided for survey participants only. "
                "Please be considerate — run small corpora (Max Repos ≤ 20)."
            ),
        )
    else:
        use_survey_token = False

    run_button = st.button(
        "▶ Run SIEVE",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    demo_button = st.button(
        "📦 Load Example Dataset",
        use_container_width=True,
        help=(
            "Load a pre-built demo corpus (54 functions/methods, 8 classes "
            "across 3 synthetic repositories). No GitHub token required."
        ),
    )


# ─── Pipeline Execution ───────────────────────────────────────────────────────

if run_button:
    # Derive granularity from max values: -1 means skip that granularity
    granularity = []
    if int(max_functions) != -1:
        granularity.append("function")
    if int(max_classes) != -1:
        granularity.append("class")

    if not granularity:
        st.error("Both Max Functions and Max Classes are set to -1. At least one granularity must be enabled.")
    else:
        # Clear previous run state before starting new run
        st.session_state.pipeline_log = []
        st.session_state.summary = None
        tmp_dir = tempfile.mkdtemp(prefix="sieve_")
        st.session_state.output_dir_path = tmp_dir

        try:
            import os
            resolved_token = (
                os.environ.get("SIEVE_SURVEY_TOKEN")
                if use_survey_token
                else (github_token if github_token else None)
            )
            config = SIEVEConfig(
                language=language,
                start_date=start_date,
                end_date=end_date,
                min_stars=int(min_stars),
                min_contributors=int(min_contributors),
                max_repos=int(max_repos)         if max_repos     > 0 else None,
                max_functions=int(max_functions) if max_functions > 0 else None,
                max_classes=int(max_classes)     if max_classes   > 0 else None,
                granularity=granularity,
                engineered_only=engineered_only,
                annotate_llm_score=annotate_llm_score,
                export_ast=export_full_ast if export_ast else False,
                deduplicate=deduplicate_flag,
                dedup_threshold=dedup_threshold,
                output_dir=tmp_dir,
                export_format=export_format,
                github_token=resolved_token,
            )
        except Exception as e:
            st.error(f"Configuration error: {e}")
            st.stop()

        st.session_state.summary = None
        st.session_state.error   = None
        st.session_state.sample  = None
        st.session_state.pipeline_log = []

        # Stream live progress during the run
        with st.status("Running SIEVE pipeline...", expanded=True) as status:
            def progress_callback(msg: str, current: int, total: int):
                ts = datetime.now().strftime("%H:%M:%S")
                st.write(f"`{ts}` {msg}")
                st.session_state.pipeline_log.append(f"{ts} {msg}")

            try:
                summary = run_pipeline(config, progress_callback=progress_callback)
                st.session_state.summary = summary
                _pipeline_summary = summary.get("pipeline_summary", "")
                _status_label = f"✅ Pipeline complete! {_pipeline_summary}"
                status.update(label=_status_label, state="complete", expanded=False)
            except Exception as e:
                st.session_state.error = str(e)
                status.update(label="❌ Pipeline failed", state="error", expanded=True)
                st.error(str(e))

# ─── Persistent Pipeline Log ─────────────────────────────────────────────────

if st.session_state.pipeline_log:
    _summary = st.session_state.get("summary") or {}
    _pipeline_summary = _summary.get("pipeline_summary", "")
    _expander_label = f"✅ Pipeline complete! {_pipeline_summary}"
    with st.expander(_expander_label, expanded=False):
        st.code("\n".join(st.session_state.pipeline_log), language=None)


# ─── Demo Dataset Load ────────────────────────────────────────────────────────

if demo_button:
    try:
        st.session_state.summary          = load_demo_dataset()
        st.session_state.error            = None
        st.session_state.sample           = None
        st.session_state.output_dir_path  = None
    except Exception as e:
        st.error(f"Could not load demo dataset: {e}")


# ─── Results ─────────────────────────────────────────────────────────────────

if st.session_state.error and not st.session_state.summary:
    st.error(f"Last run failed: {st.session_state.error}")

if st.session_state.summary:
    s = st.session_state.summary
    st.divider()

    if s.get("_is_demo"):
        st.info(
            "**📦 Demo dataset loaded.** "
            "Showing pre-built corpus from 3 synthetic repositories. "
            "Click **▶ Run SIEVE** with a GitHub token to build a real corpus.",
            icon="ℹ️",
        )

    # ── Summary Metrics ───────────────────────────────────────────────────────

    st.subheader("📊 Run Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Repos Processed", s["total_repos_processed"])
    c2.metric("Functions/Methods", s["total_functions"])
    c3.metric("Classes",         s["total_classes"])
    c4.metric("Total Records",   s["total_functions"] + s["total_classes"])
    c5.metric("Failed Repos",    s["total_repos_failed"])

    if s.get("failed_repos"):
        with st.expander(f"Failed repos ({len(s['failed_repos'])})"):
            for r in s["failed_repos"]:
                st.text(r)

    # ── Download Buttons ──────────────────────────────────────────────────────

    output_paths = s.get("output_paths", {})
    existing     = {k: v for k, v in output_paths.items() if Path(v).exists()}

    if existing and not s.get("_is_demo"):
        st.divider()
        st.subheader("⬇️ Download Corpus")

        dl_cols = st.columns(len(existing) + 1)

        for col, (label, path) in zip(dl_cols, existing.items()):
            p    = Path(path)
            data = p.read_bytes()
            col.download_button(
                label=f"📄 {p.name}",
                data=data,
                file_name=p.name,
                mime="application/json" if p.suffix == ".jsonl" else "application/octet-stream",
                use_container_width=True,
            )

        # ZIP bundle
        zip_data = _build_zip(existing)
        dl_cols[-1].download_button(
            label="🗜️ Download All (ZIP)",
            data=zip_data,
            file_name="sieve_corpus.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # ── Charts ────────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("📈 Dataset Statistics")

    repo_stats = s.get("repo_stats", [])

    if repo_stats:
        df          = pd.DataFrame(repo_stats)
        short_names = [r.split("/")[-1] for r in df["repo"]]

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Functions & Classes per Repository**")
            chart_df = pd.DataFrame({
                "Repo":  short_names * 2,
                "Count": list(df["functions"]) + list(df["classes"]),
                "Type":  ["Functions/Methods"] * len(df) + ["Classes"] * len(df),
            })
            st.bar_chart(
                chart_df.pivot(index="Repo", columns="Type", values="Count"),
                color=["#4e79a7", "#f28e2b"],
            )

        with chart_col2:
            st.markdown("**Stars vs. Extracted Functions/Methods**")
            scatter_df = pd.DataFrame({
                "Repo":              short_names,
                "Stars":             df["stars"],
                "Functions/Methods": df["functions"],
            })
            st.scatter_chart(scatter_df, x="Stars", y="Functions/Methods")

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.markdown("**Test Suite Presence**")
            test_counts = df["test_suite_present"].value_counts().reset_index()
            test_counts.columns = ["Has Test Suite", "Count"]
            test_counts["Has Test Suite"] = test_counts["Has Test Suite"].map(
                {True: "Present", False: "Absent"}
            )
            st.bar_chart(test_counts.set_index("Has Test Suite"))

        with chart_col4:
            st.markdown("**License Distribution**")
            license_counts = df["license"].value_counts().reset_index()
            license_counts.columns = ["License", "Count"]
            st.bar_chart(license_counts.set_index("License"))

        with st.expander("📋 Per-Repository Detail"):
            display_df = df[[
                "repo", "stars", "contributors",
                "functions", "classes",
                "test_suite_present", "test_confidence", "license",
            ]].copy()
            display_df.columns = [
                "Repo", "Stars", "Contributors",
                "Functions/Methods", "Classes",
                "Has Tests", "Test Confidence", "License",
            ]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Dependency Graph ──────────────────────────────────────────────────
        all_repo_meta = s.get("repo_metadata", [])
        if all_repo_meta:
            show_deps = st.toggle(
                "Show Dependency Graph",
                value=False,
                help=(
                    "Parse and display direct package dependencies for each "
                    "repository. Supports requirements.txt, package.json, "
                    "pom.xml, conanfile.txt, vcpkg.json, and CMakeLists.txt."
                ),
            )
            if show_deps:
                st.markdown("#### 📦 Dependency Graph")
                repo_names = [m["full_name"] for m in all_repo_meta]
                selected   = st.selectbox(
                    "Select repository",
                    options=repo_names,
                    key="dep_repo_select",
                )
                selected_meta = next(
                    m for m in all_repo_meta if m["full_name"] == selected
                )
                deps = selected_meta.get("dependencies", [])
                label = (f"📦 {selected} — {len(deps)} direct dependencies"
                         if deps else f"📦 {selected} — no manifest found")
                with st.expander(label, expanded=True):
                    if deps:
                        from sieve.ui.dep_viz import render_dep_graph
                        st.components.v1.html(
                            render_dep_graph(deps, selected, height=420),
                            height=420,
                            scrolling=False,
                        )
                        dep_df = pd.DataFrame(deps)[["name", "version", "kind"]]
                        dep_df["version"] = dep_df["version"].fillna("latest")
                        dep_df.columns = ["Package", "Version", "Kind"]
                        st.dataframe(dep_df, use_container_width=True, hide_index=True)
                    else:
                        st.info(
                            "No dependency manifest found for this repository. "
                            "SIEVE looks for: `requirements*.txt`, `pyproject.toml`, `setup.cfg` (Python); "
                            "`package.json` (JavaScript); `pom.xml` (Java); "
                            "`conanfile.txt`, `vcpkg.json`, `CMakeLists.txt` (C++)."
                        )

    st.divider()
    st.subheader("🎲 Random Sample Viewer")
    st.markdown("Randomly sample one record from the extracted corpus for manual inspection.")

    available = {}
    if "functions_jsonl" in output_paths and Path(output_paths["functions_jsonl"]).exists():
        available["Function/Method"] = output_paths["functions_jsonl"]
    if "classes_jsonl" in output_paths and Path(output_paths["classes_jsonl"]).exists():
        available["Class"] = output_paths["classes_jsonl"]

    if available:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

        with ctrl_col1:
            record_type = st.selectbox("Record type", options=list(available.keys()))
        with ctrl_col2:
            include_imports = st.toggle(
                "Include used imports", value=False,
                help="Prepend import statements above the extracted code.",
            )
        with ctrl_col3:
            sample_button = st.button("🎲 Draw random sample", use_container_width=True)

        if sample_button:
            try:
                lines  = Path(available[record_type]).read_text(encoding="utf-8").splitlines()
                lines  = [l for l in lines if l.strip()]
                row_id = random.randint(0, len(lines) - 1)
                record = json.loads(lines[row_id])
                st.session_state.sample = (record_type, record, row_id, len(lines))
            except Exception as e:
                st.error(f"Could not sample: {e}")

        if st.session_state.sample:
            rtype, record, row_id, total_rows = st.session_state.sample

            # Resolve syntax highlighting from record language
            rec_lang    = record.get("language", "Python")
            syntax_lang = _LANG_HIGHLIGHT.get(rec_lang, "python")

            # Build GitHub permalink if commit_sha is available
            commit_sha = record.get("commit_sha")
            file_path  = record.get("file_path", "")
            start_line = record.get("start_line")
            end_line   = record.get("end_line")
            repo_name  = record.get("repo", "")

            if commit_sha and file_path and start_line:
                gh_url = (
                    f"https://github.com/{repo_name}/blob/{commit_sha}/{file_path}"
                    f"#L{start_line}-L{end_line}"
                )
                gh_link = f"[View on GitHub ↗]({gh_url})"
            else:
                gh_link = f"[{repo_name}](https://github.com/{repo_name})"

            st.markdown(
                f"**Sampled {rtype}** — {gh_link} &nbsp;·&nbsp; "
                f"`Record #{row_id + 1} of {total_rows}`"
            )

            info_col, code_col = st.columns([1, 2])

            with info_col:
                st.markdown("**Metadata**")
                st.markdown(f"- **File:** `{record.get('file_path', '—')}`")
                st.markdown(f"- **Language:** {rec_lang}")

                line_label = (
                    f"{record.get('start_line')} – {record.get('end_line')}"
                )

                if rtype == "Function/Method":
                    st.markdown(f"- **Is Method:** {record.get('is_method', False)}")
                    if record.get("parent_class"):
                        st.markdown(f"- **Parent Class:** `{record.get('parent_class')}`")
                    params = record.get("parameters") or []
                    if isinstance(params, str):
                        params = json.loads(params)
                    st.markdown(f"- **Parameters:** `{', '.join(params) if params else 'none'}`")
                    if record.get("return_annotation"):
                        st.markdown(f"- **Returns:** `{record.get('return_annotation')}`")
                    st.markdown(f"- **Lines:** {line_label}")

                elif rtype == "Class":
                    st.markdown(f"- **Class:** `{record.get('class_name', '—')}`")
                    parents = record.get("parent_classes") or []
                    if isinstance(parents, str):
                        parents = json.loads(parents)
                    st.markdown(f"- **Inherits:** `{', '.join(parents) if parents else 'none'}`")
                    st.markdown(f"- **Methods:** {record.get('method_count', 0)}")
                    st.markdown(f"- **Has Constructor:** {record.get('has_constructor', False)}")
                    st.markdown(f"- **Lines:** {line_label}")

                llm = record.get("llm_score")
                st.markdown(
                    f"- **LLM Score:** "
                    f"{f'{llm:.3f}' if llm is not None else '`not scored`'}"
                )

                used_imports = record.get("used_imports") or []
                if isinstance(used_imports, str):
                    used_imports = json.loads(used_imports)
                if used_imports:
                    with st.expander(f"📦 Used imports ({len(used_imports)})"):
                        st.code("\n".join(used_imports), language=syntax_lang)
                else:
                    st.markdown("- **Used imports:** none detected")

            # ── Code Metrics Table ────────────────────────────────────────────
            # Collect all metrics — compute on-the-fly if not in record
            def _get_metrics(rec, lang):
                m = {
                    "loc":                   rec.get("loc"),
                    "sloc":                  rec.get("sloc"),
                    "lloc":                  rec.get("lloc"),
                    "comments":              rec.get("comments"),
                    "blank":                 rec.get("blank"),
                    "comment_ratio":         rec.get("comment_ratio"),
                    "cyclomatic_complexity": rec.get("cyclomatic_complexity"),
                    "max_nesting_depth":     rec.get("max_nesting_depth"),
                    "h1":                    rec.get("h1"),
                    "h2":                    rec.get("h2"),
                    "N1":                    rec.get("N1"),
                    "N2":                    rec.get("N2"),
                    "vocabulary":            rec.get("vocabulary"),
                    "halstead_length":       rec.get("halstead_length"),
                    "calculated_length":     rec.get("calculated_length"),
                    "volume":                rec.get("volume"),
                    "difficulty":            rec.get("difficulty"),
                    "effort":                rec.get("effort"),
                    "time":                  rec.get("time"),
                    "bugs":                  rec.get("bugs"),
                    "maintainability_index": rec.get("maintainability_index"),
                }
                if all(v is None for v in m.values()):
                    from sieve.core.metrics import compute_metrics
                    m = compute_metrics(rec.get("source_code", ""), lang)
                return m

            m = _get_metrics(record, rec_lang)

            metric_rows = [
                ("LOC",                     m.get("loc"),                   "Total lines of code"),
                ("SLOC",                    m.get("sloc"),                  "Source lines (non-blank, non-comment)"),
                ("LLOC",                    m.get("lloc"),                  "Logical lines of code (statements)"),
                ("Comments",                m.get("comments"),              "Comment lines"),
                ("Blank",                   m.get("blank"),                 "Blank lines"),
                ("Comment Ratio",           m.get("comment_ratio"),         "Comment lines / LOC"),
                ("Cyclomatic Complexity",   m.get("cyclomatic_complexity"), "McCabe: 1 + branching statements"),
                ("Max Nesting Depth",       m.get("max_nesting_depth"),     "Maximum control flow nesting depth"),
                ("h1 (Distinct Operators)", m.get("h1"),                    "Halstead: distinct operators"),
                ("h2 (Distinct Operands)",  m.get("h2"),                    "Halstead: distinct operands"),
                ("N1 (Total Operators)",    m.get("N1"),                    "Halstead: total operator occurrences"),
                ("N2 (Total Operands)",     m.get("N2"),                    "Halstead: total operand occurrences"),
                ("Vocabulary",              m.get("vocabulary"),            "Halstead: h1 + h2"),
                ("Halstead Length",         m.get("halstead_length"),       "Halstead: N1 + N2"),
                ("Calculated Length",       m.get("calculated_length"),     "Halstead: estimated length"),
                ("Volume",                  m.get("volume"),                "Halstead: information content (bits)"),
                ("Difficulty",              m.get("difficulty"),            "Halstead: difficulty to understand"),
                ("Effort",                  m.get("effort"),                "Halstead: mental effort required"),
                ("Time (s)",                m.get("time"),                  "Halstead: estimated programming time"),
                ("Bugs",                    m.get("bugs"),                  "Halstead: estimated number of bugs"),
                ("Maintainability Index",   m.get("maintainability_index"), "Composite score 0–100 (higher = more maintainable)"),
            ]
            metric_rows = [(label, val, desc) for label, val, desc in metric_rows if val is not None]

            if metric_rows:
                st.markdown("#### Code Structure Metrics")
                metrics_df = pd.DataFrame(metric_rows, columns=["Metric", "Value", "Description"])
                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Metric":      st.column_config.TextColumn(width="medium"),
                        "Value":       st.column_config.NumberColumn(width="small"),
                        "Description": st.column_config.TextColumn(width="large"),
                    }
                )

            with code_col:
                def _with_imports(code: str) -> str:
                    if include_imports and used_imports:
                        return "\n".join(used_imports) + "\n\n" + code
                    return code

                if rtype == "Function/Method":
                    tab1, tab2, tab3 = st.tabs(["Full Source", "Signature", "AST"])
                    with tab1:
                        st.code(_with_imports(record.get("source_code", "")), language=syntax_lang)
                    with tab2:
                        st.code(_with_imports(record.get("signature", "")), language=syntax_lang)
                    with tab3:
                        from sieve.ui.ast_viz import build_ast_json, render_ast_component
                        ast_json = build_ast_json(record.get("source_code", ""), rec_lang)
                        if ast_json:
                            st.components.v1.html(
                                render_ast_component(ast_json, height=520),
                                height=520,
                                scrolling=False,
                            )
                        else:
                            st.warning("AST could not be generated for this snippet.")

                elif rtype == "Class":
                    tab1, tab2, tab3 = st.tabs(["Full Source", "Skeleton", "AST"])
                    with tab1:
                        st.code(_with_imports(record.get("source_code", "")), language=syntax_lang)
                    with tab2:
                        st.code(_with_imports(record.get("skeleton", "")), language=syntax_lang)
                    with tab3:
                        from sieve.ui.ast_viz import build_ast_json, render_ast_component
                        ast_json = build_ast_json(record.get("source_code", ""), rec_lang)
                        if ast_json:
                            st.components.v1.html(
                                render_ast_component(ast_json, height=520),
                                height=520,
                                scrolling=False,
                            )
                        else:
                            st.warning("AST could not be generated for this snippet.")
    else:
        st.info("No JSONL output files found. Run the pipeline or load the example dataset.")


# ─── Welcome screen ───────────────────────────────────────────────────────────

else:
    st.markdown("""
    **How to use SIEVE:**
    1. Set your parameters in the sidebar
    2. Provide a GitHub Personal Access Token
    3. Click **▶ Run SIEVE** to start collection
    4. Browse results, download your corpus, and inspect random samples

    — or —

    Click **📦 Load Example Dataset** to explore a pre-built corpus instantly,
    with no token required.
    """)


# ─── Config Preview ───────────────────────────────────────────────────────────

with st.expander("📋 Current Configuration (JSON)"):
    try:
        config_preview = {
            "language":         language,
            "start_date":           str(start_date),
            "end_date":             str(end_date),
            "min_stars":        int(min_stars),
            "min_contributors": int(min_contributors),
            "max_repos":        int(max_repos)      if max_repos      > 0 else None,
            "max_functions":    int(max_functions) if max_functions > 0 else None,
            "max_classes":      int(max_classes)   if max_classes   > 0 else None,
            "granularity":      [g for g, v in [("function", max_functions), ("class", max_classes)] if int(v) != -1],
            "engineered_only":  engineered_only,
            "annotate_llm_score": annotate_llm_score,
            "deduplicate":      deduplicate_flag,
            "dedup_threshold":  dedup_threshold,
            "export_format":    export_format,
        }
        st.json(config_preview)
    except Exception:
        st.text("Set parameters in the sidebar to preview configuration.")