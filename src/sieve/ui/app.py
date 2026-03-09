"""
ui/app.py

Streamlit web interface for SIEVE.
Runs the pipeline synchronously in the main thread.
Progress is streamed live using st.status().

Run with: streamlit run src/sieve/ui/app.py
"""

import json
import random
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from sieve.config import SIEVEConfig, Language, Granularity, ExportFormat
from sieve.pipeline import run_pipeline
from sieve.core.quality import check_cloc, CLOC_INSTALL_INSTRUCTIONS

# ─── Demo dataset helpers ─────────────────────────────────────────────────────

_DEMO_DIR = Path(__file__).parent / "demo_data"


def load_demo_dataset() -> dict:
    """
    Load the bundled demo dataset and return a summary dict in the same
    shape as run_pipeline() — so the Results section renders unchanged.
    """
    fn_path = _DEMO_DIR / "functions.jsonl"
    cl_path = _DEMO_DIR / "classes.jsonl"
    manifest = json.loads((_DEMO_DIR / "manifest.json").read_text())

    return {
        "total_repos_discovered": 3,
        "total_repos_after_quality_filter": 3,
        "total_repos_processed": 3,
        "total_repos_failed": 0,
        "total_functions": manifest["summary"]["total_functions"],
        "total_classes": manifest["summary"]["total_classes"],
        "failed_repos": [],
        "output_paths": {
            "functions_jsonl": str(fn_path),
            "classes_jsonl": str(cl_path),
            "manifest": str(_DEMO_DIR / "manifest.json"),
        },
        "output_dir": str(_DEMO_DIR),
        "repo_stats": manifest["repo_stats"],
        "_is_demo": True,
    }


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SIEVE",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Session State Defaults ───────────────────────────────────────────────────

if "summary" not in st.session_state:
    st.session_state.summary = None
if "error" not in st.session_state:
    st.session_state.error = None
if "sample" not in st.session_state:
    st.session_state.sample = None


# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🔬 SIEVE")
st.caption("**S**oftware **I**ngestion & **E**xtraction for **V**erifiable **E**valuation")
st.markdown("Curate contamination-aware code corpora from GitHub for SE research.")
st.divider()

# ─── cloc check ──────────────────────────────────────────────────────────────

if not check_cloc():
    with st.expander("⚠️ cloc not found — click to see installation instructions", expanded=True):
        st.warning(
            "**cloc is not installed or not on your PATH.** "
            "cloc is highly recommended for accurate LOC and comment counting "
            "when using the Engineered Projects filter. Without it, SIEVE falls "
            "back to an AST-based counter which is accurate but slower on large repos."
        )
        st.code(CLOC_INSTALL_INSTRUCTIONS, language="bash")
        st.markdown(
            "After installing, restart the Streamlit server and refresh this page. "
            "You can still run SIEVE without cloc — the AST-based fallback will be used."
        )


# ─── Sidebar: Parameter Form ─────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Parameters")

    language = st.selectbox(
        "Programming Language",
        options=[l.value for l in Language],
        index=0,
    )

    start_date = st.date_input(
        "Start Date",
        value=date(2024, 1, 1),
        help="Only include repos with last commit on or after this date.",
    )
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        help="Only include repos with last commit on or before this date.",
    )
    if end_date < start_date:
        st.error("End Date must be on or after Start Date.")

    st.subheader("Repository Filters")
    min_stars = st.number_input("Minimum Stars", min_value=0, value=50, step=10)
    min_contributors = st.number_input("Minimum Contributors", min_value=1, value=5, step=1)
    max_repos = st.number_input(
        "Max Repos (0 = no cap)", min_value=0, value=100, step=10
    )
    max_functions = st.number_input(
        "Max Functions (0 = no cap)", min_value=0, value=0, step=500,
        help="Cap on total extracted functions after deduplication. A random sample is drawn if the corpus exceeds this."
    )
    max_classes = st.number_input(
        "Max Classes (0 = no cap)", min_value=0, value=0, step=100,
        help="Cap on total extracted classes after deduplication. A random sample is drawn if the corpus exceeds this."
    )

    st.subheader("Content Filters")
    granularity = st.multiselect(
        "Extraction Granularity",
        options=[g.value for g in Granularity],
        default=["function", "class"],
    )
    require_tests = st.toggle(
        "Require Test Suite", value=False,
        help="Only include repos where a test suite is detected."
    )
    engineered_only = st.toggle(
        "Engineered Projects Only", value=False,
        help=(
            "Apply Xiao et al. (2025) / Munaiah et al. (2017) engineered project filter. "
            "Excludes non-software licenses, repos without releases, and bottom-Q1 repos. "
            "Runs a two-pass discovery — slower but higher-quality corpus."
        )
    )
    annotate_llm_score = st.toggle(
        "Annotate LLM Score", value=False, disabled=True,
        help="P(LLM-generated) per record — classifier coming in a future release."
    )

    st.subheader("Processing")
    deduplicate_flag = st.toggle("Deduplicate", value=True)
    dedup_threshold = st.slider(
        "Dedup Similarity Threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
        disabled=not deduplicate_flag,
    )

    st.subheader("Output")
    output_dir = st.text_input("Output Directory", value="./sieve_output")
    export_format = st.selectbox(
        "Export Format",
        options=[f.value for f in ExportFormat],
        index=0,
    )

    st.subheader("GitHub")
    github_token = st.text_input(
        "GitHub Token (PAT)", type="password",
        help="Recommended. Without it, rate limit is 60 requests/hour."
    )

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
            "Load a pre-built demo corpus (54 functions, 8 classes across "
            "3 synthetic repositories). No GitHub token required."
        ),
    )


# ─── Pipeline Execution ───────────────────────────────────────────────────────

if run_button:
    if not granularity:
        st.error("Select at least one granularity level.")
    else:
        try:
            config = SIEVEConfig(
                language=language,
                start_date=start_date,
                end_date=end_date,
                min_stars=int(min_stars),
                min_contributors=int(min_contributors),
                max_repos=int(max_repos) if max_repos > 0 else None,
                max_functions=int(max_functions) if max_functions > 0 else None,
                max_classes=int(max_classes) if max_classes > 0 else None,
                granularity=granularity,
                require_tests=require_tests,
                engineered_only=engineered_only,
                annotate_llm_score=annotate_llm_score,
                deduplicate=deduplicate_flag,
                dedup_threshold=dedup_threshold,
                output_dir=output_dir,
                export_format=export_format,
                github_token=github_token if github_token else None,
            )
        except Exception as e:
            st.error(f"Configuration error: {e}")
            st.stop()

        st.session_state.summary = None
        st.session_state.error = None
        st.session_state.sample = None

        with st.status("Running SIEVE pipeline...", expanded=True) as status:
            def progress_callback(msg: str, current: int, total: int):
                ts = datetime.now().strftime("%H:%M:%S")
                st.write(f"`{ts}` {msg}")

            try:
                summary = run_pipeline(config, progress_callback=progress_callback)
                st.session_state.summary = summary
                status.update(label="✅ Pipeline complete!", state="complete", expanded=False)
            except Exception as e:
                st.session_state.error = str(e)
                status.update(label="❌ Pipeline failed", state="error", expanded=True)
                st.error(str(e))


# ─── Demo Dataset Load ────────────────────────────────────────────────────────

if demo_button:
    try:
        st.session_state.summary = load_demo_dataset()
        st.session_state.error = None
        st.session_state.sample = None
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
            "Showing pre-built corpus from 3 synthetic repositories "
            "(`sieve-demo/data-structures`, `sieve-demo/algorithms`, `sieve-demo/text-utils`). "
            "Click **▶ Run SIEVE** with a GitHub token to build a real corpus.",
            icon="ℹ️",
        )

    # ── Summary Metrics ───────────────────────────────────────────────────────

    st.subheader("📊 Run Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Repos Processed",  s["total_repos_processed"])
    c2.metric("Functions",        s["total_functions"])
    c3.metric("Classes",          s["total_classes"])
    c4.metric("Deduped Total",    s["total_functions"] + s["total_classes"])
    c5.metric("Failed Repos",     s["total_repos_failed"])

    if s.get("failed_repos"):
        with st.expander(f"Failed repos ({len(s['failed_repos'])})"):
            for r in s["failed_repos"]:
                st.text(r)

    # ── Charts ────────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("📈 Dataset Statistics")

    repo_stats = s.get("repo_stats", [])

    if repo_stats:

        df = pd.DataFrame(repo_stats)
        short_names = [r.split("/")[-1] for r in df["repo"]]

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Functions & Classes per Repository**")
            chart_df = pd.DataFrame({
                "Repo": short_names * 2,
                "Count": list(df["functions"]) + list(df["classes"]),
                "Type": ["Functions"] * len(df) + ["Classes"] * len(df),
            })
            st.bar_chart(
                chart_df.pivot(index="Repo", columns="Type", values="Count"),
                color=["#4e79a7", "#f28e2b"],
            )

        with chart_col2:
            st.markdown("**Stars vs. Extracted Functions**")
            scatter_df = pd.DataFrame({
                "Repo": short_names,
                "Stars": df["stars"],
                "Functions": df["functions"],
            })
            st.scatter_chart(scatter_df, x="Stars", y="Functions")

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

        # Repo detail table
        with st.expander("📋 Per-Repository Detail"):
            display_df = df[["repo", "stars", "contributors",
                              "functions", "classes",
                              "test_suite_present", "test_confidence", "license"]].copy()
            display_df.columns = ["Repo", "Stars", "Contributors",
                                   "Functions", "Classes",
                                   "Has Tests", "Test Confidence", "License"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Random Sample Viewer ──────────────────────────────────────────────────

    st.divider()
    st.subheader("🎲 Random Sample Viewer")
    st.markdown("Randomly sample one record from the extracted corpus for manual verification.")

    output_paths = s.get("output_paths", {})
    available = {}
    if "functions_jsonl" in output_paths and Path(output_paths["functions_jsonl"]).exists():
        available["Function"] = output_paths["functions_jsonl"]
    if "classes_jsonl" in output_paths and Path(output_paths["classes_jsonl"]).exists():
        available["Class"] = output_paths["classes_jsonl"]

    if available:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

        with ctrl_col1:
            record_type = st.selectbox("Record type", options=list(available.keys()))
        with ctrl_col2:
            include_imports = st.toggle(
                "Include used imports",
                value=False,
                help=(
                    "Prepend the import statements used by this program unit "
                    "above the extracted code. Import lines are not part of the "
                    "body — line numbers shown below refer to the body only."
                ),
            )
        with ctrl_col3:
            sample_button = st.button("🎲 Draw random sample", use_container_width=True)

        if sample_button:
            try:
                lines = Path(available[record_type]).read_text(encoding="utf-8").splitlines()
                lines = [l for l in lines if l.strip()]
                record = json.loads(random.choice(lines))
                st.session_state.sample = (record_type, record)
            except Exception as e:
                st.error(f"Could not sample: {e}")

        if st.session_state.sample:
            rtype, record = st.session_state.sample

            st.markdown(
                f"**Sampled {rtype}** — "
                f"[{record.get('repo')}](https://github.com/{record.get('repo')})"
            )

            info_col, code_col = st.columns([1, 2])

            with info_col:
                st.markdown("**Metadata**")
                st.markdown(f"- **File:** `{record.get('file_path', '—')}`")
                st.markdown(f"- **Language:** {record.get('language', '—')}")

                line_label = (
                    f"{record.get('start_line')} – {record.get('end_line')} "
                    f"*(body lines, excl. imports)*"
                )

                if rtype == "Function":
                    st.markdown(f"- **Function:** `{record.get('func_name', '—')}`")
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
                st.markdown(f"- **LLM Score:** {f'{llm:.3f}' if llm is not None else '`not yet scored`'}")

                # Used imports list
                used_imports = record.get("used_imports") or []
                if isinstance(used_imports, str):
                    used_imports = json.loads(used_imports)
                if used_imports:
                    with st.expander(f"📦 Used imports ({len(used_imports)})"):
                        st.code("\n".join(used_imports), language="python")
                else:
                    st.markdown("- **Used imports:** none detected")

            with code_col:
                # Helper: optionally prepend import block
                def _with_imports(code: str) -> str:
                    if include_imports and used_imports:
                        return "\n".join(used_imports) + "\n\n" + code
                    return code

                if rtype == "Function":
                    tab1, tab2 = st.tabs(["Full Source", "Signature"])
                    with tab1:
                        st.code(_with_imports(record.get("source_code", "")), language="python")
                    with tab2:
                        st.code(_with_imports(record.get("signature", "")), language="python")

                elif rtype == "Class":
                    tab1, tab2 = st.tabs(["Full Source", "Skeleton"])
                    with tab1:
                        st.code(_with_imports(record.get("source_code", "")), language="python")
                    with tab2:
                        st.code(_with_imports(record.get("skeleton", "")), language="python")
    else:
        st.info("No JSONL output files found. Run the pipeline first.")

    # ── Output File Paths ─────────────────────────────────────────────────────

    with st.expander("📁 Output Files"):
        for label, path in s.get("output_paths", {}).items():
            st.code(path, language=None)


# ─── Config Preview (always visible) ─────────────────────────────────────────

else:
    st.markdown("""
    **How to use SIEVE:**
    1. Set your parameters in the sidebar
    2. Provide a GitHub Personal Access Token
    3. Click **▶ Run SIEVE** to start collection
    4. Results, statistics, and a sample viewer will appear here

    — or —

    Click **📦 Load Example Dataset** to explore a pre-built corpus instantly,
    with no token required.
    """)

with st.expander("📋 Current Configuration (JSON)"):
    try:
        config_preview = {
            "language": language,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "min_stars": int(min_stars),
            "min_contributors": int(min_contributors),
            "max_repos": int(max_repos) if max_repos > 0 else None,
            "max_functions": int(max_functions) if max_functions > 0 else None,
            "max_classes": int(max_classes) if max_classes > 0 else None,
            "granularity": granularity,
            "require_tests": require_tests,
            "engineered_only": engineered_only,
            "deduplicate": deduplicate_flag,
            "dedup_threshold": dedup_threshold,
            "output_dir": output_dir,
            "export_format": export_format,
        }
        st.json(config_preview)
    except Exception:
        st.text("Set parameters in the sidebar to preview configuration.")