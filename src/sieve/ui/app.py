"""
ui/app.py

Streamlit web interface for SIEVE.
The UI is a thin layer over the pipeline — it collects parameters,
triggers the pipeline as a background process, and polls for progress.

Run with: streamlit run sieve/ui/app.py
"""

import json
import threading
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from sieve.config import SIEVEConfig, Language, Granularity, ExportFormat
from sieve.pipeline import run_pipeline
from sieve.core.quality import check_cloc, CLOC_INSTALL_INSTRUCTIONS


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SIEVE",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Session State Defaults ───────────────────────────────────────────────────

if "running" not in st.session_state:
    st.session_state.running = False
if "progress_log" not in st.session_state:
    st.session_state.progress_log = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "error" not in st.session_state:
    st.session_state.error = None


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

    cutoff_date = st.date_input(
        "Cutoff Date",
        value=date(2024, 1, 1),
        help="Only include repos with last commit after this date.",
    )

    st.subheader("Repository Filters")
    min_stars = st.number_input("Minimum Stars", min_value=0, value=50, step=10)
    min_contributors = st.number_input("Minimum Contributors", min_value=1, value=5, step=1)
    max_repos = st.number_input(
        "Max Repos (0 = no cap)", min_value=0, value=100, step=10
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
    deduplicate = st.toggle("Deduplicate", value=True)
    dedup_threshold = st.slider(
        "Dedup Similarity Threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
        disabled=not deduplicate,
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
        help="Recommended. Without it, rate limit is 10 requests/min."
    )

    run_button = st.button(
        "▶ Run SIEVE",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.running,
    )


# ─── Pipeline Runner ─────────────────────────────────────────────────────────

def _run_in_thread(config: SIEVEConfig):
    """Run pipeline in a background thread. Updates session state on completion."""
    def progress_callback(msg: str, current: int, total: int):
        st.session_state.progress_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        )

    try:
        summary = run_pipeline(config, progress_callback=progress_callback)
        st.session_state.summary = summary
    except Exception as e:
        st.session_state.error = str(e)
    finally:
        st.session_state.running = False


if run_button and not st.session_state.running:
    # Validate
    if not granularity:
        st.sidebar.error("Select at least one granularity level.")
    else:
        try:
            config = SIEVEConfig(
                language=language,
                cutoff_date=cutoff_date,
                min_stars=int(min_stars),
                min_contributors=int(min_contributors),
                max_repos=int(max_repos) if max_repos > 0 else None,
                granularity=granularity,
                require_tests=require_tests,
                engineered_only=engineered_only,
                annotate_llm_score=annotate_llm_score,
                deduplicate=deduplicate,
                dedup_threshold=dedup_threshold,
                output_dir=output_dir,
                export_format=export_format,
                github_token=github_token if github_token else None,
            )

            # Reset state
            st.session_state.running = True
            st.session_state.progress_log = []
            st.session_state.summary = None
            st.session_state.error = None

            thread = threading.Thread(target=_run_in_thread, args=(config,), daemon=True)
            thread.start()

        except Exception as e:
            st.sidebar.error(f"Configuration error: {e}")


# ─── Main Panel ───────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Progress")

    if st.session_state.running:
        st.info("⏳ Pipeline running... (refresh to see updates)")
        if st.button("🔄 Refresh"):
            st.rerun()

    if st.session_state.error:
        st.error(f"Pipeline failed: {st.session_state.error}")

    if st.session_state.progress_log:
        log_text = "\n".join(st.session_state.progress_log[-50:])  # Last 50 messages
        st.code(log_text, language=None)

    if not st.session_state.running and not st.session_state.progress_log:
        st.markdown("""
        **How to use SIEVE:**
        1. Set your parameters in the sidebar
        2. Provide a GitHub Personal Access Token (recommended)
        3. Click **▶ Run SIEVE** to start collection
        4. Results will be saved to your specified output directory

        Docs and source: [github.com/your-org/sieve](https://github.com)
        """)

with col2:
    st.subheader("Summary")

    if st.session_state.summary:
        s = st.session_state.summary
        st.metric("Repos Processed", s["total_repos_processed"])
        st.metric("Functions Extracted", s["total_functions"])
        st.metric("Classes Extracted", s["total_classes"])
        st.metric("Failed Repos", s["total_repos_failed"])

        if s["output_paths"]:
            st.subheader("Output Files")
            for label, path in s["output_paths"].items():
                st.code(path, language=None)

        if s["failed_repos"]:
            with st.expander(f"Failed Repos ({len(s['failed_repos'])})"):
                for r in s["failed_repos"]:
                    st.text(r)
    else:
        st.markdown("Results will appear here after the pipeline completes.")


# ─── Config Preview ───────────────────────────────────────────────────────────

with st.expander("📋 Current Configuration (JSON)"):
    try:
        config_preview = {
            "language": language,
            "cutoff_date": str(cutoff_date),
            "min_stars": int(min_stars),
            "min_contributors": int(min_contributors),
            "max_repos": int(max_repos) if max_repos > 0 else None,
            "granularity": granularity,
            "require_tests": require_tests,
            "deduplicate": deduplicate,
            "dedup_threshold": dedup_threshold,
            "output_dir": output_dir,
            "export_format": export_format,
        }
        st.json(config_preview)
    except Exception:
        st.text("Set parameters in the sidebar to preview configuration.")
