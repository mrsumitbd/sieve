"""
sieve/pipeline.py

Main orchestrator. Ties together discovery → acquisition → extraction
→ deduplication → export into a single callable pipeline.

Designed to be called from both the CLI and the Streamlit UI.
Progress is communicated via an optional callback so both interfaces
can display it in their own way.
"""

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from sieve.config import SIEVEConfig
from sieve.core.discovery import discover_repos, RepoMetadata
from sieve.core.detection import detect_test_suite
from sieve.core.extraction import extract_from_repo, FunctionRecord, ClassRecord
from sieve.core.deduplication import deduplicate
from sieve.core.export import export_dataset

logger = logging.getLogger(__name__)


def _clone_repo(repo_full_name: str, target_dir: str) -> bool:
    """Clone a GitHub repo. Returns True on success."""
    url = f"https://github.com/{repo_full_name}.git"
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, target_dir],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.warning(f"git clone failed for {repo_full_name}: {result.stderr.strip()}")
        return False
    return True


def run_pipeline(
    config: SIEVEConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """
    Execute the full SIEVE pipeline.

    Args:
        config: SIEVEConfig with all parameters set
        progress_callback: Optional callable(message, current, total) for UI progress updates

    Returns:
        dict with output paths and summary statistics
    """
    def _progress(msg: str, current: int = 0, total: int = 0):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, current, total)

    all_functions: list[FunctionRecord] = []
    all_classes: list[ClassRecord] = []
    repo_metadata_list: list[dict] = []
    failed_repos: list[str] = []

    _progress("Starting repo discovery...")

    repos = list(discover_repos(
        language=config.language,
        cutoff_date=config.cutoff_date,
        min_stars=config.min_stars,
        min_contributors=config.min_contributors,
        max_repos=config.max_repos,
        github_token=config.github_token,
    ))

    total = len(repos)
    _progress(f"Discovered {total} repositories. Beginning extraction.", 0, total)

    for idx, repo_meta in enumerate(repos):
        repo_name = repo_meta.full_name
        _progress(f"[{idx+1}/{total}] Processing {repo_name}", idx + 1, total)

        with tempfile.TemporaryDirectory() as tmpdir:
            clone_path = str(Path(tmpdir) / "repo")

            # --- Clone ---
            if not _clone_repo(repo_name, clone_path):
                failed_repos.append(repo_name)
                continue

            # --- Test suite detection ---
            test_report = detect_test_suite(clone_path, config.language)

            if config.require_tests and not test_report.is_present:
                _progress(f"  Skipping {repo_name}: no test suite detected")
                shutil.rmtree(clone_path, ignore_errors=True)
                continue

            # --- Extraction ---
            funcs, classes = extract_from_repo(
                repo_path=clone_path,
                language=config.language,
                repo_name=repo_name,
                granularities=config.granularity,
            )

            _progress(
                f"  {repo_name}: {len(funcs)} functions, {len(classes)} classes extracted",
                idx + 1, total,
            )

            all_functions.extend(funcs)
            all_classes.extend(classes)

            # Store repo metadata with test suite report attached
            meta_dict = asdict(repo_meta) if hasattr(repo_meta, "__dataclass_fields__") else vars(repo_meta)
            meta_dict["test_suite"] = test_report.to_dict()
            repo_metadata_list.append(meta_dict)

            # Cleanup handled by TemporaryDirectory context manager
            time.sleep(0.2)  # Be gentle with GitHub

    # --- Deduplication ---
    if config.deduplicate:
        _progress(f"Running deduplication (threshold={config.dedup_threshold})...")
        all_functions = deduplicate(all_functions, threshold=config.dedup_threshold)
        all_classes = deduplicate(all_classes, threshold=config.dedup_threshold)

    _progress(
        f"Extraction complete. {len(all_functions)} functions, {len(all_classes)} classes. Exporting..."
    )

    # --- Export ---
    output_paths = export_dataset(
        functions=all_functions,
        classes=all_classes,
        output_dir=config.output_dir,
        export_format=config.export_format,
        config=config.model_dump(),
        repo_metadata_list=repo_metadata_list,
    )

    summary = {
        "total_repos_discovered": total,
        "total_repos_processed": len(repo_metadata_list),
        "total_repos_failed": len(failed_repos),
        "total_functions": len(all_functions),
        "total_classes": len(all_classes),
        "failed_repos": failed_repos,
        "output_paths": output_paths,
    }

    _progress("Pipeline complete.", total, total)
    return summary
