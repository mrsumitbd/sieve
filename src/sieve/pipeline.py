"""
sieve/pipeline.py

Main orchestrator. Ties together:
  discovery → (optional: quality scoring) → acquisition
  → extraction → deduplication → export

When engineered_only=True, discovery runs in two phases:
  Phase 1: collect all candidate repos + their quality metrics
  Phase 2: compute population-level Q1 thresholds, filter, then clone

When engineered_only=False, repos are processed as they are discovered
(streaming, lower memory footprint).
"""

import logging
import math
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from sieve.config import SIEVEConfig
from sieve.core.discovery import discover_repos, RepoMetadata
from sieve.core.detection import detect_test_suite
from sieve.core.extraction import extract_from_repo, count_repo_contents, FunctionRecord, ClassRecord
from sieve.core.deduplication import deduplicate
from sieve.core.export import export_dataset
from sieve.core.quality import collect_metrics, apply_filters, RepoQualityMetrics

logger = logging.getLogger(__name__)


def _clone_repo(repo_full_name: str, target_dir: str) -> bool:
    """Shallow clone a GitHub repo. Returns True on success."""
    url = f"https://github.com/{repo_full_name}.git"
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, target_dir],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.warning(f"git clone failed for {repo_full_name}: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"git clone timed out for {repo_full_name} — repo too large, skipping")
        return False
    except Exception as e:
        logger.warning(f"git clone error for {repo_full_name}: {e}")
        return False


def _process_repo(
    repo_meta: RepoMetadata,
    config: SIEVEConfig,
    quality_metrics: Optional[RepoQualityMetrics],
    func_cap: Optional[int] = None,
    class_cap: Optional[int] = None,
) -> tuple[list[FunctionRecord], list[ClassRecord], Optional[dict]]:
    """
    Clone, detect, and extract a single repo.
    Returns (functions, classes, metadata_dict) or ([], [], None) on failure.
    """
    repo_name = repo_meta.full_name

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            clone_path = str(Path(tmpdir) / "repo")

            if not _clone_repo(repo_name, clone_path):
                return [], [], None

            # Test suite detection — used for metadata only
            test_report = detect_test_suite(clone_path, config.language)

            # Extraction with per-repo caps
            funcs, classes = extract_from_repo(
                repo_path=clone_path,
                language=config.language,
                repo_name=repo_name,
                granularities=config.granularity,
                include_ast=config.export_ast,
                func_cap=func_cap,
                class_cap=class_cap,
            )

            # Build metadata dict
            meta_dict = {
                "full_name": repo_meta.full_name,
                "url": repo_meta.url,
                "stars": repo_meta.stars,
                "contributors": repo_meta.contributors,
                "last_commit_date": repo_meta.last_commit_date,
                "default_branch": repo_meta.default_branch,
                "license_spdx": repo_meta.license_spdx,
                "language": repo_meta.language,
                "collected_at": repo_meta.collected_at,
                "topics": repo_meta.topics,
                "test_suite": test_report.to_dict(),
            }

            if quality_metrics:
                meta_dict["quality"] = {
                    "pull_request_count": quality_metrics.pull_request_count,
                    "issue_count": quality_metrics.issue_count,
                    "loc": quality_metrics.loc,
                    "comment_lines": quality_metrics.comment_lines,
                    "code_ratio": quality_metrics.code_ratio,
                    "release_count": quality_metrics.release_count,
                    "passes_engineered_filter": quality_metrics.passes_all,
                }

            return funcs, classes, meta_dict

    except Exception as e:
        logger.warning(f"Unexpected error processing {repo_name}: {e} — skipping")
        return [], [], None


def _stratified_sample(records: list, cap: int, key: str = "repo") -> list:
    """
    Sample ``cap`` records from ``records`` with proportional allocation
    across repos.  Remainder slots are awarded to repos with the largest
    fractional parts, ensuring the total is exactly ``cap``.

    Args:
        records: List of FunctionRecord or ClassRecord objects.
        cap:     Maximum number of records to keep.
        key:     Attribute name to group by (default ``"repo"``).

    Returns:
        Sampled list of length ``min(len(records), cap)``.
    """
    if len(records) <= cap:
        return records

    by_repo: dict[str, list] = defaultdict(list)
    for r in records:
        by_repo[getattr(r, key)].append(r)

    total = len(records)
    result: list = []
    remainders: dict[str, float] = {}

    for repo, items in by_repo.items():
        exact = cap * len(items) / total
        base = int(exact)
        remainders[repo] = exact - base
        result.extend(random.sample(items, min(base, len(items))))

    # Fill remainder slots from repos with the highest fractional parts
    shortfall = cap - len(result)
    if shortfall > 0:
        sorted_repos = sorted(remainders, key=lambda r: remainders[r], reverse=True)
        for repo in sorted_repos[:shortfall]:
            pool = [x for x in by_repo[repo] if x not in result]
            if pool:
                result.append(random.choice(pool))

    return result


def run_pipeline(
    config: SIEVEConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """
    Execute the full SIEVE pipeline.

    Args:
        config: SIEVEConfig with all parameters set
        progress_callback: Optional callable(message, current, total)

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

    # ── Phase 1: Discovery ────────────────────────────────────────────────────

    _progress("Starting repo discovery...")

    all_discovered: list[RepoMetadata] = list(discover_repos(
        language=config.language,
        start_date=config.start_date,
        end_date=config.end_date,
        min_stars=config.min_stars,
        min_contributors=config.min_contributors,
        max_repos=config.max_repos,
        github_token=config.github_token,
    ))

    total_discovered = len(all_discovered)
    _progress(f"Discovered {total_discovered} candidate repositories.")

    # ── Phase 2 (conditional): Engineered project filtering ───────────────────

    if config.engineered_only:
        _progress("Engineered project filter enabled — collecting quality metrics...")

        quality_map: dict[str, RepoQualityMetrics] = {}
        for idx, repo_meta in enumerate(all_discovered):
            _progress(
                f"  Quality metrics [{idx+1}/{total_discovered}]: {repo_meta.full_name}",
                idx + 1, total_discovered,
            )
            metrics = collect_metrics(
                repo_full_name=repo_meta.full_name,
                license_spdx=repo_meta.license_spdx,
                contributor_count=repo_meta.contributors,
                repo_path=None,   # No clone yet — LOC will be 0 at this stage
                github_token=config.github_token,
            )
            quality_map[repo_meta.full_name] = metrics
            time.sleep(0.3)

        filtered_metrics = apply_filters(list(quality_map.values()))
        passing_names = {m.full_name for m in filtered_metrics if m.passes_all}

        repos_to_process = [r for r in all_discovered if r.full_name in passing_names]
        _progress(
            f"Engineered filter: {total_discovered} → {len(repos_to_process)} repos passed."
        )
    else:
        repos_to_process = all_discovered
        quality_map = {}

    # ── Phase 3: Two-pass extraction ─────────────────────────────────────────
    # Pass 1: Count contents per repo (lightweight, no full extraction)
    # Pass 2: Extract with per-repo caps computed from stratified allocation

    total = len(repos_to_process)
    _progress(f"Beginning extraction on {total} repositories.", 0, total)

    # Pass 1 — Count
    repo_func_counts:  dict[str, int] = {}
    repo_class_counts: dict[str, int] = {}

    if config.max_functions is not None or config.max_classes is not None:
        _progress("Pass 1: Counting extractable records per repository...")
        for idx, repo_meta in enumerate(repos_to_process):
            repo_name = repo_meta.full_name
            with tempfile.TemporaryDirectory() as tmpdir:
                clone_path = str(Path(tmpdir) / "repo")
                if _clone_repo(repo_name, clone_path):
                    fc, cc = count_repo_contents(
                        clone_path, config.language, config.granularity
                    )
                    repo_func_counts[repo_name]  = fc
                    repo_class_counts[repo_name] = cc
            _progress(
                f"  [{idx+1}/{total}] {repo_name}: "
                f"{repo_func_counts.get(repo_name, 0)} funcs, "
                f"{repo_class_counts.get(repo_name, 0)} classes",
                idx + 1, total,
            )

        # Compute stratified per-repo caps
        def _stratified_caps(counts: dict[str, int], total_cap: int) -> dict[str, int]:
            total_available = sum(counts.values())
            if total_available == 0:
                return {r: 0 for r in counts}
            # Initial proportional allocation
            caps = {}
            for repo, cnt in counts.items():
                caps[repo] = max(1, round(cnt / total_available * total_cap))
            # Adjust to match exact total — distribute remainder to largest repos
            current_total = sum(caps.values())
            diff = current_total - total_cap
            # Sort by count descending for fair adjustment
            sorted_repos = sorted(counts.keys(), key=lambda r: -counts[r])
            i = 0
            while diff > 0:
                repo = sorted_repos[i % len(sorted_repos)]
                if caps[repo] > 0:
                    caps[repo] -= 1
                    diff -= 1
                i += 1
            while diff < 0:
                repo = sorted_repos[i % len(sorted_repos)]
                caps[repo] += 1
                diff += 1
                i += 1
            return caps

        func_caps  = (_stratified_caps(repo_func_counts,  config.max_functions)
                      if config.max_functions  is not None else {})
        class_caps = (_stratified_caps(repo_class_counts, config.max_classes)
                      if config.max_classes is not None else {})
    else:
        func_caps  = {}
        class_caps = {}

    # Pass 2 — Extract with caps
    _progress("Pass 2: Extracting code with per-repository caps...")
    for idx, repo_meta in enumerate(repos_to_process):
        repo_name = repo_meta.full_name
        _progress(f"[{idx+1}/{total}] Processing {repo_name}", idx + 1, total)

        qm = quality_map.get(repo_name)

        # Per-repo caps from stratified allocation
        fc = func_caps.get(repo_name)
        cc = class_caps.get(repo_name)

        funcs, classes, meta_dict = _process_repo(
            repo_meta, config, qm,
            func_cap=fc,
            class_cap=cc,
        )

        if meta_dict is None:
            failed_repos.append(repo_name)
            continue

        _progress(
            f"  {repo_name}: {len(funcs)} functions, {len(classes)} classes",
            idx + 1, total,
        )

        all_functions.extend(funcs)
        all_classes.extend(classes)
        repo_metadata_list.append(meta_dict)
        time.sleep(0.2)

    # ── Phase 4: Deduplication ────────────────────────────────────────────────

    if config.deduplicate:
        _progress(f"Deduplicating (threshold={config.dedup_threshold})...")
        all_functions = deduplicate(all_functions, threshold=config.dedup_threshold)
        all_classes   = deduplicate(all_classes,   threshold=config.dedup_threshold)

    # ── Phase 4b: Final corpus size caps (safety net for edge cases) ─────────
    # These caps are a safety net in case per-repo caps slightly overshoot
    # due to rounding. The main capping happens in Pass 2 above.

    if config.max_functions is not None and len(all_functions) > config.max_functions:
        _progress(
            f"Applying final function cap: {len(all_functions)} → {config.max_functions}"
        )
        all_functions = _stratified_sample(all_functions, config.max_functions)

    if config.max_classes is not None and len(all_classes) > config.max_classes:
        _progress(
            f"Applying final class cap: {len(all_classes)} → {config.max_classes}"
        )
        all_classes = _stratified_sample(all_classes, config.max_classes)

    # ── Phase 5: LLM score annotation ────────────────────────────────────────

    if config.annotate_llm_score:
        from sieve.models.classifier import LLMCodeClassifier

        model_dir = Path(__file__).parent / "models" / "artifacts"

        try:
            _progress("Loading LLM classifier (downloading from HuggingFace Hub if needed)...")
            clf = LLMCodeClassifier.load(
                model_dir=model_dir,
                hf_token=config.hf_token,
            )
            all_records   = all_functions + all_classes  # type: ignore
            total_records = len(all_records)
            _progress(f"Scoring {total_records} records with LLM classifier...")

            snippets = [r.source_code for r in all_records]
            scores   = clf.score_batch(snippets, batch_size=64)

            for record, score in zip(all_records, scores):
                record.llm_score = score

            scored = sum(1 for s in scores if s is not None)
            _progress(f"LLM scoring complete — {scored}/{total_records} records scored.")
        except Exception as e:
            _progress(f"LLM scoring failed — {e}. Continuing without scores.")

    # ── Phase 6: Export ───────────────────────────────────────────────────────

    _progress(
        f"Extraction complete — {len(all_functions)} functions, "
        f"{len(all_classes)} classes. Exporting..."
    )

    output_paths = export_dataset(
        functions=all_functions,
        classes=all_classes,
        output_dir=config.output_dir,
        export_format=config.export_format,
        config=config.model_dump(mode="json"),
        repo_metadata_list=repo_metadata_list,
    )

    summary = {
        "total_repos_discovered": total_discovered,
        "total_repos_after_quality_filter": len(repos_to_process),
        "total_repos_processed": len(repo_metadata_list),
        "total_repos_failed": len(failed_repos),
        "total_functions": len(all_functions),
        "total_classes": len(all_classes),
        "failed_repos": failed_repos,
        "output_paths": output_paths,
        "output_dir": config.output_dir,
        # Per-repo counts for charts
        "repo_stats": [
            {
                "repo": m["full_name"],
                "stars": m["stars"],
                "contributors": m["contributors"],
                "functions": sum(1 for f in all_functions if f.repo == m["full_name"]),
                "classes": sum(1 for c in all_classes if c.repo == m["full_name"]),
                "test_suite_present": m["test_suite"]["test_suite_present"],
                "test_confidence": m["test_suite"]["confidence"],
                "license": m.get("license_spdx") or "Unknown",
            }
            for m in repo_metadata_list
        ],
    }

    _progress("Pipeline complete.", total, total)
    return summary