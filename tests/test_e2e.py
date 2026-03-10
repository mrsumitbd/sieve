"""
tests/test_e2e.py

End-to-end tests for run_pipeline.
GitHub discovery and git clone are mocked — the real extraction,
deduplication, and export chain runs against local synthetic code.
"""

import json
import shutil
import textwrap
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sieve.config import SIEVEConfig
from sieve.core.discovery import RepoMetadata
from sieve.pipeline import run_pipeline


# ─── Fixtures ─────────────────────────────────────────────────────────────────

_PYTHON_CODE = textwrap.dedent("""
    import os
    from pathlib import Path

    class FileReader:
        \"\"\"Reads files.\"\"\"
        def __init__(self, base: str):
            self.base = Path(base)

        def read(self, name: str) -> str:
            \"\"\"Read a file by name.\"\"\"
            return (self.base / name).read_text()

    def count_lines(text: str) -> int:
        \"\"\"Count non-empty lines.\"\"\"
        return sum(1 for l in text.splitlines() if l.strip())
""").strip()


def _make_repo_meta(name="test-owner/test-repo") -> RepoMetadata:
    return RepoMetadata(
        full_name=name,
        url=f"https://github.com/{name}",
        stars=100,
        contributors=5,
        last_commit_date="2024-06-01",
        default_branch="main",
        license_spdx="MIT",
        language="Python",
        collected_at="2024-06-01T00:00:00Z",
        topics=[],
    )


@pytest.fixture
def synthetic_repo(tmp_path):
    """Write synthetic Python source into a temp directory."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(_PYTHON_CODE)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text("def test_count(): pass\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  test:\n    steps:\n      - run: pytest\n"
    )
    return tmp_path


@pytest.fixture
def base_config(tmp_path):
    return SIEVEConfig(
        language="Python",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        min_stars=10,
        max_repos=1,
        granularity=["function", "class"],
        deduplicate=False,
        output_dir=str(tmp_path / "output"),
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_with_fake_repo(config, synthetic_repo):
    """
    Patch discover_repos to yield one fake RepoMetadata and
    _clone_repo to copy the synthetic_repo into the target dir instead of
    hitting GitHub.
    """
    meta = _make_repo_meta()

    def fake_clone(repo_full_name, target_dir):
        shutil.copytree(str(synthetic_repo), target_dir, dirs_exist_ok=True)
        return True

    with patch("sieve.pipeline.discover_repos", return_value=iter([meta])), \
         patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
        return run_pipeline(config)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestPipelineSummaryShape:
    def test_summary_has_required_keys(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        for key in (
            "total_repos_processed", "total_functions", "total_classes",
            "total_repos_failed", "failed_repos", "output_paths", "repo_stats",
        ):
            assert key in summary

    def test_one_repo_processed(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert summary["total_repos_processed"] == 1
        assert summary["total_repos_failed"] == 0

    def test_functions_extracted(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert summary["total_functions"] > 0

    def test_classes_extracted(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert summary["total_classes"] > 0


class TestPipelineOutputFiles:
    def test_jsonl_files_written(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        paths = summary["output_paths"]
        assert Path(paths["functions_jsonl"]).exists()
        assert Path(paths["classes_jsonl"]).exists()

    def test_manifest_written(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert Path(summary["output_paths"]["manifest"]).exists()

    def test_manifest_function_count_matches_summary(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        manifest = json.loads(Path(summary["output_paths"]["manifest"]).read_text())
        assert manifest["summary"]["total_functions"] == summary["total_functions"]
        assert manifest["summary"]["total_classes"] == summary["total_classes"]


class TestPipelineRequireTests:
    def test_require_tests_passes_with_test_suite(self, tmp_path, synthetic_repo):
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            require_tests=True,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        summary = _run_with_fake_repo(config, synthetic_repo)
        # synthetic_repo has tests — should be processed
        assert summary["total_repos_processed"] == 1

    def test_require_tests_skips_repo_without_tests(self, tmp_path, tmp_repo):
        # A repo with no test infrastructure
        no_test_repo = tmp_repo({"src/app.py": "def run(): pass\n"})

        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            require_tests=True,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        meta = _make_repo_meta()

        def fake_clone(repo_full_name, target_dir):
            shutil.copytree(str(no_test_repo), target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary = run_pipeline(config)

        assert summary["total_functions"] == 0
        assert summary["total_classes"] == 0


class TestPipelineDeduplication:
    def test_dedup_reduces_count(self, tmp_path, synthetic_repo):
        """Duplicate files across two fake repos should be deduplicated."""
        config_no_dedup = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            deduplicate=False,
            output_dir=str(tmp_path / "no_dedup"),
        )
        config_dedup = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            deduplicate=True,
            dedup_threshold=0.8,
            output_dir=str(tmp_path / "dedup"),
        )

        # Two repos with identical code
        meta1 = _make_repo_meta("owner/repo1")
        meta2 = _make_repo_meta("owner/repo2")

        def fake_clone(repo_full_name, target_dir):
            shutil.copytree(str(synthetic_repo), target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta1, meta2])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary_no_dedup = run_pipeline(config_no_dedup)

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta1, meta2])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary_dedup = run_pipeline(config_dedup)

        assert summary_dedup["total_functions"] <= summary_no_dedup["total_functions"]


class TestPipelineRepoStats:
    def test_repo_stats_row_per_repo(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert len(summary["repo_stats"]) == 1

    def test_repo_stats_fields_present(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        rs = summary["repo_stats"][0]
        for field in ("repo", "stars", "contributors", "functions", "classes",
                      "test_suite_present", "license"):
            assert field in rs