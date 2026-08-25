"""
tests/test_quality.py

Unit tests for the engineered project quality filters.
No network access — tests pure functions and filesystem LOC counters only.
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from sieve.core.quality import (
    _check_license,
    _check_hard_exclusions,
    apply_filters,
    RepoQualityMetrics,
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _make_metrics(
    full_name="owner/repo",
    license_spdx="MIT",
    contributor_count=10,
    release_count=3,
    pull_request_count=50,
    issue_count=30,
    loc=1000,
    comment_lines=200,
    code_ratio=0.83,
    passes_stage1=True,
    passes_stage2=True,
) -> RepoQualityMetrics:
    return RepoQualityMetrics(
        full_name=full_name,
        license_spdx=license_spdx,
        contributor_count=contributor_count,
        release_count=release_count,
        pull_request_count=pull_request_count,
        issue_count=issue_count,
        loc=loc,
        comment_lines=comment_lines,
        code_ratio=code_ratio,
        passes_stage1=passes_stage1,
        passes_stage2=passes_stage2,
    )


# ─── Stage 1: License ─────────────────────────────────────────────────────────

class TestCheckLicense:
    @pytest.mark.parametrize("spdx", ["MIT", "Apache-2.0", "GPL-2.0", "BSD-2-Clause"])
    def test_software_licenses_pass(self, spdx):
        assert _check_license(spdx) is True

    @pytest.mark.parametrize("spdx", ["CC-BY-4.0", "CC0-1.0", "CC-BY-SA-4.0", "OFL-1.1"])
    def test_non_software_licenses_fail(self, spdx):
        assert _check_license(spdx) is False

    def test_none_license_fails(self):
        # EXCLUDE_NO_LICENSE is True by default
        assert _check_license(None) is False

    def test_noassertion_fails(self):
        assert _check_license("NOASSERTION") is False


# ─── Stage 2: Hard exclusions ─────────────────────────────────────────────────

class TestCheckHardExclusions:
    def test_passes_with_sufficient_contributors_and_releases(self):
        assert _check_hard_exclusions(5, 2) is True

    def test_fails_with_one_contributor(self):
        assert _check_hard_exclusions(1, 5) is False

    def test_fails_with_zero_releases(self):
        assert _check_hard_exclusions(10, 0) is False

    def test_exactly_two_contributors_passes(self):
        assert _check_hard_exclusions(2, 1) is True

    def test_exactly_one_release_passes(self):
        assert _check_hard_exclusions(3, 1) is True


# ─── Stage 3: Distributional filtering ───────────────────────────────────────

class TestApplyFilters:
    def _population(self, n=8):
        """Return a well-distributed population of n repos."""
        return [
            _make_metrics(
                full_name=f"owner/repo{i}",
                pull_request_count=50 + i * 10,
                issue_count=30 + i * 5,
                loc=1000 + i * 200,
                code_ratio=0.80 + i * 0.01,
            )
            for i in range(n)
        ]

    def test_high_quality_repos_pass(self):
        population = self._population(8)
        result = apply_filters(population)
        passed = [m for m in result if m.passes_all]
        # Upper half of the distribution should pass Q1 filters
        assert len(passed) > 0

    def test_q1_repos_excluded(self):
        population = self._population(8)
        result = apply_filters(population)
        # The bottom-PR repo (pull_request_count=50) should fail Q1
        bottom = next(m for m in result if m.full_name == "owner/repo0")
        assert bottom.passes_stage3 is False

    def test_no_qualifying_repos_returns_unchanged(self):
        """When all repos fail Stage 1+2, Stage 3 cannot be computed."""
        bad = [_make_metrics(passes_stage1=False, passes_stage2=False) for _ in range(4)]
        result = apply_filters(bad)
        # All should still have passes_stage3=False (not mutated to True)
        assert all(not m.passes_stage3 for m in result)

    def test_passes_all_property(self):
        m = _make_metrics()
        m.passes_stage3 = True
        assert m.passes_all is True

    def test_passes_all_false_if_any_stage_fails(self):
        m = _make_metrics(passes_stage1=False)
        m.passes_stage3 = True
        assert m.passes_all is False


# ─── check_cloc ───────────────────────────────────────────────────────────────

class TestCheckCloc:
    def test_returns_true_when_cloc_available(self):
        from unittest.mock import patch, MagicMock
        from sieve.core.quality import check_cloc

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.94\n"

        with patch("sieve.core.quality.subprocess.run", return_value=mock_result):
            assert check_cloc() is True

    def test_returns_false_when_cloc_not_found(self):
        from unittest.mock import patch
        from sieve.core.quality import check_cloc
        import subprocess

        with patch("sieve.core.quality.subprocess.run",
                   side_effect=FileNotFoundError("cloc not found")):
            assert check_cloc() is False

    def test_returns_false_when_cloc_times_out(self):
        from unittest.mock import patch
        from sieve.core.quality import check_cloc
        import subprocess

        with patch("sieve.core.quality.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("cloc", 5)):
            assert check_cloc() is False

    def test_returns_false_when_cloc_nonzero_exit(self):
        from unittest.mock import patch, MagicMock
        from sieve.core.quality import check_cloc

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("sieve.core.quality.subprocess.run", return_value=mock_result):
            assert check_cloc() is False


# ─── _count_loc_cloc ─────────────────────────────────────────────────────────

class TestCountLocCloc:
    def test_uses_cloc_when_available(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sieve.core.quality import _count_loc_cloc
        import json

        cloc_output = json.dumps({
            "SUM": {"code": 500, "comment": 100}
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = cloc_output

        with patch("sieve.core.quality.subprocess.run", return_value=mock_result):
            loc, comments = _count_loc_cloc(str(tmp_path))

        assert loc == 500
        assert comments == 100

    def test_falls_back_to_ast_when_cloc_fails(self, tmp_path):
        from unittest.mock import patch
        from sieve.core.quality import _count_loc_cloc

        # Write a small Python file
        (tmp_path / "test.py").write_text(
            "def foo():\n    # comment\n    return 1\n"
        )

        with patch("sieve.core.quality.subprocess.run",
                   side_effect=FileNotFoundError("cloc not found")):
            loc, comments = _count_loc_cloc(str(tmp_path))

        assert loc > 0  # AST fallback found something

    def test_returns_zeros_on_empty_dir(self, tmp_path):
        from unittest.mock import patch
        from sieve.core.quality import _count_loc_cloc

        with patch("sieve.core.quality.subprocess.run",
                   side_effect=FileNotFoundError):
            loc, comments = _count_loc_cloc(str(tmp_path))

        assert loc == 0
        assert comments == 0


# ─── collect_quality_metrics ──────────────────────────────────────────────────

class TestCollectQualityMetrics:
    def test_returns_metrics_with_mocked_github(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sieve.core.quality import collect_metrics

        mock_repo = MagicMock()
        mock_repo.get_releases.return_value.totalCount = 5
        mock_repo.get_pulls.return_value.totalCount = 20
        mock_repo.get_issues.return_value.totalCount = 30

        with patch("sieve.core.quality.Github") as mock_gh, \
             patch("sieve.core.quality._count_loc_cloc", return_value=(1000, 200)):
            mock_gh.return_value.get_repo.return_value = mock_repo
            metrics = collect_metrics(
                repo_full_name="owner/repo",
                contributor_count=10,
                license_spdx="MIT",
                repo_path=str(tmp_path),
            )

        assert metrics.full_name == "owner/repo"
        assert metrics.release_count == 5
        assert metrics.loc == 1000
        assert metrics.passes_stage1 is True
        assert metrics.passes_stage2 is True

    def test_handles_github_exception_gracefully(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from github import GithubException
        from sieve.core.quality import collect_metrics

        with patch("sieve.core.quality.Github") as mock_gh, \
             patch("sieve.core.quality._count_loc_cloc", return_value=(0, 0)):
            mock_gh.return_value.get_repo.side_effect = GithubException(404, "Not Found")
            metrics = collect_metrics(
                repo_full_name="owner/missing",
                contributor_count=5,
                license_spdx="MIT",
                repo_path=str(tmp_path),
            )

        assert metrics.release_count == 0
        assert metrics.pull_request_count == 0