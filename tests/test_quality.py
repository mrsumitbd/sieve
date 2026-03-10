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