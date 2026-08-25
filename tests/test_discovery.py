"""
tests/test_discovery.py

Unit tests for GitHub search query construction.
Network is not required — _build_query is a pure function.
"""

import re
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from sieve.core.discovery import _build_query, _get_contributor_count


class TestBuildQuery:
    def test_pushed_range_format(self):
        q = _build_query("Python", date(2024, 1, 1), date(2025, 3, 9), 50)
        m = re.search(r"pushed:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", q)
        assert m is not None
        assert m.group(1) == "2024-01-01"
        assert m.group(2) == "2025-03-09"

    def test_single_day_range(self):
        q = _build_query("Java", date(2024, 6, 15), date(2024, 6, 15), 10)
        m = re.search(r"pushed:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", q)
        assert m.group(1) == m.group(2) == "2024-06-15"

    def test_stars_filter(self):
        q = _build_query("Python", date(2024, 1, 1), date(2025, 1, 1), 100)
        assert "stars:>=100" in q

    def test_mandatory_filters_always_present(self):
        q = _build_query("Python", date(2024, 1, 1), date(2025, 1, 1), 10)
        assert "fork:false" in q
        assert "archived:false" in q

    @pytest.mark.parametrize("lang,expected", [
        ("Python", "language:python"),
        ("Java", "language:java"),
        ("JavaScript", "language:javascript"),
    ])
    def test_language_mapping(self, lang, expected):
        q = _build_query(lang, date(2024, 1, 1), date(2025, 1, 1), 10)
        assert expected in q


class TestGetContributorCount:
    def test_returns_count_on_success(self):
        mock_repo = MagicMock()
        mock_repo.get_contributors.return_value.totalCount = 42
        assert _get_contributor_count(mock_repo, "token") == 42

    def test_returns_zero_on_github_exception(self):
        from github import GithubException
        mock_repo = MagicMock()
        mock_repo.full_name = "owner/repo"
        mock_repo.get_contributors.side_effect = GithubException(403, "Forbidden")
        assert _get_contributor_count(mock_repo, "token") == 0

    def test_calls_get_contributors_with_anon_false(self):
        mock_repo = MagicMock()
        mock_repo.get_contributors.return_value.totalCount = 10
        _get_contributor_count(mock_repo, None)
        mock_repo.get_contributors.assert_called_once_with(anon=False)


class TestDiscoverRepos:
    """Tests for the discover_repos generator — GitHub API is fully mocked."""

    def _make_mock_repo(self, full_name="owner/repo", stars=100,
                         contributor_count=5, has_license=True):
        repo = MagicMock()
        repo.full_name = full_name
        repo.html_url = f"https://github.com/{full_name}"
        repo.stargazers_count = stars
        repo.default_branch = "main"
        repo.pushed_at = MagicMock()
        repo.pushed_at.isoformat.return_value = "2024-06-01T00:00:00"
        repo.get_commits.return_value = [MagicMock()]
        repo.get_commits.return_value[0].commit.committer.date.isoformat.return_value = "2024-06-01T00:00:00"
        repo.get_topics.return_value = []
        if has_license:
            repo.license = MagicMock()
            repo.license.spdx_id = "MIT"
        else:
            repo.license = None
        return repo

    def test_yields_repo_metadata(self):
        from sieve.core.discovery import discover_repos
        mock_repo = self._make_mock_repo()
        mock_results = MagicMock()
        mock_results.totalCount = 1
        mock_results.__iter__ = MagicMock(return_value=iter([mock_repo]))

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.core.discovery._get_contributor_count", return_value=10):
            mock_gh.return_value.search_repositories.return_value = mock_results
            results = list(discover_repos(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                min_stars=50,
                min_contributors=5,
            ))

        assert len(results) == 1
        assert results[0].full_name == "owner/repo"
        assert results[0].stars == 100

    def test_skips_repos_below_min_contributors(self):
        from sieve.core.discovery import discover_repos
        mock_repo = self._make_mock_repo()
        mock_results = MagicMock()
        mock_results.totalCount = 1
        mock_results.__iter__ = MagicMock(return_value=iter([mock_repo]))

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.core.discovery._get_contributor_count", return_value=2):
            mock_gh.return_value.search_repositories.return_value = mock_results
            results = list(discover_repos(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                min_stars=50,
                min_contributors=5,
            ))

        assert len(results) == 0

    def test_respects_max_repos_cap(self):
        from sieve.core.discovery import discover_repos
        repos = [self._make_mock_repo(f"owner/repo{i}") for i in range(5)]
        mock_results = MagicMock()
        mock_results.totalCount = 5
        mock_results.__iter__ = MagicMock(return_value=iter(repos))

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.core.discovery._get_contributor_count", return_value=10), \
             patch("sieve.core.discovery.time.sleep"):
            mock_gh.return_value.search_repositories.return_value = mock_results
            results = list(discover_repos(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                min_stars=50,
                min_contributors=5,
                max_repos=2,
            ))

        assert len(results) == 2

    def test_handles_missing_license(self):
        from sieve.core.discovery import discover_repos
        mock_repo = self._make_mock_repo(has_license=False)
        mock_results = MagicMock()
        mock_results.totalCount = 1
        mock_results.__iter__ = MagicMock(return_value=iter([mock_repo]))

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.core.discovery._get_contributor_count", return_value=10):
            mock_gh.return_value.search_repositories.return_value = mock_results
            results = list(discover_repos(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                min_stars=50,
                min_contributors=5,
            ))

        assert results[0].license_spdx is None