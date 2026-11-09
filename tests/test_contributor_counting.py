"""
tests/test_contributor_counting.py

Contributor counting in core/discovery.py: bots are not contributors, API
errors are no longer silently turned into "0 contributors", and a repository
whose contributor list is too large for the API is kept with an unknown count.

PyGithub is mocked; no network is used.
"""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests
from github import Github, GithubException
from github.GithubException import BadCredentialsException, RateLimitExceededException
from github.Repository import Repository

from sieve.config import SIEVEConfig
from sieve.core import discovery
from sieve.core.discovery import (
    DiscoveryReport,
    RepoMetadata,
    count_contributors,
    discover_repos,
    make_client,
)
from sieve.core.github_limits import GitHubQuotaError
from sieve.core.export import write_manifest
from sieve.core.quality import _check_hard_exclusions, collect_metrics
from sieve.pipeline import run_pipeline

TOKEN = "ghp_SECRET_TOKEN_1234567890"
START, END = date(2024, 1, 1), date(2024, 12, 31)

TOO_LARGE = GithubException(
    403,
    {"message": "The history or contributor list is too large to list contributors "
                "for this repository via the API."},
    {},
)


# ─── Fakes ────────────────────────────────────────────────────────────────────

def human(login):
    return SimpleNamespace(login=login, type="User")


def bot_by_suffix(login="dependabot[bot]"):
    return SimpleNamespace(login=login, type="User")      # type is irrelevant: the suffix decides


def bot_by_type(login="some-app"):
    return SimpleNamespace(login=login, type="Bot")       # no suffix: the type decides


class FakeContributors:
    """What repo.get_contributors() returns. totalCount counts everyone, bots included, like GitHub."""

    def __init__(self, users=(), error=None):
        self._users = list(users)
        self._error = error

    @property
    def totalCount(self):
        if self._error is not None:
            raise self._error
        return len(self._users)

    def __iter__(self):
        if self._error is not None:
            raise self._error
        return iter(self._users)


def contributors_of(*users):
    return FakeContributors(users)


def failing(exc):
    return FakeContributors(error=exc)


def make_repo(name, *outcomes, stars=100):
    """
    A mock repository. Each call to get_contributors() consumes the next
    outcome (the last one repeats): a FakeContributors, a list of users, or an
    exception to raise.
    """
    repo = MagicMock()
    repo.full_name = name
    repo.html_url = f"https://github.com/{name}"
    repo.stargazers_count = stars
    repo.default_branch = "main"
    repo.license = None
    repo.topics = []
    repo.pushed_at = None
    repo.get_commits.return_value = [MagicMock()]

    queue = list(outcomes)

    def get_contributors(*args, **kwargs):
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            return failing(outcome)
        if isinstance(outcome, FakeContributors):
            return outcome
        return FakeContributors(outcome)

    repo.get_contributors = MagicMock(side_effect=get_contributors)
    return repo


def server_error(status=502):
    return GithubException(status, {"message": "Server Error"}, {})


def rate_limit():
    return RateLimitExceededException(403, {"message": "API rate limit exceeded"}, {})


@pytest.fixture
def no_sleep():
    """
    Neither the retry backoff nor the pause between repos should slow the tests.
    Patches time.sleep process-wide, so the Streamlit test must not use it.
    """
    with patch("sieve.core.discovery.time.sleep") as sleep:
        yield sleep


def run(repos, report=None, **kwargs):
    kwargs.setdefault("min_contributors", 1)
    kwargs.setdefault("max_repos", 50)
    kwargs.setdefault("github_token", TOKEN)
    with patch("sieve.core.discovery.Github") as mock_gh:
        mock_gh.return_value.search_repositories.return_value = list(repos)
        return list(discover_repos("Python", START, END, report=report, **kwargs))


# ─── 1. Bots ──────────────────────────────────────────────────────────────────

@pytest.mark.usefixtures("no_sleep")
class TestBotsAreNotContributors:
    def test_login_suffix_bot_is_not_counted(self):
        repo = make_repo("o/r", [human("alice"), human("bob"), bot_by_suffix("dependabot[bot]")])
        assert count_contributors(repo) == 2

    @pytest.mark.parametrize("login", ["github-actions[bot]", "pre-commit-ci[bot]", "Renovate[BOT]"])
    def test_common_bots_are_not_counted(self, login):
        repo = make_repo("o/r", [human("alice"), bot_by_suffix(login)])
        assert count_contributors(repo) == 1

    def test_type_bot_is_not_counted_even_without_the_suffix(self):
        repo = make_repo("o/r", [human("alice"), human("bob"), bot_by_type("some-app")])
        assert count_contributors(repo) == 2

    def test_human_only_repo_is_unchanged(self):
        repo = make_repo("o/r", [human("a"), human("b"), human("c")])
        assert count_contributors(repo) == 3

    def test_a_human_whose_name_merely_contains_bot_is_counted(self):
        repo = make_repo("o/r", [human("robot-fan"), human("bot"), human("botond")])
        assert count_contributors(repo) == 3

    def test_repo_with_only_bots_has_zero(self):
        repo = make_repo("o/r", [bot_by_suffix(), bot_by_type()])
        assert count_contributors(repo) == 0

    def test_one_human_plus_a_bot_does_not_meet_min_contributors_two(self):
        repo = make_repo("o/r", [human("alice"), bot_by_suffix("github-actions[bot]")])
        assert run([repo], min_contributors=2) == []

    def test_two_humans_plus_a_bot_records_exactly_two(self):
        repo = make_repo("o/r", [human("a"), human("b"), bot_by_suffix()])
        (meta,) = run([repo], min_contributors=2)
        assert meta.contributors == 2

    def test_count_is_exact_not_a_threshold_test(self):
        # The recorded value goes into the manifest and the quality filter, so it must not stop early.
        repo = make_repo("o/r", [human(f"user{i}") for i in range(250)] + [bot_by_suffix()])
        (meta,) = run([repo], min_contributors=2)
        assert meta.contributors == 250


class TestPageSize:
    def _real_repo(self, pages):
        """A real PyGithub Repository whose HTTP layer is replaced by scripted pages."""
        client = make_client(TOKEN)
        calls = []

        def request(verb, url, parameters=None, headers=None, input=None, **kwargs):
            calls.append({"url": url, "parameters": dict(parameters or {})})
            return pages[len(calls) - 1]

        client.requester.requestJsonAndCheck = request
        repo = Repository(
            client.requester, {}, {"url": "https://api.github.com/repos/o/r", "full_name": "o/r"}, True
        )
        return repo, calls

    def test_contributors_are_requested_100_per_page(self):
        users = [{"login": f"u{i}", "type": "User"} for i in range(3)]
        repo, calls = self._real_repo([({}, users)])
        assert count_contributors(repo) == 3
        assert len(calls) == 1
        assert calls[0]["url"].endswith("/repos/o/r/contributors")
        assert calls[0]["parameters"]["per_page"] == 100

    def test_pages_are_followed_and_bots_on_later_pages_are_skipped(self):
        link = '<https://api.github.com/repos/o/r/contributors?page=2&per_page=100>; rel="next"'
        page1 = ({"link": link}, [{"login": "a", "type": "User"}, {"login": "dependabot[bot]", "type": "Bot"}])
        page2 = ({}, [{"login": "b", "type": "User"}, {"login": "x-app", "type": "Bot"}])
        repo, calls = self._real_repo([page1, page2])
        assert count_contributors(repo) == 2
        assert len(calls) == 2


# ─── 2. API errors ────────────────────────────────────────────────────────────

@pytest.mark.usefixtures("no_sleep")
class TestContributorCountErrors:
    def test_too_large_403_returns_none(self):
        repo = make_repo("o/r", TOO_LARGE)
        assert count_contributors(repo) is None

    def test_transient_5xx_is_retried_then_succeeds(self, no_sleep):
        repo = make_repo("o/r", server_error(502), server_error(503), [human("a"), human("b")])
        assert count_contributors(repo) == 2
        assert repo.get_contributors.call_count == 3
        assert no_sleep.call_count == 2          # a short backoff before each retry

    def test_connection_error_is_retried_then_succeeds(self):
        repo = make_repo("o/r", requests.exceptions.ConnectionError("reset"), [human("a")])
        assert count_contributors(repo) == 1
        assert repo.get_contributors.call_count == 2

    def test_persistent_error_gives_up_after_three_retries(self, no_sleep):
        repo = make_repo("o/r", server_error(500))
        with pytest.raises(discovery.ContributorCountError) as info:
            count_contributors(repo)
        assert repo.get_contributors.call_count == 4     # first try + 3 retries
        assert no_sleep.call_count == 3
        assert "500" in info.value.reason

    def test_non_transient_error_is_not_retried(self):
        repo = make_repo("o/r", GithubException(404, {"message": "Not Found"}, {}))
        with pytest.raises(discovery.ContributorCountError):
            count_contributors(repo)
        assert repo.get_contributors.call_count == 1

    def test_rate_limit_is_raised_and_not_retried(self):
        repo = make_repo("o/r", rate_limit())
        with pytest.raises(RateLimitExceededException):
            count_contributors(repo)
        assert repo.get_contributors.call_count == 1

    def test_bad_credentials_is_raised_and_not_retried(self):
        repo = make_repo("o/r", BadCredentialsException(401, {"message": "Bad credentials"}, {}))
        with pytest.raises(BadCredentialsException):
            count_contributors(repo)
        assert repo.get_contributors.call_count == 1

    @pytest.mark.parametrize("error", [
        GithubException(429, {"message": "Too Many Requests"}, {}),
        GithubException(403, {"message": "You have exceeded a secondary rate limit."}, {}),
    ], ids=["http-429", "secondary-403"])
    def test_other_rate_limit_shapes_become_quota_errors(self, error):
        repo = make_repo("o/r", error)
        with pytest.raises(GitHubQuotaError):
            count_contributors(repo)
        assert repo.get_contributors.call_count == 1

    def test_reason_never_contains_the_token(self):
        repo = make_repo("o/r", GithubException(500, {"message": f"boom {TOKEN}"}, {}))
        with pytest.raises(discovery.ContributorCountError) as info:
            count_contributors(repo, TOKEN)
        assert TOKEN not in info.value.reason


@pytest.mark.usefixtures("no_sleep")
class TestDiscoveryWithErrors:
    def test_too_large_repo_is_kept_with_unknown_count(self):
        report = DiscoveryReport()
        big = make_repo("o/big", TOO_LARGE)
        small = make_repo("o/small", [human("a"), human("b")])
        found = run([big, small], report=report, min_contributors=2)
        assert [(m.full_name, m.contributors) for m in found] == [("o/big", None), ("o/small", 2)]
        assert report.contributors_unknown == ["o/big"]
        assert report.excluded == []

    def test_transient_error_then_success_is_not_excluded(self):
        report = DiscoveryReport()
        repo = make_repo("o/r", server_error(503), [human("a"), human("b")])
        (meta,) = run([repo], report=report, min_contributors=2)
        assert meta.contributors == 2
        assert report.excluded == [] and report.contributors_unknown == []

    def test_persistent_error_excludes_the_repo_and_records_why(self):
        report = DiscoveryReport()
        broken = make_repo("o/broken", server_error(500))
        fine = make_repo("o/fine", [human("a")])
        found = run([broken, fine], report=report)
        assert [m.full_name for m in found] == ["o/fine"]
        assert len(report.excluded) == 1
        entry = report.excluded[0]
        assert entry["repo"] == "o/broken"
        assert "500" in entry["reason"]
        assert set(entry) == {"repo", "reason"}
        assert report.contributors_unknown == []

    def test_excluded_repo_marks_the_discovery_incomplete(self):
        report = DiscoveryReport()
        run([make_repo("o/broken", server_error(500))], report=report)
        assert report.complete is False

    def test_a_clean_run_reports_nothing_excluded(self):
        report = DiscoveryReport()
        run([make_repo("o/r", [human("a")])], report=report)
        assert report.excluded == [] and report.contributors_unknown == []
        assert report.complete is True

    def test_rate_limit_still_propagates_from_discovery(self):
        repos = [make_repo("o/a", [human("x")]), make_repo("o/b", rate_limit())]
        got = []
        with patch("sieve.core.discovery.Github") as mock_gh:
            mock_gh.return_value.search_repositories.return_value = repos
            with pytest.raises(RateLimitExceededException):
                for meta in discover_repos("Python", START, END, max_repos=10):
                    got.append(meta.full_name)
        assert got == ["o/a"]

    def test_rate_limit_becomes_github_quota_error_in_the_pipeline(self, tmp_path):
        config = SIEVEConfig(language="Python", start_date=START, end_date=END, max_repos=50,
                             output_dir=str(tmp_path / "out"), github_token=TOKEN)
        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.pipeline._fetch_quota", return_value=None):
            mock_gh.return_value.search_repositories.return_value = [make_repo("o/a", rate_limit())]
            with pytest.raises(GitHubQuotaError):
                run_pipeline(config)


# ─── 3. A repo with an unknown contributor count flows through every consumer ─

def _meta(name="o/big", contributors=None):
    return RepoMetadata(
        full_name=name, url=f"https://github.com/{name}", stars=100, contributors=contributors,
        last_commit_date="2024-01-01T00:00:00", default_branch="main", license_spdx="MIT",
        language="Python", collected_at="2024-06-01T00:00:00+00:00", topics=[],
    )


@pytest.mark.usefixtures("no_sleep")
class TestUnknownCountConsumers:
    def test_metadata_accepts_none(self):
        assert _meta(contributors=None).contributors is None

    def test_hard_exclusion_passes_an_unknown_count(self):
        # The list was too large to count, i.e. far above the minimum of 2.
        assert _check_hard_exclusions(None, release_count=3) is True
        assert _check_hard_exclusions(None, release_count=0) is False   # releases still apply

    def test_known_counts_are_still_checked(self):
        assert _check_hard_exclusions(1, release_count=3) is False
        assert _check_hard_exclusions(2, release_count=3) is True

    def test_collect_metrics_accepts_none(self):
        repo = MagicMock()
        repo.get_releases.return_value.totalCount = 3
        repo.get_pulls.return_value.totalCount = 4
        repo.get_issues.return_value.totalCount = 9
        with patch("sieve.core.quality.Github") as mock_gh:
            mock_gh.return_value.get_repo.return_value = repo
            metrics = collect_metrics("o/big", "MIT", None, repo_path=None)
        assert metrics.contributor_count is None
        assert metrics.passes_stage2 is True

    def test_manifest_stores_null(self, tmp_path):
        write_manifest({}, [{"full_name": "o/big", "contributors": None}], 0, 0, tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["repos"][0]["contributors"] is None

    def _pipeline(self, tmp_path, report, **overrides):
        config = SIEVEConfig(language="Python", start_date=START, end_date=END,
                             output_dir=str(tmp_path / "out"), github_token=TOKEN, **overrides)

        def fake_discover(**kwargs):
            kwargs["report"].excluded.extend(report.excluded)
            kwargs["report"].contributors_unknown.extend(report.contributors_unknown)
            yield _meta("o/big", None)

        def fake_clone(repo_full_name, target_dir):
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            (Path(target_dir) / "m.py").write_text("def f():\n    return 1\n")
            return True

        with patch("sieve.pipeline.discover_repos", side_effect=fake_discover), \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            return run_pipeline(config)

    def test_pipeline_summary_and_manifest_carry_none(self, tmp_path):
        report = DiscoveryReport(contributors_unknown=["o/big"])
        summary = self._pipeline(tmp_path, report)
        assert summary["total_repos_processed"] == 1
        assert summary["repo_stats"][0]["contributors"] is None
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["repos"][0]["contributors"] is None

    def test_engineered_filter_keeps_an_unknown_count_repo(self, tmp_path):
        repo = MagicMock()
        repo.get_releases.return_value.totalCount = 3
        repo.get_pulls.return_value.totalCount = 4
        repo.get_issues.return_value.totalCount = 9
        with patch("sieve.core.quality.Github") as mock_gh:
            mock_gh.return_value.get_repo.return_value = repo
            summary = self._pipeline(tmp_path, DiscoveryReport(), engineered_only=True)
        assert summary["total_repos_processed"] == 1


class TestUnknownCountInTheUI:
    def test_ui_renders_an_unknown_count(self):
        from streamlit.testing.v1 import AppTest

        home = str(Path(__file__).resolve().parents[1] / "src" / "sieve" / "ui" / "Home.py")
        at = AppTest.from_file(home, default_timeout=30)
        at.session_state["summary"] = {
            "total_repos_discovered": 2, "total_repos_processed": 2, "total_repos_failed": 0,
            "total_functions": 3, "total_classes": 0, "failed_repos": [],
            "output_paths": {}, "repo_metadata": [], "rate_limited": False,
            "repo_stats": [
                {"repo": "o/big", "stars": 10, "contributors": None, "functions": 2, "classes": 0,
                 "test_suite_present": True, "test_confidence": 0.9, "license": "MIT"},
                {"repo": "o/small", "stars": 20, "contributors": 5, "functions": 1, "classes": 0,
                 "test_suite_present": False, "test_confidence": 0.1, "license": "MIT"},
            ],
        }
        at.run()
        assert not at.exception
        shown = [df.value for df in at.dataframe if "Contributors" in df.value.columns]
        assert shown, "per-repository table not rendered"
        # Known counts must stay whole numbers next to the unknown one (not 5.0).
        assert str(shown[0]["Contributors"].tolist()[1]) == "5"


# ─── 4. The manifest's discovery block ────────────────────────────────────────

@pytest.mark.usefixtures("no_sleep")
class TestManifestDiscoveryBlock:
    def test_report_defaults(self):
        report = DiscoveryReport()
        assert report.excluded == [] and report.contributors_unknown == []

    def test_manifest_block_has_the_new_fields(self, tmp_path):
        report = DiscoveryReport()
        repos = [
            make_repo("o/big", TOO_LARGE),
            make_repo("o/broken", server_error(500)),
            make_repo("o/fine", [human("a"), human("b")]),
        ]
        config = SIEVEConfig(language="Python", start_date=START, end_date=END, max_repos=50,
                             output_dir=str(tmp_path / "out"), github_token=TOKEN)

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", return_value=False):
            mock_gh.return_value.search_repositories.return_value = repos
            run_pipeline(config)

        text = (tmp_path / "out" / "manifest.json").read_text()
        block = json.loads(text)["discovery"]
        assert block["contributors_unknown"] == ["o/big"]
        assert [e["repo"] for e in block["excluded"]] == ["o/broken"]
        assert "500" in block["excluded"][0]["reason"]
        assert set(block["excluded"][0]) == {"repo", "reason"}
        assert block["complete"] is False
        assert "SECRET_TOKEN" not in text

    def test_clean_run_block_has_empty_lists(self, tmp_path):
        config = SIEVEConfig(language="Python", start_date=START, end_date=END, max_repos=50,
                             output_dir=str(tmp_path / "out"), github_token=TOKEN)
        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", return_value=False):
            mock_gh.return_value.search_repositories.return_value = [make_repo("o/r", [human("a")])]
            run_pipeline(config)
        block = json.loads((tmp_path / "out" / "manifest.json").read_text())["discovery"]
        assert block["excluded"] == [] and block["contributors_unknown"] == []
        assert block["complete"] is True

