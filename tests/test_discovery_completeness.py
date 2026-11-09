"""
tests/test_discovery_completeness.py

Discovery must not silently return fewer repositories than the filters match.
GitHub search returns at most 1000 results per query and caps the totalCount
that PyGithub exposes, so discovery splits the creation-date range into
windows and records what it could and could not cover.

No network: a small in-memory fake stands in for the GitHub client. It models
the 1000-result cap, so a test only passes if discovery really windows.
"""

import json
import logging
import random
import re
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from github import GithubException
from github.GithubException import RateLimitExceededException

from sieve.config import Language, SIEVEConfig
from sieve.core.discovery import LANGUAGE_QUERY_MAP, discover_repos
from sieve.core.github_limits import GitHubQuotaError, build_exhausted_message
from sieve.pipeline import run_pipeline

START, END = date(2024, 1, 1), date(2024, 12, 31)
RESET = datetime(2030, 1, 1, 14, 32, tzinfo=timezone.utc)
SEARCH_CAP = 1000

_CREATED = re.compile(r"created:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})")
_STARS = re.compile(r"stars:>=(\d+)")


# ─── In-memory GitHub ─────────────────────────────────────────────────────────

class FakeRepo:
    def __init__(self, name, stars, created, client, contributors=10):
        self.full_name = name
        self.html_url = f"https://github.com/{name}"
        self.stargazers_count = stars
        self.created = created
        self.default_branch = "main"
        self.pushed_at = None
        self.license = None
        self.topics = ["from-search-payload"]
        self._client = client
        self._contributors = contributors

    def get_topics(self):
        self._client.topic_calls += 1
        return []

    def get_commits(self):
        return []

    def get_contributors(self, anon=False):
        return SimpleNamespace(totalCount=self._contributors)


def rate_limit_error(reset=RESET):
    headers = {"X-RateLimit-Reset": str(int(reset.timestamp()))} if reset else None
    return RateLimitExceededException(403, {"message": "API rate limit exceeded"}, headers)


class FakeResults:
    """Lazy search result. totalCount is capped like PyGithub's; reads are counted on the client."""

    def __init__(self, client, items, error=None):
        self._client = client
        self._items = items
        self._error = error

    @property
    def totalCount(self):
        self._client.total_count_reads += 1
        return len(self._items)

    def __iter__(self):
        if self._error is not None:
            raise self._error
        return iter(self._items)


class FakeGithub:
    """
    Stands in for github.Github. Search returns at most 1000 repos per query
    (sorted by stars, then name); the raw requester returns the true total_count.
    """

    def __init__(self, repos, incomplete_results=False,
                 search_error=None, search_error_at=None,
                 count_error=None, count_error_at=None):
        self.repos = repos
        self.incomplete_results = incomplete_results
        self.search_error, self.search_error_at = search_error, search_error_at
        self.count_error, self.count_error_at = count_error, count_error_at
        self.search_queries: list[str] = []
        self.count_queries: list[str] = []
        self.count_params: list[dict] = []
        self.topic_calls = 0
        self.total_count_reads = 0
        self.requester = SimpleNamespace(requestJsonAndCheck=self._raw)
        for r in repos:
            r._client = self

    def _matching(self, query):
        lo, hi = (date.fromisoformat(x) for x in _CREATED.search(query).groups())
        min_stars = int(_STARS.search(query).group(1))
        hits = [r for r in self.repos if lo <= r.created <= hi and r.stargazers_count >= min_stars]
        return sorted(hits, key=lambda r: (-r.stargazers_count, r.full_name))

    def search_repositories(self, query, sort=None, order=None, **_):
        self.search_queries.append(query)
        hits = self._matching(query)
        error = self.search_error if len(self.search_queries) == self.search_error_at else None
        return FakeResults(self, hits[:SEARCH_CAP], error)

    def _raw(self, verb, url, parameters=None, **_):
        assert (verb, url) == ("GET", "/search/repositories")
        self.count_queries.append(parameters["q"])
        self.count_params.append(parameters)
        if len(self.count_queries) == self.count_error_at:
            raise self.count_error
        data = {
            "total_count": len(self._matching(parameters["q"])),
            "incomplete_results": self.incomplete_results,
            "items": [],
        }
        return {}, data


def make_repos(n, *, seed=7, start=START, end=END):
    """n repos with unique names, tied star counts and creation dates spread over [start, end]."""
    rng = random.Random(seed)
    span = (end - start).days + 1
    return [
        FakeRepo(f"owner{i % 97}/repo{i}", rng.randrange(10, 5000),
                 start + timedelta(days=rng.randrange(span)), None)
        for i in range(n)
    ]


def top_by_stars(repos, k):
    return [r.full_name for r in sorted(repos, key=lambda r: (-r.stargazers_count, r.full_name))[:k]]


def run_discovery(fake, **kwargs):
    kwargs.setdefault("min_stars", 10)
    kwargs.setdefault("min_contributors", 1)
    with patch("sieve.core.discovery.Github", return_value=fake), \
         patch("sieve.core.discovery.time.sleep"):
        return list(discover_repos("Python", kwargs.pop("start", START), kwargs.pop("end", END), **kwargs))


def windows_searched(fake):
    return sorted(
        tuple(date.fromisoformat(x) for x in _CREATED.search(q).groups()) for q in fake.search_queries
    )


# ─── Bug 1: the 1000-result cap ───────────────────────────────────────────────

class TestDateWindowing:
    def test_15770_results_are_split_into_windows_that_tile_the_range(self):
        repos = make_repos(15_770)
        fake = FakeGithub(repos)

        found = run_discovery(fake)

        windows = windows_searched(fake)
        assert len(windows) > 1
        assert windows[0][0] == START
        assert windows[-1][1] == END
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
            assert next_start == prev_end + timedelta(days=1)   # no gap, no overlap
        for lo, hi in windows:
            assert len([r for r in repos if lo <= r.created <= hi]) <= SEARCH_CAP

        names = [m.full_name for m in found]
        assert len(names) == len(set(names)) == 15_770

    def test_no_window_is_counted_twice(self):
        fake = FakeGithub(make_repos(15_770))
        run_discovery(fake)
        assert len(fake.count_queries) > 1
        assert len(fake.count_queries) == len(set(fake.count_queries))

    def test_total_at_or_below_the_search_cap_is_one_window(self):
        fake = FakeGithub(make_repos(600))
        found = run_discovery(fake)
        assert windows_searched(fake) == [(START, END)]
        assert len(found) == 600


class TestMaxReposCap:
    @pytest.mark.parametrize("cap", [1, 50, 1000])
    def test_cap_up_to_1000_is_one_query_and_no_count_calls(self, cap):
        repos = make_repos(15_770)
        fake = FakeGithub(repos)

        found = run_discovery(fake, max_repos=cap)

        assert len(fake.search_queries) == 1
        assert f"created:{START}..{END}" in fake.search_queries[0]
        assert fake.count_queries == []
        assert [m.full_name for m in found] == top_by_stars(repos, cap)

    @pytest.mark.parametrize("cap", [1001, 1500, 4000])
    def test_cap_above_1000_returns_the_global_top_k_by_stars(self, cap):
        repos = make_repos(15_770)
        fake = FakeGithub(repos)

        found = run_discovery(fake, max_repos=cap)

        assert len(windows_searched(fake)) > 1
        assert [m.full_name for m in found] == top_by_stars(repos, cap)

    def test_ties_in_stars_are_broken_by_full_name(self):
        repos = [FakeRepo(f"o/r{i:04d}", 100, START + timedelta(days=i % 300), None) for i in range(2500)]
        found = run_discovery(FakeGithub(repos), max_repos=1200)
        assert [m.full_name for m in found] == sorted(r.full_name for r in repos)[:1200]

    def test_cap_above_1000_applies_to_repos_that_pass_the_contributor_filter(self):
        repos = make_repos(3_000)
        for r in repos[::2]:
            r._contributors = 0
        passing = [r for r in repos if r._contributors >= 1]

        found = run_discovery(FakeGithub(repos), max_repos=1100)

        assert [m.full_name for m in found] == top_by_stars(passing, 1100)

    def test_cap_of_zero_makes_no_requests(self):
        fake = FakeGithub(make_repos(50))
        assert run_discovery(fake, max_repos=0) == []
        assert fake.search_queries == [] and fake.count_queries == []


# ─── Bug 2: the true total ────────────────────────────────────────────────────

class TestTrueTotal:
    def test_total_is_read_from_the_raw_response_not_totalcount(self, caplog):
        fake = FakeGithub(make_repos(15_770))

        with caplog.at_level(logging.INFO, logger="sieve.core.discovery"):
            run_discovery(fake)

        assert fake.total_count_reads == 0
        assert "15770" in caplog.text
        assert fake.count_params[0] == {"q": fake.count_queries[0], "per_page": 1}

    def test_count_results_reads_total_count(self):
        from sieve.core.discovery import build_query, count_results

        fake = FakeGithub(make_repos(15_770))
        query = build_query("Python", START, END, 10)
        with patch("sieve.core.discovery.Github", return_value=fake):
            assert count_results(query, "ghp_token") == 15_770
        assert fake.count_params == [{"q": query, "per_page": 1}]

    def test_warns_when_more_than_1000_match(self, caplog):
        fake = FakeGithub(make_repos(2_500))
        with caplog.at_level(logging.WARNING, logger="sieve.core.discovery"):
            run_discovery(fake)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("2500" in w and "1000" in w for w in warnings)

    def test_no_warning_when_1000_or_fewer_match(self, caplog):
        fake = FakeGithub(make_repos(400))
        with caplog.at_level(logging.WARNING, logger="sieve.core.discovery"):
            run_discovery(fake)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_incomplete_results_is_recorded_and_warned(self, caplog):
        from sieve.core.discovery import DiscoveryReport

        report = DiscoveryReport()
        with caplog.at_level(logging.WARNING, logger="sieve.core.discovery"):
            run_discovery(FakeGithub(make_repos(300), incomplete_results=True), report=report)

        assert report.incomplete_results is True
        assert report.complete is False
        assert any("incomplete" in r.getMessage().lower() for r in caplog.records)


# ─── DiscoveryReport ──────────────────────────────────────────────────────────

class TestDiscoveryReport:
    def test_complete_windowed_run(self):
        from sieve.core.discovery import DiscoveryReport

        report = DiscoveryReport()
        found = run_discovery(FakeGithub(make_repos(15_770)), report=report)

        assert len(found) == 15_770
        assert report.complete is True
        assert report.total_matching == 15_770
        assert report.slices > 1
        assert report.truncated_windows == []
        assert report.incomplete_results is False

    def test_single_day_over_1000_is_recorded_as_truncated(self, caplog):
        from sieve.core.discovery import DiscoveryReport

        busy = date(2024, 6, 15)
        repos = make_repos(2_000) + [FakeRepo(f"busy/r{i}", 50 + i, busy, None) for i in range(1_500)]
        report = DiscoveryReport()

        with caplog.at_level(logging.WARNING, logger="sieve.core.discovery"):
            found = run_discovery(FakeGithub(repos), report=report)

        assert report.complete is False
        assert len(report.truncated_windows) == 1
        window = report.truncated_windows[0]
        assert (window["start"], window["end"]) == ("2024-06-15", "2024-06-15")
        assert window["total_count"] == 1_500 + len([r for r in repos[:2000] if r.created == busy])
        assert report.total_matching == len(repos)
        created = {r.full_name: r.created for r in repos}
        busy_found = [m.full_name for m in found if created[m.full_name] == busy]
        busy_top = top_by_stars([r for r in repos if r.created == busy], SEARCH_CAP)
        assert busy_found == busy_top   # the most-starred 1000 of that day
        assert any("2024-06-15" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    def test_single_query_that_runs_out_at_1000_before_the_cap_is_flagged(self):
        from sieve.core.discovery import DiscoveryReport

        repos = make_repos(3_000)
        for r in repos[300:]:
            r._contributors = 0      # only 300 repos pass the contributor filter
        report = DiscoveryReport()

        found = run_discovery(FakeGithub(repos), max_repos=900, report=report)

        # The first 1000 by stars were scanned and fewer than 900 passed: more may exist beyond them.
        assert len(found) < 900
        assert report.complete is False

    def test_single_query_that_reaches_the_cap_is_complete(self):
        from sieve.core.discovery import DiscoveryReport

        report = DiscoveryReport()
        run_discovery(FakeGithub(make_repos(3_000)), max_repos=100, report=report)
        assert report.complete is True
        assert report.slices == 1

    def test_report_is_optional(self):
        assert len(run_discovery(FakeGithub(make_repos(20)))) == 20

    def test_positional_signature_is_backward_compatible(self):
        fake = FakeGithub(make_repos(20))
        with patch("sieve.core.discovery.Github", return_value=fake), patch("sieve.core.discovery.time.sleep"):
            found = list(discover_repos("Python", START, END, 10, 1, 5, None))
        assert len(found) == 5


# ─── Bug 3: rate limits ───────────────────────────────────────────────────────

class TestRateLimits:
    def test_rate_limit_mid_run_raises_quota_error_with_reset_time(self):
        from sieve.core.discovery import DiscoveryReport

        fake = FakeGithub(make_repos(15_770), search_error=rate_limit_error(), search_error_at=4)
        report = DiscoveryReport()
        got = []

        with patch("sieve.core.discovery.Github", return_value=fake), patch("sieve.core.discovery.time.sleep"):
            with pytest.raises(GitHubQuotaError) as info:
                for meta in discover_repos("Python", START, END, 10, 1, None, None, report=report):
                    got.append(meta)

        assert got                                  # earlier windows were already yielded
        assert str(info.value) == build_exhausted_message(RESET, authenticated=False)
        assert "14:32 UTC" in str(info.value)
        assert report.complete is False

    def test_rate_limit_without_reset_header_still_raises_quota_error(self):
        fake = FakeGithub(make_repos(300), search_error=rate_limit_error(reset=None), search_error_at=1)
        with pytest.raises(GitHubQuotaError) as info:
            run_discovery(fake)
        assert "unknown time" in str(info.value)

    def test_rate_limit_while_counting_raises_quota_error(self):
        fake = FakeGithub(make_repos(5_000), count_error=rate_limit_error(), count_error_at=3)
        with pytest.raises(GitHubQuotaError):
            run_discovery(fake)

    def test_token_changes_the_message(self):
        fake = FakeGithub(make_repos(300), search_error=rate_limit_error(), search_error_at=1)
        with pytest.raises(GitHubQuotaError) as info:
            run_discovery(fake, github_token="ghp_x")
        assert info.value.args[0] == build_exhausted_message(RESET, authenticated=True)
        assert "public_repo" not in str(info.value)

    @pytest.mark.parametrize("error", [
        GithubException(429, {"message": "Too Many Requests"}, {}),
        GithubException(403, {"message": "You have exceeded a secondary rate limit. Please wait."}, {}),
    ], ids=["http-429", "secondary-403"])
    def test_secondary_rate_limits_become_quota_errors(self, error):
        fake = FakeGithub(make_repos(300), search_error=error, search_error_at=1)
        with pytest.raises(GitHubQuotaError):
            run_discovery(fake)

    def test_other_github_errors_are_raised_and_mark_the_report_incomplete(self):
        from sieve.core.discovery import DiscoveryReport

        fake = FakeGithub(make_repos(5_000), search_error=GithubException(500, "boom"), search_error_at=2)
        report = DiscoveryReport()
        with pytest.raises(GithubException) as info:
            run_discovery(fake, report=report)
        assert not isinstance(info.value, GitHubQuotaError)
        assert report.complete is False

    def test_rate_limit_in_a_per_repo_call_is_still_not_swallowed(self):
        repos = make_repos(5)
        repos[2].get_contributors = lambda anon=False: (_ for _ in ()).throw(rate_limit_error())
        with pytest.raises(RateLimitExceededException):
            run_discovery(FakeGithub(repos), max_repos=5)


# ─── Topics, aliases, language mapping ────────────────────────────────────────

class TestSearchPayloadAndNames:
    def test_topics_come_from_the_search_payload(self):
        fake = FakeGithub(make_repos(5))
        found = run_discovery(fake)
        assert fake.topic_calls == 0
        assert all(m.topics == ["from-search-payload"] for m in found)

    def test_public_names_with_underscore_aliases(self):
        import sieve.core.discovery as d

        assert d.build_query is d._build_query
        assert d.count_contributors is d._get_contributor_count
        assert callable(d.count_results)

    def test_contributor_counting_is_unchanged(self):
        from unittest.mock import MagicMock
        from sieve.core.discovery import count_contributors

        repo = MagicMock()
        repo.get_contributors.return_value.totalCount = 42
        assert count_contributors(repo, None) == 42
        repo.get_contributors.assert_called_once_with(anon=False)

        repo.get_contributors.side_effect = GithubException(403, "Forbidden")
        assert count_contributors(repo, None) == 0


class TestCppLanguageMapping:
    """
    "C++" has no entry in LANGUAGE_QUERY_MAP, so it falls through to language.lower() == "c++".
    That is correct: GitHub's search matches language:c++ case-insensitively and returns the same
    results as language:cpp and language:C++ (checked against the live API), and PyGithub
    percent-encodes the "+" as %2B, so the query reaches GitHub intact. This test pins the
    behaviour; the explicit map entries are not needed for it to work.
    """

    def test_cpp_falls_through_to_lowercase(self):
        from sieve.core.discovery import build_query

        assert "C++" not in LANGUAGE_QUERY_MAP
        assert build_query("C++", START, END, 10).startswith("language:c++ ")

    def test_every_supported_language_gets_a_language_qualifier(self):
        from sieve.core.discovery import build_query

        expected = {"Python": "python", "Java": "java", "JavaScript": "javascript", "C++": "c++"}
        for lang in Language:
            assert build_query(lang.value, START, END, 10).startswith(f"language:{expected[lang.value]} ")

    def test_enum_member_and_plain_string_give_the_same_query(self):
        from sieve.core.discovery import build_query

        assert build_query(Language.CPP, START, END, 10) == build_query("C++", START, END, 10)


# ─── Pipeline and manifest ────────────────────────────────────────────────────

TOKEN = "ghp_SECRET_TOKEN_1234567890"


def _config(tmp_path, **overrides):
    values = dict(
        language="Python", start_date=START, end_date=END, min_stars=10,
        output_dir=str(tmp_path / "out"), github_token=TOKEN, hf_token="hf_SECRET_TOKEN_1234567890",
    )
    values.update(overrides)
    return SIEVEConfig(**values)


def _run(tmp_path, fake, **overrides):
    with patch("sieve.core.discovery.Github", return_value=fake), \
         patch("sieve.core.discovery.time.sleep"), \
         patch("sieve.pipeline._fetch_quota", return_value=None), \
         patch("sieve.pipeline._clone_repo", return_value=False):
        return run_pipeline(_config(tmp_path, **overrides))


def _manifest(tmp_path):
    return json.loads((tmp_path / "out" / "manifest.json").read_text())


class TestPipelineManifest:
    def test_manifest_has_a_discovery_block(self, tmp_path):
        _run(tmp_path, FakeGithub(make_repos(2_500)))
        discovery = _manifest(tmp_path)["discovery"]
        assert discovery == {
            "complete": True,
            "total_matching": 2_500,
            "slices": discovery["slices"],
            "truncated_windows": [],
            "incomplete_results": False,
        }
        assert discovery["slices"] > 1

    def test_no_token_anywhere_in_the_manifest(self, tmp_path):
        _run(tmp_path, FakeGithub(make_repos(2_500)))
        text = (tmp_path / "out" / "manifest.json").read_text()
        assert "SECRET_TOKEN" not in text

    def test_truncated_day_is_in_the_manifest(self, tmp_path):
        busy = date(2024, 6, 15)
        repos = make_repos(100) + [FakeRepo(f"busy/r{i}", 50 + i, busy, None) for i in range(1_200)]
        _run(tmp_path, FakeGithub(repos))
        discovery = _manifest(tmp_path)["discovery"]
        assert discovery["complete"] is False
        assert [(w["start"], w["end"]) for w in discovery["truncated_windows"]] == [("2024-06-15", "2024-06-15")]

    def test_capped_run_manifest_reports_a_single_slice(self, tmp_path):
        _run(tmp_path, FakeGithub(make_repos(3_000)), max_repos=20)
        discovery = _manifest(tmp_path)["discovery"]
        assert discovery["complete"] is True
        assert discovery["slices"] == 1

    def test_export_dataset_without_discovery_omits_the_block(self, tmp_path):
        from sieve.core.export import export_dataset

        export_dataset([], [], str(tmp_path), "jsonl", {}, [])
        assert "discovery" not in json.loads((tmp_path / "manifest.json").read_text())


class TestPipelineRateLimit:
    def test_limit_before_anything_is_found_raises_and_writes_no_manifest(self, tmp_path):
        fake = FakeGithub(make_repos(5_000), count_error=rate_limit_error(), count_error_at=1)
        with pytest.raises(GitHubQuotaError) as info:
            _run(tmp_path, fake)
        assert "14:32 UTC" in str(info.value)
        assert not (tmp_path / "out" / "manifest.json").exists()

    def test_limit_mid_run_never_writes_a_manifest_claiming_complete(self, tmp_path):
        fake = FakeGithub(make_repos(15_770), search_error=rate_limit_error(), search_error_at=4)
        messages = []
        with patch("sieve.core.discovery.Github", return_value=fake), \
             patch("sieve.core.discovery.time.sleep"), \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", return_value=False):
            summary = run_pipeline(_config(tmp_path), progress_callback=lambda m, c, t: messages.append(m))

        assert summary["rate_limited"] is True
        assert summary["rate_limit_reset"] == RESET.isoformat()
        assert 0 < summary["total_repos_discovered"] < 15_770
        assert _manifest(tmp_path)["discovery"]["complete"] is False
        assert any("rate limit" in m.lower() and "partial" in m.lower() for m in messages)

    def test_other_discovery_error_writes_no_manifest(self, tmp_path):
        fake = FakeGithub(make_repos(5_000), search_error=GithubException(500, "boom"), search_error_at=1)
        with pytest.raises(GithubException):
            _run(tmp_path, fake)
        assert not (tmp_path / "out" / "manifest.json").exists()
