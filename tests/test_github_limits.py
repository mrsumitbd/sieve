"""
tests/test_github_limits.py

Rate-limit handling: the fail-fast retry policy, quota helpers, messages,
and how discovery and the pipeline react to an exhausted limit.
No external network: the HTTP test talks to a loopback server.
"""

import json
import threading
import time
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from github.GithubException import RateLimitExceededException

from sieve.config import SIEVEConfig
from sieve.core.discovery import RepoMetadata, discover_repos, make_client
from sieve.core.github_limits import (
    GitHubQuotaError,
    QuotaInfo,
    build_exhausted_message,
    estimate_repos,
    format_quota_line,
    read_quota,
    reset_time_from_exception,
)
from sieve.pipeline import run_pipeline


# ─── Loopback HTTP server ─────────────────────────────────────────────────────

class _Server:
    """Serves a scripted list of (status, headers, body) responses in order."""

    def __init__(self, responses):
        outer = self
        self.responses = list(responses)
        self.hits = 0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                idx = min(outer.hits, len(outer.responses) - 1)
                outer.hits += 1
                status, headers, body = outer.responses[idx]
                payload = json.dumps(body).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def _rate_limit_body():
    return {"message": "API rate limit exceeded for 1.2.3.4.", "documentation_url": "https://docs.github.com"}


_RATE_LIMIT_PAYLOAD = {
    "resources": {
        "core": {"limit": 60, "used": 0, "remaining": 60, "reset": 4102444800},
        "search": {"limit": 10, "used": 0, "remaining": 10, "reset": 4102444800},
    },
    "rate": {"limit": 60, "used": 0, "remaining": 60, "reset": 4102444800},
}


class TestFailFastClient:
    def test_primary_rate_limit_raises_immediately(self):
        reset = str(int(time.time()) + 3600)
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": "60", "X-RateLimit-Reset": reset}
        with _Server([(403, headers, _rate_limit_body())]) as server:
            client = make_client(None, base_url=server.url)
            start = time.monotonic()
            with pytest.raises(RateLimitExceededException) as info:
                client.get_repo("octocat/hello")
            elapsed = time.monotonic() - start

        assert elapsed < 2
        assert server.hits == 1
        assert reset_time_from_exception(info.value) == datetime.fromtimestamp(int(reset), timezone.utc)

    def test_transient_server_errors_are_still_retried(self):
        ok = (200, {}, _RATE_LIMIT_PAYLOAD)
        with _Server([(502, {}, {"message": "Bad Gateway"}), (500, {}, {"message": "oops"}), ok]) as server:
            client = make_client("token", base_url=server.url)
            quota = read_quota(client, authenticated=True)

        assert server.hits == 3
        assert quota.core_remaining == 60


# ─── Helpers ──────────────────────────────────────────────────────────────────

_RESET = datetime(2030, 1, 1, 14, 32, tzinfo=timezone.utc)


def _quota(remaining=54, authenticated=False):
    return QuotaInfo(
        core_remaining=remaining,
        core_limit=60,
        core_reset=_RESET,
        search_remaining=10,
        authenticated=authenticated,
    )


class TestHelpers:
    def test_read_quota_from_client(self):
        client = MagicMock()
        res = client.get_rate_limit.return_value.resources
        res.core.remaining, res.core.limit, res.core.reset = 54, 60, _RESET
        res.search.remaining = 9

        quota = read_quota(client, authenticated=False)

        assert quota == QuotaInfo(54, 60, _RESET, 9, False)

    def test_read_quota_makes_naive_reset_aware(self):
        client = MagicMock()
        res = client.get_rate_limit.return_value.resources
        res.core.remaining, res.core.limit, res.core.reset = 5, 60, datetime(2030, 1, 1, 14, 32)
        res.search.remaining = 9

        assert read_quota(client, authenticated=True).core_reset.tzinfo is not None

    def test_quota_line_without_token(self):
        assert format_quota_line(_quota()) == (
            "GitHub API: 54/60 requests remaining, resets at 14:32 UTC (no token)"
        )

    def test_quota_line_with_token(self):
        assert format_quota_line(_quota(4990, True)).endswith("(token)")

    def test_estimate_repos(self):
        assert estimate_repos(60) == 10
        assert estimate_repos(6) == 1
        assert estimate_repos(5) == 0
        assert estimate_repos(2) == 0
        assert estimate_repos(-5) == 0

    def test_unauthenticated_message_explains_token(self):
        msg = build_exhausted_message(_RESET, authenticated=False)
        assert "14:32 UTC" in msg
        assert "60 requests per hour" in msg
        assert "public_repo" in msg
        assert "https://github.com/settings/tokens" in msg
        assert "GitHub Token" in msg
        assert "GITHUB_TOKEN" in msg
        assert "--github-token" in msg

    def test_authenticated_message_has_no_token_advice(self):
        msg = build_exhausted_message(_RESET, authenticated=True)
        assert "14:32 UTC" in msg
        assert "public_repo" not in msg

    def test_message_without_reset_time(self):
        assert "unknown time" in build_exhausted_message(None, authenticated=False)

    def test_reset_time_from_exception_without_headers(self):
        assert reset_time_from_exception(RateLimitExceededException(403, {}, None)) is None


# ─── Discovery and pipeline ───────────────────────────────────────────────────

def _mock_repo(name):
    repo = MagicMock()
    repo.full_name = name
    repo.html_url = f"https://github.com/{name}"
    repo.stargazers_count = 100
    repo.default_branch = "main"
    repo.license = None
    repo.get_topics.return_value = []
    repo.pushed_at = None
    repo.get_commits.return_value = [MagicMock()]
    return repo


def _results(repos):
    results = MagicMock()
    results.__iter__.return_value = iter(repos)
    return results


def _meta(name):
    return RepoMetadata(
        full_name=name, url=f"https://github.com/{name}", stars=100, contributors=5,
        last_commit_date="2024-01-01T00:00:00", default_branch="main", license_spdx="MIT",
        language="Python", collected_at="2024-06-01T00:00:00+00:00", topics=[],
    )


def _limit_error():
    return RateLimitExceededException(
        403, {"message": "API rate limit exceeded"},
        {"X-RateLimit-Reset": str(int(_RESET.timestamp()))},
    )


def _config(tmp_path, token=None, engineered=False):
    return SIEVEConfig(
        language="Python",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        engineered_only=engineered,
        output_dir=str(tmp_path / "out"),
        github_token=token,
    )


def _gen(items, then_raise=None):
    def gen(**kwargs):
        yield from items
        if then_raise is not None:
            raise then_raise
    return gen


class TestDiscoveryRateLimit:
    def test_rate_limit_in_contributor_count_is_not_swallowed(self):
        repos = [_mock_repo("a/one"), _mock_repo("a/two")]
        repos[0].get_contributors.return_value.totalCount = 5
        repos[1].get_contributors.side_effect = _limit_error()

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.core.discovery.time.sleep"):
            mock_gh.return_value.search_repositories.return_value = _results(repos)
            got = []
            with pytest.raises(RateLimitExceededException):
                for meta in discover_repos("Python", date(2024, 1, 1), date(2024, 12, 31)):
                    got.append(meta.full_name)

        assert got == ["a/one"]

    def test_rate_limit_in_last_commit_lookup_is_not_swallowed(self):
        repo = _mock_repo("a/one")
        repo.get_contributors.return_value.totalCount = 5
        repo.get_commits.side_effect = _limit_error()

        with patch("sieve.core.discovery.Github") as mock_gh:
            mock_gh.return_value.search_repositories.return_value = _results([repo])
            with pytest.raises(RateLimitExceededException):
                list(discover_repos("Python", date(2024, 1, 1), date(2024, 12, 31)))


class TestPipelineRateLimit:
    def test_mid_discovery_returns_partial_and_signals_it(self, tmp_path):
        messages = []
        discovered = [_meta("a/one"), _meta("a/two")]

        with patch("sieve.pipeline.discover_repos", side_effect=_gen(discovered, _limit_error())), \
             patch("sieve.pipeline._clone_repo", return_value=False):
            summary = run_pipeline(_config(tmp_path), progress_callback=lambda m, c, t: messages.append(m))

        assert summary["total_repos_discovered"] == 2
        assert summary["rate_limited"] is True
        assert summary["rate_limit_reset"] == _RESET.isoformat()
        warning = [m for m in messages if "rate limit reached" in m]
        assert len(warning) == 1
        assert "partial" in warning[0]
        assert "14:32 UTC" in warning[0]

    def test_complete_run_is_not_flagged(self, tmp_path):
        with patch("sieve.pipeline.discover_repos", side_effect=_gen([_meta("a/one")])), \
             patch("sieve.pipeline._clone_repo", return_value=False):
            summary = run_pipeline(_config(tmp_path))

        assert summary["rate_limited"] is False
        assert summary["rate_limit_reset"] is None

    def test_limit_before_any_repo_raises_clear_error(self, tmp_path):
        with patch("sieve.pipeline.discover_repos", side_effect=_gen([], _limit_error())):
            with pytest.raises(GitHubQuotaError) as info:
                run_pipeline(_config(tmp_path))

        assert str(info.value) == build_exhausted_message(_RESET, authenticated=False)

    def test_limit_before_any_repo_with_token_omits_token_advice(self, tmp_path):
        with patch("sieve.pipeline.discover_repos", side_effect=_gen([], _limit_error())):
            with pytest.raises(GitHubQuotaError) as info:
                run_pipeline(_config(tmp_path, token="ghp_x"))

        assert "public_repo" not in str(info.value)

    def test_low_quota_fails_before_discovery(self, tmp_path):
        with patch("sieve.pipeline._fetch_quota", return_value=_quota(remaining=9)), \
             patch("sieve.pipeline.discover_repos") as mock_discover:
            with pytest.raises(GitHubQuotaError) as info:
                run_pipeline(_config(tmp_path))

        mock_discover.assert_not_called()
        assert "14:32 UTC" in str(info.value)

    def test_ten_remaining_is_enough_and_is_reported(self, tmp_path):
        messages = []
        with patch("sieve.pipeline._fetch_quota", return_value=_quota(remaining=10)), \
             patch("sieve.pipeline.discover_repos", side_effect=_gen([_meta("a/one")])), \
             patch("sieve.pipeline._clone_repo", return_value=False):
            run_pipeline(_config(tmp_path), progress_callback=lambda m, c, t: messages.append(m))

        assert "GitHub API: 10/60 requests remaining, resets at 14:32 UTC (no token)" in messages

    def test_unreadable_quota_does_not_block_the_run(self, tmp_path):
        from sieve.pipeline import _fetch_quota

        with patch("sieve.pipeline.make_client", side_effect=OSError("offline")):
            assert _fetch_quota(None) is None

    def test_limit_during_engineered_filter_keeps_checked_repos(self, tmp_path):
        from sieve.core.quality import RepoQualityMetrics

        passing = RepoQualityMetrics(
            full_name="a/one", license_spdx="MIT", contributor_count=5, release_count=3,
            pull_request_count=10, issue_count=10, loc=1000, comment_lines=100, code_ratio=0.9,
            passes_stage1=True, passes_stage2=True, passes_stage3=True,
        )
        messages = []
        with patch("sieve.pipeline.discover_repos", side_effect=_gen([_meta("a/one"), _meta("a/two")])), \
             patch("sieve.pipeline.collect_metrics", side_effect=[passing, _limit_error()]), \
             patch("sieve.pipeline.apply_filters", side_effect=lambda ms: ms), \
             patch("sieve.pipeline._clone_repo", return_value=False), \
             patch("sieve.pipeline.time.sleep"):
            summary = run_pipeline(
                _config(tmp_path, engineered=True),
                progress_callback=lambda m, c, t: messages.append(m),
            )

        assert summary["rate_limited"] is True
        assert summary["total_repos_after_quality_filter"] == 1
        assert any("engineered filter" in m and "partial" in m for m in messages)

    def test_limit_in_engineered_filter_before_any_check_raises(self, tmp_path):
        with patch("sieve.pipeline.discover_repos", side_effect=_gen([_meta("a/one")])), \
             patch("sieve.pipeline.collect_metrics", side_effect=_limit_error()), \
             patch("sieve.pipeline.time.sleep"):
            with pytest.raises(GitHubQuotaError):
                run_pipeline(_config(tmp_path, engineered=True))
