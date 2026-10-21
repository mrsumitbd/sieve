"""
core/github_limits.py

GitHub API rate-limit handling: a retry policy that fails fast on primary
rate limits, quota helpers and user-facing messages.

Without a token GitHub allows 60 REST requests per hour. Discovery spends
roughly three calls per qualifying repository (contributors, latest commit,
topics), so an anonymous run is exhausted after about 20 repositories.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from github import GithubRetry
from github.GithubException import RateLimitExceededException

logger = logging.getLogger(__name__)

UNAUTHENTICATED_LIMIT = 60
CALLS_PER_REPO = 3
MIN_CALLS_REQUIRED = 10

# A primary rate limit that resets sooner than this is waited out (this is the
# normal case for the search resource, which resets every minute). Anything
# longer raises immediately instead of sleeping inside the HTTP client.
MAX_PRIMARY_WAIT_SECONDS = 60

TOKEN_URL = "https://github.com/settings/tokens"


@dataclass
class QuotaInfo:
    """Snapshot of the GitHub API quota for the current client."""
    core_remaining: int
    core_limit: int
    core_reset: datetime          # timezone-aware, UTC
    search_remaining: int
    authenticated: bool


class GitHubQuotaError(RuntimeError):
    """Raised when the GitHub quota is too low to do useful work."""


# ─── Retry policy ─────────────────────────────────────────────────────────────

def _header(headers: Mapping[str, Any], name: str) -> Optional[str]:
    """Case-insensitive header lookup."""
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return value
    return None


def reset_time_from_headers(headers: Optional[Mapping[str, Any]]) -> Optional[datetime]:
    """Parse X-RateLimit-Reset (epoch seconds) into an aware UTC datetime."""
    if not headers:
        return None
    value = _header(headers, "X-RateLimit-Reset")
    if value is None or not str(value).isdigit():
        return None
    return datetime.fromtimestamp(int(value), timezone.utc)


def reset_time_from_exception(exc: Exception) -> Optional[datetime]:
    """Reset time carried by a PyGithub exception, if the response had one."""
    return reset_time_from_headers(getattr(exc, "headers", None))


class FailFastRetry(GithubRetry):
    """
    GithubRetry that raises RateLimitExceededException on a primary rate limit
    instead of sleeping until X-RateLimit-Reset (which can be up to an hour).

    Transient errors (5xx, connection errors) and secondary rate limits keep
    the retry behaviour of GithubRetry.
    """

    def increment(self, method=None, url=None, response=None, error=None,
                  _pool=None, _stacktrace=None):
        if response is not None and response.status == 403:
            headers = response.headers
            if (
                _header(headers, "Retry-After") is None
                and str(_header(headers, "X-RateLimit-Remaining")) == "0"
            ):
                reset = reset_time_from_headers(headers)
                wait = (reset - datetime.now(timezone.utc)).total_seconds() if reset else None
                if wait is None or wait > MAX_PRIMARY_WAIT_SECONDS:
                    raise self._rate_limit_exception(response, url, headers)
        return super().increment(method, url, response, error, _pool, _stacktrace)

    def _rate_limit_exception(self, response, url, headers) -> RateLimitExceededException:
        try:
            data = json.loads(self.get_content(response, url))
        except Exception:
            data = {"message": "API rate limit exceeded"}
        return RateLimitExceededException(403, data, dict(headers))


def fail_fast_retry() -> FailFastRetry:
    """Retry policy used for every GitHub client in SIEVE."""
    return FailFastRetry(
        total=5,
        connect=None,
        read=None,
        redirect=None,
        status=None,
        secondary_rate_wait=30,
    )


# ─── Quota ────────────────────────────────────────────────────────────────────

def read_quota(client, authenticated: bool) -> QuotaInfo:
    """
    Read the current quota via get_rate_limit().

    This endpoint does not count against the core limit.
    """
    resources = client.get_rate_limit().resources
    reset = resources.core.reset
    if reset.tzinfo is None:
        reset = reset.replace(tzinfo=timezone.utc)
    return QuotaInfo(
        core_remaining=resources.core.remaining,
        core_limit=resources.core.limit,
        core_reset=reset,
        search_remaining=resources.search.remaining,
        authenticated=authenticated,
    )


def format_reset(reset: Optional[datetime]) -> str:
    if reset is None:
        return "an unknown time"
    return reset.astimezone(timezone.utc).strftime("%H:%M UTC")


def estimate_repos(remaining_calls: int) -> int:
    """Rough number of repositories that can be processed with N calls."""
    return max(remaining_calls, 0) // CALLS_PER_REPO


def format_quota_line(quota: QuotaInfo) -> str:
    mode = "token" if quota.authenticated else "no token"
    return (
        f"GitHub API: {quota.core_remaining}/{quota.core_limit} requests remaining, "
        f"resets at {format_reset(quota.core_reset)} ({mode})"
    )


def token_hint() -> str:
    return (
        "Create a classic token with the public_repo scope at "
        f"{TOKEN_URL}, then enter it in the \"GitHub Token\" field in the web UI, "
        "or set the GITHUB_TOKEN environment variable or pass --github-token in the CLI."
    )


def build_exhausted_message(reset: Optional[datetime], authenticated: bool) -> str:
    """User-facing message for an exhausted GitHub rate limit."""
    msg = f"GitHub API rate limit reached. It resets at {format_reset(reset)}."
    if authenticated:
        return f"{msg} Wait until then and run again, or lower the repository cap."
    return (
        f"{msg} Without a token GitHub allows {UNAUTHENTICATED_LIMIT} requests per hour, "
        f"enough for roughly {estimate_repos(UNAUTHENTICATED_LIMIT)} repositories. "
        f"{token_hint()}"
    )
