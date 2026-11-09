"""
core/github_limits.py

GitHub API rate-limit handling: a retry policy that fails fast on primary
rate limits, an opt-in waiter that sleeps a limit out, quota-aware pacing,
quota helpers and user-facing messages.

Without a token GitHub allows 60 REST requests per hour. Discovery spends,
per qualifying repository, ceil(n/100) requests listing its n contributors
(100 per page) plus one for the latest commit; topics come with the search
results and cost nothing. Every scanned repository that is then rejected
still costs its contributor pages. Search requests (including the result
counts used to split date windows) are counted against the separate search
limit, not this one. See CALLS_PER_REPO for the planning figure derived from
this.
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from itertools import islice
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, TypeVar

from github import GithubRetry
from github.GithubException import BadCredentialsException, GithubException, RateLimitExceededException
from github.PaginatedList import PaginatedListBase

logger = logging.getLogger(__name__)

T = TypeVar("T")

UNAUTHENTICATED_LIMIT = 60
MIN_CALLS_REQUIRED = 10

# ─── Planning figures (core REST requests) ────────────────────────────────────
#
# Contributors are listed 100 per page (the page size of the client built by
# discovery.make_client, GITHUB_SEARCH_PAGE_SIZE there), so a repository with n
# contributors, bots included, costs ceil(n/100) requests, at least one.
CONTRIBUTORS_PER_PAGE = 100
# The latest-commit lookup is one request. Topics cost none: they are part of
# the search payload.
LATEST_COMMIT_CALLS = 1
# Contributor pages assumed for an accepted repository. Past runs kept in
# sieve_output/ (min_stars 500) averaged 1.1 (Java), 1.3 (C++) and 1.9
# (JavaScript) pages per accepted repository; 2 rounds the worst of them up.
PLANNED_CONTRIBUTOR_PAGES = 2
# Scanned-and-rejected repositories per accepted one, one contributors page
# each. An assumption, not a measurement: it is 0 with min_contributors=1 and
# grows with the threshold.
PLANNED_REJECTED_PER_ACCEPTED = 2
# Planning figure per accepted repository: 1 + 2 + 2 = 5 (was 6, from when it
# was three calls per repository doubled for scan overhead).
CALLS_PER_REPO = LATEST_COMMIT_CALLS + PLANNED_CONTRIBUTOR_PAGES + PLANNED_REJECTED_PER_ACCEPTED


def contributor_pages(contributors: int) -> int:
    """Requests needed to list a repository's contributors: ceil(n/100), at least 1."""
    return max(1, math.ceil(max(contributors, 0) / CONTRIBUTORS_PER_PAGE))


def calls_for_repo(contributors: int) -> int:
    """Core requests discovery spends on one accepted repository with this many contributors."""
    return contributor_pages(contributors) + LATEST_COMMIT_CALLS


# A primary rate limit that resets sooner than this is waited out (this is the
# normal case for the search resource, which resets every minute). Anything
# longer raises immediately instead of sleeping inside the HTTP client.
MAX_PRIMARY_WAIT_SECONDS = 60

# ─── Waiting a limit out (opt-in, see RateLimitWaiter) ────────────────────────
WAIT_BUFFER_SECONDS = 5               # added to the reset time: clock skew, reset races
WAIT_CHUNK_SECONDS = 5                # sleep granularity, so Ctrl-C is answered promptly
WAIT_PROGRESS_INTERVAL_SECONDS = 60   # how often a long wait reports what is left
SECONDARY_DEFAULT_WAIT_SECONDS = 60   # secondary limit without a Retry-After header
UNKNOWN_RESET_WAIT_SECONDS = 60       # primary limit whose response carried no reset time
MAX_WAITS_PER_CALL = 5                # consecutive waits for one call before giving up

# ─── Pacing ───────────────────────────────────────────────────────────────────
# Requests are sent one after another with no pause while plenty of quota is
# left. Once the remaining quota falls to this fraction of the window (a tenth:
# 6 of 60, 500 of 5000) each repository is followed by a short pause.
PACING_LOW_FRACTION = 0.10
PACING_DELAY_SECONDS = 1.0

TOKEN_URL = "https://github.com/settings/tokens"


@dataclass
class QuotaInfo:
    """Snapshot of the GitHub API quota for the current client."""
    core_remaining: int
    core_limit: int
    core_reset: datetime          # timezone-aware, UTC
    search_remaining: int
    authenticated: bool


BAD_TOKEN_MESSAGE = (
    "GitHub rejected the token (401 Bad credentials). Check that it is a valid "
    "classic token with the public_repo scope and has not expired."
)


class GitHubQuotaError(RuntimeError):
    """Raised when the GitHub quota is too low to do useful work."""


class GitHubAuthError(RuntimeError):
    """Raised when GitHub rejects the token (HTTP 401)."""

    def __init__(self, message: str = BAD_TOKEN_MESSAGE):
        super().__init__(message)


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


# ─── Telling the two kinds of rate limit apart ────────────────────────────────

@dataclass(frozen=True)
class RateLimitHit:
    """A rate limit that GitHub reported.

    kind:         "primary" (the hourly quota is used up) or "secondary" (too
                  many requests too quickly; GitHub asks for a pause).
    reset:        Primary only: when the quota refills (aware UTC), if the
                  response said.
    retry_after:  Secondary only: seconds GitHub asked us to wait, if it said.
    """
    kind: str
    reset: Optional[datetime] = None
    retry_after: Optional[float] = None


def _retry_after(headers: Mapping[str, Any]) -> Optional[float]:
    """Retry-After as seconds. GitHub sends a number; an HTTP date is accepted too."""
    value = _header(headers, "Retry-After")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(str(value))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def classify_rate_limit(exc: BaseException) -> Optional[RateLimitHit]:
    """
    Primary or secondary rate limit behind a PyGithub exception; None if it is
    not a rate limit (a rejected token, a missing repository, a 5xx, ...).

    Secondary: a message naming a "secondary rate limit", any 429, or a 403 that
    carries Retry-After. Primary: RateLimitExceededException, or a 403 that
    names a rate limit or reports X-RateLimit-Remaining: 0.
    """
    if not isinstance(exc, GithubException) or isinstance(exc, BadCredentialsException):
        return None
    headers = getattr(exc, "headers", None) or {}
    data = getattr(exc, "data", None)
    message = str(data.get("message", "") if isinstance(data, dict) else data or "").lower()
    retry_after = _retry_after(headers)

    if (
        "secondary rate limit" in message
        or exc.status == 429
        or (exc.status == 403 and retry_after is not None)
    ):
        return RateLimitHit("secondary", retry_after=retry_after)
    if (
        isinstance(exc, RateLimitExceededException)
        or (exc.status == 403 and ("rate limit" in message
                                   or str(_header(headers, "X-RateLimit-Remaining")) == "0"))
    ):
        return RateLimitHit("primary", reset=reset_time_from_headers(headers))
    return None


def format_duration(seconds: float) -> str:
    """45s, 2m 05s, 1h 02m 05s (rounded up to whole seconds)."""
    total = max(0, math.ceil(seconds - 1e-9))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _resume(source: Iterable[T], consumed: int) -> Iterator[T]:
    """
    Iterate `source` starting after the first `consumed` items, which the
    caller already has.

    A PyGithub PaginatedList caches the pages it has fetched and a failed page
    fetch leaves it unchanged, so slicing it from `consumed` continues with the
    page that failed: cached pages are neither requested nor yielded again.
    (Iterating it afresh would yield every cached item a second time.) Any
    other iterable is skipped through instead.
    """
    if consumed == 0:
        return iter(source)
    if isinstance(source, PaginatedListBase):
        return iter(source[consumed:])
    return islice(iter(source), consumed, None)


class RateLimitWaiter:
    """
    Runs GitHub calls and, when enabled, waits out the rate limits they hit.

    Wrap every GitHub call in call() (a single request) or iterate() (a lazy
    paginated list: PyGithub fetches page N+1 while you iterate, so a limit can
    surface in the middle of a for loop). When enabled, a primary limit sleeps
    until its reset time plus WAIT_BUFFER_SECONDS, a secondary limit sleeps for
    Retry-After (SECONDARY_DEFAULT_WAIT_SECONDS if absent), and the same call is
    then retried. The sleep is cut into WAIT_CHUNK_SECONDS pieces so Ctrl-C is
    answered promptly (KeyboardInterrupt propagates untouched), and a progress
    message gives the reset time and the wait left. Anything that is not a rate
    limit, and every error once a call has waited MAX_WAITS_PER_CALL times in a
    row without result, is raised as it came.

    Disabled (the default), call() and iterate() are transparent: the original
    exception reaches the caller exactly as before.

    Messages contain times only, never exception text, headers or tokens.
    waits and seconds accumulate over the waiter's life; stats() reports them.
    """

    def __init__(
        self,
        enabled: bool = False,
        progress: Optional[Callable[[str], None]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        now: Optional[Callable[[], datetime]] = None,
    ):
        self.enabled = enabled
        self._progress = progress
        self._sleep = sleep                  # None: time.sleep, looked up at call time
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.waits = 0
        self.seconds = 0.0

    def stats(self) -> dict:
        """{"count": waits, "seconds": total seconds slept}, as recorded in the manifest."""
        return {"count": self.waits, "seconds": round(self.seconds, 1)}

    # -- the two entry points --------------------------------------------------

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """fn(*args, **kwargs), waiting out rate limits when enabled."""
        if not self.enabled:
            return fn(*args, **kwargs)
        waits = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except GithubException as exc:
                hit = classify_rate_limit(exc)
                if hit is None or waits >= MAX_WAITS_PER_CALL:
                    raise
            # Waiting happens outside the except block so that Ctrl-C is not
            # reported as raised "during handling of" the rate limit.
            self._wait_out(hit)
            waits += 1

    def iterate(self, make_iterable: Callable[[], Iterable[T]]) -> Iterator[T]:
        """
        Items of make_iterable() (e.g. lambda: repo.get_contributors(anon=False)),
        waiting out rate limits, including one that hits while a page is being
        fetched mid-iteration. After a wait the iteration resumes at the first
        item not yet handed out: nothing is repeated and nothing is skipped
        (see _resume). Lazy: pages are fetched as the caller advances.
        """
        if not self.enabled:
            yield from make_iterable()
            return
        source: Optional[Iterable[T]] = None
        consumed = waits = 0
        while True:
            try:
                if source is None:
                    source = make_iterable()
                for item in _resume(source, consumed):
                    consumed += 1
                    waits = 0
                    yield item
                return
            except GithubException as exc:
                hit = classify_rate_limit(exc)
                if hit is None or waits >= MAX_WAITS_PER_CALL:
                    raise
            self._wait_out(hit)
            waits += 1

    def wait_for_reset(self, reset: Optional[datetime]) -> None:
        """Wait out a primary limit that is already known (the start-of-run quota check)."""
        self._wait_out(RateLimitHit("primary", reset=reset))

    # -- waiting ---------------------------------------------------------------

    def _say(self, message: str) -> None:
        (self._progress or logger.info)(message)

    def _wait_out(self, hit: RateLimitHit) -> None:
        now = self._now()
        if hit.kind == "secondary":
            delay = hit.retry_after if hit.retry_after is not None else SECONDARY_DEFAULT_WAIT_SECONDS
            self._say(
                f"GitHub secondary rate limit reached. Waiting {format_duration(delay)} as GitHub "
                f"asked (until about {format_reset(now + timedelta(seconds=delay))}), then resuming."
            )
        elif hit.reset is None:
            delay = UNKNOWN_RESET_WAIT_SECONDS
            self._say(
                f"GitHub rate limit reached; its reset time is unknown. Waiting "
                f"{format_duration(delay)}, then resuming."
            )
        else:
            delay = max((hit.reset - now).total_seconds(), 0.0) + WAIT_BUFFER_SECONDS
            self._say(
                f"GitHub rate limit reached. Waiting for it to reset at {format_reset(hit.reset)} "
                f"({format_duration(delay)} left, including a {WAIT_BUFFER_SECONDS}s margin); "
                f"the run resumes by itself."
            )
        self._sleep_through(delay)
        self._say("GitHub rate limit wait over; resuming.")

    def _sleep_through(self, delay: float) -> None:
        self.waits += 1
        deadline = self._now() + timedelta(seconds=delay)
        sleep = self._sleep or time.sleep
        slept = reported = 0.0
        # Stops at `delay` of actual sleeping, or sooner if the wall clock got there first
        # (the machine was suspended meanwhile).
        while slept < delay and self._now() < deadline:
            step = min(WAIT_CHUNK_SECONDS, delay - slept)
            sleep(step)
            slept += step
            self.seconds += step
            if slept < delay and slept - reported >= WAIT_PROGRESS_INTERVAL_SECONDS:
                self._say(
                    f"Still waiting for the GitHub rate limit: {format_duration(delay - slept)} left."
                )
                reported = slept


# ─── Pacing ───────────────────────────────────────────────────────────────────

def pace_requests(client, sleep: Optional[Callable[[float], None]] = None) -> float:
    """
    Pause briefly if the client's remaining quota is low; return the seconds slept.

    Reads client.rate_limiting, the (remaining, limit) PyGithub records from
    the headers of the last response, so it costs no request. The pause
    (PACING_DELAY_SECONDS) applies only once remaining/limit has fallen to
    PACING_LOW_FRACTION; with plenty of quota nothing sleeps. If the quota
    cannot be read as two numbers (no response yet, a stubbed client, an
    error) nothing sleeps either.

    The recorded figures belong to the resource of the last response (core or
    search); the ratio is what is tested, so either one works.
    """
    try:
        remaining, limit = client.rate_limiting
    except Exception:
        return 0.0
    if not isinstance(remaining, int) or not isinstance(limit, int) or limit <= 0 or remaining < 0:
        return 0.0
    if remaining / limit > PACING_LOW_FRACTION:
        return 0.0
    (sleep or time.sleep)(PACING_DELAY_SECONDS)
    return PACING_DELAY_SECONDS


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


def describe_rate_limit_error(exc: Exception, authenticated: bool) -> str:
    """User-facing text for a rate-limit, quota or token error."""
    if isinstance(exc, (GitHubQuotaError, GitHubAuthError)):
        return str(exc)
    if isinstance(exc, BadCredentialsException):
        return BAD_TOKEN_MESSAGE
    return build_exhausted_message(reset_time_from_exception(exc), authenticated)


def partial_results_message(reset_iso: Optional[str], authenticated: bool) -> str:
    """Warning shown when a run finished early because of the rate limit."""
    reset = datetime.fromisoformat(reset_iso) if reset_iso else None
    msg = (
        "Results are partial because the GitHub rate limit was reached. "
        f"It resets at {format_reset(reset)}."
    )
    if not authenticated:
        msg += " Add a GitHub token for full runs."
    return msg
