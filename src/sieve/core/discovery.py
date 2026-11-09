"""
core/discovery.py

Discovers GitHub repositories matching user-defined filters.
Excludes forks and archived repos by design.

GitHub search returns at most 1000 results per query, so a query that matches
more is split by repository creation date into windows of at most 1000
results each (see discover_repos). What discovery could and could not cover
is recorded in a DiscoveryReport.
"""

import time
import logging
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Union

import requests
from github import Auth, Github, GithubException
from github.GithubException import BadCredentialsException, RateLimitExceededException
from github.Repository import Repository
from pydantic import SecretStr

from sieve.config import reveal_secret
from sieve.core.github_limits import (
    GitHubQuotaError,
    build_exhausted_message,
    fail_fast_retry,
    reset_time_from_exception,
)

logger = logging.getLogger(__name__)


@dataclass
class RepoMetadata:
    """Snapshot of repo metadata at collection time."""
    full_name: str           # e.g. "owner/repo"
    url: str
    stars: int
    contributors: Optional[int]   # None: GitHub would not list them (too many to count)
    last_commit_date: str    # ISO format
    default_branch: str
    license_spdx: Optional[str]
    language: str
    collected_at: str        # ISO format timestamp
    topics: list[str]


@dataclass
class DiscoveryReport:
    """
    What a discovery run covered. Pass one to discover_repos(report=...) and
    read it once the generator has been consumed (or has raised).

    complete:           True when every matching repository was reachable: no
                        window hit the 1000-result limit, GitHub did not flag
                        the search as incomplete, and discovery did not stop
                        on an error. A max_repos cap does not make a run
                        incomplete.
    total_matching:     Repositories GitHub reports for the filters (the sum
                        of the true total_count of every window searched).
                        A single-query run (max_repos <= 1000) does not ask
                        GitHub for a total; there this is the number of search
                        results scanned.
    slices:             Search queries (windows) run.
    truncated_windows:  Single-day windows with more than 1000 results, which
                        GitHub cannot return in full and discovery cannot
                        split further. Each entry is
                        {"start", "end", "total_count"} (ISO dates).
    incomplete_results: GitHub flagged at least one count as incomplete (the
                        search timed out, so results may be missing).
    excluded:           Repositories dropped because their contributors could not
                        be fetched (after retries). Each entry is
                        {"repo", "reason"}. Any entry also makes `complete` False:
                        the repository matched the search but was not evaluated.
    contributors_unknown:
                        Names of repositories kept although GitHub refused to
                        list their contributors because the list is too large.
                        They are recorded with contributors=None and pass any
                        min_contributors threshold.
    """
    complete: bool = True
    total_matching: int = 0
    slices: int = 0
    truncated_windows: list = field(default_factory=list)
    incomplete_results: bool = False
    excluded: list = field(default_factory=list)
    contributors_unknown: list = field(default_factory=list)


# Map language names to GitHub search query values. Names not listed here are
# lowercased, which GitHub accepts for every language SIEVE supports (e.g.
# "C++" -> "c++").
LANGUAGE_QUERY_MAP = {
    "Python": "python",
    "Java": "java",
    "JavaScript": "javascript",
}

# GitHub search returns at most this many results per query, however many match.
GITHUB_SEARCH_RESULT_LIMIT = 1000

# Results per page; also the page size of the PyGithub client.
GITHUB_SEARCH_PAGE_SIZE = 100


def make_client(github_token: Union[str, SecretStr, None] = None, **kwargs) -> Github:
    """Build the GitHub client used for discovery (fails fast on rate limits)."""
    return Github(
        auth=Auth.Token(reveal_secret(github_token)) if github_token else None,
        per_page=GITHUB_SEARCH_PAGE_SIZE,
        retry=fail_fast_retry(),
        **kwargs,
    )


def build_query(
    language: str,
    start_date: date,
    end_date: date,
    min_stars: int,
) -> str:
    """GitHub search query for the filters. The created range is inclusive on both ends."""
    start = start_date.strftime("%Y-%m-%d")
    end   = end_date.strftime("%Y-%m-%d")
    lang  = LANGUAGE_QUERY_MAP.get(language, language.lower())
    return (
        f"language:{lang} "
        f"created:{start}..{end} "
        f"stars:>={min_stars} "
        f"fork:false "
        f"archived:false"
    )


# Contributor counting retries transient errors this many times (so up to
# 1 + CONTRIBUTOR_RETRIES requests), sleeping CONTRIBUTOR_BACKOFF_SECONDS * 2**n between them.
CONTRIBUTOR_RETRIES = 3
CONTRIBUTOR_BACKOFF_SECONDS = 0.5


class ContributorCountError(Exception):
    """The contributors of a repository could not be fetched (after retries)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _is_bot(user) -> bool:
    """GitHub App and automation accounts: login ending in "[bot]" or type "Bot"."""
    login = getattr(user, "login", None)
    if isinstance(login, str) and login.lower().endswith("[bot]"):
        return True
    return getattr(user, "type", None) == "Bot"


def _is_contributors_too_large(exc: GithubException) -> bool:
    """GitHub's 403 for repositories whose contributor list it will not produce."""
    if exc.status != 403 or not isinstance(exc.data, dict):
        return False
    return "too large" in str(exc.data.get("message", "")).lower()


def _is_transient(exc: Exception) -> bool:
    """Server errors and connection problems, which are worth another attempt."""
    if isinstance(exc, GithubException):
        return exc.status >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def _describe_error(exc: Exception, token: Union[str, SecretStr, None]) -> str:
    """Short reason for the report; never contains the token."""
    if isinstance(exc, GithubException):
        message = exc.data.get("message") if isinstance(exc.data, dict) else exc.data
        reason = f"HTTP {exc.status}: {message}" if message else f"HTTP {exc.status}"
    else:
        reason = f"{type(exc).__name__}: {exc}"
    secret = reveal_secret(token)
    return reason.replace(secret, "***") if secret else reason


def count_contributors(
    repo: Repository, token: Union[str, SecretStr, None] = None
) -> Optional[int]:
    """
    Number of human contributors (bots excluded), counted exactly.

    Bot accounts (login ending in "[bot]", or type "Bot") are regular
    contributors as far as the API is concerned (anon=False only drops
    anonymous, email-only ones), so they are skipped here. That needs the
    list itself, not just its length: it is fetched 100 per page (the page
    size of the client, see make_client). The value is recorded in the
    manifest and read by the quality filter, so it is never cut short.

    Returns None when GitHub refuses to list the contributors because the
    list is too large: the repository clearly has many.

    Raises:
        RateLimitExceededException, BadCredentialsException: as GitHub reports them.
        GitHubQuotaError: a rate limit reported as a bare 429 or a secondary 403.
        ContributorCountError: any other error, after retrying transient ones
            (5xx, connection errors) CONTRIBUTOR_RETRIES times.
    """
    attempt = 0
    while True:
        try:
            return sum(1 for user in repo.get_contributors(anon=False) if not _is_bot(user))
        except (RateLimitExceededException, BadCredentialsException):
            raise
        except GithubException as e:
            if _is_contributors_too_large(e):
                logger.info(f"{repo.full_name}: contributor list too large to fetch; count unknown")
                return None
            _raise_if_rate_limited(e, bool(reveal_secret(token)))
            error: Exception = e
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            error = e

        if not _is_transient(error) or attempt >= CONTRIBUTOR_RETRIES:
            reason = _describe_error(error, token)
            logger.warning(f"Could not fetch contributors for {repo.full_name}: {reason}")
            raise ContributorCountError(reason) from error
        time.sleep(CONTRIBUTOR_BACKOFF_SECONDS * 2 ** attempt)
        attempt += 1


# Kept so existing imports keep working.
_build_query = build_query
_get_contributor_count = count_contributors


# ─── Result counts ────────────────────────────────────────────────────────────

def _is_rate_limit(exc: GithubException) -> bool:
    """Primary or secondary rate limit, however PyGithub or GitHub reports it."""
    if isinstance(exc, RateLimitExceededException) or exc.status == 429:
        return True
    if exc.status == 403 and isinstance(exc.data, dict):
        return "rate limit" in str(exc.data.get("message", "")).lower()
    return False


def _raise_if_rate_limited(exc: GithubException, authenticated: bool) -> None:
    """Turn a rate limit into GitHubQuotaError (with the reset time) so the pipeline and UI show it."""
    if _is_rate_limit(exc):
        raise GitHubQuotaError(
            build_exhausted_message(reset_time_from_exception(exc), authenticated)
        ) from exc


def _search_total(g: Github, query: str) -> tuple[int, bool]:
    """
    (total_count, incomplete_results) for a repository search.

    PyGithub's results.totalCount is capped at 1000 for search, so the true
    total is read from the raw response. per_page=1 keeps the response small.
    """
    _, data = g.requester.requestJsonAndCheck(
        "GET", "/search/repositories", parameters={"q": query, "per_page": 1}
    )
    return int(data["total_count"]), bool(data.get("incomplete_results", False))


def count_results(query: str, github_token: Union[str, SecretStr, None] = None) -> int:
    """Number of repositories GitHub reports for a search query (not capped at 1000)."""
    try:
        return _search_total(make_client(github_token), query)[0]
    except GithubException as e:
        _raise_if_rate_limited(e, bool(reveal_secret(github_token)))
        raise


# ─── Date windows ─────────────────────────────────────────────────────────────

def _split_windows(
    count: Callable[[date, date], int],
    start: date,
    end: date,
    report: DiscoveryReport,
) -> Iterator[tuple[date, date, int]]:
    """
    Yield (start, end, total) for whole-day windows covering [start, end] with
    no gap or overlap, each holding at most 1000 results, oldest first.

    A window over the limit is split at its midpoint day. A single day over the
    limit cannot be split: it is yielded as is (GitHub returns its 1000
    most-starred repositories) and recorded in report.truncated_windows.
    Empty windows are skipped.
    """
    total = count(start, end)
    if total == 0:
        return
    if total <= GITHUB_SEARCH_RESULT_LIMIT:
        yield start, end, total
        return
    if start == end:
        logger.warning(
            f"{total} repositories were created on {start}; GitHub search returns at most "
            f"{GITHUB_SEARCH_RESULT_LIMIT} per query and a single day cannot be split further. "
            f"Only the {GITHUB_SEARCH_RESULT_LIMIT} most-starred of that day are collected."
        )
        report.truncated_windows.append(
            {"start": start.isoformat(), "end": end.isoformat(), "total_count": total}
        )
        report.complete = False
        yield start, end, total
        return
    mid = start + timedelta(days=(end - start).days // 2)
    yield from _split_windows(count, start, mid, report)
    yield from _split_windows(count, mid + timedelta(days=1), end, report)


def _search(g: Github, query: str, authenticated: bool) -> Iterator[Repository]:
    """Iterate a search sorted by stars, turning a rate limit on any page into GitHubQuotaError."""
    try:
        yield from g.search_repositories(query=query, sort="stars", order="desc")
    except GithubException as e:
        _raise_if_rate_limited(e, authenticated)
        raise


def _to_metadata(
    repo: Repository,
    language: str,
    min_contributors: int,
    github_token: Union[str, SecretStr, None],
    report: Optional[DiscoveryReport] = None,
) -> Optional[RepoMetadata]:
    """
    Fetch the per-repo details; None if the repo is skipped.

    A repo is skipped when it has too few contributors, or when its
    contributors could not be fetched (recorded in report.excluded). A repo
    whose contributor list is too large to fetch is kept with contributors=None
    (recorded in report.contributors_unknown).
    """
    # Contributor count check — this costs at least one API call per repo
    try:
        contributor_count = count_contributors(repo, github_token)
    except ContributorCountError as e:
        if report is not None:
            report.excluded.append({"repo": repo.full_name, "reason": e.reason})
            report.complete = False
        return None

    if contributor_count is None:
        if report is not None:
            report.contributors_unknown.append(repo.full_name)
    elif contributor_count < min_contributors:
        logger.debug(f"Skipping {repo.full_name}: {contributor_count} contributors < {min_contributors}")
        return None

    # Get last commit date
    try:
        last_commit = repo.get_commits()[0].commit.committer.date
        last_commit_str = last_commit.isoformat()
    except (RateLimitExceededException, BadCredentialsException):
        raise
    except (GithubException, IndexError):
        last_commit_str = repo.pushed_at.isoformat() if repo.pushed_at else "unknown"

    license_spdx = None
    if repo.license:
        license_spdx = repo.license.spdx_id

    logger.info(f"Discovered: {repo.full_name} ({repo.stargazers_count} stars, "
                f"{'unknown number of' if contributor_count is None else contributor_count} contributors)")
    return RepoMetadata(
        full_name=repo.full_name,
        url=repo.html_url,
        stars=repo.stargazers_count,
        contributors=contributor_count,
        last_commit_date=last_commit_str,
        default_branch=repo.default_branch,
        license_spdx=license_spdx,
        language=language,
        collected_at=datetime.now(timezone.utc).isoformat(),
        # Topics come with the search payload; get_topics() would cost a request per repo
        topics=list(repo.topics),
    )


def discover_repos(
    language: str,
    start_date: date,
    end_date: date,
    min_stars: int = 10,
    min_contributors: int = 1,
    max_repos: Optional[int] = None,
    github_token: Union[str, SecretStr, None] = None,
    report: Optional[DiscoveryReport] = None,
) -> Iterator[RepoMetadata]:
    """
    Generator that yields RepoMetadata for repos matching all filters.

    GitHub search returns at most 1000 results per query, so:

    * max_repos <= 1000: one query. The first 1000 results by stars are the
      exact global top N.
    * otherwise: the creation-date range is split recursively at the midpoint
      day into whole-day windows (no gap, no overlap) until each holds at most
      1000 results, using the true total_count of each window.
      With a cap above 1000, candidates from every window are collected and
      ranked by stars (then full_name), and the first max_repos that pass the
      contributor filter are yielded: the exact global top by stars.
      With no cap, repos are yielded window by window, oldest window first
      (lazily; stars order holds only within a window).
      A single day with more than 1000 repos cannot be split; its 1000
      most-starred are collected and the day is recorded in
      report.truncated_windows.

    Args:
        language:         Programming language (e.g. "Python")
        start_date:       Only include repos created on or after this date.
                          E.g. an LLM's release date for contamination-aware
                          evaluation, or any reference date to bound a corpus
                          to a development era. This bounds repository age
                          only; it does not verify where individual snippets
                          came from.
        end_date:         Only include repos created on or before this date.
        min_stars:        Minimum star count
        min_contributors: Minimum unique contributor count (bots are not counted).
                          A repository whose contributor list GitHub refuses to
                          list because it is too large passes, with
                          contributors=None.
        max_repos:        If set, stop after yielding this many repos (must be >= 0)
        github_token:     GitHub PAT. Without a token, GitHub allows 60 REST requests per hour and 10 search requests per minute.
        report:           Optional DiscoveryReport that is filled in as discovery
                          runs; read it after the generator is consumed or raises.

    Yields:
        RepoMetadata for each qualifying repo

    Raises:
        GitHubQuotaError: a primary or secondary rate limit was hit while searching
            or counting. Rate limits in per-repo calls (contributors, commits)
            propagate as RateLimitExceededException.
        GithubException: any other GitHub error. In every error case the report,
            if given, is marked incomplete.

    A repository whose contributors cannot be fetched after retries is skipped
    and listed in report.excluded; one with a too-large contributor list is
    kept and listed in report.contributors_unknown.
    """
    if max_repos is not None and max_repos < 0:
        raise ValueError(f"max_repos must be >= 0 or None, got {max_repos}")

    if report is None:
        report = DiscoveryReport()
    if max_repos == 0:
        return

    g = make_client(github_token)
    authenticated = bool(reveal_secret(github_token))

    try:
        if max_repos is not None and max_repos <= GITHUB_SEARCH_RESULT_LIMIT:
            yield from _discover_single_query(
                g, authenticated, language, start_date, end_date, min_stars,
                min_contributors, max_repos, github_token, report,
            )
        else:
            yield from _discover_windows(
                g, authenticated, language, start_date, end_date, min_stars,
                min_contributors, max_repos, github_token, report,
            )
    except Exception:
        report.complete = False
        raise


def _discover_single_query(
    g, authenticated, language, start_date, end_date, min_stars,
    min_contributors, max_repos, github_token, report,
) -> Iterator[RepoMetadata]:
    query = build_query(language, start_date, end_date, min_stars)
    logger.info(f"GitHub search query: {query}")
    report.slices += 1

    yielded = scanned = 0
    for repo in _search(g, query, authenticated):
        scanned += 1
        metadata = _to_metadata(repo, language, min_contributors, github_token, report)
        if metadata is None:
            continue
        yield metadata
        yielded += 1
        if yielded >= max_repos:
            logger.info(f"Reached max_repos cap of {max_repos}. Stopping discovery.")
            report.total_matching = scanned
            return
        # Respect rate limits — sleep briefly between contributor fetches
        time.sleep(0.5)

    report.total_matching = scanned
    if scanned >= GITHUB_SEARCH_RESULT_LIMIT:
        logger.warning(
            f"Scanned all {GITHUB_SEARCH_RESULT_LIMIT} search results GitHub returns for this "
            f"query but only {yielded} of the requested {max_repos} repositories passed the "
            f"filters; more may exist beyond them. Lower min_contributors or narrow the dates."
        )
        report.complete = False


def _discover_windows(
    g, authenticated, language, start_date, end_date, min_stars,
    min_contributors, max_repos, github_token, report,
) -> Iterator[RepoMetadata]:
    counts: dict[tuple[date, date], int] = {}

    def count(lo: date, hi: date) -> int:
        if (lo, hi) not in counts:
            query = build_query(language, lo, hi, min_stars)
            try:
                total, incomplete = _search_total(g, query)
            except GithubException as e:
                _raise_if_rate_limited(e, authenticated)
                raise
            if incomplete:
                logger.warning(
                    f"GitHub flagged the result count for {lo}..{hi} as incomplete "
                    f"(the search timed out); some repositories may be missing."
                )
                report.incomplete_results = True
                report.complete = False
            counts[(lo, hi)] = total
        return counts[(lo, hi)]

    report.total_matching = total = count(start_date, end_date)
    if total > GITHUB_SEARCH_RESULT_LIMIT:
        logger.warning(
            f"GitHub search matches {total} repositories but returns at most "
            f"{GITHUB_SEARCH_RESULT_LIMIT} per query; splitting {start_date}..{end_date} "
            f"into creation-date windows."
        )
    else:
        logger.info(f"GitHub search matches {total} repositories.")

    def window_repos() -> Iterator[Repository]:
        for lo, hi, window_total in _split_windows(count, start_date, end_date, report):
            query = build_query(language, lo, hi, min_stars)
            logger.info(f"GitHub search query ({window_total} results): {query}")
            report.slices += 1
            yield from _search(g, query, authenticated)

    if max_repos is None:
        repos: Iterator[Repository] = window_repos()
    else:
        # Cap above 1000: rank every candidate so the cut is the global top by stars.
        candidates = {repo.full_name: repo for repo in window_repos()}
        repos = iter(sorted(candidates.values(), key=lambda r: (-r.stargazers_count, r.full_name)))

    yielded = 0
    for repo in repos:
        metadata = _to_metadata(repo, language, min_contributors, github_token, report)
        if metadata is None:
            continue
        yield metadata
        yielded += 1
        if max_repos is not None and yielded >= max_repos:
            logger.info(f"Reached max_repos cap of {max_repos}. Stopping discovery.")
            return
        # Respect rate limits — sleep briefly between contributor fetches
        time.sleep(0.5)
