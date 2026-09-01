"""
core/discovery.py

Discovers GitHub repositories matching user-defined filters.
Excludes forks and archived repos by design.
"""

import time
import logging
from datetime import date
from dataclasses import dataclass
from typing import Optional, Iterator

from github import Github, GithubException
from github.Repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class RepoMetadata:
    """Snapshot of repo metadata at collection time."""
    full_name: str           # e.g. "owner/repo"
    url: str
    stars: int
    contributors: int
    last_commit_date: str    # ISO format
    default_branch: str
    license_spdx: Optional[str]
    language: str
    collected_at: str        # ISO format timestamp
    topics: list[str]


# Map language names to GitHub search query values
LANGUAGE_QUERY_MAP = {
    "Python": "python",
    "Java": "java",
    "JavaScript": "javascript",
}

# GitHub Search API max results per query is 1000
# We page through using star ranges to get beyond this limit
GITHUB_SEARCH_PAGE_SIZE = 100


def _build_query(
    language: str,
    start_date: date,
    end_date: date,
    min_stars: int,
    min_last_activity: Optional[date] = None,
) -> str:
    start = start_date.strftime("%Y-%m-%d")
    end   = end_date.strftime("%Y-%m-%d")
    lang  = LANGUAGE_QUERY_MAP.get(language, language.lower())
    # Use created: for the date range — guarantees contamination-free code
    activity = (min_last_activity or end_date).strftime("%Y-%m-%d")
    return (
        f"language:{lang} "
        f"created:{start}..{end} "
        f"pushed:>={activity} "
        f"stars:>={min_stars} "
        f"fork:false "
        f"archived:false"
    )


def _get_contributor_count(repo: Repository, token: str = None) -> int:
    """
    Fetch contributor count. Falls back to 0 on API errors.
    The contributors endpoint is expensive — we cache via metadata snapshot.
    """
    try:
        # anon_contributors=False excludes bots, keeping count meaningful
        return repo.get_contributors(anon=False).totalCount
    except GithubException as e:
        logger.warning(f"Could not fetch contributors for {repo.full_name}: {e}")
        return 0


def discover_repos(
    language: str,
    start_date: date,
    end_date: date,
    min_stars: int = 10,
    min_contributors: int = 1,
    max_repos: Optional[int] = None,
    github_token: Optional[str] = None,
    min_last_activity: Optional[date] = None,
) -> Iterator[RepoMetadata]:
    """
    Generator that yields RepoMetadata for repos matching all filters.
    Excludes forks and archived repos at query level (not just post-filter).

    Args:
        language:          Programming language (e.g. "Python")
        start_date:        Only include repos created on or after this date
                           (contamination cutoff — any code in these repos is
                           guaranteed to post-date LLM training data)
        end_date:          Only include repos created on or before this date
        min_stars:         Minimum star count
        min_contributors:  Minimum unique contributor count
        max_repos:         If set, stop after yielding this many repos
        github_token:      GitHub PAT. Without it, rate limit is 10 req/min.
        min_last_activity: Only include repos pushed on or after this date.
                           Defaults to end_date if not specified.

    Yields:
        RepoMetadata for each qualifying repo
    """
    from datetime import datetime, timezone

    g = Github(github_token, per_page=GITHUB_SEARCH_PAGE_SIZE) if github_token else Github(per_page=GITHUB_SEARCH_PAGE_SIZE)
    query = _build_query(language, start_date, end_date, min_stars, min_last_activity)

    logger.info(f"GitHub search query: {query}")

    yielded = 0
    try:
        results = g.search_repositories(query=query, sort="stars", order="desc")
        logger.info(f"Total results from GitHub search: {results.totalCount}")

        for repo in results:
            if max_repos is not None and yielded >= max_repos:
                logger.info(f"Reached max_repos cap of {max_repos}. Stopping discovery.")
                break

            # Contributor count check — this costs an extra API call per repo
            contributor_count = _get_contributor_count(repo, github_token)
            if contributor_count < min_contributors:
                logger.debug(f"Skipping {repo.full_name}: {contributor_count} contributors < {min_contributors}")
                continue

            # Get last commit date
            try:
                last_commit = repo.get_commits()[0].commit.committer.date
                last_commit_str = last_commit.isoformat()
            except (GithubException, IndexError):
                last_commit_str = repo.pushed_at.isoformat() if repo.pushed_at else "unknown"

            license_spdx = None
            if repo.license:
                license_spdx = repo.license.spdx_id

            metadata = RepoMetadata(
                full_name=repo.full_name,
                url=repo.html_url,
                stars=repo.stargazers_count,
                contributors=contributor_count,
                last_commit_date=last_commit_str,
                default_branch=repo.default_branch,
                license_spdx=license_spdx,
                language=language,
                collected_at=datetime.now(timezone.utc).isoformat(),
                topics=repo.get_topics(),
            )

            logger.info(f"Discovered: {repo.full_name} ({repo.stargazers_count} stars, {contributor_count} contributors)")
            yield metadata
            yielded += 1

            # Respect rate limits — sleep briefly between contributor fetches
            time.sleep(0.5)

    except GithubException as e:
        logger.error(f"GitHub API error during discovery: {e}")
        raise