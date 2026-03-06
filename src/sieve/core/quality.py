"""
core/quality.py

Engineered project filtering based on Xiao et al. (2025) / Munaiah et al. (2017).

Xiao et al. apply three stages to identify "engineered" software projects:

  Stage 1 — License filter:
      Exclude repos with no license or non-software licenses
      (CC BY 4.0, CC0, CC BY-SA 4.0, SIL OFL 1.1).

  Stage 2 — Hard exclusions:
      No GitHub releases, fewer than 2 contributors, archived repos.
      (archived is already excluded at query time in discovery.py)

  Stage 3 — Distributional filtering:
      Exclude repos in Q1 (bottom 25%) for pull requests, issues, and LOC
      per language. Also exclude repos whose code ratio
      (LOC / (LOC + comment_lines)) falls outside the 97% CI.

Because Stage 3 requires population-level statistics, this module
works in two passes:
  Pass 1: collect_metrics()  — fetches raw metrics for all candidate repos
  Pass 2: apply_filters()    — computes thresholds and returns qualifying repos

References:
  Xiao et al. (2025) "Self-Admitted GenAI Usage in Open-Source Software"
  Munaiah et al. (2017) "Curating GitHub for engineered software projects"
"""

import logging
import subprocess
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from github import Github, GithubException

logger = logging.getLogger(__name__)


# ─── cloc availability ────────────────────────────────────────────────────────

CLOC_INSTALL_INSTRUCTIONS = """
cloc (Count Lines of Code) is not installed or not on your PATH.
cloc is highly recommended for accurate LOC and comment counting across
all languages. Without it, SIEVE falls back to an AST-based counter
(Python: tokenize, Java/JavaScript: tree-sitter) which is accurate but
slower on large repositories.

Install cloc:
  macOS:    brew install cloc
  Ubuntu:   sudo apt install cloc
  Windows:  choco install cloc       (Chocolatey)
            winget install AlDanial.Cloc
  pip:      pip install cloc          (cross-platform wrapper)
  Manual:   https://github.com/AlDanial/cloc/releases
""".strip()


def check_cloc() -> bool:
    """
    Check whether cloc is available on PATH.
    Logs a clear warning with installation instructions if not found.
    Returns True if cloc is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["cloc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().splitlines()[0]
            logger.info(f"cloc found: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.warning(f"\n{'='*60}\n{CLOC_INSTALL_INSTRUCTIONS}\n{'='*60}")
    return False


# ─── Non-software licenses to exclude (Xiao et al.) ──────────────────────────

EXCLUDED_LICENSE_SPDX = {
    "CC-BY-4.0",
    "CC0-1.0",
    "CC-BY-SA-4.0",
    "OFL-1.1",        # SIL Open Font License 1.1
}

# Repos with no license at all are also excluded
EXCLUDE_NO_LICENSE = True


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RepoQualityMetrics:
    """Raw metrics for a single repo used in engineered project scoring."""
    full_name: str
    license_spdx: Optional[str]
    contributor_count: int
    release_count: int
    pull_request_count: int
    issue_count: int
    loc: int                    # Lines of code (non-comment, non-blank)
    comment_lines: int          # Comment lines
    code_ratio: float           # LOC / (LOC + comment_lines), 0 if no data
    passes_stage1: bool = False
    passes_stage2: bool = False
    passes_stage3: bool = False  # Set during apply_filters()

    @property
    def passes_all(self) -> bool:
        return self.passes_stage1 and self.passes_stage2 and self.passes_stage3


# ─── Stage 1: License filter ─────────────────────────────────────────────────

def _check_license(license_spdx: Optional[str]) -> bool:
    """Return True if license passes Stage 1."""
    if license_spdx is None or license_spdx == "NOASSERTION":
        return not EXCLUDE_NO_LICENSE
    return license_spdx not in EXCLUDED_LICENSE_SPDX


# ─── Stage 2: Hard exclusions ────────────────────────────────────────────────

def _check_hard_exclusions(contributor_count: int, release_count: int) -> bool:
    """Return True if repo passes Stage 2 hard exclusions."""
    if contributor_count < 2:
        return False
    if release_count == 0:
        return False
    return True


# ─── LOC counting ────────────────────────────────────────────────────────────

def _count_loc_cloc(repo_path: str) -> tuple[int, int]:
    """
    Count lines of code and comment lines.

    Strategy (in order of preference):
      1. cloc — external tool, handles all languages, very accurate
      2. AST-based fallback — uses Python tokenize for .py files and
         tree-sitter for .java / .js / .ts files. No regex anywhere.

    Returns:
        (loc, comment_lines)
    """
    try:
        result = subprocess.run(
            ["cloc", "--json", repo_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            total = data.get("SUM", {})
            return total.get("code", 0), total.get("comment", 0)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # AST-based fallback
    loc, comments = 0, 0
    root = Path(repo_path)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".py":
            f_loc, f_comments = _count_python_ast(path)
        elif path.suffix == ".java":
            f_loc, f_comments = _count_treesitter(path, "Java")
        elif path.suffix in (".js", ".ts", ".jsx", ".tsx"):
            f_loc, f_comments = _count_treesitter(path, "JavaScript")
        else:
            continue
        loc += f_loc
        comments += f_comments

    return loc, comments


def _count_python_ast(path: Path) -> tuple[int, int]:
    """
    Count code and comment lines in a Python file using the tokenize module.

    tokenize is Python's own lexer — it correctly handles:
      - Inline comments (x = 1  # comment)
      - Standalone comment lines
      - Multi-line strings (not counted as comments unless used as docstrings)
      - Continuation lines
      - Blank lines

    We count a line as a comment line if its first meaningful token is COMMENT.
    We count a line as a code line if it contains any non-COMMENT, non-NL,
    non-NEWLINE, non-ENCODING, non-INDENT, non-DEDENT token.
    """
    import tokenize
    import io

    try:
        source = path.read_bytes()
        tokens = list(tokenize.tokenize(io.BytesIO(source).readline))
    except Exception as e:
        logger.debug(f"tokenize failed for {path}: {e}")
        return 0, 0

    # Track which lines have comments and which have code
    comment_lines: set[int] = set()
    code_lines: set[int] = set()

    NON_CODE_TYPES = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
        tokenize.ENCODING, tokenize.INDENT, tokenize.DEDENT,
        tokenize.ENDMARKER,
    }

    for tok in tokens:
        lineno = tok.start[0]
        if tok.type == tokenize.COMMENT:
            comment_lines.add(lineno)
        elif tok.type not in NON_CODE_TYPES:
            code_lines.add(lineno)

    # A line with both a comment and code token counts as code (inline comment)
    pure_comment_lines = comment_lines - code_lines

    return len(code_lines), len(pure_comment_lines)


def _count_treesitter(path: Path, language: str) -> tuple[int, int]:
    """
    Count code and comment lines using tree-sitter node types.

    Comment node types per language:
      Java:       "line_comment", "block_comment"
      JavaScript: "comment" (covers both // and /* */)

    Any line covered by a comment node is a comment line.
    Any line that contains a non-comment, non-whitespace node is a code line.
    Lines with both (e.g. trailing comments) count as code.
    """
    try:
        from tree_sitter import Language as TSLanguage, Parser

        if language == "Java":
            import tree_sitter_java as tsjava
            lang = TSLanguage(tsjava.language())
            comment_types = {"line_comment", "block_comment"}
        elif language == "JavaScript":
            import tree_sitter_javascript as tsjavascript
            lang = TSLanguage(tsjavascript.language())
            comment_types = {"comment"}
        else:
            return 0, 0

        source = path.read_bytes()
        parser = Parser(lang)
        tree = parser.parse(source)

        comment_lines: set[int] = set()
        code_lines: set[int] = set()

        def walk(node):
            if node.type in comment_types:
                for lineno in range(node.start_point[0] + 1, node.end_point[0] + 2):
                    comment_lines.add(lineno)
            elif not node.is_named and node.child_count == 0:
                # Leaf punctuation/whitespace — skip
                pass
            elif node.is_named and node.child_count == 0:
                # Named leaf node = actual code token
                code_lines.add(node.start_point[0] + 1)
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        pure_comment_lines = comment_lines - code_lines
        return len(code_lines), len(pure_comment_lines)

    except Exception as e:
        logger.debug(f"tree-sitter LOC count failed for {path}: {e}")
        return 0, 0


# ─── Metrics Collection ───────────────────────────────────────────────────────

def collect_metrics(
    repo_full_name: str,
    license_spdx: Optional[str],
    contributor_count: int,
    repo_path: Optional[str],
    github_token: Optional[str] = None,
) -> RepoQualityMetrics:
    """
    Collect all quality metrics for a single repo.

    Args:
        repo_full_name: "owner/repo"
        license_spdx: SPDX license identifier from RepoMetadata
        contributor_count: Already fetched during discovery
        repo_path: Path to cloned repo (for LOC counting). None = skip LOC.
        github_token: GitHub PAT for API calls

    Returns:
        RepoQualityMetrics with Stage 1 and Stage 2 pass/fail populated.
        Stage 3 is left False — set by apply_filters() after population stats.
    """
    g = Github(github_token) if github_token else Github()

    release_count = 0
    pull_request_count = 0
    issue_count = 0

    try:
        repo = g.get_repo(repo_full_name)
        release_count = repo.get_releases().totalCount
        # Issues API returns both issues and PRs unless filtered
        pull_request_count = repo.get_pulls(state="all").totalCount
        issue_count = repo.get_issues(state="all").totalCount - pull_request_count
        issue_count = max(issue_count, 0)
    except GithubException as e:
        logger.warning(f"Could not fetch GitHub metrics for {repo_full_name}: {e}")

    loc, comment_lines = 0, 0
    if repo_path:
        loc, comment_lines = _count_loc_cloc(repo_path)

    total_lines = loc + comment_lines
    code_ratio = loc / total_lines if total_lines > 0 else 0.0

    passes_stage1 = _check_license(license_spdx)
    passes_stage2 = _check_hard_exclusions(contributor_count, release_count)

    return RepoQualityMetrics(
        full_name=repo_full_name,
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
        passes_stage3=False,
    )


# ─── Stage 3: Distributional Filtering ───────────────────────────────────────

def apply_filters(metrics_list: list[RepoQualityMetrics]) -> list[RepoQualityMetrics]:
    """
    Apply Stage 3 distributional filtering across the full candidate pool.

    Per Xiao et al.:
      - Exclude repos in Q1 (bottom 25%) for pull requests, issues, and LOC
      - Exclude repos whose code_ratio falls outside the 97% CI

    Only repos that already passed Stage 1 and Stage 2 are considered
    for Stage 3 statistics (so noise doesn't pollute thresholds).

    Mutates passes_stage3 in place and returns the full list.
    """
    qualifying = [m for m in metrics_list if m.passes_stage1 and m.passes_stage2]

    if not qualifying:
        logger.warning("No repos passed Stage 1+2 — Stage 3 thresholds cannot be computed.")
        return metrics_list

    prs    = np.array([m.pull_request_count for m in qualifying])
    issues = np.array([m.issue_count for m in qualifying])
    locs   = np.array([m.loc for m in qualifying])
    ratios = np.array([m.code_ratio for m in qualifying if m.code_ratio > 0])

    q1_pr    = np.percentile(prs, 25)
    q1_issue = np.percentile(issues, 25)
    q1_loc   = np.percentile(locs, 25)

    # 97% CI on code_ratio — equivalent to mean ± 2.17 * std
    if len(ratios) > 1:
        ratio_mean = ratios.mean()
        ratio_std  = ratios.std()
        ratio_lo   = ratio_mean - 2.17 * ratio_std
        ratio_hi   = ratio_mean + 2.17 * ratio_std
    else:
        ratio_lo, ratio_hi = 0.0, 1.0

    logger.info(
        f"Stage 3 thresholds — Q1 PRs: {q1_pr:.0f}, Q1 issues: {q1_issue:.0f}, "
        f"Q1 LOC: {q1_loc:.0f}, code_ratio CI: [{ratio_lo:.3f}, {ratio_hi:.3f}]"
    )

    for m in metrics_list:
        if not (m.passes_stage1 and m.passes_stage2):
            m.passes_stage3 = False
            continue

        fails_q1 = (
            m.pull_request_count < q1_pr or
            m.issue_count        < q1_issue or
            m.loc                < q1_loc
        )
        fails_ratio = (
            m.code_ratio > 0 and
            not (ratio_lo <= m.code_ratio <= ratio_hi)
        )

        m.passes_stage3 = not fails_q1 and not fails_ratio

    passed = sum(1 for m in metrics_list if m.passes_all)
    logger.info(
        f"Engineered project filter: {len(metrics_list)} candidates → "
        f"{passed} passed all three stages."
    )

    return metrics_list
