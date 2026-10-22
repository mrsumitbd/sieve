"""
tests/test_e2e.py

End-to-end tests for run_pipeline.
GitHub discovery and git clone are mocked — the real extraction,
deduplication, and export chain runs against local synthetic code.
"""

import json
import shutil
import textwrap
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sieve.config import SIEVEConfig
from sieve.core.discovery import RepoMetadata
from sieve.pipeline import run_pipeline


# ─── Fixtures ─────────────────────────────────────────────────────────────────

_PYTHON_CODE = textwrap.dedent("""
    import os
    from pathlib import Path

    class FileReader:
        \"\"\"Reads files.\"\"\"
        def __init__(self, base: str):
            self.base = Path(base)

        def read(self, name: str) -> str:
            \"\"\"Read a file by name.\"\"\"
            return (self.base / name).read_text()

    def count_lines(text: str) -> int:
        \"\"\"Count non-empty lines.\"\"\"
        return sum(1 for l in text.splitlines() if l.strip())
""").strip()


def _make_repo_meta(name="test-owner/test-repo") -> RepoMetadata:
    return RepoMetadata(
        full_name=name,
        url=f"https://github.com/{name}",
        stars=100,
        contributors=5,
        last_commit_date="2024-06-01",
        default_branch="main",
        license_spdx="MIT",
        language="Python",
        collected_at="2024-06-01T00:00:00Z",
        topics=[],
    )


@pytest.fixture
def synthetic_repo(tmp_path):
    """Write synthetic Python source into a temp directory."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(_PYTHON_CODE)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text("def test_count(): pass\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  test:\n    steps:\n      - run: pytest\n"
    )
    return tmp_path


@pytest.fixture
def base_config(tmp_path):
    return SIEVEConfig(
        language="Python",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        min_stars=10,
        max_repos=1,
        granularity=["function", "class"],
        deduplicate=False,
        output_dir=str(tmp_path / "output"),
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_with_fake_repo(config, synthetic_repo):
    """
    Patch discover_repos to yield one fake RepoMetadata and
    _clone_repo to copy the synthetic_repo into the target dir instead of
    hitting GitHub.
    """
    meta = _make_repo_meta()

    def fake_clone(repo_full_name, target_dir):
        shutil.copytree(str(synthetic_repo), target_dir, dirs_exist_ok=True)
        return True

    with patch("sieve.pipeline.discover_repos", return_value=iter([meta])), \
         patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
        return run_pipeline(config)


def _write_tree(root: Path, files: dict) -> Path:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


def _run_on_repos(config, tmp_path, repos, fail_first_clone=()):
    """
    Run the pipeline over several fake repos.

    repos maps "owner/name" -> {relative path: file content}. Repos named in
    fail_first_clone fail their first clone (Pass 1's counting clone) and
    succeed afterwards. Returns (summary, progress messages).
    """
    sources = {
        name: _write_tree(tmp_path / "fake_repos" / name.replace("/", "__"), files)
        for name, files in repos.items()
    }
    metas = [_make_repo_meta(name) for name in repos]
    clone_calls = {}
    messages = []

    def fake_clone(repo_full_name, target_dir):
        clone_calls[repo_full_name] = clone_calls.get(repo_full_name, 0) + 1
        if repo_full_name in fail_first_clone and clone_calls[repo_full_name] == 1:
            return False
        shutil.copytree(str(sources[repo_full_name]), target_dir, dirs_exist_ok=True)
        return True

    with patch("sieve.pipeline.discover_repos", return_value=iter(metas)), \
         patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
        summary = run_pipeline(config, progress_callback=lambda m, c, t: messages.append(m))
    return summary, messages


def _read_jsonl(summary, key):
    path = summary["output_paths"].get(key)
    if path is None:
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _per_repo(records):
    counts = {}
    for r in records:
        counts[r["repo"]] = counts.get(r["repo"], 0) + 1
    return counts


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestPipelineSummaryShape:
    def test_summary_has_required_keys(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        for key in (
            "total_repos_processed", "total_functions", "total_classes",
            "total_repos_failed", "failed_repos", "output_paths", "repo_stats",
        ):
            assert key in summary

    def test_one_repo_processed(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert summary["total_repos_processed"] == 1
        assert summary["total_repos_failed"] == 0

    def test_functions_extracted(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert summary["total_functions"] > 0

    def test_classes_extracted(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert summary["total_classes"] > 0


class TestPipelineOutputFiles:
    def test_jsonl_files_written(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        paths = summary["output_paths"]
        assert Path(paths["functions_jsonl"]).exists()
        assert Path(paths["classes_jsonl"]).exists()

    def test_manifest_written(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert Path(summary["output_paths"]["manifest"]).exists()

    def test_manifest_function_count_matches_summary(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        manifest = json.loads(Path(summary["output_paths"]["manifest"]).read_text())
        assert manifest["summary"]["total_functions"] == summary["total_functions"]
        assert manifest["summary"]["total_classes"] == summary["total_classes"]


class TestPipelineDeduplication:
    def test_dedup_reduces_count(self, tmp_path, synthetic_repo):
        """The same code in two repos is kept once with dedup on, twice with it off."""
        def cfg(dedup, out):
            return SIEVEConfig(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                deduplicate=dedup,
                dedup_threshold=0.8,
                output_dir=str(tmp_path / out),
            )

        meta1 = _make_repo_meta("owner/repo1")
        meta2 = _make_repo_meta("owner/repo2")

        def fake_clone(repo_full_name, target_dir):
            shutil.copytree(str(synthetic_repo), target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta1, meta2])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary_no_dedup = run_pipeline(cfg(False, "no_dedup"))

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta1, meta2])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary_dedup = run_pipeline(cfg(True, "dedup"))

        # synthetic_repo holds 3 functions (count_lines, FileReader.__init__,
        # FileReader.read) and 1 class (FileReader); both repos are identical.
        assert summary_no_dedup["total_functions"] == 6
        assert summary_no_dedup["total_classes"] == 2
        assert summary_dedup["total_functions"] == 3
        assert summary_dedup["total_classes"] == 1


# Near-duplicate fixtures. Jaccard similarity of the token sets (identifiers
# and punctuation, string literals collapsed), computed independently of
# sieve's tokenizer:
#   total_price ~ total_cost      0.58   (renamed identifiers only)
#   binary_search vs either       0.23
#   ServerSettings ~ ClientSettings  0.60
#   Matrix vs either              0.13
_DEDUP_MODULE = '''\
def total_price(items, tax_rate):
    subtotal = 0
    for item in items:
        subtotal += item.price * item.quantity
    return subtotal * (1 + tax_rate)


def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def total_cost(entries, vat):
    amount = 0
    for entry in entries:
        amount += entry.price * entry.quantity
    return amount * (1 + vat)


class ServerSettings:
    host = "localhost"
    port = 8080
    timeout = 30
    retries = 3
    verbose = False


class Matrix:
    rows = []
    cols = None
    shape = (0, 0)
    dtype = float


class ClientSettings:
    hostname = "localhost"
    port = 8080
    timeout = 30
    attempts = 3
    verbose = False
'''


class TestPipelineDedupWiring:
    def _run(self, tmp_path, deduplicate, threshold):
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            deduplicate=deduplicate,
            dedup_threshold=threshold,
            output_dir=str(tmp_path / "out"),
        )
        summary, _ = _run_on_repos(
            config, tmp_path, {"owner/repo": {"mod.py": _DEDUP_MODULE}}
        )
        names = lambda key, field: [r[field] for r in _read_jsonl(summary, key)]
        return (
            sorted(names("functions_jsonl", "func_name")),
            sorted(names("classes_jsonl", "class_name")),
            summary,
        )

    def test_dedup_off_keeps_near_duplicates(self, tmp_path):
        funcs, classes, summary = self._run(tmp_path, deduplicate=False, threshold=0.4)
        assert funcs == ["binary_search", "total_cost", "total_price"]
        assert classes == ["ClientSettings", "Matrix", "ServerSettings"]
        assert (summary["total_functions"], summary["total_classes"]) == (3, 3)

    def test_dedup_on_drops_later_near_duplicate(self, tmp_path):
        # threshold 0.4 is below both near-duplicate pairs (0.58, 0.60) and
        # above every distinct pair (<= 0.23): the first of each pair survives.
        funcs, classes, summary = self._run(tmp_path, deduplicate=True, threshold=0.4)
        assert funcs == ["binary_search", "total_price"]
        assert classes == ["Matrix", "ServerSettings"]
        assert (summary["total_functions"], summary["total_classes"]) == (2, 2)

    def test_strict_threshold_keeps_what_lenient_threshold_drops(self, tmp_path):
        # threshold 0.95 is above both near-duplicate pairs, so nothing is dropped.
        funcs, classes, _ = self._run(tmp_path, deduplicate=True, threshold=0.95)
        assert funcs == ["binary_search", "total_cost", "total_price"]
        assert classes == ["ClientSettings", "Matrix", "ServerSettings"]


class TestPipelineRepoStats:
    def test_repo_stats_row_per_repo(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        assert len(summary["repo_stats"]) == 1

    def test_repo_stats_fields_present(self, base_config, synthetic_repo):
        summary = _run_with_fake_repo(base_config, synthetic_repo)
        rs = summary["repo_stats"][0]
        for field in ("repo", "stars", "contributors", "functions", "classes",
                      "test_suite_present", "license"):
            assert field in rs


class TestPipelineCloneFailure:
    def test_failed_clone_counted_in_failed_repos(self, base_config):
        meta = _make_repo_meta()

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta])), \
             patch("sieve.pipeline._clone_repo", return_value=False):
            summary = run_pipeline(base_config)

        assert summary["total_repos_failed"] == 1
        assert "test-owner/test-repo" in summary["failed_repos"]
        assert summary["total_functions"] == 0
        assert summary["total_classes"] == 0

    def test_partial_failure_still_exports_successful_repos(self, tmp_path, synthetic_repo):
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        meta_good = _make_repo_meta("owner/good-repo")
        meta_bad = _make_repo_meta("owner/bad-repo")

        def selective_clone(repo_full_name, target_dir):
            if "bad" in repo_full_name:
                return False
            shutil.copytree(str(synthetic_repo), target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta_good, meta_bad])), \
             patch("sieve.pipeline._clone_repo", side_effect=selective_clone):
            summary = run_pipeline(config)

        assert summary["total_repos_processed"] == 1
        assert summary["total_repos_failed"] == 1
        assert summary["total_functions"] > 0


class TestPipelineEngineeredOnly:
    def test_engineered_only_calls_apply_filters(self, tmp_path, synthetic_repo):
        from sieve.core.quality import RepoQualityMetrics

        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            engineered_only=True,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        meta = _make_repo_meta()

        passing_metrics = RepoQualityMetrics(
            full_name=meta.full_name,
            license_spdx="MIT",
            contributor_count=10,
            release_count=3,
            pull_request_count=50,
            issue_count=30,
            loc=1000,
            comment_lines=100,
            code_ratio=0.9,
            passes_stage1=True,
            passes_stage2=True,
        )
        passing_metrics.passes_stage3 = True

        def fake_clone(repo_full_name, target_dir):
            shutil.copytree(str(synthetic_repo), target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone), \
             patch("sieve.pipeline.collect_metrics", return_value=passing_metrics), \
             patch("sieve.pipeline.apply_filters", return_value=[passing_metrics]) as mock_filter:
            summary = run_pipeline(config)

        mock_filter.assert_called_once()
        assert summary["total_repos_processed"] == 1

    def test_engineered_only_excludes_failing_repos(self, tmp_path):
        from sieve.core.quality import RepoQualityMetrics

        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            engineered_only=True,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        meta = _make_repo_meta()

        failing_metrics = RepoQualityMetrics(
            full_name=meta.full_name,
            license_spdx=None,
            contributor_count=1,
            release_count=0,
            pull_request_count=0,
            issue_count=0,
            loc=0,
            comment_lines=0,
            code_ratio=0.0,
            passes_stage1=False,
            passes_stage2=False,
        )

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta])), \
             patch("sieve.pipeline._clone_repo", return_value=True), \
             patch("sieve.pipeline.collect_metrics", return_value=failing_metrics), \
             patch("sieve.pipeline.apply_filters", return_value=[failing_metrics]):
            summary = run_pipeline(config)

        assert summary["total_repos_processed"] == 0
        assert summary["total_functions"] == 0


# ─── _clone_repo ──────────────────────────────────────────────────────────────

class TestCloneRepo:
    def test_returns_true_on_success(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sieve.pipeline import _clone_repo

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("sieve.pipeline.subprocess.run", return_value=mock_result):
            assert _clone_repo("owner/repo", str(tmp_path / "repo")) is True

    def test_returns_false_on_nonzero_exit(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sieve.pipeline import _clone_repo

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: repository not found"
        with patch("sieve.pipeline.subprocess.run", return_value=mock_result):
            assert _clone_repo("owner/repo", str(tmp_path / "repo")) is False

    def test_returns_false_on_timeout(self, tmp_path):
        from unittest.mock import patch
        from sieve.pipeline import _clone_repo
        import subprocess

        with patch("sieve.pipeline.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("git", 300)):
            assert _clone_repo("owner/repo", str(tmp_path / "repo")) is False

    def test_returns_false_on_exception(self, tmp_path):
        from unittest.mock import patch
        from sieve.pipeline import _clone_repo

        with patch("sieve.pipeline.subprocess.run",
                   side_effect=OSError("git not found")):
            assert _clone_repo("owner/repo", str(tmp_path / "repo")) is False


# ─── Two-pass extraction with caps ───────────────────────────────────────────

class TestTwoPassExtraction:
    def test_max_functions_cap_respected(self, tmp_path, synthetic_repo):
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=2,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        summary = _run_with_fake_repo(config, synthetic_repo)
        assert summary["total_functions"] <= 2

    def test_max_classes_cap_respected(self, tmp_path, synthetic_repo):
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_classes=1,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        summary = _run_with_fake_repo(config, synthetic_repo)
        assert summary["total_classes"] <= 1

    def test_no_cap_extracts_all(self, tmp_path, synthetic_repo):
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=None,
            max_classes=None,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        summary = _run_with_fake_repo(config, synthetic_repo)
        assert summary["total_functions"] > 0

    def test_two_repos_stratified_allocation(self, tmp_path, synthetic_repo, tmp_repo):
        """With two repos and a function cap, both repos contribute."""
        import shutil

        repo2 = tmp_repo({
            "src/b.py": "def beta(): pass\ndef gamma(): pass\n",
        })

        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=2,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        meta1 = _make_repo_meta("owner/repo1")
        meta2 = _make_repo_meta("owner/repo2")

        def fake_clone(repo_full_name, target_dir):
            src = str(synthetic_repo) if "repo1" in repo_full_name else str(repo2)
            shutil.copytree(src, target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos",
                   return_value=iter([meta1, meta2])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary = run_pipeline(config)

        # repo1 has 3 functions (count_lines, __init__, read), repo2 has 2.
        # Cap 2 with 2 non-empty repos: each gets its guaranteed 1, nothing extra.
        assert summary["total_repos_processed"] == 2
        assert _per_repo(_read_jsonl(summary, "functions_jsonl")) == {
            "owner/repo1": 1, "owner/repo2": 1,
        }


def _n_functions(n: int) -> dict:
    return {"mod.py": "".join(f"def f{i}():\n    return {i}\n\n" for i in range(n))}


class TestStratifiedAllocationEndToEnd:
    """
    Per-repo record counts produced by the whole pipeline for repos of
    different sizes. Expected values follow from proportional allocation with
    a guaranteed 1 per non-empty repo (largest remainder for the rest):

      sizes (6, 2), cap 4:  each repo gets 1, extra 2 shared by spare 5 : 1
                            -> 1 + 10/6 and 1 + 2/6 -> floors 2 and 1,
                            one leftover unit goes to the larger remainder
                            (4/6 vs 2/6) -> 3 and 1.
      sizes (2, 6), cap 4:  the same by symmetry -> 1 and 3.
    """

    @pytest.mark.parametrize("repo_sizes", [
        [("owner/big", 6), ("owner/small", 2)],
        [("owner/small", 2), ("owner/big", 6)],
    ])
    def test_exact_per_repo_counts(self, tmp_path, repo_sizes):
        repos = {name: _n_functions(n) for name, n in repo_sizes}
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=4,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )

        summary, _ = _run_on_repos(config, tmp_path, repos)

        assert summary["total_functions"] == 4
        assert _per_repo(_read_jsonl(summary, "functions_jsonl")) == {
            "owner/big": 3, "owner/small": 1,
        }

    def test_first_repo_does_not_take_whole_cap(self, tmp_path):
        repos = {"owner/first": _n_functions(6), "owner/second": _n_functions(6)}
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=4,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )

        summary, _ = _run_on_repos(config, tmp_path, repos)

        # Equal repos split the cap evenly.
        assert _per_repo(_read_jsonl(summary, "functions_jsonl")) == {
            "owner/first": 2, "owner/second": 2,
        }


# ─── LLM score annotation ─────────────────────────────────────────────────────

class TestLLMScoreAnnotation:
    _REPOS = {"owner/repo": {"mod.py": _PYTHON_CODE}}

    @staticmethod
    def _score(snippets, batch_size=64):
        # Score depends only on the snippet, so a record carrying another
        # record's score is detectable.
        return [len(snippet) / 1000 for snippet in snippets]

    def _config(self, tmp_path, annotate):
        return SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            annotate_llm_score=annotate,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )

    def _records(self, summary):
        records = _read_jsonl(summary, "functions_jsonl") + _read_jsonl(summary, "classes_jsonl")
        # _PYTHON_CODE: count_lines, FileReader.__init__, FileReader.read + class FileReader
        assert len(records) == 4
        return records

    def test_enabled_scores_every_record_with_its_own_score(self, tmp_path):
        classifier = MagicMock()
        classifier.score_batch.side_effect = self._score

        with patch("sieve.models.classifier.LLMCodeClassifier.load",
                   return_value=classifier):
            summary, _ = _run_on_repos(self._config(tmp_path, True), tmp_path, self._REPOS)

        for record in self._records(summary):
            assert record["llm_score"] == len(record["source_code"]) / 1000

    def test_disabled_leaves_scores_unset_and_never_loads_model(self, tmp_path):
        with patch("sieve.models.classifier.LLMCodeClassifier.load") as load:
            summary, _ = _run_on_repos(self._config(tmp_path, False), tmp_path, self._REPOS)

        load.assert_not_called()
        assert all(r["llm_score"] is None for r in self._records(summary))

    def test_model_load_failure_continues_without_scores(self, tmp_path):
        with patch("sieve.models.classifier.LLMCodeClassifier.load",
                   side_effect=Exception("Model unavailable")):
            summary, messages = _run_on_repos(self._config(tmp_path, True), tmp_path, self._REPOS)

        assert all(r["llm_score"] is None for r in self._records(summary))
        assert any("LLM scoring failed" in m and "Model unavailable" in m for m in messages)


# ─── Phase 4b: final safety-net cap (NOT Pass 3 — separate mechanism) ────────
# Phase 4b (pipeline.py "Final corpus size caps") is a final _stratified_sample()
# truncation that runs *after* Pass 3, independent of it.

class TestFinalCapEnforcement:
    def test_final_cap_applied_when_overshoot(self, tmp_path, synthetic_repo):
        """Caps hold end to end. (Phase 4b itself is exercised in TestPhase4bSafetyNet:
        this scenario never overshoots, so that code is not reached here.)"""
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=1,
            max_classes=1,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        summary = _run_with_fake_repo(config, synthetic_repo)
        assert summary["total_functions"] <= 1
        assert summary["total_classes"] <= 1


class TestPhase4bSafetyNet:
    """
    A repo whose Pass 1 counting clone fails (a transient error) has no
    allocation, so Pass 2 extracts it uncapped and the corpus overshoots the
    cap. Phase 4b must trim it back, proportionally by repo.

    Setup: repo1 (3 functions, 1 class) is counted in Pass 1; repo2
    (5 functions, 3 attribute-only classes) fails that first clone.
      functions: cap 2 -> repo1 allocated 2, repo2 uncapped 5      => 7 > 2
        trim 7 -> 2: repo1 2*2/7 = 0.57, repo2 2*5/7 = 1.43. repo2 takes its
        floor 1, the leftover slot goes to the larger remainder (repo1 0.57
        vs repo2 0.43) -> repo1 1, repo2 1.
      classes:   cap 1 -> repo1 allocated 1, repo2 uncapped 3      => 4 > 1
        trim 4 -> 1: repo1 1*1/4 = 0.25, repo2 1*3/4 = 0.75, no floors;
        the one slot goes to repo2.
    """

    _REPOS = {
        "owner/repo1": {"mod.py": _PYTHON_CODE},
        "owner/repo2": {
            "mod.py": "".join(f"def g{i}():\n    return {i}\n\n" for i in range(5))
                      + "".join(f"class C{i}:\n    x = {i}\n\n" for i in range(3)),
        },
    }

    def _run(self, tmp_path):
        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=2,
            max_classes=1,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )
        return _run_on_repos(
            config, tmp_path, self._REPOS, fail_first_clone={"owner/repo2"}
        )

    def test_overshoot_is_trimmed_to_cap_proportionally(self, tmp_path):
        summary, _ = self._run(tmp_path)

        assert summary["total_functions"] == 2
        assert summary["total_classes"] == 1
        assert _per_repo(_read_jsonl(summary, "functions_jsonl")) == {
            "owner/repo1": 1, "owner/repo2": 1,
        }
        assert _per_repo(_read_jsonl(summary, "classes_jsonl")) == {"owner/repo2": 1}

    def test_trim_is_reported(self, tmp_path):
        _, messages = self._run(tmp_path)

        assert "Applying final function cap: 7 → 2" in messages
        assert "Applying final class cap: 4 → 1" in messages


# ─── Pass 3 shortfall redistribution ─────────────────────────────────────────
#
# Shortfall (pipeline.py line 409: `if func_shortfall > 0 or class_shortfall > 0`)
# only occurs when a repo's *actual* Pass 2 yield falls short of its own
# stratified allocation from Pass 1. Since every repo's allocation is derived
# from its own Pass-1 count, simply giving repos "generous supply" (as in the
# test below) means every repo satisfies its own quota and the sum already
# equals the target after Pass 1+2 alone -- Pass 3 never runs. Genuinely
# forcing a shortfall requires a repo whose real per-repo yield diverges from
# what Pass 1 counted for it (e.g. a repo that fails during Pass 2's clone
# after being successfully counted in Pass 1, or one that legitimately
# returns less content on a second read) -- that's what the two new tests
# below construct.

class TestPass3Shortfall:
    def test_two_repo_allocation_meets_cap_without_pass3(self, tmp_path, synthetic_repo, tmp_repo):
        """
        Sanity check only: with generous, consistent per-repo supply, the
        stratified Pass 1+2 allocation already sums to the target and Pass 3
        never has to run. (Renamed from `test_shortfall_filled_from_other_repos`
        -- its old name/docstring implied Pass 3 engagement that does not
        actually occur in this scenario; see the two tests below for that.)
        """
        repo2 = tmp_repo({
            "src/b.py": "\n".join(
                [f"def func_{i}(): return {i}" for i in range(20)]
            ) + "\n",
        })

        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=5,
            max_classes=None,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )

        meta1 = _make_repo_meta("owner/repo1")
        meta2 = _make_repo_meta("owner/repo2")

        def fake_clone(repo_full_name, target_dir):
            src = str(synthetic_repo) if "repo1" in repo_full_name else str(repo2)
            shutil.copytree(src, target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos",
                   return_value=iter([meta1, meta2])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary = run_pipeline(config)

        assert summary["total_functions"] <= 5

    def test_step3a_surplus_cache_fills_shortfall_after_clone_failure(
        self, tmp_path, synthetic_repo, tmp_repo,
    ):
        """
        repo1 is successfully cloned and counted in Pass 1 (so it receives a
        real stratified allocation) but its Pass-2 clone attempt fails --
        contributing 0 despite Pass 1 having counted real content for it.
        This creates a genuine shortfall. repo2 has generous surplus beyond
        its own allocation, cached in Pass 2. Pass 3 Step 3a should pull the
        shortfall from that cache without any re-clone.
        """
        repo2 = tmp_repo({
            "src/b.py": "\n".join(
                [f"def func_{i}(): return {i}" for i in range(20)]
            ) + "\n",
        })

        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=5,
            max_classes=None,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )

        meta1 = _make_repo_meta("owner/repo1")
        meta2 = _make_repo_meta("owner/repo2")

        clone_calls = {"repo1": 0}

        def fake_clone(repo_full_name, target_dir):
            if "repo1" in repo_full_name:
                clone_calls["repo1"] += 1
                if clone_calls["repo1"] == 1:
                    # Pass 1's counting clone succeeds (repo1 gets allocated
                    # a real, nonzero share of the cap based on this).
                    shutil.copytree(str(synthetic_repo), target_dir, dirs_exist_ok=True)
                    return True
                # Pass 2's extraction clone fails -- e.g. a transient error.
                return False
            shutil.copytree(str(repo2), target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos",
                   return_value=iter([meta1, meta2])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary = run_pipeline(config)

        # repo1 contributed nothing (Pass 2 clone failed) but the cap was
        # still hit exactly -- proof the shortfall was filled from repo2's
        # cached surplus (Step 3a), not from repo1.
        assert summary["total_functions"] == 5
        assert "owner/repo1" in summary["failed_repos"]

    def test_step3b_reclone_recovers_remaining_capacity_when_cache_insufficient(
        self, tmp_path,
    ):
        """
        Single repo. Its Pass-1 counting clone sees 8 real functions (so it
        is allocated a cap of 6), but its Pass-2 extraction clone only
        exposes the first 2 of those 8 (e.g. a partial/incomplete checkout)
        -- leaving no surplus cached and a shortfall of 4. Step 3a's surplus
        cache is empty, so Step 3b must re-clone with a higher cap; this
        third clone exposes the full 8, letting extraction recover the
        remaining 4 genuinely new functions.
        """
        small_dir = tmp_path / "repo1_small"
        big_dir   = tmp_path / "repo1_big"
        small_dir.mkdir()
        big_dir.mkdir()

        # `big` is `small`'s two functions plus 6 more appended after --
        # same file_path and start_line for the first two, so the pipeline's
        # existing-key dedup correctly recognizes them as already-seen and
        # only counts the trailing 6 as new.
        small_code = "\n".join(f"def func_{i}(): return {i}" for i in range(2)) + "\n"
        big_code   = "\n".join(f"def func_{i}(): return {i}" for i in range(8)) + "\n"
        (small_dir / "main.py").write_text(small_code)
        (big_dir   / "main.py").write_text(big_code)

        config = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_functions=6,
            max_classes=None,
            deduplicate=False,
            output_dir=str(tmp_path / "out"),
        )

        meta1 = _make_repo_meta("owner/repo1")

        clone_calls = {"n": 0}

        def fake_clone(repo_full_name, target_dir):
            clone_calls["n"] += 1
            # Call 1: Pass 1 counting -> full 8-function version.
            # Call 2: Pass 2 extraction -> only the 2-function version.
            # Call 3+: Pass 3 Step 3b re-clone -> full 8-function version again.
            src = str(small_dir) if clone_calls["n"] == 2 else str(big_dir)
            shutil.copytree(src, target_dir, dirs_exist_ok=True)
            return True

        with patch("sieve.pipeline.discover_repos", return_value=iter([meta1])), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone):
            summary = run_pipeline(config)

        # Only reachable if Step 3b actually re-cloned and pulled the
        # remaining functions in -- Step 3a alone had nothing to offer.
        assert summary["total_functions"] == 6
        assert clone_calls["n"] >= 3