"""
tests/test_demo.py

End-to-end tests for the bundled demo dataset.
Verifies that load_demo_dataset() returns the expected shape,
that all JSONL/manifest files exist, and that the data is internally consistent.
"""

import json
from pathlib import Path

import pytest

from sieve.ui.app import load_demo_dataset


@pytest.fixture(scope="module")
def demo():
    return load_demo_dataset()


class TestDemoSummaryShape:
    def test_returns_dict(self, demo):
        assert isinstance(demo, dict)

    def test_required_keys_present(self, demo):
        for key in (
            "total_functions", "total_classes", "total_repos_processed",
            "total_repos_failed", "failed_repos", "output_paths",
            "repo_stats", "_is_demo",
        ):
            assert key in demo, f"Missing key: {key}"

    def test_is_demo_flag_set(self, demo):
        assert demo["_is_demo"] is True

    def test_function_count_positive(self, demo):
        assert demo["total_functions"] > 0

    def test_class_count_positive(self, demo):
        assert demo["total_classes"] > 0

    def test_failed_repos_empty(self, demo):
        assert demo["failed_repos"] == []

    def test_three_repos_processed(self, demo):
        assert demo["total_repos_processed"] == 3


class TestDemoFiles:
    def test_functions_jsonl_exists(self, demo):
        path = demo["output_paths"]["functions_jsonl"]
        assert Path(path).exists()

    def test_classes_jsonl_exists(self, demo):
        path = demo["output_paths"]["classes_jsonl"]
        assert Path(path).exists()

    def test_manifest_exists(self, demo):
        path = demo["output_paths"]["manifest"]
        assert Path(path).exists()

    def test_functions_count_matches_jsonl(self, demo):
        path = demo["output_paths"]["functions_jsonl"]
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        assert len(lines) == demo["total_functions"]

    def test_classes_count_matches_jsonl(self, demo):
        path = demo["output_paths"]["classes_jsonl"]
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        assert len(lines) == demo["total_classes"]


class TestDemoRepoStats:
    def test_repo_stats_is_list_of_three(self, demo):
        assert isinstance(demo["repo_stats"], list)
        assert len(demo["repo_stats"]) == 3

    def test_repo_stats_all_fields_present(self, demo):
        required = {"repo", "stars", "contributors", "functions", "classes",
                    "test_suite_present", "test_confidence", "license"}
        for rs in demo["repo_stats"]:
            assert required.issubset(rs.keys())

    def test_repo_stats_counts_sum_to_totals(self, demo):
        total_fns = sum(r["functions"] for r in demo["repo_stats"])
        total_cls = sum(r["classes"] for r in demo["repo_stats"])
        assert total_fns == demo["total_functions"]
        assert total_cls == demo["total_classes"]

    def test_known_repos_present(self, demo):
        repos = {r["repo"] for r in demo["repo_stats"]}
        assert "sieve-demo/data-structures" in repos
        assert "sieve-demo/algorithms" in repos
        assert "sieve-demo/text-utils" in repos


class TestDemoRecordSchema:
    def test_function_records_have_required_fields(self, demo):
        path = demo["output_paths"]["functions_jsonl"]
        required = {
            "repo", "file_path", "language", "func_name", "parameters",
            "return_annotation", "docstring", "source_code", "signature",
            "used_imports", "start_line", "end_line", "is_method",
            "parent_class", "decorators", "llm_score",
        }
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            assert required.issubset(record.keys())

    def test_class_records_have_required_fields(self, demo):
        path = demo["output_paths"]["classes_jsonl"]
        required = {
            "repo", "file_path", "language", "class_name", "parent_classes",
            "method_names", "method_count", "has_constructor", "docstring",
            "source_code", "skeleton", "decorators", "used_imports",
            "start_line", "end_line", "llm_score",
        }
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            assert required.issubset(record.keys())

    def test_llm_score_is_none(self, demo):
        """llm_score should be null — classifier not yet integrated."""
        path = demo["output_paths"]["functions_jsonl"]
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            assert record["llm_score"] is None

    def test_language_field_is_python(self, demo):
        path = demo["output_paths"]["functions_jsonl"]
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            assert record["language"] == "Python"