"""
tests/test_export.py

Integration tests for export_dataset.
Exercises JSONL, Parquet, and manifest output without network access.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from sieve.core.export import export_dataset
from sieve.core.extraction import _extract_python

# Expected columns in every function/class record
FUNCTION_COLUMNS = {
    "repo", "file_path", "language", "func_name", "parameters",
    "return_annotation", "docstring", "source_code", "signature",
    "used_imports", "start_line", "end_line", "is_method",
    "parent_class", "decorators", "llm_score", "llm_score_truncated",
}
CLASS_COLUMNS = {
    "repo", "file_path", "language", "class_name", "parent_classes",
    "method_names", "method_count", "has_constructor", "docstring",
    "source_code", "skeleton", "decorators", "used_imports",
    "start_line", "end_line", "llm_score", "llm_score_truncated",
}

_SAMPLE_CODE = """
import os

class Greeter:
    \"\"\"Say hello.\"\"\"
    def __init__(self, name: str):
        self.name = name
    def greet(self) -> str:
        return f"Hello, {self.name}"

def shout(text: str) -> str:
    \"\"\"Return uppercased text.\"\"\"
    return text.upper()
""".strip()

_CONFIG = {"language": "Python", "start_date": str(date(2024, 1, 1))}
_REPO_META = [{"full_name": "owner/repo", "stars": 100}]


@pytest.fixture
def sample_records():
    return _extract_python(_SAMPLE_CODE, "main.py", "owner/repo")


class TestJsonlExport:
    def test_jsonl_files_created(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        assert Path(paths["functions_jsonl"]).exists()
        assert Path(paths["classes_jsonl"]).exists()

    def test_row_counts_correct(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)

        fn_lines = [l for l in Path(paths["functions_jsonl"]).read_text().splitlines() if l]
        cl_lines = [l for l in Path(paths["classes_jsonl"]).read_text().splitlines() if l]
        assert len(fn_lines) == len(funcs)
        assert len(cl_lines) == len(classes)

    def test_all_function_columns_present(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        record = json.loads(Path(paths["functions_jsonl"]).read_text().splitlines()[0])
        assert FUNCTION_COLUMNS.issubset(record.keys())

    def test_all_class_columns_present(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        record = json.loads(Path(paths["classes_jsonl"]).read_text().splitlines()[0])
        assert CLASS_COLUMNS.issubset(record.keys())

    def test_source_code_preserved(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        records = [json.loads(l) for l in Path(paths["functions_jsonl"]).read_text().splitlines() if l]
        shout = next(r for r in records if r["func_name"] == "shout")
        assert "return text.upper()" in shout["source_code"]

    def test_list_fields_are_lists(self, tmp_path, sample_records):
        """List fields must be native JSON arrays, not stringified."""
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        record = json.loads(Path(paths["functions_jsonl"]).read_text().splitlines()[0])
        assert isinstance(record["parameters"], list)
        assert isinstance(record["used_imports"], list)
        assert isinstance(record["decorators"], list)


class TestParquetExport:
    def test_parquet_files_created(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "parquet", _CONFIG, _REPO_META)
        assert Path(paths["functions_parquet"]).exists()
        assert Path(paths["classes_parquet"]).exists()

    def test_parquet_row_counts_correct(self, tmp_path, sample_records):
        import polars as pl
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "parquet", _CONFIG, _REPO_META)
        df_fn = pl.read_parquet(paths["functions_parquet"])
        df_cl = pl.read_parquet(paths["classes_parquet"])
        assert len(df_fn) == len(funcs)
        assert len(df_cl) == len(classes)

    def test_parquet_all_columns_present(self, tmp_path, sample_records):
        import polars as pl
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "parquet", _CONFIG, _REPO_META)
        df = pl.read_parquet(paths["functions_parquet"])
        assert FUNCTION_COLUMNS.issubset(set(df.columns))

    def test_parquet_source_code_matches_jsonl(self, tmp_path, sample_records):
        import polars as pl
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "both", _CONFIG, _REPO_META)

        jsonl_records = {
            json.loads(l)["func_name"]: json.loads(l)["source_code"]
            for l in Path(paths["functions_jsonl"]).read_text().splitlines() if l
        }
        df = pl.read_parquet(paths["functions_parquet"])
        for row in df.iter_rows(named=True):
            assert row["source_code"] == jsonl_records[row["func_name"]]


class TestManifest:
    def test_manifest_created(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        assert Path(paths["manifest"]).exists()

    def test_manifest_counts_correct(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert manifest["summary"]["total_functions"] == len(funcs)
        assert manifest["summary"]["total_classes"] == len(classes)

    def test_manifest_has_sieve_version(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert "sieve_version" in manifest

    def test_manifest_sieve_version_matches_package(self, tmp_path, sample_records):
        """The manifest must report the installed package version, never a stale literal."""
        import sieve

        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert manifest["sieve_version"] == sieve.__version__

    def test_both_format_writes_all_four_files(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "both", _CONFIG, _REPO_META)
        assert "functions_jsonl" in paths
        assert "classes_jsonl" in paths
        assert "functions_parquet" in paths
        assert "classes_parquet" in paths

    def test_parquet_import_error_raises(self, tmp_path, sample_records):
        """If polars is unavailable, parquet export raises ImportError."""
        import sys
        from unittest.mock import patch
        funcs, classes = sample_records
        with patch.dict(sys.modules, {"polars": None}):
            with pytest.raises(Exception):
                export_dataset(funcs, classes, str(tmp_path / "out"), "parquet", _CONFIG, _REPO_META)



# ─── llm_score_truncated ──────────────────────────────────────────────────────

def _scored(records, flags):
    """Give each record a score and the next truncation flag (None = not scored)."""
    for rec, flag in zip(records, flags):
        rec.llm_score = None if flag is None else 0.5
        rec.llm_score_truncated = flag
    return records


class TestLlmScoreTruncatedExport:
    def test_unscored_records_export_null(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        rows = [json.loads(l) for l in Path(paths["functions_jsonl"]).read_text().splitlines() if l]
        assert rows and all(r["llm_score_truncated"] is None for r in rows)

    def test_jsonl_preserves_true_false_null(self, tmp_path, sample_records):
        funcs, classes = sample_records
        flags = [True, False, None, True, False][:len(funcs)]
        _scored(funcs, flags)
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        rows = [json.loads(l) for l in Path(paths["functions_jsonl"]).read_text().splitlines() if l]
        assert [r["llm_score_truncated"] for r in rows] == flags

    def test_parquet_column_is_nullable_boolean(self, tmp_path, sample_records):
        pl = pytest.importorskip("polars")
        funcs, classes = sample_records
        flags = [True, False, None, True, False][:len(funcs)]
        _scored(funcs, flags)
        paths = export_dataset(funcs, classes, str(tmp_path), "parquet", _CONFIG, _REPO_META)
        df = pl.read_parquet(paths["functions_parquet"])
        assert df.schema["llm_score_truncated"] == pl.Boolean
        assert df["llm_score_truncated"].to_list() == flags

    def test_parquet_all_unscored_still_writes_and_reads(self, tmp_path, sample_records):
        pl = pytest.importorskip("polars")
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "parquet", _CONFIG, _REPO_META)
        df = pl.read_parquet(paths["classes_parquet"])
        assert df["llm_score_truncated"].null_count() == len(classes)

    def test_manifest_counts_only_true(self, tmp_path, sample_records):
        funcs, classes = sample_records
        _scored(funcs, [True, False, None, True, True][:len(funcs)])
        _scored(classes, [True, False][:len(classes)])
        expected = sum(1 for r in funcs + classes if r.llm_score_truncated is True)
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert manifest["summary"]["total_llm_score_truncated"] == expected
        assert expected > 0

    def test_manifest_count_is_zero_when_nothing_scored(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert manifest["summary"]["total_llm_score_truncated"] == 0

    def test_manifest_keeps_existing_summary_keys(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        summary = json.loads(Path(paths["manifest"]).read_text())["summary"]
        assert {"total_repos", "total_functions", "total_classes"} <= set(summary)
        assert {"total_llm_scored", "total_llm_score_truncated"} <= set(summary)


# ─── total_llm_scored ─────────────────────────────────────────────────────────

class TestLlmScoredCount:
    def test_counts_records_with_a_non_null_score(self, tmp_path, sample_records):
        funcs, classes = sample_records
        _scored(funcs, [True, None, False, None, False][:len(funcs)])
        _scored(classes, [None, False][:len(classes)])
        expected = sum(1 for r in funcs + classes if r.llm_score is not None)
        assert 0 < expected < len(funcs) + len(classes)      # some scored, some not
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert manifest["summary"]["total_llm_scored"] == expected

    def test_zero_when_nothing_scored(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert manifest["summary"]["total_llm_scored"] == 0

    def test_score_of_zero_still_counts_as_scored(self, tmp_path, sample_records):
        """0.0 is a real score; only None means unscored."""
        funcs, classes = sample_records
        for r in funcs + classes:
            r.llm_score, r.llm_score_truncated = 0.0, False
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        manifest = json.loads(Path(paths["manifest"]).read_text())
        assert manifest["summary"]["total_llm_scored"] == len(funcs) + len(classes)

    def test_truncated_never_exceeds_scored(self, tmp_path, sample_records):
        funcs, classes = sample_records
        _scored(funcs, [True, True, None, True, False][:len(funcs)])
        paths = export_dataset(funcs, classes, str(tmp_path), "jsonl", _CONFIG, _REPO_META)
        summary = json.loads(Path(paths["manifest"]).read_text())["summary"]
        assert summary["total_llm_score_truncated"] <= summary["total_llm_scored"]

    def test_helper_counts_non_null_scores(self, sample_records):
        from sieve.core.export import count_llm_scored
        funcs, classes = sample_records
        assert count_llm_scored(funcs + classes) == 0
        _scored(funcs, [False, None, False, None, False][:len(funcs)])
        assert count_llm_scored(funcs) == sum(1 for r in funcs if r.llm_score is not None)
        assert count_llm_scored([]) == 0
