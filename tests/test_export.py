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
    "parent_class", "decorators", "llm_score",
}
CLASS_COLUMNS = {
    "repo", "file_path", "language", "class_name", "parent_classes",
    "method_names", "method_count", "has_constructor", "docstring",
    "source_code", "skeleton", "decorators", "used_imports",
    "start_line", "end_line", "llm_score",
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

    def test_both_format_writes_all_four_files(self, tmp_path, sample_records):
        funcs, classes = sample_records
        paths = export_dataset(funcs, classes, str(tmp_path), "both", _CONFIG, _REPO_META)
        assert "functions_jsonl" in paths
        assert "classes_jsonl" in paths
        assert "functions_parquet" in paths
        assert "classes_parquet" in paths