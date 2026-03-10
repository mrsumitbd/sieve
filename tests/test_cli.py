"""
tests/test_cli.py

Unit tests for the SIEVE CLI using Typer's CliRunner.
The pipeline itself is mocked — these tests cover argument parsing,
validation, error handling, and output formatting only.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sieve.cli import app

runner = CliRunner()

# ─── Minimal pipeline summary returned by the mock ────────────────────────────

_FAKE_SUMMARY = {
    "total_repos_discovered": 3,
    "total_repos_processed": 3,
    "total_repos_failed": 0,
    "total_functions": 12,
    "total_classes": 4,
    "failed_repos": [],
    "output_paths": {
        "functions_jsonl": "/tmp/sieve_output/functions.jsonl",
        "classes_jsonl": "/tmp/sieve_output/classes.jsonl",
        "manifest": "/tmp/sieve_output/manifest.json",
    },
    "repo_stats": [],
}


# ─── run command ──────────────────────────────────────────────────────────────

class TestRunCommand:
    def test_run_with_inline_args(self):
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY):
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
                "--min-stars", "50",
            ])
        assert result.exit_code == 0
        assert "Pipeline complete" in result.output

    def test_run_shows_summary_table(self):
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY):
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
            ])
        assert "Functions Extracted" in result.output
        assert "12" in result.output

    def test_run_shows_output_paths(self):
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY):
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
            ])
        assert "functions_jsonl" in result.output

    def test_run_missing_language_exits_1(self):
        result = runner.invoke(app, [
            "run",
            "--start-date", "2024-01-01",
            "--end-date", "2025-01-01",
        ])
        assert result.exit_code == 1
        assert "--language is required" in result.output

    def test_run_missing_start_date_exits_1(self):
        result = runner.invoke(app, [
            "run",
            "--language", "Python",
        ])
        assert result.exit_code == 1
        assert "--start-date is required" in result.output

    def test_run_invalid_date_exits_1(self):
        result = runner.invoke(app, [
            "run",
            "--language", "Python",
            "--start-date", "not-a-date",
        ])
        assert result.exit_code == 1
        assert "Invalid date format" in result.output

    def test_run_pipeline_exception_exits_1(self):
        with patch("sieve.cli.run_pipeline", side_effect=RuntimeError("network down")):
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
            ])
        assert result.exit_code == 1
        assert "Pipeline failed" in result.output

    def test_run_from_config_file(self, tmp_path):
        config_data = {
            "language": "Python",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "min_stars": 10,
            "min_contributors": 1,
            "output_dir": str(tmp_path / "out"),
            "export_format": "jsonl",
            "granularity": ["function", "class"],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY):
            result = runner.invoke(app, ["run", "--config", str(config_file)])
        assert result.exit_code == 0

    def test_run_with_failed_repos_shows_warning(self):
        summary_with_failures = {
            **_FAKE_SUMMARY,
            "failed_repos": ["owner/broken-repo"],
            "total_repos_failed": 1,
        }
        with patch("sieve.cli.run_pipeline", return_value=summary_with_failures):
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
            ])
        assert "owner/broken-repo" in result.output

    def test_run_progress_callback_fires(self):
        """Verify the progress_callback passed to run_pipeline actually prints."""
        captured_messages = []

        def fake_pipeline(config, progress_callback=None):
            if progress_callback:
                progress_callback("Processing repo 1/3", 1, 3)
            return _FAKE_SUMMARY

        with patch("sieve.cli.run_pipeline", side_effect=fake_pipeline):
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
            ])
        assert "[1/3]" in result.output

    def test_run_end_date_defaults_when_omitted(self):
        """When --end-date is omitted, today's date should be used without error."""
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
            ])
        assert result.exit_code == 0
        called_config = mock.call_args[0][0]
        assert called_config.end_date == date.today()

    def test_run_optional_flags_passed_through(self):
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
            result = runner.invoke(app, [
                "run",
                "--language", "Java",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
                "--require-tests",
                "--max-repos", "10",
                "--export-format", "parquet",
            ])
        assert result.exit_code == 0
        cfg = mock.call_args[0][0]
        assert cfg.language == "Java"
        assert cfg.require_tests is True
        assert cfg.max_repos == 10
        assert cfg.export_format == "parquet"


# ─── validate-config command ──────────────────────────────────────────────────

class TestValidateConfigCommand:
    def test_valid_config_exits_0(self, tmp_path):
        config_data = {
            "language": "Python",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "output_dir": str(tmp_path),
            "export_format": "jsonl",
            "granularity": ["function"],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = runner.invoke(app, ["validate-config", str(config_file)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_invalid_config_exits_1(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text(json.dumps({"language": "Rust"}))  # unsupported language

        result = runner.invoke(app, ["validate-config", str(config_file)])
        assert result.exit_code == 1
        assert "invalid" in result.output.lower()

    def test_missing_file_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, ["validate-config", str(tmp_path / "nonexistent.json")])
        assert result.exit_code != 0

    def test_valid_config_prints_json(self, tmp_path):
        config_data = {
            "language": "JavaScript",
            "start_date": "2024-06-01",
            "end_date": "2025-01-01",
            "output_dir": str(tmp_path),
            "export_format": "both",
            "granularity": ["function", "class"],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = runner.invoke(app, ["validate-config", str(config_file)])
        assert result.exit_code == 0
        assert "JavaScript" in result.output