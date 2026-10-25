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
        """When --end-date is omitted, 30 days before today should be used."""
        from datetime import timedelta
        _default_end = date.today() - timedelta(days=30)
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
            result = runner.invoke(app, [
                "run",
                "--language", "Python",
                "--start-date", "2024-01-01",
            ])
        assert result.exit_code == 0
        called_config = mock.call_args[0][0]
        assert called_config.end_date == _default_end

    def test_run_optional_flags_passed_through(self):
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
            result = runner.invoke(app, [
                "run",
                "--language", "Java",
                "--start-date", "2024-01-01",
                "--end-date", "2025-01-01",
                "--max-repos", "10",
                "--export-format", "parquet",
            ])
        assert result.exit_code == 0
        cfg = mock.call_args[0][0]
        assert cfg.language == "Java"
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

# ─── Rate limits and config errors ────────────────────────────────────────────

class TestRunErrorHandling:
    _ARGS = ["run", "--language", "Python", "--start-date", "2024-01-01", "--end-date", "2025-01-01"]

    def test_bad_config_file_gives_one_line_error(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text(json.dumps({
            "language": "Python",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "min_stars": -5,
        }))

        with patch("sieve.cli.run_pipeline") as mock_run:
            result = runner.invoke(app, ["run", "--config", str(config_file)])

        assert result.exit_code == 1
        mock_run.assert_not_called()
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("Error: invalid config: min_stars:")
        assert "pydantic" not in result.output.lower()

    def test_bad_inline_value_gives_one_line_error(self):
        with patch("sieve.cli.run_pipeline"):
            result = runner.invoke(app, self._ARGS + ["--dedup-threshold", "1.5"])

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert "dedup_threshold" in lines[0]

    def test_validate_config_error_is_one_line(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text(json.dumps({
            "language": "Python",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "dedup_threshold": 2,
        }))

        result = runner.invoke(app, ["validate-config", str(config_file)])

        assert result.exit_code == 1
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert "dedup_threshold" in lines[0]

    def test_rate_limit_exception_exits_1_with_helper_message(self):
        from github.GithubException import RateLimitExceededException
        from sieve.core.github_limits import build_exhausted_message, reset_time_from_exception

        exc = RateLimitExceededException(
            403, {"message": "API rate limit exceeded"}, {"X-RateLimit-Reset": "1893508320"}
        )
        with patch("sieve.cli.run_pipeline", side_effect=exc):
            result = runner.invoke(app, self._ARGS, env={"GITHUB_TOKEN": ""})

        assert result.exit_code == 1
        expected = build_exhausted_message(reset_time_from_exception(exc), authenticated=False)
        assert f"Error: {expected}" in result.output
        assert "--github-token" in result.output
        assert "Traceback" not in result.output

    def test_rate_limit_exception_with_token_omits_token_advice(self):
        from github.GithubException import RateLimitExceededException

        exc = RateLimitExceededException(
            403, {"message": "API rate limit exceeded"}, {"X-RateLimit-Reset": "1893508320"}
        )
        with patch("sieve.cli.run_pipeline", side_effect=exc):
            result = runner.invoke(app, self._ARGS + ["--github-token", "ghp_example"])

        assert result.exit_code == 1
        assert "Error: GitHub API rate limit reached" in result.output
        assert "public_repo" not in result.output

    def test_quota_error_exits_1_with_its_message(self):
        from sieve.core.github_limits import GitHubQuotaError

        with patch("sieve.cli.run_pipeline", side_effect=GitHubQuotaError("quota too low")):
            result = runner.invoke(app, self._ARGS)

        assert result.exit_code == 1
        assert "Error: quota too low" in result.output

    def test_partial_result_prints_warning(self):
        partial = {**_FAKE_SUMMARY, "rate_limited": True, "rate_limit_reset": "2030-01-01T14:32:00+00:00"}
        with patch("sieve.cli.run_pipeline", return_value=partial):
            result = runner.invoke(app, self._ARGS, env={"GITHUB_TOKEN": ""})

        assert result.exit_code == 0
        assert "Results are partial" in result.output
        assert "Add a GitHub token" in result.output
        assert "14:32 UTC" in result.output

    def test_rejected_token_exits_1_with_one_line_message(self):
        from sieve.core.github_limits import BAD_TOKEN_MESSAGE, GitHubAuthError

        with patch("sieve.cli.run_pipeline", side_effect=GitHubAuthError()):
            result = runner.invoke(app, self._ARGS + ["--github-token", "ghp_expired"])

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        errors = [ln for ln in result.output.splitlines() if ln.startswith("Error:")]
        assert errors == [f"Error: {BAD_TOKEN_MESSAGE}"]


# ─── Strict config, caps, new flags, env tokens ───────────────────────────────

_RUN = ["run", "--language", "Python", "--start-date", "2024-01-01", "--end-date", "2025-01-01"]
_BASE_CFG = {"language": "Python", "start_date": "2024-01-01", "end_date": "2025-01-01"}


def _run_cfg(extra_args=(), env=None):
    with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
        result = runner.invoke(app, _RUN + list(extra_args), env={"GITHUB_TOKEN": "", "HF_TOKEN": "", **(env or {})})
    assert result.exit_code == 0, result.output
    return mock.call_args[0][0], result


def _write_cfg(tmp_path, **extra):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({**_BASE_CFG, **extra}))
    return f


class TestStrictConfigFile:
    def test_run_rejects_unknown_key_readably(self, tmp_path):
        f = _write_cfg(tmp_path, max_repoz=5)
        with patch("sieve.cli.run_pipeline") as mock_run:
            result = runner.invoke(app, ["run", "--config", str(f)])
        assert result.exit_code == 1
        mock_run.assert_not_called()
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert lines == ["Error: invalid config: max_repoz: unknown config key"]

    def test_validate_config_rejects_unknown_key_readably(self, tmp_path):
        f = _write_cfg(tmp_path, hf_tokn="x", dedup=True)
        result = runner.invoke(app, ["validate-config", str(f)])
        assert result.exit_code == 1
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert "hf_tokn: unknown config key" in lines[0]
        assert "dedup: unknown config key" in lines[0]
        assert "Traceback" not in result.output

    def test_unknown_key_error_does_not_echo_its_value(self, tmp_path):
        f = _write_cfg(tmp_path, github_tokn="ghp_SECRET_VALUE")
        for args in (["run", "--config", str(f)], ["validate-config", str(f)]):
            result = runner.invoke(app, args)
            assert result.exit_code == 1
            assert "github_tokn" in result.output
            assert "ghp_SECRET_VALUE" not in result.output


class TestCapsViaCli:
    @pytest.mark.parametrize("flag", ["--max-repos", "--max-functions", "--max-classes"])
    def test_zero_rejected(self, flag):
        with patch("sieve.cli.run_pipeline") as mock_run:
            result = runner.invoke(app, _RUN + [flag, "0"])
        assert result.exit_code == 1
        mock_run.assert_not_called()
        assert flag.lstrip("-").replace("-", "_") in result.output
        assert ">= 1" in result.output

    def test_omitted_caps_are_none(self):
        cfg, _ = _run_cfg()
        assert (cfg.max_repos, cfg.max_functions, cfg.max_classes) == (None, None, None)

    def test_positive_caps_accepted(self):
        cfg, _ = _run_cfg(["--max-repos", "1", "--max-functions", "5", "--max-classes", "7"])
        assert (cfg.max_repos, cfg.max_functions, cfg.max_classes) == (1, 5, 7)

    @pytest.mark.parametrize("field", ["max_repos", "max_functions", "max_classes"])
    def test_zero_rejected_in_config_file(self, tmp_path, field):
        f = _write_cfg(tmp_path, **{field: 0})
        result = runner.invoke(app, ["validate-config", str(f)])
        assert result.exit_code == 1
        assert field in result.output and ">= 1" in result.output

    @pytest.mark.parametrize("field", ["max_repos", "max_functions", "max_classes"])
    def test_null_accepted_in_config_file(self, tmp_path, field):
        f = _write_cfg(tmp_path, **{field: None})
        assert runner.invoke(app, ["validate-config", str(f)]).exit_code == 0


class TestNewFlags:
    def test_deduplicate_default_is_off(self):
        cfg, _ = _run_cfg()
        assert cfg.deduplicate is False

    def test_deduplicate_flags(self):
        assert _run_cfg(["--deduplicate"])[0].deduplicate is True
        assert _run_cfg(["--no-deduplicate"])[0].deduplicate is False

    def test_granularity_default_is_both(self):
        assert _run_cfg()[0].granularity == ["function", "class"]

    @pytest.mark.parametrize("choice,expected", [
        ("function", ["function"]),
        ("class", ["class"]),
        ("both", ["function", "class"]),
    ])
    def test_granularity_mapping(self, choice, expected):
        assert _run_cfg(["--granularity", choice])[0].granularity == expected

    def test_granularity_invalid_choice_rejected(self):
        with patch("sieve.cli.run_pipeline") as mock_run:
            result = runner.invoke(app, _RUN + ["--granularity", "method"])
        assert result.exit_code != 0
        mock_run.assert_not_called()

    def test_annotate_llm_score_flags(self):
        assert _run_cfg()[0].annotate_llm_score is False
        assert _run_cfg(["--annotate-llm-score"])[0].annotate_llm_score is True
        assert _run_cfg(["--no-annotate-llm-score"])[0].annotate_llm_score is False

    def test_export_ast_flags(self):
        assert _run_cfg()[0].export_ast is False
        assert _run_cfg(["--export-ast"])[0].export_ast is True
        assert _run_cfg(["--no-export-ast"])[0].export_ast is False

    def test_no_hf_token_flag(self):
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
        assert "--hf-token" not in result.output
        assert "--granularity" in result.output
        assert "--annotate-llm-score" in result.output
        assert "--export-ast" in result.output


class TestEnvTokenFallback:
    GH = "ghp_ENV_GITHUB_123"
    HF = "hf_ENV_TOKEN_456"

    def _assert_hidden(self, result):
        assert self.GH not in result.output
        assert self.HF not in result.output

    def test_flags_path_reads_both_env_vars(self):
        from sieve.config import reveal_secret

        cfg, result = _run_cfg(env={"GITHUB_TOKEN": self.GH, "HF_TOKEN": self.HF})
        assert reveal_secret(cfg.github_token) == self.GH
        assert reveal_secret(cfg.hf_token) == self.HF
        self._assert_hidden(result)

    def test_flags_path_flag_beats_env(self):
        from sieve.config import reveal_secret

        cfg, result = _run_cfg(["--github-token", "ghp_FLAG"], env={"GITHUB_TOKEN": self.GH})
        assert reveal_secret(cfg.github_token) == "ghp_FLAG"
        assert "ghp_FLAG" not in result.output

    def test_config_path_reads_both_env_vars(self, tmp_path):
        from sieve.config import reveal_secret

        f = _write_cfg(tmp_path)
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
            result = runner.invoke(app, ["run", "--config", str(f)], env={"GITHUB_TOKEN": self.GH, "HF_TOKEN": self.HF})
        assert result.exit_code == 0, result.output
        cfg = mock.call_args[0][0]
        assert reveal_secret(cfg.github_token) == self.GH
        assert reveal_secret(cfg.hf_token) == self.HF
        self._assert_hidden(result)

    def test_config_path_file_value_beats_env(self, tmp_path):
        from sieve.config import reveal_secret

        f = _write_cfg(tmp_path, github_token="ghp_FILE", hf_token="hf_FILE")
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
            result = runner.invoke(app, ["run", "--config", str(f)], env={"GITHUB_TOKEN": self.GH, "HF_TOKEN": self.HF})
        cfg = mock.call_args[0][0]
        assert reveal_secret(cfg.github_token) == "ghp_FILE"
        assert reveal_secret(cfg.hf_token) == "hf_FILE"
        assert "ghp_FILE" not in result.output and "hf_FILE" not in result.output
        self._assert_hidden(result)

    def test_empty_file_value_falls_back_to_env(self, tmp_path):
        from sieve.config import reveal_secret

        f = _write_cfg(tmp_path, github_token="")
        with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
            runner.invoke(app, ["run", "--config", str(f)], env={"GITHUB_TOKEN": self.GH})
        assert reveal_secret(mock.call_args[0][0].github_token) == self.GH

    def test_unset_env_leaves_tokens_none(self):
        cfg, _ = _run_cfg()
        assert cfg.github_token is None and cfg.hf_token is None

    def test_not_in_manifest(self, tmp_path):
        f = _write_cfg(tmp_path, output_dir=str(tmp_path / "out"))
        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline.discover_repos", return_value=iter([])):
            result = runner.invoke(app, ["run", "--config", str(f)], env={"GITHUB_TOKEN": self.GH, "HF_TOKEN": self.HF})
        assert result.exit_code == 0, result.output
        self._assert_hidden(result)
        assert self.GH not in (tmp_path / "out" / "manifest.json").read_text()
        assert self.HF not in (tmp_path / "out" / "manifest.json").read_text()
