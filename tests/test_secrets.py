"""
tests/test_secrets.py

github_token and hf_token must never leak: not through repr/str, model_dump,
model_dump_json, the CLI output, or manifest.json. They must still reach the
GitHub client and the classifier loader. Only the network is mocked.
"""

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from github import Auth
from github.GithubException import GithubException
from pydantic import SecretStr
from typer.testing import CliRunner

from sieve.cli import app
from sieve.config import SIEVEConfig, reveal_secret
from sieve.core.discovery import make_client
from sieve.core.export import export_dataset
from sieve.core.quality import collect_metrics
from sieve.pipeline import _fetch_quota, run_pipeline

FAKE = "ghp_FAKE_TOKEN_123"
FIELDS = ["github_token", "hf_token"]

runner = CliRunner()

_RESOURCES = SimpleNamespace(
    core=SimpleNamespace(remaining=5000, limit=5000, reset=datetime(2030, 1, 1, tzinfo=timezone.utc)),
    search=SimpleNamespace(remaining=30),
)

_BASE = {"language": "Python", "start_date": "2024-01-01", "end_date": "2024-12-31"}

_SUMMARY = {
    "total_repos_discovered": 0,
    "total_repos_processed": 0,
    "total_repos_failed": 0,
    "total_functions": 0,
    "total_classes": 0,
    "failed_repos": [],
    "output_paths": {},
    "repo_stats": [],
}


def _cfg(field, **extra):
    return SIEVEConfig(
        language="Python",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        **{field: FAKE},
        **extra,
    )


def _write_config(tmp_path, field):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**_BASE, field: FAKE}))
    return path


@pytest.mark.parametrize("field", FIELDS)
class TestTokenNeverLeaks:
    def test_repr_and_str(self, field):
        cfg = _cfg(field)
        assert FAKE not in repr(cfg)
        assert FAKE not in str(cfg)

    def test_model_dump(self, field):
        cfg = _cfg(field)
        assert FAKE not in repr(cfg.model_dump())
        assert FAKE not in repr(cfg.model_dump(mode="json"))
        assert field not in cfg.model_dump()

    def test_model_dump_json(self, field):
        assert FAKE not in _cfg(field).model_dump_json()
        assert FAKE not in _cfg(field).model_dump_json(indent=2)

    def test_validate_config_stdout(self, field, tmp_path):
        result = runner.invoke(app, ["validate-config", str(_write_config(tmp_path, field))])
        assert result.exit_code == 0
        assert FAKE not in result.output

    def test_run_banner_stdout(self, field, tmp_path):
        with patch("sieve.cli.run_pipeline", return_value=_SUMMARY):
            result = runner.invoke(app, ["run", "--config", str(_write_config(tmp_path, field))])
        assert result.exit_code == 0
        assert "starting pipeline" in result.output
        assert FAKE not in result.output

    def test_manifest_from_export_dataset(self, field, tmp_path):
        cfg = _cfg(field, output_dir=str(tmp_path / "out"))
        export_dataset([], [], cfg.output_dir, "jsonl", cfg.model_dump(mode="json"), [])
        assert FAKE not in (tmp_path / "out" / "manifest.json").read_text()

    def test_manifest_from_full_pipeline(self, field, tmp_path):
        cfg = _cfg(field, output_dir=str(tmp_path / "out"))
        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline.discover_repos", return_value=iter([])):
            run_pipeline(cfg)
        assert FAKE not in (tmp_path / "out" / "manifest.json").read_text()

    def test_still_readable_for_the_call_that_needs_it(self, field):
        value = getattr(_cfg(field), field)
        assert isinstance(value, SecretStr)
        assert value.get_secret_value() == FAKE


class TestTokenInputs:
    def test_plain_string_in_config_json(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({**_BASE, "github_token": FAKE, "hf_token": FAKE}))
        with patch("sieve.cli.run_pipeline", return_value=_SUMMARY) as mock_run:
            result = runner.invoke(app, ["run", "--config", str(path)])
        assert result.exit_code == 0
        cfg = mock_run.call_args.args[0]
        assert reveal_secret(cfg.github_token) == FAKE
        assert reveal_secret(cfg.hf_token) == FAKE

    _ARGS = ["run", "--language", "Python", "--start-date", "2024-01-01", "--end-date", "2024-12-31"]

    def test_github_token_env_var(self):
        with patch("sieve.cli.run_pipeline", return_value=_SUMMARY) as mock_run:
            result = runner.invoke(app, self._ARGS, env={"GITHUB_TOKEN": FAKE})
        assert result.exit_code == 0
        assert reveal_secret(mock_run.call_args.args[0].github_token) == FAKE
        assert FAKE not in result.output

    def test_github_token_flag(self):
        with patch("sieve.cli.run_pipeline", return_value=_SUMMARY) as mock_run:
            result = runner.invoke(app, self._ARGS + ["--github-token", FAKE], env={"GITHUB_TOKEN": ""})
        assert result.exit_code == 0
        assert reveal_secret(mock_run.call_args.args[0].github_token) == FAKE
        assert FAKE not in result.output

    def test_empty_token_means_unauthenticated(self):
        assert reveal_secret(SIEVEConfig(**_BASE, github_token="").github_token) is None
        assert not SIEVEConfig(**_BASE, github_token="").github_token

    def test_reveal_secret_accepts_plain_str_secret_and_none(self):
        assert reveal_secret(FAKE) == FAKE
        assert reveal_secret(SecretStr(FAKE)) == FAKE
        assert reveal_secret(None) is None


class TestTokenStillUsed:
    def test_quota_lookup_passes_token_to_github_client(self, monkeypatch):
        monkeypatch.setattr("sieve.pipeline.make_client", make_client)
        cfg = _cfg("github_token")
        with patch("sieve.core.discovery.Github") as mock_gh:
            mock_gh.return_value.get_rate_limit.return_value.resources = _RESOURCES
            quota = _fetch_quota(cfg.github_token)

        assert quota is not None and quota.authenticated
        auth = mock_gh.call_args.kwargs["auth"]
        assert isinstance(auth, Auth.Token)
        assert auth.token == FAKE

    def test_pipeline_discovery_passes_token_to_github_client(self, tmp_path):
        cfg = _cfg("github_token", output_dir=str(tmp_path / "out"))
        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.core.discovery.Github") as mock_gh:
            mock_gh.return_value.search_repositories.return_value = MagicMock(
                totalCount=0, __iter__=lambda self: iter([])
            )
            run_pipeline(cfg)

        auth = mock_gh.call_args.kwargs["auth"]
        assert isinstance(auth, Auth.Token)
        assert auth.token == FAKE

    def test_no_token_means_no_auth(self, tmp_path):
        cfg = SIEVEConfig(**_BASE, output_dir=str(tmp_path / "out"))
        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.core.discovery.Github") as mock_gh:
            mock_gh.return_value.search_repositories.return_value = MagicMock(
                totalCount=0, __iter__=lambda self: iter([])
            )
            run_pipeline(cfg)
        assert mock_gh.call_args.kwargs["auth"] is None

    def test_quality_metrics_client_gets_token(self):
        with patch("sieve.core.quality.Github") as mock_gh:
            mock_gh.return_value.get_repo.side_effect = GithubException(404, {}, {})
            collect_metrics("o/r", "MIT", 3, None, github_token=SecretStr(FAKE))
        auth = mock_gh.call_args.kwargs["auth"]
        assert isinstance(auth, Auth.Token)
        assert auth.token == FAKE

    def test_classifier_load_gets_hf_token(self, tmp_path):
        cfg = _cfg("hf_token", annotate_llm_score=True, output_dir=str(tmp_path / "out"))
        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline.discover_repos", return_value=iter([])), \
             patch("sieve.models.classifier.LLMCodeClassifier.load") as mock_load:
            run_pipeline(cfg)
        assert reveal_secret(mock_load.call_args.kwargs["hf_token"]) == FAKE
