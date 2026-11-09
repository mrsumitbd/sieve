"""
tests/test_ui_home.py

Smoke tests for the token guidance and partial-result warning on the Home page,
using Streamlit's AppTest. The pipeline is never run.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

HOME = str(Path(__file__).resolve().parents[1] / "src" / "sieve" / "ui" / "Home.py")


def _app():
    return AppTest.from_file(HOME, default_timeout=30)


def _texts(elements):
    return [e.value for e in elements]


class TestHomeRateLimitGuidance:
    def test_no_token_shows_quota_hint(self):
        at = _app().run()
        assert not at.exception
        hints = [t for t in _texts(at.info) if t.startswith("No token:")]
        assert len(hints) == 1
        assert "60 API requests per hour" in hints[0]
        assert "roughly 12 repositories" in hints[0]

    def test_token_hides_quota_hint(self):
        at = _app().run()
        at.sidebar.text_input[0].set_value("ghp_example").run()
        assert not [t for t in _texts(at.info) if t.startswith("No token:")]

    def test_partial_summary_shows_warning(self):
        at = _app()
        at.session_state["summary"] = {
            "total_repos_discovered": 2, "total_repos_processed": 2, "total_repos_failed": 0,
            "total_functions": 0, "total_classes": 0, "failed_repos": [], "repo_stats": [],
            "output_paths": {}, "repo_metadata": [],
            "rate_limited": True, "rate_limit_reset": "2030-01-01T14:32:00+00:00",
        }
        at.run()
        warnings = [t for t in _texts(at.warning) if "Results are partial" in t]
        assert len(warnings) == 1
        assert "14:32 UTC" in warnings[0]
        assert "Add a GitHub token" in warnings[0]


class TestHomeTruncationVisibility:
    """Records scored on only their first 512 tokens must be visible in the UI."""

    @staticmethod
    def _summary(tmp_path, truncated=None, records=(), scored=None):
        import json
        paths = {}
        if records:
            f = tmp_path / "functions.jsonl"
            f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            paths["functions_jsonl"] = str(f)
        s = {
            "total_repos_discovered": 1, "total_repos_processed": 1, "total_repos_failed": 0,
            "total_functions": len(records), "total_classes": 0, "failed_repos": [],
            "repo_stats": [], "output_paths": paths, "repo_metadata": [],
        }
        if truncated is not None:
            s["total_llm_score_truncated"] = truncated
        if scored is not None:
            s["total_llm_scored"] = scored
        return s

    def test_run_summary_mentions_truncated_count(self, tmp_path):
        at = _app()
        at.session_state["summary"] = self._summary(tmp_path, truncated=7, scored=40)
        at.run()
        assert not at.exception
        notes = [t for t in _texts(at.caption) if "truncated" in t.lower()]
        assert len(notes) == 1 and "512" in notes[0]
        assert "7 of 40" in notes[0]            # truncated out of scored, not out of all records

    def test_run_summary_tolerates_missing_scored_count(self, tmp_path):
        at = _app()
        at.session_state["summary"] = self._summary(tmp_path, truncated=3)   # older summary dict
        at.run()
        assert not at.exception
        (note,) = [t for t in _texts(at.caption) if "truncated" in t.lower()]
        assert "3" in note

    @pytest.mark.parametrize("value", [0, None])
    def test_run_summary_is_silent_when_nothing_truncated(self, tmp_path, value):
        at = _app()
        at.session_state["summary"] = self._summary(tmp_path, truncated=value)
        at.run()
        assert not at.exception
        assert not [t for t in _texts(at.caption) if "truncated" in t.lower()]

    @staticmethod
    def _sample_markdown(tmp_path, **extra):
        record = {"repo": "o/r", "file_path": "a.py", "language": "Python",
                  "func_name": "f", "parameters": [], "source_code": "def f(): pass",
                  "used_imports": [], "start_line": 1, "end_line": 1, **extra}
        at = _app()
        at.session_state["summary"] = TestHomeTruncationVisibility._summary(tmp_path, records=[record])
        at.session_state["sample"] = ("Function/Method", record, 0, 1)
        at.run()
        assert not at.exception
        return [m.value for m in at.markdown if "LLM Score" in m.value]

    def test_sample_viewer_flags_a_truncated_score(self, tmp_path):
        (line,) = self._sample_markdown(tmp_path, llm_score=0.87, llm_score_truncated=True)
        assert "0.870" in line and "truncated" in line.lower() and "512" in line

    @pytest.mark.parametrize("extra", [
        {"llm_score": 0.87, "llm_score_truncated": False},
        {"llm_score": 0.87},                                 # file from before this field existed
        {"llm_score": None, "llm_score_truncated": None},
    ])
    def test_sample_viewer_unchanged_when_not_truncated(self, tmp_path, extra):
        (line,) = self._sample_markdown(tmp_path, **extra)
        assert "truncated" not in line.lower()
        assert ("0.870" in line) == (extra.get("llm_score") is not None)


class TestHomeWelcome:
    def test_token_step_is_optional_and_no_example_option(self):
        at = _app().run()
        assert not at.exception
        text = " ".join(_all_text(at))
        assert "Optionally add a GitHub Personal Access Token" in text
        assert "recommended for full runs" in text
        assert "Example Dataset" not in text
        assert "— or —" not in text

    def test_run_is_the_only_button(self):
        at = _app().run()
        assert [b.label for b in at.sidebar.button] == ["▶ Run SIEVE"]


# ─── Parameter widgets: same semantics as the CLI and SIEVEConfig ─────────────

CAP_REPOS = "Max Repos (0 = no cap)"
CAP_FUNCS = "Max Functions (0 = no cap)"
CAP_CLASSES = "Max Classes (0 = no cap)"
CHK_FUNCS = "Extract functions"
CHK_CLASSES = "Extract classes"
TGL_AST = "Export Full AST"


def _widget(at, kind, label):
    (w,) = [w for w in getattr(at.sidebar, kind) if w.label == label]
    return w


def _set(at, values):
    """Apply {label: value} to whichever sidebar widget carries that label.

    Sliders go last, after a rerun: the dedup slider is disabled until Deduplicate is on,
    and newer Streamlit refuses to set a widget that is still disabled in the last run.
    """
    kinds = ("number_input", "checkbox", "toggle", "slider", "selectbox")

    def widget(label):
        matches = [w for k in kinds for w in getattr(at.sidebar, k) if w.label == label]
        assert len(matches) == 1, f"expected one widget labelled {label!r}, found {len(matches)}"
        return matches[0]

    sliders = {k: v for k, v in values.items() if widget(k) in at.sidebar.slider}
    for label, value in values.items():
        if label not in sliders:
            widget(label).set_value(value)
    if sliders:
        at.run()
        for label, value in sliders.items():
            widget(label).set_value(value)


def _run_ui(values=None):
    """Set widgets, click Run with the pipeline stubbed out; return (AppTest, config or None, pipeline mock)."""
    from unittest.mock import patch

    at = _app().run()
    _set(at, values or {})
    captured = {}

    def fake(config, *a, **kw):
        captured["config"] = config
        raise RuntimeError("stop")

    with patch("sieve.pipeline.run_pipeline", side_effect=fake) as pipeline:
        at.sidebar.button[0].click().run()
    assert not at.exception
    return at, captured.get("config"), pipeline


class TestHomeDefaults:
    def test_cap_defaults(self):
        at = _app().run()
        values = {w.label: w.value for w in at.sidebar.number_input}
        assert values[CAP_REPOS] == 20
        assert values[CAP_FUNCS] == 30
        assert values[CAP_CLASSES] == 30

    def test_other_defaults_unchanged(self):
        at = _app().run()
        values = {w.label: w.value for w in at.sidebar.number_input}
        assert values["Minimum Stars"] == 50
        assert values["Minimum Contributors"] == 5

    def test_granularity_checkboxes_default_to_both(self):
        at = _app().run()
        assert {w.label: w.value for w in at.sidebar.checkbox} == {CHK_FUNCS: True, CHK_CLASSES: True}

    def test_no_minus_one_left_in_the_widgets(self):
        at = _app().run()
        for w in at.sidebar.number_input:
            assert w.min is None or w.min >= 0, f"{w.label!r} still accepts negative values"
            assert "-1" not in w.label
        assert "skip" not in " ".join(w.label.lower() for w in at.sidebar.number_input)

    def test_single_ast_toggle_replaces_the_two_old_ones(self):
        at = _app().run()
        toggles = {t.label: t.value for t in at.sidebar.toggle}
        assert toggles[TGL_AST] is False
        assert "Export AST-Derived Features" not in toggles
        assert "Include Full AST in Export" not in toggles

    def test_other_toggles_keep_labels_and_defaults(self):
        at = _app().run()
        toggles = {t.label: t.value for t in at.sidebar.toggle}
        for label in ("Engineered Projects Only", "Annotate LLM Score", "Deduplicate"):
            assert toggles[label] is False


class TestHomeGranularityWidgets:
    def test_unchecking_a_level_disables_only_its_cap(self):
        at = _app().run()
        at.sidebar.checkbox[0].uncheck().run()          # Extract functions
        assert _widget(at, "number_input", CAP_FUNCS).disabled
        assert not _widget(at, "number_input", CAP_CLASSES).disabled
        assert not _widget(at, "number_input", CAP_REPOS).disabled

    def test_unchecking_classes_disables_the_class_cap(self):
        at = _app().run()
        _widget(at, "checkbox", CHK_CLASSES).uncheck().run()
        assert _widget(at, "number_input", CAP_CLASSES).disabled
        assert not _widget(at, "number_input", CAP_FUNCS).disabled

    def test_rechecking_reenables_the_cap(self):
        at = _app().run()
        _widget(at, "checkbox", CHK_CLASSES).uncheck().run()
        _widget(at, "checkbox", CHK_CLASSES).check().run()
        assert not _widget(at, "number_input", CAP_CLASSES).disabled

    def test_both_unchecked_blocks_the_run_with_an_error(self):
        at, cfg, pipeline = _run_ui({CHK_FUNCS: False, CHK_CLASSES: False})
        assert cfg is None
        pipeline.assert_not_called()
        assert [e for e in at.main.error if "At least one" in e.value]
        assert at.session_state["summary"] is None

    def test_both_unchecked_error_is_visible_before_running(self):
        at = _app().run()
        _set(at, {CHK_FUNCS: False, CHK_CLASSES: False})
        at.run()
        assert not at.exception
        assert [e for e in at.error if "At least one" in e.value]

    def test_one_checked_shows_no_granularity_error(self):
        at = _app().run()
        _set(at, {CHK_CLASSES: False})
        at.run()
        assert not [e for e in at.error if "At least one" in e.value]


class TestHomeCapTranslation:
    """0 = no cap in the widgets becomes None in SIEVEConfig; 0 and -1 never reach the config."""

    @pytest.mark.parametrize("repos,funcs,classes,expected", [
        (0, 0, 0, (None, None, None)),
        (5, 10, 20, (5, 10, 20)),
        (1, 0, 3, (1, None, 3)),
        (0, 1, 0, (None, 1, None)),
    ])
    def test_valid_configs(self, repos, funcs, classes, expected):
        at, cfg, _ = _run_ui({CAP_REPOS: repos, CAP_FUNCS: funcs, CAP_CLASSES: classes})
        assert not [e for e in at.error if "Configuration error" in e.value]
        assert cfg is not None
        assert (cfg.max_repos, cfg.max_functions, cfg.max_classes) == expected
        assert cfg.granularity == ["function", "class"]

    @pytest.mark.parametrize("values,granularity,caps", [
        ({CHK_FUNCS: False}, ["class"], (None, 30)),
        ({CHK_CLASSES: False}, ["function"], (30, None)),
        # a disabled level's leftover cap value must not leak into the config
        ({CHK_FUNCS: False, CAP_FUNCS: 99, CAP_CLASSES: 4}, ["class"], (None, 4)),
        ({CHK_CLASSES: False, CAP_CLASSES: 99, CAP_FUNCS: 0}, ["function"], (None, None)),
    ])
    def test_unchecked_level_is_dropped_and_uncapped(self, values, granularity, caps):
        at, cfg, _ = _run_ui(values)
        assert cfg is not None
        assert cfg.granularity == granularity
        assert (cfg.max_functions, cfg.max_classes) == caps

    def test_no_zero_or_negative_cap_ever_reaches_the_config(self):
        for values in ({CAP_REPOS: 0, CAP_FUNCS: 0, CAP_CLASSES: 0}, {CHK_FUNCS: False}, {CHK_CLASSES: False}):
            _, cfg, _ = _run_ui(values)
            for cap in (cfg.max_repos, cfg.max_functions, cfg.max_classes):
                assert cap is None or cap >= 1


class TestHomeAstExport:
    def test_off_by_default(self):
        _, cfg, _ = _run_ui()
        assert cfg.export_ast is False

    def test_toggle_sets_export_ast(self):
        _, cfg, _ = _run_ui({TGL_AST: True})
        assert cfg.export_ast is True


class TestHomeConfigPreview:
    def test_preview_matches_the_config_that_runs(self):
        import json

        values = {CHK_FUNCS: False, CAP_CLASSES: 7, CAP_REPOS: 0, TGL_AST: True}
        at = _app().run()
        _set(at, values)
        at.run()
        preview = json.loads(at.json[0].value)
        _, cfg, _ = _run_ui(values)
        for key in ("max_repos", "max_functions", "max_classes", "granularity", "export_ast"):
            assert preview[key] == getattr(cfg, key), key


class TestHomeConfigGoesThroughSIEVEConfig:
    def test_every_field_the_ui_builds_is_a_config_field(self):
        from unittest.mock import patch

        from sieve.config import SIEVEConfig

        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return SIEVEConfig(**kwargs)      # extra="forbid" rejects any stale or misspelled key

        with patch("sieve.config.SIEVEConfig", side_effect=spy):
            _, cfg, _ = _run_ui({TGL_AST: True})
        assert cfg is not None
        assert seen and set(seen) <= set(SIEVEConfig.model_fields)

    def test_config_rejects_the_old_ui_key(self):
        from datetime import date

        from pydantic import ValidationError

        from sieve.config import SIEVEConfig

        with pytest.raises(ValidationError):
            SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1),
                        export_full_ast=True)


# ─── UI ↔ CLI parity ──────────────────────────────────────────────────────────

_TOKENS = {"github_token", "hf_token"}

# The UI's defaults differ from the CLI's (stars, contributors, caps), so every case spells them out.
_CLI_BASE = ["--language", "Python", "--start-date", "2024-01-01",
             "--min-stars", "50", "--min-contributors", "5"]
_R20 = ["--max-repos", "20"]            # the UI's default repo cap

_PARITY_CASES = [
    pytest.param({}, ["--max-functions", "30", "--max-classes", "30", "--granularity", "both", *_R20], id="defaults"),
    pytest.param({CHK_CLASSES: False},
                 ["--granularity", "function", "--max-functions", "30", *_R20], id="functions-only"),
    pytest.param({CHK_FUNCS: False, CAP_CLASSES: 7},
                 ["--granularity", "class", "--max-classes", "7", *_R20], id="classes-only-capped"),
    pytest.param({CHK_FUNCS: False, CAP_FUNCS: 50},       # disabled input's value is ignored
                 ["--granularity", "class", "--max-classes", "30", *_R20], id="disabled-cap-ignored"),
    pytest.param({CAP_REPOS: 0, CAP_FUNCS: 0, CAP_CLASSES: 0},
                 ["--granularity", "both"], id="no-caps"),
    pytest.param({CAP_REPOS: 3, CAP_FUNCS: 1, CAP_CLASSES: 0},
                 ["--granularity", "both", "--max-functions", "1", "--max-repos", "3"], id="mixed-caps"),
    pytest.param({TGL_AST: True},
                 ["--max-functions", "30", "--max-classes", "30", "--export-ast", *_R20], id="export-ast"),
    pytest.param({"Annotate LLM Score": True, "Engineered Projects Only": True},
                 ["--max-functions", "30", "--max-classes", "30", "--annotate-llm-score", "--engineered-only", *_R20],
                 id="llm-score-and-engineered"),
    pytest.param({"Deduplicate": True, "Dedup Similarity Threshold": 0.9, "Export Format": "both"},
                 ["--max-functions", "30", "--max-classes", "30", "--deduplicate",
                  "--dedup-threshold", "0.9", "--export-format", "both", *_R20], id="dedup-and-both-formats"),
    pytest.param({"Export Format": "parquet", CHK_CLASSES: False, TGL_AST: True, CAP_FUNCS: 0},
                 ["--granularity", "function", "--export-ast", "--export-format", "parquet", *_R20],
                 id="parquet-functions-ast"),
]


class TestUiCliParity:
    @pytest.mark.parametrize("ui_values,cli_args", _PARITY_CASES)
    def test_ui_config_equals_cli_config(self, ui_values, cli_args):
        from unittest.mock import patch

        from typer.testing import CliRunner

        from sieve.cli import app

        _, ui_cfg, _ = _run_ui(ui_values)
        assert ui_cfg is not None

        # The UI writes to a temp dir, so run the CLI against the same one.
        args = ["run", *_CLI_BASE, *cli_args, "--output-dir", ui_cfg.output_dir]
        with patch("sieve.cli.run_pipeline", return_value={
            "total_repos_discovered": 0, "total_repos_processed": 0, "total_repos_failed": 0,
            "total_functions": 0, "total_classes": 0, "failed_repos": [], "output_paths": {},
        }) as pipeline:
            result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        cli_cfg = pipeline.call_args[0][0]

        assert ui_cfg.model_dump(exclude=_TOKENS) == cli_cfg.model_dump(exclude=_TOKENS)


class TestHomeRejectedToken:
    def test_run_with_rejected_token_shows_message(self):
        from unittest.mock import patch

        from sieve.core.github_limits import BAD_TOKEN_MESSAGE, GitHubAuthError

        at = _app().run()
        at.sidebar.text_input[0].set_value("ghp_expired")
        with patch("sieve.pipeline.run_pipeline", side_effect=GitHubAuthError()):
            at.sidebar.button[0].click().run()

        assert not at.exception
        assert [t for t in _texts(at.error) if BAD_TOKEN_MESSAGE in t]
        assert at.session_state["error"] == BAD_TOKEN_MESSAGE
        assert "GitHub token rejected" in at.status[0].label


def _all_text(at):
    """Every string value rendered anywhere on the page, including inside the status box."""
    found = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            value = getattr(child, "value", None)
            if isinstance(value, str):
                found.append(value)
            label = getattr(child, "label", None)
            if isinstance(label, str):
                found.append(label)
            walk(child)

    walk(at.main)
    walk(at.sidebar)
    return found


def _count_containing(at, text):
    return sum(1 for t in _all_text(at) if text in t)


def _run_failing(exc, token=""):
    from unittest.mock import patch

    at = _app().run()
    if token:
        at.sidebar.text_input[0].set_value(token)
    with patch("sieve.pipeline.run_pipeline", side_effect=exc):
        at.sidebar.button[0].click().run()
    assert not at.exception
    return at


class TestHomeErrorShownOnce:
    def test_rate_limit_error_shown_once(self):
        from github.GithubException import RateLimitExceededException

        from sieve.core.github_limits import describe_rate_limit_error

        exc = RateLimitExceededException(
            403, {"message": "API rate limit exceeded"},
            {"x-ratelimit-reset": "1893506400"},
        )
        at = _run_failing(exc)

        message = describe_rate_limit_error(exc, False)
        assert at.session_state["error"] == message
        assert _count_containing(at, message) == 1
        assert at.status[0].label == "❌ GitHub rate limit reached"

    def test_rejected_token_error_shown_once(self):
        from sieve.core.github_limits import BAD_TOKEN_MESSAGE, GitHubAuthError

        at = _run_failing(GitHubAuthError(), token="ghp_expired")

        assert _count_containing(at, BAD_TOKEN_MESSAGE) == 1
        assert at.status[0].label == "❌ GitHub token rejected"

    def test_generic_exception_shown_once(self):
        at = _run_failing(RuntimeError("something unexpected broke"))

        assert at.session_state["error"] == "something unexpected broke"
        assert _count_containing(at, "something unexpected broke") == 1
        assert at.status[0].label == "❌ Pipeline failed"

    def test_error_survives_widget_interaction_once(self):
        at = _run_failing(RuntimeError("something unexpected broke"))
        at.sidebar.text_input[0].set_value("ghp_other").run()

        assert not at.exception
        assert _count_containing(at, "something unexpected broke") == 1

    def test_success_after_failure_clears_error(self):
        from unittest.mock import patch

        at = _run_failing(RuntimeError("something unexpected broke"))

        summary = {
            "total_repos_discovered": 1, "total_repos_processed": 1, "total_repos_failed": 0,
            "total_functions": 0, "total_classes": 0, "failed_repos": [], "repo_stats": [],
            "output_paths": {}, "repo_metadata": [], "pipeline_summary": "done",
        }
        with patch("sieve.pipeline.run_pipeline", return_value=summary):
            at.sidebar.button[0].click().run()

        assert not at.exception
        assert at.session_state["error"] is None
        assert _count_containing(at, "something unexpected broke") == 0
        assert _count_containing(at, "Last run failed") == 0
        assert not at.error
