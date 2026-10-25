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
        assert "roughly 10 repositories" in hints[0]

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


class TestHomeDefaults:
    def test_cap_defaults(self):
        at = _app().run()
        values = {w.label: w.value for w in at.sidebar.number_input}
        assert values["Max Repos (0 = no cap)"] == 20
        assert values["Max Functions (-1 = skip, 0 = no cap)"] == 30
        assert values["Max Classes (-1 = skip, 0 = no cap)"] == 30

    def test_other_defaults_unchanged(self):
        at = _app().run()
        values = {w.label: w.value for w in at.sidebar.number_input}
        assert values["Minimum Stars"] == 50
        assert values["Minimum Contributors"] == 5


class TestHomeCapTranslation:
    """0 = no cap and -1 = skip in the widgets must always yield a valid SIEVEConfig."""

    @staticmethod
    def _run_with(repos, funcs, classes):
        from unittest.mock import patch

        at = _app().run()
        values = {
            "Max Repos (0 = no cap)": repos,
            "Max Functions (-1 = skip, 0 = no cap)": funcs,
            "Max Classes (-1 = skip, 0 = no cap)": classes,
        }
        for w in at.sidebar.number_input:
            if w.label in values:
                w.set_value(values[w.label])
        captured = {}

        def fake(config, *a, **kw):
            captured["config"] = config
            raise RuntimeError("stop")

        with patch("sieve.pipeline.run_pipeline", side_effect=fake):
            at.sidebar.button[0].click().run()
        return at, captured.get("config")

    @pytest.mark.parametrize("repos,funcs,classes,expected", [
        (0, 0, 0, (None, None, None, ["function", "class"])),
        (5, 10, 20, (5, 10, 20, ["function", "class"])),
        (0, -1, 0, (None, None, None, ["class"])),
        (0, 0, -1, (None, None, None, ["function"])),
        (1, -1, 3, (1, None, 3, ["class"])),
    ])
    def test_valid_configs(self, repos, funcs, classes, expected):
        at, cfg = self._run_with(repos, funcs, classes)
        assert not [e for e in at.error if "Configuration error" in e.value]
        assert cfg is not None
        assert (cfg.max_repos, cfg.max_functions, cfg.max_classes, cfg.granularity) == expected

    def test_both_skipped_is_a_ui_error_not_a_config(self):
        at, cfg = self._run_with(0, -1, -1)
        assert cfg is None
        assert [e for e in at.error if "At least one granularity" in e.value]


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
