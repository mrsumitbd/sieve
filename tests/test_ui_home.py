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
        assert "roughly 20 repositories" in hints[0]

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
