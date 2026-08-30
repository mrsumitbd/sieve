"""
tests/test_dep_viz.py

Unit tests for sieve.ui.dep_viz — dependency graph HTML rendering.
"""

import pytest
from sieve.ui.dep_viz import render_dep_graph


SAMPLE_DEPS = [
    {"name": "requests",  "version": ">=2.28",  "kind": "main"},
    {"name": "numpy",     "version": "==1.24.0", "kind": "main"},
    {"name": "pytest",    "version": None,        "kind": "dev"},
    {"name": "black",     "version": ">=23.0",   "kind": "optional"},
]


class TestRenderDepGraph:
    def test_returns_html_string(self):
        html = render_dep_graph(SAMPLE_DEPS, "owner/repo")
        assert isinstance(html, str)
        assert len(html) > 100

    def test_contains_d3_script(self):
        html = render_dep_graph(SAMPLE_DEPS, "owner/repo")
        assert "d3" in html.lower()

    def test_repo_name_in_output(self):
        html = render_dep_graph(SAMPLE_DEPS, "owner/myrepo")
        assert "myrepo" in html

    def test_package_names_in_output(self):
        html = render_dep_graph(SAMPLE_DEPS, "owner/repo")
        assert "requests" in html
        assert "numpy" in html
        assert "pytest" in html

    def test_empty_deps_returns_placeholder(self):
        html = render_dep_graph([], "owner/repo")
        assert "No dependency manifest" in html

    def test_height_respected(self):
        html = render_dep_graph(SAMPLE_DEPS, "owner/repo", height=500)
        # height - 36 is used for graph container (controls bar takes 36px)
        assert "464" in html  # 500 - 36

    def test_color_codes_present(self):
        html = render_dep_graph(SAMPLE_DEPS, "owner/repo")
        # Main = blue, dev = purple, optional = orange
        assert "#7aa2f7" in html
        assert "#bb9af7" in html
        assert "#e0af68" in html

    def test_dep_count_in_legend(self):
        html = render_dep_graph(SAMPLE_DEPS, "owner/repo")
        assert f"{len(SAMPLE_DEPS)} direct" in html

    def test_deduplicates_same_name(self):
        deps = [
            {"name": "requests", "version": ">=2.28", "kind": "main"},
            {"name": "requests", "version": ">=2.29", "kind": "main"},
        ]
        html = render_dep_graph(deps, "owner/repo")
        # requests appears only once as a node
        assert html.count('"id": "requests"') <= 1

    def test_single_dependency(self):
        deps = [{"name": "flask", "version": ">=2.0", "kind": "main"}]
        html = render_dep_graph(deps, "owner/repo")
        assert "flask" in html
        assert "1 direct" in html

    def test_none_version_handled(self):
        deps = [{"name": "pytest", "version": None, "kind": "dev"}]
        html = render_dep_graph(deps, "owner/repo")
        assert "pytest" in html