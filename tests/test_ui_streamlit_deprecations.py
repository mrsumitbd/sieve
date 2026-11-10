"""
tests/test_ui_streamlit_deprecations.py

The UI must not call Streamlit APIs that Streamlit has deprecated: a newer Streamlit
may remove them, and the page breaks. Two independent checks:

* Dynamic: load Home.py (welcome screen, results, dependency graph, both sample
  viewers) and every page with AppTest, and fail on any deprecation warning.
  Streamlit raises these through `streamlit.deprecation_util.show_deprecation_warning`,
  which *logs* on the non-propagating logger "streamlit.deprecation_util" and, only for
  some APIs and only when client.showErrorDetails is "full", also renders an st.warning.
  So pytest's caplog never sees them; a handler is attached to that logger directly. Third-party
  warnings (e.g. pandas deprecating a keyword that Streamlit itself passes) are not collected.
* Static: `use_container_width` must not appear as a keyword anywhere in src/sieve/ui, and
  `st.components.v1.html` must not be used at all (use st.iframe). The dynamic check cannot
  see every call site: on current Streamlit, st.button and st.download_button translate the
  old argument silently with no warning, and the AppTest only reaches the branches it sets up.
"""

import ast
import json
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "src" / "sieve" / "ui"
HOME = UI / "Home.py"
PAGES = sorted((UI / "pages").glob("*.py"))

DEPRECATION_LOGGER = "streamlit.deprecation_util"


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _capture():
    """Collect Streamlit's own deprecation notices raised inside the block (a list, filled on exit).

    Only the "streamlit.deprecation_util" logger is read. Streamlit raises every deprecation of its
    own API through show_deprecation_warning; it has no deprecation category for the `warnings`
    module. Python warnings are deliberately not collected: a DeprecationWarning or FutureWarning
    from a third-party library (pandas, numpy, ...) is attributed to the file that called the
    library, which can be a Streamlit file, and says nothing about the APIs our code calls.
    """
    collector = _Collector()
    logger = logging.getLogger(DEPRECATION_LOGGER)
    logger.addHandler(collector)
    messages = []
    try:
        yield messages
    finally:
        logger.removeHandler(collector)
        messages += collector.messages


def _run(app, steps=None):
    """Run an AppTest and return every deprecation message raised while doing so."""
    with _capture() as messages:
        app.run()
        if steps:
            steps(app)
    assert not app.exception, [e.value for e in app.exception]
    # Older Streamlit versions render the notice in the page instead of (or as well as) logging it.
    messages += [w.value for w in app.warning if "will be removed" in w.value]
    return list(dict.fromkeys(messages))


def _app(path):
    return AppTest.from_file(str(path), default_timeout=60)


# ─── State that reaches every conditional call site in Home.py ────────────────

_FUNCTION = {
    "repo": "o/r", "file_path": "a.py", "language": "Python", "func_name": "f", "parameters": ["x"],
    "source_code": "def f(x):\n    return x + 1\n", "signature": "def f(x)", "used_imports": ["import os"],
    "start_line": 1, "end_line": 2, "is_method": False, "llm_score": 0.5,
}
_CLASS = {
    "repo": "o/r", "file_path": "a.py", "language": "Python", "class_name": "C", "parent_classes": [],
    "source_code": "class C:\n    pass\n", "skeleton": "class C: ...", "used_imports": [],
    "start_line": 1, "end_line": 2, "method_count": 0, "has_constructor": False,
}


@pytest.fixture
def summary(tmp_path):
    (tmp_path / "functions.jsonl").write_text(json.dumps(_FUNCTION) + "\n")
    (tmp_path / "classes.jsonl").write_text(json.dumps(_CLASS) + "\n")
    return {
        "total_repos_discovered": 2, "total_repos_processed": 2, "total_repos_failed": 0,
        "total_functions": 1, "total_classes": 1, "failed_repos": [],
        "repo_stats": [
            {"repo": "o/r", "stars": 100, "contributors": 5, "functions": 3, "classes": 1,
             "test_suite_present": True, "test_confidence": 0.9, "license": "MIT"},
            {"repo": "o/s", "stars": 60, "contributors": None, "functions": 2, "classes": 2,
             "test_suite_present": False, "test_confidence": 0.1, "license": "Apache-2.0"},
        ],
        "output_paths": {"functions_jsonl": str(tmp_path / "functions.jsonl"),
                         "classes_jsonl": str(tmp_path / "classes.jsonl")},
        "repo_metadata": [
            {"full_name": "o/r", "dependencies": [{"name": "numpy", "version": "1.0", "kind": "runtime"}]},
            {"full_name": "o/s", "dependencies": []},
        ],
    }


def _show_dependency_graph(at):
    (toggle,) = [t for t in at.toggle if t.label == "Show Dependency Graph"]
    toggle.set_value(True).run()


def _draw_sample(at):
    (button,) = [b for b in at.button if "Draw random" in b.label]
    button.click().run()


# ─── Dynamic check ────────────────────────────────────────────────────────────

class TestNoDeprecationWarnings:
    def test_capture_sees_a_deprecation(self):
        """Guard against a vacuous pass: the capture must see a notice raised the way Streamlit raises them.

        Called directly rather than inside an AppTest script: AppTest.from_function misbehaves when it
        runs after the multipage Home.py in the same process on some Streamlit versions (PagesManager
        keeps process-global state).
        """
        from streamlit.deprecation_util import show_deprecation_warning

        with _capture() as messages:
            show_deprecation_warning("canary deprecation")
        assert "canary deprecation" in messages

    @pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::DeprecationWarning")  # these tests raise them on purpose
    def test_capture_ignores_third_party_warnings_from_a_streamlit_path(self):
        """A pandas-style warning attributed to a file under a "streamlit" path is not our deprecation.

        This is what Streamlit 1.56 + pandas 3.1 produces: built_in_chart_utils.py calls
        df.reset_index(inplace=True), pandas warns, and the warning's filename is Streamlit's.
        """
        streamlit_file = "/site-packages/streamlit/elements/lib/built_in_chart_utils.py"
        pandas_msg = "The inplace keyword in DataFrame.reset_index is deprecated and will be removed"
        with _capture() as messages:
            for category in (FutureWarning, DeprecationWarning):
                warnings.warn_explicit(pandas_msg, category, streamlit_file, 651)
        assert messages == []

    @pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::DeprecationWarning")  # these tests raise them on purpose
    def test_run_fails_on_a_deprecation_and_not_on_a_third_party_warning(self, tmp_path):
        """Through _run (the path every real test takes): the same script is clean, then flagged."""
        script = tmp_path / "script.py"
        script.write_text(
            "import warnings\n"
            "import streamlit as st\n"
            "from streamlit.deprecation_util import show_deprecation_warning\n"
            "streamlit_file = '/site-packages/streamlit/elements/lib/built_in_chart_utils.py'\n"
            "for category in (FutureWarning, DeprecationWarning):\n"
            "    warnings.warn_explicit('pandas says inplace is deprecated', category, streamlit_file, 651)\n"
            "if st.session_state.get('raise_it'):\n"
            "    show_deprecation_warning('canary deprecation through the app')\n"
            "st.write('ok')\n"
        )
        assert _run(_app(script)) == []

        flagged = _app(script)
        flagged.session_state["raise_it"] = True
        assert _run(flagged) == ["canary deprecation through the app"]

    def test_welcome_screen(self):
        assert _run(_app(HOME)) == []

    def test_results_and_per_repo_table(self, summary):
        at = _app(HOME)
        at.session_state["summary"] = summary
        assert _run(at) == []

    def test_dependency_graph_and_table(self, summary):
        at = _app(HOME)
        at.session_state["summary"] = summary
        assert _run(at, _show_dependency_graph) == []

    def test_draw_sample_button(self, summary):
        at = _app(HOME)
        at.session_state["summary"] = summary
        assert _run(at, _draw_sample) == []

    @pytest.mark.parametrize("rtype,record", [("Function/Method", _FUNCTION), ("Class", _CLASS)])
    def test_sample_viewer_metrics_table(self, summary, rtype, record):
        at = _app(HOME)
        at.session_state["summary"] = summary
        at.session_state["sample"] = (rtype, record, 0, 1)
        assert _run(at) == []

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_page(self, page):
        assert _run(_app(page)) == []

    def test_every_page_is_covered(self):
        assert [p.name for p in PAGES] == ["1_Documentation.py", "2_About_the_Classifier.py", "3_FAQ.py"]


# ─── Static check ─────────────────────────────────────────────────────────────

def _use_container_width_sites():
    sites = []
    for path in sorted(UI.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Call):
                sites += [f"{path.relative_to(UI.parents[2])}:{kw.value.lineno}"
                          for kw in node.keywords if kw.arg == "use_container_width"]
    return sites


def test_ui_does_not_use_use_container_width():
    """Replace use_container_width=True with width="stretch", and =False with width="content"."""
    assert _use_container_width_sites() == []


# Names under which streamlit.components.v1 (or streamlit.components) can be reached.
_COMPONENT_MODULES = ("streamlit.components.v1", "streamlit.components")


def _components_html_sites(source, filename="<src>"):
    """Line numbers where st.components.v1.html is imported or called, however it is spelled.

    Catches `st.components.v1.html(...)`, `streamlit.components.v1.html(...)`, an aliased module
    (`import streamlit.components.v1 as c; c.html(...)`, `from streamlit import components`,
    `from streamlit.components import v1`) and `from streamlit.components.v1 import html`.
    """
    tree = ast.parse(source, filename=filename)
    aliases = set()
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname for a in node.names if a.name in _COMPONENT_MODULES and a.asname}
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in _COMPONENT_MODULES:
                sites += [node.lineno for a in node.names if a.name == "html"]
            for a in node.names:
                if f"{node.module}.{a.name}" in _COMPONENT_MODULES:
                    aliases.add(a.asname or a.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "html":
            owner = ast.unparse(node.value)
            if owner in aliases or owner.endswith(("components.v1", ".components")) or owner == "components.v1":
                sites.append(node.lineno)
    return sorted(set(sites))


def _components_html_sites_in_ui():
    return [f"{path.relative_to(UI.parents[2])}:{line}"
            for path in sorted(UI.rglob("*.py"))
            for line in _components_html_sites(path.read_text(encoding="utf-8"), str(path))]


@pytest.mark.parametrize("source", [
    "import streamlit as st\nst.components.v1.html('<p/>')",
    "import streamlit\nstreamlit.components.v1.html('<p/>', height=5)",
    "import streamlit.components.v1 as components\ncomponents.html('<p/>')",
    "import streamlit.components.v1 as c\nc.html('<p/>')",
    "from streamlit import components\ncomponents.v1.html('<p/>')",
    "from streamlit.components import v1\nv1.html('<p/>')",
    "from streamlit.components.v1 import html\nhtml('<p/>')",
])
def test_components_html_detector_finds_each_spelling(source):
    """Guard against a vacuous pass: the static check must flag every way of reaching the deprecated API."""
    assert _components_html_sites(source) != []


@pytest.mark.parametrize("source", [
    "import streamlit as st\nst.iframe('<p/>', height=5)",
    "import streamlit as st\nst.html('<p/>')",
    "import streamlit as st\nst.components.v1.declare_component('x')",
])
def test_components_html_detector_ignores_the_replacements(source):
    assert _components_html_sites(source) == []


def test_ui_does_not_use_components_v1_html():
    """st.components.v1.html is deprecated; embed HTML with st.iframe (Streamlit >= 1.56)."""
    assert _components_html_sites_in_ui() == []
