"""
tests/test_metrics.py

Unit tests for sieve.core.metrics — cross-language code metrics.
"""

import math
import pytest
from sieve.core.metrics import (
    compute_metrics,
    _cyclomatic_complexity,
    _max_nesting_depth,
    _halstead_metrics,
    _raw_metrics,
    _maintainability_index,
    _get_tree,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SIMPLE_PYTHON = "def add(a, b):\n    return a + b\n"

COMPLEX_PYTHON = """
def process(items):
    result = []
    for item in items:
        if item > 0:
            for i in range(item):
                if i % 2 == 0:
                    result.append(i)
        elif item < 0:
            continue
    return result
"""

COMMENTED_PYTHON = """
# This is a comment
def foo(x):
    # Another comment
    return x + 1
"""

SIMPLE_JAVA = """
public int add(int a, int b) {
    return a + b;
}
"""

SIMPLE_JS = """
function add(a, b) {
    return a + b;
}
"""

SIMPLE_CPP = """
int add(int a, int b) {
    return a + b;
}
"""


# ─── _get_tree ────────────────────────────────────────────────────────────────

class TestGetTree:
    def test_returns_tree_for_python(self):
        tree = _get_tree(SIMPLE_PYTHON, "Python")
        assert tree is not None
        assert tree.root_node is not None

    def test_returns_none_for_unknown_language(self):
        tree = _get_tree("some code", "COBOL")
        assert tree is None

    def test_returns_tree_for_java(self):
        tree = _get_tree(SIMPLE_JAVA, "Java")
        assert tree is not None

    def test_returns_tree_for_javascript(self):
        tree = _get_tree(SIMPLE_JS, "JavaScript")
        assert tree is not None

    def test_returns_tree_for_cpp(self):
        tree = _get_tree(SIMPLE_CPP, "C++")
        assert tree is not None


# ─── _raw_metrics ─────────────────────────────────────────────────────────────

class TestRawMetrics:
    def _get_root(self, source, language):
        return _get_tree(source, language).root_node

    def test_loc_correct(self):
        root = self._get_root(SIMPLE_PYTHON, "Python")
        m = _raw_metrics(SIMPLE_PYTHON, "Python", root)
        assert m["loc"] == len(SIMPLE_PYTHON.splitlines())

    def test_blank_lines_counted(self):
        code = "def foo():\n\n    return 1\n"
        root = self._get_root(code, "Python")
        m = _raw_metrics(code, "Python", root)
        assert m["blank"] >= 1

    def test_sloc_less_than_loc(self):
        root = self._get_root(COMMENTED_PYTHON, "Python")
        m = _raw_metrics(COMMENTED_PYTHON, "Python", root)
        assert m["sloc"] <= m["loc"]

    def test_comments_detected(self):
        root = self._get_root(COMMENTED_PYTHON, "Python")
        m = _raw_metrics(COMMENTED_PYTHON, "Python", root)
        assert m["comments"] >= 1

    def test_comment_ratio_between_0_and_1(self):
        root = self._get_root(COMMENTED_PYTHON, "Python")
        m = _raw_metrics(COMMENTED_PYTHON, "Python", root)
        assert 0.0 <= m["comment_ratio"] <= 1.0

    def test_lloc_counts_statements(self):
        root = self._get_root(SIMPLE_PYTHON, "Python")
        m = _raw_metrics(SIMPLE_PYTHON, "Python", root)
        assert m["lloc"] >= 1

    def test_loc_equals_sloc_plus_comments_plus_blank_plus_multi(self):
        root = self._get_root(COMMENTED_PYTHON, "Python")
        m = _raw_metrics(COMMENTED_PYTHON, "Python", root)
        # sloc + comments + multi + blank should equal loc
        total = m["sloc"] + m["comments"] + m["multi"] + m["blank"]
        assert total == m["loc"]


# ─── _cyclomatic_complexity ───────────────────────────────────────────────────

class TestCyclomaticComplexity:
    def _cc(self, source, language):
        root = _get_tree(source, language).root_node
        return _cyclomatic_complexity(root, language)

    def test_simple_function_is_1(self):
        assert self._cc(SIMPLE_PYTHON, "Python") == 1

    def test_complex_function_correct(self):
        # process() has: for, if, for, if, elif = 5 branches → CC = 6
        assert self._cc(COMPLEX_PYTHON, "Python") == 6

    def test_if_adds_1(self):
        code = "def foo(x):\n    if x > 0:\n        return x\n    return 0\n"
        assert self._cc(code, "Python") == 2

    def test_java_simple_is_1(self):
        assert self._cc(SIMPLE_JAVA, "Java") == 1

    def test_js_simple_is_1(self):
        assert self._cc(SIMPLE_JS, "JavaScript") == 1

    def test_cpp_simple_is_1(self):
        assert self._cc(SIMPLE_CPP, "C++") == 1

    def test_while_adds_1(self):
        code = "def foo(n):\n    while n > 0:\n        n -= 1\n    return n\n"
        assert self._cc(code, "Python") == 2


# ─── _max_nesting_depth ───────────────────────────────────────────────────────

class TestMaxNestingDepth:
    def _depth(self, source, language):
        root = _get_tree(source, language).root_node
        return _max_nesting_depth(root, language)

    def test_simple_function_is_0(self):
        assert self._depth(SIMPLE_PYTHON, "Python") == 0

    def test_one_if_is_1(self):
        code = "def foo(x):\n    if x > 0:\n        return x\n"
        assert self._depth(code, "Python") == 1

    def test_complex_function_depth_4(self):
        # for → if → for → if = depth 4
        assert self._depth(COMPLEX_PYTHON, "Python") == 4

    def test_nested_if_is_2(self):
        code = "def foo(x, y):\n    if x > 0:\n        if y > 0:\n            return x + y\n"
        assert self._depth(code, "Python") == 2


# ─── _halstead_metrics ────────────────────────────────────────────────────────

class TestHalsteadMetrics:
    def _hal(self, source, language):
        root = _get_tree(source, language).root_node
        return _halstead_metrics(root, language)

    def test_returns_all_keys(self):
        m = self._hal(SIMPLE_PYTHON, "Python")
        for key in ["h1", "h2", "N1", "N2", "vocabulary", "halstead_length",
                    "calculated_length", "volume", "difficulty", "effort",
                    "time", "bugs"]:
            assert key in m

    def test_vocabulary_equals_h1_plus_h2(self):
        m = self._hal(SIMPLE_PYTHON, "Python")
        assert m["vocabulary"] == m["h1"] + m["h2"]

    def test_length_equals_N1_plus_N2(self):
        m = self._hal(SIMPLE_PYTHON, "Python")
        assert m["halstead_length"] == m["N1"] + m["N2"]

    def test_volume_positive(self):
        m = self._hal(SIMPLE_PYTHON, "Python")
        assert m["volume"] > 0

    def test_effort_equals_difficulty_times_volume(self):
        m = self._hal(SIMPLE_PYTHON, "Python")
        assert abs(m["effort"] - round(m["difficulty"] * m["volume"], 2)) < 0.1

    def test_time_equals_effort_over_18(self):
        m = self._hal(SIMPLE_PYTHON, "Python")
        assert abs(m["time"] - round(m["effort"] / 18.0, 2)) < 0.1

    def test_bugs_equals_volume_over_3000(self):
        m = self._hal(SIMPLE_PYTHON, "Python")
        assert abs(m["bugs"] - round(m["volume"] / 3000.0, 4)) < 0.001

    def test_java_halstead(self):
        m = self._hal(SIMPLE_JAVA, "Java")
        assert m["h1"] >= 0 and m["h2"] >= 0

    def test_complex_function_higher_volume(self):
        simple = self._hal(SIMPLE_PYTHON, "Python")
        complex_ = self._hal(COMPLEX_PYTHON, "Python")
        assert complex_["volume"] > simple["volume"]


# ─── _maintainability_index ───────────────────────────────────────────────────

class TestMaintainabilityIndex:
    def test_returns_float(self):
        mi = _maintainability_index(100.0, 3, 10, 0.1)
        assert isinstance(mi, float)

    def test_between_0_and_100(self):
        mi = _maintainability_index(100.0, 3, 10, 0.1)
        assert 0.0 <= mi <= 100.0

    def test_zero_sloc_returns_100(self):
        mi = _maintainability_index(100.0, 3, 0, 0.1)
        assert mi == 100.0

    def test_higher_cc_lowers_mi(self):
        mi_low  = _maintainability_index(200.0, 2, 20, 0.1)
        mi_high = _maintainability_index(200.0, 20, 20, 0.1)
        assert mi_low > mi_high

    def test_higher_volume_lowers_mi(self):
        mi_low  = _maintainability_index(100.0, 3, 10, 0.1)
        mi_high = _maintainability_index(10000.0, 3, 10, 0.1)
        assert mi_low > mi_high


# ─── compute_metrics (public API) ────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_all_expected_keys(self):
        m = compute_metrics(SIMPLE_PYTHON, "Python")
        expected = [
            "loc", "sloc", "lloc", "comments", "multi", "blank",
            "comment_ratio", "cyclomatic_complexity", "max_nesting_depth",
            "h1", "h2", "N1", "N2", "vocabulary", "halstead_length",
            "calculated_length", "volume", "difficulty", "effort",
            "time", "bugs", "maintainability_index",
        ]
        for key in expected:
            assert key in m, f"Missing key: {key}"

    def test_returns_empty_for_unknown_language(self):
        m = compute_metrics("some code", "COBOL")
        assert m == {}

    def test_python_simple_function(self):
        m = compute_metrics(SIMPLE_PYTHON, "Python")
        assert m["cyclomatic_complexity"] == 1
        assert m["max_nesting_depth"] == 0
        assert m["loc"] > 0
        assert m["volume"] > 0
        assert 0 <= m["maintainability_index"] <= 100

    def test_complex_python_function(self):
        m = compute_metrics(COMPLEX_PYTHON, "Python")
        assert m["cyclomatic_complexity"] == 6
        assert m["max_nesting_depth"] == 4

    def test_java_function(self):
        m = compute_metrics(SIMPLE_JAVA, "Java")
        assert m["cyclomatic_complexity"] == 1
        assert m["loc"] > 0

    def test_javascript_function(self):
        m = compute_metrics(SIMPLE_JS, "JavaScript")
        assert m["cyclomatic_complexity"] == 1

    def test_cpp_function(self):
        m = compute_metrics(SIMPLE_CPP, "C++")
        assert m["cyclomatic_complexity"] == 1

    def test_all_values_are_numbers(self):
        m = compute_metrics(SIMPLE_PYTHON, "Python")
        for k, v in m.items():
            assert isinstance(v, (int, float)), f"Key {k} has non-numeric value: {v}"