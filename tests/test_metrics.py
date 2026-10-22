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

    # McCabe: 1 + one per decision point, counted by hand from the source text
    # (not from the implementation's output):
    #   Java:       for, if, while, do-while, catch              => 1 + 5 = 6
    #   JavaScript: for, if, for-of, while, do-while, catch, ?:  => 1 + 7 = 8
    #   C++:        for, if, while, do-while, catch, ?:          => 1 + 6 = 7

    def test_java_branching_constructs(self):
        code = """
class A {
    int f(int n) {
        int s = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) { s += i; }
        }
        while (s > 100) { s -= 10; }
        do { s++; } while (s < 5);
        try { s /= n; } catch (ArithmeticException e) { s = 0; }
        return s;
    }
}
"""
        assert self._cc(code, "Java") == 6

    def test_javascript_branching_constructs(self):
        code = """
function f(n, xs) {
    let s = 0;
    for (let i = 0; i < n; i++) {
        if (i % 2 === 0) { s += i; }
    }
    for (const x of xs) { s += x; }
    while (s > 100) { s -= 10; }
    do { s++; } while (s < 5);
    try { s /= n; } catch (e) { s = 0; }
    return s > 0 ? s : -s;
}
"""
        assert self._cc(code, "JavaScript") == 8

    def test_cpp_branching_constructs(self):
        code = """
int f(int n) {
    int s = 0;
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) { s += i; }
    }
    while (s > 100) { s -= 10; }
    do { ++s; } while (s < 5);
    try { s /= n; } catch (...) { s = 0; }
    return s > 0 ? s : -s;
}
"""
        assert self._cc(code, "C++") == 7

    def test_while_adds_1(self):
        code = "def foo(n):\n    while n > 0:\n        n -= 1\n    return n\n"
        assert self._cc(code, "Python") == 2


# Hand-counted McCabe values for every construct the metric counts:
#   if / else-if, for, for-each, while, do-while, case labels, catch, ?:, && and ||.
# Rules: each non-default case label adds 1 (empty fall-through labels included),
# `default` and the switch itself add 0; && / || add 1 per operator occurrence;
# every other binary operator, instanceof, await and ?? add 0.
# Expected = 1 + decision points, counted by hand per case below.
CC_CASES = [
    # ── Java ──
    # if: 1+1
    ("java-if", "Java",
     "class A { int f(int a){ if (a>0) { return 1; } return 0; } }", 2),
    # if, else-if (the else branch itself adds 0): 1+2
    ("java-else-if", "Java",
     "class A { void f(int a){ if (a>0) { } else if (a<0) { } else { } } }", 3),
    # && once: 1+1(if)+1(&&)
    ("java-and-in-if", "Java",
     "class A { boolean f(int a,int b){ if (a>0 && b>0) { return true; } return false; } }", 3),
    # || once: 1+1(if)+1(||)
    ("java-or-in-if", "Java",
     "class A { boolean f(int a,int b){ if (a>0 || b>0) { return true; } return false; } }", 3),
    # a && b && c is two operators: 1+2
    ("java-and-chain", "Java",
     "class A { boolean f(boolean a,boolean b,boolean c){ return a && b && c; } }", 3),
    # a && b || c: one && and one ||: 1+2
    ("java-and-or-mixed", "Java",
     "class A { boolean f(boolean a,boolean b,boolean c){ return a && b || c; } }", 3),
    # other binary operators (bitwise &, |, comparison, arithmetic) add 0: 1+0
    ("java-non-logical-binary", "Java",
     "class A { boolean f(int a,int b,int c){ return (a & b | c) + a * b > 1; } }", 1),
    # ternary: 1+1
    ("java-ternary", "Java",
     "class A { int f(int n){ return n > 0 ? n : -n; } }", 2),
    # for: 1+1
    ("java-for", "Java",
     "class A { void f(){ for (int i=0;i<3;i++) { } } }", 2),
    # enhanced for: 1+1
    ("java-enhanced-for", "Java",
     "class A { int f(int[] xs){ int s=0; for (int x : xs) { s += x; } return s; } }", 2),
    # while: 1+1
    ("java-while", "Java",
     "class A { void f(){ while (true) { } } }", 2),
    # do-while: 1+1
    ("java-do-while", "Java",
     "class A { void f(){ do { } while (true); } }", 2),
    # two catch clauses (try and finally add 0): 1+2
    ("java-catch", "Java",
     "class A { void f(){ try { } catch (RuntimeException e) { } catch (Error e) { } finally { } } }", 3),
    # case 1, case 2, default: switch 0, default 0, 2 labels: 1+2
    ("java-switch", "Java",
     "class A { void f(int n){ switch (n) { case 1: break; case 2: break; default: break; } } }", 3),
    # empty fall-through: case 1, case 2 each count (default 0): 1+2
    ("java-switch-fallthrough", "Java",
     "class A { void f(int n){ switch (n) { case 1: case 2: break; default: break; } } }", 3),
    # default only: 1+0
    ("java-switch-default-only", "Java",
     "class A { void f(int n){ switch (n) { default: break; } } }", 1),
    # default first, then case 3: 1+1
    ("java-switch-default-first", "Java",
     "class A { void f(int n){ switch (n) { default: case 3: break; } } }", 2),
    # arrow switch expression: case 1, case 2,3 (one label), default: 1+2
    ("java-switch-arrow", "Java",
     "class A { int f(int n){ return switch (n) { case 1 -> 1; case 2, 3 -> 2; default -> 0; }; } }", 3),
    # `case null, default` is the default label (0); case String s: 1+1
    ("java-switch-null-default", "Java",
     "class A { int f(Object o){ return switch (o) { case null, default -> 0; case String s -> 1; }; } }", 2),
    # instanceof is not a decision point: 1+0
    ("java-instanceof", "Java",
     "class A { boolean f(Object o){ return o instanceof String; } }", 1),
    # for-each, if, &&, ||, else-if, case 1, case 2, case 3, ?:, catch: 1+10
    ("java-combined", "Java",
     "class A { int f(int[] xs){ int s=0;"
     " for (int x : xs) { if (x>0 && x<10 || x==99) { s+=x; } else if (x<0) { s-=x; } }"
     " switch (s) { case 1: case 2: s++; break; case 3: s--; break; default: break; }"
     " try { s = s>5 ? s : -s; } catch (Exception e) { s = 0; }"
     " return s; } }", 11),

    # ── JavaScript ──
    # if, else-if: 1+2
    ("js-else-if", "JavaScript",
     "function f(a){ if (a>0) { } else if (a<0) { } else { } }", 3),
    # && once: 1+1(if)+1(&&)
    ("js-and-in-if", "JavaScript",
     "function f(a,b){ if (a>0 && b>0) { return true; } return false; }", 3),
    # || once: 1+1(if)+1(||)
    ("js-or-in-if", "JavaScript",
     "function f(a,b){ if (a>0 || b>0) { return true; } return false; }", 3),
    # a && b && c: 1+2
    ("js-and-chain", "JavaScript",
     "function f(a,b,c){ return a && b && c; }", 3),
    # a && b || c: 1+2
    ("js-and-or-mixed", "JavaScript",
     "function f(a,b,c){ return a && b || c; }", 3),
    # ?? , &&=, ||=, bitwise and arithmetic operators add 0: 1+0
    ("js-non-counted-operators", "JavaScript",
     "function f(a,b){ a ||= b; a &&= b; return (a ?? b) & (a | b) + 1; }", 1),
    # ternary: 1+1
    ("js-ternary", "JavaScript",
     "function f(n){ return n > 0 ? n : -n; }", 2),
    # for: 1+1
    ("js-for", "JavaScript",
     "function f(){ for (let i=0;i<3;i++) { } }", 2),
    # for-in: 1+1
    ("js-for-in", "JavaScript",
     "function f(o){ for (const k in o) { } }", 2),
    # for-of: 1+1
    ("js-for-of", "JavaScript",
     "function f(o){ for (const k of o) { } }", 2),
    # while: 1+1
    ("js-while", "JavaScript",
     "function f(){ while (true) { } }", 2),
    # do-while: 1+1
    ("js-do-while", "JavaScript",
     "function f(){ do { } while (true); }", 2),
    # catch: 1+1 (finally adds 0)
    ("js-catch", "JavaScript",
     "function f(){ try { } catch (e) { } finally { } }", 2),
    # case 1, case 2, default: 1+2
    ("js-switch", "JavaScript",
     "function f(n){ switch (n) { case 1: break; case 2: break; default: break; } }", 3),
    # empty fall-through: case 1, case 2: 1+2
    ("js-switch-fallthrough", "JavaScript",
     "function f(n){ switch (n) { case 1: case 2: break; default: break; } }", 3),
    # default only: 1+0
    ("js-switch-default-only", "JavaScript",
     "function f(n){ switch (n) { default: break; } }", 1),
    # await is not a decision point: 1+0
    ("js-await", "JavaScript",
     "async function f(){ await g(); await h(); }", 1),
    # for-of, if, &&, ||, else-if, case 1, case 2, case 3, ?:, catch (await adds 0): 1+10
    ("js-combined", "JavaScript",
     "async function f(xs){ let s=0;"
     " for (const x of xs) { if (x>0 && x<10 || x===99) { s+=x; } else if (x<0) { s-=x; } }"
     " switch (s) { case 1: case 2: s++; break; case 3: s--; break; default: break; }"
     " try { s = s>5 ? s : -s; await g(); } catch (e) { s = 0; }"
     " return s; }", 11),

    # ── C++ ──
    # if, else-if: 1+2
    ("cpp-else-if", "C++",
     "void f(int a){ if (a>0) { } else if (a<0) { } else { } }", 3),
    # && once: 1+1(if)+1(&&)
    ("cpp-and-in-if", "C++",
     "bool f(int a,int b){ if (a>0 && b>0) { return true; } return false; }", 3),
    # || once: 1+1(if)+1(||)
    ("cpp-or-in-if", "C++",
     "bool f(int a,int b){ if (a>0 || b>0) { return true; } return false; }", 3),
    # a && b && c: 1+2
    ("cpp-and-chain", "C++",
     "bool f(bool a,bool b,bool c){ return a && b && c; }", 3),
    # a && b || c: 1+2
    ("cpp-and-or-mixed", "C++",
     "bool f(bool a,bool b,bool c){ return a && b || c; }", 3),
    # alternative tokens `and` / `or` are the same operators: 1+2
    ("cpp-and-or-keywords", "C++",
     "bool f(bool a,bool b,bool c){ return a and b or c; }", 3),
    # bitwise &, |, comparison, arithmetic add 0: 1+0
    ("cpp-non-logical-binary", "C++",
     "int f(int a,int b,int c){ return (a & b | c) + a * b; }", 1),
    # ternary: 1+1
    ("cpp-ternary", "C++",
     "int f(int n){ return n > 0 ? n : -n; }", 2),
    # for: 1+1
    ("cpp-for", "C++",
     "void f(){ for (int i=0;i<3;i++) { } }", 2),
    # range-for: 1+1
    ("cpp-range-for", "C++",
     "int f(std::vector<int> v){ int s=0; for (auto x : v) { s += x; } return s; }", 2),
    # while: 1+1
    ("cpp-while", "C++",
     "void f(){ while (true) { } }", 2),
    # do-while: 1+1
    ("cpp-do-while", "C++",
     "void f(){ do { } while (true); }", 2),
    # two catch clauses, one of them catch(...): 1+2
    ("cpp-catch", "C++",
     "void f(){ try { } catch (const std::exception& e) { } catch (...) { } }", 3),
    # case 1, case 2, default: 1+2
    ("cpp-switch", "C++",
     "void f(int n){ switch (n) { case 1: break; case 2: break; default: break; } }", 3),
    # empty fall-through: case 1, case 2: 1+2
    ("cpp-switch-fallthrough", "C++",
     "void f(int n){ switch (n) { case 1: case 2: break; default: break; } }", 3),
    # default only: 1+0
    ("cpp-switch-default-only", "C++",
     "void f(int n){ switch (n) { default: break; } }", 1),
    # default first, then case 1: 1+1
    ("cpp-switch-default-first", "C++",
     "void f(int n){ switch (n) { default: case 1: n++; break; } }", 2),
    # range-for, if, &&, ||, else-if, case 1, case 2, case 3, ?:, catch: 1+10
    ("cpp-combined", "C++",
     "int f(std::vector<int> xs){ int s=0;"
     " for (auto x : xs) { if (x>0 && x<10 || x==99) { s+=x; } else if (x<0) { s-=x; } }"
     " switch (s) { case 1: case 2: s++; break; case 3: s--; break; default: break; }"
     " try { s = s>5 ? s : -s; } catch (const std::exception& e) { s = 0; }"
     " return s; }", 11),

    # ── Python control (the Python path must not change) ──
    # and in if: 1+1(if)+1(and)
    ("py-and-in-if", "Python",
     "def f(a,b):\n    if a>0 and b>0:\n        return True\n    return False\n", 3),
    # a and b and c: two boolean operators: 1+2
    ("py-and-chain", "Python",
     "def f(a,b,c):\n    return a and b and c\n", 3),
    # if, elif, for, while, except, ternary, list comprehension, or: 1+8
    ("py-combined", "Python",
     "def f(a, xs):\n"
     "    if a:\n        pass\n    elif xs:\n        pass\n"
     "    for x in xs:\n        pass\n"
     "    while a:\n        a -= 1\n"
     "    try:\n        pass\n    except Exception:\n        pass\n"
     "    y = 1 if a else 2\n"
     "    z = [i for i in xs]\n"
     "    return a or y\n", 9),
]


class TestCyclomaticComplexityPerConstruct:
    @pytest.mark.parametrize(
        "language,source,expected",
        [(lang, src, exp) for _, lang, src, exp in CC_CASES],
        ids=[name for name, *_ in CC_CASES],
    )
    def test_hand_counted(self, language, source, expected):
        root = _get_tree(source, language).root_node
        assert _cyclomatic_complexity(root, language) == expected


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