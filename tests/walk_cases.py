"""
tests/walk_cases.py

Small source snippets on which Pass 1 (count_repo_contents) used to disagree
with Pass 2 (extract_from_repo), plus a few neighbours that always agreed.
They are shared by tests/test_pass_consistency.py (Pass 1 == Pass 2) and
tests/test_extraction_walk.py (the shared parse-only walk).

Each case is (id, language, file name, source, (functions, classes)). The
expected pair is what Pass 2 returned BEFORE the walk refactor; it is pinned
here so that "Pass 1 == Pass 2" cannot be satisfied by both being wrong the
same way.
"""

import pytest


def _deep_python(depth: int) -> str:
    lines = [("  " * i) + f"def f{i}():" for i in range(depth)]
    lines.append("  " * depth + "pass")
    return "\n".join(lines) + "\n"


JS_BROKEN = "class A { m( { }\nconst ok = () => 1;\n"

# (id, language, file name, source, (functions, classes) that Pass 2 returns)
CASES = [
    # ── Java ────────────────────────────────────────────────────────────────
    ("java-method-of-anonymous-class", "Java", "A.java",
     "class A { void m() { Object o = new Object() { public void x() {} }; } }\n", (1, 1)),
    ("java-class-declared-inside-a-method", "Java", "A.java",
     "class A { void m() { class L { void l() {} } } }\n", (1, 1)),
    ("java-anonymous-class-in-a-field-initialiser-is-reached", "Java", "A.java",
     "class A { Object o = new Object() { public void x() {} }; void m() {} }\n", (2, 1)),
    ("java-nested-and-inner-classes", "Java", "A.java",
     "class A { void a() {} static class B { void b() {} class C { void c() {} } } }\n", (3, 3)),
    ("java-interface-enum-record-methods", "Java", "A.java",
     "interface I { void i(); }\nenum E { X; void e() {} }\nrecord R(int a) { int r() { return a; } }\n", (3, 0)),
    # ── JavaScript ──────────────────────────────────────────────────────────
    ("js-plain-function-declaration-counted-twice", "JavaScript", "a.js",
     "function f() { return 1; }\n", (1, 0)),
    ("js-async-function-declaration", "JavaScript", "a.js", "async function f() { return 1; }\n", (1, 0)),
    ("js-exported-function-declaration", "JavaScript", "a.js", "export function f() { return 1; }\n", (1, 0)),
    ("js-generator-function", "JavaScript", "a.js", "function* g() {}\n", (0, 0)),
    ("js-function-expression", "JavaScript", "a.js", "const h = function () {};\n", (0, 0)),
    ("js-callback-arrow", "JavaScript", "a.js", "[1].map(x => x);\n", (0, 0)),
    ("js-arrow-inside-function-body", "JavaScript", "a.js",
     "function f() { const g = () => 1; }\n", (1, 0)),
    ("js-object-literal-method", "JavaScript", "a.js", "const o = { m() {} };\n", (0, 0)),
    ("js-class-expression", "JavaScript", "a.js", "const C = class { m() {} };\n", (0, 0)),
    ("js-syntax-error-arrow-counted-inside-an-ERROR-subtree", "JavaScript", "a.js", JS_BROKEN, (0, 0)),
    ("js-anonymous-default-export-arrow", "JavaScript", "a.js", "export default () => 1;\n", (0, 0)),
    ("js-two-arrows-in-one-declaration", "JavaScript", "a.js", "const a = () => 1, b = () => 2;\n", (2, 0)),
    ("js-arrow-class-field-is-not-a-method", "JavaScript", "a.js",
     "class A { f = () => 1; m() {} }\n", (1, 1)),
    ("js-function-declaration-in-a-block", "JavaScript", "a.js", "if (x) { function g() {} }\n", (1, 0)),
    ("js-callbacks-nested-arrows-and-declarations", "JavaScript", "a.js",
     "function load(items) {\n  return items.map(i => i.id).filter(id => id > 0);\n}\n"
     "const handler = (req, res) => { res.on('data', chunk => chunk); };\n"
     "function helper() { return () => 1; }\n"
     "class Svc { run() { return [1].map(x => x); } }\n", (4, 1)),
    # ── C++ ─────────────────────────────────────────────────────────────────
    ("cpp-forward-declaration", "C++", "a.cpp", "class Fwd;\n", (0, 0)),
    ("cpp-elaborated-type-specifier", "C++", "a.cpp",
     "class Fwd;\nvoid f(class Fwd* p) {}\n", (1, 0)),
    ("cpp-class-declared-inside-a-function", "C++", "a.cpp",
     "void f() { class L { void l() {} }; }\n", (1, 0)),
    ("cpp-struct-is-not-a-class-but-its-methods-are-functions", "C++", "a.cpp",
     "struct S { int g() { return 1; } };\n", (1, 0)),
    ("cpp-nested-class-and-namespace", "C++", "a.cpp",
     "namespace n { class A { public: void a() {} class B { public: void b() {} }; }; }\n", (2, 2)),
    ("cpp-templates-and-out-of-class-method", "C++", "a.cpp",
     "template <typename T> class Box { public: T get() { return v; } T v; };\n"
     "template <typename T> T twice(T x) { return x + x; }\n"
     "int Box<int>::other() { return 0; }\n", (3, 1)),
    ("cpp-function-in-extern-c-block", "C++", "a.cpp", 'extern "C" { int c_api() { return 0; } }\n', (1, 0)),
    # ── Python ──────────────────────────────────────────────────────────────
    ("python-nesting-deeper-than-the-extractor-depth-guard", "Python", "a.py", _deep_python(250), (100, 0)),
    ("python-nested-and-decorated", "Python", "a.py",
     "import functools\n\n@functools.cache\ndef a():\n    def b():\n        pass\n    class C:\n        def d(self):\n            pass\n",
     (3, 1)),
    ("python-lambda-is-not-a-function", "Python", "a.py", "f = lambda x: x\n", (0, 0)),
]

CASE_PARAMS = [pytest.param(lang, name, src, expected, id=cid) for cid, lang, name, src, expected in CASES]
