"""
tests/test_pass_consistency.py

Pass 1 (count_repo_contents, which sizes the stratified per-repo caps) and
Pass 2 (extract_from_repo) must agree on which files are eligible. They used to
carry two hand-copied lists of test-file name rules that had drifted apart, so
Pass 1 counted records that Pass 2 later dropped.

These tests build small fake repos in tmp_path (no network) and check:

  1. the shared predicate is_test_file,
  2. the invariant  count_repo_contents(repo) == what extract_from_repo(repo)
     returns, for functions and classes separately, for every language, with
     and without caps,
  3. the allocation that _stratified_caps produces from the Pass 1 counts,
  4. the cases that used to differ at the NODE level (tests/walk_cases.py: JS
     function keyword tokens and nested or anonymous functions, parse-error
     subtrees, Java anonymous-class methods and classes in method bodies, C++
     forward declarations, the Python depth guard). Pass 1 now counts exactly
     the nodes the Pass 2 extractors turn into records, so these are ordinary
     assertions,
  5. the allocation _stratified_caps produces from Pass 1 for a JavaScript repo
     set with callbacks, nested arrows and function declarations.
"""

import re
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from sieve.config import SIEVEConfig
from sieve.core import extraction as ex
from sieve.core.discovery import RepoMetadata
from sieve.pipeline import _stratified_caps, run_pipeline

import walk_cases

BOTH = ["function", "class"]

# Every "keep" file in the fixtures holds exactly this many records, in every
# language, so expected totals are simply (files * 2, files * 1) and cannot be
# produced by two equally wrong counters agreeing with each other.
FUNCS_PER_FILE = 2
CLASSES_PER_FILE = 1


def _ident(stem: str) -> str:
    return re.sub(r"\W", "_", stem)


def source_for(language: str, rel: str) -> str:
    """A small valid file with FUNCS_PER_FILE functions and CLASSES_PER_FILE class.

    Plain constructs only; the awkward ones live in tests/walk_cases.py.
    """
    s = _ident(Path(rel).name.split(".")[0])
    if language == "Python":
        return f"def f_{s}():\n    return 1\n\n\nclass C_{s}:\n    def m(self):\n        return 2\n"
    if language == "Java":
        return (f"public class C_{s} {{\n    public C_{s}() {{}}\n"
                f"    public int m() {{ return 1; }}\n}}\n")
    if language == "JavaScript":
        return f"class C_{s} {{\n  m() {{ return 1; }}\n}}\nconst f_{s} = () => 1;\n"
    if language == "C++":
        return (f"class C_{s} {{\npublic:\n  int m() {{ return 1; }}\n}};\n"
                f"int f_{s}() {{ return 2; }}\n")
    raise AssertionError(language)


# keep: eligible, counted by both passes.
# drop_*: must be counted by neither pass (and extracted by neither).
FIXTURES = {
    "Python": {
        "keep": [
            "app.py", "util.py", "src/core/engine.py",
            # near misses that look like tests but are not
            "src/contest.py", "src/mytest.py", "src/latest.py", "src/testing_utils.py",
        ],
        "drop_test_names": [
            "test_app.py", "app_test.py", "TestApp.py", "src/Testing.py", "src/core/test_engine.py",
        ],
        "drop_test_dirs": [
            "tests/helpers.py", "test/helpers.py", "__tests__/helpers.py",
            "spec/helpers.py", "specs/helpers.py", "src/tests/helpers.py",
        ],
        "drop_skip_dirs": [
            "vendor/v.py", "node_modules/n.py", "build/b.py", "venv/e.py",
            "third_party/t.py", "src/dist/d.py",
        ],
    },
    "Java": {
        "keep": [
            "src/main/App.java", "src/main/Service.java", "src/main/Contest.java",
            "src/main/Latest.java", "src/main/Attest.java",
        ],
        "drop_test_names": [
            "src/main/TestApp.java", "src/main/AppTest.java", "src/main/TestUtils.java",
            "src/main/ServiceTest.java", "src/main/TestCase.java",
        ],
        "drop_test_dirs": [
            "src/test/java/Helper.java", "tests/Helper.java", "spec/Helper.java",
        ],
        "drop_skip_dirs": ["vendor/V.java", "build/B.java", "node_modules/N.java"],
    },
    "JavaScript": {
        "keep": [
            "src/app.js", "src/view.jsx", "lib/index.js", "src/contest.js",
            "src/latest.js", "src/test-utils.js", "src/widget.testing.js",
        ],
        "drop_test_names": [
            "src/app.test.js", "src/app.spec.js", "src/test_helpers.js",
            "src/TestWidget.js", "src/Button.test.js",
        ],
        "drop_test_dirs": [
            "test/helpers.js", "tests/helpers.js", "__tests__/helpers.js",
            "spec/helpers.js", "specs/helpers.js", "src/__tests__/view.js",
        ],
        "drop_skip_dirs": ["node_modules/dep/index.js", "vendor/lib.js", "dist/bundle.js", "build/b.js"],
        # Dropped by Pass 2's extract_from_file on the *name*: .min.js and any
        # path containing "node_modules" (even as part of a longer name).
        "drop_vendor_names": ["lib/jquery.min.js", "src/node_modules_shim.js"],
        # Dropped by both passes on the *content* (a line over 1000 characters).
        "drop_minified_content": ["src/bundle_legacy.js"],
        # Not an extension SIEVE parses: counted by neither pass.
        "drop_other_extensions": ["src/types.ts", "src/app.test.ts", "src/app.spec.ts"],
    },
    "C++": {
        "keep": [
            "src/main.cpp", "include/engine.h", "src/util.cc", "src/widget.hpp",
            "src/a.cxx", "src/b.hxx", "src/contest.cpp", "src/mytest.cpp", "src/test.cpp",
        ],
        "drop_test_names": [
            "src/x_test.cpp", "src/y_test.cc", "src/test_z.cpp", "src/test_w.cc",
            "src/test_v.h", "src/TestX.cpp", "src/TestX.h",
        ],
        "drop_test_dirs": ["test/helpers.cpp", "tests/helpers.cpp", "spec/helpers.cpp", "src/tests/helpers.h"],
        "drop_skip_dirs": ["vendor/v.cpp", "third_party/t.cc", "build/b.cpp"],
    },
}

LANGUAGES = list(FIXTURES)


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def build_repo(root: Path, language: str) -> tuple[int, int]:
    """Write the fixture repo; return the (functions, classes) that are really eligible."""
    spec = FIXTURES[language]
    for group, files in spec.items():
        for rel in files:
            if group == "drop_minified_content":
                text = "const f_bundle = () => 1;\n/*" + "x" * 1200 + "*/\n"
            elif group == "drop_other_extensions":
                text = "export const f = (x: number): number => x;\n"
            else:
                text = source_for(language, rel)
            write(root, rel, text)
    n = len(spec["keep"])
    return n * FUNCS_PER_FILE, n * CLASSES_PER_FILE


def count(root: Path, language: str, granularities=BOTH) -> tuple[int, int]:
    return ex.count_repo_contents(str(root), language, granularities)


def extract(root: Path, language: str, granularities=BOTH, **kw):
    return ex.extract_from_repo(str(root), language, "o/r", granularities, **kw)


# ─── 1. The shared predicate ──────────────────────────────────────────────────

class TestIsTestFile:
    @pytest.mark.parametrize("name", [
        # Python
        "test_app.py", "app_test.py", "TestApp.py", "Testing.py",
        # Java
        "TestApp.java", "AppTest.java", "TestUtils.java", "ServiceTest.java",
        # JavaScript / TypeScript
        "app.test.js", "app.spec.js", "app.test.ts", "app.spec.ts", "test_helpers.js", "TestWidget.js",
        # C++
        "x_test.cpp", "y_test.cc", "test_z.cpp", "test_w.cc", "test_v.h", "TestX.cpp", "TestX.h",
        # directories in front of the name never matter
        "src/main/TestApp.java", "a/b/c/app.spec.js",
    ])
    def test_test_names_are_recognised(self, name):
        assert ex.is_test_file(name) is True
        assert ex.is_test_file(Path(name)) is True

    @pytest.mark.parametrize("name", [
        "app.py", "contest.py", "mytest.py", "latest.py", "testing_utils.py",
        "App.java", "Contest.java", "Latest.java", "Attest.java",
        "app.js", "test-utils.js", "widget.testing.js", "view.jsx",
        "main.cpp", "util.cc", "engine.h", "test.cpp", "mytest.cpp",
    ])
    def test_ordinary_names_are_not(self, name):
        assert ex.is_test_file(name) is False

    def test_only_the_file_name_is_judged_not_its_directories(self):
        assert ex.is_test_file("tests/helpers.py") is False
        assert ex.is_test_file("src/TestDir/helpers.py") is False


# ─── 2. The invariant: Pass 1 count == Pass 2 records ─────────────────────────

@pytest.fixture(params=LANGUAGES)
def repo(request, tmp_path):
    language = request.param
    expected = build_repo(tmp_path, language)
    return language, tmp_path, expected


class TestPass1MatchesPass2:
    def test_extraction_returns_exactly_the_eligible_records(self, repo):
        """Guards the fixture itself: Pass 2 keeps the 'keep' files and nothing else."""
        language, root, expected = repo
        funcs, classes = extract(root, language)
        assert (len(funcs), len(classes)) == expected

    def test_count_equals_extraction_functions_and_classes(self, repo):
        language, root, expected = repo
        funcs, classes = extract(root, language)
        assert count(root, language) == (len(funcs), len(classes))

    def test_count_equals_the_eligible_records(self, repo):
        language, root, expected = repo
        assert count(root, language) == expected

    @pytest.mark.parametrize("granularities", [["function"], ["class"], ["function", "class"]])
    def test_granularity_is_respected_the_same_way(self, repo, granularities):
        language, root, expected = repo
        funcs, classes = extract(root, language, granularities)
        assert count(root, language, granularities) == (len(funcs), len(classes))

    @pytest.mark.parametrize("func_cap,class_cap", [
        (None, None), (0, 0), (1, 1), (3, 2), (5, 4), (1000, 1000), (None, 1), (2, None),
    ])
    def test_with_caps_extraction_returns_min_of_cap_and_count(self, repo, func_cap, class_cap):
        language, root, expected = repo
        n_funcs, n_classes = count(root, language)
        funcs, classes = extract(root, language, func_cap=func_cap, class_cap=class_cap)
        assert len(funcs) == (n_funcs if func_cap is None else min(func_cap, n_funcs))
        assert len(classes) == (n_classes if class_cap is None else min(class_cap, n_classes))

    def test_no_test_named_file_survives_in_the_records(self, repo):
        language, root, expected = repo
        funcs, classes = extract(root, language)
        for r in funcs + classes:
            assert not ex.is_test_file(r.file_path), r.file_path

    def test_both_passes_select_files_through_one_helper(self, repo):
        """Every record comes from a file the shared walker yields, and the walker yields no test file."""
        language, root, expected = repo
        walked = {str(p.relative_to(root)) for p in ex._iter_source_files(
            root, ex.LANGUAGE_EXTENSIONS[language], language=language)}
        assert all(not ex.is_test_file(p) for p in walked)
        funcs, classes = extract(root, language)
        assert {r.file_path for r in funcs + classes} <= walked
        # The walker yields the keep files, plus files that only the content
        # rule (minified JS) rejects afterwards, in both passes.
        spec = FIXTURES[language]
        assert walked == set(spec["keep"]) | set(spec.get("drop_minified_content", []))

    def test_count_skipped_files_is_not_changed_by_test_name_filtering(self, repo):
        """The skip-list count lists skipped trees by extension only; test names never enter into it."""
        language, root, expected = repo
        n = ex.count_skipped_files(str(root), language)
        spec = FIXTURES[language]
        expected_skipped = len(spec["drop_skip_dirs"])
        assert n == expected_skipped


class TestRobustness:
    """Awkward files must not make the two passes disagree."""

    # A file with a syntax error.
    BROKEN = {
        "Python": "def f(:\n  pass\nclass A:\n    def m(self): pass\ndef ok(): pass\n",
        "Java": "public class A { public void m( { } \n public int ok() { return 1; }\n}\n",
        "C++": "class A { int m( { }; \nint ok() { return 1; }\n",
        "JavaScript": walk_cases.JS_BROKEN,
    }
    EXT = {"Python": "py", "Java": "java", "JavaScript": "js", "C++": "cpp"}

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_syntax_errors_empty_and_non_utf8_files(self, tmp_path, language):
        ext = self.EXT[language]
        if language in self.BROKEN:
            write(tmp_path, f"broken.{ext}", self.BROKEN[language])
        write(tmp_path, f"empty.{ext}", "")
        (tmp_path / f"binary.{ext}").write_bytes(b"\xff\xfe\x00\x00 garbage \x80\x81")
        write(tmp_path, f"fine.{ext}", source_for(language, f"fine.{ext}"))
        funcs, classes = extract(tmp_path, language)
        assert count(tmp_path, language) == (len(funcs), len(classes))

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_a_directory_named_like_a_source_file_is_ignored_by_both(self, tmp_path, language):
        ext = self.EXT[language]
        (tmp_path / f"dir.{ext}").mkdir()
        write(tmp_path, f"fine.{ext}", source_for(language, f"fine.{ext}"))
        funcs, classes = extract(tmp_path, language)
        assert (len(funcs), len(classes)) == (FUNCS_PER_FILE, CLASSES_PER_FILE)
        assert count(tmp_path, language) == (FUNCS_PER_FILE, CLASSES_PER_FILE)

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_a_repo_with_only_ineligible_files_counts_zero(self, tmp_path, language):
        ext = self.EXT[language]
        test_name = {"Python": "test_a.py", "Java": "TestA.java", "JavaScript": "a.test.js",
                     "C++": "a_test.cpp"}[language]
        write(tmp_path, test_name, source_for(language, test_name))
        write(tmp_path, f"tests/b.{ext}", source_for(language, f"b.{ext}"))
        assert count(tmp_path, language) == (0, 0)
        assert extract(tmp_path, language) == ([], [])


# ─── 3. How the Pass 1 counts feed the stratified caps ────────────────────────

JAVA_REAL = [f"src/main/Real{i}.java" for i in range(4)]            # 4 files -> 8 funcs, 4 classes
JAVA_TESTS = ([f"src/main/Test{i}.java" for i in range(3)]          # 6 test files -> 12 funcs, 6 classes
              + [f"src/main/Thing{i}Test.java" for i in range(3)])


def build_java_pair(root: Path) -> dict[str, Path]:
    """Repo A: real code plus a lot of test-named files. Repo B: real code only, same amount."""
    a, b = root / "A", root / "B"
    for rel in JAVA_REAL + JAVA_TESTS:
        write(a, rel, source_for("Java", rel))
    for rel in JAVA_REAL:
        write(b, rel, source_for("Java", rel))
    return {"o/A": a, "o/B": b}


class TestStratifiedAllocationMatchesReality:
    def test_test_files_no_longer_inflate_the_repo_that_has_them(self, tmp_path):
        repos = build_java_pair(tmp_path)
        counts = {name: count(path, "Java") for name, path in repos.items()}
        # Both repos really offer 8 functions and 4 classes.
        assert counts == {"o/A": (8, 4), "o/B": (8, 4)}

        func_caps = _stratified_caps({n: c[0] for n, c in counts.items()}, 14)
        class_caps = _stratified_caps({n: c[1] for n, c in counts.items()}, 6)
        # Equal supply, equal share.
        assert func_caps == {"o/A": 7, "o/B": 7}
        assert class_caps == {"o/A": 3, "o/B": 3}

    def test_every_allocated_record_can_be_extracted_no_shortfall(self, tmp_path):
        repos = build_java_pair(tmp_path)
        counts = {name: count(path, "Java") for name, path in repos.items()}
        func_caps = _stratified_caps({n: c[0] for n, c in counts.items()}, 14)
        class_caps = _stratified_caps({n: c[1] for n, c in counts.items()}, 6)

        got_funcs = got_classes = 0
        for name, path in repos.items():
            funcs, classes = extract(path, "Java", func_cap=func_caps[name], class_cap=class_caps[name])
            assert len(funcs) == func_caps[name], f"{name}: allocated more than Pass 2 can deliver"
            assert len(classes) == class_caps[name], f"{name}: allocated more than Pass 2 can deliver"
            got_funcs += len(funcs)
            got_classes += len(classes)
        assert (got_funcs, got_classes) == (14, 6)

    def test_run_pipeline_computes_caps_from_the_eligible_counts(self, tmp_path):
        """End to end through run_pipeline: the counts handed to _stratified_caps are the real ones."""
        trees = {"o/A": JAVA_REAL + JAVA_TESTS, "o/B": JAVA_REAL}
        config = SIEVEConfig(language="Java", start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
                             output_dir=str(tmp_path / "out"), github_token="x",
                             max_functions=14, max_classes=6)

        def meta(name):
            return RepoMetadata(
                full_name=name, url=f"https://github.com/{name}", stars=100, contributors=5,
                last_commit_date="2024-01-01T00:00:00", default_branch="main", license_spdx="MIT",
                language="Java", collected_at="2024-06-01T00:00:00+00:00", topics=[],
            )

        def fake_discover(**kwargs):
            for name in trees:
                yield meta(name)

        def fake_clone(repo_full_name, target_dir):
            for rel in trees[repo_full_name]:
                write(Path(target_dir), rel, source_for("Java", rel))
            return True

        seen = []
        real = _stratified_caps

        def spy(counts, total_cap):
            result = real(counts, total_cap)
            seen.append((dict(counts), total_cap, result))
            return result

        with patch("sieve.pipeline.discover_repos", side_effect=fake_discover), \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone), \
             patch("sieve.pipeline._stratified_caps", side_effect=spy), \
             patch("sieve.pipeline.time.sleep"):
            run_pipeline(config)

        by_cap = {total: (counts, caps) for counts, total, caps in seen}
        assert by_cap[14][0] == {"o/A": 8, "o/B": 8}
        assert by_cap[14][1] == {"o/A": 7, "o/B": 7}
        assert by_cap[6][0] == {"o/A": 4, "o/B": 4}
        assert by_cap[6][1] == {"o/A": 3, "o/B": 3}


# ─── 4. Node-level cases: Pass 1 == Pass 2 ───────────────────────────────────
#
# Every case below used to make the two passes disagree on a file that BOTH
# treated as eligible (Pass 1 counted raw tree-sitter node types; the extractors
# walk the tree their own way). Both now consume one parse-only walk per
# language (extraction._find_definitions), so they agree by construction; these
# tests keep it that way. The expected pair is what Pass 2 returned before the
# refactor, so agreement cannot come from both being wrong together.

class TestNodeLevelCasesAgree:
    @pytest.mark.parametrize("language,name,source,expected", walk_cases.CASE_PARAMS)
    def test_pass_2_returns_the_pinned_records(self, tmp_path, language, name, source, expected):
        write(tmp_path, name, source)
        funcs, classes = extract(tmp_path, language)
        assert (len(funcs), len(classes)) == expected

    @pytest.mark.parametrize("language,name,source,expected", walk_cases.CASE_PARAMS)
    def test_pass_1_equals_pass_2(self, tmp_path, language, name, source, expected):
        write(tmp_path, name, source)
        funcs, classes = extract(tmp_path, language)
        assert count(tmp_path, language) == (len(funcs), len(classes)) == expected

    @pytest.mark.parametrize("language,name,source,expected", walk_cases.CASE_PARAMS)
    @pytest.mark.parametrize("granularities", [["function"], ["class"], ["function", "class"]])
    def test_every_granularity_setting(self, tmp_path, language, name, source, expected, granularities):
        write(tmp_path, name, source)
        funcs, classes = extract(tmp_path, language, granularities)
        counted = count(tmp_path, language, granularities)
        assert counted == (len(funcs), len(classes))
        assert counted == (expected[0] if "function" in granularities else 0,
                           expected[1] if "class" in granularities else 0)

    @pytest.mark.parametrize("language,name,source,expected", walk_cases.CASE_PARAMS)
    @pytest.mark.parametrize("func_cap,class_cap", [(None, None), (0, 0), (1, 1), (2, None), (None, 1), (1000, 1000)])
    def test_with_caps_pass_2_returns_min_of_cap_and_pass_1(self, tmp_path, language, name, source, expected,
                                                            func_cap, class_cap):
        write(tmp_path, name, source)
        n_funcs, n_classes = count(tmp_path, language)
        funcs, classes = extract(tmp_path, language, func_cap=func_cap, class_cap=class_cap)
        assert len(funcs) == (n_funcs if func_cap is None else min(func_cap, n_funcs))
        assert len(classes) == (n_classes if class_cap is None else min(class_cap, n_classes))

    @pytest.mark.parametrize("language,name,source,expected", walk_cases.CASE_PARAMS)
    def test_nothing_is_counted_that_pass_2_cannot_deliver(self, tmp_path, language, name, source, expected):
        """Pass 1 is never an overestimate: a cap equal to the Pass 1 count is always filled."""
        write(tmp_path, name, source)
        n_funcs, n_classes = count(tmp_path, language)
        funcs, classes = extract(tmp_path, language, func_cap=n_funcs, class_cap=n_classes)
        assert (len(funcs), len(classes)) == (n_funcs, n_classes)


# ─── 5. JavaScript: callbacks, nested arrows and function declarations ────────
#
# Before the walk was shared, Pass 1 counted the `function` keyword token as a
# second function, and every callback and nested arrow, none of which Pass 2
# extracts. A JS repo full of them looked 2-3x bigger than it was, so
# _stratified_caps gave it a share Pass 2 could not fill.

JS_CALLBACK_HEAVY = """\
function load{n}(items) {{
  return items.map(i => i.id).filter(id => id > 0).reduce((a, b) => a + b, 0);
}}
function save{n}(rows) {{
  const norm = rows.map(r => r.trim());
  return () => norm;
}}
const handler{n} = (req, res) => {{ res.on('data', chunk => chunk); }};
"""
# 3 functions per file that Pass 2 extracts (load, save, handler); the callbacks,
# `norm` and the returned arrow are inside them and are not.

JS_CLASS_ONLY = """\
class Svc{n} {{
  a() {{ return 1; }}
  b() {{ return 2; }}
  c() {{ return 3; }}
}}
"""
# 3 methods and 1 class per file, counted identically by the old and new Pass 1.

JS_TREES = {
    "o/callbacks": {f"src/cb{i}.js": JS_CALLBACK_HEAVY.format(n=i) for i in range(2)},   # 6 real functions
    "o/classes":   {f"src/cls{i}.js": JS_CLASS_ONLY.format(n=i) for i in range(2)},      # 6 real functions
}


def _build_js_repos(root: Path) -> dict[str, Path]:
    repos = {}
    for name, files in JS_TREES.items():
        repo = root / name.split("/")[1]
        for rel, text in files.items():
            write(repo, rel, text)
        repos[name] = repo
    return repos


class TestStratifiedAllocationJavaScript:
    def test_pass_1_counts_what_pass_2_extracts(self, tmp_path):
        repos = _build_js_repos(tmp_path)
        counts = {n: count(p, "JavaScript") for n, p in repos.items()}
        assert counts == {"o/callbacks": (6, 0), "o/classes": (6, 2)}
        for name, path in repos.items():
            funcs, classes = extract(path, "JavaScript")
            assert counts[name] == (len(funcs), len(classes))

    def test_equal_supply_gets_equal_caps_and_pass_2_fills_every_one(self, tmp_path):
        repos = _build_js_repos(tmp_path)
        counts = {n: count(p, "JavaScript") for n, p in repos.items()}
        func_caps = _stratified_caps({n: c[0] for n, c in counts.items()}, 8)
        class_caps = _stratified_caps({n: c[1] for n, c in counts.items()}, 2)
        # The two repos really offer 6 functions each, so they split 8 evenly.
        # (With the old inflated counts the callback repo looked about 3x larger
        # and was allotted 6 of the 8, more than it has.)
        assert func_caps == {"o/callbacks": 4, "o/classes": 4}
        assert class_caps == {"o/callbacks": 0, "o/classes": 2}

        got_funcs = got_classes = 0
        for name, path in repos.items():
            funcs, classes = extract(path, "JavaScript", func_cap=func_caps[name], class_cap=class_caps[name])
            assert len(funcs) == func_caps[name], f"{name}: allocated more than Pass 2 can deliver"
            assert len(classes) == class_caps[name], f"{name}: allocated more than Pass 2 can deliver"
            got_funcs += len(funcs)
            got_classes += len(classes)
        assert (got_funcs, got_classes) == (8, 2)

    def test_a_cap_above_the_supply_is_allocated_in_full_never_beyond_it(self, tmp_path):
        repos = _build_js_repos(tmp_path)
        counts = {n: count(p, "JavaScript")[0] for n, p in repos.items()}
        caps = _stratified_caps(counts, 1000)
        assert caps == {"o/callbacks": 6, "o/classes": 6}
        for name, path in repos.items():
            assert len(extract(path, "JavaScript", func_cap=caps[name])[0]) == caps[name]

    def test_run_pipeline_computes_caps_from_the_extractable_counts(self, tmp_path):
        """End to end: the counts handed to _stratified_caps are what Pass 2 can deliver."""
        config = SIEVEConfig(language="JavaScript", start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
                             output_dir=str(tmp_path / "out"), github_token="x",
                             max_functions=8, max_classes=2)

        def meta(name):
            return RepoMetadata(
                full_name=name, url=f"https://github.com/{name}", stars=100, contributors=5,
                last_commit_date="2024-01-01T00:00:00", default_branch="main", license_spdx="MIT",
                language="JavaScript", collected_at="2024-06-01T00:00:00+00:00", topics=[],
            )

        def fake_discover(**kwargs):
            for name in JS_TREES:
                yield meta(name)

        def fake_clone(repo_full_name, target_dir):
            for rel, text in JS_TREES[repo_full_name].items():
                write(Path(target_dir), rel, text)
            return True

        seen = []
        real = _stratified_caps

        def spy(counts, total_cap):
            result = real(counts, total_cap)
            seen.append((dict(counts), total_cap, result))
            return result

        with patch("sieve.pipeline.discover_repos", side_effect=fake_discover), \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone), \
             patch("sieve.pipeline._stratified_caps", side_effect=spy), \
             patch("sieve.pipeline.time.sleep"):
            run_pipeline(config)

        by_cap = {total: (counts, caps) for counts, total, caps in seen}
        assert by_cap[8][0] == {"o/callbacks": 6, "o/classes": 6}
        assert by_cap[8][1] == {"o/callbacks": 4, "o/classes": 4}
        assert by_cap[2][0] == {"o/callbacks": 0, "o/classes": 2}
        assert by_cap[2][1] == {"o/callbacks": 0, "o/classes": 2}
