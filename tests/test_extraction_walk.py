"""
tests/test_extraction_walk.py

Each language has ONE parse-only walk (extraction._find_definitions) that yields
the function and class nodes the extractor turns into records. Pass 2
(extract_from_file) builds its records from those nodes and Pass 1
(count_repo_contents) counts them, so the two cannot disagree about what a
file contains. These tests pin that contract:

  1. golden records: Pass 2 output is unchanged by the refactor. The golden
     file was generated from the code BEFORE the walk was factored out
     (commit 57b9763), for one rich sample file per language plus every case in
     tests/walk_cases.py.
  2. the walk and the records agree node for node (kind, order, line span),
     including on files the extractors only partly understand,
  3. Pass 1 is parse-only: it never builds a record, computes a metric or an
     AST feature, or collects imports,
  4. both passes really do go through the one walk,
  5. the one gap that is left on purpose (a record that cannot be built).

Regenerating the golden file is only legitimate when a change to the RECORDS
(not to which nodes are found) is intended:

    python tests/test_extraction_walk.py --regen
"""

import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import walk_cases  # noqa: E402
from sieve.core import extraction as ex  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "walk"
GOLDEN = FIXTURE_DIR / "golden_records.json"
SAMPLES = {
    "Python": "sample.py",
    "Java": "Sample.java",
    "JavaScript": "sample.js",
    "C++": "sample.cpp",
}
BOTH = ["function", "class"]
# Skipped in the golden file only (100 records of quadratic size); the walk
# agreement tests below still run it.
NOT_IN_GOLDEN = {"python-nesting-deeper-than-the-extractor-depth-guard"}


def _records(language: str, path: Path, rel: str):
    funcs, classes = ex.extract_from_file(str(path), language, "o/r", relative_path=rel)
    return {
        "functions": [dataclasses.asdict(r) for r in funcs],
        "classes": [dataclasses.asdict(r) for r in classes],
    }


def _golden_inputs(tmp: Path):
    for language, name in SAMPLES.items():
        yield f"sample:{name}", language, FIXTURE_DIR / name, name
    for cid, language, name, source, _ in walk_cases.CASES:
        if cid in NOT_IN_GOLDEN:
            continue
        path = tmp / f"{cid}__{name}"
        path.write_text(source)
        yield f"case:{cid}", language, path, name


def _same(actual, expected, where="record"):
    """Exact equality, except that floats are compared to 1e-9 relative: the Halstead
    metrics use math.log2, whose last bit may differ between platforms."""
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), where
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), where
        for k in expected:
            _same(actual[k], expected[k], f"{where}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected)):
            _same(a, e, f"{where}[{i}]")
    else:
        assert actual == expected, where


# ─── 1. Pass 2 output is unchanged ────────────────────────────────────────────

GOLDEN_DATA = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


class TestGoldenRecords:
    def test_the_golden_file_covers_every_input(self, tmp_path):
        keys = {key for key, *_ in _golden_inputs(tmp_path)}
        assert keys == set(GOLDEN_DATA)

    @pytest.mark.parametrize("key", sorted(GOLDEN_DATA))
    def test_records_are_identical_to_the_pre_refactor_output(self, key, tmp_path):
        inputs = {k: (lang, path, rel) for k, lang, path, rel in _golden_inputs(tmp_path)}
        language, path, rel = inputs[key]
        actual = _records(language, path, rel)
        expected = GOLDEN_DATA[key]
        assert len(actual["functions"]) == len(expected["functions"])
        assert len(actual["classes"]) == len(expected["classes"])
        _same(actual, expected, key)

    @pytest.mark.parametrize("language,name", sorted(SAMPLES.items()))
    def test_the_samples_are_rich_enough_to_mean_something(self, language, name):
        funcs, classes = ex.extract_from_file(
            str(FIXTURE_DIR / name), language, "o/r", relative_path=name)
        assert len(funcs) >= 10 and len(classes) >= 2


# ─── 2. The walk and the records agree node for node ──────────────────────────

def _all_sources():
    for language, name in SAMPLES.items():
        yield f"sample:{name}", language, (FIXTURE_DIR / name).read_text()
    for cid, language, name, source, _ in walk_cases.CASES:
        yield f"case:{cid}", language, source


def _span(node_or_record):
    if hasattr(node_or_record, "start_point"):
        return node_or_record.start_point[0] + 1, node_or_record.end_point[0] + 1
    return node_or_record.start_line, node_or_record.end_line


_EXTRACTORS = {
    "Python": ex._extract_python,
    "Java": ex._extract_java,
    "JavaScript": ex._extract_js,
    "C++": ex._extract_cpp,
}


class TestWalkMatchesRecords:
    @pytest.mark.parametrize("key,language,source", [
        pytest.param(k, lang, src, id=k)
        for k, lang, src in _all_sources()
    ])
    def test_every_definition_becomes_exactly_one_record_in_the_same_order(self, key, language, source):
        definitions = ex._parse_definitions(source, language)
        funcs, classes = _EXTRACTORS[language](source, "f", "o/r")
        assert [_span(d.node) for d in definitions if d.kind == "function"] == [_span(r) for r in funcs]
        assert [_span(d.node) for d in definitions if d.kind == "class"] == [_span(r) for r in classes]

    @pytest.mark.parametrize("language", sorted(SAMPLES))
    def test_the_walk_yields_only_function_and_class_kinds(self, language):
        source = (FIXTURE_DIR / SAMPLES[language]).read_text()
        kinds = {d.kind for d in ex._parse_definitions(source, language)}
        assert kinds == {"function", "class"}

    @pytest.mark.parametrize("key,language,source", [
        pytest.param(k, lang, src, id=k)
        for k, lang, src in _all_sources()
    ])
    def test_a_definition_carries_the_class_it_sits_in(self, key, language, source):
        """is_method / parent_class in the records come straight from the walk."""
        definitions = ex._parse_definitions(source, language)
        funcs, _ = _EXTRACTORS[language](source, "f", "o/r")
        walked = [d.parent_class for d in definitions if d.kind == "function"]
        assert len(walked) == len(funcs)
        for record, parent in zip(funcs, walked):
            if language == "C++" and parent is None:
                # An out-of-class definition (`void A::f() {}`) takes its class from
                # the qualifier when the walk has no enclosing class.
                continue
            assert record.parent_class == parent
            assert record.is_method == (parent is not None)

    def test_unsupported_language_has_no_walk(self):
        with pytest.raises(ValueError):
            ex._parse_definitions("x", "Cobol")


# ─── 3. Pass 1 is parse-only ──────────────────────────────────────────────────

def _boom(name):
    def raiser(*a, **k):
        raise AssertionError(f"Pass 1 must not call {name}")
    return raiser


# Everything Pass 2 does AFTER it knows which nodes there are.
_FORBIDDEN = [
    "FunctionRecord", "ClassRecord",
    "_compute_ast_features", "_filter_used_imports", "_clean_indentation",
    "_collect_python_imports", "_collect_java_imports", "_collect_js_imports", "_collect_cpp_includes",
    "_build_python_skeleton", "_build_java_skeleton", "_build_js_skeleton", "_build_cpp_skeleton",
    "_extract_python", "_extract_java", "_extract_js", "_extract_cpp",
    "extract_from_file",
]


class TestPassOneIsParseOnly:
    @pytest.mark.parametrize("language", sorted(SAMPLES))
    def test_counting_builds_no_records_and_computes_no_metrics(self, language, tmp_path):
        (tmp_path / SAMPLES[language]).write_text((FIXTURE_DIR / SAMPLES[language]).read_text())
        expected = (len(_records(language, tmp_path / SAMPLES[language], SAMPLES[language])["functions"]),
                    len(_records(language, tmp_path / SAMPLES[language], SAMPLES[language])["classes"]))

        patches = [patch.object(ex, name, _boom(name)) for name in _FORBIDDEN]
        patches.append(patch("sieve.core.metrics.compute_metrics", _boom("compute_metrics")))
        for p in patches:
            p.start()
        try:
            assert ex.count_repo_contents(str(tmp_path), language, BOTH) == expected
        finally:
            for p in patches:
                p.stop()

    def test_the_walk_decodes_nothing_but_class_names(self):
        """Source text is only read where the walk needs a name to pass to nested definitions."""
        source = (FIXTURE_DIR / "sample.js").read_text()
        decoded = []
        real = ex._node_text

        def spy(node, source_bytes):
            decoded.append(node.type)
            return real(node, source_bytes)

        with patch.object(ex, "_node_text", spy):
            definitions = ex._parse_definitions(source, "JavaScript")
        n_classes = sum(1 for d in definitions if d.kind == "class")
        assert len(decoded) <= n_classes
        assert set(decoded) <= {"identifier"}


# ─── 4. Both passes go through the one walk ───────────────────────────────────

class TestOneWalkForBothPasses:
    @pytest.mark.parametrize("language", sorted(SAMPLES))
    def test_pass_2_builds_records_only_from_what_the_walk_yields(self, language, tmp_path):
        """Hide the first function from the walk: Pass 2 and Pass 1 both lose exactly it."""
        name = SAMPLES[language]
        (tmp_path / name).write_text((FIXTURE_DIR / name).read_text())
        funcs0, classes0 = ex.extract_from_repo(str(tmp_path), language, "o/r", BOTH)
        real = ex._find_definitions

        def without_first_function(lang, root, source_bytes):
            definitions = real(lang, root, source_bytes)
            first = next(i for i, d in enumerate(definitions) if d.kind == "function")
            return definitions[:first] + definitions[first + 1:]

        with patch.object(ex, "_find_definitions", without_first_function):
            funcs1, classes1 = ex.extract_from_repo(str(tmp_path), language, "o/r", BOTH)
            counted = ex.count_repo_contents(str(tmp_path), language, BOTH)
        assert (len(funcs1), len(classes1)) == (len(funcs0) - 1, len(classes0))
        assert counted == (len(funcs1), len(classes1))
        assert [r.start_line for r in funcs1] == [r.start_line for r in funcs0][1:]

    @pytest.mark.parametrize("language", sorted(SAMPLES))
    def test_each_file_is_walked_once_per_pass(self, language, tmp_path):
        name = SAMPLES[language]
        (tmp_path / name).write_text((FIXTURE_DIR / name).read_text())
        calls = []
        real = ex._find_definitions

        def spy(lang, root, source_bytes):
            calls.append(lang)
            return real(lang, root, source_bytes)

        with patch.object(ex, "_find_definitions", spy):
            ex.count_repo_contents(str(tmp_path), language, BOTH)
            assert calls == [language]
            ex.extract_from_repo(str(tmp_path), language, "o/r", BOTH)
        assert calls == [language, language]


# ─── 5. The one gap left, on purpose ──────────────────────────────────────────

class TestTheOnlyRemainingGap:
    """
    Pass 1 runs the walk; Pass 2 runs the walk and then builds records. If the
    walk succeeds but building a record raises, extract_from_file drops the whole
    file (it logs a warning and returns nothing) while Pass 1 has already counted
    its definitions. No input known to SIEVE triggers this: it needs a bug in a
    record builder, and Pass 1 deliberately does not run the builders to find out
    (that would make it as expensive as Pass 2). The walk itself cannot make the
    passes disagree, because both run the same one. This test pins the behavior
    so the gap stays known and documented rather than silent.
    """

    def test_a_file_whose_records_cannot_be_built_is_counted_by_pass_1_only(self, tmp_path):
        (tmp_path / "a.py").write_text("def f():\n    pass\n\n\ndef g():\n    pass\n")
        with patch.object(ex, "_build_python_function_signature", side_effect=RuntimeError("injected")):
            assert ex.extract_from_repo(str(tmp_path), "Python", "o/r", BOTH) == ([], [])
            assert ex.count_repo_contents(str(tmp_path), "Python", BOTH) == (2, 0)
        # And without the injected failure the two agree again.
        funcs, classes = ex.extract_from_repo(str(tmp_path), "Python", "o/r", BOTH)
        assert ex.count_repo_contents(str(tmp_path), "Python", BOTH) == (len(funcs), len(classes)) == (2, 0)


if __name__ == "__main__":  # pragma: no cover
    if "--regen" not in sys.argv:
        sys.exit(__doc__)
    import tempfile
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for key, language, path, rel in _golden_inputs(Path(d)):
            out[key] = _records(language, path, rel)
    GOLDEN.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {GOLDEN} ({len(out)} inputs)")
