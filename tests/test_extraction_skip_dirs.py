"""
tests/test_extraction_skip_dirs.py

Code extraction skips vendored, generated and virtualenv directories, prunes
them during the walk instead of filtering afterwards, and judges test
directories on the path *relative to the repo root*.

Small fake repo trees are built in tmp_path; nothing touches the network.
"""

import json
import os
from collections import Counter
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from sieve.config import SIEVEConfig
from sieve.core import extraction as ex
from sieve.core.discovery import RepoMetadata
from sieve.core.export import write_manifest
from sieve.pipeline import run_pipeline

# Spelled out on purpose: the test must fail if the list changes by accident.
SKIPPED = [
    ".eggs", ".tox", ".venv", "build", "dist", "node_modules", "site-packages",
    "third-party", "third_party", "vendor", "vendored", "venv",
]

FUNC = "def {name}():\n    return 1\n"
KLASS = "class {name}:\n    pass\n"


def write(root: Path, rel: str, text: str = None) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = Path(rel).stem.replace("-", "_").replace(".", "_")
    path.write_text(text if text is not None else FUNC.format(name=f"f_{stem}") + KLASS.format(name=f"C_{stem}"))
    return path


def extract(root: Path, **kwargs):
    return ex.extract_from_repo(str(root), "Python", "o/r", ["function", "class"], **kwargs)


def func_paths(root: Path) -> list[str]:
    funcs, _ = extract(root)
    return [f.file_path for f in funcs]


def class_paths(root: Path) -> list[str]:
    _, classes = extract(root)
    return [c.file_path for c in classes]


# ─── 1. The skip list ─────────────────────────────────────────────────────────

class TestSkipList:
    def test_exact_membership(self):
        assert ex.EXTRACTION_SKIP_DIRS == frozenset(SKIPPED)
        assert isinstance(ex.EXTRACTION_SKIP_DIRS, frozenset)

    @pytest.mark.parametrize("name", ["external", "env"])
    def test_external_and_env_are_deliberately_not_in_the_list(self, name):
        assert name not in ex.EXTRACTION_SKIP_DIRS


# ─── 2. Every skipped name, both walks, functions and classes ─────────────────

class TestEachSkippedDirectory:
    @pytest.mark.parametrize("name", SKIPPED)
    def test_extract_from_repo_skips_functions_and_classes(self, tmp_path, name):
        write(tmp_path, "src/app.py")
        write(tmp_path, f"{name}/lib.py")
        write(tmp_path, f"src/{name}/inner.py")
        assert func_paths(tmp_path) == ["src/app.py"]
        assert class_paths(tmp_path) == ["src/app.py"]

    @pytest.mark.parametrize("name", SKIPPED)
    def test_count_repo_contents_skips_it_too(self, tmp_path, name):
        write(tmp_path, "src/app.py")
        write(tmp_path, f"{name}/lib.py")
        write(tmp_path, f"src/{name}/inner.py")
        assert ex.count_repo_contents(str(tmp_path), "Python", ["function", "class"]) == (1, 1)

    @pytest.mark.parametrize("name", SKIPPED)
    def test_both_walks_agree(self, tmp_path, name):
        write(tmp_path, "a.py")
        write(tmp_path, f"{name}/b.py")
        funcs, classes = extract(tmp_path)
        assert ex.count_repo_contents(str(tmp_path), "Python", ["function", "class"]) == (len(funcs), len(classes))

    def test_nested_skipped_directories(self, tmp_path):
        write(tmp_path, "keep/ok.py")
        write(tmp_path, "keep/vendor/node_modules/deep/x.py")
        write(tmp_path, "pkg/build/vendor/y.py")
        write(tmp_path, "venv/lib/python3.11/site-packages/z.py")
        assert func_paths(tmp_path) == ["keep/ok.py"]
        assert ex.count_repo_contents(str(tmp_path), "Python", ["function", "class"]) == (1, 1)

    def test_a_file_named_like_a_skipped_directory_is_still_extracted(self, tmp_path):
        write(tmp_path, "src/vendor.py")
        write(tmp_path, "src/build.py")
        assert sorted(func_paths(tmp_path)) == ["src/build.py", "src/vendor.py"]

    def test_names_must_match_exactly(self, tmp_path):
        write(tmp_path, "vendors/a.py")
        write(tmp_path, "my_build/b.py")
        write(tmp_path, "Vendor/c.py")
        assert sorted(func_paths(tmp_path)) == ["Vendor/c.py", "my_build/b.py", "vendors/a.py"]

    @pytest.mark.parametrize("language, ext, code", [
        ("JavaScript", ".js", "function f() { return 1; }\nclass C {}\n"),
        ("Java", ".java", "class C { void m() {} }\n"),
        ("C++", ".cpp", "int f() { return 1; }\nclass C {};\n"),
    ])
    def test_other_languages(self, tmp_path, language, ext, code):
        write(tmp_path, f"src/app{ext}", code)
        write(tmp_path, f"node_modules/dep/lib{ext}", code)
        write(tmp_path, f"dist/bundle{ext}", code)
        write(tmp_path, f"build/out{ext}", code)
        funcs, classes = ex.extract_from_repo(str(tmp_path), language, "o/r", ["function", "class"])
        assert {r.file_path for r in funcs + classes} == {f"src/app{ext}"}


class TestExternalAndEnvAreReal:
    def test_external_and_env_are_extracted(self, tmp_path):
        write(tmp_path, "external/mod.py")
        write(tmp_path, "src/env/Config.py")
        write(tmp_path, "env/settings.py")
        assert sorted(func_paths(tmp_path)) == ["env/settings.py", "external/mod.py", "src/env/Config.py"]
        assert sorted(class_paths(tmp_path)) == ["env/settings.py", "external/mod.py", "src/env/Config.py"]
        assert ex.count_repo_contents(str(tmp_path), "Python", ["function", "class"]) == (3, 3)

    def test_a_java_package_called_env(self, tmp_path):
        write(tmp_path, "src/main/java/com/acme/env/Env.java", "class Env { void load() {} }\n")
        funcs, classes = ex.extract_from_repo(str(tmp_path), "Java", "o/r", ["function", "class"])
        assert [r.file_path for r in funcs] == ["src/main/java/com/acme/env/Env.java"]
        assert len(classes) == 1


# ─── 3. Test-directory exclusion is judged on the relative path ───────────────

class TestRelativePathFiltering:
    @pytest.mark.parametrize("parent", ["tests", "test", "spec", "__tests__"])
    def test_clone_under_a_tests_parent_is_still_extracted(self, tmp_path, parent):
        repo = tmp_path / parent / "owner" / "repo"
        write(repo, "src/app.py")
        write(repo, "lib.py")
        assert sorted(func_paths(repo)) == ["lib.py", "src/app.py"]
        assert ex.count_repo_contents(str(repo), "Python", ["function", "class"]) == (2, 2)

    def test_clone_under_a_vendor_parent_is_still_extracted(self, tmp_path):
        repo = tmp_path / "vendor" / "node_modules" / "repo"
        write(repo, "app.py")
        assert func_paths(repo) == ["app.py"]
        assert ex.count_repo_contents(str(repo), "Python", ["function", "class"]) == (1, 1)

    @pytest.mark.parametrize("parent", ["tests", "test"])
    def test_test_dirs_inside_the_repo_are_still_skipped(self, tmp_path, parent):
        repo = tmp_path / parent / "repo"
        write(repo, "app.py")
        write(repo, "tests/test_a.py")
        write(repo, "pkg/test/b.py")
        write(repo, "pkg/specs/c.py")
        write(repo, "pkg/__tests__/d.py")
        assert func_paths(repo) == ["app.py"]
        assert ex.count_repo_contents(str(repo), "Python", ["function", "class"]) == (1, 1)

    def test_test_file_names_still_skipped(self, tmp_path):
        write(tmp_path, "app.py")
        write(tmp_path, "test_app.py")
        write(tmp_path, "app_test.py")
        assert func_paths(tmp_path) == ["app.py"]

    def test_file_paths_in_records_stay_relative_to_the_repo(self, tmp_path):
        repo = tmp_path / "tests" / "repo"
        write(repo, "src/app.py")
        assert func_paths(repo) == ["src/app.py"]


# ─── 4. Pruning, not after-the-fact filtering ─────────────────────────────────

class _ScandirSpy:
    """Records every directory that os.scandir is asked to list."""

    def __init__(self):
        self.listed: list[str] = []
        self._real = os.scandir

    def __call__(self, path="."):
        self.listed.append(os.fspath(path))
        return self._real(path)

    def touched(self, name: str) -> list[str]:
        return [p for p in self.listed if name in Path(p).parts]


class TestPruning:
    def make_repo(self, root: Path):
        write(root, "src/app.py")
        write(root, "node_modules/dep/index.py")
        write(root, "node_modules/dep/nested/vendor/deeper/x.py")
        write(root, "vendor/lib.py")
        write(root, "src/tests/test_x.py")
        return root

    @pytest.mark.parametrize("walk", ["extract", "count"])
    def test_skipped_trees_are_never_listed(self, tmp_path, monkeypatch, walk):
        root = self.make_repo(tmp_path)
        spy = _ScandirSpy()
        monkeypatch.setattr(os, "scandir", spy)
        if walk == "extract":
            extract(root)
        else:
            ex.count_repo_contents(str(root), "Python", ["function", "class"])
        assert spy.touched("node_modules") == []
        assert spy.touched("vendor") == []
        assert spy.touched("tests") == []          # test dirs are pruned too
        assert any(p.endswith("src") for p in spy.listed)       # sanity: the spy sees real listings
        assert str(root) in spy.listed

    def test_a_skipped_dir_is_never_entered_even_when_it_would_fail(self, tmp_path):
        root = self.make_repo(tmp_path)
        real_scandir = os.scandir

        def exploding(path="."):
            if "node_modules" in Path(os.fspath(path)).parts:
                raise AssertionError(f"walked into a skipped directory: {path}")
            return real_scandir(path)

        # os.walk swallows OSError but not AssertionError, so a visit would surface.
        with patch.object(os, "scandir", exploding):
            assert func_paths(root) == ["src/app.py"]
            assert ex.count_repo_contents(str(root), "Python", ["function", "class"]) == (1, 1)

    def test_a_walk_with_a_filename_filter_is_not_a_filter_after_the_fact(self, tmp_path, monkeypatch):
        # Guard against re-adding rglob: it would list every directory, skipped or not.
        root = self.make_repo(tmp_path)
        monkeypatch.setattr(Path, "rglob", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rglob used")))
        assert func_paths(root) == ["src/app.py"]
        assert ex.count_repo_contents(str(root), "Python", ["function", "class"]) == (1, 1)


# ─── 5. Same records as before; a pinned order of its own ─────────────────────

def _legacy_extract(root: Path, language="Python"):
    """The pre-change algorithm: rglob plus after-the-fact filtering of the ABSOLUTE parts."""
    extensions = ex.LANGUAGE_EXTENSIONS[language]
    funcs, classes = [], []
    for fp in root.rglob("*"):
        if not fp.is_file() or fp.suffix not in extensions:
            continue
        if any(p in fp.parts for p in {"test", "tests", "__tests__", "spec", "specs"}):
            continue
        if fp.name.startswith("test_") or fp.name.endswith("_test.py"):
            continue
        if fp.name.startswith("Test") or fp.name.endswith("Test.java"):
            continue
        f, c = ex.extract_from_file(str(fp), language, "o/r", relative_path=str(fp.relative_to(root)))
        funcs.extend(f)
        classes.extend(c)
    return funcs, classes


def _key(r):
    return (r.file_path, r.func_name if hasattr(r, "func_name") else r.class_name, r.start_line)


def _multiset(records):
    """Order-free view of whole records: sorted by a unique key (the key is checked for uniqueness)."""
    ordered = sorted(records, key=_key)
    keys = [_key(r) for r in ordered]
    assert len(keys) == len(set(keys)), "fixture must give every record a unique key"
    return ordered


class TestSameRecordsAsLegacy:
    def make_repo(self, root: Path):
        for rel in ["zz.py", "a.py", "b.py", "pkg/x.py", "pkg/sub/y.py", "pkg/sub/deeper/w.py",
                    "pkg2/z.py", "pkg2/Z2.py", "src/env/e.py", "external/ext.py", "Alpha/up.py"]:
            write(root, rel)
        return root

    def test_new_walker_extracts_the_same_records_as_the_rglob_based_extraction(self, tmp_path):
        """
        Record equality only, order ignored. The legacy rglob order followed the filesystem and
        pathlib's own traversal, and that traversal differs between Python versions (3.12 visits
        directories in a different order than 3.11), so asserting the legacy ORDER would make this
        test depend on the interpreter. The order of the new walker is pinned separately, in
        TestNewOrderIsPinned, with an explicit expected list.
        """
        root = self.make_repo(tmp_path)
        old_funcs, old_classes = _legacy_extract(root)
        new_funcs, new_classes = extract(root)
        assert len(new_funcs) == len(old_funcs) == 11
        assert len(new_classes) == len(old_classes) == 11
        # Whole records (every field), compared as multisets.
        assert _multiset(new_funcs) == _multiset(old_funcs)
        assert _multiset(new_classes) == _multiset(old_classes)
        assert Counter(map(_key, new_funcs)) == Counter(map(_key, old_funcs))
        assert Counter(map(_key, new_classes)) == Counter(map(_key, old_classes))


class TestNewOrderIsPinned:
    """
    The order of the new walker, as an explicit list: the files of a directory sorted by name,
    then its subdirectories sorted by name, depth first. Names sort by code point, so upper case
    comes before lower case. Not derived from the filesystem, so it holds on every Python version.
    """

    make_repo = TestSameRecordsAsLegacy.make_repo

    EXPECTED = [
        "a.py", "b.py", "zz.py",
        "Alpha/up.py",
        "external/ext.py",
        "pkg/x.py", "pkg/sub/y.py", "pkg/sub/deeper/w.py",
        "pkg2/Z2.py", "pkg2/z.py",
        "src/env/e.py",
    ]

    def test_order_is_files_of_a_directory_then_its_subdirectories_all_sorted(self, tmp_path):
        root = self.make_repo(tmp_path)
        assert func_paths(root) == self.EXPECTED
        assert class_paths(root) == self.EXPECTED

    def test_order_does_not_depend_on_creation_order(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        names = ["m.py", "d/x.py", "a.py", "d/b.py", "c/z.py"]
        for rel in names:
            write(a, rel)
        for rel in reversed(names):
            write(b, rel)
        expected = ["a.py", "m.py", "c/z.py", "d/b.py", "d/x.py"]
        assert func_paths(a) == expected
        assert func_paths(b) == expected

    def test_order_does_not_depend_on_the_directory_listing_order(self, tmp_path, monkeypatch):
        """Even a listing in reverse name order (the worst case for an unsorted walk) gives the pinned order."""
        root = self.make_repo(tmp_path)
        real_scandir = os.scandir

        def reversed_scandir(path="."):
            with real_scandir(path) as it:
                entries = sorted(it, key=lambda e: e.name, reverse=True)
            return _ListingIterator(entries)

        monkeypatch.setattr(os, "scandir", reversed_scandir)
        assert func_paths(root) == self.EXPECTED

    def test_caps_keep_the_first_records_in_that_order(self, tmp_path):
        root = self.make_repo(tmp_path)
        funcs, _ = extract(root, func_cap=2, class_cap=2)
        assert [f.file_path for f in funcs] == ["a.py", "b.py"]


class _ListingIterator:
    """What os.scandir returns, over a fixed list of entries (iterator and context manager)."""

    def __init__(self, entries):
        self._it = iter(entries)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


# ─── 6. files_skipped and the manifest ────────────────────────────────────────

class TestFilesSkippedCount:
    def test_counts_source_files_under_skipped_directories(self, tmp_path):
        write(tmp_path, "src/app.py")
        write(tmp_path, "node_modules/a.py")
        write(tmp_path, "node_modules/deep/b.py")
        write(tmp_path, "vendor/c.py")
        write(tmp_path, "build/readme.txt", "not source")       # wrong extension: not a candidate
        write(tmp_path, "src/dist/d.py")
        assert ex.count_skipped_files(str(tmp_path), "Python") == 4

    def test_zero_when_nothing_is_skipped(self, tmp_path):
        write(tmp_path, "src/app.py")
        write(tmp_path, "tests/test_a.py")                      # skipped as a test, not under the skip list
        assert ex.count_skipped_files(str(tmp_path), "Python") == 0

    def test_uses_the_language_extensions(self, tmp_path):
        write(tmp_path, "node_modules/a.js", "function f() {}")
        write(tmp_path, "node_modules/b.py")
        assert ex.count_skipped_files(str(tmp_path), "JavaScript") == 1

    def test_clone_under_a_skipped_parent_counts_nothing_for_the_parent(self, tmp_path):
        repo = tmp_path / "vendor" / "repo"
        write(repo, "app.py")
        assert ex.count_skipped_files(str(repo), "Python") == 0

    def test_extraction_itself_never_counts_by_listing_skipped_trees(self, tmp_path, monkeypatch):
        write(tmp_path, "src/app.py")
        write(tmp_path, "node_modules/a.py")
        spy = _ScandirSpy()
        monkeypatch.setattr(os, "scandir", spy)
        extract(tmp_path)
        assert spy.touched("node_modules") == []


class TestManifestExtractionBlock:
    def test_write_manifest_stores_the_block(self, tmp_path):
        block = {"skip_dirs": sorted(SKIPPED), "files_skipped": 7}
        write_manifest({}, [], 0, 0, tmp_path, extraction=block)
        assert json.loads((tmp_path / "manifest.json").read_text())["extraction"] == block

    def test_write_manifest_without_the_block_is_unchanged(self, tmp_path):
        write_manifest({}, [], 0, 0, tmp_path)
        assert "extraction" not in json.loads((tmp_path / "manifest.json").read_text())

    def _run(self, tmp_path, trees: dict):
        token = "ghp_SECRET_TOKEN_1234567890"
        config = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
                             output_dir=str(tmp_path / "out"), github_token=token)

        def meta(name):
            return RepoMetadata(
                full_name=name, url=f"https://github.com/{name}", stars=100, contributors=5,
                last_commit_date="2024-01-01T00:00:00", default_branch="main", license_spdx="MIT",
                language="Python", collected_at="2024-06-01T00:00:00+00:00", topics=[],
            )

        def fake_discover(**kwargs):
            for name in trees:
                yield meta(name)

        def fake_clone(repo_full_name, target_dir):
            for rel in trees[repo_full_name]:
                write(Path(target_dir), rel)
            return True

        with patch("sieve.pipeline.discover_repos", side_effect=fake_discover), \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", side_effect=fake_clone), \
             patch("sieve.pipeline.time.sleep"):
            run_pipeline(config)
        text = (tmp_path / "out" / "manifest.json").read_text()
        assert token not in text
        return json.loads(text)

    def test_pipeline_sums_files_skipped_over_repos(self, tmp_path):
        manifest = self._run(tmp_path, {
            "o/one": ["app.py", "node_modules/a.py", "node_modules/b.py", "vendor/c.py"],
            "o/two": ["lib.py", "dist/bundle.py", "external/ext.py"],
            "o/three": ["only.py"],
        })
        block = manifest["extraction"]
        assert block["skip_dirs"] == sorted(SKIPPED)
        assert block["files_skipped"] == 4
        assert set(block) == {"skip_dirs", "files_skipped"}
        assert manifest["summary"]["total_functions"] == 4          # app, lib, ext, only

    def test_a_clean_run_records_zero(self, tmp_path):
        manifest = self._run(tmp_path, {"o/one": ["app.py"]})
        assert manifest["extraction"]["files_skipped"] == 0
        assert manifest["extraction"]["skip_dirs"] == sorted(SKIPPED)
