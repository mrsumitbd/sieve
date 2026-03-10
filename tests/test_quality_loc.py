"""
tests/test_quality_loc.py

Unit tests for the AST-based and tree-sitter LOC counters in quality.py.
No network access — uses real temporary files.
"""

import textwrap
from pathlib import Path

import pytest

from sieve.core.quality import _count_python_ast, _count_treesitter


class TestCountPythonAst:
    def test_basic_code_lines_counted(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\ny = 2\nz = x + y\n")
        loc, comments = _count_python_ast(f)
        assert loc == 3
        assert comments == 0

    def test_comment_lines_counted(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("# this is a comment\nx = 1\n# another comment\n")
        loc, comments = _count_python_ast(f)
        assert loc == 1
        assert comments == 2

    def test_inline_comment_counts_as_code(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1  # inline comment\n")
        loc, comments = _count_python_ast(f)
        assert loc == 1
        assert comments == 0  # inline comments don't count as pure comment lines

    def test_blank_lines_not_counted(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n\n\ny = 2\n")
        loc, comments = _count_python_ast(f)
        assert loc == 2

    def test_function_def_counted(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text(textwrap.dedent("""
            def add(a, b):
                return a + b
        """).strip() + "\n")
        loc, comments = _count_python_ast(f)
        assert loc >= 2

    def test_empty_file_returns_zeros(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        loc, comments = _count_python_ast(f)
        assert loc == 0
        assert comments == 0

    def test_invalid_python_returns_zeros(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_bytes(b"\xff\xfe invalid utf-8 \x00")
        loc, comments = _count_python_ast(f)
        assert loc == 0
        assert comments == 0

    def test_returns_tuple_of_two_ints(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        result = _count_python_ast(f)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)


class TestCountTreesitter:
    def test_java_code_lines_counted(self, tmp_path):
        f = tmp_path / "Foo.java"
        f.write_text(textwrap.dedent("""
            public class Foo {
                public int add(int a, int b) {
                    return a + b;
                }
            }
        """).strip() + "\n")
        loc, comments = _count_treesitter(f, "Java")
        assert loc > 0

    def test_java_comment_lines_counted(self, tmp_path):
        f = tmp_path / "Foo.java"
        f.write_text(textwrap.dedent("""
            // This is a comment
            public class Foo {
                /** Javadoc */
                public void go() {}
            }
        """).strip() + "\n")
        loc, comments = _count_treesitter(f, "Java")
        assert comments > 0

    def test_javascript_code_lines_counted(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text(textwrap.dedent("""
            function add(a, b) {
                return a + b;
            }
        """).strip() + "\n")
        loc, comments = _count_treesitter(f, "JavaScript")
        assert loc > 0

    def test_javascript_comment_lines_counted(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text(textwrap.dedent("""
            // A comment
            function add(a, b) {
                return a + b; // inline
            }
        """).strip() + "\n")
        loc, comments = _count_treesitter(f, "JavaScript")
        assert comments > 0

    def test_empty_file_returns_low_count(self, tmp_path):
        """tree-sitter may count 0 or 1 lines for an empty file — both are acceptable."""
        f = tmp_path / "Empty.java"
        f.write_text("")
        loc, comments = _count_treesitter(f, "Java")
        assert loc <= 1
        assert comments == 0

    def test_returns_tuple_of_two_ints(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text("const x = 1;\n")
        result = _count_treesitter(f, "JavaScript")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)