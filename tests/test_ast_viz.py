"""
tests/test_ast_viz.py

Unit tests for the AST visualization helpers in sieve.ui.ast_viz.
No network or GPU required.
"""

import pytest
from sieve.ui.ast_viz import build_ast_json, render_ast_component


class TestBuildAstJson:
    def test_returns_dict_for_python(self):
        code = "def add(a, b): return a + b"
        result = build_ast_json(code, "Python")
        assert result is not None
        assert isinstance(result, dict)
        assert "type" in result
        assert "children" in result

    def test_root_node_type(self):
        code = "x = 1"
        result = build_ast_json(code, "Python")
        assert result["type"] == "module"

    def test_max_depth_limits_tree(self):
        code = "def foo():\n    if True:\n        return 1\n"
        shallow = build_ast_json(code, "Python", max_depth=1)
        deep    = build_ast_json(code, "Python", max_depth=10)
        # Shallow tree should have fewer or equal total nodes
        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.get("children", []))
        assert count_nodes(shallow) <= count_nodes(deep)

    def test_returns_none_for_unknown_language(self):
        result = build_ast_json("some code", "COBOL")
        assert result is None

    def test_leaf_nodes_have_text(self):
        code = "x = 42"
        result = build_ast_json(code, "Python", max_depth=10)
        def find_leaves(node):
            if not node.get("children"):
                return [node]
            leaves = []
            for child in node["children"]:
                leaves.extend(find_leaves(child))
            return leaves
        leaves = find_leaves(result)
        # Leaf nodes encode text in their "name" field (e.g. "identifier: 'x'")
        assert any(leaf.get("name") is not None for leaf in leaves)

    def test_works_for_javascript(self):
        code = "function add(a, b) { return a + b; }"
        result = build_ast_json(code, "JavaScript")
        assert result is not None
        assert result["type"] == "program"

    def test_works_for_java(self):
        code = "class Foo { int x = 1; }"
        result = build_ast_json(code, "Java")
        assert result is not None

    def test_works_for_cpp(self):
        code = "int add(int a, int b) { return a + b; }"
        result = build_ast_json(code, "C++")
        assert result is not None


class TestRenderAstComponent:
    def test_returns_html_string(self):
        code = "def foo(): pass"
        ast_json = build_ast_json(code, "Python")
        html = render_ast_component(ast_json, height=400)
        assert isinstance(html, str)
        assert "<html>" in html or "<!DOCTYPE" in html or "<svg" in html or "d3" in html.lower()

    def test_html_contains_data(self):
        code = "x = 1"
        ast_json = build_ast_json(code, "Python")
        html = render_ast_component(ast_json)
        # The AST JSON should be embedded in the HTML
        assert "module" in html

    def test_height_parameter_respected(self):
        code = "x = 1"
        ast_json = build_ast_json(code, "Python")
        html_400 = render_ast_component(ast_json, height=400)
        html_800 = render_ast_component(ast_json, height=800)
        # height - 44 is used for tree container (controls bar takes 44px)
        assert "356" in html_400  # 400 - 44
        assert "756" in html_800  # 800 - 44