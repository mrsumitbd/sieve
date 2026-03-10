"""
tests/test_extraction_python.py

Unit tests for the Python tree-sitter extractor.
"""

import textwrap

import pytest

from sieve.core.extraction import _extract_python


def extract(code: str):
    return _extract_python(textwrap.dedent(code).strip(), "test.py", "owner/repo")


class TestFunctionExtraction:
    def test_basic_function_extracted(self):
        funcs, _ = extract("""
            def add(a, b):
                return a + b
        """)
        assert len(funcs) == 1
        assert funcs[0].func_name == "add"

    def test_parameters_captured(self):
        funcs, _ = extract("""
            def greet(name: str, greeting: str = "hello") -> str:
                return f"{greeting}, {name}"
        """)
        assert "name: str" in funcs[0].parameters[0] or "name" in funcs[0].parameters[0]
        assert len(funcs[0].parameters) == 2

    def test_return_annotation_captured(self):
        funcs, _ = extract("""
            def double(x: int) -> int:
                return x * 2
        """)
        assert funcs[0].return_annotation is not None
        assert "int" in funcs[0].return_annotation

    def test_docstring_captured(self):
        funcs, _ = extract('''
            def add(a, b):
                """Add two numbers."""
                return a + b
        ''')
        assert funcs[0].docstring == "Add two numbers."

    def test_no_docstring_is_none(self):
        funcs, _ = extract("""
            def add(a, b):
                return a + b
        """)
        assert funcs[0].docstring is None

    def test_async_function_extracted(self):
        funcs, _ = extract("""
            async def fetch(url: str) -> str:
                return url
        """)
        assert funcs[0].func_name == "fetch"

    def test_is_method_false_for_top_level(self):
        funcs, _ = extract("""
            def standalone():
                pass
        """)
        assert funcs[0].is_method is False
        assert funcs[0].parent_class is None

    def test_is_method_true_inside_class(self):
        funcs, _ = extract("""
            class MyClass:
                def method(self):
                    pass
        """)
        methods = [f for f in funcs if f.func_name == "method"]
        assert len(methods) == 1
        assert methods[0].is_method is True
        assert methods[0].parent_class == "MyClass"

    def test_source_code_indentation_normalized(self):
        funcs, _ = extract("""
            class Outer:
                class Inner:
                    def deep(self):
                        return 42
        """)
        deep = next(f for f in funcs if f.func_name == "deep")
        lines = deep.source_code.splitlines()
        # body should be at 4-space indent, not 12
        assert lines[0].startswith("def deep")
        body = [l for l in lines[1:] if l.strip()]
        assert all(l.startswith("    ") for l in body)

    def test_signature_has_no_body(self):
        funcs, _ = extract('''
            def add(a: int, b: int) -> int:
                """Add two numbers."""
                return a + b
        ''')
        sig = funcs[0].signature
        assert "pass" in sig or "..." in sig
        assert "return" not in sig

    def test_line_numbers_correct(self):
        code = "def first(): pass\ndef second(): pass\n"
        funcs, _ = _extract_python(code, "test.py", "r")
        first = next(f for f in funcs if f.func_name == "first")
        second = next(f for f in funcs if f.func_name == "second")
        assert first.start_line == 1
        assert second.start_line == 2


class TestDecoratorExtraction:
    def test_single_decorator(self):
        funcs, _ = extract("""
            import functools

            @staticmethod
            def cached():
                pass
        """)
        fn = next(f for f in funcs if f.func_name == "cached")
        assert "@staticmethod" in fn.decorators

    def test_multiple_decorators(self):
        funcs, _ = extract("""
            import functools

            @staticmethod
            @functools.lru_cache(maxsize=128)
            def cached(x):
                return x
        """)
        fn = next(f for f in funcs if f.func_name == "cached")
        assert "@staticmethod" in fn.decorators
        assert any("lru_cache" in d for d in fn.decorators)
        assert len(fn.decorators) == 2

    def test_undecorated_has_empty_list(self):
        funcs, _ = extract("""
            def plain():
                pass
        """)
        assert funcs[0].decorators == []

    def test_class_decorator_captured(self):
        _, classes = extract("""
            from dataclasses import dataclass

            @dataclass
            class Point:
                x: float
                y: float
        """)
        assert "@dataclass" in classes[0].decorators

    def test_decorated_method_in_method_names(self):
        _, classes = extract("""
            class Service:
                @classmethod
                def create(cls):
                    return cls()

                @property
                def name(self):
                    return self._name
        """)
        assert "create" in classes[0].method_names
        assert "name" in classes[0].method_names


class TestClassExtraction:
    def test_basic_class_extracted(self):
        _, classes = extract("""
            class Animal:
                def speak(self):
                    pass
        """)
        assert len(classes) == 1
        assert classes[0].class_name == "Animal"

    def test_parent_classes_captured(self):
        _, classes = extract("""
            class Dog(Animal, Mammal):
                pass
        """)
        assert "Animal" in classes[0].parent_classes
        assert "Mammal" in classes[0].parent_classes

    def test_has_constructor_true(self):
        _, classes = extract("""
            class Foo:
                def __init__(self):
                    pass
        """)
        assert classes[0].has_constructor is True

    def test_has_constructor_false(self):
        _, classes = extract("""
            class Foo:
                def method(self):
                    pass
        """)
        assert classes[0].has_constructor is False

    def test_method_count(self):
        _, classes = extract("""
            class MyClass:
                def a(self): pass
                def b(self): pass
                def c(self): pass
        """)
        assert classes[0].method_count == 3

    def test_skeleton_has_no_body_implementation(self):
        _, classes = extract('''
            class Calc:
                """A calculator."""
                def add(self, a, b):
                    """Add two numbers."""
                    return a + b
        ''')
        skeleton = classes[0].skeleton
        assert "return" not in skeleton
        assert "add" in skeleton

    def test_empty_class_skeleton_has_pass(self):
        _, classes = extract("""
            class Empty:
                pass
        """)
        assert "pass" in classes[0].skeleton

    def test_class_docstring_captured(self):
        _, classes = extract('''
            class Foo:
                """This is a docstring."""
                pass
        ''')
        assert classes[0].docstring == "This is a docstring."


class TestImportDetection:
    def test_used_import_included(self):
        funcs, _ = extract("""
            import os
            import re

            def clean(text):
                return re.sub(r'\\s+', ' ', text)
        """)
        clean = next(f for f in funcs if f.func_name == "clean")
        assert any("re" in imp for imp in clean.used_imports)
        assert not any("os" in imp for imp in clean.used_imports)

    def test_from_import_detected(self):
        funcs, _ = extract("""
            from pathlib import Path

            def read(name: str) -> str:
                return Path(name).read_text()
        """)
        fn = funcs[0]
        assert any("pathlib" in imp or "Path" in imp for imp in fn.used_imports)

    def test_wildcard_always_included(self):
        funcs, _ = extract("""
            from os.path import *

            def join_paths(a, b):
                return join(a, b)
        """)
        assert any("*" in imp for imp in funcs[0].used_imports)

    def test_unused_import_excluded(self):
        funcs, _ = extract("""
            import json
            import os

            def add(a, b):
                return a + b
        """)
        fn = next(f for f in funcs if f.func_name == "add")
        assert fn.used_imports == []