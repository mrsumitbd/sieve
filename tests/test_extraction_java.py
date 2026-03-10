"""
tests/test_extraction_java.py

Unit tests for the Java tree-sitter extractor.
"""

import textwrap

import pytest

from sieve.core.extraction import _extract_java


def extract(code: str):
    return _extract_java(textwrap.dedent(code).strip(), "Test.java", "owner/repo")


class TestMethodExtraction:
    def test_basic_method_extracted(self):
        funcs, _ = extract("""
            public class Foo {
                public int add(int a, int b) {
                    return a + b;
                }
            }
        """)
        methods = [f for f in funcs if f.func_name == "add"]
        assert len(methods) == 1

    def test_parameters_captured(self):
        funcs, _ = extract("""
            public class Foo {
                public int add(int a, int b) { return a + b; }
            }
        """)
        method = next(f for f in funcs if f.func_name == "add")
        assert "a" in method.parameters
        assert "b" in method.parameters

    def test_constructor_extracted(self):
        funcs, _ = extract("""
            public class Foo {
                public Foo() {}
            }
        """)
        ctors = [f for f in funcs if f.func_name == "Foo"]
        assert len(ctors) == 1

    def test_javadoc_captured(self):
        funcs, _ = extract("""
            public class Foo {
                /** Find the index. */
                public int search(int target) {
                    return -1;
                }
            }
        """)
        method = next(f for f in funcs if f.func_name == "search")
        assert method.docstring == "Find the index."

    def test_is_method_true(self):
        funcs, _ = extract("""
            public class Foo {
                public void bar() {}
            }
        """)
        method = next(f for f in funcs if f.func_name == "bar")
        assert method.is_method is True
        assert method.parent_class == "Foo"

    def test_signature_has_no_body(self):
        funcs, _ = extract("""
            public class Foo {
                public int add(int a, int b) {
                    return a + b;
                }
            }
        """)
        method = next(f for f in funcs if f.func_name == "add")
        assert "return" not in method.signature
        assert "{ }" in method.signature


class TestAnnotationExtraction:
    def test_override_annotation_captured(self):
        funcs, _ = extract("""
            public class Foo {
                @Override
                public String toString() {
                    return "Foo";
                }
            }
        """)
        method = next(f for f in funcs if f.func_name == "toString")
        assert "@Override" in method.decorators

    def test_multiple_annotations_captured(self):
        funcs, _ = extract("""
            public class Foo {
                @Override
                @Deprecated
                public String toString() {
                    return "Foo";
                }
            }
        """)
        method = next(f for f in funcs if f.func_name == "toString")
        assert "@Override" in method.decorators
        assert "@Deprecated" in method.decorators

    def test_annotation_with_argument(self):
        funcs, _ = extract("""
            public class Foo {
                @SuppressWarnings("unchecked")
                public void process() {}
            }
        """)
        method = next(f for f in funcs if f.func_name == "process")
        assert any("SuppressWarnings" in d for d in method.decorators)

    def test_unannotated_method_empty_decorators(self):
        funcs, _ = extract("""
            public class Foo {
                public void plain() {}
            }
        """)
        method = next(f for f in funcs if f.func_name == "plain")
        assert method.decorators == []


class TestClassExtraction:
    def test_class_name_captured(self):
        _, classes = extract("""
            public class BinarySearch {
                public int search(int target) { return -1; }
            }
        """)
        assert classes[0].class_name == "BinarySearch"

    def test_superclass_captured(self):
        _, classes = extract("""
            public class Dog extends Animal {
                public void speak() {}
            }
        """)
        assert "Animal" in classes[0].parent_classes

    def test_interfaces_captured(self):
        _, classes = extract("""
            public class MyClass extends Base implements Serializable, Runnable {
                public void run() {}
            }
        """)
        assert "Base" in classes[0].parent_classes
        assert "Serializable" in classes[0].parent_classes
        assert "Runnable" in classes[0].parent_classes

    def test_has_constructor_true(self):
        _, classes = extract("""
            public class Foo {
                public Foo() {}
            }
        """)
        assert classes[0].has_constructor is True

    def test_javadoc_on_class_captured(self):
        _, classes = extract("""
            /** A useful class. */
            public class Useful {
                public void go() {}
            }
        """)
        assert classes[0].docstring == "A useful class."

    def test_skeleton_contains_method_sigs(self):
        _, classes = extract("""
            public class Calc {
                public int add(int a, int b) { return a + b; }
                public int sub(int a, int b) { return a - b; }
            }
        """)
        skeleton = classes[0].skeleton
        assert "add" in skeleton
        assert "sub" in skeleton
        assert "return" not in skeleton


class TestImportDetection:
    def test_used_import_included(self):
        funcs, _ = extract("""
            import java.util.List;
            import java.util.Map;

            public class Foo {
                public int size(List<Integer> items) {
                    return items.size();
                }
            }
        """)
        method = next(f for f in funcs if f.func_name == "size")
        assert any("List" in imp for imp in method.used_imports)
        assert not any("Map" in imp for imp in method.used_imports)