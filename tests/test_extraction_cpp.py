"""
tests/test_extraction_cpp.py

Unit tests for the C++ tree-sitter extractor.
"""

import textwrap
import pytest

# extraction.py lives in src/ — tests run with pytest from repo root
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sieve.core.extraction import _extract_cpp


def extract(code: str):
    return _extract_cpp(textwrap.dedent(code).strip(), "test.cpp", "owner/repo")


# ─── Function extraction ──────────────────────────────────────────────────────

class TestFunctionExtraction:
    def test_basic_function_extracted(self):
        funcs, _ = extract("""
            int add(int a, int b) {
                return a + b;
            }
        """)
        assert any(f.func_name == "add" for f in funcs)

    def test_parameters_captured(self):
        funcs, _ = extract("""
            void greet(const std::string& name, int times) {
                for (int i = 0; i < times; ++i) {}
            }
        """)
        fn = next(f for f in funcs if f.func_name == "greet")
        assert len(fn.parameters) == 2

    def test_return_type_captured(self):
        funcs, _ = extract("""
            int factorial(int n) {
                return n <= 1 ? 1 : n * factorial(n - 1);
            }
        """)
        fn = next(f for f in funcs if f.func_name == "factorial")
        assert fn.return_annotation is not None
        assert "int" in fn.return_annotation

    def test_docstring_captured(self):
        funcs, _ = extract("""
            /** Compute the sum of two integers. */
            int add(int a, int b) {
                return a + b;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "add")
        assert fn.docstring is not None
        assert "sum" in fn.docstring.lower()

    def test_line_comment_docstring(self):
        funcs, _ = extract("""
            // Returns the maximum of two values.
            int max_val(int a, int b) {
                return a > b ? a : b;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "max_val")
        assert fn.docstring is not None
        assert "maximum" in fn.docstring.lower()

    def test_no_docstring_is_none(self):
        funcs, _ = extract("""
            int add(int a, int b) { return a + b; }
        """)
        fn = next(f for f in funcs if f.func_name == "add")
        assert fn.docstring is None

    def test_signature_has_no_body(self):
        funcs, _ = extract("""
            int add(int a, int b) {
                return a + b;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "add")
        assert "return" not in fn.signature
        assert "{ }" in fn.signature

    def test_is_method_false_for_top_level(self):
        funcs, _ = extract("""
            void standalone() {}
        """)
        fn = next(f for f in funcs if f.func_name == "standalone")
        assert fn.is_method is False
        assert fn.parent_class is None

    def test_is_method_true_inside_class(self):
        funcs, _ = extract("""
            class Foo {
            public:
                void bar() {}
            };
        """)
        methods = [f for f in funcs if f.func_name == "bar"]
        assert len(methods) == 1
        assert methods[0].is_method is True
        assert methods[0].parent_class == "Foo"

    def test_void_return_type(self):
        funcs, _ = extract("""
            void print_hello() {
                printf("hello");
            }
        """)
        fn = next(f for f in funcs if f.func_name == "print_hello")
        assert fn.return_annotation is not None
        assert "void" in fn.return_annotation

    def test_template_function_extracted(self):
        funcs, _ = extract("""
            template<typename T>
            T max_val(T a, T b) {
                return a > b ? a : b;
            }
        """)
        assert any(f.func_name == "max_val" for f in funcs)

    def test_namespace_function_extracted(self):
        funcs, _ = extract("""
            namespace utils {
                int square(int x) { return x * x; }
            }
        """)
        assert any(f.func_name == "square" for f in funcs)

    def test_language_field(self):
        funcs, _ = extract("""
            int add(int a, int b) { return a + b; }
        """)
        assert funcs[0].language == "C++"

    def test_decorators_empty(self):
        funcs, _ = extract("""
            int add(int a, int b) { return a + b; }
        """)
        assert funcs[0].decorators == []


# ─── Class extraction ────────────────────────────────────────────────────────

class TestClassExtraction:
    def test_basic_class_extracted(self):
        _, classes = extract("""
            class Animal {
            public:
                void speak() {}
            };
        """)
        assert len(classes) == 1
        assert classes[0].class_name == "Animal"

    def test_base_class_captured(self):
        _, classes = extract("""
            class Dog : public Animal {
            public:
                void bark() {}
            };
        """)
        assert "Animal" in classes[0].parent_classes

    def test_multiple_base_classes(self):
        _, classes = extract("""
            class C : public A, public B {
            public:
                void go() {}
            };
        """)
        assert "A" in classes[0].parent_classes
        assert "B" in classes[0].parent_classes

    def test_has_constructor_true(self):
        _, classes = extract("""
            class Foo {
            public:
                Foo() {}
                void bar() {}
            };
        """)
        assert classes[0].has_constructor is True

    def test_has_constructor_false(self):
        _, classes = extract("""
            class Foo {
            public:
                void bar() {}
            };
        """)
        assert classes[0].has_constructor is False

    def test_method_names_captured(self):
        _, classes = extract("""
            class Calc {
            public:
                int add(int a, int b) { return a + b; }
                int sub(int a, int b) { return a - b; }
            };
        """)
        assert "add" in classes[0].method_names
        assert "sub" in classes[0].method_names

    def test_method_count(self):
        _, classes = extract("""
            class Calc {
            public:
                int add(int a, int b) { return a + b; }
                int sub(int a, int b) { return a - b; }
                int mul(int a, int b) { return a * b; }
            };
        """)
        assert classes[0].method_count == 3

    def test_class_docstring_captured(self):
        _, classes = extract("""
            /** A simple calculator class. */
            class Calc {
            public:
                int add(int a, int b) { return a + b; }
            };
        """)
        assert classes[0].docstring is not None
        assert "calculator" in classes[0].docstring.lower()

    def test_skeleton_has_method_sigs(self):
        _, classes = extract("""
            class Calc {
            public:
                int add(int a, int b) { return a + b; }
                int sub(int a, int b) { return a - b; }
            };
        """)
        skeleton = classes[0].skeleton
        assert "add" in skeleton
        assert "sub" in skeleton
        assert "return" not in skeleton

    def test_skeleton_no_implementation(self):
        _, classes = extract("""
            class Foo {
            public:
                void process(int x) {
                    for (int i = 0; i < x; ++i) {
                        printf("%d", i);
                    }
                }
            };
        """)
        skeleton = classes[0].skeleton
        assert "printf" not in skeleton
        assert "process" in skeleton

    def test_language_field(self):
        _, classes = extract("""
            class Foo { public: void bar() {} };
        """)
        assert classes[0].language == "C++"

    def test_decorators_empty(self):
        _, classes = extract("""
            class Foo { public: void bar() {} };
        """)
        assert classes[0].decorators == []

    def test_template_class_extracted(self):
        _, classes = extract("""
            template<typename T>
            class Stack {
            public:
                void push(T val) {}
                T pop() {}
            };
        """)
        assert any(c.class_name == "Stack" for c in classes)

    def test_namespace_class_extracted(self):
        _, classes = extract("""
            namespace utils {
                class Helper {
                public:
                    void help() {}
                };
            }
        """)
        assert any(c.class_name == "Helper" for c in classes)


# ─── Include detection ───────────────────────────────────────────────────────

class TestIncludeDetection:
    def test_used_include_detected(self):
        funcs, _ = extract("""
            #include <vector>
            #include <string>

            void process(std::vector<int>& items) {
                items.push_back(1);
            }
        """)
        fn = next(f for f in funcs if f.func_name == "process")
        assert any("vector" in imp for imp in fn.used_imports)

    def test_unused_include_excluded(self):
        funcs, _ = extract("""
            #include <vector>
            #include <string>

            int add(int a, int b) { return a + b; }
        """)
        fn = next(f for f in funcs if f.func_name == "add")
        assert fn.used_imports == []

    def test_quoted_include_detected(self):
        funcs, _ = extract("""
            #include "mylib.h"

            void init() {
                mylib::setup();
            }
        """)
        fn = next(f for f in funcs if f.func_name == "init")
        assert any("mylib" in imp for imp in fn.used_imports)


class TestPointerReturnTypes:
    def test_pointer_return_name_extracted(self):
        funcs, _ = extract("""
            Node* lca(Node* root, int n1, int n2) {
                return nullptr;
            }
        """)
        fn = next((f for f in funcs if f.func_name != "unknown"), None)
        assert fn is not None
        assert fn.func_name == "lca"

    def test_const_pointer_return_name_extracted(self):
        funcs, _ = extract("""
            const wchar_t* printError(int hr) {
                return nullptr;
            }
        """)
        fn = next((f for f in funcs if f.func_name != "unknown"), None)
        assert fn is not None
        assert fn.func_name == "printError"

    def test_scoped_pointer_return_name_extracted(self):
        funcs, _ = extract("""
            GL_ShapeDrawer::ShapeCache* GL_ShapeDrawer::cache(int shape) {
                return nullptr;
            }
        """)
        fn = next((f for f in funcs if f.func_name != "unknown"), None)
        assert fn is not None
        assert fn.func_name == "cache"

    def test_double_pointer_return_name_extracted(self):
        funcs, _ = extract("""
            char** get_args(int n) {
                return nullptr;
            }
        """)
        fn = next((f for f in funcs if f.func_name != "unknown"), None)
        assert fn is not None
        assert fn.func_name == "get_args"

    def test_qualified_method_name_extracted(self):
        funcs, _ = extract("""
            bool Log::LogMessage::valid() {
                return metadata != nullptr;
            }
        """)
        fn = next((f for f in funcs if f.func_name != "unknown"), None)
        assert fn is not None
        assert fn.func_name == "valid"

    def test_no_unknown_names_for_common_patterns(self):
        """None of the common pointer-return patterns should produce 'unknown'."""
        code = """
            Node* find(Node* root) { return nullptr; }
            const char* get_name() { return nullptr; }
            int* allocate(int n) { return nullptr; }
            bool* check(int x) { return nullptr; }
        """
        funcs, _ = extract(code)
        unknown = [f for f in funcs if f.func_name == "unknown"]
        assert len(unknown) == 0


# ─── Out-of-class qualified method definitions ────────────────────────────────

class TestOutOfClassMethods:
    def test_qualified_definition_marked_as_method(self):
        # Regression test: func_name already correctly strips the
        # "ClassName::" qualifier, but is_method/parent_class were derived
        # purely from tree-recursion context and never consulted the
        # qualifier itself -- so an out-of-class method definition (a very
        # common C++ pattern: declare in .h, define in .cpp) was
        # incorrectly reported as a free function.
        funcs, _ = extract("""
            Ref<Resource> ResourceLoader::load(const String &p_path) {
                return nullptr;
            }
        """)
        fn = funcs[0]
        assert fn.func_name == "load"
        assert fn.is_method is True
        assert fn.parent_class == "ResourceLoader"

    def test_nested_qualifier_uses_immediate_scope(self):
        funcs, _ = extract("""
            bool Log::LogMessage::valid() {
                return true;
            }
        """)
        fn = funcs[0]
        assert fn.func_name == "valid"
        assert fn.is_method is True
        assert fn.parent_class == "LogMessage"

    def test_free_function_still_not_a_method(self):
        funcs, _ = extract("""
            void free_function(int x) {}
        """)
        fn = funcs[0]
        assert fn.is_method is False
        assert fn.parent_class is None


# ─── Operator overloads and conversion operators ──────────────────────────────

class TestOperatorNames:
    def test_operator_overload_name_extracted(self):
        _, classes = extract("""
            class Foo {
                int operator[](int i) { return i; }
            };
        """)
        assert "operator[]" in classes[0].method_names
        assert "unknown" not in classes[0].method_names

    def test_conversion_operator_name_extracted(self):
        _, classes = extract("""
            class Foo {
                operator int() const { return 0; }
            };
        """)
        assert "operator int" in classes[0].method_names
        assert "unknown" not in classes[0].method_names

    def test_pointer_conversion_operator_name_extracted(self):
        # Edge case: the parameter list nests one level deeper for
        # pointer/reference conversion targets (abstract_pointer_declarator
        # wrapping abstract_function_declarator).
        _, classes = extract("""
            class Ptr {
            public:
                template<class T> operator T*() const { return nullptr; }
            };
        """)
        assert classes[0].method_names == ["operator T*"]

    def test_conversion_operator_has_no_parameters(self):
        funcs, _ = extract("""
            struct Foo {
                operator int() const { return 0; }
            };
        """)
        # class-body walk also emits a FunctionRecord for the method
        method = next(f for f in funcs if f.func_name == "operator int")
        assert method.parameters == []


# ─── Template specializations and template member methods ────────────────────

class TestTemplateSpecializations:
    def test_specialized_class_name_extracted(self):
        # Regression test: class_specifier's name is wrapped one level
        # deeper (template_type -> type_identifier) for specializations,
        # not a direct-child type_identifier like a normal class.
        _, classes = extract("""
            template<> class Array<int, 4> {
            public:
                int get() { return 0; }
            };
        """)
        assert classes[0].class_name == "Array"
        assert classes[0].class_name != "unknown"

    def test_templated_member_method_visible(self):
        # Regression test: templated methods are wrapped in
        # template_declaration inside the class body, which the class-body
        # method scan previously never unwrapped (unlike the top-level walk,
        # which already did) -- so these methods were entirely invisible.
        _, classes = extract("""
            class Ptr {
            public:
                template<class T> operator T*() const { return nullptr; }
            };
        """)
        assert classes[0].method_count == 1
        assert len(classes[0].method_names) == 1

    def test_has_constructor_collision_guard(self):
        # Regression test: when both class_name and a method name
        # independently fall back to "unknown" (e.g. an unresolved
        # specialization alongside an unresolved operator overload),
        # they must not spuriously "match" and produce a false
        # has_constructor=True.
        _, classes = extract("""
            template<> class numeric_limits<BFloat16> {
            public:
                BFloat16 operator[](int i) { return BFloat16(); }
            };
        """)
        assert classes[0].has_constructor is False