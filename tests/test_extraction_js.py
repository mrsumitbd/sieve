"""
tests/test_extraction_js.py

Unit tests for the JavaScript tree-sitter extractor.
"""

import textwrap

import pytest

from sieve.core.extraction import _extract_js


def extract(code: str):
    return _extract_js(textwrap.dedent(code).strip(), "test.js", "owner/repo")


class TestFunctionExtraction:
    def test_function_declaration_extracted(self):
        funcs, _ = extract("""
            function add(a, b) {
                return a + b;
            }
        """)
        assert any(f.func_name == "add" for f in funcs)

    def test_arrow_function_extracted(self):
        funcs, _ = extract("""
            const double = (x) => x * 2;
        """)
        assert any(f.func_name == "double" for f in funcs)

    def test_jsdoc_captured(self):
        funcs, _ = extract("""
            /** Copy a file. */
            function copyFile(src, dest) {
                return src;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "copyFile")
        assert fn.docstring == "Copy a file."

    def test_parameters_captured(self):
        funcs, _ = extract("""
            function greet(name, greeting) {
                return greeting + name;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "greet")
        assert "name" in fn.parameters
        assert "greeting" in fn.parameters

    def test_destructured_object_parameter_captured(self):
        # Regression test: object_pattern was previously silently dropped
        # entirely rather than captured, even in plain .js (not TS-specific) --
        # confirmed on a real-world Vite config's manualChunks(id, { ... }).
        funcs, _ = extract("""
            function manualChunks(id, { getModuleInfo, getModuleIds }) {
                return id;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "manualChunks")
        assert len(fn.parameters) == 2
        assert fn.parameters[0] == "id"
        assert "getModuleInfo" in fn.parameters[1]

    def test_destructured_array_parameter_captured(self):
        funcs, _ = extract("""
            function first([a, b]) {
                return a;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "first")
        assert len(fn.parameters) == 1
        assert "a" in fn.parameters[0]

    def test_signature_has_no_body(self):
        funcs, _ = extract("""
            function add(a, b) {
                return a + b;
            }
        """)
        fn = next(f for f in funcs if f.func_name == "add")
        assert "return" not in fn.signature
        assert "{ }" in fn.signature


class TestClassExtraction:
    def test_class_extracted(self):
        _, classes = extract("""
            class Animal {
                constructor(name) {
                    this.name = name;
                }
                speak() {}
            }
        """)
        assert len(classes) == 1
        assert classes[0].class_name == "Animal"

    def test_extends_captured(self):
        _, classes = extract("""
            class Dog extends Animal {
                bark() {}
            }
        """)
        assert "Animal" in classes[0].parent_classes

    def test_has_constructor_true(self):
        _, classes = extract("""
            class Foo {
                constructor() {}
            }
        """)
        assert classes[0].has_constructor is True

    def test_method_names_captured(self):
        _, classes = extract("""
            class Service {
                constructor() {}
                start() {}
                stop() {}
            }
        """)
        assert "start" in classes[0].method_names
        assert "stop" in classes[0].method_names

    def test_jsdoc_on_class_captured(self):
        _, classes = extract("""
            /** Utility class. */
            class Utils {
                static noop() {}
            }
        """)
        assert classes[0].docstring == "Utility class."

    def test_skeleton_has_no_body(self):
        _, classes = extract("""
            class Calc {
                add(a, b) { return a + b; }
                sub(a, b) { return a - b; }
            }
        """)
        skeleton = classes[0].skeleton
        assert "add" in skeleton
        assert "return" not in skeleton


class TestLanguageExtensions:
    def test_typescript_extensions_excluded(self):
        # Regression test: .ts/.tsx were previously included under
        # "JavaScript" even though SIEVE only ships a JavaScript grammar
        # with no real TypeScript support -- parsing .ts/.tsx with it
        # silently corrupted typed parameters rather than failing loudly
        # (e.g. "store: Store" misparsed with "Store" captured as if it
        # were the parameter name). TypeScript is out of scope; only
        # extensions SIEVE can actually parse correctly should be listed.
        from sieve.core.extraction import LANGUAGE_EXTENSIONS
        js_extensions = LANGUAGE_EXTENSIONS["JavaScript"]
        assert ".ts" not in js_extensions
        assert ".tsx" not in js_extensions
        assert ".js" in js_extensions
        assert ".jsx" in js_extensions


class TestImportDetection:
    def test_default_import_detected(self):
        funcs, _ = extract("""
            import fs from 'fs';

            function copy(src, dest) {
                fs.copyFileSync(src, dest);
            }
        """)
        fn = next(f for f in funcs if f.func_name == "copy")
        assert any("fs" in imp for imp in fn.used_imports)

    def test_named_import_detected(self):
        funcs, _ = extract("""
            import { readFile } from 'fs/promises';

            async function read(path) {
                return readFile(path, 'utf8');
            }
        """)
        fn = next(f for f in funcs if f.func_name == "read")
        assert any("readFile" in imp for imp in fn.used_imports)

    def test_namespace_import_detected(self):
        funcs, _ = extract("""
            import * as path from 'path';

            function join(a, b) {
                return path.join(a, b);
            }
        """)
        fn = next(f for f in funcs if f.func_name == "join")
        assert any("path" in imp for imp in fn.used_imports)

    def test_unused_import_excluded(self):
        funcs, _ = extract("""
            import fs from 'fs';
            import os from 'os';

            function add(a, b) { return a + b; }
        """)
        fn = next(f for f in funcs if f.func_name == "add")
        assert fn.used_imports == []