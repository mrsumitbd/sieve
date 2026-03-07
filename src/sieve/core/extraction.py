"""
core/extraction.py

Tree-sitter based AST extraction for functions and classes.
Supports Python, Java, and JavaScript.

Reuses the concept from OpenClassGen's extract_skeleton() but:
- Uses tree-sitter instead of Python's ast module for language agnosticism
- Extracts both functions and classes in a single pass
- Captures richer metadata (line numbers, docstrings, decorators, parameters)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class FunctionRecord:
    """CodeSearchNet-style function record."""
    repo: str
    file_path: str
    language: str
    func_name: str
    parameters: list[str]
    return_annotation: Optional[str]
    docstring: Optional[str]
    source_code: str          # Full function body
    signature: str            # def line + docstring + pass (no body)
    used_imports: list[str]   # Import statements whose names are referenced in this function
    start_line: int
    end_line: int
    is_method: bool           # True if inside a class
    parent_class: Optional[str]
    decorators: list[str]
    llm_score: Optional[float] = None   # P(LLM-generated) — populated by classifier (future)


@dataclass
class ClassRecord:
    """OpenClassEval-style class record."""
    repo: str
    file_path: str
    language: str
    class_name: str
    parent_classes: list[str]
    docstring: Optional[str]
    source_code: str          # Full class body
    skeleton: str             # Class skeleton (signatures + docstrings + pass)
    used_imports: list[str]   # Import statements whose names are referenced in this class
    method_names: list[str]
    method_count: int
    has_constructor: bool
    decorators: list[str]
    start_line: int
    end_line: int
    llm_score: Optional[float] = None   # P(LLM-generated) — populated by classifier (future)


# ─── Tree-sitter Setup ───────────────────────────────────────────────────────

_PARSERS: dict = {}

def _get_parser(language: str):
    """Lazy-load and cache tree-sitter parsers per language."""
    if language in _PARSERS:
        return _PARSERS[language]

    try:
        from tree_sitter import Language as TSLanguage, Parser

        if language == "Python":
            import tree_sitter_python as tspython
            lang = TSLanguage(tspython.language())
        elif language == "Java":
            import tree_sitter_java as tsjava
            lang = TSLanguage(tsjava.language())
        elif language == "JavaScript":
            import tree_sitter_javascript as tsjavascript
            lang = TSLanguage(tsjavascript.language())
        else:
            raise ValueError(f"Unsupported language: {language}")

        parser = Parser(lang)
        _PARSERS[language] = (parser, lang)
        return parser, lang

    except ImportError as e:
        raise ImportError(
            f"tree-sitter grammar for {language} not installed. "
            f"Run: pip install tree-sitter-{language.lower()}"
        ) from e


# ─── Generic Utilities ───────────────────────────────────────────────────────

def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _clean_indentation(snippet: str) -> str:
    """
    Normalizes indentation of extracted code snippets.

    Tree-sitter gives us node text starting at the first token (e.g. the `def`
    keyword), so the first line has zero leading spaces even for methods.
    The body lines still carry their absolute indentation from the source file:
      - Top-level function body: 4 spaces  (correct, leave unchanged)
      - Method body inside a class: 8 spaces (strip 4 → normalize to 4)
      - Method inside nested class: 12 spaces (strip 8 → normalize to 4)

    Strategy: if the minimum body indentation exceeds 4 spaces, strip the
    excess (min_indent - 4) so the body is always at one standard indent level.
    Code that is already at 4-space (or 2-space) indentation is left unchanged.
    """
    lines = snippet.split("\n")
    if len(lines) <= 1:
        return snippet
    body_lines = [l for l in lines[1:] if l.strip()]
    if not body_lines:
        return snippet
    min_indent = min(len(l) - len(l.lstrip()) for l in body_lines)
    if min_indent <= 4:
        # Already at sensible indentation — top-level function or 2-space style
        return snippet
    strip_n = min_indent - 4
    stripped_body = [
        l[strip_n:] if len(l) >= strip_n else l
        for l in lines[1:]
    ]
    return lines[0] + "\n" + "\n".join(stripped_body)


# ─── Python Extraction ───────────────────────────────────────────────────────

def _extract_python_docstring(node, source_bytes: bytes) -> Optional[str]:
    """Extract docstring from the first statement of a function/class body."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for sub in stmt.children:
                        if sub.type == "string":
                            raw = _node_text(sub, source_bytes)
                            return raw.strip('"""').strip("'''").strip('"').strip("'").strip()
            break
    return None


def _build_python_skeleton(class_node, source_bytes: bytes) -> str:
    """
    Rebuild class skeleton: class signature + method signatures + docstrings + pass.
    Mirrors the logic from OpenClassGen's extract_skeleton() but operates on tree-sitter nodes.
    """
    lines = []
    class_text = _node_text(class_node, source_bytes)

    # Class signature line
    sig_end = class_text.index(":") + 1
    class_sig = class_text[:sig_end].split("\n")[0]
    lines.append(class_sig)

    # Class docstring
    class_doc = _extract_python_docstring(class_node, source_bytes)
    if class_doc:
        lines.append(f'    """{class_doc}"""')
    lines.append("")

    # Method signatures
    for child in class_node.children:
        if child.type == "block":
            for item in child.children:
                if item.type in ("function_definition", "decorated_definition"):
                    func_node = item
                    if item.type == "decorated_definition":
                        for sub in item.children:
                            if sub.type == "function_definition":
                                func_node = sub
                                break
                        # Include decorators in skeleton
                        for sub in item.children:
                            if sub.type == "decorator":
                                lines.append(f"    {_node_text(sub, source_bytes)}")

                    # Reconstruct signature from named fields to avoid
                    # choking on colons inside type annotations (e.g. x: int)
                    name_node = func_node.child_by_field_name("name")
                    params_node = func_node.child_by_field_name("parameters")
                    ret_node = func_node.child_by_field_name("return_type")
                    func_name_str = _node_text(name_node, source_bytes) if name_node else "unknown"
                    params_str = _node_text(params_node, source_bytes) if params_node else "()"
                    ret_str = f" -> {_node_text(ret_node, source_bytes)}" if ret_node else ""
                    is_async = func_node.type == "async_function_definition"
                    prefix = "async def " if is_async else "def "
                    func_sig = f"{prefix}{func_name_str}{params_str}{ret_str}:"
                    lines.append(f"    {func_sig.lstrip()}")

                    func_doc = _extract_python_docstring(func_node, source_bytes)
                    if func_doc:
                        lines.append(f'        """{func_doc}"""')
                    lines.append("        pass")
                    lines.append("")
    # If the class has no methods at all (e.g. Enum, dataclass with only
    # assignments), the body would be empty — add pass to keep it valid Python
    has_any_method = any(
        child.type in ("function_definition", "async_function_definition", "decorated_definition")
        for block in class_node.children if block.type == "block"
        for child in block.children
    )
    if not has_any_method:
        lines.append("    pass")

    return "\n".join(lines)


def _build_python_function_signature(func_node, source_bytes: bytes) -> str:
    """
    Build a function signature stub: def line + docstring (if any) + pass.
    Mirrors the method-level logic in _build_python_skeleton.
    """
    name_node = func_node.child_by_field_name("name")
    params_node = func_node.child_by_field_name("parameters")
    ret_node = func_node.child_by_field_name("return_type")

    func_name_str = _node_text(name_node, source_bytes) if name_node else "unknown"
    params_str = _node_text(params_node, source_bytes) if params_node else "()"
    ret_str = f" -> {_node_text(ret_node, source_bytes)}" if ret_node else ""
    is_async = func_node.type == "async_function_definition"
    prefix = "async def " if is_async else "def "
    sig_line = f"{prefix}{func_name_str}{params_str}{ret_str}:"

    lines = [sig_line]
    docstring = _extract_python_docstring(func_node, source_bytes)
    if docstring:
        lines.append(f'    """{docstring}"""')
    lines.append("    pass")
    return "\n".join(lines)


def _collect_python_imports(root, source_bytes: bytes) -> list[tuple[str, list[str]]]:
    """
    Walk top-level nodes and collect all import statements.

    Returns a list of (statement_text, [names_introduced]) tuples.
    Names introduced are the identifiers that end up in the module namespace
    and that might be referenced inside functions/classes.

    Examples:
      import os                       → ("import os", ["os"])
      import os.path                  → ("import os.path", ["os"])
      import numpy as np              → ("import numpy as np", ["np"])
      from pathlib import Path        → ("from pathlib import Path", ["Path"])
      from typing import List, Dict   → ("from typing import List, Dict", ["List", "Dict"])
      from . import utils             → ("from . import utils", ["utils"])
    """
    results: list[tuple[str, list[str]]] = []

    for node in root.children:
        if node.type == "import_statement":
            stmt_text = _node_text(node, source_bytes)
            names = []
            for child in node.children:
                if child.type == "dotted_name":
                    # import os.path → "os" is what enters the namespace
                    names.append(_node_text(child, source_bytes).split(".")[0])
                elif child.type == "aliased_import":
                    # import numpy as np → "np"
                    alias = child.child_by_field_name("alias")
                    if alias:
                        names.append(_node_text(alias, source_bytes))
                    else:
                        name_part = child.child_by_field_name("name")
                        if name_part:
                            names.append(_node_text(name_part, source_bytes).split(".")[0])
            if names:
                results.append((stmt_text, names))

        elif node.type == "import_from_statement":
            stmt_text = _node_text(node, source_bytes)
            names = []
            # Children after "from X import" are the imported names
            importing = False
            for child in node.children:
                if child.type == "import":
                    importing = True
                    continue
                if importing:
                    if child.type == "dotted_name":
                        names.append(_node_text(child, source_bytes))
                    elif child.type == "aliased_import":
                        alias = child.child_by_field_name("alias")
                        if alias:
                            names.append(_node_text(alias, source_bytes))
                        else:
                            name_part = child.child_by_field_name("name")
                            if name_part:
                                names.append(_node_text(name_part, source_bytes))
                    elif child.type == "wildcard_import":
                        # from x import * — include unconditionally
                        names.append("*")
            if names:
                results.append((stmt_text, names))

    return results


def _filter_used_imports(
    imports: list[tuple[str, list[str]]],
    source_code: str,
) -> list[str]:
    """
    Return import statement strings whose introduced names appear as whole
    tokens in source_code.  Uses word-boundary regex to avoid false positives
    (e.g. 'os' matching inside 'cosmos').
    Wildcard imports (from x import *) are always included.
    """
    used = []
    for stmt_text, names in imports:
        if "*" in names:
            used.append(stmt_text)
            continue
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", source_code):
                used.append(stmt_text)
                break
    return used


def _extract_python(source: str, file_path: str, repo: str) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    parser, _ = _get_parser("Python")
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # Collect all file-level imports once — used to annotate every record
    file_imports = _collect_python_imports(root, source_bytes)

    functions: list[FunctionRecord] = []
    classes: list[ClassRecord] = []

    def walk(node, parent_class: Optional[str] = None):
        if node.type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            func_name = _node_text(name_node, source_bytes) if name_node else "unknown"

            params_node = node.child_by_field_name("parameters")
            params = []
            if params_node:
                for p in params_node.children:
                    if p.type not in (",", "(", ")", "comment"):
                        params.append(_node_text(p, source_bytes))

            ret_node = node.child_by_field_name("return_type")
            ret_annotation = _node_text(ret_node, source_bytes) if ret_node else None

            docstring = _extract_python_docstring(node, source_bytes)
            source_code = _clean_indentation(_node_text(node, source_bytes))
            signature = _build_python_function_signature(node, source_bytes)
            used_imports = _filter_used_imports(file_imports, source_code)

            functions.append(FunctionRecord(
                repo=repo,
                file_path=file_path,
                language="Python",
                func_name=func_name,
                parameters=params,
                return_annotation=ret_annotation,
                docstring=docstring,
                source_code=source_code,
                signature=signature,
                used_imports=used_imports,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                is_method=parent_class is not None,
                parent_class=parent_class,
                decorators=[],
            ))

            for child in node.children:
                walk(child, parent_class)

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(name_node, source_bytes) if name_node else "unknown"

            bases_node = node.child_by_field_name("superclasses")
            parent_classes = []
            if bases_node:
                for b in bases_node.children:
                    if b.type not in (",", "(", ")", "comment"):
                        parent_classes.append(_node_text(b, source_bytes))

            docstring = _extract_python_docstring(node, source_bytes)
            source_code = _clean_indentation(_node_text(node, source_bytes))
            skeleton = _build_python_skeleton(node, source_bytes)
            used_imports = _filter_used_imports(file_imports, source_code)

            method_names = []
            has_constructor = False
            for child in node.children:
                if child.type == "block":
                    for item in child.children:
                        if item.type in ("function_definition", "async_function_definition"):
                            mn = item.child_by_field_name("name")
                            if mn:
                                mname = _node_text(mn, source_bytes)
                                method_names.append(mname)
                                if mname == "__init__":
                                    has_constructor = True

            classes.append(ClassRecord(
                repo=repo,
                file_path=file_path,
                language="Python",
                class_name=class_name,
                parent_classes=parent_classes,
                docstring=docstring,
                source_code=source_code,
                skeleton=skeleton,
                used_imports=used_imports,
                method_names=method_names,
                method_count=len(method_names),
                has_constructor=has_constructor,
                decorators=[],
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

            for child in node.children:
                if child.type == "block":
                    for item in child.children:
                        walk(item, parent_class=class_name)
        else:
            for child in node.children:
                walk(child, parent_class)

    walk(root)
    return functions, classes


# ─── Public API ──────────────────────────────────────────────────────────────

LANGUAGE_EXTENSIONS = {
    "Python": {".py"},
    "Java": {".java"},
    "JavaScript": {".js", ".ts", ".jsx", ".tsx"},
}


def extract_from_file(
    file_path: str,
    language: str,
    repo: str,
    relative_path: str = None,
) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    """
    Extract function and class records from a single source file.

    Args:
        file_path: Absolute path to the file (used for reading)
        language: Target language
        repo: Repo name (e.g. "owner/repo")
        relative_path: Path relative to repo root stored in records.
                       Falls back to file_path if not provided.

    Returns:
        (functions, classes) — lists of records extracted from this file
    """
    stored_path = relative_path if relative_path is not None else file_path
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return [], []

    if language == "Python":
        return _extract_python(source, stored_path, repo)
    else:
        logger.warning(f"Extraction for {language} not yet implemented.")
        return [], []


def extract_from_repo(
    repo_path: str,
    language: str,
    repo_name: str,
    granularities: list[str],
) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    """
    Walk a cloned repo and extract all matching source files.

    Args:
        repo_path: Absolute path to cloned repo
        language: Target language
        repo_name: Full repo name (e.g. "owner/repo")
        granularities: List of granularity strings from config

    Returns:
        (all_functions, all_classes) across the entire repo
    """
    extensions = LANGUAGE_EXTENSIONS.get(language, set())
    all_functions: list[FunctionRecord] = []
    all_classes: list[ClassRecord] = []

    root = Path(repo_path)

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in extensions:
            continue
        # Skip test files from the extraction corpus
        # They are catalogued by detection.py, not included in generation datasets
        if any(p in file_path.parts for p in {"test", "tests", "__tests__", "spec", "specs"}):
            continue
        if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
            continue

        funcs, classes = extract_from_file(
            str(file_path),
            language,
            repo_name,
            relative_path=str(file_path.relative_to(root)),
        )

        if "function" in granularities:
            all_functions.extend(funcs)
        if "class" in granularities:
            all_classes.extend(classes)

        logger.debug(f"  {file_path.name}: {len(funcs)} functions, {len(classes)} classes")

    return all_functions, all_classes