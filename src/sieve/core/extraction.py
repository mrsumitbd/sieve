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
    llm_score: Optional[float] = None   # P(LLM-generated) — populated by classifier
    ast_depth: Optional[int] = None     # Max depth of the parse tree
    ast_num_nodes: Optional[int] = None # Total node count
    ast_node_types: Optional[dict] = None  # Node type → count
    ast: Optional[dict] = None          # Full AST as nested JSON (opt-in)


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
    llm_score: Optional[float] = None   # P(LLM-generated) — populated by classifier
    ast_depth: Optional[int] = None     # Max depth of the parse tree
    ast_num_nodes: Optional[int] = None # Total node count
    ast_node_types: Optional[dict] = None  # Node type → count
    ast: Optional[dict] = None          # Full AST as nested JSON (opt-in)


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
        elif language == "C++":
            import tree_sitter_cpp as tscpp
            lang = TSLanguage(tscpp.language())
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


def _node_to_ast_dict(node, source_bytes: bytes, max_depth: int, depth: int = 0) -> dict:
    """Recursively convert a tree-sitter node to a JSON-serializable dict."""
    is_leaf = len(node.children) == 0
    text = node.text.decode("utf-8", errors="replace") if is_leaf else None
    if text and len(text) > 80:
        text = text[:77] + "..."
    result = {
        "type":     node.type,
        "text":     text,
        "children": [],
    }
    if depth < max_depth:
        for child in node.children:
            result["children"].append(
                _node_to_ast_dict(child, source_bytes, max_depth, depth + 1)
            )
    return result


def _compute_ast_features(
    source: str,
    language: str,
    include_full_ast: bool = False,
) -> tuple[Optional[int], Optional[int], Optional[dict], Optional[dict]]:
    """
    Parse source code and return AST-derived features.

    Returns:
        (ast_depth, ast_num_nodes, ast_node_types, ast_json)
        ast_json is None unless include_full_ast=True
    """
    try:
        source_bytes = source.encode("utf-8")
        result = _get_parser(language)
        if result is None:
            return None, None, None, None
        parser, _ = result
        tree = parser.parse(source_bytes)
        root = tree.root_node

        depth_counter = [0]
        node_types: dict[str, int] = {}
        num_nodes = [0]

        def _walk(node, depth):
            num_nodes[0] += 1
            if depth > depth_counter[0]:
                depth_counter[0] = depth
            nt = node.type
            node_types[nt] = node_types.get(nt, 0) + 1
            for child in node.children:
                _walk(child, depth + 1)

        _walk(root, 0)

        ast_json = None
        if include_full_ast:
            ast_json = _node_to_ast_dict(root, source_bytes, max_depth=999)

        return depth_counter[0], num_nodes[0], node_types, ast_json

    except Exception:
        return None, None, None, None


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
                if item.type in ("function_definition", "async_function_definition", "decorated_definition"):
                    func_node = item
                    if item.type == "decorated_definition":
                        for sub in item.children:
                            if sub.type in ("function_definition", "async_function_definition"):
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
                    is_async = any(c.type == "async" for c in func_node.children)
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
    is_async = func_node.type == "async_function_definition" or any(
        c.type == "async" for c in func_node.children
    )
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

    def walk(node, parent_class: Optional[str] = None, decorators: list[str] = None, _depth: int = 0):
        if _depth > 200:
            return
        if decorators is None:
            decorators = []

        if node.type == "decorated_definition":
            # Collect all decorator texts, then recurse into the inner definition
            dec_texts = [
                _node_text(child, source_bytes)
                for child in node.children
                if child.type == "decorator"
            ]
            inner = next(
                (child for child in node.children
                 if child.type in ("function_definition", "async_function_definition", "class_definition")),
                None,
            )
            if inner:
                walk(inner, parent_class=parent_class, decorators=dec_texts, _depth=_depth + 1)
            return

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
                decorators=decorators,
            ))

            for child in node.children:
                walk(child, parent_class, _depth=_depth + 1)

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
                        inner_node = item
                        if item.type == "decorated_definition":
                            inner_node = next(
                                (c for c in item.children
                                 if c.type in ("function_definition", "async_function_definition")),
                                None,
                            )
                        if inner_node and inner_node.type in ("function_definition", "async_function_definition"):
                            mn = inner_node.child_by_field_name("name")
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
                decorators=decorators,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

            for child in node.children:
                if child.type == "block":
                    for item in child.children:
                        walk(item, parent_class=class_name, _depth=_depth + 1)
        else:
            for child in node.children:
                walk(child, parent_class, _depth=_depth + 1)

    walk(root)
    return functions, classes


# ─── Java Extraction ─────────────────────────────────────────────────────────

def _clean_block_comment(raw: str) -> Optional[str]:
    """Strip /* */ or /** */ delimiters and leading ' * ' from each line."""
    inner = re.sub(r"^/\*+", "", raw.strip())
    inner = re.sub(r"\*+/$", "", inner).strip()
    cleaned = "\n".join(re.sub(r"^\s*\*\s?", "", l) for l in inner.splitlines()).strip()
    return cleaned or None

def _java_preceding_comment(node, source_bytes: bytes) -> Optional[str]:
    """
    Return the cleaned text of the block_comment immediately preceding node,
    or None if no such comment exists.  Java docstrings are /** ... */ siblings,
    not children of the declaration node.
    """
    parent = node.parent
    if not parent:
        return None
    siblings = list(parent.children)
    idx = next((i for i, c in enumerate(siblings) if c == node), -1)
    if idx > 0:
        prev = siblings[idx - 1]
        if prev.type == "block_comment":
            raw = _node_text(prev, source_bytes)
            return _clean_block_comment(raw)
    return None


def _java_method_signature(method_node, source_bytes: bytes) -> str:
    """
    Reconstruct Java method signature: everything up to (not including) the body block.
    Returns e.g. 'public int search(List<T> list, T target)'.
    Handles both method_declaration (body=block) and constructor_declaration (body=constructor_body).
    """
    parts = []
    for child in method_node.children:
        if child.type in ("block", "constructor_body"):
            break
        parts.append(_node_text(child, source_bytes))
    return " ".join(parts).strip()


def _build_java_skeleton(class_node, source_bytes: bytes) -> str:
    """
    Build class skeleton: class signature + method signatures + empty bodies.
    """
    lines = []

    # Class signature: everything up to the class_body opening brace
    class_sig_parts = []
    for child in class_node.children:
        if child.type == "class_body":
            break
        class_sig_parts.append(_node_text(child, source_bytes))
    lines.append(" ".join(class_sig_parts).strip() + " {")

    # Class docstring
    class_doc = _java_preceding_comment(class_node, source_bytes)
    if class_doc:
        lines.append(f'    /** {class_doc} */')

    # Methods and constructors
    for child in class_node.children:
        if child.type != "class_body":
            continue
        for item in child.children:
            if item.type in ("method_declaration", "constructor_declaration"):
                doc = _java_preceding_comment(item, source_bytes)
                if doc:
                    lines.append(f"    /** {doc} */")
                sig = _java_method_signature(item, source_bytes)
                lines.append(f"    {sig} {{ }}")
                lines.append("")

    lines.append("}")
    return "\n".join(lines)


def _collect_java_imports(root, source_bytes: bytes) -> list[tuple[str, list[str]]]:
    """
    Collect Java import declarations.
    Returns (statement_text, [names_introduced]) tuples.
    For 'import java.util.List' → name is 'List' (last segment).
    For static 'import static Collections.sort' → name is 'sort'.
    """
    results = []
    for node in root.children:
        if node.type != "import_declaration":
            continue
        stmt_text = _node_text(node, source_bytes)
        for child in node.children:
            if child.type == "scoped_identifier":
                full = _node_text(child, source_bytes)
                name = full.split(".")[-1]
                if name != "*":
                    results.append((stmt_text, [name]))
                else:
                    results.append((stmt_text, ["*"]))
                break
    return results


def _extract_java(source: str, file_path: str, repo: str) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    parser, _ = _get_parser("Java")
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    file_imports = _collect_java_imports(root, source_bytes)
    functions: list[FunctionRecord] = []
    classes: list[ClassRecord] = []

    def walk(node, parent_class: Optional[str] = None, _depth: int = 0):
        if _depth > 200:
            return
        if node.type == "class_declaration":
            # Class name
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            class_name = _node_text(name_node, source_bytes) if name_node else "unknown"

            # Superclass
            parent_classes = []
            for child in node.children:
                if child.type == "superclass":
                    for sc in child.children:
                        if sc.type in ("type_identifier", "generic_type"):
                            parent_classes.append(_node_text(sc, source_bytes))
                elif child.type == "super_interfaces":
                    for sc in child.children:
                        if sc.type == "type_list":
                            for t in sc.children:
                                if t.type in ("type_identifier", "generic_type"):
                                    parent_classes.append(_node_text(t, source_bytes))

            docstring = _java_preceding_comment(node, source_bytes)
            source_code = _clean_indentation(_node_text(node, source_bytes))
            skeleton = _build_java_skeleton(node, source_bytes)
            used_imports = _filter_used_imports(file_imports, source_code)

            # Annotations on the class
            class_annotations = []
            for child in node.children:
                if child.type == "modifiers":
                    for mod in child.children:
                        if mod.type in ("marker_annotation", "annotation"):
                            class_annotations.append(_node_text(mod, source_bytes))

            method_names = []
            has_constructor = False
            for child in node.children:
                if child.type == "class_body":
                    for item in child.children:
                        if item.type == "method_declaration":
                            mn = next((c for c in item.children if c.type == "identifier"), None)
                            if mn:
                                method_names.append(_node_text(mn, source_bytes))
                        elif item.type == "constructor_declaration":
                            has_constructor = True

            classes.append(ClassRecord(
                repo=repo,
                file_path=file_path,
                language="Java",
                class_name=class_name,
                parent_classes=parent_classes,
                docstring=docstring,
                source_code=source_code,
                skeleton=skeleton,
                used_imports=used_imports,
                method_names=method_names,
                method_count=len(method_names),
                has_constructor=has_constructor,
                decorators=class_annotations,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

            # Walk class body for methods
            for child in node.children:
                if child.type == "class_body":
                    for item in child.children:
                        walk(item, parent_class=class_name, _depth=_depth + 1)

        elif node.type in ("method_declaration", "constructor_declaration"):
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            func_name = _node_text(name_node, source_bytes) if name_node else "unknown"

            # Parameters
            params = []
            for child in node.children:
                if child.type == "formal_parameters":
                    for p in child.children:
                        if p.type == "formal_parameter":
                            param_id = next(
                                (c for c in reversed(p.children) if c.type == "identifier"), None
                            )
                            if param_id:
                                params.append(_node_text(param_id, source_bytes))

            # Return type: first type-like node before identifier
            ret_annotation = None
            for child in node.children:
                if child.type == "identifier":
                    break
                if child.type not in ("modifiers", "type_parameters"):
                    ret_annotation = _node_text(child, source_bytes)

            docstring = _java_preceding_comment(node, source_bytes)
            source_code = _clean_indentation(_node_text(node, source_bytes))

            # Annotations from modifiers node
            method_annotations = []
            for child in node.children:
                if child.type == "modifiers":
                    for mod in child.children:
                        if mod.type in ("marker_annotation", "annotation"):
                            method_annotations.append(_node_text(mod, source_bytes))

            # Signature: sig line + docstring + { }
            sig_line = _java_method_signature(node, source_bytes)
            sig_parts = [sig_line + " { }"]
            if docstring:
                sig_parts = [f"/** {docstring} */", sig_line + " { }"]
            signature = "\n".join(sig_parts)

            used_imports = _filter_used_imports(file_imports, source_code)

            functions.append(FunctionRecord(
                repo=repo,
                file_path=file_path,
                language="Java",
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
                decorators=method_annotations,
            ))

        else:
            for child in node.children:
                walk(child, parent_class, _depth=_depth + 1)

    walk(root)
    return functions, classes


# ─── JavaScript Extraction ────────────────────────────────────────────────────

def _js_preceding_comment(node, source_bytes: bytes) -> Optional[str]:
    """
    Return cleaned JSDoc comment immediately preceding node, or None.
    Handles both /** ... */ block comments and // line comments.
    """
    parent = node.parent
    if not parent:
        return None
    siblings = list(parent.children)
    idx = next((i for i, c in enumerate(siblings) if c == node), -1)
    if idx > 0:
        prev = siblings[idx - 1]
        if prev.type == "comment":
            raw = _node_text(prev, source_bytes).strip()
            if raw.startswith("/**"):
                return _clean_block_comment(raw)
            elif raw.startswith("//"):
                return raw.lstrip("/").strip() or None
    return None


def _js_method_signature(method_node, source_bytes: bytes) -> str:
    """
    Reconstruct JS method signature: everything up to (not including) the body block.
    """
    parts = []
    for child in method_node.children:
        if child.type == "statement_block":
            break
        parts.append(_node_text(child, source_bytes))
    return " ".join(parts).strip()


def _build_js_skeleton(class_node, source_bytes: bytes) -> str:
    """
    Build JS class skeleton: class signature + method signatures + empty bodies.
    """
    lines = []

    # Class signature
    sig_parts = []
    for child in class_node.children:
        if child.type == "class_body":
            break
        sig_parts.append(_node_text(child, source_bytes))
    lines.append(" ".join(sig_parts).strip() + " {")

    class_doc = _js_preceding_comment(class_node, source_bytes)
    if class_doc:
        lines.append(f"    /** {class_doc} */")

    for child in class_node.children:
        if child.type != "class_body":
            continue
        for item in child.children:
            if item.type == "method_definition":
                doc = _js_preceding_comment(item, source_bytes)
                if doc:
                    lines.append(f"    /** {doc} */")
                sig = _js_method_signature(item, source_bytes)
                lines.append(f"    {sig} {{ }}")
                lines.append("")

    lines.append("}")
    return "\n".join(lines)


def _collect_js_imports(root, source_bytes: bytes) -> list[tuple[str, list[str]]]:
    """
    Collect ES6 import statements.
    Handles: default imports, named imports, namespace imports.
    """
    results = []
    for node in root.children:
        if node.type != "import_statement":
            continue
        stmt_text = _node_text(node, source_bytes)
        names = []
        for child in node.children:
            if child.type == "import_clause":
                for sub in child.children:
                    if sub.type == "identifier":
                        # default import: import fs from '...'
                        names.append(_node_text(sub, source_bytes))
                    elif sub.type == "named_imports":
                        # named: import { A, B as C }
                        for spec in sub.children:
                            if spec.type == "import_specifier":
                                # alias takes priority
                                alias = spec.child_by_field_name("alias")
                                name_node = spec.child_by_field_name("name")
                                if alias:
                                    names.append(_node_text(alias, source_bytes))
                                elif name_node:
                                    names.append(_node_text(name_node, source_bytes))
                    elif sub.type == "namespace_import":
                        # import * as path
                        for ns in sub.children:
                            if ns.type == "identifier":
                                names.append(_node_text(ns, source_bytes))
        if names:
            results.append((stmt_text, names))
    return results


def _extract_js(source: str, file_path: str, repo: str) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    parser, _ = _get_parser("JavaScript")
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    file_imports = _collect_js_imports(root, source_bytes)
    functions: list[FunctionRecord] = []
    classes: list[ClassRecord] = []

    def walk(node, parent_class: Optional[str] = None, _depth: int = 0):
        if _depth > 200:
            return
        if node.type == "class_declaration":
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            class_name = _node_text(name_node, source_bytes) if name_node else "unknown"

            parent_classes = []
            for child in node.children:
                if child.type == "class_heritage":
                    for sub in child.children:
                        if sub.type == "identifier":
                            parent_classes.append(_node_text(sub, source_bytes))

            docstring = _js_preceding_comment(node, source_bytes)
            source_code = _clean_indentation(_node_text(node, source_bytes))
            skeleton = _build_js_skeleton(node, source_bytes)
            used_imports = _filter_used_imports(file_imports, source_code)

            method_names = []
            has_constructor = False
            for child in node.children:
                if child.type == "class_body":
                    for item in child.children:
                        if item.type == "method_definition":
                            mn = next((c for c in item.children if c.type == "property_identifier"), None)
                            if mn:
                                mname = _node_text(mn, source_bytes)
                                method_names.append(mname)
                                if mname == "constructor":
                                    has_constructor = True

            classes.append(ClassRecord(
                repo=repo,
                file_path=file_path,
                language="JavaScript",
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

            # Walk class body for methods
            for child in node.children:
                if child.type == "class_body":
                    for item in child.children:
                        walk(item, parent_class=class_name, _depth=_depth + 1)

        elif node.type == "function_declaration":
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            func_name = _node_text(name_node, source_bytes) if name_node else "unknown"
            _append_js_function(node, func_name, parent_class)

        elif node.type == "method_definition":
            name_node = next((c for c in node.children if c.type == "property_identifier"), None)
            func_name = _node_text(name_node, source_bytes) if name_node else "unknown"
            _append_js_function(node, func_name, parent_class)

        elif node.type in ("lexical_declaration", "variable_declaration"):
            # const/let/var arrowFunc = (x) => x * 2
            for child in node.children:
                if child.type == "variable_declarator":
                    arrow = next(
                        (c for c in child.children if c.type == "arrow_function"), None
                    )
                    if arrow:
                        name_node = next(
                            (c for c in child.children if c.type == "identifier"), None
                        )
                        func_name = _node_text(name_node, source_bytes) if name_node else "unknown"
                        _append_js_function(arrow, func_name, parent_class, is_arrow=True)

        else:
            for child in node.children:
                walk(child, parent_class, _depth=_depth + 1)

    def _append_js_function(node, func_name: str, parent_class: Optional[str], is_arrow: bool = False):
        params = []
        for child in node.children:
            if child.type == "formal_parameters":
                for p in child.children:
                    if p.type in ("identifier", "assignment_pattern", "rest_pattern"):
                        params.append(_node_text(p, source_bytes))
            elif child.type == "identifier" and is_arrow:
                # single-param arrow: x => x * 2
                params.append(_node_text(child, source_bytes))

        docstring = _js_preceding_comment(node, source_bytes) if not is_arrow else None
        source_code = _clean_indentation(_node_text(node, source_bytes))

        if is_arrow:
            # For arrow functions, source_code should include the full assignment
            # Walk up to the lexical/variable declaration to get `const f = (x) => ...`
            decl = node.parent  # variable_declarator
            if decl and decl.type == "variable_declarator":
                lex = decl.parent  # lexical_declaration or variable_declaration
                if lex and lex.type in ("lexical_declaration", "variable_declaration"):
                    source_code = _clean_indentation(_node_text(lex, source_bytes))

            # Build signature — detect if arrow body is a block or expression
            arrow_text = _node_text(node, source_bytes)
            has_block_body = any(c.type == "statement_block" for c in node.children)

            if has_block_body:
                # Block body: const f = (x) => { ... } → signature ends with { }
                arrow_prefix = arrow_text.split("{")[0].rstrip()
                sig_line = f"{func_name} = {arrow_prefix} {{ }}"
            else:
                # Expression body: const f = (x) => x * 2 → keep as-is, no { }
                sig_line = f"{func_name} = {arrow_text}"

            sig_parts = [sig_line]
        else:
            sig_line = _js_method_signature(node, source_bytes)
            sig_parts = [sig_line + " { }"]

        if docstring:
            sig_parts = [f"/** {docstring} */"] + sig_parts
        signature = "\n".join(sig_parts)

        used_imports = _filter_used_imports(file_imports, source_code)

        functions.append(FunctionRecord(
            repo=repo,
            file_path=file_path,
            language="JavaScript",
            func_name=func_name,
            parameters=params,
            return_annotation=None,  # JS has no return type annotations in vanilla JS
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

    walk(root)
    return functions, classes


# ─── C++ Extraction ──────────────────────────────────────────────────────────

def _cpp_preceding_comment(node, source_bytes: bytes) -> Optional[str]:
    """
    Return cleaned comment immediately preceding node, or None.
    Handles both /** ... */ block comments and // line comments.
    C++ uses the same sibling-comment convention as Java.
    """
    parent = node.parent
    if not parent:
        return None
    siblings = list(parent.children)
    idx = next((i for i, c in enumerate(siblings) if c == node), -1)
    if idx > 0:
        prev = siblings[idx - 1]
        if prev.type == "comment":
            raw = _node_text(prev, source_bytes).strip()
            if raw.startswith("/**") or raw.startswith("/*"):
                return _clean_block_comment(raw)
            elif raw.startswith("//"):
                return raw.lstrip("/").strip() or None
    return None


def _cpp_find_function_declarator(func_node) -> Optional[object]:
    """
    Find the function_declarator node inside a function_definition.
    Handles direct children and pointer/reference-wrapped declarators:
      - void foo()            → function_declarator is direct child
      - Node* foo()           → pointer_declarator → function_declarator
      - const T& foo()        → reference_declarator → function_declarator
      - T** foo()             → pointer_declarator → pointer_declarator → function_declarator
    """
    def _search(node, depth=0) -> Optional[object]:
        if depth > 5:
            return None
        if node.type == "function_declarator":
            return node
        if node.type in ("pointer_declarator", "reference_declarator",
                         "abstract_pointer_declarator"):
            for child in node.children:
                result = _search(child, depth + 1)
                if result:
                    return result
        return None

    for child in func_node.children:
        result = _search(child)
        if result:
            return result
    return None


def _cpp_extract_name_from_declarator(decl_node, source_bytes: bytes) -> str:
    """
    Extract function name from a function_declarator node.
    Handles: identifier, field_identifier, qualified_identifier, destructor_name.
    """
    for sub in decl_node.children:
        if sub.type in ("identifier", "field_identifier"):
            return _node_text(sub, source_bytes)
        elif sub.type == "qualified_identifier":
            # Walk to deepest identifier — e.g. Log::LogMessage::valid → 'valid'
            def _last_id(n) -> Optional[str]:
                for c in reversed(n.children):
                    if c.type in ("identifier", "field_identifier"):
                        return _node_text(c, source_bytes)
                    elif c.type == "qualified_identifier":
                        r = _last_id(c)
                        if r:
                            return r
                return None
            name = _last_id(sub)
            if name:
                return name
        elif sub.type == "destructor_name":
            return _node_text(sub, source_bytes)
    return "unknown"


def _cpp_function_name(func_node, source_bytes: bytes) -> str:
    """
    Extract function name from a function_definition node.
    Handles plain identifiers, field_identifiers, qualified names (Foo::bar),
    pointer return types (Node* foo()), and reference return types (T& foo()).
    """
    decl = _cpp_find_function_declarator(func_node)
    if decl:
        return _cpp_extract_name_from_declarator(decl, source_bytes)
    return "unknown"


def _cpp_function_params(func_node, source_bytes: bytes) -> list[str]:
    """Extract parameter names from a function_definition node."""
    params = []
    decl = _cpp_find_function_declarator(func_node)
    if not decl:
        return params
    for sub in decl.children:
        if sub.type == "parameter_list":
            for p in sub.children:
                if p.type == "parameter_declaration":
                    # Last identifier in the declaration is the param name
                    param_id = next(
                        (c for c in reversed(p.children)
                         if c.type in ("identifier", "reference_declarator",
                                       "pointer_declarator")),
                        None,
                    )
                    if param_id:
                        params.append(_node_text(param_id, source_bytes))
    return params


def _cpp_return_type(func_node, source_bytes: bytes) -> Optional[str]:
    """
    Extract return type from a function_definition node.
    Everything before the function_declarator (or pointer_declarator wrapping it).
    """
    parts = []
    for child in func_node.children:
        # Stop at the declarator chain — pointer_declarator wraps function_declarator
        if child.type in ("function_declarator", "pointer_declarator",
                          "reference_declarator"):
            break
        if child.type not in ("storage_class_specifier", "type_qualifier",
                               "virtual", "explicit", "inline", "static"):
            text = _node_text(child, source_bytes).strip()
            if text:
                parts.append(text)
    return " ".join(parts) if parts else None


def _cpp_method_signature(func_node, source_bytes: bytes) -> str:
    """
    Reconstruct C++ method signature: everything up to (not including) the body.
    Returns e.g. 'void push(int val)'.
    """
    parts = []
    for child in func_node.children:
        if child.type == "compound_statement":
            break
        parts.append(_node_text(child, source_bytes))
    return " ".join(parts).strip()


def _build_cpp_skeleton(class_node, source_bytes: bytes) -> str:
    """
    Build C++ class skeleton: class signature + method signatures + empty bodies.
    """
    lines = []

    # Class signature: everything up to field_declaration_list
    sig_parts = []
    for child in class_node.children:
        if child.type == "field_declaration_list":
            break
        sig_parts.append(_node_text(child, source_bytes))
    lines.append(" ".join(sig_parts).strip() + " {")

    # Class docstring
    class_doc = _cpp_preceding_comment(class_node, source_bytes)
    if class_doc:
        lines.append(f"    /** {class_doc} */")

    # Methods inside field_declaration_list
    for child in class_node.children:
        if child.type != "field_declaration_list":
            continue
        for item in child.children:
            if item.type == "function_definition":
                doc = _cpp_preceding_comment(item, source_bytes)
                if doc:
                    lines.append(f"    /** {doc} */")
                sig = _cpp_method_signature(item, source_bytes)
                lines.append(f"    {sig} {{ }}")
                lines.append("")
            elif item.type == "access_specifier":
                # Preserve public:/private:/protected: labels
                lines.append(_node_text(item, source_bytes) + ":")

    lines.append("};")
    return "\n".join(lines)


def _collect_cpp_includes(root, source_bytes: bytes) -> list[tuple[str, list[str]]]:
    """
    Collect C++ #include directives.
    Returns (statement_text, [header_stem]) tuples.
    e.g. #include <vector> → ("...", ["vector"])
         #include "mylib.h" → ("...", ["mylib"])
    """
    results = []
    for node in root.children:
        if node.type != "preproc_include":
            continue
        stmt_text = _node_text(node, source_bytes)
        for child in node.children:
            if child.type in ("system_lib_string", "string_literal"):
                raw = _node_text(child, source_bytes).strip('<>"')
                # e.g. "vector", "mylib.h" → stem without extension
                stem = Path(raw).stem
                if stem:
                    results.append((stmt_text, [stem]))
                break
    return results


def _extract_cpp(source: str, file_path: str, repo: str) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    parser, _ = _get_parser("C++")
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    file_includes = _collect_cpp_includes(root, source_bytes)
    functions: list[FunctionRecord] = []
    classes: list[ClassRecord] = []

    def walk(node, parent_class: Optional[str] = None, _depth: int = 0):
        if _depth > 200:
            return
        if node.type == "class_specifier":
            name_node = next(
                (c for c in node.children if c.type == "type_identifier"), None
            )
            class_name = _node_text(name_node, source_bytes) if name_node else "unknown"

            # Base classes
            parent_classes = []
            for child in node.children:
                if child.type == "base_class_clause":
                    for sub in child.children:
                        if sub.type == "type_identifier":
                            parent_classes.append(_node_text(sub, source_bytes))

            docstring = _cpp_preceding_comment(node, source_bytes)
            source_code = _clean_indentation(_node_text(node, source_bytes))
            skeleton = _build_cpp_skeleton(node, source_bytes)
            used_imports = _filter_used_imports(file_includes, source_code)

            # Methods
            method_names = []
            has_constructor = False
            for child in node.children:
                if child.type == "field_declaration_list":
                    for item in child.children:
                        if item.type == "function_definition":
                            mname = _cpp_function_name(item, source_bytes)
                            method_names.append(mname)
                            if mname == class_name:
                                has_constructor = True

            classes.append(ClassRecord(
                repo=repo,
                file_path=file_path,
                language="C++",
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

            # Walk into class body for methods
            for child in node.children:
                if child.type == "field_declaration_list":
                    for item in child.children:
                        walk(item, parent_class=class_name, _depth=_depth + 1)

        elif node.type == "function_definition":
            func_name = _cpp_function_name(node, source_bytes)

            # Skip if this is a method we already walked via class body
            # (parent_class is set when called from class walk above)
            params = _cpp_function_params(node, source_bytes)
            ret_annotation = _cpp_return_type(node, source_bytes)
            docstring = _cpp_preceding_comment(node, source_bytes)
            source_code = _clean_indentation(_node_text(node, source_bytes))

            sig_line = _cpp_method_signature(node, source_bytes)
            sig_parts = [sig_line + " { }"]
            if docstring:
                sig_parts = [f"/** {docstring} */", sig_line + " { }"]
            signature = "\n".join(sig_parts)

            used_imports = _filter_used_imports(file_includes, source_code)

            functions.append(FunctionRecord(
                repo=repo,
                file_path=file_path,
                language="C++",
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

        elif node.type == "template_declaration":
            # Templates wrap function_definition or class_specifier — walk into them
            for child in node.children:
                if child.type in ("function_definition", "class_specifier"):
                    walk(child, parent_class, _depth=_depth + 1)

        elif node.type == "namespace_definition":
            # Walk into namespaces transparently
            for child in node.children:
                if child.type == "declaration_list":
                    for item in child.children:
                        walk(item, parent_class, _depth=_depth + 1)

        else:
            for child in node.children:
                walk(child, parent_class, _depth=_depth + 1)

    walk(root)
    return functions, classes


# ─── Public API ──────────────────────────────────────────────────────────────

LANGUAGE_EXTENSIONS = {
    "Python": {".py"},
    "Java": {".java"},
    "JavaScript": {".js", ".ts", ".jsx", ".tsx"},
    "C++": {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"},
}


def extract_from_file(
    file_path: str,
    language: str,
    repo: str,
    relative_path: str = None,
    include_ast: bool = False,
) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    """
    Extract function and class records from a single source file.

    Args:
        file_path:    Absolute path to the file (used for reading)
        language:     Target language
        repo:         Repo name (e.g. "owner/repo")
        relative_path: Path relative to repo root stored in records.
                       Falls back to file_path if not provided.
        include_ast:  If True, populate ast_depth, ast_num_nodes,
                      ast_node_types, and full ast JSON on every record.
                      If False, only ast_depth, ast_num_nodes, ast_node_types
                      are populated (full ast stays None).

    Returns:
        (functions, classes) — lists of records extracted from this file
    """
    stored_path = relative_path if relative_path is not None else file_path
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return [], []

    # ── Minification filter (JS only) ─────────────────────────────────────────
    if language == "JavaScript":
        fp = stored_path.lower()
        if fp.endswith(".min.js") or "node_modules" in fp:
            logger.debug(f"Skipping minified/vendor JS: {stored_path}")
            return [], []
        lines = source.splitlines()
        if lines:
            max_line = max((len(l) for l in lines), default=0)
            avg_line = sum(len(l) for l in lines) / len(lines)
            if max_line > 1000 or avg_line > 300:
                logger.debug(f"Skipping minified JS (max_line={max_line}, avg={avg_line:.0f}): {stored_path}")
                return [], []

    try:
        if language == "Python":
            functions, classes = _extract_python(source, stored_path, repo)
        elif language == "Java":
            functions, classes = _extract_java(source, stored_path, repo)
        elif language == "JavaScript":
            functions, classes = _extract_js(source, stored_path, repo)
        elif language == "C++":
            functions, classes = _extract_cpp(source, stored_path, repo)
        else:
            logger.warning(f"Extraction for {language} not yet implemented.")
            return [], []
    except RecursionError:
        logger.warning(f"Recursion limit hit parsing {file_path} — skipping file")
        return [], []
    except Exception as e:
        logger.warning(f"Extraction failed for {file_path}: {e} — skipping file")
        return [], []

    # ── AST feature annotation ────────────────────────────────────────────────
    for record in functions + classes:
        depth, num_nodes, node_types, ast_json = _compute_ast_features(
            record.source_code, language, include_full_ast=include_ast
        )
        record.ast_depth      = depth
        record.ast_num_nodes  = num_nodes
        record.ast_node_types = node_types
        record.ast            = ast_json

    return functions, classes


def count_repo_contents(
    repo_path: str,
    language: str,
    granularities: list[str],
) -> tuple[int, int]:
    """
    Lightweight pass 1: count extractable functions and classes in a repo
    without full extraction. Uses tree-sitter node counting only.

    Returns:
        (num_functions, num_classes)
    """
    extensions = LANGUAGE_EXTENSIONS.get(language, set())
    num_functions = 0
    num_classes   = 0

    # Node types that represent functions and classes per language
    FUNC_TYPES = {
        "Python":     {"function_definition", "async_function_definition"},
        "Java":       {"method_declaration", "constructor_declaration"},
        "JavaScript": {"function_declaration", "function", "arrow_function",
                       "method_definition"},
        "C++":        {"function_definition"},
    }
    CLASS_TYPES = {
        "Python":     {"class_definition"},
        "Java":       {"class_declaration"},
        "JavaScript": {"class_declaration"},
        "C++":        {"class_specifier"},
    }

    func_types  = FUNC_TYPES.get(language, set())
    class_types = CLASS_TYPES.get(language, set())

    root = Path(repo_path)

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in extensions:
            continue
        # Same test file exclusions as extract_from_repo
        if any(p in file_path.parts for p in {"test", "tests", "__tests__", "spec", "specs"}):
            continue
        if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
            continue
        if file_path.name.endswith(".test.js") or file_path.name.endswith(".spec.js"):
            continue
        if file_path.name.endswith("_test.cpp") or file_path.name.endswith("_test.cc"):
            continue

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")

            # Skip minified JS
            if language == "JavaScript":
                lines = source.splitlines()
                if lines:
                    max_line = max((len(l) for l in lines), default=0)
                    avg_line = sum(len(l) for l in lines) / len(lines)
                    if max_line > 1000 or avg_line > 300:
                        continue

            parser = _get_parser(language)
            if parser is None:
                continue
            parser, _ = parser
            tree = parser.parse(source.encode("utf-8"))

            def _count(node):
                fc, cc = 0, 0
                if node.type in func_types:
                    fc += 1
                if node.type in class_types:
                    cc += 1
                for child in node.children:
                    cf, cl = _count(child)
                    fc += cf
                    cc += cl
                return fc, cc

            fc, cc = _count(tree.root_node)
            if "function" in granularities:
                num_functions += fc
            if "class" in granularities:
                num_classes += cc

        except Exception:
            continue

    return num_functions, num_classes


def extract_from_repo(
    repo_path: str,
    language: str,
    repo_name: str,
    granularities: list[str],
    include_ast: bool = False,
    func_cap: Optional[int] = None,
    class_cap: Optional[int] = None,
) -> tuple[list[FunctionRecord], list[ClassRecord]]:
    """
    Walk a cloned repo and extract all matching source files.

    Args:
        repo_path:    Absolute path to cloned repo
        language:     Target language
        repo_name:    Full repo name (e.g. "owner/repo")
        granularities: List of granularity strings from config
        include_ast:  If True, populate full AST JSON on every record
        func_cap:     Stop extracting functions after this many (None = no cap)
        class_cap:    Stop extracting classes after this many (None = no cap)

    Returns:
        (all_functions, all_classes) across the entire repo
    """
    extensions = LANGUAGE_EXTENSIONS.get(language, set())
    all_functions: list[FunctionRecord] = []
    all_classes: list[ClassRecord] = []

    root = Path(repo_path)

    for file_path in root.rglob("*"):
        # Early stop if both caps reached
        if (func_cap is not None and len(all_functions) >= func_cap and
                class_cap is not None and len(all_classes) >= class_cap):
            break
        if not file_path.is_file():
            continue
        if file_path.suffix not in extensions:
            continue
        # Skip test files from the extraction corpus
        if any(p in file_path.parts for p in {"test", "tests", "__tests__", "spec", "specs"}):
            continue
        if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
            continue
        if file_path.name.startswith("Test") or file_path.name.endswith("Test.java"):
            continue
        if file_path.name.endswith(".test.js") or file_path.name.endswith(".spec.js"):
            continue
        if file_path.name.endswith(".test.ts") or file_path.name.endswith(".spec.ts"):
            continue
        if file_path.name.endswith("_test.cpp") or file_path.name.endswith("_test.cc"):
            continue
        if file_path.name.startswith("test_") and file_path.suffix in {".cpp", ".cc", ".h"}:
            continue

        funcs, classes = extract_from_file(
            str(file_path),
            language,
            repo_name,
            relative_path=str(file_path.relative_to(root)),
            include_ast=include_ast,
        )

        if "function" in granularities:
            if func_cap is not None:
                remaining = func_cap - len(all_functions)
                all_functions.extend(funcs[:remaining])
            else:
                all_functions.extend(funcs)

        if "class" in granularities:
            if class_cap is not None:
                remaining = class_cap - len(all_classes)
                all_classes.extend(classes[:remaining])
            else:
                all_classes.extend(classes)

        logger.debug(f"  {file_path.name}: {len(funcs)} functions, {len(classes)} classes")

    return all_functions, all_classes