"""
sieve/core/metrics.py

Cross-language code metrics computed from tree-sitter ASTs.

Metrics computed:
  Raw:       loc, sloc, lloc, comments, multi, blank, comment_ratio
  Cyclomatic: cyclomatic_complexity (McCabe)
  Nesting:   max_nesting_depth
  Halstead:  h1, h2, N1, N2, vocabulary, length, calculated_length,
             volume, difficulty, effort, time, bugs
  Composite: maintainability_index

All metrics are computed from isolated code snippets — fan-in and fan-out
require whole-repo call graph analysis and are not supported.

Supports: Python, Java, JavaScript, C++
"""

import math
import re
from typing import Optional


# ─── Operator / Operand Node Type Sets ───────────────────────────────────────

# Operators: nodes that represent operations, keywords, punctuation with semantics
_OPERATORS: dict[str, set[str]] = {
    "Python": {
        "binary_operator", "unary_operator", "comparison_operator",
        "boolean_operator", "augmented_assignment", "assignment",
        "return_statement", "yield", "yield_from",
        "raise_statement", "assert_statement", "delete_statement",
        "import_statement", "import_from_statement",
        "if_statement", "for_statement", "while_statement",
        "with_statement", "try_statement", "except_clause",
        "lambda", "list_comprehension", "set_comprehension",
        "dictionary_comprehension", "generator_expression",
        "conditional_expression",
        "+", "-", "*", "/", "//", "%", "**",
        "==", "!=", "<", ">", "<=", ">=",
        "and", "or", "not", "in", "not in", "is", "is not",
        "&", "|", "^", "~", "<<", ">>",
        "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=",
        "&=", "|=", "^=", "<<=", ">>=",
    },
    "Java": {
        "binary_expression", "unary_expression", "assignment_expression",
        "update_expression", "ternary_expression",
        "instanceof_expression", "cast_expression",
        "if_statement", "for_statement", "while_statement", "do_statement",
        "switch_expression", "return_statement", "throw_statement",
        "try_statement", "catch_clause",
        "+", "-", "*", "/", "%",
        "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "!",
        "&", "|", "^", "~", "<<", ">>", ">>>",
        "=", "+=", "-=", "*=", "/=", "%=",
        "&=", "|=", "^=", "<<=", ">>=", ">>>=",
        "++", "--",
    },
    "JavaScript": {
        "binary_expression", "unary_expression", "assignment_expression",
        "update_expression", "ternary_expression", "await_expression",
        "yield_expression", "sequence_expression",
        "if_statement", "for_statement", "for_in_statement",
        "while_statement", "do_statement", "switch_statement",
        "return_statement", "throw_statement", "try_statement",
        "+", "-", "*", "/", "%", "**",
        "==", "!=", "===", "!==", "<", ">", "<=", ">=",
        "&&", "||", "!", "??",
        "&", "|", "^", "~", "<<", ">>", ">>>",
        "=", "+=", "-=", "*=", "/=", "%=", "**=",
        "&=", "|=", "^=", "<<=", ">>=", ">>>=", "??=",
        "++", "--", "=>",
    },
    "C++": {
        "binary_expression", "unary_expression", "assignment_expression",
        "update_expression", "conditional_expression", "comma_expression",
        "pointer_expression", "cast_expression", "sizeof_expression",
        "if_statement", "for_statement", "while_statement", "do_statement",
        "switch_statement", "return_statement", "throw_statement",
        "try_statement",
        "+", "-", "*", "/", "%",
        "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "!",
        "&", "|", "^", "~", "<<", ">>",
        "=", "+=", "-=", "*=", "/=", "%=",
        "&=", "|=", "^=", "<<=", ">>=",
        "++", "--", "->", ".",
    },
}

# Operands: identifiers, literals
_OPERAND_TYPES: dict[str, set[str]] = {
    "Python": {
        "identifier", "integer", "float", "string", "true", "false", "none",
        "concatenated_string", "bytes",
    },
    "Java": {
        "identifier", "decimal_integer_literal", "hex_integer_literal",
        "octal_integer_literal", "binary_integer_literal",
        "decimal_floating_point_literal", "hex_floating_point_literal",
        "string_literal", "character_literal", "true", "false", "null_literal",
        "type_identifier",
    },
    "JavaScript": {
        "identifier", "number", "string", "template_string",
        "regex", "true", "false", "null", "undefined",
    },
    "C++": {
        "identifier", "number_literal", "string_literal", "char_literal",
        "true", "false", "nullptr", "type_identifier", "field_identifier",
    },
}

# Branching nodes for cyclomatic complexity
_BRANCH_NODES: dict[str, set[str]] = {
    "Python": {
        "if_statement", "elif_clause", "for_statement", "while_statement",
        "except_clause", "with_statement", "assert_statement",
        "boolean_operator", "conditional_expression",
        "list_comprehension", "set_comprehension",
        "dictionary_comprehension", "generator_expression",
    },
    "Java": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "switch_expression", "catch_clause", "conditional_expression",
        "instanceof_expression",
    },
    "JavaScript": {
        "if_statement", "for_statement", "for_in_statement", "while_statement",
        "do_statement", "switch_case", "catch_clause", "ternary_expression",
        "logical_expression", "await_expression",
    },
    "C++": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "switch_statement", "case_statement", "catch_clause",
        "conditional_expression",
    },
}

# Control flow nodes for nesting depth
_CONTROL_NODES: dict[str, set[str]] = {
    "Python": {
        "if_statement", "for_statement", "while_statement",
        "with_statement", "try_statement", "except_clause",
        "match_statement",
    },
    "Java": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "try_statement", "catch_clause", "switch_expression",
    },
    "JavaScript": {
        "if_statement", "for_statement", "for_in_statement", "while_statement",
        "do_statement", "try_statement", "catch_clause", "switch_statement",
    },
    "C++": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "try_statement", "catch_clause", "switch_statement",
    },
}

# Comment node types
_COMMENT_TYPES: dict[str, set[str]] = {
    "Python":     {"comment"},
    "Java":       {"line_comment", "block_comment"},
    "JavaScript": {"comment", "html_comment"},
    "C++":        {"comment"},
}


# ─── Tree-sitter Parser Helper ────────────────────────────────────────────────

def _get_tree(source: str, language: str):
    """Parse source and return (tree, root_node) or None on failure."""
    try:
        from tree_sitter import Language, Parser
        if language == "Python":
            import tree_sitter_python as ts_lang
        elif language == "Java":
            import tree_sitter_java as ts_lang
        elif language == "JavaScript":
            import tree_sitter_javascript as ts_lang
        elif language == "C++":
            import tree_sitter_cpp as ts_lang
        else:
            return None
        lang   = Language(ts_lang.language())
        parser = Parser(lang)
        tree   = parser.parse(source.encode("utf-8"))
        return tree
    except Exception:
        return None


# ─── Raw Metrics ─────────────────────────────────────────────────────────────

def _raw_metrics(source: str, language: str, root_node) -> dict:
    """
    Compute raw line-count metrics.

    Returns: loc, sloc, lloc, comments, multi, blank, comment_ratio
    """
    lines = source.splitlines()
    loc   = len(lines)
    blank = sum(1 for l in lines if not l.strip())

    # Collect comment line numbers from AST
    comment_types  = _COMMENT_TYPES.get(language, set())
    comment_lines: set[int] = set()
    multi_lines:   set[int] = set()

    def _collect_comments(node):
        if node.type in comment_types:
            start = node.start_point[0]
            end   = node.end_point[0]
            if end > start:
                for ln in range(start, end + 1):
                    multi_lines.add(ln)
            else:
                comment_lines.add(start)
        for child in node.children:
            _collect_comments(child)

    _collect_comments(root_node)

    comments = len(comment_lines)
    multi    = len(multi_lines)

    # SLOC: non-blank, non-comment lines
    all_comment = comment_lines | multi_lines
    sloc = sum(
        1 for i, l in enumerate(lines)
        if l.strip() and i not in all_comment
    )

    # LLOC: count statement nodes
    _STMT_TYPES = {
        "Python":     {"expression_statement", "return_statement", "assignment",
                       "augmented_assignment", "raise_statement", "assert_statement",
                       "delete_statement", "pass_statement", "break_statement",
                       "continue_statement", "import_statement", "import_from_statement"},
        "Java":       {"expression_statement", "return_statement", "throw_statement",
                       "assert_statement", "break_statement", "continue_statement",
                       "local_variable_declaration"},
        "JavaScript": {"expression_statement", "return_statement", "throw_statement",
                       "break_statement", "continue_statement", "variable_declaration",
                       "lexical_declaration"},
        "C++":        {"expression_statement", "return_statement", "throw_statement",
                       "break_statement", "continue_statement", "declaration"},
    }
    stmt_types = _STMT_TYPES.get(language, set())
    lloc = [0]
    def _count_stmts(node):
        if node.type in stmt_types:
            lloc[0] += 1
        for child in node.children:
            _count_stmts(child)
    _count_stmts(root_node)

    comment_ratio = round((comments + multi) / loc, 4) if loc > 0 else 0.0

    return {
        "loc":           loc,
        "sloc":          sloc,
        "lloc":          lloc[0],
        "comments":      comments,
        "multi":         multi,
        "blank":         blank,
        "comment_ratio": comment_ratio,
    }


# ─── Cyclomatic Complexity ────────────────────────────────────────────────────

def _cyclomatic_complexity(root_node, language: str) -> int:
    """McCabe cyclomatic complexity = 1 + number of branching nodes."""
    branch_types = _BRANCH_NODES.get(language, set())
    count = [0]
    def _walk(node):
        if node.type in branch_types:
            count[0] += 1
        for child in node.children:
            _walk(child)
    _walk(root_node)
    return 1 + count[0]


# ─── Max Nesting Depth ────────────────────────────────────────────────────────

def _max_nesting_depth(root_node, language: str) -> int:
    """Maximum depth of nested control flow structures."""
    ctrl_types = _CONTROL_NODES.get(language, set())
    max_depth  = [0]
    def _walk(node, depth):
        if node.type in ctrl_types:
            depth += 1
            if depth > max_depth[0]:
                max_depth[0] = depth
        for child in node.children:
            _walk(child, depth)
    _walk(root_node, 0)
    return max_depth[0]


# ─── Halstead Metrics ─────────────────────────────────────────────────────────

def _halstead_metrics(root_node, language: str) -> dict:
    """
    Compute Halstead metrics from the AST.

    Returns:
        h1 (distinct operators), h2 (distinct operands),
        N1 (total operators), N2 (total operands),
        vocabulary, length, calculated_length,
        volume, difficulty, effort, time, bugs
    """
    op_types      = _OPERATORS.get(language, set())
    operand_types = _OPERAND_TYPES.get(language, set())

    operators: list[str] = []
    operands:  list[str] = []

    def _walk(node):
        nt   = node.type
        text = node.text.decode("utf-8", errors="replace") if node.text else nt

        if nt in op_types:
            operators.append(text if len(node.children) == 0 else nt)
        if nt in operand_types:
            operands.append(text)

        for child in node.children:
            _walk(child)

    _walk(root_node)

    h1 = len(set(operators))   # distinct operators
    h2 = len(set(operands))    # distinct operands
    N1 = len(operators)        # total operators
    N2 = len(operands)         # total operands

    vocabulary         = h1 + h2
    length             = N1 + N2
    calculated_length  = (
        (h1 * math.log2(h1) if h1 > 0 else 0) +
        (h2 * math.log2(h2) if h2 > 0 else 0)
    )
    volume     = length * math.log2(vocabulary) if vocabulary > 1 else 0.0
    difficulty = (h1 / 2) * (N2 / h2) if h2 > 0 else 0.0
    effort     = difficulty * volume
    time       = effort / 18.0
    bugs       = volume / 3000.0

    return {
        "h1":                round(h1),
        "h2":                round(h2),
        "N1":                round(N1),
        "N2":                round(N2),
        "vocabulary":        round(vocabulary),
        "halstead_length":   round(length),
        "calculated_length": round(calculated_length, 2),
        "volume":            round(volume, 2),
        "difficulty":        round(difficulty, 2),
        "effort":            round(effort, 2),
        "time":              round(time, 2),
        "bugs":              round(bugs, 4),
    }


# ─── Maintainability Index ────────────────────────────────────────────────────

def _maintainability_index(volume: float, cc: int, sloc: int,
                            comment_ratio: float) -> float:
    """
    Maintainability Index (Radon variant).

    MI = max(0, 100 * (171 - 5.2*ln(V) - 0.23*G - 16.2*ln(L)
                        + 50*sin(sqrt(2.4*C))) / 171)

    Where V=volume, G=cyclomatic, L=sloc, C=comment_ratio
    """
    if sloc <= 0 or volume <= 0:
        return 100.0
    try:
        mi = (
            171
            - 5.2  * math.log(volume)
            - 0.23 * cc
            - 16.2 * math.log(sloc)
            + 50   * math.sin(math.sqrt(2.4 * comment_ratio))
        )
        return round(max(0.0, 100.0 * mi / 171.0), 2)
    except Exception:
        return 0.0


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_metrics(source: str, language: str) -> dict:
    """
    Compute all code metrics for a source code snippet.

    Args:
        source:   Source code string
        language: One of Python, Java, JavaScript, C++

    Returns:
        dict with all metric fields, or empty dict on failure.
        All values are None if computation fails for that metric.
    """
    tree = _get_tree(source, language)
    if tree is None:
        return {}

    root = tree.root_node

    raw  = _raw_metrics(source, language, root)
    cc   = _cyclomatic_complexity(root, language)
    nest = _max_nesting_depth(root, language)
    hal  = _halstead_metrics(root, language)
    mi   = _maintainability_index(
        hal["volume"], cc, raw["sloc"], raw["comment_ratio"]
    )

    return {
        # Raw
        "loc":                   raw["loc"],
        "sloc":                  raw["sloc"],
        "lloc":                  raw["lloc"],
        "comments":              raw["comments"],
        "multi":                 raw["multi"],
        "blank":                 raw["blank"],
        "comment_ratio":         raw["comment_ratio"],
        # Cyclomatic
        "cyclomatic_complexity": cc,
        # Nesting
        "max_nesting_depth":     nest,
        # Halstead
        "h1":                    hal["h1"],
        "h2":                    hal["h2"],
        "N1":                    hal["N1"],
        "N2":                    hal["N2"],
        "vocabulary":            hal["vocabulary"],
        "halstead_length":       hal["halstead_length"],
        "calculated_length":     hal["calculated_length"],
        "volume":                hal["volume"],
        "difficulty":            hal["difficulty"],
        "effort":                hal["effort"],
        "time":                  hal["time"],
        "bugs":                  hal["bugs"],
        # Composite
        "maintainability_index": mi,
    }