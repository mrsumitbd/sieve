"""
scripts/data_collection/clean_generated_code.py

Post-processes raw LLM output to extract the largest contiguous parseable
code block, stripping markdown fences, prose explanations, and anything
that would prevent the snippet from being parsed by an AST tool.

Methodology mirrors the original Python-only approach (ast.parse) but
extends it to Java, JavaScript, and C++ via tree-sitter.

Usage:
    python clean_generated_code.py \\
        --input   data/generated/deepseek_java_func.csv \\
        --language Java \\
        --granularity function

    python clean_generated_code.py \\
        --input   data/generated/qwen_js_class.csv \\
        --language JavaScript \\
        --granularity class

Adds a 'generated_code_cleaned' column in-place and overwrites the input CSV.
Rows where no parseable code was found get None in that column.
"""

import argparse
import ast
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Markdown fence stripper ─────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences (```python ... ``` etc.) from LLM output.
    Returns the inner content if fences are found, otherwise returns text unchanged.
    """
    # Match ```<lang>\n...\n``` or just ```\n...\n```
    fence_pattern = re.compile(
        r"```(?:[a-zA-Z+#]*)\n(.*?)```", re.DOTALL
    )
    matches = fence_pattern.findall(text)
    if matches:
        # Return the largest match (most content)
        return max(matches, key=len).strip()
    return text.strip()


# ─── Language-specific cleaners ──────────────────────────────────────────────

def _clean_python(raw: str, granularity: str) -> str | None:
    """
    Extract the first valid top-level function or class block using ast.parse.
    Matches the original methodology exactly.
    """
    text = _strip_markdown_fences(raw)

    # Try to parse the whole thing first
    try:
        tree = ast.parse(text)
        target_types = (ast.FunctionDef, ast.AsyncFunctionDef) if granularity == "function" else (ast.ClassDef,)
        for node in tree.body:
            if isinstance(node, target_types):
                lines = text.splitlines()
                snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                return snippet.strip()
    except SyntaxError:
        pass

    # Fall back: try progressively shorter suffixes
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            tree = ast.parse(candidate)
            target_types = (ast.FunctionDef, ast.AsyncFunctionDef) if granularity == "function" else (ast.ClassDef,)
            for node in tree.body:
                if isinstance(node, target_types):
                    snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                    return snippet.strip()
        except SyntaxError:
            continue

    return None


def _clean_treesitter(raw: str, language: str, granularity: str) -> str | None:
    """
    Extract the first valid function/class node using tree-sitter.
    Used for Java, JavaScript, and C++.

    Strategy:
    1. Strip markdown fences
    2. Try to parse the full text
    3. If no clean node found, progressively strip leading lines
       (handles LLM responses with prose before the code)
    """
    try:
        from tree_sitter import Language as TSLanguage, Parser
    except ImportError:
        logger.error("tree-sitter not installed. Run: pip install tree-sitter")
        return None

    text = _strip_markdown_fences(raw)

    # Load parser and target node types
    try:
        if language == "Java":
            import tree_sitter_java as tsjava
            lang = TSLanguage(tsjava.language())
            target_types = (
                {"method_declaration", "constructor_declaration"}
                if granularity == "function"
                else {"class_declaration"}
            )
        elif language == "JavaScript":
            import tree_sitter_javascript as tsjavascript
            lang = TSLanguage(tsjavascript.language())
            target_types = (
                {"function_declaration", "method_definition", "lexical_declaration"}
                if granularity == "function"
                else {"class_declaration"}
            )
        elif language == "C++":
            import tree_sitter_cpp as tscpp
            lang = TSLanguage(tscpp.language())
            target_types = (
                {"function_definition"}
                if granularity == "function"
                else {"class_specifier"}
            )
        else:
            logger.error(f"Unsupported language: {language}")
            return None
    except ImportError as e:
        logger.error(f"tree-sitter grammar not installed: {e}")
        return None

    parser = Parser(lang)

    def _find_target(source_bytes: bytes) -> str | None:
        """Parse source_bytes and return the first clean target node."""
        tree = parser.parse(source_bytes)

        def walk(node, depth=0):
            if depth > 4:
                return None
            if node.type in target_types:
                has_error = any(c.type == "ERROR" for c in node.children)
                if not has_error:
                    return source_bytes[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="replace"
                    ).strip()
            for child in node.children:
                result = walk(child, depth + 1)
                if result:
                    return result
            return None

        return walk(tree.root_node)

    # Try full text first
    result = _find_target(text.encode("utf-8", errors="replace"))
    if result:
        return result

    # Progressive leading-line strip — handles prose before code blocks
    lines = text.splitlines()
    for start in range(1, min(len(lines), 20)):
        candidate = "\n".join(lines[start:])
        if not candidate.strip():
            continue
        result = _find_target(candidate.encode("utf-8", errors="replace"))
        if result:
            return result

    return None


# ─── Dispatcher ──────────────────────────────────────────────────────────────

def clean_snippet(raw: str | None, language: str, granularity: str) -> str | None:
    """
    Clean a single raw LLM output string.
    Returns the cleaned code string, or None if no parseable block was found.
    """
    if not raw or not str(raw).strip():
        return None

    raw = str(raw)

    if language == "Python":
        return _clean_python(raw, granularity)
    else:
        return _clean_treesitter(raw, language, granularity)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean raw LLM-generated code and add generated_code_cleaned column"
    )
    parser.add_argument("--input",       required=True,
                        help="CSV file with a 'generated_code' column (modified in place)")
    parser.add_argument("--language",    required=True,
                        choices=["Python", "Java", "JavaScript", "C++"])
    parser.add_argument("--granularity", required=True,
                        choices=["function", "class"])
    parser.add_argument("--input-col",   default="generated_code",
                        help="Column name containing raw LLM output (default: generated_code)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    if args.input_col not in df.columns:
        logger.error(f"Column '{args.input_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    cleaned = []
    n_success = 0
    for i, raw in enumerate(df[args.input_col]):
        result = clean_snippet(raw, args.language, args.granularity)
        cleaned.append(result)
        if result is not None:
            n_success += 1
        if (i + 1) % 500 == 0:
            logger.info(f"Cleaned {i + 1}/{len(df)} ({n_success} parseable so far)")

    df["generated_code_cleaned"] = cleaned
    df.to_csv(args.input, index=False)

    n_failed = len(df) - n_success
    logger.info(
        f"Done. {n_success}/{len(df)} snippets parseable "
        f"({n_failed} failed → None). Saved to {args.input}"
    )


if __name__ == "__main__":
    main()