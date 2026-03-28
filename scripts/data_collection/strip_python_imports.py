"""
scripts/data_collection/strip_python_imports.py

Strips file-level imports from Python human and AI code columns,
leaving only the first function or class definition.

This ensures Python samples are consistent with Java/JS/C++ samples,
which contain only the function/class body without file-level imports.

Usage:
    python scripts/data_collection/strip_python_imports.py \
        --input  data/existing_python_data/claude-3-haiku_func_level_filtered.csv \
        --granularity function

Modifies in place:
    - human code column (complete_extracted_code or human_written_code)
    - generated_code_cleaned column
"""

import ast
import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_first_node(source: str, granularity: str) -> str | None:
    """
    Parse Python source and return the source text of the first
    function or class definition, stripping everything else
    (imports, module-level statements, etc.).

    Returns None if no parseable function/class is found.
    """
    if not source or not str(source).strip():
        return None

    source = str(source)

    target_types = (
        (ast.FunctionDef, ast.AsyncFunctionDef)
        if granularity == "function"
        else (ast.ClassDef,)
    )

    # Try parsing the full source first
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, target_types):
                lines = source.splitlines()
                snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                return snippet.strip()
    except SyntaxError:
        pass

    # Progressive strip — try removing leading lines until it parses
    lines = source.splitlines()
    for start in range(1, min(len(lines), 50)):
        candidate = "\n".join(lines[start:])
        try:
            tree = ast.parse(candidate)
            for node in tree.body:
                if isinstance(node, target_types):
                    snippet_lines = candidate.splitlines()
                    snippet = "\n".join(snippet_lines[node.lineno - 1 : node.end_lineno])
                    return snippet.strip()
        except SyntaxError:
            continue

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Strip file-level imports from Python code columns"
    )
    parser.add_argument("--input",       required=True, help="CSV file to clean (modified in place)")
    parser.add_argument("--granularity", required=True, choices=["function", "class"])
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")

    human_col = (
        "complete_extracted_code" if args.granularity == "function"
        else "human_written_code"
    )

    # ── Strip human code ──────────────────────────────────────────────────────
    if human_col in df.columns:
        before = df[human_col].notna().sum()
        df[human_col] = df[human_col].apply(
            lambda x: extract_first_node(x, args.granularity)
            if pd.notna(x) else None
        )
        after = df[human_col].notna().sum()
        logger.info(f"Human col ({human_col}): {before} → {after} (lost {before - after})")
    else:
        logger.warning(f"Column '{human_col}' not found — skipping human strip")

    # ── Strip AI generated code ───────────────────────────────────────────────
    if "generated_code_cleaned" in df.columns:
        before = df["generated_code_cleaned"].notna().sum()
        df["generated_code_cleaned"] = df["generated_code_cleaned"].apply(
            lambda x: extract_first_node(x, args.granularity)
            if pd.notna(x) else None
        )
        after = df["generated_code_cleaned"].notna().sum()
        logger.info(f"AI col (generated_code_cleaned): {before} → {after} (lost {before - after})")
    else:
        logger.warning("Column 'generated_code_cleaned' not found — skipping AI strip")

    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned file to {path}")


if __name__ == "__main__":
    main()