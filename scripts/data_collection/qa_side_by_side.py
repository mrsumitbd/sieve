"""
scripts/data_collection/qa_side_by_side.py

Side-by-side QA of human vs AI-generated code for all languages,
granularities, and LLMs. Each pair shows the SAME function/class:
human original vs AI implementation given the same signature/skeleton.

Usage:
    python scripts/data_collection/qa_side_by_side.py

Output:
    data/qa/side_by_side_qa.txt
"""

import random
import pandas as pd
from pathlib import Path
from datetime import datetime

random.seed(42)

BASE = Path(__file__).parent.parent.parent  # repo root

# ── Dataset definitions ───────────────────────────────────────────────────────

NEW_LANG_CONFIGS = [
    {
        "lang":       "javascript",
        "gran":       "function",
        "human_path": "sieve_output/javascript/js_functions_human.csv",
        "human_col":  "complete_extracted_code",
        "input_col":  "func_signature",
        "repo_col":   "repository_name",
    },
    {
        "lang":       "javascript",
        "gran":       "class",
        "human_path": "sieve_output/javascript/js_classes_human.csv",
        "human_col":  "human_written_code",
        "input_col":  "class_skeleton",
        "repo_col":   "repository_name",
    },
    {
        "lang":       "java",
        "gran":       "function",
        "human_path": "sieve_output/java/java_functions_human.csv",
        "human_col":  "complete_extracted_code",
        "input_col":  "func_signature",
        "repo_col":   "repository_name",
    },
    {
        "lang":       "java",
        "gran":       "class",
        "human_path": "sieve_output/java/java_classes_human.csv",
        "human_col":  "human_written_code",
        "input_col":  "class_skeleton",
        "repo_col":   "repository_name",
    },
    {
        "lang":       "cpp",
        "gran":       "function",
        "human_path": "sieve_output/cpp/cpp_functions_human.csv",
        "human_col":  "complete_extracted_code",
        "input_col":  "func_signature",
        "repo_col":   "repository_name",
    },
    {
        "lang":       "cpp",
        "gran":       "class",
        "human_path": "sieve_output/cpp/cpp_classes_human.csv",
        "human_col":  "human_written_code",
        "input_col":  "class_skeleton",
        "repo_col":   "repository_name",
    },
]

NEW_LANG_MODELS = [
    "deepseek-v3-1",
    "llama-3-3-70b-instruct-turbo",
    "llama-4-maverick-17b-128e-instruct-fp8",
    "mistral-small-24b-instruct-2501",
    "gpt-oss-20b",
]

PYTHON_MODELS = {
    "claude-3-haiku":   ("claude-3-haiku_func_level_filtered.csv",
                         "claude-3-haiku_class_level_filtered.csv"),
    "claude-4-5-haiku": ("claude-4_5-haiku_func_level_filtered.csv",
                         "claude-4_5-haiku_class_level_filtered.csv"),
    "gpt-3-5":          ("gpt-3_5_func_level_filtered.csv",
                         "gpt-3_5_class_level_filtered.csv"),
    "gpt-oss":          ("gpt-oss_func_level_filtered.csv",
                         "gpt-oss_class_level_filtered.csv"),
}

PYTHON_DATA_DIR = BASE / "data/existing_python_data"
OUTPUT_DIR      = BASE / "data/qa"
OUTPUT_FILE     = OUTPUT_DIR / "side_by_side_qa.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Formatting ────────────────────────────────────────────────────────────────

SEP_MAJOR = "=" * 80
SEP_MINOR = "-" * 80

def section(title):
    return f"\n{SEP_MAJOR}\n{title}\n{SEP_MAJOR}\n"

def subsection(title):
    return f"\n{SEP_MINOR}\n{title}\n{SEP_MINOR}"

def format_pair(input_label, input_sig, human_label, human_code, ai_label, ai_code):
    return (
        f"\nINPUT ({input_label}):\n{input_sig}\n"
        f"\n[HUMAN] {human_label}\n{human_code}\n"
        f"\n[AI]    {ai_label}\n{ai_code}\n"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

lines = []
lines.append("SIEVE - Side-by-Side QA Report (Matched Pairs)")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("Each pair: same signature/skeleton -> human original vs AI implementation")
lines.append("Random seed: 42  |  1 matched pair per (language x granularity x model)")

# ── New languages ─────────────────────────────────────────────────────────────

for cfg in NEW_LANG_CONFIGS:
    lang       = cfg["lang"]
    gran       = cfg["gran"]
    input_col  = cfg["input_col"]
    human_col  = cfg["human_col"]
    repo_col   = cfg["repo_col"]

    lines.append(section(f"{lang.upper()} - {gran.upper()}"))

    df_human = pd.read_csv(BASE / cfg["human_path"])
    df_human = df_human[df_human[human_col].notna() & df_human[input_col].notna()]
    human_index = df_human.set_index(input_col)

    for model in NEW_LANG_MODELS:
        ai_path = BASE / f"data/generated/{model}/{lang}_{gran}.csv"
        if not ai_path.exists():
            lines.append(f"\n[MISSING] {model}: {ai_path}")
            continue

        df_ai = pd.read_csv(ai_path)
        df_ai = df_ai[
            df_ai["generated_code_cleaned"].notna() &
            df_ai[input_col].notna()
        ]

        df_ai_matched = df_ai[df_ai[input_col].isin(human_index.index)]

        if df_ai_matched.empty:
            lines.append(f"\n[NO MATCH] {model}: no overlapping signatures found")
            continue

        ai_row     = df_ai_matched.sample(1, random_state=42).iloc[0]
        sig        = ai_row[input_col]
        ai_code    = str(ai_row["generated_code_cleaned"])
        human_rows = human_index.loc[[sig]]
        human_row  = human_rows.iloc[0]
        human_code = str(human_row[human_col])
        repo       = str(human_row[repo_col])

        lines.append(subsection(f"Model: {model}"))
        lines.append(format_pair(
            input_label="signature/skeleton",
            input_sig=sig,
            human_label=repo,
            human_code=human_code,
            ai_label=model,
            ai_code=ai_code,
        ))

# ── Python ────────────────────────────────────────────────────────────────────

for gran in ("function", "class"):
    lines.append(section(f"PYTHON - {gran.upper()}"))

    human_col = "complete_extracted_code" if gran == "function" else "human_written_code"
    sig_col   = "func_signature"          if gran == "function" else "class_skeleton"
    repo_col  = "repository_name"

    for model_name, (func_file, cls_file) in PYTHON_MODELS.items():
        fname = func_file if gran == "function" else cls_file
        fpath = PYTHON_DATA_DIR / fname

        if not fpath.exists():
            lines.append(f"\n[MISSING] {model_name}: {fpath}")
            continue

        df = pd.read_csv(fpath)
        df = df[
            df[human_col].notna() &
            df["generated_code_cleaned"].notna() &
            df[sig_col].notna()
        ]
        if df.empty:
            lines.append(f"\n[EMPTY] {model_name}: no usable rows")
            continue

        row        = df.sample(1, random_state=42).iloc[0]
        sig        = str(row[sig_col])
        human_code = str(row[human_col])
        ai_code    = str(row["generated_code_cleaned"])
        repo       = str(row[repo_col])

        lines.append(subsection(f"Model: {model_name}"))
        lines.append(format_pair(
            input_label="signature/skeleton",
            input_sig=sig,
            human_label=repo,
            human_code=human_code,
            ai_label=model_name,
            ai_code=ai_code,
        ))

# ── Write ─────────────────────────────────────────────────────────────────────

report = "\n".join(lines)
OUTPUT_FILE.write_text(report, encoding="utf-8")
print(f"QA report saved to: {OUTPUT_FILE}")
print(f"Total pairs written: {report.count('[HUMAN]')}")