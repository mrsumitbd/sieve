"""
scripts/data_collection/build_dataset.py

Merges human and AI-generated code samples across all languages and
granularities into a single unified dataset for classifier training.

Design:
  - label 0 = human-written
  - label 1 = AI-generated
  - Adds 'language' and 'granularity' columns for model conditioning
  - Balances Python to match other languages (~10K functions, ~5K classes)
  - Outputs train/val/test splits (80/10/10) stratified by
    language × granularity × label

Output files (in data/dataset/):
  train.csv, val.csv, test.csv, full.csv

Each row has columns:
  code, label, language, granularity, model, repository

Usage:
    python scripts/data_collection/build_dataset.py
"""

import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

BASE       = Path(__file__).parent.parent.parent
OUT_DIR    = BASE / "data/dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

# Target sizes after downsampling Python to match other languages
PYTHON_FUNC_TARGET  = 10_000   # human samples (AI will match)
PYTHON_CLASS_TARGET = 5_000

# New language AI models
NEW_LANG_MODELS = [
    "deepseek-v3-1",
    "llama-3-3-70b-instruct-turbo",
    "llama-4-maverick-17b-128e-instruct-fp8",
    "mistral-small-24b-instruct-2501",
    "gpt-oss-20b",
]

# Python models → file stems
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

NEW_LANG_CONFIGS = [
    {
        "language":    "JavaScript",
        "granularity": "function",
        "human_path":  "sieve_output/javascript/js_functions_human.csv",
        "human_col":   "complete_extracted_code",
        "input_col":   "func_signature",
        "repo_col":    "repository_name",
        "slug":        "javascript_function",
    },
    {
        "language":    "JavaScript",
        "granularity": "class",
        "human_path":  "sieve_output/javascript/js_classes_human.csv",
        "human_col":   "human_written_code",
        "input_col":   "class_skeleton",
        "repo_col":    "repository_name",
        "slug":        "javascript_class",
    },
    {
        "language":    "Java",
        "granularity": "function",
        "human_path":  "sieve_output/java/java_functions_human.csv",
        "human_col":   "complete_extracted_code",
        "input_col":   "func_signature",
        "repo_col":    "repository_name",
        "slug":        "java_function",
    },
    {
        "language":    "Java",
        "granularity": "class",
        "human_path":  "sieve_output/java/java_classes_human.csv",
        "human_col":   "human_written_code",
        "input_col":   "class_skeleton",
        "repo_col":    "repository_name",
        "slug":        "java_class",
    },
    {
        "language":    "C++",
        "granularity": "function",
        "human_path":  "sieve_output/cpp/cpp_functions_human.csv",
        "human_col":   "complete_extracted_code",
        "input_col":   "func_signature",
        "repo_col":    "repository_name",
        "slug":        "cpp_function",
    },
    {
        "language":    "C++",
        "granularity": "class",
        "human_path":  "sieve_output/cpp/cpp_classes_human.csv",
        "human_col":   "human_written_code",
        "input_col":   "class_skeleton",
        "repo_col":    "repository_name",
        "slug":        "cpp_class",
    },
]


# ── Helper ────────────────────────────────────────────────────────────────────

def make_record(code, label, language, granularity, model, repository):
    return {
        "code":        code,
        "label":       label,
        "language":    language,
        "granularity": granularity,
        "model":       model,        # model name or "human"
        "repository":  repository,
    }


# ── Python data ───────────────────────────────────────────────────────────────

def load_python(gran: str, target: int) -> list[dict]:
    """
    Load Python human + AI pairs, downsample human to target,
    then match AI volume proportionally across 4 models.
    """
    human_col = "complete_extracted_code" if gran == "function" else "human_written_code"
    sig_col   = "func_signature"          if gran == "function" else "class_skeleton"

    records = []

    # ── Human: load once from any file (same across models), downsample
    first_model = list(PYTHON_MODELS.keys())[0]
    func_f, cls_f = PYTHON_MODELS[first_model]
    fname = func_f if gran == "function" else cls_f
    df_h = pd.read_csv(BASE / "data/existing_python_data" / fname)
    df_h = df_h[df_h[human_col].notna()].copy()
    df_h = df_h.sample(min(target, len(df_h)), random_state=42)

    for _, row in df_h.iterrows():
        records.append(make_record(
            code=str(row[human_col]),
            label=0,
            language="Python",
            granularity=gran,
            model="human",
            repository=str(row.get("repository_name", "")),
        ))
    logger.info(f"Python {gran} human: {len(df_h)} samples")

    # ── AI: per-model, sample target // 4 rows, match on signature
    per_model = target // len(PYTHON_MODELS)
    human_sigs = set(df_h[sig_col].dropna().tolist())

    for model_name, (func_f, cls_f) in PYTHON_MODELS.items():
        fname = func_f if gran == "function" else cls_f
        df_ai = pd.read_csv(BASE / "data/existing_python_data" / fname)
        df_ai = df_ai[
            df_ai["generated_code_cleaned"].notna() &
            df_ai[sig_col].isin(human_sigs)
        ].copy()
        df_ai = df_ai.sample(min(per_model, len(df_ai)), random_state=42)

        for _, row in df_ai.iterrows():
            records.append(make_record(
                code=str(row["generated_code_cleaned"]),
                label=1,
                language="Python",
                granularity=gran,
                model=model_name,
                repository=str(row.get("repository_name", "")),
            ))
        logger.info(f"Python {gran} AI [{model_name}]: {len(df_ai)} samples")

    return records


# ── New language data ─────────────────────────────────────────────────────────

def load_new_lang(cfg: dict) -> list[dict]:
    """Load human + AI records for one language/granularity combination."""
    language    = cfg["language"]
    granularity = cfg["granularity"]
    human_col   = cfg["human_col"]
    input_col   = cfg["input_col"]
    repo_col    = cfg["repo_col"]
    slug        = cfg["slug"]

    records = []

    # ── Human
    df_h = pd.read_csv(BASE / cfg["human_path"])
    df_h = df_h[df_h[human_col].notna()].copy()

    for _, row in df_h.iterrows():
        records.append(make_record(
            code=str(row[human_col]),
            label=0,
            language=language,
            granularity=granularity,
            model="human",
            repository=str(row.get(repo_col, "")),
        ))
    logger.info(f"{language} {granularity} human: {len(df_h)} samples")

    # ── AI: all models
    human_sigs = set(df_h[input_col].dropna().tolist())

    for model in NEW_LANG_MODELS:
        ai_path = BASE / f"data/generated/{model}/{slug}.csv"
        if not ai_path.exists():
            logger.warning(f"Missing: {ai_path}")
            continue

        df_ai = pd.read_csv(ai_path)
        df_ai = df_ai[
            df_ai["generated_code_cleaned"].notna() &
            df_ai[input_col].isin(human_sigs)
        ].copy()

        for _, row in df_ai.iterrows():
            records.append(make_record(
                code=str(row["generated_code_cleaned"]),
                label=1,
                language=language,
                granularity=granularity,
                model=model,
                repository=str(row.get("repository_name", "")),
            ))
        logger.info(f"{language} {granularity} AI [{model}]: {len(df_ai)} samples")

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_records = []

    # Python
    logger.info("Loading Python data...")
    all_records.extend(load_python("function", PYTHON_FUNC_TARGET))
    all_records.extend(load_python("class",    PYTHON_CLASS_TARGET))

    # New languages
    for cfg in NEW_LANG_CONFIGS:
        logger.info(f"Loading {cfg['language']} {cfg['granularity']}...")
        all_records.extend(load_new_lang(cfg))

    df = pd.DataFrame(all_records)
    df = df.dropna(subset=["code"])
    df = df[df["code"].str.strip() != ""]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"\nFull dataset: {len(df):,} samples")
    logger.info("\nBreakdown:")
    breakdown = df.groupby(["language", "granularity", "label"]).size().reset_index(name="count")
    print(breakdown.to_string(index=False))

    # ── Save full dataset
    df.to_csv(OUT_DIR / "full.csv", index=False)
    logger.info(f"\nSaved full.csv ({len(df):,} rows)")

    # ── Stratified train/val/test split 80/10/10
    # Stratify on language × granularity × label
    df["_strat"] = df["language"] + "_" + df["granularity"] + "_" + df["label"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["_strat"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["_strat"]
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.drop(columns=["_strat"])
        split_df.to_csv(OUT_DIR / f"{split_name}.csv", index=False)
        logger.info(f"Saved {split_name}.csv ({len(split_df):,} rows)")

    # ── Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples:  {len(df):,}")
    logger.info(f"  Human (0):    {(df.label==0).sum():,}")
    logger.info(f"  AI (1):       {(df.label==1).sum():,}")
    logger.info(f"Train:          {len(train_df):,}")
    logger.info(f"Val:            {len(val_df):,}")
    logger.info(f"Test:           {len(test_df):,}")
    logger.info(f"\nOutput dir: {OUT_DIR}")

    # ── Per-language model coverage
    logger.info("\nAI model coverage:")
    model_counts = df[df.label==1].groupby(["language", "model"]).size().unstack(fill_value=0)
    print(model_counts.to_string())


if __name__ == "__main__":
    main()