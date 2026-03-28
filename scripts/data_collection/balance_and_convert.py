"""
scripts/data_collection/balance_and_convert.py

Balances and converts SIEVE JSONL output to CSV files ready for AI generation.

Steps performed:
  1. For each language (JavaScript, Java, C++):
     a. Check distribution and apply per-repo cap (300 functions, 200 classes)
     b. Apply total cap (10K functions, 5K classes)
     c. Filter garbage entries (unknown names, main(), short signatures)
     d. Convert to CSV with column names matching the generation script schema

  2. Run QA checks on all output CSVs

Output CSVs (in sieve_output/<lang>/):
  js_functions_human.csv   js_classes_human.csv
  java_functions_human.csv java_classes_human.csv
  cpp_functions_human.csv  cpp_classes_human.csv

Usage:
    python scripts/data_collection/balance_and_convert.py
"""

import json
import logging
import random
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

random.seed(42)

BASE = Path(__file__).parent.parent.parent

# ── Config ────────────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "language":          "javascript",
        "func_jsonl":        "sieve_output/javascript/functions_balanced.jsonl",
        "cls_jsonl":         "sieve_output/javascript/classes.jsonl",
        "func_out":          "sieve_output/javascript/js_functions_human.csv",
        "cls_out":           "sieve_output/javascript/js_classes_human.csv",
        "func_per_repo_cap": 300,
        "func_total_cap":    10_000,
        "cls_per_repo_cap":  200,
        "cls_total_cap":     10_000,   # JS has fewer classes — keep all
        "bad_repos": {
            "MostlyAdequate/mostly-adequate-guide",
            "ryanmcdermott/clean-code-javascript",
            "goldbergyoni/javascript-testing-best-practices",
            "verekia/js-stack-from-scratch",
            "adam-golab/react-developer-roadmap",
            "jondot/awesome-react-native",
            "goabstract/Awesome-Design-Tools",
            "elsewhencode/project-guidelines",
            "viatsko/awesome-vscode",
            "ascoders/weekly",
            "ryanhanwu/How-To-Ask-Questions-The-Smart-Way",
            "hackerkid/Mind-Expanding-Books",
            "apsdehal/awesome-ctf",
            "jojoldu/junior-recruit-scheduler",
            "Hackl0us/SS-Rule-Snippet",
            "apachecn/apachecn-algo-zh",
            "kautukkundan/Awesome-Profile-README-templates",
            "chaozh/awesome-blockchain-cn",
        },
    },
    {
        "language":          "java",
        "func_jsonl":        "sieve_output/java/functions_balanced.jsonl",
        "cls_jsonl":         "sieve_output/java/classes_balanced.jsonl",
        "func_out":          "sieve_output/java/java_functions_human.csv",
        "cls_out":           "sieve_output/java/java_classes_human.csv",
        "func_per_repo_cap": 300,
        "func_total_cap":    10_000,
        "cls_per_repo_cap":  200,
        "cls_total_cap":     5_000,
        "bad_repos":         set(),
    },
    {
        "language":          "cpp",
        "func_jsonl":        "sieve_output/cpp/functions_balanced.jsonl",
        "cls_jsonl":         "sieve_output/cpp/classes_balanced.jsonl",
        "func_out":          "sieve_output/cpp/cpp_functions_human.csv",
        "cls_out":           "sieve_output/cpp/cpp_classes_human.csv",
        "func_per_repo_cap": 300,
        "func_total_cap":    10_000,
        "cls_per_repo_cap":  200,
        "cls_total_cap":     5_000,
        "bad_repos":         set(),
    },
]

MIN_SIG_LEN  = 10    # drop signatures shorter than this
BAD_FUNC_NAMES = {"unknown", "main"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_sample(records: list[dict], per_repo_cap: int,
                      total_cap: int, key: str = "repo") -> list[dict]:
    """
    Apply per-repo cap then total cap with proportional allocation.
    Same logic as SIEVE's _stratified_sample.
    """
    by_repo = defaultdict(list)
    for r in records:
        by_repo[r[key]].append(r)

    # Per-repo cap
    capped = []
    for repo, items in by_repo.items():
        capped.extend(random.sample(items, min(len(items), per_repo_cap)))

    logger.info(f"  After per-repo cap ({per_repo_cap}): {len(capped)}")

    # Total cap
    if len(capped) > total_cap:
        capped = random.sample(capped, total_cap)
    logger.info(f"  After total cap ({total_cap}): {len(capped)}")

    return capped


def show_top_repos(records: list[dict], n: int = 10):
    counts = Counter(r["repo"] for r in records)
    logger.info(f"  Top {n} repos:")
    for repo, cnt in counts.most_common(n):
        logger.info(f"    {repo:<55} {cnt:>6}")


def func_to_row(r: dict) -> dict:
    return {
        "repository_name":         r["repo"],
        "func_path_in_repository": r["file_path"],
        "func_name":               r["func_name"],
        "complete_extracted_code": r["source_code"],
        "func_signature":          r["signature"],
    }


def cls_to_row(r: dict) -> dict:
    return {
        "repository_name":   r["repo"],
        "file_path":         r["file_path"],
        "class_name":        r["class_name"],
        "human_written_code": r["source_code"],
        "class_skeleton":    r["skeleton"],
    }


def qa_check(path: Path, col: str) -> bool:
    df = pd.read_csv(path)
    nulls = df[col].isna().sum()
    empty = (df[col].str.strip() == "").sum()
    short = (df[col].str.len() < MIN_SIG_LEN).sum()
    ok    = nulls + empty + short == 0
    status = "OK" if ok else "ISSUES FOUND"
    logger.info(
        f"  QA [{status}] {path.name}: "
        f"rows={len(df)} nulls={nulls} empty={empty} short={short}"
    )
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def process_language(cfg: dict):
    lang = cfg["language"].upper()
    logger.info(f"\n{'='*60}")
    logger.info(f"{lang}")
    logger.info(f"{'='*60}")

    # ── Functions ─────────────────────────────────────────────────────────────
    logger.info(f"\n[{lang}] Functions")
    func_path = BASE / cfg["func_jsonl"]
    if not func_path.exists():
        logger.error(f"  Missing: {func_path}")
        return

    funcs = load_jsonl(func_path)
    logger.info(f"  Loaded: {len(funcs)}")
    show_top_repos(funcs)

    # Filter bad repos
    if cfg["bad_repos"]:
        before = len(funcs)
        funcs = [r for r in funcs if r["repo"] not in cfg["bad_repos"]]
        logger.info(f"  After bad repo filter: {len(funcs)} (dropped {before - len(funcs)})")

    # Filter garbage function names and short signatures
    before = len(funcs)
    funcs = [
        r for r in funcs
        if r.get("func_name", "").strip() not in BAD_FUNC_NAMES
        and len(r.get("signature", "")) >= MIN_SIG_LEN
    ]
    logger.info(f"  After QA filter: {len(funcs)} (dropped {before - len(funcs)})")

    # Balance
    funcs = stratified_sample(
        funcs,
        cfg["func_per_repo_cap"],
        cfg["func_total_cap"],
    )

    # Convert and save
    out_path = BASE / cfg["func_out"]
    pd.DataFrame([func_to_row(r) for r in funcs]).to_csv(out_path, index=False)
    logger.info(f"  Saved: {out_path} ({len(funcs)} rows)")

    # ── Classes ───────────────────────────────────────────────────────────────
    logger.info(f"\n[{lang}] Classes")
    cls_path = BASE / cfg["cls_jsonl"]
    if not cls_path.exists():
        logger.error(f"  Missing: {cls_path}")
        return

    classes = load_jsonl(cls_path)
    logger.info(f"  Loaded: {len(classes)}")
    show_top_repos(classes)

    # Filter bad repos
    if cfg["bad_repos"]:
        before = len(classes)
        classes = [r for r in classes if r["repo"] not in cfg["bad_repos"]]
        logger.info(f"  After bad repo filter: {len(classes)} (dropped {before - len(classes)})")

    # Balance
    classes = stratified_sample(
        classes,
        cfg["cls_per_repo_cap"],
        cfg["cls_total_cap"],
    )

    # Convert and save
    out_path = BASE / cfg["cls_out"]
    pd.DataFrame([cls_to_row(r) for r in classes]).to_csv(out_path, index=False)
    logger.info(f"  Saved: {out_path} ({len(classes)} rows)")


def main():
    logger.info("SIEVE — Balance and Convert")
    logger.info("Converts JSONL corpus to generation-ready CSVs")

    for cfg in CONFIGS:
        process_language(cfg)

    # ── QA all outputs ────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("QA CHECKS")
    logger.info(f"{'='*60}")

    checks = [
        ("sieve_output/javascript/js_functions_human.csv",  "func_signature"),
        ("sieve_output/javascript/js_classes_human.csv",    "class_skeleton"),
        ("sieve_output/java/java_functions_human.csv",       "func_signature"),
        ("sieve_output/java/java_classes_human.csv",         "class_skeleton"),
        ("sieve_output/cpp/cpp_functions_human.csv",         "func_signature"),
        ("sieve_output/cpp/cpp_classes_human.csv",           "class_skeleton"),
    ]

    all_ok = True
    for rel_path, col in checks:
        path = BASE / rel_path
        if not path.exists():
            logger.warning(f"  MISSING: {path}")
            all_ok = False
            continue
        ok = qa_check(path, col)
        if not ok:
            all_ok = False

    logger.info("")
    if all_ok:
        logger.info("All QA checks passed. Ready for AI generation.")
        logger.info("Next step: bash generate_<model>.sh (run all 5 in parallel)")
    else:
        logger.warning("Some QA checks failed — review before proceeding.")


if __name__ == "__main__":
    main()