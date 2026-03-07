"""
core/export.py

Serializes extracted records to JSONL and/or Parquet format.
Each granularity level (function, class) gets its own file.
A metadata manifest is always written alongside the data files.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from sieve.core.extraction import FunctionRecord, ClassRecord
from sieve.core.discovery import RepoMetadata

logger = logging.getLogger(__name__)


def _record_to_dict(record: Union[FunctionRecord, ClassRecord]) -> dict:
    return asdict(record)


def _write_jsonl(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_record_to_dict(record), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records → {output_path}")


def _write_parquet(records: list, output_path: Path) -> None:
    try:
        import polars as pl
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dicts = [_record_to_dict(r) for r in records]
        # Flatten list fields to JSON strings for Parquet compatibility
        for d in dicts:
            for k, v in d.items():
                if isinstance(v, list):
                    d[k] = json.dumps(v)
        df = pl.DataFrame(dicts)
        df.write_parquet(str(output_path))
        logger.info(f"Wrote {len(records)} records → {output_path}")
    except ImportError:
        logger.error("polars not installed. Cannot write Parquet. Run: pip install polars")
        raise


def write_manifest(
    config: dict,
    repo_metadata_list: list[dict],
    function_count: int,
    class_count: int,
    output_dir: Path,
) -> None:
    """Write a JSON manifest capturing config, repo list, and summary stats."""
    manifest = {
        "sieve_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": {
            "total_repos": len(repo_metadata_list),
            "total_functions": function_count,
            "total_classes": class_count,
        },
        "repos": repo_metadata_list,
    }
    def _default(obj):
        """JSON serializer for types not handled by default."""
        if hasattr(obj, "isoformat"):   # date, datetime
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_default)
    logger.info(f"Manifest written → {manifest_path}")


def export_dataset(
    functions: list[FunctionRecord],
    classes: list[ClassRecord],
    output_dir: str,
    export_format: str,  # "jsonl", "parquet", "both"
    config: dict,
    repo_metadata_list: list[dict],
) -> dict:
    """
    Export all extracted records to the output directory.

    Returns:
        dict with output file paths and record counts
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    if export_format in ("jsonl", "both"):
        if functions:
            fp = out / "functions.jsonl"
            _write_jsonl(functions, fp)
            output_paths["functions_jsonl"] = str(fp)
        if classes:
            cp = out / "classes.jsonl"
            _write_jsonl(classes, cp)
            output_paths["classes_jsonl"] = str(cp)

    if export_format in ("parquet", "both"):
        if functions:
            fp = out / "functions.parquet"
            _write_parquet(functions, fp)
            output_paths["functions_parquet"] = str(fp)
        if classes:
            cp = out / "classes.parquet"
            _write_parquet(classes, cp)
            output_paths["classes_parquet"] = str(cp)

    write_manifest(
        config=config,
        repo_metadata_list=repo_metadata_list,
        function_count=len(functions),
        class_count=len(classes),
        output_dir=out,
    )
    output_paths["manifest"] = str(out / "manifest.json")

    return output_paths