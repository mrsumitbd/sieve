"""
cli.py

Headless CLI for SIEVE. Useful for scripted runs, cluster jobs,
and reproducibility (config file in, dataset out).

Usage:
    python -m sieve.cli run --config config.json
    python -m sieve.cli run --language Python --start-date 2024-01-01 --end-date 2025-01-01 --min-stars 50
"""

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv()

from sieve.config import SIEVEConfig, Language, Granularity, ExportFormat
from sieve.pipeline import run_pipeline

app = typer.Typer(
    name="sieve",
    help="SIEVE — Software Ingestion & Extraction for Verifiable Evaluation",
    add_completion=False,
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@app.command("run")
def run(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a JSON config file"
    ),
    language: Optional[str] = typer.Option(None, help="Programming language"),
    start_date: Optional[str] = typer.Option(None, help="Start date — repos created on or after (YYYY-MM-DD). Set to LLM training cutoff for contamination-free data."),
    end_date: Optional[str] = typer.Option(None, help="End date — repos created on or before (YYYY-MM-DD). Defaults to one month before today."),
    min_last_activity: Optional[str] = typer.Option(None, help="Only include repos pushed on or after this date (YYYY-MM-DD). Defaults to end_date."),
    min_stars: int = typer.Option(10, help="Minimum stars"),
    min_contributors: int = typer.Option(1, help="Minimum contributors"),
    max_repos: Optional[int] = typer.Option(None, help="Max repos to process (default: no cap)"),
    max_functions: Optional[int] = typer.Option(None, help="Max functions after deduplication — stratified by repo (default: no cap)"),
    max_classes: Optional[int] = typer.Option(None, help="Max classes after deduplication — stratified by repo (default: no cap)"),
    engineered_only: bool = typer.Option(False, help="Apply engineered project filter (Munaiah et al., 2017)"),
    deduplicate: bool = typer.Option(True, help="Enable deduplication"),
    dedup_threshold: float = typer.Option(0.8, help="Dedup similarity threshold"),
    output_dir: str = typer.Option("./sieve_output", help="Output directory"),
    export_format: str = typer.Option("jsonl", help="Export format: jsonl, parquet, or both"),
    github_token: Optional[str] = typer.Option(None, envvar="GITHUB_TOKEN", help="GitHub PAT"),
):
    """Run the SIEVE pipeline."""
    # Config file takes precedence over individual flags
    if config_file:
        with open(config_file) as f:
            raw = json.load(f)
        config = SIEVEConfig(**raw)
    else:
        if not language:
            console.print("[red]--language is required when not using a config file.[/red]")
            raise typer.Exit(1)
        if not start_date:
            console.print("[red]--start-date is required when not using a config file.[/red]")
            raise typer.Exit(1)

        try:
            from datetime import timedelta
            parsed_start    = date.fromisoformat(start_date)
            _default_end    = (date.today().replace(day=1) - timedelta(days=1)).replace(day=1)
            parsed_end      = date.fromisoformat(end_date) if end_date else _default_end
            parsed_activity = date.fromisoformat(min_last_activity) if min_last_activity else None
        except ValueError as e:
            console.print(f"[red]Invalid date format: {e}. Use YYYY-MM-DD.[/red]")
            raise typer.Exit(1)

        config = SIEVEConfig(
            language=language,
            start_date=parsed_start,
            end_date=parsed_end,
            min_last_activity=parsed_activity,
            min_stars=min_stars,
            min_contributors=min_contributors,
            max_repos=max_repos,
            max_functions=max_functions,
            max_classes=max_classes,
            granularity=["function", "class"],
            engineered_only=engineered_only,
            deduplicate=deduplicate,
            dedup_threshold=dedup_threshold,
            output_dir=output_dir,
            export_format=export_format,
            github_token=github_token,
        )

    console.print(f"\n[bold cyan]🔬 SIEVE[/bold cyan] — starting pipeline\n")
    console.print_json(config.model_dump_json(indent=2))
    console.print()

    def progress_callback(msg: str, current: int, total: int):
        if total > 0:
            console.print(f"  [{current}/{total}] {msg}")
        else:
            console.print(f"  {msg}")

    try:
        summary = run_pipeline(config, progress_callback=progress_callback)
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        raise typer.Exit(1)

    # Print summary table
    console.print("\n[bold green]✓ Pipeline complete[/bold green]\n")
    table = Table(title="SIEVE Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Repos Discovered",  str(summary["total_repos_discovered"]))
    table.add_row("Repos Processed",   str(summary["total_repos_processed"]))
    table.add_row("Repos Failed",      str(summary["total_repos_failed"]))
    table.add_row("Functions Extracted", str(summary["total_functions"]))
    table.add_row("Classes Extracted",   str(summary["total_classes"]))
    console.print(table)

    console.print("\n[bold]Output files:[/bold]")
    for label, path in summary["output_paths"].items():
        console.print(f"  {label}: {path}")

    if summary["failed_repos"]:
        console.print(f"\n[yellow]Failed repos ({len(summary['failed_repos'])}):[/yellow]")
        for r in summary["failed_repos"]:
            console.print(f"  {r}")


@app.command("validate-config")
def validate_config(config_file: Path = typer.Argument(..., help="Path to JSON config file")):
    """Validate a SIEVE config file without running the pipeline."""
    try:
        with open(config_file) as f:
            raw = json.load(f)
        config = SIEVEConfig(**raw)
        console.print("[green]✓ Config is valid[/green]")
        console.print_json(config.model_dump_json(indent=2))
    except Exception as e:
        console.print(f"[red]✗ Config invalid: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()