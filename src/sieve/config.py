from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from typing import Optional, Union
from datetime import date
from enum import Enum


class Language(str, Enum):
    PYTHON = "Python"
    JAVA = "Java"
    JAVASCRIPT = "JavaScript"
    CPP = "C++"


class Granularity(str, Enum):
    FUNCTION = "function"
    CLASS = "class"


class ExportFormat(str, Enum):
    JSONL = "jsonl"
    PARQUET = "parquet"
    BOTH = "both"


def reveal_secret(value: Union[str, SecretStr, None]) -> Optional[str]:
    """Return the raw text of a token for the call that needs it; None if unset or empty."""
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    return value or None


class SIEVEConfig(BaseModel):
    # Unknown keys are rejected so a typo in a config file fails loudly instead of being ignored.
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # --- Core filters ---
    language: Language = Field(..., description="Target programming language")
    start_date: date = Field(..., description="Only include repos created on or after this date — e.g. an LLM's release date for contamination-aware evaluation, or any reference date to bound a corpus to a development era (bounds repo age, not the origin of each snippet)")
    end_date: date = Field(..., description="Only include repos created on or before this date. Required here (no default in the config or API); the CLI and web UI default it to 30 days before today")

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("end_date must be on or after start_date")
        return v

    # --- Repo quality filters ---
    min_stars: int = Field(default=10, ge=0, description="Minimum number of GitHub stars")
    min_contributors: int = Field(default=1, ge=1, description="Minimum number of unique contributors")
    max_repos: Optional[int] = Field(default=None, description="Cap on number of repos to process. None (default) = no cap; otherwise must be >= 1")
    max_functions: Optional[int] = Field(default=None, description="Cap on total functions after deduplication, stratified by repo. None (default) = no cap; otherwise must be >= 1. To skip functions, drop 'function' from granularity")
    max_classes: Optional[int] = Field(default=None, description="Cap on total classes after deduplication, stratified by repo. None (default) = no cap; otherwise must be >= 1. To skip classes, drop 'class' from granularity")

    @field_validator("max_repos", "max_functions", "max_classes")
    @classmethod
    def cap_is_none_or_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError(f"must be null/None (no cap) or an integer >= 1, got {v}")
        return v

    # --- Content filters ---
    granularity: list[Granularity] = Field(
        default=[Granularity.FUNCTION, Granularity.CLASS],
        description="Extraction granularity levels: 'function', 'class' or both (default: both). At least one is required"
    )
    engineered_only: bool = Field(
        default=False,
        description=(
            "Apply Xiao et al. / Munaiah et al. engineered project filter. "
            "Runs a two-pass discovery: collects all candidates first, "
            "computes population-level Q1 thresholds, then filters."
        )
    )

    # --- LLM-generated code ---
    annotate_llm_score: bool = Field(
        default=False,
        description="Annotate each record with P(LLM-generated). Default: False. Classifier weights are downloaded from the Hugging Face Hub on first use (hf_token, or the HF_TOKEN env var in the CLI, is optional)"
    )
    export_ast: bool = Field(
        default=False,
        description="Include full AST JSON in each exported record. Default: False (increases file size significantly)"
    )

    # --- Deduplication ---
    deduplicate: bool = Field(default=False, description="Apply MinHash near-duplicate removal. Default: False")
    dedup_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Jaccard similarity threshold for dedup (0.0-1.0, default 0.8). Only used when deduplicate is True")

    # --- Output ---
    output_dir: str = Field(default="./sieve_output", description="Local path to save extracted dataset")
    export_format: ExportFormat = Field(default=ExportFormat.JSONL, description="Output format: jsonl (default), parquet or both")

    # --- GitHub ---
    # Secrets: excluded from model_dump / model_dump_json (and so the CLI banner and manifest.json),
    # masked in repr/str. Accept plain strings as input.
    github_token: Optional[SecretStr] = Field(default=None, exclude=True, description="GitHub PAT for higher rate limits. The CLI falls back to the GITHUB_TOKEN env var when unset")
    hf_token: Optional[SecretStr] = Field(default=None, exclude=True, description="HuggingFace token for downloading model weights. The CLI falls back to the HF_TOKEN env var when unset")

    @field_validator("granularity")
    @classmethod
    def at_least_one_granularity(cls, v):
        if len(v) == 0:
            raise ValueError("At least one granularity level must be selected")
        return v
