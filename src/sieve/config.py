from pydantic import BaseModel, Field, field_validator
from typing import Optional
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
    FILE = "file"


class ExportFormat(str, Enum):
    JSONL = "jsonl"
    PARQUET = "parquet"
    BOTH = "both"


class SIEVEConfig(BaseModel):
    # --- Core filters ---
    language: Language = Field(..., description="Target programming language")
    start_date: date = Field(..., description="Only include repos created on or after this date (contamination cutoff)")
    end_date: date = Field(..., description="Only include repos created on or before this date (default: one month before today)")
    min_last_activity: Optional[date] = Field(default=None, description="Only include repos with last push on or after this date (default: same as end_date)")

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("end_date must be on or after start_date")
        return v

    @field_validator("min_last_activity")
    @classmethod
    def activity_after_end(cls, v, info):
        if v is not None and "end_date" in info.data and v < info.data["end_date"]:
            raise ValueError("min_last_activity cannot be before end_date")
        return v

    # --- Repo quality filters ---
    min_stars: int = Field(default=10, ge=0, description="Minimum number of GitHub stars")
    min_contributors: int = Field(default=1, ge=1, description="Minimum number of unique contributors")
    max_repos: Optional[int] = Field(default=None, description="Cap on number of repos to process. None = no cap")
    max_functions: Optional[int] = Field(default=None, ge=1, description="Cap on total functions after deduplication. None = no cap")
    max_classes: Optional[int] = Field(default=None, ge=1, description="Cap on total classes after deduplication. None = no cap")

    # --- Content filters ---
    granularity: list[Granularity] = Field(
        default=[Granularity.FUNCTION, Granularity.CLASS],
        description="Extraction granularity levels"
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
        description="Annotate each record with P(LLM-generated). Requires classifier to be built first."
    )
    export_ast: bool = Field(
        default=False,
        description="Include full AST JSON in each exported record (increases file size significantly)"
    )

    # --- Deduplication ---
    deduplicate: bool = Field(default=False, description="Apply MinHash near-duplicate removal")
    dedup_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Jaccard similarity threshold for dedup")

    # --- Output ---
    output_dir: str = Field(default="./sieve_output", description="Local path to save extracted dataset")
    export_format: ExportFormat = Field(default=ExportFormat.JSONL)

    # --- GitHub ---
    github_token: Optional[str] = Field(default=None, description="GitHub PAT for higher rate limits")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token for downloading model weights")

    @field_validator("granularity")
    @classmethod
    def at_least_one_granularity(cls, v):
        if len(v) == 0:
            raise ValueError("At least one granularity level must be selected")
        return v

    class Config:
        use_enum_values = True