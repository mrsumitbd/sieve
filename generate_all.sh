#!/bin/bash
# =============================================================================
# generate_all.sh
#
# Runs all 24 LLM code generation jobs (4 models × 3 languages × 2 granularities)
# sequentially via Together.ai batch API.
#
# Usage:
#   chmod +x generate_all.sh
#   ./generate_all.sh
#
# Output files land in: data/generated/<model>/<lang>_<gran>.csv
# Logs land in:         logs/generate_all_<timestamp>.log
#
# Requirements:
#   pip install -r scripts/data_collection/requirements.txt
#   TOGETHER_API_KEY set in .env at project root
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE="$SCRIPT_DIR/scripts/data_collection/generate_llm_code.py"
SIEVE_OUT="$SCRIPT_DIR/sieve_output"
DATA_OUT="$SCRIPT_DIR/data/generated"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/generate_all_${TIMESTAMP}.log"

mkdir -p "$DATA_OUT" "$LOG_DIR"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS=(
    "deepseek-ai/DeepSeek-V3.1"
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "mistralai/Mistral-Small-24B-Instruct-2501"
)

# Convert model ID to a filesystem-safe slug
model_slug() {
    echo "$1" \
        | sed 's|.*/||' \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's/[^a-z0-9]/-/g' \
        | sed 's/-\+/-/g' \
        | sed 's/-$//'
}

# ── Input files and targets ───────────────────────────────────────────────────
# Format: "input_csv language granularity n_samples"
declare -a JOBS=(
    "$SIEVE_OUT/javascript/js_functions_human.csv  JavaScript function 2500"
    "$SIEVE_OUT/javascript/js_classes_human.csv    JavaScript class   214"
    "$SIEVE_OUT/java/java_functions_human.csv      Java       function 2500"
    "$SIEVE_OUT/java/java_classes_human.csv        Java       class    1250"
    "$SIEVE_OUT/cpp/cpp_functions_human.csv        C++        function 2263"
    "$SIEVE_OUT/cpp/cpp_classes_human.csv          C++        class    1250"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log_section() {
    local sep="$(printf '=%.0s' {1..72})"
    log "$sep"
    log "$*"
    log "$sep"
}

# ── Preflight checks ──────────────────────────────────────────────────────────
log_section "PREFLIGHT CHECKS"

if [ ! -f "$GENERATE" ]; then
    log "ERROR: generate_llm_code.py not found at $GENERATE"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    log "WARNING: .env file not found — TOGETHER_API_KEY must be set in environment"
else
    log "OK: .env file found"
fi

python3 -c "import together" 2>/dev/null || {
    log "ERROR: 'together' package not installed. Run: pip install together"
    exit 1
}

log "OK: together package available"
log "Log file: $LOG_FILE"
log "Output dir: $DATA_OUT"
log ""

# Count total jobs
TOTAL_JOBS=$(( ${#MODELS[@]} * ${#JOBS[@]} ))
COMPLETED=0
FAILED=0
SKIPPED=0

log_section "STARTING $TOTAL_JOBS GENERATION JOBS"
log "Models: ${#MODELS[@]}  ×  Language/granularity pairs: ${#JOBS[@]}"
log ""

START_TIME=$SECONDS

# ── Main loop ─────────────────────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    SLUG="$(model_slug "$MODEL")"
    MODEL_OUT="$DATA_OUT/$SLUG"
    mkdir -p "$MODEL_OUT"

    log_section "MODEL: $MODEL  (slug: $SLUG)"

    for JOB in "${JOBS[@]}"; do
        # Parse job fields
        read -r INPUT LANG GRAN N_SAMPLES <<< "$JOB"

        # Derive output filename: e.g. javascript_function.csv
        LANG_SLUG="$(echo "$LANG" | tr '[:upper:]' '[:lower:]' | tr '+' 'p' | tr ' ' '_')"
        OUTPUT="$MODEL_OUT/${LANG_SLUG}_${GRAN}.csv"

        COMPLETED=$(( COMPLETED + 1 ))
        JOB_NUM="$COMPLETED/$TOTAL_JOBS"

        log ""
        log "── Job $JOB_NUM ──────────────────────────────────────────"
        log "  Model:       $MODEL"
        log "  Language:    $LANG"
        log "  Granularity: $GRAN"
        log "  Input:       $INPUT"
        log "  Output:      $OUTPUT"
        log "  Target rows: $N_SAMPLES"

        # Skip if output already exists and is non-empty
        if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
            EXISTING=$(python3 -c "import pandas as pd; df=pd.read_csv('$OUTPUT'); print(len(df))" 2>/dev/null || echo 0)
            if [ "$EXISTING" -gt 0 ]; then
                log "  STATUS: SKIPPED (already exists with $EXISTING rows)"
                SKIPPED=$(( SKIPPED + 1 ))
                COMPLETED=$(( COMPLETED - 1 ))
                continue
            fi
        fi

        JOB_START=$SECONDS

        # Run generation — end=N_SAMPLES so we take the first N rows as input
        if python3 "$GENERATE" \
            --input       "$INPUT" \
            --output      "$OUTPUT" \
            --model       "$MODEL" \
            --language    "$LANG" \
            --granularity "$GRAN" \
            --end         "$N_SAMPLES" \
            --mode        sequential \
            >> "$LOG_FILE" 2>&1; then

            JOB_ELAPSED=$(( SECONDS - JOB_START ))
            ROWS=$(python3 -c "import pandas as pd; df=pd.read_csv('$OUTPUT'); print(df['generated_code'].notna().sum())" 2>/dev/null || echo "?")
            log "  STATUS: OK  ($ROWS rows generated, ${JOB_ELAPSED}s)"
        else
            JOB_ELAPSED=$(( SECONDS - JOB_START ))
            log "  STATUS: FAILED (${JOB_ELAPSED}s) — check log for details"
            FAILED=$(( FAILED + 1 ))
        fi

    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( TOTAL_ELAPSED / 3600 ))
MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 ))
SECS=$(( TOTAL_ELAPSED % 60 ))

log ""
log_section "GENERATION COMPLETE"
log "  Total jobs:  $TOTAL_JOBS"
log "  Completed:   $COMPLETED"
log "  Skipped:     $SKIPPED  (already existed)"
log "  Failed:      $FAILED"
log "  Elapsed:     ${HOURS}h ${MINUTES}m ${SECS}s"
log "  Output dir:  $DATA_OUT"
log "  Log file:    $LOG_FILE"
log ""

if [ "$FAILED" -gt 0 ]; then
    log "WARNING: $FAILED jobs failed — check $LOG_FILE for details"
    log "Re-run this script to retry failed jobs (completed jobs will be skipped)"
    exit 1
else
    log "All jobs completed successfully."
    log ""
    log "Next step: run clean_generated_code.py on each output file"
    log "  python3 scripts/data_collection/clean_generated_code.py \\"
    log "      --input data/generated/<model>/<lang>_<gran>.csv \\"
    log "      --language <lang> --granularity <func|class>"
fi