#!/bin/bash
# =============================================================================
# clean_all.sh
#
# Runs clean_generated_code.py on all 24 generated CSV files to produce
# the generated_code_cleaned column for each.
#
# Run this after generate_all.sh completes.
#
# Usage:
#   chmod +x clean_all.sh
#   ./clean_all.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN="$SCRIPT_DIR/scripts/data_collection/clean_generated_code.py"
DATA_OUT="$SCRIPT_DIR/data/generated"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/clean_all_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# ── Models and slug function ──────────────────────────────────────────────────
MODELS=(
    "deepseek-ai/DeepSeek-V3.1"
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "mistralai/Mistral-Small-24B-Instruct-2501"
    "openai/gpt-oss-20b"
)

model_slug() {
    echo "$1" \
        | sed 's|.*/||' \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's/[^a-z0-9]/-/g' \
        | sed 's/-\+/-/g' \
        | sed 's/-$//'
}

# ── Jobs: "slug language granularity" ─────────────────────────────────────────
declare -a JOBS=(
    "javascript function"
    "javascript class"
    "java       function"
    "java       class"
    "cpp        function"
    "cpp        class"
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

# Language display name from slug
lang_display() {
    case "$1" in
        javascript) echo "JavaScript" ;;
        java)       echo "Java" ;;
        cpp)        echo "C++" ;;
        *)          echo "$1" ;;
    esac
}

log_section "CLEANING ALL GENERATED FILES"
log "Log: $LOG_FILE"
log ""

TOTAL=$(( ${#MODELS[@]} * ${#JOBS[@]} ))
DONE=0
FAILED=0
SKIPPED=0
START_TIME=$SECONDS

for MODEL in "${MODELS[@]}"; do
    SLUG="$(model_slug "$MODEL")"
    log_section "Model: $MODEL  (slug: $SLUG)"

    for JOB in "${JOBS[@]}"; do
        read -r LANG_SLUG GRAN <<< "$JOB"
        LANG="$(lang_display "$LANG_SLUG")"
        INPUT="$DATA_OUT/$SLUG/${LANG_SLUG}_${GRAN}.csv"

        DONE=$(( DONE + 1 ))
        log "── $DONE/$TOTAL  $SLUG / $LANG / $GRAN"

        # Check input exists
        if [ ! -f "$INPUT" ]; then
            log "  STATUS: SKIPPED (input not found: $INPUT)"
            SKIPPED=$(( SKIPPED + 1 ))
            DONE=$(( DONE - 1 ))
            continue
        fi

        # Check if already cleaned
        ALREADY=$(python3 -c "
import pandas as pd
df = pd.read_csv('$INPUT')
if 'generated_code_cleaned' in df.columns:
    print(df['generated_code_cleaned'].notna().sum())
else:
    print(-1)
" 2>/dev/null || echo -1)

        if [ "$ALREADY" -gt 0 ] 2>/dev/null; then
            log "  STATUS: SKIPPED (already cleaned, $ALREADY rows)"
            SKIPPED=$(( SKIPPED + 1 ))
            DONE=$(( DONE - 1 ))
            continue
        fi

        JOB_START=$SECONDS

        if python3 "$CLEAN" \
            --input       "$INPUT" \
            --language    "$LANG" \
            --granularity "$GRAN" \
            >> "$LOG_FILE" 2>&1; then

            JOB_ELAPSED=$(( SECONDS - JOB_START ))
            CLEAN_COUNT=$(python3 -c "
import pandas as pd
df = pd.read_csv('$INPUT')
print(df['generated_code_cleaned'].notna().sum())
" 2>/dev/null || echo "?")
            log "  STATUS: OK  ($CLEAN_COUNT parseable rows, ${JOB_ELAPSED}s)"
        else
            JOB_ELAPSED=$(( SECONDS - JOB_START ))
            log "  STATUS: FAILED (${JOB_ELAPSED}s)"
            FAILED=$(( FAILED + 1 ))
        fi

    done
done

TOTAL_ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( TOTAL_ELAPSED / 3600 ))
MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 ))
SECS=$(( TOTAL_ELAPSED % 60 ))

log ""
log_section "CLEANING COMPLETE"
log "  Total:    $TOTAL"
log "  Done:     $DONE"
log "  Skipped:  $SKIPPED"
log "  Failed:   $FAILED"
log "  Elapsed:  ${HOURS}h ${MINUTES}m ${SECS}s"
log "  Log:      $LOG_FILE"

if [ "$FAILED" -gt 0 ]; then
    log "WARNING: $FAILED jobs failed — check $LOG_FILE"
    exit 1
fi

# ── Strip imports from Python data ────────────────────────────────────────────
STRIP="$SCRIPT_DIR/scripts/data_collection/strip_python_imports.py"
PYTHON_DATA="$SCRIPT_DIR/data/existing_python_data"

log ""
log_section "STRIPPING IMPORTS FROM PYTHON DATA"

PYTHON_FILES_FUNC=(
    "claude-3-haiku_func_level_filtered.csv"
    "claude-4_5-haiku_func_level_filtered.csv"
    "gpt-3_5_func_level_filtered.csv"
    "gpt-oss_func_level_filtered.csv"
)

PYTHON_FILES_CLS=(
    "claude-3-haiku_class_level_filtered.csv"
    "claude-4_5-haiku_class_level_filtered.csv"
    "gpt-3_5_class_level_filtered.csv"
    "gpt-oss_class_level_filtered.csv"
)

PY_FAILED=0

for fname in "${PYTHON_FILES_FUNC[@]}"; do
    fpath="$PYTHON_DATA/$fname"
    if [ ! -f "$fpath" ]; then
        log "  SKIPPED (not found): $fname"
        continue
    fi
    log "  function: $fname"
    if python3 "$STRIP" --input "$fpath" --granularity function >> "$LOG_FILE" 2>&1; then
        log "    OK"
    else
        log "    FAILED"
        PY_FAILED=$(( PY_FAILED + 1 ))
    fi
done

for fname in "${PYTHON_FILES_CLS[@]}"; do
    fpath="$PYTHON_DATA/$fname"
    if [ ! -f "$fpath" ]; then
        log "  SKIPPED (not found): $fname"
        continue
    fi
    log "  class: $fname"
    if python3 "$STRIP" --input "$fpath" --granularity class >> "$LOG_FILE" 2>&1; then
        log "    OK"
    else
        log "    FAILED"
        PY_FAILED=$(( PY_FAILED + 1 ))
    fi
done

if [ "$PY_FAILED" -gt 0 ]; then
    log "WARNING: $PY_FAILED Python strip jobs failed — check $LOG_FILE"
    exit 1
else
    log ""
    log "All done. Next step: merge human + AI data for classifier training."
fi