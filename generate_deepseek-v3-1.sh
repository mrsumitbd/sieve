#!/usr/bin/env bash
# Auto-generated — run in parallel with other per-model scripts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE="$SCRIPT_DIR/scripts/data_collection/generate_llm_code.py"
SIEVE_OUT="$SCRIPT_DIR/sieve_output"
DATA_OUT="$SCRIPT_DIR/data/generated"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="deepseek-ai/DeepSeek-V3.1"
SLUG="deepseek-v3-1"

LOG_FILE="$LOG_DIR/generate_${SLUG}_${TIMESTAMP}.log"
mkdir -p "$DATA_OUT/$SLUG" "$LOG_DIR"

log() { local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"; echo "$msg"; echo "$msg" >> "$LOG_FILE"; }

declare -a JOBS=(
    "$SIEVE_OUT/javascript/js_functions_human.csv  JavaScript function 2500"
    "$SIEVE_OUT/javascript/js_classes_human.csv    JavaScript class   214"
    "$SIEVE_OUT/java/java_functions_human.csv      Java       function 2500"
    "$SIEVE_OUT/java/java_classes_human.csv        Java       class    1250"
    "$SIEVE_OUT/cpp/cpp_functions_human.csv        C++        function 2263"
    "$SIEVE_OUT/cpp/cpp_classes_human.csv          C++        class    1250"
)

log "Starting 6 jobs for $MODEL"
FAILED=0

for JOB in "${JOBS[@]}"; do
    read -r INPUT LANG GRAN N_SAMPLES <<< "$JOB"
    LANG_SLUG="$(echo "$LANG" | tr '[:upper:]' '[:lower:]' | tr '+' 'p' | tr ' ' '_')"
    OUTPUT="$DATA_OUT/$SLUG/${LANG_SLUG}_${GRAN}.csv"

    log "── $LANG / $GRAN → $OUTPUT (n=$N_SAMPLES)"

    if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
        EXISTING=$(python3 -c "import pandas as pd; df=pd.read_csv('$OUTPUT'); print(len(df[df['generated_code'].notna()]))" 2>/dev/null || echo 0)
        if [ "$EXISTING" -gt 0 ]; then
            log "  SKIPPED (exists: $EXISTING rows)"
            continue
        fi
    fi

    JOB_START=$SECONDS
    if python3 "$GENERATE" \
        --input "$INPUT" --output "$OUTPUT" \
        --model "$MODEL" --language "$LANG" \
        --granularity "$GRAN" --end "$N_SAMPLES" \
        --mode sequential >> "$LOG_FILE" 2>&1; then
        ROWS=$(python3 -c "import pandas as pd; df=pd.read_csv('$OUTPUT'); print(df['generated_code'].notna().sum())" 2>/dev/null || echo "?")
        log "  OK ($ROWS rows, $(( SECONDS - JOB_START ))s)"
    else
        log "  FAILED ($(( SECONDS - JOB_START ))s)"
        FAILED=$(( FAILED + 1 ))
    fi
done

log "Done. Failed: $FAILED/6"
[ "$FAILED" -eq 0 ] && exit 0 || exit 1