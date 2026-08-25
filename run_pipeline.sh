#!/usr/bin/env bash
#
# run_pipeline.sh
# Runs the full pipeline end to end and logs everything.
#
# Usage:
#   ./run_pipeline.sh              full run (rebuilds database)
#   ./run_pipeline.sh --keep-db    keep existing database
#   ./run_pipeline.sh --skip-fe    skip feature engineering (slowest step)
#
# Logs go to logs/pipeline_YYYYMMDD_HHMMSS.log

set -u  # error on undefined variables

REBUILD_DB=1
SKIP_FE=0

for arg in "$@"; do
    case $arg in
        --keep-db) REBUILD_DB=0 ;;
        --skip-fe) SKIP_FE=1 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/pipeline_${TS}.log"

# send everything to both terminal and log file
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Pipeline run started: $(date)"
echo "Log file: $LOG"
echo "============================================================"

START_ALL=$(date +%s)

# runs one step, times it, stops the whole script if it fails
run_step () {
    local name="$1"; shift
    echo ""
    echo "------------------------------------------------------------"
    echo ">> $name"
    echo "   started $(date +%H:%M:%S)"
    echo "------------------------------------------------------------"

    local t0=$(date +%s)
    if ! "$@"; then
        echo ""
        echo "!! FAILED: $name"
        echo "!! Pipeline stopped. See $LOG"
        exit 1
    fi
    local mins=$(( ($(date +%s) - t0) / 60 ))
    local secs=$(( ($(date +%s) - t0) % 60 ))
    echo ""
    echo "   done in ${mins}m ${secs}s"
}

# check disk space before starting - feature engineering needs room
AVAIL=$(df --output=avail -BG /workspaces | tail -1 | tr -dc '0-9')
echo ""
echo "Disk space available: ${AVAIL}GB"
if [ "$AVAIL" -lt 3 ]; then
    echo "WARNING: less than 3GB free. Consider clearing space first."
fi

if [ "$REBUILD_DB" -eq 1 ]; then
    echo ""
    echo "Removing existing database for a clean rebuild..."
    rm -f data/freshwater_risk.db
fi

run_step "1/6  Create database tables"        python src/db_loader.py

if [ "$SKIP_FE" -eq 0 ]; then
    run_step "2/6  Feature engineering (slow - land cover extraction)" \
             python src/feature_engineering.py --no-staging
else
    echo ""
    echo ">> 2/6  Feature engineering SKIPPED (--skip-fe)"
fi

run_step "3/6  Model evaluation (traditional vs ML, ablation)" python src/model_evaluation.py
run_step "4/6  Train production model"                          python src/model_training.py
run_step "5/6  SHAP explainability analysis"                    python src/shap_analysis.py
run_step "6/6  Generate predictions"                            python src/inference.py

TOTAL=$(( ($(date +%s) - START_ALL) / 60 ))

echo ""
echo "============================================================"
echo "Pipeline complete in ${TOTAL} minutes"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Key results to check:"
grep -E "Sanity check|Overall weighted F1|Total predictions" "$LOG" || true
echo ""
echo "Full log: $LOG"
echo ""
echo "Start the dashboard with:  streamlit run src/dashboard.py"