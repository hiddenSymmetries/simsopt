#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"

OUTBASE="$SCRIPT_DIR/output_permanent_magnet_GPMO_MUSE"
RUNROOT="$OUTBASE/paper_runs"
OUTDIR="$RUNROOT/paper_03_gb50uh"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/paper_03_gb50uh.log"
exec > >(tee -a "$LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

run_if_missing() {
  local marker="$1"; shift
  if [[ -f "$marker" ]]; then
    echo "[$(ts)] Skip (exists): $marker"
    return 0
  fi
  echo "[$(ts)] Run: $*"
  "$@"
}

# Preset: gb50uh (high-field study)
K=50000
N=64
DS=1
MAT="GB50UH"
BT=200
NADJ=12
NMAX=40000
KMM=100
HISTORY_EVERY=10

BASE="K${K}_nphi${N}_ntheta${N}_ds${DS}_mat${MAT}_bt${BT}_Nadj${NADJ}_nmax${NMAX}"
RID_GPMO="${BASE}_GPMO"
RID_GPMOMR="${BASE}_kmm${KMM}_GPMOmr"

run_if_missing "$OUTDIR/runhistory_${RID_GPMO}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset gb50uh --algorithm GPMO --history-every "$HISTORY_EVERY" --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_GPMOMR}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset gb50uh --algorithm GPMOmr --mm-refine-every "$KMM" --history-every "$HISTORY_EVERY" --outdir "$OUTDIR"

# Plots (2 runs => mse + deltam)
echo "[$(ts)] Plot: Combined_MSE_history.png"
"$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE_plots.py" \
  --outdir "$OUTDIR" --mode mse \
  --distinct-n-active \
  --runs "$RID_GPMO" "$RID_GPMOMR"

run_if_missing "$OUTDIR/plots/Histogram_DeltaM_log_GPMO_vs_GPMOmr.png" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE_plots.py" \
  --outdir "$OUTDIR" --mode deltam --compare "$RID_GPMO" "$RID_GPMOMR"

echo "[$(ts)] Done."
