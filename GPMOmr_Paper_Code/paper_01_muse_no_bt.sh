#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"

MUSE_SCRIPT="$REPO_ROOT/examples/2_Intermediate/permanent_magnet_MUSE.py"
PLOTS_SCRIPT="$REPO_ROOT/examples/2_Intermediate/permanent_magnet_MUSE_plots.py"

OUTBASE="$SCRIPT_DIR/output_permanent_magnet_GPMO_MUSE"
RUNROOT="$OUTBASE/paper_runs"
OUTDIR="$RUNROOT/paper_01_muse_no_bt"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/paper_01_muse_no_bt.log"
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

# Preset: muse_no_bt (paper baseline without backtracking)
K=20000
N=64
DS=1
MAT="N52"
BT=0
NADJ=12
NMAX=20000
HISTORY_EVERY=10

BASE="K${K}_nphi${N}_ntheta${N}_ds${DS}_mat${MAT}_bt${BT}_Nadj${NADJ}_nmax${NMAX}"

RID_GPMO="${BASE}_GPMO"
RID_KMM1="${BASE}_kmm1_GPMOmr"
RID_KMM25="${BASE}_kmm25_GPMOmr"
RID_KMM50="${BASE}_kmm50_GPMOmr"

run_if_missing "$OUTDIR/runhistory_${RID_GPMO}.csv" \
  "$PYTHON" "$MUSE_SCRIPT" \
  --preset muse_no_bt --algorithm GPMO --history-every "$HISTORY_EVERY" --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_KMM1}.csv" \
  "$PYTHON" "$MUSE_SCRIPT" \
  --preset muse_no_bt --algorithm GPMOmr --mm-refine-every 1 --history-every "$HISTORY_EVERY" --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_KMM25}.csv" \
  "$PYTHON" "$MUSE_SCRIPT" \
  --preset muse_no_bt --algorithm GPMOmr --mm-refine-every 25 --history-every "$HISTORY_EVERY" --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_KMM50}.csv" \
  "$PYTHON" "$MUSE_SCRIPT" \
  --preset muse_no_bt --algorithm GPMOmr --mm-refine-every 50 --history-every "$HISTORY_EVERY" --outdir "$OUTDIR"

# Plots
echo "[$(ts)] Plot: Combined_MSE_history.png"
"$PYTHON" "$PLOTS_SCRIPT" \
  --outdir "$OUTDIR" --mode mse --no-n-active \
  --runs "$RID_GPMO" "$RID_KMM1" "$RID_KMM25" "$RID_KMM50"

# One representative ΔM comparison (paper default uses kmm=50)
echo "[$(ts)] Plot: Histogram_DeltaM_log_GPMO_vs_GPMOmr.png"
"$PYTHON" "$PLOTS_SCRIPT" \
  --outdir "$OUTDIR" --mode deltam --compare "$RID_GPMO" "$RID_KMM50"

echo "[$(ts)] Done."
