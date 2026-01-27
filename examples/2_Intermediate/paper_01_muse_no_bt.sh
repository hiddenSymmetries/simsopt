#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"

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

BASE="K${K}_nphi${N}_ntheta${N}_ds${DS}_mat${MAT}_bt${BT}_Nadj${NADJ}_nmax${NMAX}"

RID_GPMO="${BASE}_GPMO"
RID_KMM1="${BASE}_kmm1_GPMOmr"
RID_KMM25="${BASE}_kmm25_GPMOmr"
RID_KMM50="${BASE}_kmm50_GPMOmr"

run_if_missing "$OUTDIR/runhistory_${RID_GPMO}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset muse_no_bt --algorithm GPMO --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_KMM1}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset muse_no_bt --algorithm GPMOmr --mm-refine-every 1 --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_KMM25}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset muse_no_bt --algorithm GPMOmr --mm-refine-every 25 --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_KMM50}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset muse_no_bt --algorithm GPMOmr --mm-refine-every 50 --outdir "$OUTDIR"

# Plots
run_if_missing "$OUTDIR/plots/Combined_MSE_history.png" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE_plots.py" \
  --outdir "$OUTDIR" --mode mse

# One representative ΔM comparison (paper default uses kmm=50)
run_if_missing "$OUTDIR/plots/Histogram_DeltaM_log_GPMO_vs_GPMOmr.png" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE_plots.py" \
  --outdir "$OUTDIR" --mode deltam --compare "$RID_GPMO" "$RID_KMM50"

echo "[$(ts)] Done."
