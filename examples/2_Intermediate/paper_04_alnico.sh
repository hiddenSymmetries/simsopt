#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"

OUTBASE="$SCRIPT_DIR/output_permanent_magnet_GPMO_MUSE"
RUNROOT="$OUTBASE/paper_runs"
OUTDIR="$RUNROOT/paper_04_alnico"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/paper_04_alnico.log"
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

# Preset: alnico (low-remanence / stronger coupling study)
K=25000
N=64
DS=1
MAT="AlNiCo"
BT=200
NADJ=12
NMAX=40000
KMM=50

BASE="K${K}_nphi${N}_ntheta${N}_ds${DS}_mat${MAT}_bt${BT}_Nadj${NADJ}_nmax${NMAX}"
RID_GPMO="${BASE}_GPMO"
RID_GPMOMR="${BASE}_kmm${KMM}_GPMOmr"

run_if_missing "$OUTDIR/runhistory_${RID_GPMO}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset alnico --algorithm GPMO --outdir "$OUTDIR"

run_if_missing "$OUTDIR/runhistory_${RID_GPMOMR}.csv" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE.py" \
  --preset alnico --algorithm GPMOmr --mm-refine-every "$KMM" --outdir "$OUTDIR"

# Plots (2 runs => mse + deltam)
run_if_missing "$OUTDIR/plots/Combined_MSE_history.png" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE_plots.py" \
  --outdir "$OUTDIR" --mode mse

run_if_missing "$OUTDIR/plots/Histogram_DeltaM_log_GPMO_vs_GPMOmr.png" \
  "$PYTHON" "$SCRIPT_DIR/permanent_magnet_MUSE_plots.py" \
  --outdir "$OUTDIR" --mode deltam --compare "$RID_GPMO" "$RID_GPMOMR"

# Post-processing (integrated here since it relies on the AlNiCo run outputs).
NPZ_NAME="dipoles_final_matAlNiCo_bt200_Nadj12_nmax40000_GPMO.npz"
POSTDIR="$OUTDIR/postprocess"
mkdir -p "$POSTDIR"
run_if_missing "$POSTDIR/surface_Bn_delta_AlNiCo.vtp" \
  "$PYTHON" "$SCRIPT_DIR/macromag_MUSE_AlNiCo_post_processing.py" \
  --npz-path "$OUTDIR/$NPZ_NAME" \
  --outdir "$POSTDIR"

echo "[$(ts)] Done."
