#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"

MUSE_SCRIPT="$REPO_ROOT/examples/2_Intermediate/permanent_magnet_MUSE.py"
PLOTS_SCRIPT="$REPO_ROOT/examples/2_Intermediate/permanent_magnet_MUSE_plots.py"
POST_SCRIPT="$REPO_ROOT/examples/2_Intermediate/macromag_MUSE_AlNiCo_post_processing.py"

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

# Like run_if_missing, but re-runs if any dependency is newer than the marker.
# Usage: run_if_missing_or_stale <marker> <dep1> [dep2 ...] -- <command...>
run_if_missing_or_stale() {
  local marker="$1"; shift
  local -a deps=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    deps+=("$1")
    shift
  done
  if [[ $# -eq 0 ]]; then
    echo "run_if_missing_or_stale: missing -- separator" >&2
    return 2
  fi
  shift  # consume --

  if [[ -f "$marker" ]]; then
    for dep in "${deps[@]}"; do
      # If a dependency is missing or newer, treat output as stale.
      if [[ ! -e "$dep" || "$dep" -nt "$marker" ]]; then
        echo "[$(ts)] Run (stale): $*"
        "$@"
        return $?
      fi
    done
    echo "[$(ts)] Skip (up-to-date): $marker"
    return 0
  fi

  echo "[$(ts)] Run (missing): $*"
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
HISTORY_EVERY=10
SNAP_K=18000

BASE="K${K}_nphi${N}_ntheta${N}_ds${DS}_mat${MAT}_bt${BT}_Nadj${NADJ}_nmax${NMAX}"
RID_GPMO="${BASE}_GPMO"
RID_GPMOMR="${BASE}_kmm${KMM}_GPMOmr"

run_if_missing "$OUTDIR/dipoles_snapshot_K${SNAP_K}_mat${MAT}_bt${BT}_Nadj${NADJ}_nmax${NMAX}_GPMO.npz" \
  "$PYTHON" "$MUSE_SCRIPT" \
  --preset alnico --algorithm GPMO --history-every "$HISTORY_EVERY" --snapshot-k "$SNAP_K" --outdir "$OUTDIR"

run_if_missing "$OUTDIR/dipoles_snapshot_K${SNAP_K}_mat${MAT}_bt${BT}_Nadj${NADJ}_nmax${NMAX}_kmm${KMM}_GPMOmr.npz" \
  "$PYTHON" "$MUSE_SCRIPT" \
  --preset alnico --algorithm GPMOmr --mm-refine-every "$KMM" --history-every "$HISTORY_EVERY" --snapshot-k "$SNAP_K" --outdir "$OUTDIR"

# Plots (2 runs => mse + deltam)
echo "[$(ts)] Plot: Combined_MSE_history.png"
"$PYTHON" "$PLOTS_SCRIPT" \
  --outdir "$OUTDIR" --mode mse \
  --mark-k "$SNAP_K" \
  --runs "$RID_GPMO" "$RID_GPMOMR"

echo "[$(ts)] Plot: Histogram_DeltaM_log_GPMO_vs_GPMOmr_K${SNAP_K}.png"
"$PYTHON" "$PLOTS_SCRIPT" \
  --outdir "$OUTDIR" --mode deltam --deltam-k "$SNAP_K" --compare "$RID_GPMO" "$RID_GPMOMR"

# Post-processing (integrated here since it relies on the AlNiCo run outputs).
NPZ_NAME="dipoles_final_matAlNiCo_bt200_Nadj12_nmax40000_GPMO.npz"
POSTDIR="$OUTDIR/postprocess"
mkdir -p "$POSTDIR"
run_if_missing_or_stale "$POSTDIR/surface_Bn_delta_AlNiCo.vts" \
  "$POST_SCRIPT" \
  "$OUTDIR/$NPZ_NAME" \
  -- \
  "$PYTHON" "$POST_SCRIPT" \
    --npz-path "$OUTDIR/$NPZ_NAME" \
    --outdir "$POSTDIR"

echo "[$(ts)] Done."
