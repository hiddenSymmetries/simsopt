#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"

POST_SCRIPT="$REPO_ROOT/examples/2_Intermediate/macromag_MUSE_post_processing.py"

OUTBASE="$SCRIPT_DIR/output_permanent_magnet_GPMO_MUSE"
RUNROOT="$OUTBASE/paper_runs"
OUTDIR="$RUNROOT/paper_00_postprocess_muse_grid"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/paper_00_postprocess_muse_grid.log"
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

run_if_missing "$OUTDIR/surface_Bn_delta_MUSE_post_processing.vts" \
  "$PYTHON" "$POST_SCRIPT" --outdir "$OUTDIR"

echo "[$(ts)] Done."
