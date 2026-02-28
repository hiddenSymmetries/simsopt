#!/usr/bin/env python
r"""
Finite-permeability permanent magnet optimization on the MUSE grid.

This script runs GPMOmr (GPMO with macromagnetic refinement) on the
MUSE magnet grid to optimise permanent magnets while accounting for
finite-permeability coupling and demagnetization effects.  A standard
GPMO run is also performed for comparison.

The macromagnetic refinement follows:

  A. Ulrich, M. Haberle, and A. A. Kaptanoglu,
  "Permanent magnet optimization of stellarators with coupling from
  finite permeability and demagnetization effects." 2025.
  arXiv:2512.14997

The baseline GPMO algorithm is described in:

  A. A. Kaptanoglu, R. Conlin, and M. Landreman,
  Greedy permanent magnet optimization,
  Nuclear Fusion 63, 036016 (2023)

Usage
-----
    python permanent_magnet_MUSE_finitemu_optimization.py

For higher-resolution and more realistic GPMOmr designs, please use the
paper-specific scripts available at:
  https://github.com/armulrich/simsopt/tree/gpmomr_paper_code
"""

import time
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier
from simsopt.util import (
    FocusData,
    in_github_actions,
    calculate_modB_on_major_radius,
    initialize_coils_for_pm_optimization,
    build_polarization_vectors_from_focus_data,
    compute_m_maxima_and_cube_dim_from_focus_file,
    run_gpmo_optimization_on_muse_grid,
    GPMORunConfig,
)

t_start = time.time()

# ---------------------------------------------------------------------------
# Material presets
# ---------------------------------------------------------------------------
MATERIALS = {
    "N52": {
        "B_max": 1.465,
        "mu_ea": 1.05,
        "mu_oa": 1.15,
        "scale_coils": False,
        "target_B0": None,
    },
    "GB50UH": {
        "B_max": 1.410,
        "mu_ea": 1.05,
        "mu_oa": 1.15,
        "scale_coils": True,
        "target_B0": 0.5,
    },
    "AlNiCo": {
        "B_max": 0.72,
        "mu_ea": 3.00,
        "mu_oa": 3.00,
        "scale_coils": True,
        "target_B0": 0.05,
    },
}

# ---------------------------------------------------------------------------
# Run presets
# ---------------------------------------------------------------------------
RUN_PRESETS = {
    "example_lowres_compare": {
        "material": "N52",
        "nphi": 16,
        "nIter_max": 2500,
        "nBacktracking": 200,
        "max_nMagnets": 5000,
        "downsample": 6,
        "history_every": 200,
        "mm_refine_every": 50,
        "Nadjacent": 12,
    },
}
DEFAULT_PRESET = "example_lowres_compare"

# Output directory
default_out_dir = (
    Path(__file__).resolve().parent
    / "output_permanent_magnet_GPMO_MUSE"
    / "example_lowres_compare"
)

# In CI we force a tiny configuration.
if in_github_actions:
    preset = {
        "material": "N52",
        "nphi": 8,
        "nIter_max": 100,
        "nBacktracking": 0,
        "max_nMagnets": 20,
        "mm_refine_every": 10,
        "downsample": 100,
        "history_every": 10,
        "Nadjacent": 12,
    }
    preset_name = "ci"
else:
    preset_name = DEFAULT_PRESET
    preset = dict(RUN_PRESETS[preset_name])

material_name = str(preset["material"])
material = MATERIALS[material_name]

nphi = int(preset["nphi"])
nIter_max = int(preset["nIter_max"])
nBacktracking = int(preset["nBacktracking"])
max_nMagnets = int(preset["max_nMagnets"])
downsample = int(preset["downsample"])
history_every = int(preset["history_every"])
ntheta = nphi
dr = 0.01
dx = dy = dz = dr
nHistory = max(1, int(nIter_max // history_every))
Nadjacent = int(preset["Nadjacent"])
mm_refine_every = int(preset["mm_refine_every"])
thresh_angle = np.pi - (5 * np.pi / 180)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / "zot80.focus"
surface_filename = TEST_DIR / "input.muse"
coil_path = TEST_DIR / "muse_tf_coils.focus"

out_dir = Path(default_out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Magnet grid preparation
# ---------------------------------------------------------------------------
mag_data = FocusData(famus_filename, downsample=downsample)
pol_vectors = build_polarization_vectors_from_focus_data(mag_data)
m_maxima, cube_dim = compute_m_maxima_and_cube_dim_from_focus_file(
    famus_filename, downsample, float(material["B_max"])
)

# Plasma surface
s = SurfaceRZFourier.from_focus(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)
s_inner = SurfaceRZFourier.from_focus(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)
s_outer = SurfaceRZFourier.from_focus(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)
_ = (s_inner, s_outer)

# Coils (optionally rescaled for a target on-axis field)
base_curves, curves, coils = initialize_coils_for_pm_optimization(
    "muse_famus", TEST_DIR, s, out_dir
)
_ = (base_curves, curves)

current_scale = 1.0
if bool(material["scale_coils"]):
    from simsopt.field import Coil

    bs_tmp = BiotSavart(coils)
    B0 = float(calculate_modB_on_major_radius(bs_tmp, s))
    target_B0 = float(material["target_B0"])
    current_scale = target_B0 / max(B0, 1e-30)
    coils = [Coil(c.curve, c.current * current_scale) for c in coils]
    bs_tmp = BiotSavart(coils)
    B0_check = float(calculate_modB_on_major_radius(bs_tmp, s))
    print(
        f"[INFO] B0 before: {B0:.6g} T, scale: {current_scale:.6g}, "
        f"B0 after: {B0_check:.6g} T"
    )

bs = BiotSavart(coils)
calculate_modB_on_major_radius(bs, s)

bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# ---------------------------------------------------------------------------
# Build the run configuration dataclass
# ---------------------------------------------------------------------------
config = GPMORunConfig(
    downsample=downsample,
    dr=dr,
    dx=dx,
    dy=dy,
    dz=dz,
    nphi=nphi,
    ntheta=ntheta,
    nIter_max=nIter_max,
    nHistory=nHistory,
    history_every=history_every,
    nBacktracking=nBacktracking,
    Nadjacent=Nadjacent,
    max_nMagnets=max_nMagnets,
    thresh_angle=thresh_angle,
    mm_refine_every=mm_refine_every,
    coil_path=coil_path,
    test_dir=TEST_DIR,
    surface_filename=surface_filename,
    material_name=material_name,
    material=material,
)

# ---------------------------------------------------------------------------
# Run GPMO (baseline, without macromagnetic refinement)
# ---------------------------------------------------------------------------
run_gpmo_optimization_on_muse_grid(
    "GPMO",
    subdir="GPMO",
    s=s,
    bs=bs,
    Bnormal=Bnormal,
    pol_vectors=pol_vectors,
    m_maxima=m_maxima,
    cube_dim=cube_dim,
    current_scale=current_scale,
    out_dir=out_dir,
    famus_filename=famus_filename,
    config=config,
)

# ---------------------------------------------------------------------------
# Run GPMOmr (with macromagnetic refinement)
# ---------------------------------------------------------------------------
run_gpmo_optimization_on_muse_grid(
    "GPMOmr",
    subdir="GPMOmr",
    s=s,
    bs=bs,
    Bnormal=Bnormal,
    pol_vectors=pol_vectors,
    m_maxima=m_maxima,
    cube_dim=cube_dim,
    current_scale=current_scale,
    out_dir=out_dir,
    famus_filename=famus_filename,
    config=config,
)

t_end = time.time()
print(f"\nTotal time = {t_end - t_start:.3f} s")
