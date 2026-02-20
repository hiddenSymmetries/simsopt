#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving permanent
magnet optimization on the MUSE grid. This
algorithm is described in the following paper:
    A. A. Kaptanoglu, R. Conlin, and M. Landreman,
    Greedy permanent magnet optimization,
    Nuclear Fusion 63, 036016 (2023)

This script includes the GPMOmr extension, which incorporates 
implementation components from:

  A. Ulrich, M. Haberle, and A. A. Kaptanoglu,
  “Permanent magnet optimization of stellarators with coupling from finite
  permeability and demagnetization effects.” 2025. arXiv:2512.14997


The script should be run as:
    mpirun -n 1 python permanent_magnet_MUSE.py
on a cluster machine but
    python permanent_magnet_MUSE.py
is sufficient on other machines. Note that this code does not use MPI, but is
parallelized via OpenMP and XSIMD, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

For high-resolution and more realistic designs, please see the script files at
https://github.com/akaptano/simsopt_permanent_magnet_advanced_scripts.git

For higher-resolution and more realistic GPMOmr designs, 
please use the paper-specific scripts available at:
  https://github.com/armulrich/simsopt/tree/gpmomr_paper_code
"""

import time
import csv
import datetime
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.coil_optimization_helper_functions import \
    calculate_modB_on_major_radius
from simsopt.util.permanent_magnet_helper_functions import \
    initialize_default_kwargs, \
    initialize_coils_for_pm_optimization

t_start = time.time()

# ----------------------------
# Run configuration presets
# ----------------------------

# Material presets used in the paper / studies.
# - B_max controls the per-site dipole moment cap m_max = (B_max/mu0) * V.
# - mu_ea/mu_oa are the (relative) permeabilities along / perpendicular to easy axis.
# - target_B0 controls coil current scaling (if scale_coils=True).
MATERIALS = {
    # Baseline MUSE NdFeB (close to an ideal permanent-magnet source).
    "N52": {
        "B_max": 1.465,  # Tesla
        "mu_ea": 1.05,
        "mu_oa": 1.15,
        "scale_coils": False,
        "target_B0": None,
    },
    # High-coercivity NdFeB grade for higher-field studies.
    "GB50UH": {
        "B_max": 1.410,  # Tesla
        "mu_ea": 1.05,
        "mu_oa": 1.15,
        "scale_coils": True,
        "target_B0": 0.5,  # Tesla
    },
    # Low-remanence material with deliberately stronger effective permeability.
    "AlNiCo": {
        "B_max": 0.72,  # Tesla
        "mu_ea": 3.00,
        "mu_oa": 3.00,
        "scale_coils": True,
        "target_B0": 0.05,  # Tesla
    },
}

# ----------------------------
# Example run configuration
# ----------------------------

# Preset controlling *both* runs (GPMO then GPMOmr) so the results are directly comparable.
RUN_PRESETS = {
    "example_lowres_compare": {
        "material": "N52",
        "nphi": 8,
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

# Output directory (kept inside examples so it's easy to find when learning).
default_out_dir = (
    Path(__file__).resolve().parent / "output_permanent_magnet_GPMO_MUSE" / "example_lowres_compare"
)

# Set parameters -- in CI we force a tiny configuration.
if in_github_actions:
    preset = {
        "material": "N52",
        "nphi": 2,
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

# Keep these assignments explicit: this makes it obvious what settings were used
# when the script is copied into other workflows.
nphi = int(preset["nphi"])
nIter_max = int(preset["nIter_max"])
nBacktracking = int(preset["nBacktracking"])
max_nMagnets = int(preset["max_nMagnets"])
downsample = int(preset["downsample"])
history_every = int(preset["history_every"])

ntheta = nphi
dr = 0.01  # Radial extent in meters of the cylindrical permanent magnet bricks
dx = dy = dz = dr
input_name = "input.muse"

nHistory = max(1, int(nIter_max // history_every))
Nadjacent = int(preset["Nadjacent"])
mm_refine_every = int(preset["mm_refine_every"])
thresh_angle = np.pi - (5 * np.pi / 180)  # remove if adjacent angle exceeds this

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / "zot80.focus"
surface_filename = TEST_DIR / input_name
coil_path = TEST_DIR / "muse_tf_coils.focus"


def ts() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds")


def write_runhistory(out_subdir: Path, run_id: str, pm_opt, fB_hist, absBn_hist):
    out_subdir.mkdir(parents=True, exist_ok=True)
    run_csv = out_subdir / f"runhistory_{run_id}.csv"

    k_hist = getattr(pm_opt, "k_history", None)
    if k_hist is None or len(k_hist) != len(fB_hist):
        k_hist = np.arange(len(fB_hist), dtype=int)

    n_active = getattr(pm_opt, "num_nonzeros", None)
    if n_active is None or len(n_active) != len(fB_hist):
        n_active = np.zeros(len(fB_hist), dtype=int)

    with run_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "fB", "absBn", "n_active"])
        w.writeheader()
        for i in range(len(fB_hist)):
            w.writerow(
                {
                    "k": int(k_hist[i]),
                    "fB": float(fB_hist[i]),
                    "absBn": float(absBn_hist[i]),
                    "n_active": int(n_active[i]),
                }
            )
    return run_csv


def write_run_yaml(out_subdir: Path, run_id: str, *, algorithm: str, current_scale: float, metrics: dict):
    doc = {
        "run_id": run_id,
        "timestamp_utc": ts(),
        "algorithm": algorithm,
        "material": {"name": material_name, **material},
        "params": {
            "nphi": nphi,
            "ntheta": ntheta,
            "downsample": downsample,
            "dr": dr,
            "K": nIter_max,
            "nhistory": nHistory,
            "history_every": history_every,
            "backtracking": nBacktracking,
            "Nadjacent": Nadjacent,
            "max_nMagnets": max_nMagnets,
            "thresh_angle": float(thresh_angle),
            "mm_refine_every": mm_refine_every if algorithm == "GPMOmr" else 0,
            "current_scale": float(current_scale),
        },
        "paths": {
            "test_dir": str(TEST_DIR),
            "famus_filename": str(famus_filename),
            "surface_filename": str(surface_filename),
            "coil_path": str(coil_path),
        },
        "metrics": metrics,
    }

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml_path = out_subdir / f"run_{run_id}.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(doc, f)
    return yaml_path


def build_pol_vectors(mag_data: FocusData) -> np.ndarray:
    pol_axes = np.zeros((0, 3))
    pol_type = np.zeros(0, dtype=int)
    pol_axes_f, pol_type_f = polarization_axes(["face"])
    ntype_f = int(len(pol_type_f) / 2)
    pol_axes_f = pol_axes_f[:ntype_f, :]
    pol_type_f = pol_type_f[:ntype_f]
    pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_f))

    ophi = np.arctan2(mag_data.oy, mag_data.ox)
    discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
    pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
    pol_vectors[:, :, 0] = mag_data.pol_x
    pol_vectors[:, :, 1] = mag_data.pol_y
    pol_vectors[:, :, 2] = mag_data.pol_z
    return pol_vectors


def compute_m_maxima_and_cube_dim(focus_path: Path, downsample: int, B_max: float):
    # Reference B_max used when generating the MUSE .focus (FAMUS) grid.
    B_ref = 1.465
    mu0 = 4 * np.pi * 1e-7

    ox, oy, oz, Ic, M0s = np.loadtxt(
        focus_path, skiprows=3, usecols=[3, 4, 5, 6, 7], delimiter=",", unpack=True
    )

    inds_total = np.arange(len(ox))
    inds_downsampled = inds_total[::downsample]
    nonzero_inds = np.intersect1d(np.ravel(np.where(Ic == 1.0)), inds_downsampled)

    m_maxima = M0s[nonzero_inds] * (B_max / B_ref)

    cell_vol = M0s * mu0 / B_ref
    cube_dim = float(cell_vol[0] ** (1.0 / 3.0))
    return m_maxima, cube_dim


def compute_final_fB(*, s_plot: SurfaceRZFourier, bs: BiotSavart, pm_opt: PermanentMagnetGrid) -> float:
    qphi = nphi
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    b_dipole = DipoleField(
        pm_opt.dipole_grid_xyz,
        pm_opt.m,
        nfp=s_plot.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    return float(SquaredFlux(s_plot, b_dipole, -Bnormal).J())


def run_one(
    algorithm: str,
    *,
    subdir: str,
    s: SurfaceRZFourier,
    bs: BiotSavart,
    Bnormal: np.ndarray,
    pol_vectors: np.ndarray,
    m_maxima: np.ndarray,
    cube_dim: float,
    current_scale: float,
    out_dir: Path,
):
    out_subdir = out_dir / subdir
    out_subdir.mkdir(parents=True, exist_ok=True)

    # grid + optimizer object
    kwargs_grid = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr, "m_maxima": m_maxima}
    pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs_grid)
    print(f"[INFO] Number of available dipoles = {pm_opt.ndipoles}")

    # solver kwargs (same for both algorithms)
    kwargs = initialize_default_kwargs("GPMO")
    kwargs["K"] = int(nIter_max)
    kwargs["nhistory"] = int(nHistory)
    kwargs["backtracking"] = int(nBacktracking)
    kwargs["Nadjacent"] = int(Nadjacent)
    kwargs["dipole_grid_xyz"] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs["max_nMagnets"] = int(max_nMagnets)
    kwargs["thresh_angle"] = float(thresh_angle)

    if algorithm == "GPMOmr":
        kwargs["cube_dim"] = float(cube_dim)
        kwargs["mu_ea"] = float(material["mu_ea"])
        kwargs["mu_oa"] = float(material["mu_oa"])
        kwargs["use_coils"] = True
        kwargs["use_demag"] = True
        kwargs["coil_path"] = coil_path
        kwargs["mm_refine_every"] = int(mm_refine_every)
        kwargs["current_scale"] = float(current_scale)

    print(
        f"\n[{ts()}] Run: {algorithm} (nphi={nphi}, downsample={downsample}, "
        f"K={nIter_max}, max_nMagnets={max_nMagnets})"
    )
    fB_hist, absBn_hist, _m_history = GPMO(pm_opt, algorithm=algorithm, **kwargs)

    run_id = f"lowres_nphi{nphi}_ds{downsample}_mat{material_name}_K{nIter_max}_nmax{max_nMagnets}_{algorithm}"
    run_csv = write_runhistory(out_subdir, run_id, pm_opt, fB_hist, absBn_hist)

    fB_final = compute_final_fB(s_plot=s, bs=bs, pm_opt=pm_opt)
    fB_last_hist = float(fB_hist[-1]) if len(fB_hist) else float("nan")
    print(f"[INFO] f_B (last history)   = {fB_last_hist:.16e}")
    print(f"[INFO] f_B (recomputed)    = {fB_final:.16e}")

    yaml_path = write_run_yaml(
        out_subdir,
        run_id,
        algorithm=algorithm,
        current_scale=current_scale,
        metrics={
            "fB_last_history": fB_last_hist,
            "fB_recomputed": fB_final,
        },
    )

    # final dipoles (for later integration / comparison)
    npz_path = out_subdir / f"dipoles_final_{run_id}.npz"
    np.savez(
        npz_path,
        xyz=pm_opt.dipole_grid_xyz,
        m=pm_opt.m.reshape(pm_opt.ndipoles, 3),
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )

    # Also write a FAMUS-style file and a VTK file for quick visualization.
    pm_opt.write_to_famus(out_subdir)
    b_final = DipoleField(
        pm_opt.dipole_grid_xyz,
        pm_opt.m,
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )
    b_final._toVTK(out_subdir / f"dipoles_final_{run_id}", dx, dy, dz)

    print(f"[INFO] Wrote {run_csv}")
    print(f"[INFO] Wrote {yaml_path}")
    print(f"[INFO] Wrote {npz_path}")


out_dir = Path(default_out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

mag_data = FocusData(famus_filename, downsample=downsample)
pol_vectors = build_pol_vectors(mag_data)

m_maxima, cube_dim = compute_m_maxima_and_cube_dim(famus_filename, downsample, float(material["B_max"]))

# Read in the plasma equilibrium / boundary surface.
s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
_ = (s_inner, s_outer)  # reserved for future extensions / comparisons

# Initialize the coils (scaled if requested by the chosen material).
base_curves, curves, coils = initialize_coils_for_pm_optimization("muse_famus", TEST_DIR, s, out_dir)
_ = (base_curves, curves)

current_scale = 1.0
if bool(material["scale_coils"]):
    from simsopt.field import Coil

    bs2 = BiotSavart(coils)
    B0 = float(calculate_modB_on_major_radius(bs2, s))
    target_B0 = float(material["target_B0"])
    current_scale = target_B0 / max(B0, 1e-30)
    coils = [Coil(c.curve, c.current * current_scale) for c in coils]

    # Verify:
    bs2 = BiotSavart(coils)
    B0_check = float(calculate_modB_on_major_radius(bs2, s))
    print(f"[INFO] B0 before: {B0:.6g} T, scale: {current_scale:.6g}, B0 after: {B0_check:.6g} T")

bs = BiotSavart(coils)
calculate_modB_on_major_radius(bs, s)

# Set up correct Bnormal from TF coils on the optimization surface.
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

run_one(
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
)
run_one(
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
)

t_end = time.time()
print(f"\nTotal time = {t_end - t_start:.3f} s")
