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
  https://github.com/armulrich/simsopt/tree/GPMOmr_paper_code
"""

import argparse
import time
import csv
import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from ruamel.yaml import YAML

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.coil_optimization_helper_functions import \
    calculate_modB_on_major_radius, \
    make_qfm
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
        "B_max": 1.465,   # Tesla
        "mu_ea": 1.05,
        "mu_oa": 1.15,
        "scale_coils": False,
        "target_B0": None,
    },
    # High-coercivity NdFeB grade for higher-field studies.
    "GB50UH": {
        "B_max": 1.410,   # Tesla
        "mu_ea": 1.05,
        "mu_oa": 1.15,
        "scale_coils": True,
        "target_B0": 0.5,  # Tesla
    },
    # Low-remanence material with deliberately stronger effective permeability.
    "AlNiCo": {
        "B_max": 0.72,    # Tesla
        "mu_ea": 3.00,
        "mu_oa": 3.00,
        "scale_coils": True,
        "target_B0": 0.05,  # Tesla
    },
}

# Paper-style run presets (choose one).
# You can run the same preset twice (GPMO vs GPMOmr) by only changing `algorithm`.
RUN_PRESETS = {
    # Baseline MUSE with backtracking (used in most paper figures).
    "muse_bt": {
        "material": "N52",
        "nphi": 64,
        "nIter_max": 25000,
        "nBacktracking": 200,
        "max_nMagnets": 40000,
        "mm_refine_every": 50,
        "downsample": 1,
        "history_every": 1000,
    },
    # MUSE without backtracking (stage-two style run).
    "muse_no_bt": {
        "material": "N52",
        "nphi": 64,
        "nIter_max": 20000,
        "nBacktracking": 0,
        "max_nMagnets": 20000,
        "mm_refine_every": 50,
        "downsample": 1,
        "history_every": 1000,
    },
    # High-field study.
    "gb50uh": {
        "material": "GB50UH",
        "nphi": 64,
        "nIter_max": 50000,
        "nBacktracking": 200,
        "max_nMagnets": 40000,
        "mm_refine_every": 100,
        "downsample": 1,
        "history_every": 1000,
    },
    # Low-remanence / stronger coupling study.
    "alnico": {
        "material": "AlNiCo",
        "nphi": 64,
        "nIter_max": 25000,
        "nBacktracking": 200,
        "max_nMagnets": 40000,
        "mm_refine_every": 50,
        "downsample": 1,
        "history_every": 1000,
    },
}

DEFAULT_PRESET = "muse_bt"
DEFAULT_ALGORITHM = "GPMOmr"
DEFAULT_OUTDIR = Path("output_permanent_magnet_GPMO_MUSE")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run greedy permanent-magnet optimization (MUSE) and write run.yaml + runhistory.csv artifacts."
    )
    p.add_argument(
        "--preset",
        choices=sorted(RUN_PRESETS.keys()),
        default=DEFAULT_PRESET,
        help="Paper-style run preset (selects material + key numerical parameters).",
    )
    p.add_argument(
        "--material",
        choices=sorted(MATERIALS.keys()),
        default=None,
        help="Override preset material.",
    )
    p.add_argument(
        "--algorithm",
        default=DEFAULT_ALGORITHM,
        help="GPMO algorithm string (e.g. GPMO or GPMOmr).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for artifacts.",
    )
    p.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Disable optional diagnostic plots (default: enabled).",
    )
    p.add_argument(
        "--vmec",
        action="store_true",
        help="Run a VMEC solve using the final field (default: disabled).",
    )
    p.add_argument("--nphi", type=int, default=None, help="Override surface resolution (nphi=ntheta).")
    p.add_argument("--n-iter-max", type=int, default=None, help="Override maximum greedy iterations K.")
    p.add_argument("--n-backtracking", type=int, default=None, help="Override backtracking interval (0 disables).")
    p.add_argument("--max-n-magnets", type=int, default=None, help="Override maximum number of active magnets.")
    p.add_argument("--mm-refine-every", type=int, default=None, help="Override macromag refinement interval k_mm.")
    p.add_argument("--downsample", type=int, default=None, help="Override magnet-grid downsample factor.")
    p.add_argument("--history-every", type=int, default=None, help="Override history logging period in K.")
    p.add_argument(
        "--snapshot-k",
        type=int,
        action="append",
        default=[],
        help=(
            "Optionally save additional dipole-moment snapshots at the requested greedy iteration K. "
            "May be provided multiple times, e.g. `--snapshot-k 18000 --snapshot-k 25000`."
        ),
    )
    p.add_argument("--list-presets", action="store_true", help="Print available presets and exit.")
    p.add_argument("--list-materials", action="store_true", help="Print available materials and exit.")
    return p.parse_args()


args = _parse_args()
if args.list_presets:
    print("Available presets:")
    for name in sorted(RUN_PRESETS.keys()):
        print(f"  - {name}")
    raise SystemExit(0)
if args.list_materials:
    print("Available materials:")
    for name in sorted(MATERIALS.keys()):
        print(f"  - {name}")
    raise SystemExit(0)

save_plots = not bool(args.no_save_plots)
vmec_flag = bool(args.vmec)

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
    }
    preset_name = "ci"
else:
    preset_name = str(args.preset)
    preset = dict(RUN_PRESETS[preset_name])

if args.material is not None:
    preset["material"] = str(args.material)
if args.nphi is not None:
    preset["nphi"] = int(args.nphi)
if args.n_iter_max is not None:
    preset["nIter_max"] = int(args.n_iter_max)
if args.n_backtracking is not None:
    preset["nBacktracking"] = int(args.n_backtracking)
if args.max_n_magnets is not None:
    preset["max_nMagnets"] = int(args.max_n_magnets)
if args.mm_refine_every is not None:
    preset["mm_refine_every"] = int(args.mm_refine_every)
if args.downsample is not None:
    preset["downsample"] = int(args.downsample)
if args.history_every is not None:
    preset["history_every"] = int(args.history_every)

material_name = str(preset["material"])
material = MATERIALS[material_name]

nphi = int(preset["nphi"])
nIter_max = int(preset["nIter_max"])
nBacktracking = int(preset["nBacktracking"])
max_nMagnets = int(preset["max_nMagnets"])
downsample = int(preset["downsample"])
history_every = int(preset["history_every"])

ntheta = nphi  # same as above
dr = 0.01  # Radial extent in meters of the cylindrical permanent magnet bricks
dx = dy = dz = dr
input_name = 'input.muse'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / 'zot80.focus'
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory -- warning, saved data can get big!
# On NERSC, recommended to change this directory to point to SCRATCH!
out_dir = Path(args.outdir)
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils_for_pm_optimization('muse_famus', TEST_DIR, s, out_dir)

scale_coils = bool(material["scale_coils"])
current_scale = 1
if(scale_coils):
    from simsopt.field import Coil
    # compute current B0 (really B0avg on major radius)
    bs2 = BiotSavart(coils)
    B0 = calculate_modB_on_major_radius(bs2, s)   # Tesla (if geometry in meters, currents in Amps)

    # picking a target and scale currents linearly
    target_B0 = float(material["target_B0"])
    current_scale = target_B0 / B0

    coils = [Coil(c.curve, c.current * current_scale) for c in coils]

    # verify
    bs2 = BiotSavart(coils)
    B0_check = calculate_modB_on_major_radius(bs2, s)
    print("[INFO] B0 before:", B0, "scale:", current_scale, "B0 after:", B0_check)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(bs, s)

# Surface for final metric reporting / optional VMEC handoff.
# Use the same discretization as the optimization surface so the reported f_B
# matches the runhistory CSV (and the paper convention nphi = ntheta = 64).
qphi = nphi
s_plot = s

# Set up correct Bnormal from TF coils
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Load a downsampled version of the magnet grid from a FAMUS file
mag_data = FocusData(famus_filename, downsample=downsample)

# Set the allowable orientations of the magnets to be face-aligned
# i.e. the local x, y, or z directions
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

# Optionally add additional types of allowed orientations
PM4Stell_orientations = False
if PM4Stell_orientations:
    pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
    ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
    pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
    pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
    pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fe_ftri))
    pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
    ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
    pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
    pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
    pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fc_ftri))

ophi = np.arctan2(mag_data.oy, mag_data.ox)
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z
print('pol_vectors_shape = ', pol_vectors.shape)

# m_maxima: Here we want keep cell volume fixed and rescale by material B_max (the logic here is same cube dimensions but different material)
# This ensures the optimizer sees the correct dipole-moment cap for the chosen material.
B_ref = 1.465  # reference B_max used when generating the MUSE .focus (FAMUS) grid 
B_max = float(material["B_max"])
mu0 = 4 * np.pi * 1e-7

ox, oy, oz, Ic, M0s = np.loadtxt(
    famus_filename, skiprows=3, usecols=[3, 4, 5, 6, 7],
    delimiter=',', unpack=True
)

inds_total = np.arange(len(ox))
inds_downsampled = inds_total[::downsample]
nonzero_inds = np.intersect1d(np.ravel(np.where(Ic == 1.0)), inds_downsampled)

# M0s stores the reference dipole magnitudes; rescale to new material at fixed cell volume:
# m_maxima_new = m_maxima_ref * (B_max / B_ref)
m_maxima = M0s[nonzero_inds] * (B_max / B_ref)

# Infer the (fixed) cell volume from the reference material, then cube_dim from V = cube_dim^3:
cell_vol = M0s * mu0 / B_ref
cube_dim = float(cell_vol[0] ** (1.0 / 3.0))

# Print + verify cube dimensions and consistency with m_maxima scaling
vol_from_cube = cube_dim ** 3
rel_err_vol = abs(vol_from_cube - float(cell_vol[0])) / max(float(cell_vol[0]), 1e-300)
print(f"[INFO] MacroMag cube dims (m): {cube_dim:.8g} x {cube_dim:.8g} x {cube_dim:.8g}")
print(f"[INFO] Volume check: V_ref(from file)={float(cell_vol[0]):.8g}, V(from cube_dim^3)={vol_from_cube:.8g}, rel_err={rel_err_vol:.3e}")

# Optional sanity check: predicted full-magnet dipole magnitude from geometry + B_max should match m_maxima
mmax_pred0 = (B_max / mu0) * vol_from_cube
mmax_file0 = float(m_maxima[0]) if len(m_maxima) > 0 else np.nan
rel_err_mmax0 = abs(mmax_pred0 - mmax_file0) / max(abs(mmax_file0), 1e-300)
print(f"[INFO] m_maxima check (first active tile): mmax_pred={mmax_pred0:.8g}, mmax_from_file={mmax_file0:.8g}, rel_err={rel_err_mmax0:.3e}")

# pol_vectors is only used for the greedy algorithms with cartesian coordinate_flag
# which is the default, so no need to specify it here.
kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr, "m_maxima": m_maxima}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
# Python+Macromag
algorithm = str(args.algorithm)
nAdjacent = 12  # How many magnets to consider "adjacent" to one another
if history_every <= 0:
    raise ValueError("history_every must be > 0")
nHistory = max(1, int(nIter_max // history_every))
thresh_angle = np.pi - (5 * np.pi / 180)  # The angle between two "adjacent" dipoles such that they should be removed
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = nIter_max  # Maximum number of GPMO iterations to run
kwargs['nhistory'] = nHistory
if algorithm in ("GPMO_Backtracking", "GPMO", "GPMO_py", "GPMOmr"):
    kwargs['backtracking'] = nBacktracking  # How often to perform the backtrackinig
    kwargs['Nadjacent'] = nAdjacent
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs['max_nMagnets'] = max_nMagnets
    if algorithm in ("GPMO", "GPMO_py", "GPMOmr"):
        kwargs['thresh_angle'] = thresh_angle

# Macromag branch (GPMOmr)
param_suffix = f"_bt{nBacktracking}_Nadj{nAdjacent}_nmax{max_nMagnets}"
mm_suffix = ""
if algorithm == "GPMOmr":
    kwargs['cube_dim'] = cube_dim
    kwargs['mu_ea'] = float(material["mu_ea"])
    kwargs['mu_oa'] = float(material["mu_oa"])
    
    kwargs['use_coils'] = True
    kwargs['use_demag'] = True
    kwargs['coil_path'] = TEST_DIR / 'muse_tf_coils.focus'
    kwargs['mm_refine_every'] = int(preset["mm_refine_every"])
    kwargs['current_scale'] = current_scale
    mm_suffix = f"_kmm{kwargs['mm_refine_every']}"

mat_suffix = f"_mat{material_name}"
full_suffix = mat_suffix + param_suffix + mm_suffix


# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
if save_plots:
    k_plot = getattr(pm_opt, "k_history", None)
    if k_plot is None or len(k_plot) != len(R2_history):
        k_plot = np.arange(len(R2_history), dtype=int)
    plt.figure()
    plt.semilogy(k_plot, R2_history, label=r'$f_B$')
    plt.semilogy(k_plot, Bn_history, label=r'$<|Bn|>$')
    plt.grid(True)
    plt.xlabel('K')
    plt.ylabel('Metric values')
    plt.legend()
    plt.savefig(out_dir / 'GPMO_MSE_history.png')
    plt.close()

# Set final m to the minimum achieved during the optimization
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])

#For solo magnet case since otherwise outputted glyphs are empty
# H = m_history.shape[2]
# filled = np.linalg.norm(m_history, axis=1).sum(axis=0) > 0   # (H,)
# last_idx = np.where(filled)[0][-1]
# pm_opt.m = m_history[:, :, last_idx].ravel()


# Print effective permanent magnet volume
M_max = B_max / mu0
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics (on the same surface discretization used
# during optimization).
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
if vmec_flag:
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    mpi = MpiPartition(ngroups=1)

    # Make the QFM surfaces
    t1 = time.time()
    Bfield = bs + b_dipole
    Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s_plot, Bfield)
    qfm_surf = qfm_surf.surface
    t2 = time.time()
    print("Making the QFM surface took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()

    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

from simsopt.util import in_github_actions, proc0_print, comm_world
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)

def trace_fieldlines(bfield, label):
    t1 = time.time()
    # Set initial grid of points for field line tracing, going from
    # the magnetic axis to the surface. The actual plasma boundary is
    # at R=1.300425, but the outermost initial point is a bit inward
    # from that, R = 1.295, so the SurfaceClassifier does not think we
    # have exited the surface
    R0 = np.linspace(0.32, 0.36, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/s.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm_world is None or comm_world.rank == 0:
        particles_to_vtk(fieldlines_tys, str(out_dir / f'fieldlines_{label}'))
        plot_poincare_data(fieldlines_phi_hits, phis, str(out_dir / f'poincare_fieldline_{label}.png'), dpi=150)


# uncomment this to run tracing using the biot savart field (very slow!)
# trace_fieldlines(bs, 'bs')


# Bounds for the interpolated magnetic field chosen so that the surface is
# entirely contained in it
nfieldlines = 30
tmax_fl = 20000 
degree = 2 
n = 20
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/s.nfp, n*2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n//2)


def skip(rs, phis, zs):
    # The RegularGrindInterpolant3D class allows us to specify a function that
    # is used in order to figure out which cells to be skipped.  Internally,
    # the class will evaluate this function on the nodes of the regular mesh,
    # and if *all* of the eight corners are outside the domain, then the cell
    # is skipped.  Since the surface may be curved in a way that for some
    # cells, all mesh nodes are outside the surface, but the surface still
    # intersects with a cell, we need to have a bit of buffer in the signed
    # distance (essentially blowing up the surface a bit), to avoid ignoring
    # cells that shouldn't be ignored
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
    proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip


bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=True, skip=skip
)

bsh.set_points(s.gamma().reshape((-1, 3)))
Bh = bsh.B()
trace_fieldlines(bsh, 'bsh')

t_end = time.time()
print('Total time = ', t_end - t_start)
# plt.show()


### Complete output of key PM optimization files
# Write solution to FAMUS-type file (kept outside the optional plotting block).
pm_opt.write_to_famus(out_dir)

b_final = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,                        # flat (3N,) is fine;  or pm_opt.m.reshape(-1, 3)
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima         # enables normalized |m| in the VTK
)


b_final._toVTK(out_dir / f"dipoles_final{full_suffix}_{algorithm}", dx, dy, dz)
print(f"[SIMSOPT] Wrote dipoles_final{full_suffix}_{algorithm}.vtu")


### add B \dot n output
min_res = 164
qphi_view   = max(2 * nphi,   min_res)
qtheta_view = max(2 * ntheta, min_res)
quad_phi    = np.linspace(0, 1, qphi_view,   endpoint=True)
quad_theta  = np.linspace(0, 1, qtheta_view, endpoint=True)

s_view = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quad_phi,
    quadpoints_theta=quad_theta
)

# Evaluate fields on that surface
pts = s_view.gamma().reshape((-1, 3))
bs.set_points(pts)
b_final.set_points(pts)

# Compute B·n components
n_hat    = s_view.unitnormal().reshape(qphi_view, qtheta_view, 3)
B_coils  = bs.B().reshape(qphi_view, qtheta_view, 3)
B_mags   = b_final.B().reshape(qphi_view, qtheta_view, 3)

Bn_coils   = np.sum(B_coils * n_hat, axis=2)          # shape (qphi_view, qtheta_view)
Bn_magnets = np.sum(B_mags  * n_hat, axis=2)
Bn_target  = -Bn_coils                                # target: magnets cancel coil Bn
Bn_total   = Bn_coils + Bn_magnets
Bn_error   = Bn_magnets - Bn_target                   # = Bn_total
Bn_error_abs = np.abs(Bn_error)

def as_field3(a2):
    return np.ascontiguousarray(a2, dtype=np.float64)[:, :, None]

extra_data = {
    "Bn_total":     as_field3(Bn_total),
    "Bn_coils":     as_field3(Bn_coils),
    "Bn_magnets":   as_field3(Bn_magnets),
    "Bn_target":    as_field3(Bn_target),
    "Bn_error":     as_field3(Bn_error),
    "Bn_error_abs": as_field3(Bn_error_abs),
}

surface_fname = out_dir / f"surface_Bn_fields{full_suffix}_{algorithm}"
s_view.to_vtk(surface_fname, extra_data=extra_data)
print(f"[SIMSOPT] Wrote {surface_fname}.vtp")

### saving for later poincare plotting
np.savez(
    out_dir / f"dipoles_final{full_suffix}_{algorithm}.npz",
    xyz=pm_opt.dipole_grid_xyz,
    m=pm_opt.m.reshape(pm_opt.ndipoles, 3),
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
print(f"[SIMSOPT] Saved dipoles_final{full_suffix}_{algorithm}.npz")

# ----------------------------
# Run artifacts: run.yaml + runhistory.csv (replaces legacy *.txt histories)
# ----------------------------

run_id = f"K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}_ds{downsample}{full_suffix}_{algorithm}"
run_yaml = out_dir / f"run_{run_id}.yaml"
run_csv = out_dir / f"runhistory_{run_id}.csv"

record_every = max(1, int(kwargs["K"]) // int(kwargs["nhistory"]))
k_hist = getattr(pm_opt, "k_history", None)
if k_hist is None:
    # Fallback: use the nominal schedule (may be off by 1 for some algorithms).
    k_hist = list(range(len(R2_history)))

n_active = getattr(pm_opt, "num_nonzeros", None)
if n_active is None:
    # Fallback: compute from the returned history tensor (in-memory only).
    norm2 = np.sum(m_history * m_history, axis=1)  # (ndipoles, H)
    n_active = np.count_nonzero(norm2 > 0, axis=0)[: len(R2_history)]

rows = []
for i in range(len(R2_history)):
    rows.append(
        {
            "k": int(k_hist[i]),
            "fB": float(R2_history[i]),
            "absBn": float(Bn_history[i]),
            "n_active": int(n_active[i]),
        }
    )

with run_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["k", "fB", "absBn", "n_active"])
    w.writeheader()
    w.writerows(rows)

snapshot_entries = []
snapshot_targets = [int(k) for k in (args.snapshot_k or [])]
snapshot_targets = sorted(set(snapshot_targets))
if snapshot_targets:
    k_hist_arr = np.asarray(k_hist[: len(R2_history)], dtype=int)
    if m_history.shape[2] < len(k_hist_arr):
        raise RuntimeError(
            f"m_history length mismatch: m_history.shape[2]={m_history.shape[2]} < len(k_hist)={len(k_hist_arr)}"
        )
    for k_target in snapshot_targets:
        idx_le = np.where(k_hist_arr <= int(k_target))[0]
        idx_used = int(idx_le[-1]) if len(idx_le) else 0
        k_used = int(k_hist_arr[idx_used])

        snap_name = f"dipoles_snapshot_K{k_target}{full_suffix}_{algorithm}.npz"
        snap_path = out_dir / snap_name
        print(f"[INFO] Snapshot request K={k_target} -> using recorded K={k_used} (index {idx_used})")
        np.savez(
            snap_path,
            xyz=pm_opt.dipole_grid_xyz,
            m=m_history[:, :, idx_used],
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
            k_target=np.int64(k_target),
            k_used=np.int64(k_used),
        )
        print(f"[SIMSOPT] Saved {snap_path}")
        snapshot_entries.append({"k_target": int(k_target), "k_used": int(k_used), "dipoles_npz": snap_name})

yaml = YAML(typ="safe")
yaml.default_flow_style = False
run_doc = {
    "schema_version": 1,
    "created_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds"),
    "preset": preset_name,
    "run_id": run_id,
    "material": {
        "name": material_name,
        "B_ref_T": float(B_ref),
        "B_max_T": float(B_max),
        "mu_ea": float(material["mu_ea"]),
        "mu_oa": float(material["mu_oa"]),
    },
    "coil_scaling": {
        "enabled": bool(scale_coils),
        "target_B0_T": None if not scale_coils else float(material["target_B0"]),
        "current_scale": float(current_scale),
    },
    "grid": {
        "focus_file": str(famus_filename),
        "downsample": int(downsample),
        "ndipoles": int(pm_opt.ndipoles),
    },
    "surface": {"nphi": int(nphi), "ntheta": int(ntheta)},
    "algorithm": algorithm,
    "params": {
        "K": int(kwargs["K"]),
        "nhistory": int(kwargs["nhistory"]),
        "record_every": int(record_every),
        "history_every_requested": int(history_every),
        "save_plots": bool(save_plots),
        "vmec": bool(vmec_flag),
        "backtracking": int(kwargs.get("backtracking", 0)),
        "Nadjacent": int(kwargs.get("Nadjacent", 0)),
        "max_nMagnets": int(kwargs.get("max_nMagnets", 0)),
        "thresh_angle_rad": float(kwargs.get("thresh_angle", np.pi)),
        "mm_refine_every": int(kwargs.get("mm_refine_every", 0)),
        "use_coils": bool(kwargs.get("use_coils", False)),
        "use_demag": bool(kwargs.get("use_demag", False)),
        "coil_path": None if kwargs.get("coil_path", None) is None else str(kwargs.get("coil_path")),
        "cube_dim_m": float(kwargs.get("cube_dim", cube_dim)),
    },
    "artifacts": {
        "run_yaml": run_yaml.name,
        "runhistory_csv": run_csv.name,
        "dipoles_final_npz": f"dipoles_final{full_suffix}_{algorithm}.npz",
        "dipoles_final_vtu": f"dipoles_final{full_suffix}_{algorithm}.vtu",
        "surface_Bn_fields_vtp": f"surface_Bn_fields{full_suffix}_{algorithm}.vtp",
        "dipoles_snapshots": snapshot_entries,
    },
}
with run_yaml.open("w", encoding="utf-8") as f:
    yaml.dump(run_doc, f)

print(f"[SIMSOPT] Wrote {run_yaml}")
print(f"[SIMSOPT] Wrote {run_csv}")
