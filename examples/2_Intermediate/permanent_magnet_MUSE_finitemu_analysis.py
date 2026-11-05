#!/usr/bin/env python
"""
Post-processing script for MUSE permanent magnet analysis.

This script applies the macromagnetic refinement introduced in:

  A. Ulrich, M. Haberle, and A. A. Kaptanoglu,
  “Permanent magnet optimization of stellarators with coupling from finite
  permeability and demagnetization effects.” 2025. arXiv:2512.14997

to quantify the impact of magnet coupling on the MUSE permanent magnet grid.

It evaluates the effect of magnet–magnet and magnet–coil coupling by comparing:
(i) Uncoupled magnets (original MUSE solution)
(ii) Magnet–magnet coupling only (no coil coupling)
(iii) Full coupling (magnet–magnet + magnet–coil)

The script reports diagnostics including:
- Magnetization tilt angles and magnitude changes
- Surface B·n (magnetic field normal component) on the plasma boundary
- Total surface B·n error (coils + magnets)
- f_B objective function values
- VTK outputs for visualization
"""

import math
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve.macromagnetics import MacroMag, SolverParams, muse2tiles
from simsopt.solve.permanent_magnet_optimization import _connectivity_matrix_py
from simsopt.util.permanent_magnet_helper_functions import (
    check_magnet_volume_stellarator_symmetry,
    compute_angle_between_vectors_degrees,
    compute_normal_field_component_from_dipoles,
    initialize_coils_for_pm_optimization,
    print_bnormal_error_summary_statistics,
    reshape_to_vtk_field_format,
)
from simsopt.util import in_github_actions

# ============================================================================
# Configuration and file paths
# ============================================================================

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
muse_grid_filename = TEST_DIR / "magtense_zot80_3d.csv"  # MUSE magnet grid CSV file
famus_filename = TEST_DIR / "input.muse"  # FOCUS format plasma surface file
coil_path = TEST_DIR / "muse_tf_coils.focus"  # FOCUS format coil file

# Relative permeability values for easy axis (ea) and orthogonal axis (oa)
# These characterize the magnetic material properties
mu_ea = 1.05  # Relative permeability along easy axis
mu_oa = 1.15  # Relative permeability along orthogonal axis

# Medium resolution for smooth surface plots
nphi = 64  # Number of toroidal grid points
ntheta = nphi  # Number of poloidal grid points (set equal to nphi)

if in_github_actions:
    nphi = 8
    ntheta = 8

# Number of field periods for MUSE
# Note: NFP=1 is used since the grid from Towered MUSE setup is already set
# (i.e., no repeating portion needs to be handled)
nfp = 2


# ============================================================================
# Output directory (aligned with examples/2_Intermediate/permanent_magnet_MUSE.py)
# ============================================================================

out_base = Path(__file__).resolve().parent / "output_permanent_magnet_GPMO_MUSE" / "example_lowres_compare"
out_dir = out_base / "postprocess_muse_grid"
out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Load and initialize magnet tiles
# ============================================================================

# Build tiles from MUSE CSV file
# This loads the magnet grid positions, sizes, and orientations
tiles = muse2tiles(muse_grid_filename, (mu_ea, mu_oa))

# Case (i): Uncoupled MUSE magnetization
# This is the original MUSE solution without any coupling effects
# M0 = M_rem * u_ea (remnant magnetization along easy axis)
M0 = tiles.M_rem[:, None] * tiles.u_ea

# Full physical dimensions and volumes of each magnet tile
full_sizes = tiles.size  # Shape: (N, 3) - [dx, dy, dz] for each tile
vol = np.prod(full_sizes, axis=1)  # Volume = dx * dy * dz for each tile
print(f"minimum volume: {vol.min():.2e}")
print(f"maximum volume: {vol.max():.2e}")

# Check if tiles satisfy stellarator symmetry
check_magnet_volume_stellarator_symmetry(tiles.offset, vol, nfp=nfp)

# Per tile box sizes for VTK glyphs (used for visualization)
dx_tile = full_sizes[:, 0]  # x-dimension of each tile
dy_tile = full_sizes[:, 1]  # y-dimension of each tile
dz_tile = full_sizes[:, 2]  # z-dimension of each tile

# Initialize MacroMag solver with the tiles
mac = MacroMag(tiles)
mac.load_coils(coil_path)  # Load coil geometry for magnet-coil coupling

# Build neighbor list for near-field demag approximation.
# NOTE: direct_solve_neighbor_sparse is an approximation: the full demagnetization
# tensor is generally dense (each tile couples to all others), but we truncate to
# only the Nadjacent nearest tiles per dipole. This reduces assembly from O(N^2)
# to O(k*N) blocks and yields a sparse system, at the cost of neglecting far-field
# coupling. For grids with ~50+ neighbors per tile, results are often acceptable
# for analysis; increase Nadjacent for higher accuracy at the cost of runtime.
Nadjacent = 100
neighbors = _connectivity_matrix_py(tiles.offset, Nadjacent)

# ============================================================================
# Solve for coupled magnetizations (near-field approximation)
# ============================================================================

# Case (ii): Magnet-magnet coupling only (no coil coupling)
# Solves for magnetization with demag truncated to nearest neighbors.
mac, A_mm = mac.direct_solve_neighbor_sparse(
    neighbors=neighbors,
    params=SolverParams(use_coils=False, print_progress=True, krylov_it=200, krylov_tol=1e-10),
)
M_mm = tiles.M.copy()  # Store the magnet-magnet coupled magnetization

# Case (iii): Full coupling (magnet-magnet + magnet-coil)
# Solves for magnetization with both demag (neighbors) and coil field coupling.
mac, A_full = mac.direct_solve_neighbor_sparse(
    neighbors=neighbors,
    params=SolverParams(use_coils=True, print_progress=True, krylov_it=200, krylov_tol=1e-10),
)
M_mc = tiles.M.copy()  # Store the fully coupled magnetization

# ============================================================================
# Analyze magnetization changes
# ============================================================================

# Compute tilt angles: how much the magnetization vector deviates from easy axis
theta_mm = compute_angle_between_vectors_degrees(M_mm, tiles.u_ea)
theta_mc = compute_angle_between_vectors_degrees(M_mc, tiles.u_ea)

# Compute magnetization magnitudes
mag0 = np.linalg.norm(M0, axis=1)  # Uncoupled magnetization magnitude
mag_mm = np.linalg.norm(M_mm, axis=1)  # Magnet-magnet coupled magnitude
mag_mc = np.linalg.norm(M_mc, axis=1)  # Fully coupled magnitude

# Compute changes in magnetization magnitude
dmag_mm = mag_mm - mag0  # Change for magnet-magnet coupling
dmag_mc = mag_mc - mag0  # Change for full coupling

# Relative ΔM (normalized by M_rem per tile)
M_rem = tiles.M_rem
rel_dmag_mm = np.abs(dmag_mm) / (M_rem + 1e-30)
rel_dmag_mc = np.abs(dmag_mc) / (M_rem + 1e-30)

# Print statistics comparing coupled vs uncoupled magnetizations
print("\nTilt and magnitude changes relative to uncoupled M0")
print("Magnet magnet coupling only (no coil coupling):")
print(f"mean Δθ [deg] = {theta_mm.mean():.2e}")
print(f"max  Δθ [deg] = {theta_mm.max():.2e}")
print(f"mean Δ|M| [A/m] = {np.abs(dmag_mm).mean():.2e}")
print(f"max  Δ|M| [A/m] = {np.abs(dmag_mm).max():.2e}")
print(f"mean Δ|M|/M_rem = {100 * rel_dmag_mm.mean():.2f}%")
print(f"max  Δ|M|/M_rem = {100 * rel_dmag_mm.max():.2f}%")

print("\nMagnet magnet and coil coupling:")
print(f"mean Δθ [deg] = {theta_mc.mean():.2e}")
print(f"max  Δθ [deg] = {theta_mc.max():.2e}")
print(f"mean Δ|M| [A/m] = {np.abs(dmag_mc).mean():.2e}")
print(f"max  Δ|M| [A/m] = {np.abs(dmag_mc).max():.2e}")
print(f"mean Δ|M|/M_rem = {100 * rel_dmag_mc.mean():.2f}%")
print(f"max  Δ|M|/M_rem = {100 * rel_dmag_mc.max():.2f}%")


# ============================================================================
# Compute B·n (magnetic field normal component) on plasma surface
# ============================================================================

# Create a high-resolution surface for B·n diagnostics
# Using full torus to capture all field periods
qphi_bn = 2 * nphi  # Toroidal resolution for B·n evaluation
quadpoints_phi_bn = np.linspace(0.0, 1.0, qphi_bn, endpoint=True)
quadpoints_theta_bn = np.linspace(0.0, 1.0, ntheta, endpoint=True)

# Load plasma surface from FOCUS file
surf_bn = SurfaceRZFourier.from_focus(
    famus_filename,
    quadpoints_phi=quadpoints_phi_bn,
    quadpoints_theta=quadpoints_theta_bn,
    range="full torus",  # Full torus (not just half period)
)

# Extract surface points and normal vectors
PTS_BN = surf_bn.gamma().reshape(-1, 3)  # Surface points: shape (N, 3)
NORM_BN = surf_bn.unitnormal().reshape(-1, 3)  # Surface normals: shape (N, 3)
nphi_grid, ntheta_grid = qphi_bn, ntheta  # Grid dimensions

# Get magnet center positions
centers = tiles.offset.copy()

# Convert magnetization M [A/m] to dipole moments m = M * V [A·m²]
# Volume is needed because dipole moment = magnetization × volume
m_uncoupled = M0 * vol[:, None]  # Uncoupled dipole moments
m_mag_only = M_mm * vol[:, None]  # Magnet-magnet coupled dipole moments
m_mag_coil = M_mc * vol[:, None]  # Fully coupled dipole moments

# Compute B·n (normal component of magnetic field) from magnets only
Bn_uncoupled = compute_normal_field_component_from_dipoles(
    centers, m_uncoupled, PTS_BN, NORM_BN
)
Bn_mag_only = compute_normal_field_component_from_dipoles(
    centers, m_mag_only, PTS_BN, NORM_BN
)
Bn_mag_coil = compute_normal_field_component_from_dipoles(
    centers, m_mag_coil, PTS_BN, NORM_BN
)

# B field on axis [T] for normalizing ΔBn
B_axis = 0.15

# Compute relative ΔBn (normalized by B_axis) between uncoupled and coupled cases
delta_Bn_mm = Bn_mag_only - Bn_uncoupled  # Magnet-magnet coupling only
delta_Bn_mc = Bn_mag_coil - Bn_uncoupled  # Fully coupled
rel_dBn_mm = np.abs(delta_Bn_mm) / B_axis
rel_dBn_mc = np.abs(delta_Bn_mc) / B_axis

print("\nRelative ΔBn (ΔBn/B_axis, B_axis = 0.15 T) between uncoupled and coupled magnet fields")
print("Magnet magnet coupling only (no coil coupling):")
print(f"mean |ΔBn|/B_axis = {100 * rel_dBn_mm.mean():.2f}%")
print(f"max  |ΔBn|/B_axis = {100 * rel_dBn_mm.max():.2f}%")
print("\nMagnet magnet and coil coupling:")
print(f"mean |ΔBn|/B_axis = {100 * rel_dBn_mc.mean():.2f}%")
print(f"max  |ΔBn|/B_axis = {100 * rel_dBn_mc.max():.2f}%")

# Change: uncoupled - fully coupled (used for VTK and downstream statistics)
change_Bn = Bn_uncoupled - Bn_mag_coil


# Reshape B·n arrays from 1D to 2D grid for visualization and analysis
Bn_uncoupled_grid = Bn_uncoupled.reshape(nphi_grid, ntheta_grid)
Bn_mag_only_grid = Bn_mag_only.reshape(nphi_grid, ntheta_grid)
Bn_mag_coil_grid = Bn_mag_coil.reshape(nphi_grid, ntheta_grid)

# Reshape change in B·n for analysis
dBn_grid = change_Bn.reshape(nphi_grid, ntheta_grid)
dBn_abs = np.abs(dBn_grid)  # Absolute value of change

# Prepare data for VTK output
extra_data = {
    "Bn_pre": reshape_to_vtk_field_format(Bn_uncoupled_grid),
    "Bn_post": reshape_to_vtk_field_format(Bn_mag_coil_grid),
    "Bn_mag_only": reshape_to_vtk_field_format(Bn_mag_only_grid),
    "Bn_delta_mag_only": reshape_to_vtk_field_format(Bn_mag_only_grid - Bn_uncoupled_grid),
    "Bn_delta": reshape_to_vtk_field_format(dBn_grid),
    "Bn_abs_delta": reshape_to_vtk_field_format(dBn_abs),
}

# Write VTK file for visualization
vtk_path = out_dir / "surface_Bn_delta_MUSE_post_processing"
surf_bn.to_vtk(vtk_path, extra_data=extra_data)
print(f"[SIMSOPT] Wrote {vtk_path}.vts with fields: {', '.join(extra_data.keys())}")

# ============================================================================
# Statistics on unique half-period wedge
# ============================================================================

# Extract unique half-period wedge for statistics
# For stellarators, we only need to analyze one half-period due to symmetry
change_Bn_grid = change_Bn.reshape(nphi_grid, ntheta_grid)
nphi_unique = nphi_grid // (2 * nfp)  # Number of phi points in half period
if nphi_grid % (2 * nfp) != 0:
    raise ValueError(f"nphi={nphi_grid} must be divisible by 2*nfp={2 * nfp}.")

# Extract the unique half-period wedge
change_Bn_unique = change_Bn_grid[:nphi_unique, :].ravel()

# Print statistics on the unique half-period wedge
print("\nΔ(B·n) summary (unique half period wedge, uncoupled minus fully coupled)")
print(f"mean = {change_Bn_unique.mean():.2e}")
print(f"max  = {np.abs(change_Bn_unique).max():.2e}")
print(f"rms  = {math.sqrt(np.mean(change_Bn_unique ** 2)):.2e}")

# ============================================================================
# f_B objective function evaluation
# ============================================================================

# f_B evaluation on nphi x ntheta surface grid (half-period wedge)
# f_B measures how well the magnets cancel the coil field on the plasma surface
B_max = 1.465  # Maximum magnetic field strength [T] used in MUSE optimization

# Permeability of free space in SI units [H/m]
# BUG FIX: Changed from 1e-14 to 1e-7 (correct SI value: 4π × 10^-7)
mu0 = 4 * np.pi * 1e-7

# Maximum magnetization [A/m] corresponding to B_max
# M_max = B_max / mu0 (from B = mu0 * M for permanent magnets)
M_max = B_max / mu0

# Maximum dipole moments [A·m²] for each tile
# Used to normalize the f_B objective function
m_maxima = vol * M_max
print(f"Global max m_maxima = {m_maxima.max():.2e}")
print(f"Global min m_maxima = {m_maxima.min():.2e}")

# Load plasma surface for f_B evaluation (half-period wedge)
input_name = "input.muse"
surface_filename = TEST_DIR / input_name
s_half = SurfaceRZFourier.from_focus(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)

# Initialize coils for Biot-Savart field computation
base_curves, curves, coils = initialize_coils_for_pm_optimization(
    "muse_famus", TEST_DIR, s_half, out_dir
)
bs = BiotSavart(coils)  # Biot-Savart field from coils

# Compute coil field normal component on the full-torus surface
bs.set_points(PTS_BN)
B_coils_bn = bs.B().reshape((nphi_grid, ntheta_grid, 3))  # Coil field: shape (nphi, ntheta, 3)
NORM_BN_grid = NORM_BN.reshape((nphi_grid, ntheta_grid, 3))  # Normals: shape (nphi, ntheta, 3)
Bnormal_coils_bn = np.sum(B_coils_bn * NORM_BN_grid, axis=2)  # B_coils · n

# Total B·n = (B_coils + B_magnets) · n for each coupling case
# This is the total normal field that should be zero for perfect plasma confinement
Btotal_unc_grid = Bnormal_coils_bn + Bn_uncoupled_grid  # Uncoupled case
Btotal_mm_grid = Bnormal_coils_bn + Bn_mag_only_grid  # Magnet-magnet coupling only
Btotal_mc_grid = Bnormal_coils_bn + Bn_mag_coil_grid  # Full coupling

# Compute absolute values for error visualization
Btotal_unc_abs = np.abs(Btotal_unc_grid)
Btotal_mm_abs = np.abs(Btotal_mm_grid)
Btotal_mc_abs = np.abs(Btotal_mc_grid)

# Prepare total B·n error data for VTK output
extra_error_data = {
    "Btotal_unc": reshape_to_vtk_field_format(Btotal_unc_grid),
    "Btotal_mm": reshape_to_vtk_field_format(Btotal_mm_grid),
    "Btotal_mc": reshape_to_vtk_field_format(Btotal_mc_grid),
    "Btotal_unc_abs": reshape_to_vtk_field_format(Btotal_unc_abs),
    "Btotal_mm_abs": reshape_to_vtk_field_format(Btotal_mm_abs),
    "Btotal_mc_abs": reshape_to_vtk_field_format(Btotal_mc_abs),
    "Bn_coils": reshape_to_vtk_field_format(Bnormal_coils_bn),
}

# Write VTK file with total B·n error fields
vtk_path_err = out_dir / "surface_Btotal_error_MUSE_post_processing"
surf_bn.to_vtk(vtk_path_err, extra_data=extra_error_data)
print(
    f"[SIMSOPT] Wrote {vtk_path_err}.vts with fields: "
    f"{', '.join(extra_error_data.keys())}"
)

# ============================================================================
# Total B·n error statistics on unique half-period wedge
# ============================================================================

# Extract unique half-period wedge for error statistics
Btotal_unc_unique = Btotal_unc_grid[:nphi_unique, :].ravel()
Btotal_mm_unique = Btotal_mm_grid[:nphi_unique, :].ravel()
Btotal_mc_unique = Btotal_mc_grid[:nphi_unique, :].ravel()


# Print error statistics for each coupling case
print_bnormal_error_summary_statistics("Uncoupled", Btotal_unc_unique)
print_bnormal_error_summary_statistics("Magnet-magnet only", Btotal_mm_unique)
print_bnormal_error_summary_statistics("Magnet-magnet + coil", Btotal_mc_unique)

# ============================================================================
# Compute f_B objective function
# ============================================================================

# Create surface for f_B evaluation (full torus for proper evaluation)
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta,
)

# Set evaluation points and compute coil field normal component
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pts_fb = s_plot.gamma().reshape(-1, 3)  # Points for f_B evaluation

# Compute Bnormal from coils (this is what the magnets should cancel)
Bnormal = np.sum(
    bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2
)

# Case (i): Uncoupled magnets
# f_B measures how well magnets cancel the coil field
# SquaredFlux(surface, B_magnets, -B_coils) = ||B_magnets + B_coils||^2
b_dipole_uncoupled = DipoleField(
    centers,
    m_uncoupled,  # Uncoupled dipole moments
    nfp=1,
    stellsym=False,
    coordinate_flag="cartesian",
    m_maxima=m_maxima,  # Maximum moments for normalization
)
b_dipole_uncoupled.set_points(pts_fb)
fB_uncoupled = SquaredFlux(s_plot, b_dipole_uncoupled, -Bnormal).J()

# Case (ii): Magnet-magnet coupling only
b_dipole_mag_only = DipoleField(
    centers,
    m_mag_only,  # Magnet-magnet coupled dipole moments
    nfp=1,
    stellsym=False,
    coordinate_flag="cartesian",
    m_maxima=m_maxima,
)
b_dipole_mag_only.set_points(pts_fb)
fB_mag_only = SquaredFlux(s_plot, b_dipole_mag_only, -Bnormal).J()

# Case (iii): Full coupling (magnet-magnet + magnet-coil)
b_dipole_mag_coil = DipoleField(
    centers,
    m_mag_coil,  # Fully coupled dipole moments
    nfp=1,
    stellsym=False,
    coordinate_flag="cartesian",
    m_maxima=m_maxima,
)
b_dipole_mag_coil.set_points(pts_fb)
fB_mag_coil = SquaredFlux(s_plot, b_dipole_mag_coil, -Bnormal).J()

# Print f_B values for each case
print("\nf_B on Nphi = Ntheta = ", nphi)
print("case (i) MUSE solution without coupling:")
print(f"f_B (uncoupled M0)                 = {fB_uncoupled:.2e}")
print("\ncase (ii) magnet magnet coupling only (no coil coupling in macromag solve):")
print(f"f_B (magnet magnet only)           = {fB_mag_only:.2e}")
print("\ncase (iii) magnet magnet and coil coupling:")
print(f"f_B (magnet magnet + coil)         = {fB_mag_coil:.2e}")

# Compute relative change in f_B
# Positive values indicate improvement (lower f_B is better)
rel_change_mag_only = (fB_uncoupled - fB_mag_only) / fB_uncoupled
rel_change_mag_coil = (fB_uncoupled - fB_mag_coil) / fB_uncoupled

print("\nRelative change in f_B compared to uncoupled M0")
print(f"magnet magnet only          = {rel_change_mag_only:.2%}")
print(f"magnet magnet + coil        = {rel_change_mag_coil:.2%}")

# ============================================================================
# Write VTK files for visualization
# ============================================================================

# Define output paths for VTK files
vtk_uncoupled = out_dir / "dipoles_uncoupled"
vtk_mag_only = out_dir / "dipoles_magnet_only"
vtk_coupled = out_dir / "dipoles_coupled"

# Write voxel VTK files with per-tile box sizes
# Each magnet is represented as a box with its actual physical dimensions
b_dipole_uncoupled._toVTK(vtk_uncoupled, dx_tile, dy_tile, dz_tile)
b_dipole_mag_only._toVTK(vtk_mag_only, dx_tile, dy_tile, dz_tile)
b_dipole_mag_coil._toVTK(vtk_coupled, dx_tile, dy_tile, dz_tile)

print(f"[SIMSOPT] Wrote {vtk_uncoupled}.vtu for uncoupled magnets")
print(f"[SIMSOPT] Wrote {vtk_mag_only}.vtu for magnets with magnet magnet coupling only")
print(f"[SIMSOPT] Wrote {vtk_coupled}.vtu for magnets with magnet magnet and coil coupling")
