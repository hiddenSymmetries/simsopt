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

import argparse
from pathlib import Path
import math
import sys
import numpy as np

from simsopt.field import DipoleField, BiotSavart
from simsopt.solve.macromag import muse2tiles, MacroMag
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.util.permanent_magnet_helper_functions import *


# ============================================================================
# Configuration and file paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TEST_DIR = (REPO_ROOT / "tests" / "test_files").resolve()
muse_grid_filename = TEST_DIR / "magtense_zot80_3d.csv"  # MUSE magnet grid CSV file
famus_filename = TEST_DIR / "input.muse"  # FOCUS format plasma surface file
coil_path = TEST_DIR / "muse_tf_coils.focus"  # FOCUS format coil file

# Relative permeability values for easy axis (ea) and orthogonal axis (oa)
# These characterize the magnetic material properties
mu_ea = 1.05  # Relative permeability along easy axis
mu_oa = 1.15  # Relative permeability along orthogonal axis

# High resolution for smooth surface plots
nphi = 1024  # Number of toroidal grid points
ntheta = nphi  # Number of poloidal grid points (set equal to nphi)

# Number of field periods for MUSE
# Note: NFP=1 is used since the grid from Towered MUSE setup is already set
# (i.e., no repeating portion needs to be handled)
nfp = 2


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MUSE macromagnetic post-processing (fixed published grid).")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_permanent_magnet_GPMO_MUSE/paper_runs/paper_00_postprocess_muse_grid"),
        help="Output directory for VTK artifacts and logs.",
    )
    return p.parse_args()


args = _parse_args()
out_dir = args.outdir
out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helper functions
# ============================================================================

def unit(v, eps=1e-30):
    """
    Normalize vectors to unit length.
    
    Args:
        v: Array of shape (N, 3) containing vectors to normalize
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Normalized vectors of same shape as input
    """
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


def angle_deg(a, b, eps=1e-30):
    """
    Compute the angle in degrees between two sets of vectors.
    
    Args:
        a: Array of shape (N, 3) containing first set of vectors
        b: Array of shape (N, 3) containing second set of vectors
        eps: Small epsilon for numerical stability
        
    Returns:
        Array of shape (N,) containing angles in degrees
    """
    an, bn = unit(a, eps), unit(b, eps)
    c = np.clip(np.sum(an * bn, axis=1), -1.0, 1.0)  # Dot product, clipped for arccos
    return np.degrees(np.arccos(c))


def Bn_from_dipoles(centers, moments, pts, normals):
    """
    Compute the normal component of magnetic field (B·n) at evaluation points
    due to dipole moments.
    
    Args:
        centers: Array of shape (N, 3) containing dipole center positions
        moments: Array of shape (N, 3) containing dipole moments (m = M * V)
        pts: Array of shape (M, 3) containing evaluation point positions
        normals: Array of shape (M, 3) containing surface normal vectors at pts
        
    Returns:
        Array of shape (M,) containing B·n at each evaluation point
    """
    dip = DipoleField(centers, moments, nfp=1, stellsym=False, coordinate_flag="cartesian")
    dip.set_points(pts)
    B = dip.B()  # Magnetic field at evaluation points
    return np.sum(B * normals, axis=1)  # Dot product: B·n


# ============================================================================
# Symmetry diagnostics for magnet volumes
# ============================================================================

def check_stellarator_symmetry(centers, volumes, nfp=2, z_tol=1e-6, rel_tol=1e-3):
    """
    Check whether the magnet volumes obey stellarator, midplane, and inversion symmetries.

    Args:
        centers : (N,3) array of magnet center coordinates [x,y,z]
        volumes : (N,) array of magnet volumes
        nfp      : integer, number of field periods
        z_tol    : tolerance [m] for matching midplane symmetry in z
        rel_tol  : relative tolerance for volume equality
    """
    from scipy.spatial.transform import Rotation as R

    N = len(volumes)
    phi_period = 2 * np.pi / nfp

    print("\n[Symmetry check] --- Magnet volume symmetries ---")

    # Toroidal (nfp) periodicity test
    rot = R.from_euler("z", phi_period).as_matrix()
    rotated = (rot @ centers.T).T

    # For each magnet, find nearest partner in rotated set
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    dists, idx = tree.query(rotated, k=1)

    vol_err = np.abs(volumes - volumes[idx]) / np.maximum(volumes, 1e-20)
    print(f"  nfp={nfp} toroidal symmetry:")
    print(f"    mean |ΔV/V| = {vol_err.mean():.2e}")
    print(f"    max  |ΔV/V| = {vol_err.max():.2e}")
    print(f"    mean position mismatch = {dists.mean():.3e} m")
    print(f"    max  position mismatch = {dists.max():.3e} m")

    # Midplane (pancake) Z -> -Z symmetry
    flipped = centers.copy()
    flipped[:, 2] *= -1
    treeZ = cKDTree(centers)
    dZ, idxZ = treeZ.query(flipped, k=1)
    mask = dZ < z_tol
    if np.any(mask):
        vol_errZ = np.abs(volumes[mask] - volumes[idxZ[mask]]) / np.maximum(volumes[mask], 1e-20)
        print("  midplane (Z->-Z) symmetry:")
        print(f"    matched {mask.sum()} / {N} tiles within {z_tol:.1e} m")
        print(f"    mean |ΔV/V| = {vol_errZ.mean():.2e}")
        print(f"    max  |ΔV/V| = {vol_errZ.max():.2e}")
    else:
        print("  midplane symmetry: no matching pairs found within tolerance")

    # Combined inversion (R,φ,Z)->(R,φ+π/nfp,-Z)
    rot_half = R.from_euler("z", phi_period / 2).as_matrix()
    inv = (rot_half @ centers.T).T
    inv[:, 2] *= -1
    treeI = cKDTree(centers)
    dI, idxI = treeI.query(inv, k=1)
    vol_errI = np.abs(volumes - volumes[idxI]) / np.maximum(volumes, 1e-20)
    print("  inversion (φ+π/nfp, Z->-Z) symmetry:")
    print(f"    mean |ΔV/V| = {vol_errI.mean():.2e}")
    print(f"    max  |ΔV/V| = {vol_errI.max():.2e}")
    print(f"    mean position mismatch = {dI.mean():.3e} m")
    print(f"    max  position mismatch = {dI.max():.3e} m")

    print("-------------------------------------------------------------\n")


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
print(f"minimum volume: {vol.min()}")
print(f"maximum volume: {vol.max()}")

# Check if tiles satisfy stellerator symmetry
check_stellarator_symmetry(tiles.offset, vol, nfp=nfp)

# Per tile box sizes for VTK glyphs (used for visualization)
dx_tile = full_sizes[:, 0]  # x-dimension of each tile
dy_tile = full_sizes[:, 1]  # y-dimension of each tile
dz_tile = full_sizes[:, 2]  # z-dimension of each tile

# Initialize MacroMag solver with the tiles
mac = MacroMag(tiles)
mac.load_coils(coil_path)  # Load coil geometry for magnet-coil coupling

# Compute full demagnetization tensor
# Note: The negative sign accounts for the demagnetizing field direction
# This tensor describes how each magnet affects all other magnets
N_full = -mac.fast_get_demag_tensor(cache=True)

# ============================================================================
# Solve for coupled magnetizations
# ============================================================================

# Case (ii): Magnet-magnet coupling only (no coil coupling)
# This solves for the magnetization accounting for how magnets affect each other,
# but ignores the effect of coils on the magnets
mac, A_mm = mac.direct_solve(
    use_coils=False,  # Don't include coil coupling
    N_new_rows=N_full,  # Use full demagnetization tensor
    print_progress=False,
    A_prev=None,  # No previous solution matrix
)
M_mm = tiles.M.copy()  # Store the magnet-magnet coupled magnetization

# Case (iii): Full coupling (magnet-magnet + magnet-coil)
# This solves for the magnetization accounting for both:
# - How magnets affect each other (demagnetization)
# - How coils affect the magnets (coil field coupling)
mac, A_full = mac.direct_solve(
    use_coils=True,  # Include coil coupling
    N_new_rows=N_full,  # Use full demagnetization tensor
    print_progress=False,
    A_prev=None,  # No previous solution matrix
)
M_mc = tiles.M.copy()  # Store the fully coupled magnetization

# ============================================================================
# Analyze magnetization changes
# ============================================================================

# Compute tilt angles: how much the magnetization vector deviates from easy axis
theta_mm = angle_deg(M_mm, tiles.u_ea)  # Tilt for magnet-magnet coupling case
theta_mc = angle_deg(M_mc, tiles.u_ea)  # Tilt for full coupling case

# Compute magnetization magnitudes
mag0 = np.linalg.norm(M0, axis=1)  # Uncoupled magnetization magnitude
mag_mm = np.linalg.norm(M_mm, axis=1)  # Magnet-magnet coupled magnitude
mag_mc = np.linalg.norm(M_mc, axis=1)  # Fully coupled magnitude

# Compute changes in magnetization magnitude
dmag_mm = mag_mm - mag0  # Change for magnet-magnet coupling
dmag_mc = mag_mc - mag0  # Change for full coupling

# Print statistics comparing coupled vs uncoupled magnetizations
print("\nTilt and magnitude changes relative to uncoupled M0")
print("Magnet magnet coupling only (no coil coupling):")
print(f"mean Δθ [deg] = {theta_mm.mean():.12f}")
print(f"max  Δθ [deg] = {theta_mm.max():.12f}")
print(f"mean Δ|M| [A/m] = {np.abs(dmag_mm).mean():.12f}")
print(f"max  Δ|M| [A/m] = {np.abs(dmag_mm).max():.12f}")

print("\nMagnet magnet and coil coupling:")
print(f"mean Δθ [deg] = {theta_mc.mean():.12f}")
print(f"max  Δθ [deg] = {theta_mc.max():.12f}")
print(f"mean Δ|M| [A/m] = {np.abs(dmag_mc).mean():.12f}")
print(f"max  Δ|M| [A/m] = {np.abs(dmag_mc).max():.12f}")


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
# This is the field contribution from permanent magnets (no coils)
Bn_uncoupled = Bn_from_dipoles(centers, m_uncoupled, PTS_BN, NORM_BN)
Bn_mag_only = Bn_from_dipoles(centers, m_mag_only, PTS_BN, NORM_BN)
Bn_mag_coil = Bn_from_dipoles(centers, m_mag_coil, PTS_BN, NORM_BN)

# Compute relative change between uncoupled and fully coupled cases
# The 0.15 value is a normalization factor (likely a typical B·n scale)
delta_Bn = Bn_mag_coil - Bn_uncoupled  # Change: fully coupled - uncoupled
rel = (np.abs(delta_Bn) / 0.15).max()
print("\nRelative error between uncoupled and fully coupled magnet fields")
print(f"max(||ΔBn||/0.15) = {rel:.2%}")

# Also compute the reverse difference for statistics
# This represents how much the uncoupled solution differs from the coupled one
change_Bn = Bn_uncoupled - Bn_mag_coil  # Change: uncoupled - fully coupled


def as_field3(a2):
    """
    Convert 2D array to 3D array with singleton third dimension.
    Required for VTK field data format.
    
    Args:
        a2: 2D array of shape (nphi, ntheta)
        
    Returns:
        3D array of shape (nphi, ntheta, 1)
    """
    return np.ascontiguousarray(a2, dtype=np.float64)[:, :, None]


# Reshape B·n arrays from 1D to 2D grid for visualization and analysis
Bn_uncoupled_grid = Bn_uncoupled.reshape(nphi_grid, ntheta_grid)
Bn_mag_only_grid = Bn_mag_only.reshape(nphi_grid, ntheta_grid)
Bn_mag_coil_grid = Bn_mag_coil.reshape(nphi_grid, ntheta_grid)

# Reshape change in B·n for analysis
dBn_grid = change_Bn.reshape(nphi_grid, ntheta_grid)
dBn_abs = np.abs(dBn_grid)  # Absolute value of change

# Prepare data for VTK output
# These fields will be visualized on the plasma surface
extra_data = {
    "Bn_pre": as_field3(Bn_uncoupled_grid),  # B·n before coupling (uncoupled)
    "Bn_post": as_field3(Bn_mag_coil_grid),  # B·n after full coupling
    "Bn_mag_only": as_field3(Bn_mag_only_grid),  # B·n with magnet-magnet coupling only
    "Bn_delta_mag_only": as_field3(Bn_mag_only_grid - Bn_uncoupled_grid),  # Change in B·n (magnet-magnet coupling only - uncoupled)
    "Bn_delta": as_field3(dBn_grid),  # Change in B·n (uncoupled - fully coupled)
    "Bn_abs_delta": as_field3(dBn_abs),  # Absolute value of change
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
print(f"mean = {change_Bn_unique.mean():.12e}")
print(f"max  = {np.abs(change_Bn_unique).max():.12e}")
print(f"rms  = {math.sqrt(np.mean(change_Bn_unique ** 2)):.12e}")

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
print(f"Global max m_maxima = {m_maxima.max():.3e}")
print(f"Global min m_maxima = {m_maxima.min():.3e}")

# Load plasma surface for f_B evaluation (half-period wedge)
input_name = "input.muse"
surface_filename = TEST_DIR / input_name
s_half = SurfaceRZFourier.from_focus(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)

# Initialize coils for Biot-Savart field computation
base_curves, curves, coils = initialize_coils("muse_famus", TEST_DIR, s_half, out_dir)
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
    "Btotal_unc": as_field3(Btotal_unc_grid),  # Total B·n (uncoupled magnets)
    "Btotal_mm": as_field3(Btotal_mm_grid),  # Total B·n (magnet-magnet coupling)
    "Btotal_mc": as_field3(Btotal_mc_grid),  # Total B·n (full coupling)
    "Btotal_unc_abs": as_field3(Btotal_unc_abs),  # |Total B·n| (uncoupled)
    "Btotal_mm_abs": as_field3(Btotal_mm_abs),  # |Total B·n| (magnet-magnet)
    "Btotal_mc_abs": as_field3(Btotal_mc_abs),  # |Total B·n| (full coupling)
    "Bn_coils": as_field3(Bnormal_coils_bn),  # B·n from coils
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


def summarize_error(name, arr):
    """
    Print summary statistics for B·n error.
    
    Args:
        name: Descriptive name for the case being summarized
        arr: Array of B·n error values
    """
    print(f"\n{name} total B·n error (unique half period wedge)")
    print(f"mean = {arr.mean():.12e}")
    print(f"max  = {np.abs(arr).max():.12e}")
    print(f"rms  = {math.sqrt(np.mean(arr ** 2)):.12e}")


# Print error statistics for each coupling case
summarize_error("Uncoupled", Btotal_unc_unique)
summarize_error("Magnet-magnet only", Btotal_mm_unique)
summarize_error("Magnet-magnet + coil", Btotal_mc_unique)

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

# Generate Bnormal plots from unoptimized coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

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
print(f"f_B (uncoupled M0)                 = {fB_uncoupled:.12e}")
print("\ncase (ii) magnet magnet coupling only (no coil coupling in macromag solve):")
print(f"f_B (magnet magnet only)           = {fB_mag_only:.12e}")
print("\ncase (iii) magnet magnet and coil coupling:")
print(f"f_B (magnet magnet + coil)         = {fB_mag_coil:.12e}")

# Compute relative change in f_B
# Positive values indicate improvement (lower f_B is better)
rel_change_mag_only = (fB_uncoupled - fB_mag_only) / fB_uncoupled * 100.0
rel_change_mag_coil = (fB_uncoupled - fB_mag_coil) / fB_uncoupled * 100.0

print("\nRelative change in f_B compared to uncoupled M0")
print(f"magnet magnet only          = {rel_change_mag_only:.2f}%")
print(f"magnet magnet + coil        = {rel_change_mag_coil:.2f}%")

# ============================================================================
# Write VTK files for visualization
# ============================================================================

# Define output paths for VTK files
vtk_uncoupled = out_dir / "dipoles_uncoupled"
vtk_mag_only = out_dir / "dipoles_magnet_only"
vtk_coupled = out_dir / "dipoles_coupled"

# Write voxel VTK files with per-tile box sizes
# Each magnet is represented as a box with its actual physical dimensions
b_dipole_uncoupled._toVTK_boxes_per_tile(vtk_uncoupled, dx_tile, dy_tile, dz_tile)
b_dipole_mag_only._toVTK_boxes_per_tile(vtk_mag_only, dx_tile, dy_tile, dz_tile)
b_dipole_mag_coil._toVTK_boxes_per_tile(vtk_coupled, dx_tile, dy_tile, dz_tile)

print(f"[SIMSOPT] Wrote {vtk_uncoupled}.vtu for uncoupled magnets")
print(f"[SIMSOPT] Wrote {vtk_mag_only}.vtu for magnets with magnet magnet coupling only")
print(f"[SIMSOPT] Wrote {vtk_coupled}.vtu for magnets with magnet magnet and coil coupling")
