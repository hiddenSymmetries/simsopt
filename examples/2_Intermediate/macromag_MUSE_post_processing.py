from pathlib import Path
import math
import numpy as np

from simsopt.field import DipoleField, BiotSavart
from simsopt.solve.macromag import muse2tiles, MacroMag
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.util.permanent_magnet_helper_functions import *


TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
muse_grid_filename = TEST_DIR / "magtense_zot80_3d.csv"
famus_filename = TEST_DIR / "input.muse"
coil_path = TEST_DIR / "muse_tf_coils.focus"

mu_ea = 1.05
mu_oa = 1.15

# High resolution for smooth surface plots
nphi = 64
ntheta = nphi
## USING NFP=1 since grid from Towered MUSE setup is already set..... i.e no repeating portion....  -> Any comments or other points on this let me know. 
nfp = 1  # number of field periods for MUSE


def unit(v, eps=1e-30):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


def angle_deg(a, b, eps=1e-30):
    an, bn = unit(a, eps), unit(b, eps)
    c = np.clip(np.sum(an * bn, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def Bn_from_dipoles(centers, moments, pts, normals):
    dip = DipoleField(centers, moments)
    dip.set_points(pts)
    B = dip.B()
    return np.sum(B * normals, axis=1)


# Build tiles from MUSE CSV
tiles = muse2tiles(muse_grid_filename, (mu_ea, mu_oa))

# Uncoupled MUSE magnetization
M0 = tiles.M_rem[:, None] * tiles.u_ea

# Full physical dimensions and volumes
full_sizes = tiles.size 
vol = np.prod(full_sizes, axis=1)

# Per tile box sizes for VTK glyphs
dx_tile = full_sizes[:, 0]
dy_tile = full_sizes[:, 1]
dz_tile = full_sizes[:, 2]

mac = MacroMag(tiles)
mac.load_coils(coil_path)

# Full demagnetization tensor (note sign)
N_full = -mac.fast_get_demag_tensor(cache=True)

# Case (ii): magnet magnet coupling only
mac, A_mm = mac.direct_solve(
    use_coils=False,
    N_new_rows=N_full,
    print_progress=False,
    A_prev=None,
)
M_mm = tiles.M.copy()

# Case (iii): magnet magnet and coil coupling
mac, A_full = mac.direct_solve(
    use_coils=True,
    N_new_rows=N_full,
    print_progress=False,
    A_prev=None,
)
M_mc = tiles.M.copy()

# Magnet statistics
theta_mm = angle_deg(M_mm, tiles.u_ea)
theta_mc = angle_deg(M_mc, tiles.u_ea)

mag0 = np.linalg.norm(M0, axis=1)
mag_mm = np.linalg.norm(M_mm, axis=1)
mag_mc = np.linalg.norm(M_mc, axis=1)

dmag_mm = mag_mm - mag0
dmag_mc = mag_mc - mag0

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


# Magnet-only B·n diagnostics on a closed full-torus surface
qphi_bn = 2 * nphi
quadpoints_phi_bn = np.linspace(0.0, 1.0, qphi_bn, endpoint=True)
quadpoints_theta_bn = np.linspace(0.0, 1.0, ntheta, endpoint=True)

surf_bn = SurfaceRZFourier.from_focus(
    famus_filename,
    quadpoints_phi=quadpoints_phi_bn,
    quadpoints_theta=quadpoints_theta_bn,
    range="full torus",
)

PTS_BN = surf_bn.gamma().reshape(-1, 3)
NORM_BN = surf_bn.unitnormal().reshape(-1, 3)
nphi_grid, ntheta_grid = qphi_bn, ntheta

centers = tiles.offset.copy()

m_uncoupled = M0 * vol[:, None]
m_mag_only = M_mm * vol[:, None]
m_mag_coil = M_mc * vol[:, None]

Bn_uncoupled = Bn_from_dipoles(centers, m_uncoupled, PTS_BN, NORM_BN)
Bn_mag_only = Bn_from_dipoles(centers, m_mag_only, PTS_BN, NORM_BN)
Bn_mag_coil = Bn_from_dipoles(centers, m_mag_coil, PTS_BN, NORM_BN)

# Relative change
delta_Bn = Bn_mag_coil - Bn_uncoupled
rel = (np.abs(delta_Bn) / 0.15).max()
print("\nRelative error between uncoupled and fully coupled magnet fields")
print(f"max(||ΔBn||/0.15) = {rel:.2%}")

change_Bn = Bn_uncoupled - Bn_mag_coil


def as_field3(a2):
    return np.ascontiguousarray(a2, dtype=np.float64)[:, :, None]


Bn_uncoupled_grid = Bn_uncoupled.reshape(nphi_grid, ntheta_grid)
Bn_mag_only_grid = Bn_mag_only.reshape(nphi_grid, ntheta_grid)
Bn_mag_coil_grid = Bn_mag_coil.reshape(nphi_grid, ntheta_grid)

dBn_grid = change_Bn.reshape(nphi_grid, ntheta_grid)
dBn_abs = np.abs(dBn_grid)

extra_data = {
    "Bn_pre": as_field3(Bn_uncoupled_grid),
    "Bn_post": as_field3(Bn_mag_coil_grid),
    "Bn_mag_only": as_field3(Bn_mag_only_grid),
    "Bn_delta": as_field3(dBn_grid),
    "Bn_abs_delta": as_field3(dBn_abs),
}

out_dir = Path("output_permanent_magnet_GPMO_MUSE/MUSE_post_processing")
out_dir.mkdir(parents=True, exist_ok=True)
vtk_path = out_dir / "surface_Bn_delta_MUSE_post_processing"
surf_bn.to_vtk(vtk_path, extra_data=extra_data)
print(f"[SIMSOPT] Wrote {vtk_path}.vtp with fields: {', '.join(extra_data.keys())}")

# Δ(B·n) stats on unique half period wedge
change_Bn_grid = change_Bn.reshape(nphi_grid, ntheta_grid)
nphi_unique = nphi_grid // (2 * nfp)
if nphi_grid % (2 * nfp) != 0:
    raise ValueError(f"nphi={nphi_grid} must be divisible by 2*nfp={2 * nfp}.")

change_Bn_unique = change_Bn_grid[:nphi_unique, :].ravel()

print("\nΔ(B·n) summary (unique half period wedge, uncoupled minus fully coupled)")
print(f"mean = {change_Bn_unique.mean():.12e}")
print(f"max  = {np.abs(change_Bn_unique).max():.12e}")
print(f"rms  = {math.sqrt(np.mean(change_Bn_unique ** 2)):.12e}")

# f_B evaluation on 64 x 64 surface grid (half-period wedge)
B_max = 1.465 # Using same B_max as was used in MUSE stuff
mu0 = 4 * np.pi * 1e-14
M_max = B_max / mu0

m_maxima = vol * M_max
print(f"Global max m_maxima = {m_maxima.max():.3e}")

input_name = "input.muse"
surface_filename = TEST_DIR / input_name
s_half = SurfaceRZFourier.from_focus(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)

base_curves, curves, coils = initialize_coils("muse_famus", TEST_DIR, s_half, out_dir)
bs = BiotSavart(coils)

bs.set_points(PTS_BN)
B_coils_bn = bs.B().reshape((nphi_grid, ntheta_grid, 3))
NORM_BN_grid = NORM_BN.reshape((nphi_grid, ntheta_grid, 3))
Bnormal_coils_bn = np.sum(B_coils_bn * NORM_BN_grid, axis=2)

# Total B*n = (B_coils + B_magnets)*n for each case
Btotal_unc_grid = Bnormal_coils_bn + Bn_uncoupled_grid
Btotal_mm_grid = Bnormal_coils_bn + Bn_mag_only_grid
Btotal_mc_grid = Bnormal_coils_bn + Bn_mag_coil_grid

Btotal_unc_abs = np.abs(Btotal_unc_grid)
Btotal_mm_abs = np.abs(Btotal_mm_grid)
Btotal_mc_abs = np.abs(Btotal_mc_grid)

extra_error_data = {
    "Btotal_unc": as_field3(Btotal_unc_grid),
    "Btotal_mm": as_field3(Btotal_mm_grid),
    "Btotal_mc": as_field3(Btotal_mc_grid),
    "Btotal_unc_abs": as_field3(Btotal_unc_abs),
    "Btotal_mm_abs": as_field3(Btotal_mm_abs),
    "Btotal_mc_abs": as_field3(Btotal_mc_abs),
}

vtk_path_err = out_dir / "surface_Btotal_error_MUSE_post_processing"
surf_bn.to_vtk(vtk_path_err, extra_data=extra_error_data)
print(
    f"[SIMSOPT] Wrote {vtk_path_err}.vtp with fields: "
    f"{', '.join(extra_error_data.keys())}"
)

# HEre we have the summary of total Bn error on unique half-period wedge
Btotal_unc_unique = Btotal_unc_grid[:nphi_unique, :].ravel()
Btotal_mm_unique = Btotal_mm_grid[:nphi_unique, :].ravel()
Btotal_mc_unique = Btotal_mc_grid[:nphi_unique, :].ravel()

def summarize_error(name, arr):
    print(f"\n{name} total B·n error (unique half period wedge)")
    print(f"mean = {arr.mean():.12e}")
    print(f"max  = {np.abs(arr).max():.12e}")
    print(f"rms  = {math.sqrt(np.mean(arr ** 2)):.12e}")

summarize_error("Uncoupled", Btotal_unc_unique)
summarize_error("Magnet-magnet only", Btotal_mm_unique)
summarize_error("Magnet-magnet + coil", Btotal_mc_unique)

qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta,
)

# Initial Bnormal on plasma surface from unoptimized coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pts_fb = s_plot.gamma().reshape(-1, 3)

Bnormal = np.sum(
    bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2
)

# Uncoupled case
b_dipole_uncoupled = DipoleField(
    centers,
    m_uncoupled,
    nfp=nfp,
    coordinate_flag="cartesian",
    m_maxima=m_maxima,
)
b_dipole_uncoupled.set_points(pts_fb)
fB_uncoupled = SquaredFlux(s_plot, b_dipole_uncoupled, -Bnormal).J()

# Magnet magnet coupling only
b_dipole_mag_only = DipoleField(
    centers,
    m_mag_only,
    nfp=nfp,
    coordinate_flag="cartesian",
    m_maxima=m_maxima,
)
b_dipole_mag_only.set_points(pts_fb)
fB_mag_only = SquaredFlux(s_plot, b_dipole_mag_only, -Bnormal).J()

# Magnet magnet and coil coupling
b_dipole_mag_coil = DipoleField(
    centers,
    m_mag_coil,
    nfp=nfp,
    coordinate_flag="cartesian",
    m_maxima=m_maxima,
)
b_dipole_mag_coil.set_points(pts_fb)
fB_mag_coil = SquaredFlux(s_plot, b_dipole_mag_coil, -Bnormal).J()

print("\nf_B on Nphi = Ntheta = ", nphi)
print("case (i) MUSE solution without coupling:")
print(f"f_B (uncoupled M0)                 = {fB_uncoupled:.12e}")
print("\ncase (ii) magnet magnet coupling only (no coil coupling in macromag solve):")
print(f"f_B (magnet magnet only)           = {fB_mag_only:.12e}")
print("\ncase (iii) magnet magnet and coil coupling:")
print(f"f_B (magnet magnet + coil)         = {fB_mag_coil:.12e}")

rel_change_mag_only = (fB_uncoupled - fB_mag_only) / fB_uncoupled * 100.0
rel_change_mag_coil = (fB_uncoupled - fB_mag_coil) / fB_uncoupled * 100.0

print("\nRelative change in f_B compared to uncoupled M0")
print(f"magnet magnet only          = {rel_change_mag_only:.2f}%")
print(f"magnet magnet + coil        = {rel_change_mag_coil:.2f}%")

vtk_uncoupled = out_dir / "dipoles_uncoupled"
vtk_mag_only = out_dir / "dipoles_magnet_only"
vtk_coupled = out_dir / "dipoles_coupled"

# Write voxel VTK with per tile box sizes using the new helper
b_dipole_uncoupled._toVTK_boxes_per_tile(vtk_uncoupled, dx_tile, dy_tile, dz_tile)
b_dipole_mag_only._toVTK_boxes_per_tile(vtk_mag_only, dx_tile, dy_tile, dz_tile)
b_dipole_mag_coil._toVTK_boxes_per_tile(vtk_coupled, dx_tile, dy_tile, dz_tile)

print(f"[SIMSOPT] Wrote {vtk_uncoupled}.vtu for uncoupled magnets")
print(f"[SIMSOPT] Wrote {vtk_mag_only}.vtu for magnets with magnet magnet coupling only")
print(f"[SIMSOPT] Wrote {vtk_coupled}.vtu for magnets with magnet magnet and coil coupling")
