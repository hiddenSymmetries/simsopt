"""
Post-processing for uncoupled GPMO AlNiCo magnets on MUSE grid.
Computes magnet-magnet and magnet-coil coupling corrections,
evaluates B·n and f_B before and after coupling, and writes VTK output.

Adds B·n summary logs (mean/max/rms) for magnet-only and total (coils+magnets),
and writes normalized Δ(B·n)/|B| style fields to VTK for plotting.

Author: Armin Ulrich
"""

import argparse
import math
from pathlib import Path
import numpy as np

from simsopt.field import BiotSavart, DipoleField, Coil
from simsopt.solve.macromag import MacroMag, Tiles
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.util.permanent_magnet_helper_functions import (
    initialize_coils,
    calculate_modB_on_major_radius,
)

# =====================================================================
# CONFIGURATION
# =====================================================================

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / "input.muse"
coil_path = TEST_DIR / "muse_tf_coils.focus"

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlNiCo macromagnetic post-processing (uses dipoles_final_*.npz).")
    p.add_argument(
        "--npz-path",
        type=Path,
        default=None,
        help="Path to `dipoles_final_*.npz` produced by `permanent_magnet_MUSE.py` (GPMO AlNiCo run).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_permanent_magnet_GPMO_MUSE/paper_runs/paper_04_alnico/postprocess"),
        help="Output directory for post-processing artifacts.",
    )
    return p.parse_args()


args = _parse_args()

if args.npz_path is not None:
    npz_path = args.npz_path
else:
    # Prefer the paper-run folder layout, but keep backwards-compatible fallbacks.
    npz_candidates = [
        Path("output_permanent_magnet_GPMO_MUSE/paper_runs/paper_04_alnico")
        / "dipoles_final_matAlNiCo_bt200_Nadj12_nmax40000_GPMO.npz",
        Path("output_permanent_magnet_GPMO_MUSE")
        / "dipoles_final_matAlNiCo_bt200_Nadj12_nmax40000_GPMO.npz",
    ]
    npz_path = next((p for p in npz_candidates if p.exists()), npz_candidates[0])

# AlNiCo material parameters
B_max = 0.72         # Tesla
mu_ea = 3.0
mu_oa = 3.0
target_B0 = 0.05     # Tesla (desired coil field on axis)

# Surface resolution
nphi = 64
ntheta = nphi

# Output directory
out_dir = args.outdir
out_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Helpers
# =====================================================================

def summarize_bn(name: str, arr: np.ndarray):
    """Print mean, max abs, rms for a B·n array (Tesla)."""
    arr = np.asarray(arr).ravel()
    mean = arr.mean()
    mx = np.max(np.abs(arr))
    rms = math.sqrt(np.mean(arr**2))
    print(f"\n[{name}] B·n summary (Tesla)")
    print(f"  mean = {mean:.12e}")
    print(f"  max  = {mx:.12e}")
    print(f"  rms  = {rms:.12e}")
    return mean, mx, rms

def summarize_bn_unique(name: str, grid: np.ndarray, nfp: int):
    """Same stats but restricted to unique half-period wedge."""
    nphi_grid = grid.shape[0]
    nphi_unique = nphi_grid // (2 * nfp)
    if nphi_grid % (2 * nfp) != 0:
        raise ValueError(f"nphi_grid={nphi_grid} must be divisible by 2*nfp={2*nfp}.")
    arr = grid[:nphi_unique, :].ravel()
    return summarize_bn(name + " (unique half-period wedge)", arr)

def as_field3(a2):
    return np.ascontiguousarray(a2, dtype=np.float64)[:, :, None]

# =====================================================================
# LOAD UNCOUPLED DIPOLES (.npz) + metadata needed for consistent eval
# =====================================================================

data = np.load(npz_path, allow_pickle=True)

centers_all = np.asarray(data["xyz"])
m_unc_all = np.asarray(data["m"])              # (N,3)
m_maxima_all = np.asarray(data["m_maxima"]).reshape(-1)

# Prefer metadata from file if present
nfp = int(data["nfp"]) if "nfp" in data else 1

coord_flag_raw = data["coordinate_flag"] if "coordinate_flag" in data else "cartesian"
if isinstance(coord_flag_raw, (bytes, np.bytes_)):
    coordinate_flag = coord_flag_raw.decode()
else:
    coordinate_flag = str(coord_flag_raw)

mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
vol_all = m_maxima_all / M_max

# Restrict to active magnets only (for MacroMag solves / VTK box output)
active = np.linalg.norm(m_unc_all, axis=1) > 0
centers = centers_all[active]
m_uncoupled = m_unc_all[active]
m_maxima = m_maxima_all[active]
vol = vol_all[active]

fill = np.linalg.norm(m_uncoupled, axis=1) / np.maximum(m_maxima, 1e-300)
u_ea = m_uncoupled / np.maximum(np.linalg.norm(m_uncoupled, axis=1, keepdims=True), 1e-300)
M_rem = fill * M_max

cube_dim = np.cbrt(vol)
full_sizes = np.column_stack([cube_dim, cube_dim, cube_dim])
M0 = m_uncoupled / vol[:, None]  # A/m

dx_tile, dy_tile, dz_tile = cube_dim, cube_dim, cube_dim

# =====================================================================
# BUILD Tiles STRUCTURE (MagTense-compatible)
# =====================================================================

n_tiles = len(centers)
tiles = Tiles(n_tiles)

tiles.offset   = centers
tiles.size     = full_sizes
tiles.u_ea     = u_ea
tiles.M_rem    = M_rem
tiles.mu_r_ea  = np.full(n_tiles, mu_ea)
tiles.mu_r_oa  = np.full(n_tiles, mu_oa)
tiles.M        = M0.copy()
tiles.rot      = np.zeros((n_tiles, 3))   # aligned cubes (no rotation)

# =====================================================================
# INITIALIZE MacroMag WITH COILS
# =====================================================================

mac = MacroMag(tiles)
mac.load_coils(coil_path)

# Rescale coils ONCE to get ~target_B0 on-axis
s_half = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=64, ntheta=64)
base_curves, curves, coils = initialize_coils("muse_famus", TEST_DIR, s_half, out_dir)
bs_tmp = BiotSavart(coils)
B0 = calculate_modB_on_major_radius(bs_tmp, s_half)
current_scale = target_B0 / B0

coils = [Coil(c.curve, c.current * current_scale) for c in coils]
bs = BiotSavart(coils)
print(f"[INFO] Scaled coil currents: B0 before = {B0:.4f}, after scale = {target_B0:.4f}")

# Attach scaled coils to MacroMag if applicable (distinct objects from `coils` above)
if hasattr(mac, "coils"):
    for c in mac.coils:
        c.current *= current_scale

# =====================================================================
# DEMAG TENSOR AND MACROMAG SOLVES
# =====================================================================

N_full = -mac.fast_get_demag_tensor(cache=True)

# Magnet-magnet coupling only
mac, A_mm = mac.direct_solve(
    use_coils=False,
    N_new_rows=N_full,
    print_progress=False,
    A_prev=None,
)
M_mm = tiles.M.copy()
m_mag_only = M_mm * vol[:, None]

# Magnet-magnet + coil coupling
mac, A_full = mac.direct_solve(
    use_coils=True,
    N_new_rows=N_full,
    print_progress=False,
    A_prev=None,
)
M_mc = tiles.M.copy()
m_mag_coil = M_mc * vol[:, None]

# =====================================================================
# Δ(B·n) DIAGNOSTICS (uncoupled vs fully coupled) + VTK
#   Δ(B·n) = (B·n)_unc - (B·n)_mc
#   Note: coils cancel in this difference (same coils both cases), so this is
#   identical to the total-field difference on the boundary.
# =====================================================================

qphi_bn = 2 * nphi
# Include the periodic endpoints so VTK exports (SurfaceRZFourier.to_vtk) are
# watertight in both toroidal and poloidal directions. Without the duplicated
# endpoint ring, the structured grid is an open strip that renders with a gap.
quad_phi = np.linspace(0.0, 1.0, qphi_bn, endpoint=True)
quad_theta = np.linspace(0.0, 1.0, ntheta, endpoint=True)

surf_bn = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quad_phi,
    quadpoints_theta=quad_theta,
    range="full torus",
)
PTS_BN = surf_bn.gamma().reshape(-1, 3)
NORM_BN = surf_bn.unitnormal().reshape(-1, 3)

def Bn_from_dipoles(centers_, moments_, pts_, normals_):
    dip = DipoleField(
        centers_,
        moments_,
        nfp=nfp,
        stellsym=False,
        coordinate_flag=coordinate_flag,
    )
    dip.set_points(pts_)
    return np.sum(dip.B() * normals_, axis=1)

Bn_unc = Bn_from_dipoles(centers, m_uncoupled, PTS_BN, NORM_BN)
Bn_mc  = Bn_from_dipoles(centers, m_mag_coil,  PTS_BN, NORM_BN)

nphi_grid, ntheta_grid = qphi_bn, ntheta
Bn_unc_grid = Bn_unc.reshape(nphi_grid, ntheta_grid)
Bn_mc_grid  = Bn_mc.reshape(nphi_grid, ntheta_grid)

# Δ(B·n) = unc - fully coupled
dBn_grid = (Bn_unc_grid - Bn_mc_grid)
dBn = dBn_grid.ravel()

# Summary logs for Δ(B·n)
summarize_bn("Δ(B·n) = unc - mc (full surface)", dBn)
mean_u, max_u, rms_u = summarize_bn_unique("Δ(B·n) = unc - mc", dBn_grid, nfp=nfp)

Bref = max(target_B0, 1e-30)
print(f"\n[Normalized] max |Δ(B·n)| / {target_B0:.3e} T = {np.max(np.abs(dBn_grid)) / Bref:.6e}")
print(f"[Normalized] <Δ(B·n)>_unique / {target_B0:.3e} T = {mean_u / Bref:.6e}")

extra_delta = {
    "Bn_unc":            as_field3(Bn_unc_grid),
    "Bn_mc":             as_field3(Bn_mc_grid),
    "Bn_delta":          as_field3(dBn_grid),                 # unc - mc
    "Bn_delta_norm":     as_field3(dBn_grid / Bref),
}
surf_bn.to_vtk(out_dir / "surface_Bn_delta_AlNiCo", extra_data=extra_delta)
print(f"[SIMSOPT] Wrote {out_dir}/surface_Bn_delta_AlNiCo.vts")

# =====================================================================
# f_B EVALUATION  (match pm_opt convention: unitnormal, metadata, arrays)
# =====================================================================

qphi = 2 * nphi
quadpoints_phi   = np.linspace(0.0, 1.0, qphi,   endpoint=True)
quadpoints_theta = np.linspace(0.0, 1.0, ntheta, endpoint=True)

s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta,
)

pts_fb = s_plot.gamma().reshape((-1, 3))
bs.set_points(pts_fb)

# IMPORTANT: use unitnormal()
Bnormal = np.sum(
    bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(),
    axis=2
)

centers_fb = centers_all
mmax_fb    = m_maxima_all

# Full-length moment arrays for consistent f_B evaluation
m_unc_fb = m_unc_all

m_mm_all = np.zeros_like(m_unc_all)
m_mc_all = np.zeros_like(m_unc_all)
m_mm_all[active] = m_mag_only
m_mc_all[active] = m_mag_coil

b_dip_unc = DipoleField(
    centers_fb,
    m_unc_fb,
    nfp=nfp,
    coordinate_flag=coordinate_flag,
    m_maxima=mmax_fb,
)
b_dip_mm = DipoleField(
    centers_fb,
    m_mm_all,
    nfp=nfp,
    coordinate_flag=coordinate_flag,
    m_maxima=mmax_fb,
)
b_dip_mc = DipoleField(
    centers_fb,
    m_mc_all,
    nfp=nfp,
    coordinate_flag=coordinate_flag,
    m_maxima=mmax_fb,
)

b_dip_unc.set_points(pts_fb)
b_dip_mm.set_points(pts_fb)
b_dip_mc.set_points(pts_fb)

fB_unc = SquaredFlux(s_plot, b_dip_unc, -Bnormal).J()
fB_mm  = SquaredFlux(s_plot, b_dip_mm,  -Bnormal).J()
fB_mc  = SquaredFlux(s_plot, b_dip_mc,  -Bnormal).J()

print("\n--- f_B evaluation (Nφ=Nθ={}) ---".format(nphi))
print(f"Uncoupled:               f_B = {fB_unc:.6e}")
print(f"Magnet-magnet only:      f_B = {fB_mm:.6e}")
print(f"Magnet+coil coupling:    f_B = {fB_mc:.6e}")
print(f"Δf_B (mag only): {(fB_unc - fB_mm)/fB_unc*100:.2f}%")
print(f"Δf_B (full coupling): {(fB_unc - fB_mc)/fB_unc*100:.2f}%")

# =====================================================================
# VTK EXPORTS (dipole boxes)
# =====================================================================

b_dip_unc_active = DipoleField(
    centers,
    m_uncoupled,
    nfp=nfp,
    stellsym=False,
    coordinate_flag=coordinate_flag,
    m_maxima=m_maxima,
)

b_dip_unc_active._toVTK_boxes_per_tile(out_dir / "dipoles_uncoupled_AlNiCo",
                                      dx_tile, dy_tile, dz_tile)

# For boxes, use active-only objects so dimensions match active list
b_dip_mm_active = DipoleField(
    centers,
    m_mag_only,
    nfp=nfp,
    stellsym=False,
    coordinate_flag=coordinate_flag,
    m_maxima=m_maxima,
)
b_dip_mc_active = DipoleField(
    centers,
    m_mag_coil,
    nfp=nfp,
    stellsym=False,
    coordinate_flag=coordinate_flag,
    m_maxima=m_maxima,
)

b_dip_mm_active._toVTK_boxes_per_tile(out_dir / "dipoles_mm_AlNiCo",
                                     dx_tile, dy_tile, dz_tile)
b_dip_mc_active._toVTK_boxes_per_tile(out_dir / "dipoles_mc_AlNiCo",
                                     dx_tile, dy_tile, dz_tile)

print("[SIMSOPT] Wrote dipole box VTKs in", out_dir)
print("\n[Done] Post-processing complete for AlNiCo case.")
