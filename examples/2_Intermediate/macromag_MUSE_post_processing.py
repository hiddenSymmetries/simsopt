#!/usr/bin/env python

from pathlib import Path
import math
import numpy as np

from simsopt.field import DipoleField
from simsopt.solve.macromag import muse2tiles, MacroMag
from simsopt.geo import SurfaceRZFourier

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
muse_grid_filename = TEST_DIR / "magtense_zot80_3d.csv"
famus_filename     = TEST_DIR / "input.muse"
coil_path = TEST_DIR / "muse_tf_coils.focus"

mu_ea = 1.05
mu_oa = 1.15

# High res to get smooth plots
N_THETA, N_ZETA = 64, 256


def unit(v, eps=1e-30):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


def angle_deg(a, b, eps=1e-30):
    an, bn = unit(a, eps), unit(b, eps)
    c = np.clip(np.sum(an * bn, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def surface_from_focus(path):
    nphi   = N_ZETA    # toroidal resolution
    ntheta = N_THETA   # poloidal resolution

    s = SurfaceRZFourier.from_focus(
        path,
        nphi=nphi,
        ntheta=ntheta,
        range="full torus",
    )

    # Simsopt returns arrays of shape (nphi, ntheta, 3)
    pts  = s.gamma().reshape(-1, 3)
    try:
        n_hat = s.unitnormal().reshape(-1, 3)  # preferred
    except AttributeError:
        n_hat = unit(s.normal().reshape(-1, 3))  # fallback

    shape = (nphi, ntheta)  # keep this consistent with gamma() layout
    return s, pts, n_hat, shape

def Bn_from_dipoles(centers, moments, pts, normals):
    dip = DipoleField(centers, moments)
    dip.set_points(pts)
    B = dip.B()
    return np.sum(B * normals, axis=1)


tiles = muse2tiles(muse_grid_filename, (mu_ea, mu_oa))
M0 = tiles.M_rem[:, None] * tiles.u_ea
vol = np.prod(tiles.size, axis=1)

mac = MacroMag(tiles)
mac.load_coils(coil_path)

# Precompute full demag tensor. Convention note:
# fast_get_demag_tensor() returns -N (since H_d = -N M), so flip the sign.
N_full = -mac.fast_get_demag_tensor(cache=True)   # shape: (n, n, 3, 3)

mac, A_full = mac.direct_solve(
    use_coils=True,
    N_new_rows=N_full,      # (n, n, 3, 3)
    print_progress=False,
    A_prev=None)

M1 = tiles.M.copy()

theta_after = angle_deg(M1, tiles.u_ea)
mag0, mag1 = np.linalg.norm(M0, axis=1), np.linalg.norm(M1, axis=1)
dmag = mag1 - mag0

print("\nTilt and magnitude changes")
print(f"mean Δθ [deg] = {theta_after.mean():.12f}")
print(f"max  Δθ [deg] = {theta_after.max():.12f}")
print(f"mean Δ|M| [A/m] = {np.abs(dmag).mean():.12f}")
print(f"max  Δ|M| [A/m] = {np.abs(dmag).max():.12f}")

surf, PTS, NORM, shape = surface_from_focus(famus_filename)
m0, m1 = M0 * vol[:, None], M1 * vol[:, None]
centers = tiles.offset.copy()
Bn0, Bn1 = Bn_from_dipoles(centers, m0, PTS, NORM), Bn_from_dipoles(centers, m1, PTS, NORM)

delta_Bn = Bn1 - Bn0
rel = (np.abs(delta_Bn) / 0.15).max()
print("\nRelative error")
print(f"max(‖ΔBn‖/0.15) = {rel:.2%}")

change_Bn = Bn0 - Bn1

def as_field3(a2):
    """(nphi, ntheta) -> contiguous (nphi, ntheta, 1) float64"""
    return np.ascontiguousarray(a2, dtype=np.float64)[:, :, None]

nphi, ntheta = shape
Bn0_grid   = Bn0.reshape(nphi, ntheta)
Bn1_grid   = Bn1.reshape(nphi, ntheta)
dBn_grid   = change_Bn.reshape(nphi, ntheta)      # pre - post 
dBn_abs    = np.abs(dBn_grid)

extra_data = {
    "Bn_pre":       as_field3(Bn0_grid),
    "Bn_post":      as_field3(Bn1_grid),
    "Bn_delta":     as_field3(dBn_grid),         # delta (B·n) = pre − post
    "Bn_abs_delta": as_field3(dBn_abs),
}

out_dir = Path("output_permanent_magnet_GPMO_MUSE")
out_dir.mkdir(parents=True, exist_ok=True)
vtk_path = out_dir / "surface_Bn_delta_MUSE_post_processing"
surf.to_vtk(vtk_path, extra_data=extra_data)
print(f"[SIMSOPT] Wrote {vtk_path}.vtp with fields: {', '.join(extra_data.keys())}")

print("\nΔ(B·n) summary")
print(f"mean = {change_Bn.mean():.12e}")
print(f"max  = {np.abs(change_Bn).max():.12e}")
print(f"rms  = {math.sqrt(np.mean(change_Bn**2)):.12e}")
