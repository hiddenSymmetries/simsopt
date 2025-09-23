#!/usr/bin/env python

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import relax_and_split
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *
# === NEW: MacroMag tools for post-processing ===
from simsopt.solve.macromag import MacroMag, Tiles

t_start = time.time()

# === PARAMS ===
nphi = 64
downsample = 4
ntheta = nphi
dr = 0.01                      # Radial extent in meters of the cylindrical permanent magnet bricks
input_name = 'input.muse'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / 'zot80.focus'
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory -- warning, saved data can get big!
out_dir = Path("output_permanent_magnet_GPMO_MUSE")
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

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

# pol_vectors is only used for the greedy algorithms; safe to pass through geosetup but unused by relax_and_split
kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Relax-and-split ONLY, with no sparsity (reg_l0=reg_l1=0) and lambda=0
rs_kwargs = {
    "reg_l0": 0.0,
    "reg_l1": 0.0,
}

# Run optimization
t1 = time.time()
errors, m_history, m_proxy_history = relax_and_split(pm_opt, **rs_kwargs)
t2 = time.time()
print(f'relax-and-split took t = {t2 - t1:.2f} s')

# Plot the MSE history we have (errors)
iterations = np.arange(len(errors))
plt.figure()
plt.semilogy(iterations, errors, label=r'$f_B$')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Metric values')
plt.legend()
plt.savefig(out_dir / 'relax_and_split_MSE_history.png')

# If m_history is a list/array of snapshots, pick the best per errors:
if len(errors) > 0 and hasattr(m_history, "shape") and m_history.ndim == 3:
    min_ind = int(np.argmin(errors))
    pm_opt.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = False
if save_plots:
    # Save the history if desired — guard keys that used to come from GPMO
    np.savetxt(out_dir / f"mhistory_relax_and_split_nphi{nphi}_ntheta{ntheta}.txt",
               dipoles.reshape(pm_opt.ndipoles * 3))
    np.savetxt(out_dir / f"errors_relax_and_split_nphi{nphi}_ntheta{ntheta}.txt",
               np.asarray(errors))

    # Plot the SIMSOPT relax_and_split solution
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")

    # Also write a VTK of the dipole-only Bn if you want parity with old paths, etc.
    b_tmp = DipoleField(
        pm_opt.dipole_grid_xyz,
        pm_opt.m,
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )
    b_tmp.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_dipoles = np.sum(b_tmp.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles
    make_Bnormal_plots(b_tmp, s_plot, out_dir, "only_m_optimized_relax_and_split")
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(out_dir / "m_optimized_relax_and_split", extra_data=pointData)

    # write solution to FAMUS-type file
    pm_opt.write_to_famus(out_dir)

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
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
vmec_flag = False
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

t_end = time.time()
print('Total time = ', t_end - t_start)
# plt.show()


### Armin output inspection (temporary) additions
b_final = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,                        # flat (3N,) is fine; or pm_opt.m.reshape(-1, 3)
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima         # enables normalized |m| in the VTK
)

b_final._toVTK(out_dir / f"dipoles_final_relax-and-split")
print(f"[SIMSOPT] Wrote dipoles_final_relax-and-split.vtu")

### add B \dot n output
min_res = 164
qphi_view   = max(2 * nphi,   min_res)
qtheta_view = max(2 * ntheta, min_res)
quad_phi    = np.linspace(0, 1, qphi_view,   endpoint=False)
quad_theta  = np.linspace(0, 1, qtheta_view, endpoint=False)

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

surface_fname = out_dir / f"surface_Bn_fields_relax-and-split"
s_view.to_vtk(surface_fname, extra_data=extra_data)
print(f"[SIMSOPT] Wrote {surface_fname}.vtp")

### saving for later poincare plotting
np.savez(
    out_dir / f"dipoles_final_relax-and-split.npz",
    xyz=pm_opt.dipole_grid_xyz,
    m=pm_opt.m.reshape(pm_opt.ndipoles, 3),
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
print(f"[SIMSOPT] Saved dipoles_final_relax-and-split.npz")


# === NEW: MacroMag direct-solve post-processing on the relax-and-split grid ===

# Helper funcs mirroring the CSV-based post-processing (kept local & minimal)
def _unit(v, eps=1e-30):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + eps)

def _angle_deg(a, b, eps=1e-30):
    an, bn = _unit(a, eps), _unit(b, eps)
    c = np.clip(np.sum(an * bn, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def _surface_from_focus(path, N_THETA=64, N_ZETA=256):
    s_full = SurfaceRZFourier.from_focus(
        path,
        nphi=N_ZETA,
        ntheta=N_THETA,
        range="full torus",
    )
    pts  = s_full.gamma().reshape(-1, 3)
    try:
        n_hat = s_full.unitnormal().reshape(-1, 3)
    except AttributeError:
        n_hat = _unit(s_full.normal().reshape(-1, 3))
    shape = (N_ZETA, N_THETA)
    return s_full, pts, n_hat, shape

def _Bn_from_dipoles(centers, moments, pts, normals):
    dip = DipoleField(centers, moments)
    dip.set_points(pts)
    B = dip.B()
    return np.sum(B * normals, axis=1)

# Build a MacroMag "Tiles" model from the relax-and-split solution
N = pm_opt.ndipoles
centers = pm_opt.dipole_grid_xyz.copy()

# Choose a cube size (meters) for each tile; keep constant for simplicity
cube_dim = 0.004
sizes = np.full((N, 3), cube_dim, dtype=np.float64)
rots  = np.zeros((N, 3), dtype=np.float64)

# Use relax-and-split directions as the easy axes
m_vec = pm_opt.m.reshape(N, 3)
u_ea = _unit(m_vec)  # directions only

# Remanence and anisotropic mus (match your request)
mu_ea = 1.05
mu_oa = 1.15
M_rem_value = B_max / mu0  # 1.465 T / mu0 ~ 1.1659e6 A/m

tiles = Tiles(N)
tiles.size    = sizes
tiles.offset  = centers
tiles.rot     = rots
tiles.M_rem   = np.full(N, M_rem_value)
tiles.mu_r_ea = np.full(N, mu_ea)
tiles.mu_r_oa = np.full(N, mu_oa)
tiles.u_ea    = u_ea   # this also initializes tiles.M = M_rem * u_ea

# Keep the initial magnetization M0 for comparison (hard-magnet only)
M0 = tiles.M.copy()
vol = np.prod(tiles.size, axis=1)

# Load coils, solve with MacroMag (full demag tensor path)
coil_path = TEST_DIR / "muse_tf_coils.focus"
mac = MacroMag(tiles)
mac.load_coils(coil_path)

# Convention note: fast_get_demag_tensor returns -N (since H_d = -N M), so flip sign
N_full = -mac.fast_get_demag_tensor(cache=True)

# Run direct solve with coils to get the self-consistent M
mac, A_full = mac.direct_solve(
    use_coils=True,
    N_new_rows=N_full,
    print_progress=False,
    A_prev=None
)

M1 = tiles.M.copy()

# Tilt & magnitude changes
theta_after = _angle_deg(M1, tiles.u_ea)
mag0, mag1 = np.linalg.norm(M0, axis=1), np.linalg.norm(M1, axis=1)
dmag = mag1 - mag0

print("\nTilt and magnitude changes")
print(f"mean Δθ [deg] = {theta_after.mean():.12f}")
print(f"max  Δθ [deg] = {theta_after.max():.12f}")
print(f"mean Δ|M| [A/m] = {np.abs(dmag).mean():.12f}")
print(f"max  Δ|M| [A/m] = {np.abs(dmag).max():.12f}")

# B·n pre/post using dipole equivalent (M * volume)
surf_full, PTS, NORM, shape = _surface_from_focus(surface_filename, N_THETA=64, N_ZETA=256)
m0 = (M0 * vol[:, None])
m1 = (M1 * vol[:, None])

Bn0 = _Bn_from_dipoles(centers, m0, PTS, NORM)
Bn1 = _Bn_from_dipoles(centers, m1, PTS, NORM)

delta_Bn = Bn1 - Bn0
rel = (np.abs(delta_Bn) / 0.15).max()
print("\nRelative error")
print(f"max(‖ΔBn‖/0.15) = {rel:.2%}")

# Pack fields and write a VTK with pre/post/delta
change_Bn = Bn0 - Bn1    # pre − post (matches inspiration)
nphi_vtk, ntheta_vtk = shape
Bn0_grid = Bn0.reshape(nphi_vtk, ntheta_vtk)
Bn1_grid = Bn1.reshape(nphi_vtk, ntheta_vtk)
dBn_grid = change_Bn.reshape(nphi_vtk, ntheta_vtk)
dBn_abs  = np.abs(dBn_grid)

extra_delta = {
    "Bn_pre":       as_field3(Bn0_grid),
    "Bn_post":      as_field3(Bn1_grid),
    "Bn_delta":     as_field3(dBn_grid),
    "Bn_abs_delta": as_field3(dBn_abs),
}

vtk_path = out_dir / "surface_Bn_delta_MUSE_post_processing"
surf_full.to_vtk(vtk_path, extra_data=extra_delta)
print(f"[SIMSOPT] Wrote {vtk_path}.vtp with fields: {', '.join(extra_delta.keys())}")

print("\nΔ(B·n) summary")
print(f"mean = {change_Bn.mean():.12e}")
print(f"max  = {np.abs(change_Bn).max():.12e}")
print(f"rms  = {np.sqrt(np.mean(change_Bn**2)):.12e}")
