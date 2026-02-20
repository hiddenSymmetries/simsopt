#!/usr/bin/env python
'''
    This example sript uses the GPMO backtracking algorithm with 
    arbitrarily-defined polarization vectors to optimize magnets in an 
    arrangement around the NCSX plasma in the C09R00 configuration with 
    an average magnetic field strength on axis of about 0.5 T.
    
    This test applies the backtracking approach to the PM4Stell magnet arrangement
    with face-triplet polarizations. The threshold angle for removal of magnet 
    pairs is set by the user, unlike the normal backtracking algorithm
    which defaults to threshold_angle = 180 degrees (only remove adjacent 
    dipoles if they are exactly equal and opposite).

    PM4Stell functionality and related code obtained courtesy of
    Ken Hammond and the PM4Stell + MAGPIE teams. 
'''

from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np

from simsopt.field import BiotSavart, DipoleField, Coil
from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
from simsopt.solve import GPMO
from simsopt.objectives import SquaredFlux
from simsopt.util import initialize_default_kwargs
from simsopt.util import FocusPlasmaBnormal, FocusData, read_focus_coils, in_github_actions
from simsopt.util import polarization_axes, orientation_phi, discretize_polarizations

t_start = time.time()

# Set some parameters -- warning this is super low resolution!
if in_github_actions:
    N = 2  # >= 64 for high-resolution runs
    nIter_max = 100
    max_nMagnets = 20
    nBacktracking = 0
    downsample = 100  # drastically downsample the grid if running CI
else:
    N = 16  # >= 64 for high-resolution runs
    nIter_max = 3500
    max_nMagnets = 3500
    nBacktracking = 0
    downsample = 10
    

nphi = N
ntheta = N
algorithm = 'GPMO' # Or GPMOmr
nAdjacent = 10
thresh_angle = np.pi  # / np.sqrt(2)
nHistory = 10
angle = int(thresh_angle * 180 / np.pi)
out_dir = Path("PM4Stell") 
out_dir.mkdir(parents=True, exist_ok=True)
print('out directory = ', out_dir)

# Obtain the plasma boundary for the NCSX configuration
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
fname_plasma = TEST_DIR / 'c09r00_B_axis_half_tesla_PM4Stell.plasma'
lcfs_ncsx = SurfaceRZFourier.from_focus(
    fname_plasma, range='half period', nphi=nphi, ntheta=ntheta
)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    fname_plasma,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Obtain the normal field on the plasma boundary arising from plasma currents
bnormal_obj_ncsx = FocusPlasmaBnormal(fname_plasma)
bn_plasma = bnormal_obj_ncsx.bnormal_grid(nphi, ntheta, 'half period')

# Obtain the NCSX TF coil data and calculate their normal field on the boundary
fname_ncsx_coils = TEST_DIR / 'tf_only_half_tesla_symmetry_baxis_PM4Stell.focus'
base_curves, base_currents, ncoils = read_focus_coils(fname_ncsx_coils)
coils = []
for i in range(ncoils):
    coils.append(Coil(base_curves[i], base_currents[i]))
base_currents[0].fix_all()

# fix all the coil shapes
for i in range(ncoils):
    base_curves[i].fix_all()

# Obtain Bnormal from the plasma and the coils
ncsx_tfcoils = coils
bs_tfcoils = BiotSavart(ncsx_tfcoils)
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
bn_tfcoils = np.sum(
    bs_tfcoils.B().reshape((nphi, ntheta, 3)) * lcfs_ncsx.unitnormal(),
    axis=2
)
bn_total = bn_plasma + bn_tfcoils
bs_tfcoils.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
s_plot.to_vtk(out_dir / "biot_savart_initial", extra_data={"B_N": Bnormal[:, :, None]})

# Obtain data on the magnet arrangement
fname_argmt = TEST_DIR / 'magpie_trial104b_PM4Stell.focus'
fname_corn = TEST_DIR / 'magpie_trial104b_corners_PM4Stell.csv'
mag_data = FocusData(fname_argmt, downsample=downsample)
nMagnets_tot = mag_data.nMagnets

# Determine the allowable polarization types and reject the negatives
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

# When optimizing with GPMOmr, its adviced to disable the pole-orientation block below to reduce computational cost.
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

# Read in the phi coordinates and set the pol_vectors
ophi = orientation_phi(fname_corn)[:nMagnets_tot]
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((nMagnets_tot, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z

# Not settnig M_max such that pm_ncsx generates this using default args
kwargs_geo = {"pol_vectors": pol_vectors, "downsample": downsample}

# Initialize the permanent magnet grid from the PM4Stell arrangement
pm_ncsx = PermanentMagnetGrid.geo_setup_from_famus(
    lcfs_ncsx, bn_total, fname_argmt, **kwargs_geo
)

# Optimize with the GPMO algorithm
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = nIter_max
kwargs['nhistory'] = nHistory
if algorithm in ("GPMO_Backtracking", "GPMO", "GPMO_py", "GPMOmr"):
    kwargs['backtracking'] = nBacktracking
    kwargs['Nadjacent'] = nAdjacent
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_ncsx.dipole_grid_xyz)
    if algorithm in ("GPMO", "GPMO_py", "GPMOmr"):
        kwargs['thresh_angle'] = thresh_angle
        kwargs['max_nMagnets'] = max_nMagnets
        
        
# Macromag branch
if algorithm == "GPMOmr":
    kwargs['cube_dim'] = 0.0296
    kwargs['mu_ea'] = 1.05
    kwargs['mu_oa'] = 1.15
    kwargs['use_coils'] = True
    kwargs['use_demag'] = True
    kwargs['coil_path'] = TEST_DIR / 'muse_tf_coils.focus'
    kwargs['mm_refine_every'] = 50
    
    
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_ncsx, algorithm, **kwargs)
dt = time.time() - t1
print('GPMO took t = ', dt, ' s')
# Print effective permanent magnet volume
dipoles = pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / pm_ncsx.m_maxima)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Save files
if True:
    # Magnet dimensions for VTK output (cube_dim from m_maxima for FAMUS grid)
    B_max = 1.465
    mu0 = 4 * np.pi * 1e-7
    cube_dim = (np.mean(pm_ncsx.m_maxima) * mu0 / B_max) ** (1.0 / 3.0)
    dx = dy = dz = cube_dim
    # Make BiotSavart object from the dipoles and plot solution
    b_dipole = DipoleField(
        pm_ncsx.dipole_grid_xyz,
        pm_ncsx.m,
        nfp=s_plot.nfp,
        coordinate_flag=pm_ncsx.coordinate_flag,
        m_maxima=pm_ncsx.m_maxima,
    )
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole._toVTK(out_dir / "Dipole_Fields", dx, dy, dz)
    Bnormal_coils = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_plasma = bnormal_obj_ncsx.bnormal_grid(qphi, ntheta, 'full torus')
    Bnormal_total = Bnormal_plasma + Bnormal_coils + Bnormal_dipoles
    pointData = {"B_N": Bnormal_plasma[:, :, None]}
    s_plot.to_vtk(out_dir / "Bnormal_plasma", extra_data=pointData)
    pointData = {"B_N": Bnormal_dipoles[:, :, None]}
    s_plot.to_vtk(out_dir / "Bnormal_dipoles", extra_data=pointData)
    pointData = {"B_N": Bnormal_coils[:, :, None]}
    s_plot.to_vtk(out_dir / "Bnormal_coils", extra_data=pointData)
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(out_dir / "Bnormal_total", extra_data=pointData)
    pm_ncsx.write_to_famus(out_dir)
    b_dipole.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
    bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
    Bnormal_plasma = bnormal_obj_ncsx.bnormal_grid(nphi, ntheta, 'half period')
    f_B_sf = SquaredFlux(lcfs_ncsx, b_dipole + bs_tfcoils, -Bnormal_plasma).J()
    print('f_B = ', f_B_sf) 
    np.savetxt(out_dir / 'R2_history.txt', R2_history)
    np.savetxt(out_dir / 'absBn_history.txt', Bn_history)
    nmags = m_history.shape[0]
    nhist = m_history.shape[2]
    m_history_2d = m_history.reshape((nmags*m_history.shape[1], nhist))
    np.savetxt(out_dir / f"m_history_nmags={nmags}_nhist={nhist}.txt", m_history_2d)
t_end = time.time()
print('Script took in total t = ', t_end - t_start, ' s')

# Plot optimization results as function of iterations
plt.figure()
plt.semilogy(R2_history, label=r'$f_B$')
plt.semilogy(Bn_history, label=r'$<|Bn|>$')
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Metric values')
plt.legend()
# plt.show()

# Armin's Output

B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
dipoles = pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = True
if save_plots:
    # Save the MSE history and history of the m vectors
    H = m_history.shape[2]
    np.savetxt(
        out_dir / f"mhistory_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}_{algorithm}.txt",
        m_history.reshape(pm_ncsx.ndipoles * 3, H)
    )
    np.savetxt(
        out_dir / f"R2history_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}_{algorithm}.txt",
        R2_history
    )
    np.savetxt(out_dir / f"Bn_history_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}_{algorithm}.txt", Bn_history)

    # Plot the SIMSOPT solution
    bs_tfcoils.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    Bn = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    pointData = {"B_N": Bn[:, :, None]}
    s_plot.to_vtk(out_dir / "biot_savart_optimized", extra_data=pointData)

    # Look through the solutions as function of K and make plots
    for k in range(0, kwargs["nhistory"] + 1, 50):
        mk = m_history[:, :, k].reshape(pm_ncsx.ndipoles * 3)
        b_dipole = DipoleField(
            pm_ncsx.dipole_grid_xyz,
            mk,
            nfp=lcfs_ncsx.nfp,
            coordinate_flag=pm_ncsx.coordinate_flag,
            m_maxima=pm_ncsx.m_maxima,
        )
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        K_save = int(kwargs['K'] / kwargs['nhistory'] * k)
        b_dipole._toVTK(out_dir / f"Dipole_Fields_K{K_save}_nphi{nphi}_ntheta{ntheta}_{algorithm}", dx, dy, dz)
        print("Total fB = ", 0.5 * np.sum((pm_ncsx.A_obj @ mk - pm_ncsx.b_obj) ** 2))
        Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
        Bnormal_total = Bnormal + Bnormal_dipoles

        # For plotting Bn on the full torus surface at the end with just the dipole fields
        pointData = {"B_N": Bnormal_dipoles[:, :, None]}
        s_plot.to_vtk(out_dir / "Bnormal_dipoles_K{K_save}_nphi{nphi}_ntheta{ntheta}", extra_data=pointData)
        pointData = {"B_N": Bnormal_total[:, :, None]}
        s_plot.to_vtk(out_dir / "m_optimized_K{K_save}_nphi{nphi}_ntheta{ntheta}", extra_data=pointData)

    # write solution to FAMUS-type file
    pm_ncsx.write_to_famus(out_dir)

# Compute metrics with permanent magnet results
dipoles_m = pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_ncsx.ndipoles * 100
print("Number of possible dipoles = ", pm_ncsx.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
b_dipole = DipoleField(
    pm_ncsx.dipole_grid_xyz,
    pm_ncsx.m,
    nfp=lcfs_ncsx.nfp,
    coordinate_flag=pm_ncsx.coordinate_flag,
    m_maxima=pm_ncsx.m_maxima,
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
bs_tfcoils.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
total_volume = np.sum(np.sqrt(np.sum(pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3) ** 2, axis=-1))) * lcfs_ncsx.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)


b_final = DipoleField(
    pm_ncsx.dipole_grid_xyz,
    pm_ncsx.m,                        # flat (3N,) is fine; or pm_ncsx.m.reshape(-1, 3)
    nfp=s_plot.nfp,
    coordinate_flag=pm_ncsx.coordinate_flag,
    m_maxima=pm_ncsx.m_maxima         # enables normalized |m| in the VTK
)

# Magnet dimensions for VTK (B_max, mu0 defined above)
cube_dim_final = (np.mean(pm_ncsx.m_maxima) * mu0 / B_max) ** (1.0 / 3.0)
dx = dy = dz = cube_dim_final
b_final._toVTK(out_dir / f"dipoles_final_{algorithm}", dx, dy, dz)
print(f"[SIMSOPT] Wrote dipoles_final_{algorithm}.vtu")

# --- add B·n output ---
min_res = 164
qphi_view   = max(2 * nphi,   min_res)
qtheta_view = max(2 * ntheta, min_res)
quad_phi    = np.linspace(0, 1, qphi_view,   endpoint=False)
quad_theta  = np.linspace(0, 1, qtheta_view, endpoint=False)

s_view = SurfaceRZFourier.from_focus(
    fname_plasma,
    quadpoints_phi=quad_phi,
    quadpoints_theta=quad_theta
)

# Evaluate fields on that surface
pts = s_view.gamma().reshape((-1, 3))
bs_tfcoils.set_points(pts)
b_final.set_points(pts)

# Compute B·n components
n_hat    = s_view.unitnormal().reshape(qphi_view, qtheta_view, 3)
B_coils  = bs_tfcoils.B().reshape(qphi_view, qtheta_view, 3)
B_mags   = b_final.B().reshape(qphi_view, qtheta_view, 3)

Bn_coils   = np.sum(B_coils * n_hat, axis=2)
Bn_magnets = np.sum(B_mags  * n_hat, axis=2)
Bn_target  = -Bn_coils
Bn_total   = Bn_coils + Bn_magnets
Bn_error   = Bn_magnets - Bn_target
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

surface_fname = out_dir / f"surface_Bn_fields_{algorithm}"
s_view.to_vtk(surface_fname, extra_data=extra_data)
print(f"[SIMSOPT] Wrote {surface_fname}.vtp")

np.savez(
    out_dir / f"dipoles_final_{algorithm}.npz",
    xyz=pm_ncsx.dipole_grid_xyz,
    m=pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3),
    nfp=s_plot.nfp,
    coordinate_flag=pm_ncsx.coordinate_flag,
    m_maxima=pm_ncsx.m_maxima,
)
print(f"[SIMSOPT] Saved dipoles_final_{algorithm}.npz")
