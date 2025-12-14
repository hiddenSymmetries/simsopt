#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving permanent 
magnet optimization on the MUSE grid. This 
algorithm is described in the following paper:
    A. A. Kaptanoglu, R. Conlin, and M. Landreman, 
    Greedy permanent magnet optimization, 
    Nuclear Fusion 63, 036016 (2023)

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
"""

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

high_res_run = True

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 2
    nIter_max = 100
    nBacktracking = 0
    max_nMagnets = 20
    downsample = 100  # downsample the FAMUS grid of magnets by this factor
elif high_res_run: 
    nphi = 64
    nIter_max = 50000
    nBacktracking = 200
    max_nMagnets = 40000
    downsample = 1    
else:
    nphi = 16  # >= 64 for high-resolution runs
    nIter_max = 35000
    nBacktracking = 0
    max_nMagnets = 35000
    downsample = 1

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
out_dir = Path("output_permanent_magnet_GPMO_MUSE")
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)

scale_coils = True
current_scale = 1
if(scale_coils):
    from simsopt.field import Coil
    # compute current B0 (really B0avg on major radius)
    bs2 = BiotSavart(coils)
    B0 = calculate_modB_on_major_radius(bs2, s)   # Tesla (if geometry in meters, currents in Amps)

    # picking a target and scale currents linearly
    target_B0 = 0.5  # Tesla
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

# pol_vectors is only used for the greedy algorithms with cartesian coordinate_flag
# which is the default, so no need to specify it here.
kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
# Python+Macromag
algorithm = 'ArbVec_backtracking_macromag_py'  # Algorithm to use
# algorithm = 'ArbVec_backtracking'  # Algorithm to use
nAdjacent = 12  # How many magnets to consider "adjacent" to one another
nHistory = nIter_max
thresh_angle = np.pi - (5 * np.pi / 180)  # The angle between two "adjacent" dipoles such that they should be removed
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = nIter_max  # Maximum number of GPMO iterations to run
kwargs['nhistory'] = nHistory
if algorithm == 'backtracking' or algorithm == 'ArbVec_backtracking_macromag_py' or algorithm == 'ArbVec_backtracking':
    kwargs['backtracking'] = nBacktracking  # How often to perform the backtrackinig
    kwargs['Nadjacent'] = nAdjacent
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs['max_nMagnets'] = max_nMagnets
    if algorithm == 'ArbVec_backtracking_macromag_py' or algorithm == 'ArbVec_backtracking':
        kwargs['thresh_angle'] = thresh_angle

# Macromag branch (GPMOmr)
param_suffix = f"_bt{nBacktracking}_Nadj{nAdjacent}_nmax{max_nMagnets}"
mm_suffix = ""
if algorithm == "ArbVec_backtracking_macromag_py": 
    kwargs['cube_dim'] = 0.004
    kwargs['mu_ea'] = 1.05
    kwargs['mu_oa'] = 1.15
    kwargs['use_coils'] = True
    kwargs['use_demag'] = True
    kwargs['coil_path'] = TEST_DIR / 'muse_tf_coils.focus'
    kwargs['mm_refine_every'] = 100
    kwargs['current_scale'] = current_scale
    mm_suffix = f"_kmm{kwargs['mm_refine_every']}"

full_suffix = param_suffix + mm_suffix

    
# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
iterations = np.linspace(0, kwargs['max_nMagnets'], len(R2_history), endpoint=False)
plt.figure()
plt.semilogy(iterations, R2_history, label=r'$f_B$')
plt.semilogy(iterations, Bn_history, label=r'$<|Bn|>$')
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Metric values')
plt.legend()
plt.savefig(out_dir / 'GPMO_MSE_history.png')

# Set final m to the minimum achieved during the optimization
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])

#For solo magnet case since otherwise outputted glyphs are empty
# H = m_history.shape[2]
# filled = np.linalg.norm(m_history, axis=1).sum(axis=0) > 0   # (H,)
# last_idx = np.where(filled)[0][-1]                           
# pm_opt.m = m_history[:, :, last_idx].ravel()


# Print effective permanent magnet volume
#B_max = 1.465
B_max = 1.410  # Tesla, GB50UH 
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = True
if save_plots:
    # Save the MSE history and history of the m vectors
    H = m_history.shape[2]
    np.savetxt(
        out_dir / f"mhistory_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}{full_suffix}_{algorithm}.txt",
        m_history.reshape(pm_opt.ndipoles * 3, H)
    )
    np.savetxt(
        out_dir / f"R2history_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}{full_suffix}_{algorithm}.txt",
        R2_history
    )
    np.savetxt(
        out_dir / f"Bn_history_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}{full_suffix}_{algorithm}.txt",
        Bn_history
    )

    # Plot the SIMSOPT GPMO solution
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")

    # Only log and export the final iterate
    k_final = m_history.shape[2] - 1
    mk = m_history[:, :, k_final].reshape(pm_opt.ndipoles * 3)

    b_dipole = DipoleField(
        pm_opt.dipole_grid_xyz,
        mk,
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))

    K_save = int(kwargs['K'])  # nominal final iteration number
    b_dipole._toVTK(out_dir / f"Dipole_Fields_K{K_save}_nphi{nphi}_ntheta{ntheta}{full_suffix}_{algorithm}", dx, dy, dz)

    print("Total fB = ", 0.5 * np.sum((pm_opt.A_obj @ mk - pm_opt.b_obj) ** 2))

    # Compute and export final Bnormal fields
    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles

    # For plotting Bn on the full torus surface at the end with just the dipole fields
    make_Bnormal_plots(b_dipole, s_plot, out_dir, f"only_m_optimized_K{K_save}_nphi{nphi}{full_suffix}_ntheta{ntheta}")
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(out_dir / f"m_optimized_K{K_save}_nphi{nphi}{full_suffix}_ntheta{ntheta}", extra_data=pointData)

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
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
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


### Armin ouput inspection (temporary) additions
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
