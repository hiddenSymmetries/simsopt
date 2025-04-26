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
import simsoptpp as sopp
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *
# from pyevtk.hl import polyLinesToVTK, pointsToVTK

t_start = time.time()

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 2
    nIter_max = 100
    nBacktracking = 50
    max_nMagnets = 20
    downsample = 100  # downsample the FAMUS grid of magnets by this factor
else:
    nphi = 32  # >= 64 for high-resolution runs
    nIter_max = 10000
    downsample = 2

ntheta = nphi  # same as above
dr = 0.01  # Radial extent in meters of the cylindrical permanent magnet bricks
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

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

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

# pol_vectors is only used for the greedy algorithms with cartesian coordinate_flag
# which is the default, so no need to specify it here.
kwargs = {"downsample": downsample, "dr": dr}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)
# mt = mag_data.mt
# mp = mag_data.mp
# m0 = mag_data.M_0
# mx = m0 * np.cos(mp)*np.sin(mt)
# my = m0 * np.sin(mp)*np.sin(mt)
# mz = m0 * np.cos(mt)
# m = np.array([mx, my, mz]).T
# force_matrix = pm_opt.net_force_matrix(m)
# print(force_matrix)
# positions = pm_opt.dipole_grid_xyz
# Fx = np.ascontiguousarray(force_matrix[:,0])
# Fy = np.ascontiguousarray(force_matrix[:,1])
# Fz = np.ascontiguousarray(force_matrix[:,2])
# x = np.ascontiguousarray(positions[:,0])
# y = np.ascontiguousarray(positions[:,1])
# z = np.ascontiguousarray(positions[:,2])
# data = {'Forces':(Fx,Fy,Fz)}
# pointsToVTK('MUSE_Force_visualization',x, y, z, data = data)
# print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
algorithm = 'baseline'  # Algorithm to use
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = nIter_max  # Maximum number of GPMO iterations to run

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
iterations = np.linspace(0, nIter_max, len(R2_history), endpoint=False)
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
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = True
if save_plots:
    # Save the MSE history and history of the m vectors
    np.savetxt(
        out_dir / f"mhistory_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}.txt",
        m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1)
    )
    np.savetxt(
        out_dir / f"R2history_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}.txt",
        R2_history
    )
    # Plot the SIMSOPT GPMO solution
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")
    print(m_history.shape)
    # Look through the solutions as function of K and make plots
    for k in range(0, m_history.shape[-1], 100):
        #mk = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
        mk = m_history[:, :, k]
        print(mk.shape)
        #Find indices where there are and aren't dipole moments
        mk_nonzero_indices = np.where(np.sum(mk ** 2, axis=-1) > 1e-10)[0]
        mk_zero_indices = np.where(np.sum(mk ** 2, axis=-1) <= 1e-10)[0]
        #Do net force calcs where there are nonzero dipole moments and make a list
        t_force_calc_start = time.time()
        net_forces_nonzero = sopp.net_force_matrix(
                np.ascontiguousarray(mk[mk_nonzero_indices, :]), 
                np.ascontiguousarray(pm_opt.dipole_grid_xyz[mk_nonzero_indices, :])
            )
        net_forces = np.zeros((pm_opt.ndipoles, 3))
        net_forces[mk_nonzero_indices, :] = net_forces_nonzero
        net_forces[mk_zero_indices, :] = 0.0
        t_force_calc_end = time.time()
        print('Time to calc force = ', t_force_calc_end - t_force_calc_start)
        #Do net torque calcs where there are nonzero dipole moments and make a list
        t_torque_calc_start = time.time()
        net_torques_nonzero = sopp.net_torque_matrix(
                np.ascontiguousarray(mk[mk_nonzero_indices, :]), 
                np.ascontiguousarray(pm_opt.dipole_grid_xyz[mk_nonzero_indices, :])
            )
        net_torques = np.zeros((pm_opt.ndipoles, 3))
        net_torques[mk_nonzero_indices, :] = net_torques_nonzero
        net_torques[mk_zero_indices, :] = 0.0
        t_torque_calc_end = time.time()
        print('Time to calc torque = ', t_torque_calc_end - t_torque_calc_start)
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            mk,
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
            net_forces=net_forces,
            net_torques = net_torques
        )
        
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        K_save = int(kwargs['K'] / kwargs['nhistory'] * k)
        b_dipole._toVTK(out_dir / f"Dipole_Fields_K{K_save}_nphi{nphi}_ntheta{ntheta}")
        #print("Total fB = ", 0.5 * np.sum((pm_opt.A_obj @ mk - pm_opt.b_obj) ** 2))
        Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
        Bnormal_total = Bnormal + Bnormal_dipoles

        # For plotting Bn on the full torus surface at the end with just the dipole fields
        make_Bnormal_plots(b_dipole, s_plot, out_dir, "only_m_optimized_K{K_save}_nphi{nphi}_ntheta{ntheta}")
        pointData = {"B_N": Bnormal_total[:, :, None]}
        s_plot.to_vtk(out_dir / "m_optimized_K{K_save}_nphi{nphi}_ntheta{ntheta}", extra_data=pointData)
        
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
# b_dipole = DipoleField(
#     pm_opt.dipole_grid_xyz,
#     pm_opt.m,
#     nfp=s.nfp,
#     coordinate_flag=pm_opt.coordinate_flag,
#     m_maxima=pm_opt.m_maxima,
#     net_forces=pm_opt.net_force_matrix(pm_opt.m),
# )
# b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
# bs.set_points(s_plot.gamma().reshape((-1, 3)))
# Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
# f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
# print('f_B = ', f_B_sf)
# total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
# print('Total volume = ', total_volume)

t_end = time.time()
print('Total time = ', t_end - t_start)
# plt.show()
