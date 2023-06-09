#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving permanent 
magnet optimization on the NCSX grid. This 
algorithm is described in the following paper:
    A. A. Kaptanoglu, R. Conlin, and M. Landreman, 
    Greedy permanent magnet optimization, 
    Nuclear Fusion 63, 036016 (2023)

The script should be run as:
    mpirun -n 1 python permanent_magnet_NCSX.py
on a cluster machine but 
    python permanent_magnet_NCSX.py
is sufficient on other machines. Note that the code is 
parallelized via OpenMP and XSIMD, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

For high-resolution and more realistic designs, please see the script files at
https://github.com/akaptano/simsopt_permanent_magnet_advanced_scripts.git
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import DipoleField, ToroidalField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters -- if doing CI, lower the resolution
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
if ci:
    nphi = 4  # change to 64 for a real run
    downsample = 100  # drastically downsample the PM grid for CI
else:
    nphi = 16
    downsample = 1

ntheta = nphi
surface_flag = 'wout'
input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
famus_filename = 'init_orient_pm_nonorm_5E4_q4_dp.focus'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
famus_filename = TEST_DIR / famus_filename
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory -- warning, saved data can get big!
# On NERSC, recommended to change this directory to point to SCRATCH!
out_dir = Path("output_permanent_magnet_GPMO_NCSX")
out_dir.mkdir(parents=True, exist_ok=True)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename,
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# NCSX grid uses imaginary coils...
# Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
net_poloidal_current_Amperes = 3.7713e+6
mu0 = 4 * np.pi * 1e-7
RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
bs = ToroidalField(R0=1, B0=RB)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
kwargs_geo = {"coordinate_flag": "cylindrical", "downsample": downsample}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(
    s, Bnormal, famus_filename, **kwargs_geo
)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = 500
kwargs['nhistory'] = 20

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
iterations = np.linspace(0, kwargs['K'], len(R2_history), endpoint=False)
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

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0 
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = False 
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

    # Look through the solutions as function of K and make plots
    for k in range(0, kwargs["nhistory"] + 1, 50):
        mk = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            mk,
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
        )
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        K_save = int(kwargs['K'] / kwargs['nhistory'] * k)
        b_dipole._toVTK(out_dir / f"Dipole_Fields_K{K_save}_nphi{nphi}_ntheta{ntheta}")
        print("Total fB = ",
              0.5 * np.sum((pm_opt.A_obj @ mk - pm_opt.b_obj) ** 2))
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
    from mpi4py import MPI
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
    vmec_input = os.path.join(os.getcwd(), "..", "..", "tests", "test_files", "input.LandremanPaul2021_QA")
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

t_end = time.time()
print('Total time = ', t_end - t_start)
# plt.show()
