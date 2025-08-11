#!/usr/bin/env python
r"""
This simple example script allows the user to explore building
permanent magnet configurations for the Landreman/Paul QA design with
basic cylindrical brick magnets.

For realistic designs, please see the script files at
https://github.com/akaptano/simsopt_permanent_magnet_advanced_scripts.git
which can generate all of the results in the recent relax-and-split
and GPMO permanent magnet optimization papers:
    
    A. A. Kaptanoglu, R. Conlin, and M. Landreman, 
    Greedy permanent magnet optimization, 
    Nuclear Fusion 63, 036016 (2023)
    
    A. A. Kaptanoglu, T. Qian, F. Wechsung, and M. Landreman. 
    Permanent-Magnet Optimization for Stellarators as Sparse Regression.
    Physical Review Applied 18, no. 4 (2022): 044006.

This example uses the relax-and-split algorithm for 
high-dimensional sparse regression. See the other examples 
for using the greedy GPMO algorithm to solve the problem.

The script should be run as:
    mpirun -n 1 python permanent_magnet_QA.py
on a cluster machine but 
    python permanent_magnet_QA.py
is sufficient on other machines. Note that this code does not use MPI, but is 
parallelized via OpenMP and XSIMD, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).
"""

import time
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import relax_and_split
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    dr = 0.05  # cylindrical bricks with radial extent 5 cm
else:
    nphi = 16  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = 16
    dr = 0.02  # cylindrical bricks with radial extent 2 cm

coff = 0.1  # PM grid starts offset ~ 10 cm from the plasma surface
poff = 0.05  # PM grid end offset ~ 15 cm from the plasma surface
input_name = 'input.LandremanPaul2021_QA_lowres'

# Read in the plas/ma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_projected_normal(poff)
s_outer.extend_via_projected_normal(poff + coff)

# Make the output directory
out_dir = Path("permanent_magnet_QA_output")
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils
bs = coil_optimization(s, bs, base_curves, curves, out_dir)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# check after-optimization average on-axis magnetic field strength
calculate_modB_on_major_radius(bs, s)

# Set up correct Bnormal from TF coils
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Finally, initialize the permanent magnet class
kwargs_geo = {"dr": dr, "coordinate_flag": "cylindrical"}
pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
    s, Bnormal, s_inner, s_outer, **kwargs_geo
)

reg_l0 = 0.05  # Threshold off magnets with 5% or less strength
nu = 1e10  # how strongly to make proxy variable w close to values in m

# Rescale the hyperparameters and then add contributions to ATA and ATb
reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu)

# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs()
kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
kwargs['max_iter'] = 10  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = 10  # Number of total iterations of the relax-and-split algorithm
kwargs['reg_l0'] = reg_l0

# Optimize the permanent magnets. This actually solves
# 2 full relax-and-split problems, and uses the result of each
# problem to initialize the next, increasing L0 threshold each time,
# until thresholding over all magnets with strengths < 50% the max.
m0 = np.zeros(pm_opt.ndipoles * 3)
total_m_history = []
total_mproxy_history = []
total_RS_history = []
for i in range(2):
    print('Relax-and-split iteration ', i, ', L0 threshold = ', reg_l0)
    reg_l0_scaled = reg_l0 * (i + 1) / 2.0
    kwargs['reg_l0'] = reg_l0_scaled
    RS_history, m_history, m_proxy_history = relax_and_split(pm_opt, m0=m0, **kwargs)
    total_RS_history.append(RS_history)
    total_m_history.append(m_history)
    total_mproxy_history.append(m_proxy_history)
    m0 = pm_opt.m

total_RS_history = np.ravel(np.array(total_RS_history))
print('Done optimizing the permanent magnet object')

# Try to make a mp4 movie of the optimization progress
try:
    make_optimization_plots(total_RS_history, total_m_history, total_mproxy_history, pm_opt, out_dir)
except ValueError:
    print(
        'Attempted to make a mp4 of optimization progress but ValueError was raised. '
        'This is probably an indication that a mp4 python writer was not available for use.'
    )

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Plot the sparse and less sparse solutions from SIMSOPT
b_dipole_proxy = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m_proxy,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
b_dipole_proxy.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole_proxy._toVTK(out_dir / "Dipole_Fields_Sparse")
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / "Dipole_Fields")

# Print optimized metrics
print("Total fB = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
print("Total fB (sparse) = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m_proxy - pm_opt.b_obj) ** 2))

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
Bnormal_dipoles_proxy = np.sum(b_dipole_proxy.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total_proxy = Bnormal + Bnormal_dipoles_proxy

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(b_dipole, s_plot, out_dir, "only_m_optimized")
make_Bnormal_plots(b_dipole_proxy, s_plot, out_dir, "only_m_proxy_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "m_optimized", extra_data=pointData)
pointData = {"B_N": Bnormal_total_proxy[:, :, None]}
s_plot.to_vtk(out_dir / "m_proxy_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
total_volume_sparse = np.sum(np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume for m and m_proxy = ', total_volume, total_volume_sparse)
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m_proxy,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
f_B_sp = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B_sparse = ', f_B_sp)
dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
num_nonzero_sparse = np.count_nonzero(np.sum(dipoles ** 2, axis=-1)) / pm_opt.ndipoles * 100

# write solution to FAMUS-type file
pm_opt.write_to_famus(out_dir)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    mpi = MpiPartition(ngroups=1)

    # Make the QFM surface
    t1 = time.time()
    Bfield = bs + b_dipole
    Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    Bfield_proxy.set_points(s_plot.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s_plot, Bfield)
    qfm_surf = qfm_surf.surface

    # repeat QFM calculation for the proxy solution
    Bfield_proxy = bs + b_dipole_proxy
    qfm_surf_proxy = make_qfm(s, Bfield_proxy)
    qfm_surf_proxy = qfm_surf_proxy.surface
    qfm_surf_proxy.plot()
    qfm_surf_proxy = qfm_surf
    t2 = time.time()
    print("Making the two QFM surfaces took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()

    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

    ### Always use the QH VMEC file for the proxy solution and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QH_reactorScale_lowres"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf_proxy
    equil.run()

t_end = time.time()
print('Total time = ', t_end - t_start)
# plt.show()
