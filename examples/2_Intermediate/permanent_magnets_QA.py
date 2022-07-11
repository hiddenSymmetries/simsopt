#!/usr/bin/env python
r"""
This simple example script allows the user to explore building
permanent magnet configurations for the Landreman/Paul QA design with
basic cylindrical brick magnets.

For realistic designs, please see the full script in src/simsopt/util,
which can generate all of the results in our recent relax-and-split
permanent magnet optimization paper.

The script should be run as:
    mpirun -n 1 python permanent_magnet_QA.py

"""

import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.permanent_magnet_grid import PermanentMagnetGrid
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, ScaledCurrent, Coil, coils_via_symmetries
from simsopt.geo import curves_to_vtk
import time
from simsopt.mhd.vmec import Vmec
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters
comm = None
nphi = 16
ntheta = 16
dr = 0.02  # cylindrical bricks with radial extent 2 cm
coff = 0.1  # PM grid starts offset ~ 10 cm from the plasma surface
poff = 0.05  # PM grid end offset ~ 15 cm from the plasma surface
input_name = 'input.LandremanPaul2021_QA'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory
OUT_DIR = 'permanent_magnet_QA_output/'
os.makedirs(OUT_DIR, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('qa', TEST_DIR, OUT_DIR, s)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make higher resolution surface for plotting Bnormal
quadpoints_phi = np.linspace(0, 1, 2 * nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

# optimize the currents in the TF coils
s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, config_flag)

# check after-optimization average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

# Plot Bnormal on plasma surface from optimized BiotSavart coils
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal_plot = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s_plot, bs).J()
print('BiotSavart f_B = ', f_B_sf)
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid(
    s, coil_offset=coff, dr=dr, plasma_offset=poff,
    Bn=Bnormal,
    filename=surface_filename,
    coordinate_flag='cylindrical'
)

# Do optimization on pre-made grid of dipoles
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Set some hyperparameters for the optimization
kwargs = {}
kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
kwargs['max_iter_MwPGP'] = 100  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = 100  # Number of total iterations of the relax-and-split algorithm

reg_l0 = 0.05  # Threshold off magnets with 10% or less strength

# Optimize the permanent magnets, increasing L0 threshold each iteration
total_m_history = []
total_mproxy_history = []
total_RS_history = []
for i in range(20):
    reg_l0_scaled = reg_l0 * (1 + i / 2.0)
    kwargs['reg_l0'] = reg_l0_scaled
    RS_history, m_history, m_proxy_history = relax_and_split(pm_opt, **kwargs)
    total_RS_history.append(RS_history)
    total_m_history.append(m_history)
    total_mproxy_history.append(m_proxy_history)
    m0 = pm_opt.m

total_RS_history = np.ravel(np.array(total_RS_history))
print('Done optimizing the permanent magnet object')
make_optimization_plots(total_RS_history, total_m_history, total_mproxy_history, pm_opt, OUT_DIR)

# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Plot the sparse and less sparse solutions from SIMSOPT
m_copy = np.copy(pm_opt.m)
pm_opt.m = pm_opt.m_proxy
b_dipole_proxy = DipoleField(pm_opt)
b_dipole_proxy.set_points(s.gamma().reshape((-1, 3)))
b_dipole_proxy._toVTK(OUT_DIR + "Dipole_Fields_Sparse")
pm_opt.m = m_copy
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields")

# Print optimized metrics
print("Total fB = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
print("Total fB (sparse) = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m_proxy - pm_opt.b_obj) ** 2))

# Compute metrics with permanent magnet results
Bnormal_dipoles = np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
print('F_B INITIAL = ', SquaredFlux(s, b_dipole, -Bnormal).J())
print('F_B INITIAL * 2 * nfp = ', 2 * s.nfp * SquaredFlux(pm_opt.plasma_boundary, b_dipole, -pm_opt.Bn).J())

dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")
make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized")
make_Bnormal_plots(b_dipole_proxy, s_plot, OUT_DIR, "only_m_proxy_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(OUT_DIR + "m_optimized", extra_data=pointData)
pointData = {"B_N": Bnormal_total_proxy[:, :, None]}
s_plot.to_vtk(OUT_DIR + "m_proxy_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
total_volume_sparse = np.sum(np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume for m and m_proxy = ', total_volume, total_volume_sparse)
pm_opt.m = pm_opt.m_proxy
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
f_B_sp = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B_sparse = ', f_B_sp)
dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
num_nonzero_sparse = np.count_nonzero(np.sum(dipoles ** 2, axis=-1)) / pm_opt.ndipoles * 100

# write solution to FAMUS-type file
write_pm_optimizer_to_famus(OUT_DIR, pm_opt)

# write sparse solution to FAMUS-type file
# m_copy = np.copy(pm_opt.m)
# pm_opt.m = pm_opt.m_proxy
# write_pm_optimizer_to_famus(OUT_DIR, pm_opt)
# pm_opt.m  = m_copy

t_end = time.time()
print('Total time = ', t_end - t_start)
plt.show()
