#!/usr/bin/env python
r"""
This simple example script allows the user to explore building
permanent magnet configurations for the Landreman/Paul QA design with
basic cartesian brick magnets.

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
from simsopt.geo import PermanentMagnetGrid
from simsopt.solve import GPMO 
import time
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters
comm = None
nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 64
dr = 0.01
poff = 0.05
coff = 0.12
input_name = 'input.LandremanPaul2021_QA'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory
OUT_DIR = 'permanent_magnet_QA_output_nphi' + str(nphi) + '/'
os.makedirs(OUT_DIR, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('qa', TEST_DIR, OUT_DIR, s)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

# optimize the currents in the TF coils
s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, 'qa')
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# check after-optimization average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid(
    s, coil_offset=coff, dr=dr, plasma_offset=poff,
    Bn=Bnormal, coordinate_flag='toroidal',
    filename=surface_filename,
)
print(pm_opt.plasma_offset, pm_opt.coil_offset, pm_opt.dr)

kwargs = initialize_default_kwargs('GPMO')
algorithm = 'baseline'
kwargs['K'] = 40000  # Number of magnets to place... 50000 for a full run perhaps
kwargs['verbose'] = True
#kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz  # grid data needed for backtracking
#kwargs['backtracking'] = 200  # frequency with which to backtrack
#kwargs['Nadjacent'] = 1  # Number of neighbor dipoles to consider as adjacent
#kwargs['nhistory'] = 100  # Number of neighbor dipoles to consider as adjacent

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Plot the sparse and less sparse solutions from SIMSOPT
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields")

# Print optimized metrics
print("Total fB = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# For plotting Bn on the full torus surface at the end with just the dipole fields
# Do optimization on pre-made grid of dipoles
make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(OUT_DIR + "m_optimized", extra_data=pointData)

for k in range(0, len(R2_history), 100):
    pm_opt.m = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    print("Total fB = ",
          0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles
    # For plotting Bn on the full torus surface at the end with just the dipole fields
    make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)), extra_data=pointData)

# Print optimized f_B and other metrics
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

# write solution to FAMUS-type file
write_pm_optimizer_to_famus(OUT_DIR, pm_opt)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from mpi4py import MPI
    from simsopt.util.mpi import MpiPartition
    from simsopt.mhd.vmec import Vmec
    mpi = MpiPartition(ngroups=4)
    comm = MPI.COMM_WORLD

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
plt.show()
