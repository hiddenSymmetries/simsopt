#!/usr/bin/env python
r"""
This example script uses the binary orthogonal pursuit (GPMO)
greedy algorithm for solving the permanent magnet optimization.

The script should be run as:
    mpirun -n 1 python permanent_magnet_GPMO.py

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
from simsopt.solve import relax_and_split, GPMO 
from simsopt._core import Optimizable
import pickle
import time
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters
comm = None
nphi = 64
ntheta = 64
dr = 0.01
coff = 0.1
poff = 0.02
input_name = 'input.muse'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# initialize the coils
#base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, OUT_DIR, s)

# Set up BiotSavart fields
#bs = BiotSavart(coils)
IN_DIR = "/global/cscratch1/sd/akaptano/muse_famus_toroidal_nphi" + str(nphi) + "_ntheta" + str(ntheta) + "_dr1.00e-02_coff1.00e-01_poff2.00e-02/"
bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')

# Calculate average, approximate on-axis B field strength
#calculate_on_axis_B(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
#make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

# optimize the currents in the TF coils if doing QA/QH
# s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, 'qa')
#bs.set_points(s.gamma().reshape((-1, 3)))
#Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# check after-optimization average on-axis magnetic field strength
#calculate_on_axis_B(bs, s)

# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Finally, initialize the permanent magnet class
#pm_opt = PermanentMagnetGrid(
#    s, coil_offset=coff, dr=dr, plasma_offset=poff,
#    Bn=Bnormal, surface_flag='focus',
#    filename=surface_filename,
#    coordinate_flag='toroidal',
#    famus_filename='zot80.focus'
#)
# Make a subdirectory for the optimization output

#
pickle_name = IN_DIR + "PM_optimizer_muse_famus.pickle"
pm_opt = pickle.load(open(pickle_name, "rb", -1))
pm_opt.m0 = np.zeros(pm_opt.ndipoles * 3)
pm_opt.m = np.zeros(pm_opt.ndipoles * 3)
pm_opt.m_proxy = np.zeros(pm_opt.ndipoles * 3)
pm_opt.plasma_boundary = s
#pm_opt.Bn = Bnormal

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the relax-and-split optimization
#kwargs = initialize_default_kwargs()

# Optimize the permanent magnets with relax-and-split for comparison
#R2_history, m_history, m_proxy_history = relax_and_split(pm_opt, **kwargs)

# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = 50000
kwargs['nhistory'] = 1000
#kwargs['verbose'] = False

OUT_DIR = '/global/cscratch1/sd/akaptano/permanent_magnet_GPMO_MC' + '_K' + str(kwargs['K']) + '_output/'
os.makedirs(OUT_DIR, exist_ok=True)

t1 = time.time()
# Optimize the permanent magnets greedily
R2_history, m_history = GPMO(pm_opt, algorithm='mutual_coherence', **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')
np.savetxt(OUT_DIR + 'mhistory_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1))
np.savetxt(OUT_DIR + 'R2history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', R2_history)
iterations = np.linspace(0, kwargs['K'], kwargs['nhistory'] + 1, endpoint=False)
plt.figure()
plt.semilogy(iterations, R2_history)
plt.grid(True)
plt.savefig(OUT_DIR + 'GPMO_MSE_history.png')

min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])

R2_history = np.ravel(np.array(R2_history))
print('Done optimizing the permanent magnet object')

# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")
# Plot the SIMSOPT GBPMO solution 
for k in range(0, kwargs["nhistory"] + 1, 50):
    pm_opt.m = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))

    # Print optimized metrics
    print("Total fB = ",
          0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles

    # For plotting Bn on the full torus surface at the end with just the dipole fields
    # Do optimization on pre-made grid of dipoles
    make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)), extra_data=pointData)

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

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
# This is probably worthless unless plasma
# surface is 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from mpi4py import MPI
    from simsopt.util.mpi import MpiPartition
    from simsopt.mhd.vmec import Vmec
    mpi = MpiPartition(ngroups=4)
    comm = MPI.COMM_WORLD

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
#plt.show()
