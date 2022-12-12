#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving the 
permanent magnet optimization on the NCSX grid.

The script should be run as:
    mpirun -n 1 python permanent_magnet_GPMO_NCSX.py

"""

import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField, ToroidalField
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
nphi = 64  # need to set this to 64 for a real run
ntheta = 64  # same as above
dr = 0.02
coff = 0.02
poff = 0.1
surface_flag = 'wout'
input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# NCSX grid uses imaginary coils...
# Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
net_poloidal_current_Amperes = 3.7713e+6
mu0 = 4 * np.pi * 1e-7
RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
bs = ToroidalField(R0=1, B0=RB)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid(
    s, coil_offset=coff, dr=dr, plasma_offset=poff,
    Bn=Bnormal, surface_flag='focus',
    filename=surface_filename,
    coordinate_flag='cylindrical',
    famus_filename='init_orient_pm_nonorm_5E4_q4_dp.focus'
)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
print('NCSX surface area = ', s_plot.area())
algorithm = 'baseline'  # 'backtracking'
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = 35000  # Number of magnets to place... 50000 for a full run perhaps
#kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz  # grid data needed for backtracking
#kwargs['backtracking'] = 100  # frequency with which to backtrack
#kwargs['Nadjacent'] = 100  # Number of neighbor dipoles to consider as adjacent
kwargs['nhistory'] = 500  # Number of neighbor dipoles to consider as adjacent

# Make the output directory
OUT_DIR = 'output_permanent_magnet_GPMO_NCSX_' + algorithm + '/'
os.makedirs(OUT_DIR, exist_ok=True)

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# optionally save the whole solution history
# np.savetxt(OUT_DIR + 'mhistory_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1))
np.savetxt(OUT_DIR + 'R2history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', R2_history)

# Note backtracking uses num_nonzeros since many magnets get removed 
plt.figure()
if algorithm != "backtracking":
    plt.semilogy(R2_history)
else:
    plt.semilogy(pm_opt.num_nonzeros, R2_history[1:])
plt.grid(True)
plt.xlabel('K')
plt.ylabel('$f_B$')
plt.savefig(OUT_DIR + 'GPMO_MSE_history.png')

plt.figure()
if algorithm != "backtracking":
    plt.semilogy(Bn_history)
else:
    plt.semilogy(pm_opt.num_nonzeros, Bn_history[1:])
plt.grid(True)
plt.xlabel('K')
plt.ylabel('$<|B_n|>$')
plt.savefig(OUT_DIR + 'GPMO_absBn_history.png')

mu0 = 4 * np.pi * 1e-7
Bmax = 1.465
vol_eff = np.sum(np.sqrt(np.sum(m_history ** 2, axis=1)), axis=0) * mu0 * 2 * s.nfp / Bmax
np.savetxt(OUT_DIR + 'eff_vol_history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', vol_eff)

# Plot the MSE history versus the effective magnet volume
plt.figure()
if algorithm != "backtracking":
    plt.semilogy(vol_eff, R2_history)
else:
    plt.semilogy(vol_eff[:len(pm_opt.num_nonzeros) + 1], R2_history)
plt.grid(True)
plt.xlabel('$V_{eff}$')
plt.ylabel('$f_B$')
plt.savefig(OUT_DIR + 'GPMO_Volume_MSE_history.png')

# Solution is the m vector that minimized the fb
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Plot the SIMSOPT GPMO solution 
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")
Nnorms = np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1)))

for k in range(0, len(R2_history), 5):
    pm_opt.m = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    print("Total fB = ",
          0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
    print("Total <|Bn|> = ",
          np.sum(np.abs(pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) * np.sqrt(Nnorms / len(pm_opt.b_obj))))

    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles
    # For plotting Bn on the full torus surface at the end with just the dipole fields
    #make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    #pointData = {"B_N": Bnormal_total[:, :, None]}
    #s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)), extra_data=pointData)

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

# Compute the full Bfield and average it over the plasma surface
Bfield = bs + b_dipole
Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
Bmag = np.sqrt(np.sum(Bfield.B().reshape((qphi * ntheta, 3)) ** 2, axis=-1))

# repeat for Bn
abs_Bnormal = np.ravel(abs(np.sum(Bfield.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)))

Nnorms = np.ravel(np.sqrt(np.sum(s_plot.normal() ** 2, axis=-1)))
Ngrid = qphi * ntheta
print('<|Bn|> = ', np.sum(abs_Bnormal * Nnorms) / Ngrid)
print('<|B|> = ', np.sum(Bmag * Nnorms) / Ngrid)
print('<|B|^2> = ', np.sum(Bmag ** 2 * Nnorms) / Ngrid)
print('<|Bn|> / <|B|> = ', np.sum(abs_Bnormal * Nnorms) / np.sum(Bmag * Nnorms))

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

    # Make the QFM surfaces
    t1 = time.time()
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
plt.show()
