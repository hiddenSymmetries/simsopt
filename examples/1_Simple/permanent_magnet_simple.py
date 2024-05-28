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
    mpirun -n 1 python permanent_magnet_simple.py
on a cluster machine but 
    python permanent_magnet_simple.py
is sufficient on other machines. Note that the code is 
parallelized via OpenMP and XSIMD, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

This script is substantially simplified to provide a simple
illustration of performing permanent magnet optimization in SIMSOPT.
"""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from simsopt.field import DipoleField, ToroidalField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.solve import GPMO
from simsopt.util.permanent_magnet_helper_functions import *

nphi = 16  # change to 64 for a real run
ntheta = 16
input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
famus_filename = 'init_orient_pm_nonorm_5E4_q4_dp.focus'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
famus_filename = TEST_DIR / famus_filename
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
net_poloidal_current_Amperes = 3.7713e+6
mu0 = 4 * np.pi * 1e-7
RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
bs = ToroidalField(R0=1, B0=RB)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Finally, initialize the permanent magnet class
kwargs_geo = {"coordinate_flag": "cylindrical"}  # , "downsample": downsample}
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(
    s, Bnormal, famus_filename, **kwargs_geo
)
print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = 40000
kwargs['nhistory'] = 100

# Optimize the permanent magnets greedily
R2_history, Bn_history, m_history = GPMO(pm_opt, **kwargs)

b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
b_dipole.set_points(s.gamma().reshape((-1, 3)))
Bnormal_dipoles = np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
pointData = {"B_N": Bnormal[:, :, None]}
s.to_vtk("CoilsBn", extra_data=pointData)
pointData = {"B_N": Bnormal_dipoles[:, :, None]}
s.to_vtk("MagnetsBn", extra_data=pointData)
pointData = {"B_N": Bnormal_total[:, :, None]}
s.to_vtk("TotalBn", extra_data=pointData)

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

pm_opt.write_to_famus()

plt.show()
