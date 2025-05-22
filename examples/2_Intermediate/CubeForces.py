#!/usr/bin/env python

import time
import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
import simsoptpp as sopp
from simsopt.util import FocusData, discretize_polarizations, polarization_axes
from simsopt.util.permanent_magnet_helper_functions import *

# Set number of quadrature points - make them consistent
nphi = 32  # Number of phi quadrature points
ntheta = 32  # Number of theta quadrature points
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=False)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=False)

dr = 0.01  # Radial extent in meters of the cylindrical permanent magnet bricks


out_dir = Path("CubeForces_Output")
out_dir.mkdir(parents=True, exist_ok=True)

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / 'MagnetForceTest.focus'

#taking muse plasma boundary and coils just to set up permanent magnet grid
surface_filename = TEST_DIR / 'input.muse'
s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Create plotting surface with same quadrature points
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)


# Load a downsampled version of the magnet grid from a FAMUS file
mag_data = FocusData(famus_filename, downsample=1)

# Convert spherical coordinates (M_0, mt, mp) arrays to cartesian (mx, my, mz)
mx = np.array(mag_data.M_0) * np.sin(np.array(mag_data.mt)) * np.cos(np.array(mag_data.mp))
my = np.array(mag_data.M_0) * np.sin(np.array(mag_data.mt)) * np.sin(np.array(mag_data.mp))
mz = np.array(mag_data.M_0) * np.cos(np.array(mag_data.mt))

# Combine into array with shape (nMagnets, 3) 
m = np.column_stack((mx, my, mz))

#Set up dipole position grid by calling geo_setup_from_famus
pm_grid = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename)

# Calculate forces and torques
net_forces, net_torques = pm_grid.force_torque_calc(m)

# Create dipole field object
b_dipole = DipoleField(
    pm_grid.dipole_grid_xyz,
    m,
    nfp=s.nfp,
    coordinate_flag=pm_grid.coordinate_flag,
    m_maxima=pm_grid.m_maxima,
    net_forces=net_forces,
    net_torques=net_torques
)

# Calculate and plot Bnormal
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / f"Dipole_Fields")





