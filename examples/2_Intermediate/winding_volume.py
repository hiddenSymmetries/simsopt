#!/usr/bin/env python
r"""
This example uses the winding volume method 
outline in Kaptanoglu & Landreman 2023 in order
to make finite-build coils with no multi-filament
approximation. 

The script should be run as:
    mpirun -n 1 python winding_volume.py 

"""

import os
#from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import eye as sparse_eye
from scipy.sparse.linalg import inv as sparse_inv
import simsoptpp as sopp
from simsopt.geo import SurfaceRZFourier, CurveRZFourier, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import WindingVolumeGrid
from simsopt.util.permanent_magnet_helper_functions import *
import time

t_start = time.time()

# Set some parameters
nphi = 8  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 8
dx = 0.02
dy = dx
dz = dx
coff = 0.05  # PM grid starts offset ~ 5 cm from the plasma surface
poff = 0.1  # PM grid end offset ~ 10 cm from the plasma surface
#input_name = 'input.LandremanPaul2021_QA'
input_name = 'wout_LandremanPaul_QH_variant.nc'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
#s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory
OUT_DIR = 'winding_volume/'
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

#s_plot = SurfaceRZFourier.from_vmec_input(
#    surface_filename, range="full torus",
#    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
#)
s_plot = SurfaceRZFourier.from_wout(
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

# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Define a curve to define a Itarget loss term
# Make circle at Z = 0
numquadpoints = nphi * s.nfp * 2
order = 4
curve = CurveRZFourier(numquadpoints, order, nfp=1, stellsym=False)
r_min = 0.4
curve.rc[0] = r_min
curve.x = curve.get_dofs()
curve.x = curve.x  # need to do this to transfer data to C++
curves_to_vtk([curve], OUT_DIR + f"Itarget_curve")
Itarget = 1e5  # 0.1 MA

# Set up Bnormal on the curve
bs.set_points(curve.gamma().reshape((-1, 3)))
Bnormal_Itarget = np.sum((bs.B() * curve.gammadash()).reshape(-1, 3), axis=-1)

# Finally, initialize the winding volume 
wv_grid = WindingVolumeGrid(
    s, Itarget_curve=curve, Itarget=Itarget, 
    coil_offset=coff, 
    dx=dx, dy=dy, dz=dz, 
    plasma_offset=poff,
    Bn=Bnormal,
    Bn_Itarget=Bnormal_Itarget,
    filename=surface_filename,
    surface_flag='wout',
    OUT_DIR=OUT_DIR
)

wv_grid._toVTK(OUT_DIR + 'grid')
phis = wv_grid._polynomial_basis(nx=2, ny=2, nz=2)
print(phis.shape)
wv_grid._construct_geo_factor()
print(wv_grid.geo_factor.shape)
print(wv_grid.flux_jump_matrix.shape)
for j in range(6): 
    for i in range(11 * wv_grid.N_grid):
        if wv_grid.flux_jump_matrix[j, i] != 0.0:
            print(j, i, wv_grid.flux_jump_matrix[j, i])

C = csc_matrix(wv_grid.flux_jump_matrix)  # matrix is way too big but it is very sparse
CT = C.transpose()
CCT = C @ CT
CCT_inv = sparse_inv(CCT)
projection_onto_constraints = sparse_eye(len(wv_grid.alphas), format="csc") - CT @ CCT_inv @ C 
wv_grid.alphas = projection_onto_constraints.todense() @ wv_grid.alphas
wv_grid._toVTK(OUT_DIR + 'grid_with_flux_jump_constraints')

t_end = time.time()
print('Total time = ', t_end - t_start)
