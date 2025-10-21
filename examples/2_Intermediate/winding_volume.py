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
from scipy.sparse import eye as sparse_eye
from scipy.sparse.linalg import inv as sparse_inv
from simsopt.geo import SurfaceRZFourier, CurveRZFourier, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import WindingVolumeField
from simsopt.geo import WindingVolumeGrid
from simsopt.solve import projected_gradient_descent_Tikhonov 
from simsopt.util.permanent_magnet_helper_functions import *
import time

t_start = time.time()

# Set some parameters
nphi = 32  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 32
dx = 0.015
dy = dx
dz = dx
coff = 0.05  # PM grid starts offset ~ 5 cm from the plasma surface
poff = 0.02  # PM grid end offset ~ 10 cm from the plasma surface
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
#qphi = 2 * nphi
#quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
#quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)

#s_plot = SurfaceRZFourier.from_vmec_input(
#    surface_filename, range="full torus",
#    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
#)
#s_plot = SurfaceRZFourier.from_wout(
#    surface_filename, range="full torus",
#    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
#)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s, OUT_DIR, "biot_savart_initial")
#make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

# optimize the currents in the TF coils
s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, 'qa')

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
curves_to_vtk([curve], OUT_DIR + "Itarget_curve")
Itarget = 1e5  # 0.1 MA

# Set up Bnormal on the curve
bs.set_points(curve.gamma().reshape((-1, 3)))
Bnormal_Itarget = np.sum((bs.B() * curve.gammadash()).reshape(-1, 3), axis=-1)

nx = 8
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
    OUT_DIR=OUT_DIR,
    nx=nx, ny=nx, nz=nx
)

wv_grid._toVTK(OUT_DIR + 'grid')

#print(wv_grid.geo_factor.shape)
#print(wv_grid.flux_constraint_matrix.shape)
#for j in range(6): 
#    for i in range(11 * wv_grid.N_grid):
#        if wv_grid.flux_constraint_matrix[j, i] != 0.0:
#            print(j, i, wv_grid.flux_constraint_matrix[j, i])

if False:
    t1 = time.time()
    C = wv_grid.flux_constraint_matrix  # matrix is way too big but it is very sparse
    # Need to append Itarget constraint to the flux jump constraints
    # C = vstack([C, wv_grid.Itarget_matrix], format="csc")
    CT = C.transpose()
    CCT = C @ CT
    t2 = time.time()
    print('Time to make CCT = ', t2 - t1, ' s')
    t1 = time.time()
    CCT_inv = sparse_inv(CCT)
    t2 = time.time()
    print('Time to make CCT_inv = ', t2 - t1, ' s')
    t1 = time.time()
    CT_CCT_inv = CT @ CCT_inv
    CT_CCT_inv_d = CT_CCT_inv[:, -1] * wv_grid.Itarget_rhs
    projection_onto_constraints = sparse_eye(wv_grid.N_grid * wv_grid.n_functions, format="csc") - CT @ CCT_inv @ C 
    wv_grid.alphas = projection_onto_constraints.dot(np.ravel(wv_grid.alphas)).reshape(wv_grid.alphas.shape)
    t2 = time.time()
    print('Time to make projection operator and project alpha = ', t2 - t1, ' s')
    wv_grid._toVTK(OUT_DIR + 'grid_with_flux_jump_constraints')
nfp = wv_grid.plasma_boundary.nfp
print('fB initial = ', 0.5 * np.linalg.norm(wv_grid.B_matrix @ wv_grid.alphas - wv_grid.b_rhs, ord=2) ** 2 * nfp)
t1 = time.time()
alpha_opt, fB, fK = projected_gradient_descent_Tikhonov(wv_grid, lam=1e-20)
print('alpha_opt = ', alpha_opt)
t2 = time.time()
print('Gradient Descent Tikhonov solve time = ', t2 - t1, ' s')
plt.figure()
plt.semilogy(fB, label='fB')
plt.semilogy(fK, label='fK')
plt.semilogy(fB + fK, label='total')
plt.grid(True)
plt.legend()
wv_grid.alphas = alpha_opt
wv_grid._toVTK(OUT_DIR + 'grid_after_Tikhonov_solve')
# print('fB after optimization = ', fB) 

# set up WindingVolume Bfield
bs_wv = WindingVolumeField(wv_grid)
bs_wv.set_points(s.gamma().reshape((-1, 3)))
Bnormal_wv = np.sum(bs_wv.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
normN = np.linalg.norm(s.normal().reshape(-1, 3), axis=-1)
print('Bnormal direct = ', Bnormal_wv)
print('Bnormal lstsq = ', wv_grid.B_matrix @ alpha_opt * np.sqrt(nphi * ntheta) / np.sqrt(normN))
print('Bnormal coils = ', Bnormal)
print('fB direct = ', np.sum(normN * np.ravel(Bnormal_wv + Bnormal) ** 2) * 0.5 / (nphi * ntheta))
fB_direct = SquaredFlux(s, bs_wv, -Bnormal).J()
print('fB_direct = ', fB_direct)

make_Bnormal_plots(bs_wv, s, OUT_DIR, "biot_savart_only_winding_volume")
make_Bnormal_plots(bs + bs_wv, s, OUT_DIR, "biot_savart_total")

t_end = time.time()
print('Total time = ', t_end - t_start)
plt.show()
