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
# from sksparse.cholmod import cholesky
from simsopt.geo import SurfaceRZFourier, CurveRZFourier, curves_to_vtk
from simsopt.field.magneticfieldclasses import WindingVolumeField
from simsopt.geo import WindingVolumeGrid
from simsopt.solve import projected_gradient_descent_Tikhonov 
from simsopt.util.permanent_magnet_helper_functions import *
import time

t_start = time.time()

# Set some parameters
nphi = 8  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 8
dx = 0.05
dy = dx
dz = dx
poff = 0.5  # PM grid end offset ~ 10 cm from the plasma surface
coff = 0.05  # PM grid starts offset ~ 5 cm from the plasma surface
input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

qphi = s.nfp * nphi * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
OUT_DIR = 'wv_NCSX/'
os.makedirs(OUT_DIR, exist_ok=True)

# No external coils
Bnormal = np.zeros((nphi, ntheta))

# Define a curve to define a Itarget loss term
# Make circle at Z = 0
numquadpoints = nphi * 10 * s.nfp * 2
order = 40
# curve = Curve(s.gamma()[:, 0, :])
curve = CurveRZFourier(numquadpoints, order, nfp=1, stellsym=False)
for m in range(s.mpol + 1):
    if m == 0:
        nmin = 0
    else: 
        nmin = -s.ntor
    for n in range(nmin, s.ntor + 1):
        curve.rc[s.nfp * int(abs(n))] += s.get_rc(m, n)
        curve.zs[s.nfp * int(abs(n))] += s.get_zs(m, n) * np.sign(n)

curve.x = curve.get_dofs()
curve.x = curve.x  # need to do this to transfer data to C++
curves_to_vtk([curve], OUT_DIR + "Itarget_curve")
Itarget = 1e6  # 1 MA

nx = 8
# Finally, initialize the winding volume 
wv_grid = WindingVolumeGrid(
    s, Itarget_curve=curve, Itarget=Itarget, 
    coil_offset=coff, 
    dx=dx, dy=dy, dz=dz, 
    plasma_offset=poff,
    Bn=Bnormal,
    Bn_Itarget=np.zeros(curve.gammadash().reshape(-1, 3).shape[0]),
    filename=surface_filename,
    surface_flag='wout',
    OUT_DIR=OUT_DIR,
    nx=nx, ny=nx, nz=nx
)

wv_grid._toVTK(OUT_DIR + 'grid')

nfp = wv_grid.plasma_boundary.nfp
print('fB initial = ', 0.5 * np.linalg.norm(wv_grid.B_matrix @ wv_grid.alphas - wv_grid.b_rhs, ord=2) ** 2 * nfp)
t1 = time.time()

lam = 1e-20  # Strength of Tikhonov regularization
alpha_opt, fB, fK, fI = projected_gradient_descent_Tikhonov(wv_grid, lam=lam)
print('alpha_opt = ', alpha_opt)
t2 = time.time()
print('Gradient Descent Tikhonov solve time = ', t2 - t1, ' s')
plt.figure()
plt.semilogy(fB, label='fB')
plt.semilogy(lam * fK, label='lam * fK')
plt.semilogy(fI, label='fI')
plt.semilogy(fB + fI + lam * fK, label='total')
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
# print('Bnormal direct = ', Bnormal_wv)
Bnormal_lstsq = wv_grid.B_matrix @ alpha_opt * np.sqrt(nphi * ntheta) / np.sqrt(normN)
# print('Bnormal coils = ', Bnormal)
print('fB lstsq = ', np.sum(normN * np.ravel(Bnormal_lstsq) ** 2) * 0.5 / (nphi * ntheta))
print('fB direct = ', np.sum(normN * np.ravel(Bnormal_wv + Bnormal) ** 2) * 0.5 / (nphi * ntheta))
# fB_direct = SquaredFlux(s, bs_wv, -Bnormal).J()
# print('fB_direct = ', fB_direct)

# make_Bnormal_plots(bs_wv, s, OUT_DIR, "biot_savart_partial_surface")
make_Bnormal_plots(bs_wv, s_plot, OUT_DIR, "biot_savart_winding_volume")
# make_Bnormal_plots(bs + bs_wv, s, OUT_DIR, "biot_savart_total")

t_end = time.time()
print('Total time = ', t_end - t_start)
plt.show()
