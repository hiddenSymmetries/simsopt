#!/usr/bin/env python
r"""
This example uses the current voxels method 
outline in Kaptanoglu & Landreman 2023 in order
to make finite-build coils with no multi-filament
approximation. 

The script should be run as:
    mpirun -n 1 python current_voxels.py 

"""

import os
import logging
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field import InterpolatedField, SurfaceClassifier
from simsopt.field.magneticfieldclasses import CurrentVoxelsField
from simsopt.geo import CurrentVoxelsGrid
from simsopt.solve import ras_minres
from simsopt.util.permanent_magnet_helper_functions import *
import time
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

t_start = time.time()

t1 = time.time()
# Set some parameters
nphi = 32  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = nphi
# coil_range = 'full torus'
coil_range = 'half period'
input_name = 'input.circular_tokamak' 

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range=coil_range, nphi=nphi, ntheta=ntheta)

qphi = s.nfp * nphi * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)
if coil_range == 'half period':
    s.nfp = 2
    s.stellsym = True
    s_plot.nfp = 2
    s_plot.stellsym = True
else:
    s.stellsym = False

# Make the output directory
OUT_DIR = 'current_voxels_axisymmetric_convex/'
os.makedirs(OUT_DIR, exist_ok=True)

# No external coils
Bnormal = np.zeros((nphi, ntheta))
t2 = time.time()
print('First setup took time = ', t2 - t1, ' s')

# Define a curve to define a Itarget loss term
# As the boundary of the stellarator at theta = 0
t1 = time.time()
numquadpoints = nphi * s.nfp * 2
curve = make_curve_at_theta0(s, numquadpoints)
curves_to_vtk([curve], OUT_DIR + "Itarget_curve")
Itarget = 30e6
t2 = time.time()
print('Curve initialization took time = ', t2 - t1, ' s')

fac1 = 1.2
fac2 = 2
fac3 = 3

#create the outside boundary for the PMs
s_out = SurfaceRZFourier.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=coil_range, nfp=s.nfp, stellsym=s.stellsym)
s_out.set_rc(0, 0, s.get_rc(0, 0) * fac1)
s_out.set_rc(1, 0, s.get_rc(1, 0) * fac3)
s_out.set_zs(1, 0, s.get_rc(1, 0) * fac3)
s_out.to_vtk(OUT_DIR + "surf_out")

#create the inside boundary for the PMs
s_in = SurfaceRZFourier.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=coil_range, nfp=s.nfp, stellsym=s.stellsym)
s_in.set_rc(0, 0, s.get_rc(0, 0) * fac1)
s_in.set_rc(1, 0, s.get_rc(1, 0) * fac2)
s_in.set_zs(1, 0, s.get_rc(1, 0) * fac2)
s_in.to_vtk(OUT_DIR + "surf_in")

nx = 10
Nx = 20
Ny = Nx
Nz = Nx 
# Finally, initialize the current voxels 
t1 = time.time()
wv_grid = CurrentVoxelsGrid(
    s, Itarget_curve=curve, Itarget=Itarget, 
    # coil_offset=coff, 
    rz_inner_surface=s_in,
    rz_outer_surface=s_out,
    Nx=Nx, Ny=Ny, Nz=Nz, 
    # plasma_offset=poff,
    Bn=Bnormal,
    Bn_Itarget=np.zeros(curve.gammadash().reshape(-1, 3).shape[0]),
    filename=surface_filename,
    OUT_DIR=OUT_DIR,
    nx=nx, ny=nx, nz=nx,
    sparse_constraint_matrix=True,
    coil_range=coil_range
)
wv_grid.rz_inner_surface.to_vtk(OUT_DIR + 'inner')
wv_grid.rz_outer_surface.to_vtk(OUT_DIR + 'outer')
wv_grid.to_vtk_before_solve(OUT_DIR + 'grid_before_solve_Nx' + str(Nx))
t2 = time.time()
print('WV grid initialization took time = ', t2 - t1, ' s')

t1 = time.time()
#max_iter = 20
#rs_max_iter = 120
#nu = 1e13
#lam = 1e-30 
#l0_threshold = 5e4  # 1e4
# best: max_iter = 20, rs_max_iter = 100, nu=1e13, l0 = 5e4, 20, 40
#l0_thresholds = np.linspace(l0_threshold, 25 * l0_threshold, 100, endpoint=True)
max_iter = 200
rs_max_iter = 200
nu = 1e1
kappa = 1e-14
sigma = 1  # 1e-2
l0_threshold = 5e4
l0_thresholds = np.linspace(l0_threshold, 30 * l0_threshold, 30, endpoint=True)
alpha_opt, fB, fK, fI, fRS, f0, fC, fminres, fBw, fKw, fIw, fRSw, fCw = ras_minres( 
    #alpha_opt, fB, fK, fI, fRS, f0, fC, fminres, fBw, fKw, fIw, fRSw, fCw = ras_preconditioned_minres( 
    #alpha_opt, fB, fK, fI, fRS, f0, fBw, fKw, fIw = relax_and_split_increasingl0(
    wv_grid, kappa=kappa, nu=nu, max_iter=max_iter,
    l0_thresholds=l0_thresholds,
    sigma=sigma,
    rs_max_iter=rs_max_iter,
    print_iter=40,
    OUT_DIR=OUT_DIR
)
t2 = time.time()
print('MINRES solve time = ', t2 - t1, ' s')    

t1 = time.time()
wv_grid.to_vtk_after_solve(OUT_DIR + 'grid_after_Tikhonov_solve_Nx' + str(Nx))
t2 = time.time()
print('Time to plot the optimized grid = ', t2 - t1, ' s')
print('fB after optimization = ', fB[-1]) 
print('fB check = ', 0.5 * np.linalg.norm(wv_grid.B_matrix @ alpha_opt - wv_grid.b_rhs) ** 2 * s.nfp * 2)

# set up CurrentVoxels Bfield
bs_wv = CurrentVoxelsField(wv_grid.J, wv_grid.XYZ_integration, wv_grid.grid_scaling, wv_grid.coil_range, nfp=s.nfp, stellsym=s.stellsym)
t1 = time.time()
bs_wv.set_points(s.gamma().reshape((-1, 3)))
Bnormal_wv = np.sum(bs_wv.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
normN = np.linalg.norm(s.normal().reshape(-1, 3), axis=-1)
contig = np.ascontiguousarray
print('fB direct = ', np.sum(normN * np.ravel(Bnormal_wv + Bnormal) ** 2) * 0.5 / (nphi * ntheta) * s.nfp * 2)

t2 = time.time()
print('Time to compute Bnormal_wv = ', t2 - t1, ' s')
fB_direct = SquaredFlux(s, bs_wv, -Bnormal).J() * 2 * s.nfp
print('fB_direct = ', fB_direct)

gamma = curve.gamma().reshape(-1, 3)
bs_wv.set_points(gamma)
gammadash = curve.gammadash().reshape(-1, 3)
Bnormal_Itarget_curve = np.sum(bs_wv.B() * gammadash, axis=-1)
mu0 = 4 * np.pi * 1e-7
# print(curve.quadpoints)
Itarget_check = np.sum(Bnormal_Itarget_curve) / mu0 / len(curve.quadpoints)
print('Itarget_check = ', Itarget_check)
print('Itarget second check = ', wv_grid.Itarget_matrix @ alpha_opt / mu0) 

t1 = time.time()
make_Bnormal_plots(bs_wv, s_plot, OUT_DIR, "biot_savart_current_voxels_Nx" + str(Nx))
t2 = time.time()

print('Time to plot Bnormal_wv = ', t2 - t1, ' s')

plt.figure()
plt.semilogy(fB, 'r', label=r'$f_B$')
plt.semilogy(fK, 'b', label=r'$\lambda \|\alpha\|^2$')
plt.semilogy(fC, 'c', label=r'$f_C$')
plt.semilogy(fI, 'm', label=r'$f_I$')
plt.semilogy(fminres, 'k--', label=r'MINRES residual')
if l0_thresholds[-1] > 0:
    fRS[np.abs(fRS) < 1e-20] = 0.0
    plt.semilogy(fRS, 'g', label=r'$\nu^{-1} \|\alpha - w\|^2$')
plt.grid(True)
plt.legend()
plt.figure()
plt.semilogy(fBw, 'r', label=r'$f_Bw$')
plt.semilogy(fKw, 'b', label=r'$\lambda \|w\|^2$')
plt.semilogy(fCw, 'c', label=r'$f_Cw$')
plt.semilogy(fIw, 'm', label=r'$f_Iw$')
plt.semilogy(fRSw, 'g', label=r'$f_RSw$')
plt.semilogy(fBw + fIw + fKw + fRSw, 'k', label='Total objective (not incl. l0)')
plt.grid(True)
plt.legend()

# plt.savefig(OUT_DIR + 'optimization_progress.jpg')
t1 = time.time()
#wv_grid.check_fluxes()
t2 = time.time()
print('Time to check all the flux constraints = ', t2 - t1, ' s')

calculate_on_axis_B(bs_wv, s)

trace_field = False
if trace_field:
    t1 = time.time()
    bs_wv.set_points(s_plot.gamma().reshape((-1, 3)))
    print('R0 = ', s.get_rc(0, 0), ', r0 = ', s.get_rc(1, 0))
    n = 20
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 4

    # compute the fieldlines from the initial locations specified above
    sc_fieldline = SurfaceClassifier(s_plot, h=0.03, p=2)

    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    # Load in the optimized coils from stage_two_optimization.py:
    bsh = InterpolatedField(
        bs_wv, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym,  # skip=skip
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    bs_wv.set_points(s_plot.gamma().reshape((-1, 3)))
    make_Bnormal_plots(bsh, s_plot, OUT_DIR, "biot_savart_interpolated")
    Bh = bsh.B()
    B = bs_wv.B()
    print("Mean(|B|) on plasma surface =", np.mean(bs_wv.AbsB()))
    print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
    nfieldlines = 10
    R0 = np.linspace(6, 7.9, nfieldlines)
    trace_fieldlines(bsh, 'current_voxels_axisymmetric_poincare', s_plot, None, OUT_DIR, R0)
    t2 = time.time()
print(OUT_DIR)

t_end = time.time()
print('Total time = ', t_end - t_start)
print('f fB fI fK fC')
print(f0[-1], fB[-1], fI[-1], fK[-1], fC[-1])
plt.show()
