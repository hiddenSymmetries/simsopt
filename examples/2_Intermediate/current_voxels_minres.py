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
from simsopt.geo import SurfaceRZFourier, CurveRZFourier, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field import InterpolatedField, SurfaceClassifier
from simsopt.field.magneticfieldclasses import CurrentVoxelsField
from simsopt.geo import CurrentVoxelsGrid
from simsopt.solve import ras_minres
from simsopt.util.permanent_magnet_helper_functions import *
import time
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

t_start = time.time()

t1 = time.time()
# Set some parameters
nphi = 16  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = nphi
poff = 0.5
coff = 0.4
# input_name = 'input.circular_tokamak' 
input_name = 'input.LandremanPaul2021_QA'

lam = 1e-40
nu = 1e100

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

qphi = s.nfp * nphi * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename,
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
out_dir = 'wv_test/'
os.makedirs(out_dir, exist_ok=True)

# No external coils
Bnormal = np.zeros((nphi, ntheta))
t2 = time.time()
print('First setup took time = ', t2 - t1, ' s')

# Define a curve to define a Itarget loss term
# As the boundary of the stellarator at theta = 0
t1 = time.time()
numquadpoints = nphi * s.nfp * 2
order = s.ntor + 1
quadpoints = np.linspace(0, 1, numquadpoints, endpoint=True)
curve = CurveRZFourier(quadpoints, order, s.nfp, stellsym=True)
r_mn = np.zeros((s.mpol + 1, 2 * s.ntor + 1))
z_mn = np.zeros((s.mpol + 1, 2 * s.ntor + 1))
for m in range(s.mpol + 1):
    if m == 0:
        nmin = 0
    else: 
        nmin = -s.ntor
    for n in range(nmin, s.ntor + 1):
        r_mn[m, n + s.ntor] = s.get_rc(m, n)
        z_mn[m, n + s.ntor] = s.get_zs(m, n)
r_n = np.sum(r_mn, axis=0)
z_n = np.sum(z_mn, axis=0)
for n in range(s.ntor + 1):
    if n == 0:
        curve.rc[n] = r_n[n + s.ntor]
    else:
        curve.rc[n] = r_n[n + s.ntor] + r_n[-n + s.ntor]
        curve.zs[n - 1] = -z_n[n + s.ntor] + z_n[-n + s.ntor]

curve.x = curve.get_dofs()
curve.x = curve.x  # need to do this to transfer data to C++
curves_to_vtk([curve], out_dir + "Itarget_curve")
Itarget = 0.45e6
t2 = time.time()
print('Curve initialization took time = ', t2 - t1, ' s')

fac1 = 1.2
fac2 = 5
fac3 = 7
#create the outside boundary for the PMs
s_out = SurfaceRZFourier.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range='half period', nfp=2, stellsym=True)
s_out.set_rc(0, 0, s.get_rc(0, 0) * fac1)
s_out.set_rc(1, 0, s.get_rc(1, 0) * fac3)
s_out.set_zs(1, 0, s.get_rc(1, 0) * fac3)
s_out.to_vtk(out_dir + "surf_out")

#create the inside boundary for the PMs
s_in = SurfaceRZFourier.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range='half period', nfp=2, stellsym=True)
s_in.set_rc(0, 0, s.get_rc(0, 0) * fac1)
s_in.set_rc(1, 0, s.get_rc(1, 0) * fac2)
s_in.set_zs(1, 0, s.get_rc(1, 0) * fac2)
s_in.to_vtk(out_dir + "surf_in")

nx = 10
Nx = 24
Ny = Nx
Nz = Nx 
# Finally, initialize the current voxels 
t1 = time.time()
wv_grid = CurrentVoxelsGrid(
    s, Itarget_curve=curve, Itarget=Itarget, 
    #plasma_offset=poff, 
    #coil_offset=coff, 
    rz_inner_surface=s_in,
    rz_outer_surface=s_out,
    Nx=Nx, Ny=Ny, Nz=Nz, 
    Bn=Bnormal,
    Bn_Itarget=np.zeros(curve.gammadash().reshape(-1, 3).shape[0]),
    filename=surface_filename,
    OUT_DIR=out_dir,
    nx=nx, ny=nx, nz=nx,
    sparse_constraint_matrix=True,
)
wv_grid.rz_inner_surface.to_vtk(out_dir + 'inner')
wv_grid.rz_outer_surface.to_vtk(out_dir + 'outer')
wv_grid.to_vtk_before_solve(out_dir + 'grid_before_solve_Nx' + str(Nx))
t2 = time.time()
print('WV grid initialization took time = ', t2 - t1, ' s')

max_iter = 20
rs_max_iter = 200
nu = 1e2  # 1e1
l0_threshold = 5e4  # 60 below line
l0_thresholds = np.linspace(l0_threshold, 150 * l0_threshold, 40, endpoint=True)
alpha_opt, fB, fK, fI, fRS, f0, fC, fBw, fKw, fIw = ras_minres( 
    wv_grid, lam=lam, nu=nu, max_iter=max_iter,
    l0_thresholds=l0_thresholds, 
    rs_max_iter=rs_max_iter,
    print_iter=100,
    OUT_DIR=out_dir
)
print('solution shape = ', alpha_opt.shape)

t2 = time.time()
print('Gradient Descent Tikhonov solve time = ', t2 - t1, ' s')    
#print('||P * alpha_opt - alpha_opt|| / ||alpha_opt|| = ', np.linalg.norm(wv_grid.P.dot(alpha_opt) - alpha_opt) / np.linalg.norm(alpha_opt))
#print('||P * w_opt - w_opt|| / ||w_opt|| = ', np.linalg.norm(wv_grid.P.dot(wv_grid.w) - wv_grid.w) / np.linalg.norm(wv_grid.w))

t1 = time.time()
wv_grid.to_vtk_after_solve(out_dir + 'grid_after_Tikhonov_solve_Nx' + str(Nx))
t2 = time.time()
print('Time to plot the optimized grid = ', t2 - t1, ' s')
print('fB after optimization = ', fB[-1]) 
print('fB check = ', 0.5 * np.linalg.norm(wv_grid.B_matrix @ alpha_opt - wv_grid.b_rhs) ** 2 * s.nfp * 2)

# set up CurrentVoxels Bfield
bs_wv = CurrentVoxelsField(wv_grid.J, wv_grid.XYZ_integration, wv_grid.grid_scaling, wv_grid.coil_range, nfp=s.nfp, stellsym=s.stellsym)
bs_wv_sparse = CurrentVoxelsField(wv_grid.J_sparse, wv_grid.XYZ_integration, wv_grid.grid_scaling, wv_grid.coil_range, nfp=s.nfp, stellsym=s.stellsym)
t1 = time.time()
bs_wv.set_points(s.gamma().reshape((-1, 3)))
bs_wv_sparse.set_points(s.gamma().reshape((-1, 3)))
Bnormal_wv = np.sum(bs_wv.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
Bnormal_wv_sparse = np.sum(bs_wv_sparse.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
normN = np.linalg.norm(s.normal().reshape(-1, 3), axis=-1)
contig = np.ascontiguousarray
print('fB direct = ', np.sum(normN * np.ravel(Bnormal_wv + Bnormal) ** 2) * 0.5 / (nphi * ntheta) * s.nfp * 2)
print('fB direct (sparse) = ', np.sum(normN * np.ravel(Bnormal_wv_sparse + Bnormal) ** 2) * 0.5 / (nphi * ntheta) * s.nfp * 2)

t2 = time.time()
print('Time to compute Bnormal_wv = ', t2 - t1, ' s')
fB_direct = SquaredFlux(s, bs_wv, -Bnormal).J() * 2 * s.nfp
print('fB_direct = ', fB_direct)

bs_wv.set_points(curve.gamma().reshape((-1, 3)))
Bnormal_Itarget_curve = np.sum(bs_wv.B() * curve.gammadash().reshape(-1, 3), axis=-1)
mu0 = 4 * np.pi * 1e-7
print(curve.quadpoints)
Itarget_check = np.sum(Bnormal_Itarget_curve) / mu0 / len(curve.quadpoints)
print('Itarget_check = ', Itarget_check)
print('Itarget second check = ', wv_grid.Itarget_matrix @ alpha_opt / mu0) 

t1 = time.time()
make_Bnormal_plots(bs_wv, s_plot, out_dir, "biot_savart_current_voxels_Nx" + str(Nx))
make_Bnormal_plots(bs_wv_sparse, s_plot, out_dir, "biot_savart_current_voxels_sparse_Nx" + str(Nx))
t2 = time.time()

print('Time to plot Bnormal_wv = ', t2 - t1, ' s')

w_range = np.linspace(0, len(fB), len(fBw), endpoint=True)
plt.figure()
plt.semilogy(fB, 'r', label=r'$f_B$')
#plt.semilogy(lam * fK, 'b', label=r'$\lambda \|\alpha\|^2$')
plt.semilogy(fI, 'm', label=r'$f_I$')
plt.semilogy(fC, 'k', label=r'$f_C$')
#plt.semilogy(fRS, 'k', label=r'$f_{RS}$')
#plt.semilogy(w_range, fBw, 'r--', label=r'$f_Bw$')
#plt.semilogy(w_range, lam * fKw, 'b--', label=r'$\lambda \|w\|^2$')
#plt.semilogy(w_range, fIw, 'm--', label=r'$f_Iw$')
if l0_thresholds[-1] > 0:
    plt.semilogy(fRS / nu, label=r'$\nu^{-1} \|\alpha - w\|^2$')
    # plt.semilogy(f0, label=r'$\|\alpha\|_0^G$')
plt.semilogy(fB + fI + lam * fK, 'g', label='Total objective (not incl. l0)')
#plt.semilogy(w_range, fBw + fIw + lam * fKw, 'g--', label='Total w objective (not incl. l0)')
plt.grid(True)
plt.legend()

# plt.savefig(out_dir + 'optimization_progress.jpg')
t1 = time.time()
wv_grid.check_fluxes()
t2 = time.time()
print('Time to check all the flux constraints = ', t2 - t1, ' s')

bs_wv.set_points(s_plot.gamma().reshape((-1, 3)))
make_Bnormal_plots(bs_wv, s_plot, out_dir, "biot_savart_current_voxels")
B = bs_wv.B()
calculate_on_axis_B(bs_wv, s)
print("Mean(|B|) on plasma surface =", np.mean(bs_wv.AbsB()))

if False: 
    t1 = time.time()
    # biotsavart_json_str = bs_wv.save(filename=out_dir + 'BiotSavart.json')
    print('R0 = ', s.get_rc(0, 0), ', r0 = ', s.get_rc(1, 0))
    n = 20
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 4

    # compute the fieldlines from the initial locations specified above

    ####### s -> s_plot here is critical!!!
    sc_fieldline = SurfaceClassifier(s_plot, h=0.03, p=2)
    sc_fieldline.to_vtk(out_dir + 'levelset', h=0.02)

    def skip(rs, phis, zs):
        # The RegularGrindInterpolant3D class allows us to specify a function that
        # is used in order to figure out which cells to be skipped.  Internally,
        # the class will evaluate this function on the nodes of the regular mesh,
        # and if *all* of the eight corners are outside the domain, then the cell
        # is skipped.  Since the surface may be curved in a way that for some
        # cells, all mesh nodes are outside the surface, but the surface still
        # intersects with a cell, we need to have a bit of buffer in the signed
        # distance (essentially blowing up the surface a bit), to avoid ignoring
        # cells that shouldn't be ignored
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    # Load in the optimized coils from stage_two_optimization.py:
    bsh = InterpolatedField(
        # bs, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym, skip=skip
        bs_wv, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym, skip=skip
    )
    # bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
    make_Bnormal_plots(bsh, s_plot, out_dir, "biot_savart_interpolated")
    # trace_fieldlines(bs_wv, 'poincare_torus', s_plot, comm, out_dir)
    nfieldlines = 30
    R0 = np.linspace(1.2125346, 1.295, nfieldlines)
    trace_fieldlines(bsh, 'poincare_torus', s_plot, None, out_dir, R0)
    t2 = time.time()
    print(out_dir)

t_end = time.time()
print('Total time = ', t_end - t_start)
plt.show()
