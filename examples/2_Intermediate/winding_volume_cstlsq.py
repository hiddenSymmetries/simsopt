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
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier, CurveRZFourier, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import WindingVolumeField
from simsopt.geo import WindingVolumeGrid
from simsopt.solve import cstlsq
from simsopt.util.permanent_magnet_helper_functions import *
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

t_start = time.time()

t1 = time.time()
# Set some parameters
nphi = 16  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 16
poff = 0.25  # grid end offset ~ 10 cm from the plasma surface
coff = 0.4  # grid starts offset ~ 5 cm from the plasma surface
input_name = 'input.LandremanPaul2021_QA'
# input_name = 'input.circular_tokamak' 

lam = 1e-22
l0_threshold = 0.0  # 5e4
nu = 1e100  # 1e15

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s.nfp = 2
s.stellsym = True

qphi = s.nfp * nphi * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)
s_plot.nfp = 2
s_plot.stellsym = True

# Make the output directory
OUT_DIR = 'wv_cstlsq/'
os.makedirs(OUT_DIR, exist_ok=True)

# No external coils
Bnormal = np.zeros((nphi, ntheta))
t2 = time.time()
print('First setup took time = ', t2 - t1, ' s')

fB_all = []
fI_all = []
fK_all = []
t_algorithm = []
# Define a curve to define a Itarget loss term
# Make circle at Z = 0
t1 = time.time()
numquadpoints = nphi * s.nfp * 2  # * 5
order = 20
quadpoints = np.linspace(0, 1, numquadpoints, endpoint=True)
curve = CurveRZFourier(quadpoints, order, nfp=1, stellsym=False)
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
Itarget = 0.5e6
t2 = time.time()
print('Curve initialization took time = ', t2 - t1, ' s')

# params = [1, 2, 3, 4, 5, 10, 20, 30]
params = [16]
num_cells = []
cell_volumes = []
# for nx in params:
for Nx in params:
    nx = 6
    # Nx = 12
    Ny = Nx
    Nz = Nx 
    # Finally, initialize the winding volume 
    t1 = time.time()
    wv_grid = WindingVolumeGrid(
        s, Itarget_curve=curve, Itarget=Itarget, 
        coil_offset=coff, 
        Nx=Nx, Ny=Ny, Nz=Nz, 
        plasma_offset=poff,
        Bn=Bnormal,
        Bn_Itarget=np.zeros(curve.gammadash().reshape(-1, 3).shape[0]),
        filename=surface_filename,
        surface_flag='vmec',
        OUT_DIR=OUT_DIR,
        # coil_range="full torus",
        nx=nx, ny=nx, nz=nx,
        sparse_constraint_matrix=False,  # Need this so can delete elements out of P I guess
    )
    t2 = time.time()
    print('WV grid initialization took time = ', t2 - t1, ' s')
    #wv_grid.to_vtk_before_solve(OUT_DIR + 'grid_before_solve_Nx' + str(Nx))

    max_iter = 300
    rs_max_iter = 10

    #l0_thresholds = np.linspace(l0_threshold, 60 * l0_threshold, 60, endpoint=True)
    l0_thresholds = [1e5, 5e5, 1e6] 
    alpha_opt, fB, fK, fI, f0 = cstlsq( 
        wv_grid, lam=lam, max_iter=max_iter,
        l0_thresholds=l0_thresholds, 
        stlsq_max_iter=rs_max_iter,
        print_iter=100,
    )

    # print('alpha_opt = ', alpha_opt)
    #if wv_grid.P is not None:
    #    print('P * alpha_opt - alpha_opt = ', wv_grid.P.dot(alpha_opt) - alpha_opt)
    #    print('||P * alpha_opt - alpha_opt|| / ||alpha_opt|| = ', np.linalg.norm(wv_grid.P.dot(alpha_opt) - alpha_opt) / np.linalg.norm(alpha_opt))
    t2 = time.time()
    print('Gradient Descent Tikhonov solve time = ', t2 - t1, ' s')    
    t_algorithm.append(t2 - t1)

    t1 = time.time()
    wv_grid.to_vtk_after_solve(OUT_DIR + 'grid_after_Tikhonov_solve_Nx' + str(Nx))
    t2 = time.time()
    print('Time to plot the optimized grid = ', t2 - t1, ' s')
    print('fB after optimization = ', fB[-1]) 
    print('fB check = ', 0.5 * np.linalg.norm(wv_grid.B_matrix @ alpha_opt - wv_grid.b_rhs) ** 2 * s.nfp * 2)

    # set up WindingVolume Bfield
    bs_wv = WindingVolumeField(wv_grid.J, wv_grid.XYZ_integration, wv_grid.grid_scaling, wv_grid.coil_range)
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

    bs_wv.set_points(curve.gamma().reshape((-1, 3)))
    Bnormal_Itarget_curve = np.sum(bs_wv.B() * curve.gammadash().reshape(-1, 3), axis=-1)
    mu0 = 4 * np.pi * 1e-7
    print(curve.quadpoints)
    Itarget_check = np.sum(Bnormal_Itarget_curve) / mu0 / len(curve.quadpoints)
    print('Itarget_check = ', Itarget_check)
    print('Itarget second check = ', wv_grid.Itarget_matrix @ alpha_opt / mu0) 

    t1 = time.time()
    make_Bnormal_plots(bs_wv, s_plot, OUT_DIR, "biot_savart_winding_volume_Nx" + str(Nx))
    t2 = time.time()

    print('Time to plot Bnormal_wv = ', t2 - t1, ' s')

    plt.figure()
    plt.semilogy(fB, 'r', label=r'$f_B$')
    plt.semilogy(lam * fK, 'b', label=r'$\lambda \|\alpha\|^2$')
    plt.semilogy(fI, 'm', label=r'$f_I$')
    plt.semilogy(fB + fI + lam * fK, 'g', label='Total objective (not incl. l0)')
    plt.grid(True)
    plt.legend()
    plt.show()

    #t1 = time.time()
    #wv_grid.check_fluxes()
    #t2 = time.time()
    #print('Time to check all the flux constraints = ', t2 - t1, ' s')

    # t1 = time.time()
    # biotsavart_json_str = bs_wv.save(filename=OUT_DIR + 'BiotSavart.json')
    # bs_wv.set_points(s.gamma().reshape((-1, 3)))
    # trace_fieldlines(bs_wv, 'poincare_qa', 'qa', s_plot, comm, OUT_DIR)
    # t2 = time.time()
    print(OUT_DIR)

    t_end = time.time()
    print('Total time = ', t_end - t_start)
