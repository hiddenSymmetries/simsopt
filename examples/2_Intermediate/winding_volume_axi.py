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
from simsopt.geo import SurfaceRZFourier, CurveRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import WindingVolumeField
from simsopt.geo import WindingVolumeGrid
from simsopt.solve import relax_and_split_increasingl0
from simsopt.util.permanent_magnet_helper_functions import *
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

t_start = time.time()

t1 = time.time()
# Set some parameters
nphi = 32  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 32
poff = 2.0  # grid end offset ~ 10 cm from the plasma surface
coff = 1.0  # grid starts offset ~ 5 cm from the plasma surface
# input_name = 'input.LandremanPaul2021_QA'
input_name = 'input.circular_tokamak' 

lam = 1e-20
l0_threshold = 5e3
nu = 1e15

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
OUT_DIR = 'wv_axi_fake_QA/'
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
# curves_to_vtk([curve], OUT_DIR + f"Itarget_curve")
Itarget = 0.5e6
t2 = time.time()
print('Curve initialization took time = ', t2 - t1, ' s')

# params = [1, 2, 3, 4, 5, 10, 20, 30]
params = [22]
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
        sparse_constraint_matrix=True,
    )
    t2 = time.time()
    print('WV grid initialization took time = ', t2 - t1, ' s')
    wv_grid.to_vtk_before_solve(OUT_DIR + 'grid_before_solve_Nx' + str(Nx))

    max_iter = 100
    rs_max_iter = 10

    l0_thresholds = np.linspace(l0_threshold, 20 * l0_threshold, 30, endpoint=True)
    alpha_opt, fB, fK, fI, fRS, f0, fBw, fKw, fIw = relax_and_split_increasingl0(
        wv_grid, lam=lam, nu=nu, max_iter=max_iter,
        l0_thresholds=l0_thresholds, 
        rs_max_iter=rs_max_iter,
        print_iter=10,
    )

    # print('alpha_opt = ', alpha_opt)
    if wv_grid.P is not None:
        print('P * alpha_opt - alpha_opt = ', wv_grid.P.dot(alpha_opt) - alpha_opt)
        print('P * w_opt - w_opt = ', wv_grid.P.dot(wv_grid.w) - wv_grid.w)
        print('||P * alpha_opt - alpha_opt|| / ||alpha_opt|| = ', np.linalg.norm(wv_grid.P.dot(alpha_opt) - alpha_opt) / np.linalg.norm(alpha_opt))
        print('||P * w_opt - w_opt|| / ||w_opt|| = ', np.linalg.norm(wv_grid.P.dot(wv_grid.w) - wv_grid.w) / np.linalg.norm(wv_grid.w))
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
    bs_wv_sparse = WindingVolumeField(wv_grid.J_sparse, wv_grid.XYZ_integration, wv_grid.grid_scaling, wv_grid.coil_range)
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

    t1 = time.time()
    make_Bnormal_plots(bs_wv, s_plot, OUT_DIR, "biot_savart_winding_volume_Nx" + str(Nx))
    make_Bnormal_plots(bs_wv_sparse, s_plot, OUT_DIR, "biot_savart_winding_volume_sparse_Nx" + str(Nx))
    t2 = time.time()
    print('Time to plot Bnormal_wv = ', t2 - t1, ' s')

    bs_wv.set_points(curve.gamma().reshape((-1, 3)))
    Bnormal_Itarget_curve = np.sum(bs_wv.B() * curve.gammadash().reshape(-1, 3), axis=-1)
    mu0 = 4 * np.pi * 1e-7
    print(curve.quadpoints)
    Itarget_check = np.sum(Bnormal_Itarget_curve) / mu0 / len(curve.quadpoints)
    print('Itarget_check = ', Itarget_check)
    print('Itarget second check = ', wv_grid.Itarget_matrix @ alpha_opt / mu0) 

    w_range = np.linspace(0, len(fB), len(fBw), endpoint=True)
    plt.figure()
    plt.semilogy(fB, 'r', label=r'$f_B$')
    plt.semilogy(lam * fK, 'b', label=r'$\lambda \|\alpha\|^2$')
    plt.semilogy(fI, 'm', label=r'$f_I$')
    plt.semilogy(w_range, fBw, 'r--', label=r'$f_Bw$')
    plt.semilogy(w_range, lam * fKw, 'b--', label=r'$\lambda \|w\|^2$')
    plt.semilogy(w_range, fIw, 'm--', label=r'$f_Iw$')
    if l0_threshold > 0:
        plt.semilogy(fRS, label=r'$\nu^{-1} \|\alpha - w\|^2$')
        # plt.semilogy(f0, label=r'$\|\alpha\|_0^G$')
    plt.semilogy(fB + fI + lam * fK + fRS, 'g', label='Total objective (not incl. l0)')
    plt.semilogy(w_range, fBw + fIw + lam * fKw, 'g--', label='Total w objective (not incl. l0)')
    plt.grid(True)
    plt.legend()

    alpha_history = np.squeeze(np.array(wv_grid.alpha_history))
    alpha_history = alpha_history.reshape(alpha_history.shape[0], wv_grid.N_grid, wv_grid.n_functions)
    alpha_history += np.ones(alpha_history.shape)
    w_history = np.squeeze(np.array(wv_grid.w_history))
    w_history = w_history.reshape(w_history.shape[0], wv_grid.N_grid, wv_grid.n_functions)
    w_history += np.ones(w_history.shape)
    plt.figure()
    fig, ax = plt.subplots()
    colors = ['r', 'b']
    for i, datum in enumerate([alpha_history, w_history]):
        # Code from https://matplotlib.org/stable/gallery/animation/animated_histogram.html
        def prepare_animation(bar_container):
            def animate(frame_number):
                plt.title('Iteration # ' + str(frame_number))
                data = np.linalg.norm(datum[frame_number, :, :], axis=-1)
                n, _ = np.histogram(data, np.logspace(0, 6, 40))
                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)
                return bar_container.patches
            return animate

        # make histogram animation of the dipoles at each relax-and-split save
        data = np.linalg.norm(datum[0, :, :], axis=-1)
        _, _, bar_container = ax.hist(data, bins=np.logspace(0, 6, 40), log=True, alpha=0.5, color=colors[i], edgecolor='k')
        ax.set_ylim(top=wv_grid.N_grid)  # set safe limit to ensure that all data is visible.
        plt.grid(True)
        plt.xscale('log')
        #if i == 1:
        #    plt.legend([np.array([r'$\alpha^*$', r'w$^*$'])])

        plt.xlabel(r'Magnitude of each cell group')
        plt.ylabel('Number of cells')
    ani = animation.FuncAnimation(
        fig, prepare_animation(bar_container),
        range(0, alpha_history.shape[0]),
        repeat=False, blit=True
    )
    ani.save(OUT_DIR + 'history_l2{0:.2e}'.format(lam) + '_l0{0:.2e}'.format(l0_threshold) + '_nu{0:.2e}'.format(nu) + '_N' + str(Nx) + '.mp4')

    t1 = time.time()
    wv_grid.check_fluxes()
    t2 = time.time()
    print('Time to check all the flux constraints = ', t2 - t1, ' s')

    # t1 = time.time()
    # biotsavart_json_str = bs_wv.save(filename=OUT_DIR + 'BiotSavart.json')
    # bs_wv.set_points(s.gamma().reshape((-1, 3)))
    # trace_fieldlines(bs_wv, 'poincare_qa', 'qa', s_plot, comm, OUT_DIR)
    # t2 = time.time()
    print(OUT_DIR)

    t_end = time.time()
    print('Total time = ', t_end - t_start)

plt.show()
