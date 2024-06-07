#!/usr/bin/env python
r"""
This example script optimizes a set of relatively simple toroidal field coils
and passive superconducting coils (PSCs)
for an ARIES-CS reactor-scale version of the precise-QH stellarator from 
Landreman and Paul. 

The script should be run as:
    mpirun -n 1 python NCSX_psc_example.py
on a cluster machine but 
    python NCSX_psc_example.py
is sufficient on other machines. Note that this code does not use MPI, but is 
parallelized via OpenMP, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

"""
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.geo.psc_grid import PSCgrid
from simsopt.objectives import SquaredFlux
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *
import time

# np.random.seed(1)  # set a seed so that the same PSCs are initialized each time

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
else:
    # Resolution needs to be reasonably high if you are doing permanent magnets
    # or small coils because the fields are quite local
    nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    # Make higher resolution surface for plotting Bnormal
    qphi = nphi * 4
    quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta * 4, endpoint=True)

poff = 0.2  # PSC grid will be offset 'poff' meters from the plasma surface
coff = 0.6  # PSC grid will be initialized between 1 m and 2 m from plasma

# Read in the plasma equilibrium file
input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
range_param = 'half period'
s = SurfaceRZFourier.from_wout(
    surface_filename, range=range_param, nphi=nphi, ntheta=ntheta
)
# Print major and minor radius
print('s.R = ', s.get_rc(0, 0))
print('s.r = ', s.get_rc(1, 0))

# Make inner and outer toroidal surfaces very high resolution,
# which helps to initialize coils precisely between the surfaces. 
s_inner = SurfaceRZFourier.from_wout(
    surface_filename, range=range_param, nphi=nphi * 2, ntheta=ntheta * 2
)
s_outer = SurfaceRZFourier.from_wout(
    surface_filename, range=range_param, nphi=nphi * 2, ntheta=ntheta * 2
)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

# Make the output directory
out_str = "NCSX_psc_output/"
out_dir = Path("NCSX_psc_output")
out_dir.mkdir(parents=True, exist_ok=True)

# Save the inner and outer surfaces for debugging purposes
s_inner.to_vtk(out_str + 'inner_surf')
s_outer.to_vtk(out_str + 'outer_surf')

# initialize the coils

def initialize_coils_NCSX():
    """
    Initializes NCSX coils
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import curves_to_vtk

    # generate planar TF coils
    ncoils = 2
    R0 = 1.5
    R1 = 1.4
    order = 5

    from simsopt.mhd.vmec import Vmec
    total_current = Vmec(surface_filename).external_current() / (2 * s.nfp) / 7.131 * 6.5
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
    base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    # fix all the coil shapes so only the currents are optimized
    for i in range(ncoils):
        base_curves[i].fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir / "curves_init")
    return base_curves, curves, coils

base_curves, curves, coils = initialize_coils_NCSX()
currents = np.array([coil.current.get_value() for coil in coils])

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, 
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils and plot results
# fix all the coil shapes so only the currents are optimized
# for i in range(ncoils):
#     base_curves[i].fix_all()
bs = coil_optimization(s, bs, base_curves, curves, out_dir)
curves_to_vtk(curves, out_dir / "TF_coils", close=True)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized")

# check after-optimization average on-axis magnetic field strength
B_axis = calculate_on_axis_B(bs, s)
# B_axis = 1.0  # Don't rescale
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized", B_axis)

# Finally, initialize the psc class
kwargs_geo = {"Nx": 12, "out_dir": out_str, 
              # "initialization": "plasma",
              "poff": poff,
              "plasma_boundary_full": s_plot} 
psc_array = PSCgrid.geo_setup_between_toroidal_surfaces(
    s, coils, s_inner, s_outer,  **kwargs_geo
)
print('x = ', psc_array.kappas)
print('Number of PSC locations = ', len(psc_array.grid_xyz))

currents = []
for i in range(psc_array.num_psc):
    currents.append(Current(psc_array.I[i]))
all_coils = coils_via_symmetries(
    psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
)
B_PSC = BiotSavart(all_coils)
# Plot initial errors from only the PSCs, and then together with the TF coils
make_Bnormal_plots(B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
make_Bnormal_plots(bs + B_PSC, s_plot, out_dir, "PSC_and_TF_initial", B_axis)

# from simsopt.field import BiotSavart, InterpolatedField
# Bn = psc_array.Bn_PSC_full.reshape(qphi, 4 * ntheta)[:, :, None] * 1e-7 / B_axis
# pointData = {"B_N": Bn}
# s_plot.to_vtk(out_dir / "direct_Bn_PSC", extra_data=pointData)
# Bn = psc_array.b_opt.reshape(nphi, ntheta)[:, :, None] * 1e-7 / B_axis
# pointData = {"B_N": Bn}
# s.to_vtk(out_dir / "direct_Bn_TF", extra_data=pointData)

# Check SquaredFlux values using different ways to calculate it
x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
fB = SquaredFlux(s, bs, np.zeros((nphi, ntheta))).J()
print('fB only TF coils = ', fB / (B_axis ** 2 * s.area()))
bs.set_points(s.gamma().reshape(-1, 3))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
print('fB only TF direct = ', np.sum(Bnormal.reshape(-1) ** 2 * psc_array.grid_normalization ** 2
                                    ) / (2 * B_axis ** 2 * s.area()))
make_Bnormal_plots(B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
fB = SquaredFlux(s, B_PSC, np.zeros((nphi, ntheta))).J()
print(fB/ (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, B_PSC + bs, np.zeros((nphi, ntheta))).J()
print('fB with both, before opt = ', fB / (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, B_PSC, -Bnormal).J()
print('fB with both (minus sign), before opt = ', fB / (B_axis ** 2 * s.area()))

# exit()
# Actually do the minimization now
from scipy.optimize import minimize
# from scipy.optimize import lbfgsb


print('beginning optimization: ')
eps = 1e-6
options = {"disp": True, "maxiter": 40}
verbose = True

# Run STLSQ with BFGS in the loop
kwargs_manual = {
                 "out_dir": out_str, 
                 "plasma_boundary" : s,
                 "coils_TF" : coils
                 }

from scipy.optimize import approx_fprime, check_grad, basinhopping, dual_annealing, direct, differential_evolution, OptimizeResult
# from scipy.optimize import lbfgsb
def callback(x):
    print('fB: ', psc_array.least_squares(x))
    print('approx: ', approx_fprime(x, psc_array.least_squares, 1E-8))
    print('exact: ', psc_array.least_squares_jacobian(x))
    print('x = ', x)
    print('-----')
    print(check_grad(psc_array.least_squares, psc_array.least_squares_jacobian, x) / np.linalg.norm(psc_array.least_squares_jacobian(x)))
    
def callback_annealing(x, f, context):
    print('fB: ', psc_array.least_squares(x))
    return (context == 100)

# print('Dual annealing: ')
# t1 = time.time()
# x_opt = dual_annealing(psc_array.least_squares, opt_bounds, callback=callback_annealing, maxiter=30)
# t2 = time.time()
# print('Dual annealing time: ', t2 - t1)

# x0 = np.zeros(2 * psc_array.num_psc)  # np.hstack(((np.random.rand(psc_array.num_psc) - 0.5) * np.pi, (np.random.rand(psc_array.num_psc) - 0.5) * 2 * np.pi))
I_threshold = 6e4
I_threshold_scaling = 1.1
STLSQ_max_iters = 10
BdotN2_list = []
num_pscs = []
for k in range(STLSQ_max_iters):
    I_threshold *= I_threshold_scaling
    x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
    num_pscs.append(len(x0) // 2)
    print('Number of PSCs = ', len(x0) // 2, ' in iteration ', k)
    print('I_threshold = ', I_threshold)
    opt_bounds1 = tuple([(-np.pi / 2.0 + eps, np.pi / 2.0 - eps) for i in range(psc_array.num_psc)])
    opt_bounds2 = tuple([(-np.pi + eps, np.pi - eps) for i in range(psc_array.num_psc)])
    opt_bounds = np.vstack((opt_bounds1, opt_bounds2))
    opt_bounds = tuple(map(tuple, opt_bounds))
    x_opt = minimize(psc_array.least_squares, 
                     x0, 
                     args=(verbose,),
                     method='L-BFGS-B',
                     bounds=opt_bounds,
                     jac=psc_array.least_squares_jacobian, 
                     options=options,
                     tol=1e-20,
                      # callback=callback
                     )
    psc_array.setup_curves()
    psc_array.plot_curves('final_Ithresh_{0:.3e}_'.format(I_threshold))
    currents = []
    for i in range(psc_array.num_psc):
        currents.append(Current(psc_array.I[i]))
    all_coils = coils_via_symmetries(
        psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
    )
    B_PSC = BiotSavart(all_coils)

    # Check that direct Bn calculation agrees with optimization calculation
    fB = SquaredFlux(s, B_PSC + bs, np.zeros((nphi, ntheta))).J()
    print('fB with both, after opt = ', fB / (B_axis ** 2 * s.area()))
    make_Bnormal_plots(B_PSC, s_plot, out_dir, 'PSC_final_Ithresh_{0:.3e}'.format(I_threshold), B_axis)
    make_Bnormal_plots(bs + B_PSC, s_plot, out_dir, 'PSC_and_TF_final_Ithresh_{0:.3e}'.format(I_threshold), B_axis)
    I = psc_array.I
    grid_xyz = psc_array.grid_xyz
    alphas = psc_array.alphas
    deltas = psc_array.deltas
    if len(BdotN2_list) > 0:
        print(BdotN2_list, np.array(psc_array.BdotN2_list))
        BdotN2_list = np.hstack((BdotN2_list, np.array(psc_array.BdotN2_list)))
    else:
        BdotN2_list = np.array(psc_array.BdotN2_list)
    big_I_inds = np.ravel(np.where(np.abs(I) > I_threshold))
    if len(big_I_inds) != psc_array.num_psc:
        grid_xyz = grid_xyz[big_I_inds, :]
        alphas = alphas[big_I_inds]
        deltas = deltas[big_I_inds]
    else:
        print('STLSQ converged, breaking out of loop')
        break
    kwargs_manual["alphas"] = alphas
    kwargs_manual["deltas"] = deltas
    # Initialize new PSC array with coils only at the remaining locations
    # with initial orientations from the solve using BFGS
    psc_array = PSCgrid.geo_setup_manual(
        grid_xyz, psc_array.R, **kwargs_manual
    )
BdotN2_list = np.ravel(BdotN2_list)
    
from matplotlib import pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.semilogy(BdotN2_list)
plt.subplot(1, 2, 2)
plt.plot(num_pscs)
    
# psc_array.setup_orientations(x_opt.x[:len(x_opt) // 2], x_opt.x[len(x_opt) // 2:])

# N = 20
# alphas = np.linspace(-np.pi, np.pi, N)
# deltas = np.linspace(-np.pi, np.pi, N)
# fB = np.zeros((N, N))
# for i in range(N):
#     for j in range(N):
#         if len(psc_array.alphas[1:]) > 1:
#             alphas_i = np.hstack((alphas[i], psc_array.alphas[1:]))
#             deltas_j = np.hstack((deltas[j], psc_array.deltas[1:]))
#         else:
#             alphas_i = alphas[i]
#             deltas_j = deltas[j]

#         kappas = np.hstack((alphas_i, deltas_j))
#         fB[i, j] = psc_array.least_squares(kappas)
# plt.figure()
# plt.contourf(alphas, deltas, fB.T) # np.log10(fB.T))
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$\delta$')
# # plt.legend([r'$\log(f_B)$'])
# plt.colorbar()

# if len(psc_array.alphas[1:]) > 1:
#     fB = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             alphas_i = np.hstack((alphas[i], np.hstack((alphas[j], psc_array.alphas[2:]))))
#             kappas = np.hstack((alphas_i, psc_array.deltas))
#             fB[i, j] = psc_array.least_squares(kappas)
#     plt.figure()
#     plt.contourf(alphas, deltas, fB.T) # np.log10(fB.T))
#     plt.xlabel(r'$\alpha$')
#     plt.ylabel(r'$\delta$')
#     # plt.legend([r'$\log(f_B)$'])
#     plt.colorbar()
plt.show()
print('end')
