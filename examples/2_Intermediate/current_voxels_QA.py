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
import simsoptpp as sopp
import simsopt
from simsopt.geo import SurfaceRZFourier, Curve, curves_to_vtk
from scipy.optimize import minimize
from simsopt.objectives import SquaredFlux
from simsopt.objectives import Weight
from simsopt.field.biotsavart import BiotSavart
from simsopt.field import InterpolatedField, SurfaceClassifier, Coil
from simsopt.field.magneticfieldclasses import CurrentVoxelsField
from simsopt.geo import CurrentVoxelsGrid
from simsopt.solve import relax_and_split_minres 
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.objectives import QuadraticPenalty
from simsopt.field import Current
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LpCurveTorsion
import time

t_start = time.time()

# Set some parameters
nphi = 32  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = nphi
input_name = 'input.LandremanPaul2021_QA'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range='half period', nphi=nphi, ntheta=ntheta)

qphi = s.nfp * nphi * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename,
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
out_dir = 'current_voxels_QA/'
os.makedirs(out_dir, exist_ok=True)

# Initialize an inner and outer toroidal surface using normal vectors of the plasma boundary
poff = 0.5
coff = 0.3
s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_projected_normal(poff)
s_outer.extend_via_projected_normal(poff + coff)

Nx = 20
Ny = Nx
Nz = Nx

# Finally, initialize the current voxels 
t1 = time.time()
kwargs = {}
kwargs = {"Nx": Nx, "Ny": Ny, "Nz": Nz}
current_voxels_grid = CurrentVoxelsGrid(
    s, s_inner, s_outer, **kwargs 
)
curves_to_vtk([current_voxels_grid.Itarget_curve], out_dir + f"Itarget_curve")
current_voxels_grid.inner_toroidal_surface.to_vtk(out_dir + 'inner')
current_voxels_grid.outer_toroidal_surface.to_vtk(out_dir + 'outer')
current_voxels_grid.to_vtk_before_solve(out_dir + 'grid_before_solve_Nx' + str(Nx))
t2 = time.time()
print('WV grid initialization took time = ', t2 - t1, ' s')

max_iter = 100  # 10000
rs_max_iter = 100  # 100 # 1
nu = 1e2
kappa = 1e-3
l0_threshold = 1e4
l0_thresholds = np.linspace(l0_threshold, 100 * l0_threshold, 5, endpoint=True)
kwargs = {"out_dir": out_dir, "kappa": kappa, "max_iter": 5000, "precondition": True}
alpha_opt, fB, fK, fI, fRS, f0, fC, fminres, fBw, fKw, fIw, fRSw, fCw, _, _, _ = relax_and_split_minres( 
    current_voxels_grid, **kwargs 
)
kwargs['alpha0'] = alpha_opt
kwargs['l0_thresholds'] = l0_thresholds
kwargs['nu'] = nu
kwargs['max_iter'] = max_iter
kwargs['rs_max_iter'] = rs_max_iter
alpha_opt, fB, fK, fI, fRS, f0, fC, fminres, fBw, fKw, fIw, fRSw, fCw, _, _, _ = relax_and_split_minres( 
    current_voxels_grid, **kwargs 
)
t2 = time.time()
print('(Preconditioned) MINRES solve time = ', t2 - t1, ' s')    

t1 = time.time()
current_voxels_grid.to_vtk_after_solve(out_dir + 'grid_after_Tikhonov_solve_Nx' + str(Nx))
t2 = time.time()
print('Time to plot the optimized grid = ', t2 - t1, ' s')
print('fB after optimization = ', fB[-1]) 
print('fB check = ', 0.5 * np.linalg.norm(current_voxels_grid.B_matrix @ alpha_opt - current_voxels_grid.b_rhs) ** 2 * s.nfp * 2)

# set up CurrentVoxels Bfield
bs_current_voxels = CurrentVoxelsField(
    current_voxels_grid.J, 
    current_voxels_grid.XYZ_integration, 
    current_voxels_grid.grid_scaling, 
    nfp=s.nfp, 
    stellsym=s.stellsym
)
t1 = time.time()
bs_current_voxels.set_points(s.gamma().reshape((-1, 3)))
Bnormal_current_voxels = np.sum(bs_current_voxels.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
normN = np.linalg.norm(s.normal().reshape(-1, 3), axis=-1)
contig = np.ascontiguousarray
print('fB direct = ', np.sum(normN * np.ravel(Bnormal_current_voxels + current_voxels_grid.Bn) ** 2) * 0.5 / (nphi * ntheta) * s.nfp * 2)

t2 = time.time()
print('Time to compute Bnormal_current_voxels = ', t2 - t1, ' s')
fB_direct = SquaredFlux(s, bs_current_voxels, -current_voxels_grid.Bn).J() * 2 * s.nfp
print('fB_direct = ', fB_direct)

gamma = current_voxels_grid.Itarget_curve.gamma().reshape(-1, 3)
bs_current_voxels.set_points(gamma)
gammadash = current_voxels_grid.Itarget_curve.gammadash().reshape(-1, 3)
Bnormal_Itarget_curve = np.sum(bs_current_voxels.B() * gammadash, axis=-1)
mu0 = 4 * np.pi * 1e-7
Itarget_check = np.sum(current_voxels_grid.Bn_Itarget) / mu0 / len(current_voxels_grid.Itarget_curve.quadpoints)
print('Itarget_check = ', Itarget_check)
print('Itarget second check = ', current_voxels_grid.Itarget_matrix @ alpha_opt / mu0) 

t1 = time.time()
make_Bnormal_plots(bs_current_voxels, s_plot, out_dir, "biot_savart_current_voxels_Nx" + str(Nx))
t2 = time.time()

print('Time to plot Bnormal_current_voxels = ', t2 - t1, ' s')

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

# plt.savefig(out_dir + 'optimization_progress.jpg')
t1 = time.time()
#current_voxels_grid.check_fluxes()
t2 = time.time()
print('Time to check all the flux constraints = ', t2 - t1, ' s')

calculate_on_axis_B(bs_current_voxels, s)

print(out_dir)


t_end = time.time()
print('Total time = ', t_end - t_start)
print('f fB fI fK fC')
print(f0[-1], fB[-1], fI[-1], fK[-1], fC[-1])

filament_curve = make_filament_from_voxels(current_voxels_grid, l0_thresholds[-1], num_fourier=30)
curves_to_vtk([filament_curve], out_dir + f"filament_curve")
current = Current(current_voxels_grid.Itarget / 2.0)
current.fix_all()
coil = [Coil(filament_curve, current)]
print('coil dofs = ', coil[0].dof_names)
bs = BiotSavart(coil)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(out_dir + "surf_filament_initial", extra_data=pointData)
make_Bnormal_plots(bs, s_plot, out_dir, "surf_filament_initial2")

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(5e-7)

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0
CS_WEIGHT = 1e-22

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 0.01
CURVATURE_WEIGHT = 1e-22
TORSION_WEIGHT = 1e-22

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 0.01
MSC_WEIGHT = 1e-22

# Define the individual terms objective function:
Jf = SquaredFlux(s_plot, bs)
Jls = [CurveLength(filament_curve)]
Jcsdist = CurveSurfaceDistance([filament_curve], s_plot, CS_THRESHOLD)
Jcs = [LpCurveCurvature(filament_curve, 2, CURVATURE_THRESHOLD)]
Jts = [LpCurveTorsion(filament_curve, 2, CURVATURE_THRESHOLD)]
Jmscs = [MeanSquaredCurvature(filament_curve)]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * sum(Jls) \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + TORSION_WEIGHT * sum(Jts) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in [filament_curve])
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
MAXITER = 2000
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
curves_to_vtk([filament_curve], out_dir + f"curves_opt_long")
pointData = {"B_N": np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(out_dir + "surf_opt_long", extra_data=pointData)
calculate_on_axis_B(bs, s_plot)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
bs.save(out_dir + "biot_savart_opt.json")
trace_field = True
if trace_field:
    t1 = time.time()
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    print('R0 = ', s.get_rc(0, 0), ', r0 = ', s.get_rc(1, 0))
    n = 20
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 2

    # compute the fieldlines from the initial locations specified above
    sc_fieldline = SurfaceClassifier(s_plot, h=0.03, p=2)

    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym, skip=skip
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    make_Bnormal_plots(bsh, s_plot, out_dir, "biot_savart_interpolated")
    make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_filaments")
    Bh = bsh.B()
    B = bs.B()
    print("Mean(|B|) on plasma surface =", np.mean(bs_current_voxels.AbsB()))
    print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
    nfieldlines = 30
    R0 = np.linspace(1.2125346, 1.295, nfieldlines)
    trace_fieldlines(bsh, 'current_voxels_QA_long', s_plot, None, R0, out_dir)
    t2 = time.time()

plt.show()
