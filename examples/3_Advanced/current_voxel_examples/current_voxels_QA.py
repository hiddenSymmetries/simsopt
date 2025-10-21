#!/usr/bin/env python
r"""
This example uses the current voxels method 
outlined in Kaptanoglu, Langlois & Landreman (2023) in order
to generate a figure-eight coil for the Landreman-Paul 
QA stellarator. Then this solution is used to initialize
a filament optimization of a single helical figure-eight
coil that further reduces the errors. 

The script should be run as:
    mpirun -n 1 python current_voxels_QA.py 
or
    python current_voxels_QA.py 
"""

import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field.biotsavart import BiotSavart
from simsopt.field import Coil, Current
from simsopt.field.magneticfieldclasses import CurrentVoxelsField
from simsopt.geo import CurrentVoxelsGrid
from simsopt.solve import relax_and_split_minres 
from simsopt.util.permanent_magnet_helper_functions import \
    make_filament_from_voxels, perform_filament_optimization, make_Bnormal_plots, \
    calculate_modB_on_major_radius
import time

t_start = time.time()

# Set surface parameters
nphi = 32
ntheta = nphi
input_name = 'input.LandremanPaul2021_QA'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range='half period', nphi=nphi, ntheta=ntheta)

# Make high-resolution, full-torus version of the surface for plots
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

# Finally, initialize the current voxels 
t1 = time.time()
Nx = 30  # 20 voxels in each direction in the voxel volume
kwargs = {}
kwargs = {"Nx": Nx, "Ny": Nx, "Nz": Nx}
current_voxels_grid = CurrentVoxelsGrid(
    s, s_inner, s_outer, **kwargs 
)

# Save vtks of the Itarget_curve, inner and outer toroidal surfaces (used to generate
# the volume of current voxels), and current voxel grid.
curves_to_vtk([current_voxels_grid.Itarget_curve], out_dir + "Itarget_curve")
current_voxels_grid.inner_toroidal_surface.to_vtk(out_dir + 'inner')
current_voxels_grid.outer_toroidal_surface.to_vtk(out_dir + 'outer')
current_voxels_grid.to_vtk_before_solve(out_dir + 'grid_before_solve_Nx' + str(Nx))
t2 = time.time()
print('WV grid initialization took time = ', t2 - t1, ' s')

# Optimize the voxels without sparsity to generate an initial guess for the full optimization
kappa = 1e-3  # Tikhonov regularization 
kwargs = {"out_dir": out_dir, "kappa": kappa, "max_iter": 5000, "precondition": True}
minres_dict = relax_and_split_minres( 
    current_voxels_grid, **kwargs 
)

# Optimize the voxels with the group sparsity term active
max_iter = 100  # max number of MINRES iterations for each call 
rs_max_iter = 100  # max number of relax-and-split iterations
nu = 1e2  # stength of the "relaxation" loss term 
l0_threshold = 1e4  # threshold for the group l0 loss term
l0_thresholds = np.linspace(  # Sequence of increasing thresholds
    l0_threshold, 100 * l0_threshold, 5, endpoint=True
)
kwargs['alpha0'] = minres_dict['alpha_opt']
kwargs['l0_thresholds'] = l0_thresholds
kwargs['nu'] = nu
kwargs['max_iter'] = max_iter
kwargs['rs_max_iter'] = rs_max_iter
minres_dict = relax_and_split_minres( 
    current_voxels_grid, **kwargs 
)

# Unpack final ouputs and save the solution to vtk
alpha_opt = minres_dict['alpha_opt']
fB = minres_dict['fB']
fI = minres_dict['fI']
fK = minres_dict['fK']
fRS = minres_dict['fRS']
f0 = minres_dict['f0']
fC = minres_dict['fC']
fminres = minres_dict['fminres']
current_voxels_grid.to_vtk_after_solve(out_dir + 'grid_after_Tikhonov_solve_Nx' + str(Nx))

# set up CurrentVoxels Bfield and check fB value
bs_current_voxels = CurrentVoxelsField(
    current_voxels_grid.J, 
    current_voxels_grid.XYZ_integration, 
    current_voxels_grid.grid_scaling, 
    nfp=s.nfp, 
    stellsym=s.stellsym
)
bs_current_voxels.set_points(s.gamma().reshape((-1, 3)))
Bnormal_current_voxels = np.sum(bs_current_voxels.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
normN = np.linalg.norm(s.normal().reshape(-1, 3), axis=-1)
fB_direct = SquaredFlux(s, bs_current_voxels, -current_voxels_grid.Bn).J() * 2 * s.nfp
print('fB = ', fB_direct)

# Check optimization achieved the desired Itarget value
gamma = current_voxels_grid.Itarget_curve.gamma().reshape(-1, 3)
bs_current_voxels.set_points(gamma)
gammadash = current_voxels_grid.Itarget_curve.gammadash().reshape(-1, 3)
Bnormal_Itarget_curve = np.sum(bs_current_voxels.B() * gammadash, axis=-1)
mu0 = 4 * np.pi * 1e-7
Itarget_check = np.sum(current_voxels_grid.Bn_Itarget) / mu0 / len(current_voxels_grid.Itarget_curve.quadpoints)
print('Itarget_check = ', Itarget_check)

# Save Bnormal plots from the voxel fields
make_Bnormal_plots(bs_current_voxels, s_plot, out_dir, f"biot_savart_current_voxels_{Nx}")

# Make a figure of algorithm progress
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

# Check the divergence-free constraints are well-satisfied
current_voxels_grid.check_fluxes()

# Check the average |B| along the major radius
calculate_modB_on_major_radius(bs_current_voxels, s)

t_end = time.time()
print('Total voxels time = ', t_end - t_start)
print('f fB fI fK fC')
print(f0[-1], fB[-1], fI[-1], fK[-1], fC[-1])

# Initialize a filament optimization from the current voxels solution
filament_curve = make_filament_from_voxels(current_voxels_grid, l0_thresholds[-1], num_fourier=30)
curves = [filament_curve]
curves_to_vtk(curves, out_dir + "filament_curve")
current = Current(current_voxels_grid.Itarget / 2.0)
current.fix_all()
coil = [Coil(filament_curve, current)]
bs = BiotSavart(coil)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(out_dir + "surf_filament_initial", extra_data=pointData)

# perform the filament optimization
perform_filament_optimization(s_plot, bs, curves)
plt.show()
