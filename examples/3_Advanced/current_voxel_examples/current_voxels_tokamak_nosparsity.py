#!/usr/bin/env python
r"""
This example uses the current voxels method 
outline in Kaptanoglu, Langlois & Landreman 2024 in order
to make finite-build coils with the voxels method:

Kaptanoglu, A. A., Langlois, G. P., & Landreman, M. (2024). 
Topology optimization for inverse magnetostatics as sparse regression: 
Application to electromagnetic coils for stellarators. 
Computer Methods in Applied Mechanics and Engineering, 418, 116504.

The script should be run as:
    mpirun -n 1 python current_voxels_tokamak_nosparsity.py
or
    python current_voxels_tokamak_nosparsity.py

"""

import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import CurrentVoxelsField
from simsopt.geo import CurrentVoxelsGrid
from simsopt.solve import relax_and_split_minres
from simsopt.util import make_curve_at_theta0, calculate_modB_on_major_radius
from simsopt.util import in_github_actions
import time
t_start = time.time()

t1 = time.time()
# Set some parameters
if in_github_actions:
    nphi = 16  # nphi = ntheta >= 64 needed for accurate full-resolution runs
else:
    nphi = 64
ntheta = nphi
# coil_range = 'full torus'
coil_range = 'half period'
input_name = 'input.circular_tokamak' 

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
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
out_dir = 'current_voxels_axisymmetric_convex/'
os.makedirs(out_dir, exist_ok=True)

# No external coils
Bnormal = np.zeros((nphi, ntheta))
t2 = time.time()
print('First setup took time = ', t2 - t1, ' s')

# Define a curve to define a Itarget loss term
# As the boundary of the stellarator at theta = 0
t1 = time.time()
numquadpoints = nphi * s.nfp * 2
curve = make_curve_at_theta0(s, numquadpoints)
curves_to_vtk([curve], out_dir + "Itarget_curve")
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
s_out.to_vtk(out_dir + "surf_out")

#create the inside boundary for the PMs
s_in = SurfaceRZFourier.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=coil_range, nfp=s.nfp, stellsym=s.stellsym)
s_in.set_rc(0, 0, s.get_rc(0, 0) * fac1)
s_in.set_rc(1, 0, s.get_rc(1, 0) * fac2)
s_in.set_zs(1, 0, s.get_rc(1, 0) * fac2)
s_in.to_vtk(out_dir + "surf_in")

Nx = 30
Ny = Nx
Nz = Nx 
# Finally, initialize the current voxels 
t1 = time.time()
kwargs = {}
kwargs = {"Nx": Nx, "Ny": Nx, "Nz": Nx, "Itarget": Itarget}
current_voxels_grid = CurrentVoxelsGrid(
    s, s_in, s_out, **kwargs 
)
current_voxels_grid.inner_toroidal_surface.to_vtk(out_dir + 'inner')
current_voxels_grid.outer_toroidal_surface.to_vtk(out_dir + 'outer')
current_voxels_grid.to_vtk_before_solve(out_dir + 'grid_before_solve_Nx' + str(Nx))
t2 = time.time()
print('WV grid initialization took time = ', t2 - t1, ' s')

t1 = time.time()
kappa = 1e-5
kwargs = {"out_dir": out_dir, "kappa": kappa, "max_iter": 5000, "precondition": True}
minres_dict = relax_and_split_minres( 
    current_voxels_grid, **kwargs 
)
alpha_opt = minres_dict['alpha_opt']
fB = minres_dict['fB']
fI = minres_dict['fI']
fK = minres_dict['fK']
fRS = minres_dict['fRS']
f0 = minres_dict['f0']
fC = minres_dict['fC']
fminres = minres_dict['fminres']
t2 = time.time()
print('MINRES solve time = ', t2 - t1, ' s')    

curves_to_vtk([current_voxels_grid.Itarget_curve], out_dir + "Itarget_curve")
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
calculate_modB_on_major_radius(bs_current_voxels, s_plot)
bs_current_voxels.set_points(s_plot.gamma().reshape((-1, 3)))
Bn_voxels = np.sum(bs_current_voxels.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
pointData = {"B_N": Bn_voxels[:, :, None]}
s_plot.to_vtk(out_dir + "surf_voxels_full", extra_data=pointData)

# Make a figure of algorithm progress
plt.figure()
plt.semilogy(fB, 'r', label=r'$f_B$')
plt.semilogy(fK, 'b', label=r'$\lambda \|\alpha\|^2$')
plt.semilogy(fC, 'c', label=r'$f_C$')
plt.semilogy(fI, 'm', label=r'$f_I$')
plt.semilogy(fminres, 'k--', label=r'MINRES residual')
plt.grid(True)
plt.legend()
# plt.savefig(out_dir + 'optimization_progress.jpg')

# Check the divergence-free constraints are well-satisfied
# current_voxels_grid.check_fluxes()

t_end = time.time()
print('Total voxels time = ', t_end - t_start)
plt.show()
