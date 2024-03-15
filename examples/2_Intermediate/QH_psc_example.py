#!/usr/bin/env python
r"""
"""
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.geo.psc_grid import PSCgrid
from simsopt.objectives import SquaredFlux
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *
import time


# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
else:
    nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = 64
    # Make higher resolution surface for plotting Bnormal
    qphi = nphi * 8
    quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta * 4, endpoint=True)

coff = 1.0  # PM grid starts offset ~ 10 cm from the plasma surface
poff = 1.0  # PM grid end offset ~ 15 cm from the plasma surface
input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'

# Read in the plas/ma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
range_param = 'half period'
s = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi, ntheta=ntheta
)
print('s.R = ', s.get_rc(0, 0))
print('s.r = ', s.get_rc(1, 0))

# input_name = 'tests/test_files/input.circular_tokamak'
# s_inner = SurfaceRZFourier(
#     stellsym=True,
#     quadpoints_phi=s.quadpoints_phi, 
#     quadpoints_theta=s.quadpoints_theta
# )
# s_inner.set_rc(0, 0, s.get_rc(0, 0))
# s_inner.set_rc(1, 0, s.get_rc(1, 0))
# s_inner.set_zs(1, 0, s.get_rc(1, 0))
# s_outer = SurfaceRZFourier(
#     stellsym=True,
#     quadpoints_phi=s.quadpoints_phi, 
#     quadpoints_theta=s.quadpoints_theta
# )
# s_outer.set_rc(0, 0, s.get_rc(0, 0))
# s_outer.set_rc(1, 0, s.get_rc(1, 0))
# s_outer.set_zs(1, 0, s.get_rc(1, 0))
s_inner = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4
)
s_outer = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4
)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

# Make the output directory
out_str = "QH_psc_output/"
out_dir = Path("QH_psc_output")
out_dir.mkdir(parents=True, exist_ok=True)
s_inner.to_vtk(out_str + 'inner_surf')
s_outer.to_vtk(out_str + 'outer_surf')

# initialize the coils
base_curves, curves, coils = initialize_coils('qh', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, 
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils
bs = coil_optimization(s, bs, base_curves, curves, out_dir)
curves_to_vtk(curves, out_dir / "TF_coils", close=True)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized")

# check after-optimization average on-axis magnetic field strength
B_axis = calculate_on_axis_B(bs, s)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized", B_axis)

# Finally, initialize the psc class
kwargs_geo = {"Nx": 4, "out_dir": out_str, "random_initialization": True, "poff": poff} 
# note Bnormal below is additional Bnormal (not from the TF coils or the PSCs)
print(Bnormal.shape)
psc_array = PSCgrid.geo_setup_between_toroidal_surfaces(
    s, np.zeros(Bnormal.shape), bs, s_inner, s_outer,  **kwargs_geo
)

# print('Currents = ', psc_array.I)
psc_array.setup_psc_biotsavart()
make_Bnormal_plots(psc_array.B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
make_Bnormal_plots(psc_array.B_PSC_all, s_plot, out_dir, "biot_savart_PSC_initial_all", B_axis)
# make_Bnormal_plots(psc_array.B_PSC_direct, s_plot, out_dir, "biot_savart_PSC_initial_direct")
# make_Bnormal_plots(psc_array.B_TF, s_plot, out_dir, "biot_savart_InterpolatedTF", B_axis)
make_Bnormal_plots(bs + psc_array.B_PSC, s_plot, out_dir, "PSC_and_TF_initial", B_axis)

# Try and optimize with scipy
t1 = time.time()
x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
psc_array.least_squares(x0)
t2 = time.time()
print('Time for call to least-squares = ', t2 - t1)
# psc_array.least_squares(x0 + np.random.rand(len(x0)))

psc_array.setup_psc_biotsavart()
# make_Bnormal_plots(psc_array.B_PSC_all, s_plot, out_dir, "PSC_initial", B_axis)
# make_Bnormal_plots(bs + psc_array.B_PSC_all, s_plot, out_dir, "PSC_and_TF_initial", B_axis)

from scipy.optimize import minimize
fB = SquaredFlux(s, bs, np.zeros((nphi, ntheta))).J()
print('fB only TF coils = ', fB / (B_axis ** 2 * s.area()))
psc_array.least_squares(np.zeros(x0.shape))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
print('fB only TF direct = ', np.sum(Bnormal.reshape(-1) ** 2 * psc_array.grid_normalization ** 2
                                    ) / (2 * B_axis ** 2 * s.area()))
# exit()
make_Bnormal_plots(psc_array.B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
make_Bnormal_plots(psc_array.B_PSC_all, s_plot, out_dir, "biot_savart_PSC_initial_all", B_axis)
fB = SquaredFlux(s, psc_array.B_PSC, np.zeros((nphi, ntheta))).J()
print(fB/ (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, psc_array.B_PSC_all + bs, np.zeros((nphi, ntheta))).J()
print('fB with both, before opt = ', fB / (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, psc_array.B_PSC_all, -Bnormal).J()
print('fB with both (minus sign), before opt = ', fB / (B_axis ** 2 * s.area()))
psc_array.least_squares(np.zeros(x0.shape))
# exit()

print('beginning optimization: ')
options = {"disp": True, "maxiter": 100}
x0 = np.random.rand(len(np.ravel(np.array([psc_array.alphas, psc_array.deltas]
                                          )))) * 2 * np.pi
print(x0)
x0 = psc_array.kappas
# print(x0)
x_opt = minimize(psc_array.least_squares, x0, tol=1e-20, options=options)
# print(x_opt)
# print('currents = ', psc_array.I)
psc_array.setup_curves()
psc_array.plot_curves('final_')
# Check that direct Bn calculation agrees with optimization calculation
psc_array.setup_psc_biotsavart()
fB = SquaredFlux(s, psc_array.B_PSC_all + bs, np.zeros((nphi, ntheta))).J()
print('fB with both, after opt = ', fB / (B_axis ** 2 * s.area()))
make_Bnormal_plots(psc_array.B_PSC_all, s_plot, out_dir, "PSC_final", B_axis)
make_Bnormal_plots(bs + psc_array.B_PSC_all, s_plot, out_dir, "PSC_and_TF_final", B_axis)
print('end')
# plt.show()
