#!/usr/bin/env python
r"""
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from simsopt.field import BiotSavart, ExactField, DipoleField
from simsopt.geo import ExactMagnetGrid, SurfaceRZFourier, PermanentMagnetGrid
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO, relax_and_split
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    dx = 0.05  # bricks with radial extent 5 cm
else:
    nphi = 32  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    Nx = 32 # bricks with radial extent ??? cm

coff = 0.2  # PM grid starts offset ~ 10 cm from the plasma surface
poff = 0.1  # PM grid end offset ~ 15 cm from the plasma surface
input_name = 'input.LandremanPaul2021_QA_lowres'

# Read in the plas/ma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_projected_normal(poff)
s_outer.extend_via_projected_normal(poff + coff)

# Make the output directory
out_str = f"exact_QA_noSparsity"
out_dir = Path(out_str)
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, 
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils
bs = coil_optimization(s, bs, base_curves, curves, out_dir)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# check after-optimization average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Finally, initialize the permanent magnet class
kwargs_geo = {"Nx": Nx} #, "Ny": Nx * 2, "Nz": Nx * 3}  
pm_opt = ExactMagnetGrid.geo_setup_between_toroidal_surfaces(
    s, Bnormal, s_inner, s_outer, **kwargs_geo
)

kwargs_geo = {"Nx":Nx}
pm_comp = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
    s, Bnormal, s_inner, s_outer, **kwargs_geo
)

print(pm_opt.phiThetas)

m0 = np.zeros(pm_opt.ndipoles * 3)
reg_l0 = 0.05  # Threshold off magnets with 5% or less strength
nu = 1e10  # how strongly to make proxy variable w close to values in m
reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu) 
# reg_l0 = 0.0  # No sparsity
# nu = 1e100
kwargs = initialize_default_kwargs()
kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
kwargs['max_iter'] = 1000  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = 40  # Number of total iterations of the relax-and-split algorithm
kwargs['reg_l0'] = reg_l0
print('nu, lambda = ', kwargs['nu'], kwargs['reg_l0'])
# RS_history, m_history, m_proxy_history = relax_and_split(pm_opt, m0=m0, **kwargs)
# m0 = pm_opt.m

# Optimize the permanent magnets. This actually solves
# 2 full relax-and-split problems, and uses the result of each
# problem to initialize the next, increasing L0 threshold each time,
# until thresholding over all magnets with strengths < 50% the max.
m0 = np.zeros(pm_opt.ndipoles * 3) 
total_m_history = []
total_mproxy_history = []
total_RS_history = []
for i in range(20):
    print('Relax-and-split iteration ', i, ', L0 threshold = ', reg_l0)
    reg_l0_scaled = reg_l0 * (i + 1) / 2.0
    kwargs['reg_l0'] = reg_l0_scaled
    RS_history, m_history, m_proxy_history = relax_and_split(pm_opt, m0=m0, **kwargs)
    total_RS_history.append(RS_history)
    total_m_history.append(m_history)
    total_mproxy_history.append(m_proxy_history)
    m_norms = np.linalg.norm(m_proxy_history[-1].reshape(-1, 3), axis=-1)
    # bar_x = np.linspace(0, np.max(m_norms), 20)
    plt.figure()
    plt.hist(m_norms, bins=100)
    m0 = pm_opt.m

plt.show()
# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0 
magnets = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(magnets ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(magnets ** 2, axis=-1))))

print('magnets have dimensions ',pm_opt.dims, ' with volume v = ',np.prod(pm_opt.dims))

b_magnet = ExactField(
    pm_opt.pm_grid_xyz,
    pm_opt.m,
    pm_opt.dims,
    pm_opt.phiThetas,
    stellsym=s_plot.stellsym,
    nfp=s_plot.nfp,
    m_maxima=pm_opt.m_maxima,
)
b_magnet.set_points(s_plot.gamma().reshape((-1, 3)))
b_magnet._toVTK(out_dir / "magnet_fields", pm_opt.dx, pm_opt.dy, pm_opt.dz)

# Print optimized metrics
fB = 0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2)
print("Total fB = ", fB)

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_coils")
Bnormal_magnets = np.sum(b_magnet.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_magnets
make_Bnormal_plots(bs + b_magnet, s_plot, out_dir, "biot_savart_optimized")

# Compute metrics with permanent magnet results
magnets_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(magnets_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible magnets = ", pm_opt.ndipoles)
print("% of magnets that are nonzero = ", num_nonzero)

# For plotting Bn on the full torus surface at the end with just the magnet fields
make_Bnormal_plots(b_magnet, s_plot, out_dir, "only_m_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "m_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
b_magnet.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s, b_magnet, -Bnormal).J()

print('f_B = ', f_B_sf)
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

assert np.all(pm_comp.m == 0.0)
assert np.all(pm_comp.dipole_grid_xyz == pm_opt.pm_grid_xyz)
assert np.all(pm_comp.phiThetas == pm_opt.phiThetas)

assert pm_comp.dx == pm_opt.dx
assert pm_comp.dy == pm_opt.dy
assert pm_comp.dz == pm_opt.dz

assert all(pm_comp.b_obj == pm_opt.b_obj)

b_dipole = DipoleField(
    pm_comp.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag, #check this one, which flag to use
    m_maxima=pm_opt.m_maxima
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / "dipole_magnet_fields", pm_opt.dx, pm_opt.dy, pm_opt.dz)

# Print optimized metrics
fBc = 0.5 * np.sum((pm_comp.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2)
print("Total fBc = ", fBc)

# Print optimized f_B and other metrics
b_dipole.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
Bcnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
f_Bc_sf = SquaredFlux(s, b_dipole, -Bcnormal).J()

print('f_Bc = ', f_Bc_sf)
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

print('fB diff (ex - dip) = ', fB - fBc)
print('f_B (squared flux function) diff', f_B_sf - f_Bc_sf)

t_end = time.time()
print('Total time = ', t_end - t_start)
plt.show()

