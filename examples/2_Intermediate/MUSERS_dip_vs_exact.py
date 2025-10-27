#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving permanent 
magnet optimization on the MUSE grid. This 
algorithm is described in the following paper:
    A. A. Kaptanoglu, R. Conlin, and M. Landreman, 
    Greedy permanent magnet optimization, 
    Nuclear Fusion 63, 036016 (2023)

The script should be run as:
    mpirun -n 1 python permanent_magnet_MUSE.py
on a cluster machine but 
    python permanent_magnet_MUSE.py
is sufficient on other machines. Note that this code does not use MPI, but is 
parallelized via OpenMP and XSIMD, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

For high-resolution and more realistic designs, please see the script files at
https://github.com/akaptano/simsopt_permanent_magnet_advanced_scripts.git
"""

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from simsopt.field import BiotSavart, DipoleField, ExactField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier, ExactMagnetGrid
from simsopt.objectives import SquaredFlux
from simsopt.solve import relax_and_split
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

nphi = 4  # >= 64 for high-resolution runs
max_iter = 10
downsample = 10

ntheta = nphi  # same as above
dr = 0.01  # Radial extent in meters of the cylindrical permanent magnet bricks
input_name = 'input.muse'

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / 'zot80.focus'
surface_filename = TEST_DIR / input_name

objectives = dict.fromkeys(['dipole_dipole',
                            'dipole_exact',
                            'exact_exact',
                            'exact_dipole'])

#################################################
#################################################

s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# these surfaces not referenced again after this, grid is loaded in
# maybe distance vs performance is not a necessary test with MUSE

# Make the output directory -- warning, saved data can get big!
# On NERSC, recommended to change this directory to point to SCRATCH!
out_str = f"MUSE_RS_output"
out_dir = Path(out_str)
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Load a downsampled version of the magnet grid from a FAMUS file
mag_data = FocusData(famus_filename, downsample=downsample)

# Set the allowable orientations of the magnets to be face-aligned
# i.e. the local x, y, or z directions
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

ophi = np.arctan2(mag_data.oy, mag_data.ox) 
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z
print('pol_vectors_shape = ', pol_vectors.shape)

# pol_vectors is only used for the greedy algorithms with cartesian coordinate_flag
# which is the default, so no need to specify it here. 
kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr}

# HERE IS WHERE A MATRIX IS ENCODED
# encoded in _opt_setup, not which grid initializer you use, so it should be good to go?
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

pm_comp = ExactMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

print('Number of available dipoles = ', pm_opt.ndipoles)

m0 = np.zeros(pm_opt.ndipoles * 3) 
reg_l0 = 0.0  # No sparsity
nu = 1e100
kwargs = initialize_default_kwargs()
kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
kwargs['max_iter'] = max_iter  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = 1  # Number of total iterations of the relax-and-split algorithm
kwargs['reg_l0'] = reg_l0
RS_history, m_history, m_proxy_history = relax_and_split(pm_opt, m0=m0, **kwargs)
m0 = pm_opt.m

# # Set final m to the minimum achieved during the optimization
# min_ind = np.argmin(RS_history)
# pm_opt.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0 
magnets = pm_opt.m.reshape(pm_opt.ndipoles, 3)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
# Save files
b_dipole = DipoleField(
        pm_opt.dipole_grid_xyz,
        pm_opt.m,
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima
    )
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / "dipdip_normal_Fields", pm_opt.dx, pm_opt.dy, pm_opt.dz)

dipfB = 0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2)
objectives['dipole_dipole'] = dipfB

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100

# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(b_dipole, s_plot, out_dir, "only_dipdip_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mdip_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
bs.set_points(s.gamma().reshape(-1, 3))
b_dipole.set_points(s.gamma().reshape(-1, 3))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s, b_dipole, -Bnormal).J()

total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max

# field for cubic magents in dipole optimization positions
b_comp = ExactField(
    pm_comp.pm_grid_xyz,
    pm_opt.m,
    pm_opt.dims,
    pm_opt.phiThetas,
    nfp = s.nfp,
    stellsym = s.stellsym,
    m_maxima = pm_opt.m_maxima
)

assert np.all(pm_comp.m == 0.0)
assert np.all(pm_comp.pm_grid_xyz == pm_opt.dipole_grid_xyz)
assert np.all(pm_comp.phiThetas == pm_opt.phiThetas)

assert pm_comp.dx == pm_opt.dx
assert pm_comp.dy == pm_opt.dy
assert pm_comp.dz == pm_opt.dz

b_comp.set_points(s_plot.gamma().reshape((-1, 3)))
b_comp._toVTK(out_dir / "exdip_normal_fields", pm_comp.dx, pm_comp.dy, pm_comp.dz)

# Print optimized metrics
assert all(pm_comp.b_obj == pm_opt.b_obj)
compfB = 0.5 * np.sum((pm_comp.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2)
objectives['dipole_exact'] = compfB

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bcnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")
Bcnormal_magnets = np.sum(b_comp.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bcnormal_total = Bcnormal + Bcnormal_magnets

# For plotting Bn on the full torus surface at the end with just the magnet fields
make_Bnormal_plots(b_comp, s_plot, out_dir, "only_exdip_optimized")
pointData = {"B_N": Bcnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mdip_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
b_comp.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
Bcnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
f_Bc_sf = SquaredFlux(s, b_comp, -Bcnormal).J()

print('FINISHED WITH DIPOLE')

#########################
###### NOW DO EXACT #####
#########################

print('BEGINNING EXACT')

s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# these surfaces not referenced again after this, grid is loaded in
# maybe distance vs performance is not a necessary test with MUSE

# Make the output directory -- warning, saved data can get big!
# On NERSC, recommended to change this directory to point to SCRATCH!
out_str = f"MUSE_RS_output/"
out_dir = Path(out_str)
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Load a downsampled version of the magnet grid from a FAMUS file
mag_data = FocusData(famus_filename, downsample=downsample)

# Set the allowable orientations of the magnets to be face-aligned
# i.e. the local x, y, or z directions
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

ophi = np.arctan2(mag_data.oy, mag_data.ox) 
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z
print('pol_vectors_shape = ', pol_vectors.shape)

# pol_vectors is only used for the greedy algorithms with cartesian coordinate_flag
# which is the default, so no need to specify it here. 
kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr}

# HERE IS WHERE A MATRIX IS ENCODED
# encoded in _opt_setup, not which grid initializer you use, so it should be good to go?
pm_opt = ExactMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

pm_comp = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

print('Number of available dipoles = ', pm_opt.ndipoles)

m0 = np.zeros(pm_opt.ndipoles * 3) 
reg_l0 = 0.0  # No sparsity
nu = 1e100
kwargs = initialize_default_kwargs()
kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
kwargs['max_iter'] = max_iter  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = 1  # Number of total iterations of the relax-and-split algorithm
kwargs['reg_l0'] = reg_l0
RS_history, m_history, m_proxy_history = relax_and_split(pm_opt, m0=m0, **kwargs)
m0 = pm_opt.m

# # Set final m to the minimum achieved during the optimization
# min_ind = np.argmin(RS_history)
# pm_opt.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0 
magnets = pm_opt.m.reshape(pm_opt.ndipoles, 3)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
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
b_magnet._toVTK(out_dir / "exex_normal_fields", pm_opt.dx, pm_opt.dy, pm_opt.dz)

# Print optimized metrics
fB = 0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2)
objectives['exact_exact'] = fB

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_coils")
Bnormal_magnets = np.sum(b_magnet.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_magnets
make_Bnormal_plots(bs + b_magnet, s_plot, out_dir, "biot_savart_optimized")

# Compute metrics with permanent magnet results
magnets_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(magnets_m ** 2, axis=-1)) / pm_opt.ndipoles * 100

# For plotting Bn on the full torus surface at the end with just the magnet fields
make_Bnormal_plots(b_magnet, s_plot, out_dir, "only_exex_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mex_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
b_magnet.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s, b_magnet, -Bnormal).J()

total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max

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
b_dipole._toVTK(out_dir / "dipex_normal_fields", pm_opt.dx, pm_opt.dy, pm_opt.dz)

# Print optimized metrics
fBc = 0.5 * np.sum((pm_comp.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2)
objectives['exact_dipole'] = fBc

make_Bnormal_plots(b_magnet, s_plot, out_dir, "only_dipex_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mex_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
b_dipole.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
Bcnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
f_Bc_sf = SquaredFlux(s, b_dipole, -Bcnormal).J()

total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max

########################
########################

df = pd.DataFrame([objectives])
df.to_csv(out_dir / 'objectives.csv', index=False)

t_end = time.time()
print('Total time = ', t_end - t_start)
# plt.show()
