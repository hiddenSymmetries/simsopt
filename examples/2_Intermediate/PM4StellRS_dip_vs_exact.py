#!/usr/bin/env python
'''
    This example sript uses the GPMO backtracking algorithm with 
    arbitrarily-defined polarization vectors to optimize magnets in an 
    arrangement around the NCSX plasma in the C09R00 configuration with 
    an average magnetic field strength on axis of about 0.5 T.
    
    This test applies the backtracking approach to the PM4Stell magnet arrangement
    with face-triplet polarizations. The threshold angle for removal of magnet 
    pairs is set by the user, unlike the normal backtracking algorithm
    which defaults to threshold_angle = 180 degrees (only remove adjacent 
    dipoles if they are exactly equal and opposite).

    PM4Stell functionality and related code obtained courtesy of
    Ken Hammond and the PM4Stell + MAGPIE teams. 
'''

from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from simsopt.field import BiotSavart, DipoleField, Coil, ExactField
from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid, ExactMagnetGrid
from simsopt.solve import relax_and_split
from simsopt.util.permanent_magnet_helper_functions \
    import initialize_default_kwargs, make_Bnormal_plots
from simsopt.util import FocusPlasmaBnormal, FocusData, read_focus_coils, in_github_actions
from simsopt.util.polarization_project import (polarization_axes, orientation_phi,
                                               discretize_polarizations)
from simsopt.objectives import SquaredFlux

t_start = time.time()

N = 8  # >= 64 for high-resolution runs
nIter_max = 10
downsample = 10

nphi = N
ntheta = N
# angle = int(thresh_angle * 180 / np.pi)
out_dir = Path("PM4StellRS_comp") 
out_dir.mkdir(parents=True, exist_ok=True)
print('out directory = ', out_dir)

# Obtain the plasma boundary for the NCSX configuration
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
fname_plasma = TEST_DIR / 'c09r00_B_axis_half_tesla_PM4Stell.plasma'

objectives = dict.fromkeys(['dipole_dipole',
                            'dipole_exact',
                            'exact_exact',
                            'exact_dipole'])

######################
######################

lcfs_ncsx = SurfaceRZFourier.from_focus(
    fname_plasma, range='half period', nphi=nphi, ntheta=ntheta
)
s1 = SurfaceRZFourier.from_focus(
    fname_plasma, range='half period', nphi=nphi, ntheta=ntheta
)
s2 = SurfaceRZFourier.from_focus(
    fname_plasma, range='half period', nphi=nphi, ntheta=ntheta
)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    fname_plasma,
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Obtain the normal field on the plasma boundary arising from plasma currents
bnormal_obj_ncsx = FocusPlasmaBnormal(fname_plasma)
bn_plasma = bnormal_obj_ncsx.bnormal_grid(nphi, ntheta, 'half period')

# Obtain the NCSX TF coil data and calculate their normal field on the boundary
fname_ncsx_coils = TEST_DIR / 'tf_only_half_tesla_symmetry_baxis_PM4Stell.focus'
base_curves, base_currents, ncoils = read_focus_coils(fname_ncsx_coils)
coils = []
for i in range(ncoils):
    coils.append(Coil(base_curves[i], base_currents[i]))
base_currents[0].fix_all()

# fix all the coil shapes
for i in range(ncoils):
    base_curves[i].fix_all()

# Obtain Bnormal from the plasma and the coils
ncsx_tfcoils = coils
bs_tfcoils = BiotSavart(ncsx_tfcoils)
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
bn_tfcoils = np.sum(
    bs_tfcoils.B().reshape((nphi, ntheta, 3)) * lcfs_ncsx.unitnormal(), 
    axis=2
)
bn_total = bn_plasma + bn_tfcoils
make_Bnormal_plots(bs_tfcoils, s_plot, out_dir, "biot_savart_initial")

# Obtain data on the magnet arrangement
fname_argmt = TEST_DIR / 'magpie_trial104b_PM4Stell.focus'
fname_corn = TEST_DIR / 'magpie_trial104b_corners_PM4Stell.csv'
mag_data = FocusData(fname_argmt, downsample=downsample)
nMagnets_tot = mag_data.nMagnets

# Determine the allowable polarization types and reject the negatives
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))
pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
pol_type = np.concatenate((pol_type, pol_type_fe_ftri))
pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
pol_type = np.concatenate((pol_type, pol_type_fc_ftri))

# Read in the phi coordinates and set the pol_vectors
ophi = orientation_phi(fname_corn)[:nMagnets_tot]
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((nMagnets_tot, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z

kwargs_geo = {"pol_vectors": pol_vectors, "downsample": downsample}

# Initialize the permanent magnet grid from the PM4Stell arrangement
pm_ncsx = PermanentMagnetGrid.geo_setup_from_famus(
    lcfs_ncsx, bn_total, fname_argmt, **kwargs_geo
)

pm_comp = ExactMagnetGrid.geo_setup_from_famus(
    lcfs_ncsx, bn_total, fname_argmt, **kwargs_geo
)

pm_ncsx.phiThetas = np.zeros((nMagnets_tot, 2))
pm_ncsx.phiThetas[:,0] = ophi

# Optimize with the GPMO algorithm
m0 = np.zeros(pm_ncsx.ndipoles * 3) 
reg_l0 = 0.0  # No sparsity
nu = 1e100
kwargs = initialize_default_kwargs()
kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
kwargs['max_iter'] = nIter_max  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = 1  # Number of total iterations of the relax-and-split algorithm
kwargs['reg_l0'] = reg_l0
RS_history, m_history, m_proxy_history = relax_and_split(pm_ncsx, m0=m0, **kwargs)
m0 = pm_ncsx.m

# # Set final m to the minimum achieved during the optimization
# min_ind = np.argmin(RS_history)
# pm_ncsx.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0 
magnets = pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3)

# Save files
b_dipole = DipoleField(
        pm_ncsx.dipole_grid_xyz,
        pm_ncsx.m,
        nfp=lcfs_ncsx.nfp,
        coordinate_flag=pm_ncsx.coordinate_flag,
        m_maxima=pm_ncsx.m_maxima
    )
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / "dipdip_normal_Fields", pm_ncsx.dx, pm_ncsx.dy, pm_ncsx.dz)

dipfB = 0.5 * np.sum((pm_ncsx.A_obj @ pm_ncsx.m - pm_ncsx.b_obj) ** 2)
objectives['dipole_dipole'] = dipfB

bs_tfcoils.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs_tfcoils, s_plot, out_dir, "biot_savart_optimized")
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles

# Compute metrics with permanent magnet results
dipoles_m = pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_ncsx.ndipoles * 100

# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(b_dipole, s_plot, out_dir, "only_dipdip_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mdip_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape(-1, 3))
b_dipole.set_points(lcfs_ncsx.gamma().reshape(-1, 3))
Bnormal = np.sum(bs_tfcoils.B().reshape((nphi, ntheta, 3)) * lcfs_ncsx.unitnormal(), axis=2)
f_B_sf = SquaredFlux(lcfs_ncsx, b_dipole, -Bnormal).J()

total_volume = np.sum(np.sqrt(np.sum(pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3) ** 2, axis=-1))) * lcfs_ncsx.nfp * 2 * mu0 / B_max

# field for cubic magents in dipole optimization positions
b_comp = ExactField(
    pm_comp.pm_grid_xyz,
    pm_ncsx.m,
    pm_ncsx.dims,
    pm_ncsx.phiThetas,
    nfp = lcfs_ncsx.nfp,
    stellsym = lcfs_ncsx.stellsym,
    m_maxima = pm_ncsx.m_maxima
)

assert np.all(pm_comp.m == 0.0)
assert np.all(pm_comp.pm_grid_xyz == pm_ncsx.dipole_grid_xyz)

assert pm_comp.dx == pm_ncsx.dx
assert pm_comp.dy == pm_ncsx.dy
assert pm_comp.dz == pm_ncsx.dz

b_comp.set_points(s_plot.gamma().reshape((-1, 3)))
b_comp._toVTK(out_dir / "exdip_normal_fields", pm_comp.dx, pm_comp.dy, pm_comp.dz)

# Print optimized metrics
assert all(pm_comp.b_obj == pm_ncsx.b_obj)
compfB = 0.5 * np.sum((pm_comp.A_obj @ pm_ncsx.m - pm_ncsx.b_obj) ** 2)
objectives['dipole_exact'] = compfB

bs_tfcoils.set_points(s_plot.gamma().reshape((-1, 3)))
Bcnormal = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs_tfcoils, s_plot, out_dir, "biot_savart_optimized")
Bcnormal_magnets = np.sum(b_comp.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bcnormal_total = Bcnormal + Bcnormal_magnets

# For plotting Bn on the full torus surface at the end with just the magnet fields
make_Bnormal_plots(b_comp, s_plot, out_dir, "only_exdip_optimized")
pointData = {"B_N": Bcnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mdip_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
b_comp.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
Bcnormal = np.sum(bs_tfcoils.B().reshape((nphi, ntheta, 3)) * lcfs_ncsx.unitnormal(), axis=2)
f_Bc_sf = SquaredFlux(lcfs_ncsx, b_comp, -Bcnormal).J()

#################################
#### NOW OPTIMIZE WITH EXACT ####
#################################

lcfs_ncsx = SurfaceRZFourier.from_focus(
    fname_plasma, range='half period', nphi=nphi, ntheta=ntheta
)
s1 = SurfaceRZFourier.from_focus(
    fname_plasma, range='half period', nphi=nphi, ntheta=ntheta
)
s2 = SurfaceRZFourier.from_focus(
    fname_plasma, range='half period', nphi=nphi, ntheta=ntheta
)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    fname_plasma,
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Obtain the normal field on the plasma boundary arising from plasma currents
bnormal_obj_ncsx = FocusPlasmaBnormal(fname_plasma)
bn_plasma = bnormal_obj_ncsx.bnormal_grid(nphi, ntheta, 'half period')

# Obtain the NCSX TF coil data and calculate their normal field on the boundary
fname_ncsx_coils = TEST_DIR / 'tf_only_half_tesla_symmetry_baxis_PM4Stell.focus'
base_curves, base_currents, ncoils = read_focus_coils(fname_ncsx_coils)
coils = []
for i in range(ncoils):
    coils.append(Coil(base_curves[i], base_currents[i]))
base_currents[0].fix_all()

# fix all the coil shapes
for i in range(ncoils):
    base_curves[i].fix_all()

# Obtain Bnormal from the plasma and the coils
ncsx_tfcoils = coils
bs_tfcoils = BiotSavart(ncsx_tfcoils)
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
bn_tfcoils = np.sum(
    bs_tfcoils.B().reshape((nphi, ntheta, 3)) * lcfs_ncsx.unitnormal(), 
    axis=2
)
bn_total = bn_plasma + bn_tfcoils
make_Bnormal_plots(bs_tfcoils, s_plot, out_dir, "biot_savart_initial")

# Obtain data on the magnet arrangement
fname_argmt = TEST_DIR / 'magpie_trial104b_PM4Stell.focus'
fname_corn = TEST_DIR / 'magpie_trial104b_corners_PM4Stell.csv'
mag_data = FocusData(fname_argmt, downsample=downsample)
nMagnets_tot = mag_data.nMagnets

# Determine the allowable polarization types and reject the negatives
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))
pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
pol_type = np.concatenate((pol_type, pol_type_fe_ftri))
pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
pol_type = np.concatenate((pol_type, pol_type_fc_ftri))

# Read in the phi coordinates and set the pol_vectors
ophi = orientation_phi(fname_corn)[:nMagnets_tot]
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((nMagnets_tot, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z

kwargs_geo = {"pol_vectors": pol_vectors, "downsample": downsample}

# Initialize the permanent magnet grid from the PM4Stell arrangement
pm_comp = PermanentMagnetGrid.geo_setup_from_famus(
    lcfs_ncsx, bn_total, fname_argmt, **kwargs_geo
)

pm_ncsx = ExactMagnetGrid.geo_setup_from_famus(
    lcfs_ncsx, bn_total, fname_argmt, **kwargs_geo
)

pm_ncsx.phiThetas = np.zeros((nMagnets_tot, 2))
pm_ncsx.phiThetas[:,0] = ophi

# Optimize with the GPMO algorithm
m0 = np.zeros(pm_ncsx.ndipoles * 3) 
reg_l0 = 0.0  # No sparsity
nu = 1e100
kwargs = initialize_default_kwargs()
kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
kwargs['max_iter'] = nIter_max  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = 1  # Number of total iterations of the relax-and-split algorithm
kwargs['reg_l0'] = reg_l0
RS_history, m_history, m_proxy_history = relax_and_split(pm_ncsx, m0=m0, **kwargs)
m0 = pm_ncsx.m

# # Set final m to the minimum achieved during the optimization
# min_ind = np.argmin(RS_history)
# pm_ncsx.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0 
magnets = pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3)

b_magnet = ExactField(
        pm_ncsx.pm_grid_xyz,
        pm_ncsx.m,
        pm_ncsx.dims,
        pm_ncsx.phiThetas,
        stellsym=s_plot.stellsym,
        nfp=s_plot.nfp,
        m_maxima=pm_ncsx.m_maxima,
    )
b_magnet.set_points(s_plot.gamma().reshape((-1, 3)))
b_magnet._toVTK(out_dir / "exex_normal_fields", pm_ncsx.dx, pm_ncsx.dy, pm_ncsx.dz)

# Print optimized metrics
fB = 0.5 * np.sum((pm_ncsx.A_obj @ pm_ncsx.m - pm_ncsx.b_obj) ** 2)
objectives['exact_exact'] = fB

bs_tfcoils.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs_tfcoils.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs_tfcoils, s_plot, out_dir, "biot_savart_coils")
Bnormal_magnets = np.sum(b_magnet.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_magnets
make_Bnormal_plots(bs_tfcoils + b_magnet, s_plot, out_dir, "biot_savart_optimized")

# Compute metrics with permanent magnet results
magnets_m = pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(magnets_m ** 2, axis=-1)) / pm_ncsx.ndipoles * 100

# For plotting Bn on the full torus surface at the end with just the magnet fields
make_Bnormal_plots(b_magnet, s_plot, out_dir, "only_exex_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mex_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
b_magnet.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs_tfcoils.B().reshape((nphi, ntheta, 3)) * lcfs_ncsx.unitnormal(), axis=2)
f_B_sf = SquaredFlux(lcfs_ncsx, b_magnet, -Bnormal).J()

total_volume = np.sum(np.sqrt(np.sum(pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3) ** 2, axis=-1))) * lcfs_ncsx.nfp * 2 * mu0 / B_max

assert np.all(pm_comp.m == 0.0)
assert np.all(pm_comp.dipole_grid_xyz == pm_ncsx.pm_grid_xyz)

assert pm_comp.dx == pm_ncsx.dx
assert pm_comp.dy == pm_ncsx.dy
assert pm_comp.dz == pm_ncsx.dz

assert all(pm_comp.b_obj == pm_ncsx.b_obj)

b_dipole = DipoleField(
    pm_comp.dipole_grid_xyz,
    pm_ncsx.m,
    nfp=lcfs_ncsx.nfp,
    coordinate_flag=pm_ncsx.coordinate_flag, #check this one, which flag to use
    m_maxima=pm_ncsx.m_maxima
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / "dipex_normal_fields", pm_ncsx.dx, pm_ncsx.dy, pm_ncsx.dz)

# Print optimized metrics
fBc = 0.5 * np.sum((pm_comp.A_obj @ pm_ncsx.m - pm_ncsx.b_obj) ** 2)
objectives['exact_dipole'] = fBc

make_Bnormal_plots(b_magnet, s_plot, out_dir, "only_dipex_optimized")
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "mex_optimized", extra_data=pointData)

# Print optimized f_B and other metrics
b_dipole.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
Bcnormal = np.sum(bs_tfcoils.B().reshape((nphi, ntheta, 3)) * lcfs_ncsx.unitnormal(), axis=2)
f_Bc_sf = SquaredFlux(lcfs_ncsx, b_dipole, -Bcnormal).J()

total_volume = np.sum(np.sqrt(np.sum(pm_ncsx.m.reshape(pm_ncsx.ndipoles, 3) ** 2, axis=-1))) * lcfs_ncsx.nfp * 2 * mu0 / B_max

######################
######################

df = pd.DataFrame([objectives])
df.to_csv(out_dir / 'objectives.csv', index=False)

# Plot optimization results as function of iterations
# plt.figure()
# plt.semilogy(RS_history, label=r'$f_B$')
# plt.semilogy(Bn_history, label=r'$<|Bn|>$')
# plt.grid(True)
# plt.xlabel('K')
# plt.ylabel('Metric values')
# plt.legend()
# plt.show()
