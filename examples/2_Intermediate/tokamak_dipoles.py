#!/usr/bin/env python
r"""
"""

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField, Current, \
    coils_via_symmetries
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier, \
    curves_to_vtk, create_equally_spaced_curves
from simsopt.solve import GPMO
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
else:
    nphi = 16  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    Nx = 40  # cartesian bricks but note that we are not modelling the cubic geometry!
    Ny = Nx
    Nz = Nx

coff = 0.1  # PM grid starts offset ~ 10 cm from the plasma surface
poff = 0.25  # PM grid end offset ~ 15 cm from the plasma surface
input_name = 'input.circular_tokamak'

# Read in the plas/ma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
range_param = 'full torus'
s = SurfaceRZFourier.from_vmec_input(surface_filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range=range_param, nphi=nphi, ntheta=ntheta)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

s.stellsym=False
s_inner.stellsym=False
s_outer.stellsym=False

# Make the output directory
out_dir = Path("tokamak_dipole")
out_dir.mkdir(parents=True, exist_ok=True)

s_inner.to_vtk(out_dir / 's_inner')
s_outer.to_vtk(out_dir / 's_outer')

# initialize the coils
# generate planar TF coils
def initialize_tokamak_coils():
    ncoils = 4
    R0 = s.get_rc(0, 0)  # get the major radius of the tokamak
    R1 = s.get_rc(1, 0) * 2  # get minor radius times some factor
    order = 1

    total_current = 1e6  # 1 MA total current through all the coils
    base_curves = create_equally_spaced_curves(
        ncoils, nfp=1, stellsym=False, R0=R0, R1=R1, order=order, numquadpoints=128
    )
    base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    # fix all the coil shapes and currents so that these coils are 
    # completely fixed
    for i in range(ncoils):
        base_curves[i].fix_all()
        base_currents[i].fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    return base_curves, curves, coils

base_curves, curves, coils = initialize_tokamak_coils()
curves_to_vtk(curves, out_dir / "curves_init")

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
s_plot.stellsym=False

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Finally, initialize the permanent magnet class
kwargs_geo = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "coordinate_flag": "cartesian"}  
pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
    s, Bnormal, s_inner, s_outer, **kwargs_geo
)

# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs('GPMO')
nIter_max = 10000
# max_nMagnets = 400
algorithm = 'baseline'
# algorithm = 'ArbVec_backtracking'
# nBacktracking = 200 
# nAdjacent = 10
# thresh_angle = np.pi  # / np.sqrt(2)
nHistory = 50
# angle = int(thresh_angle * 180 / np.pi)

kwargs['K'] = nIter_max
kwargs['nhistory'] = nHistory
if algorithm == 'backtracking' or algorithm == 'ArbVec_backtracking':
    kwargs['backtracking'] = nBacktracking
    kwargs['Nadjacent'] = nAdjacent
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_opt.pm_grid_xyz)
    if algorithm == 'ArbVec_backtracking':
        kwargs['thresh_angle'] = thresh_angle
        kwargs['max_nMagnets'] = max_nMagnets
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
dt = time.time() - t1
print('GPMO took t = ', dt, ' s')

# Save files
if True:
    # Make BiotSavart object from the dipoles and plot solution 
    b_dipole = DipoleField(
        pm_opt.dipole_grid_xyz,
        pm_opt.m,
        stellsym=s_plot.stellsym,
        nfp=s_plot.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )
    print('bfield = ',b_dipole)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole._toVTK(out_dir / "Dipole_Fields")
    make_Bnormal_plots(bs + b_dipole, s_plot, out_dir, "biot_savart_optimized")
    Bnormal_coils = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal_coils + Bnormal_dipoles 
    pointData = {"B_N": Bnormal_dipoles[:, :, None]}
    s_plot.to_vtk(out_dir / "Bnormal_dipoles", extra_data=pointData)
    pointData = {"B_N": Bnormal_coils[:, :, None]}
    s_plot.to_vtk(out_dir / "Bnormal_coils", extra_data=pointData)
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(out_dir / "Bnormal_total", extra_data=pointData)
    # pm_opt.write_to_famus(out_dir)
    np.savetxt(out_dir / 'R2_history.txt', R2_history)
    np.savetxt(out_dir / 'absBn_history.txt', Bn_history)
    nmags = m_history.shape[0]
    nhist = m_history.shape[2]
    m_history_2d = m_history.reshape((nmags*m_history.shape[1], nhist))
    np.savetxt(out_dir / 'm_history_nmags={nmags}_nhist={nhist}.txt', m_history_2d)
t_end = time.time()  
print('Script took in total t = ', t_end - t_start, ' s')

# Plot optimization results as function of iterations
plt.figure()
plt.semilogy(R2_history, label=r'$f_B$')
plt.semilogy(Bn_history, label=r'$<|Bn|>$')
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Metric values')
plt.legend()
plt.show()