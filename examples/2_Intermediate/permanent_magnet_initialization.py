#!/usr/bin/env python
r"""
In this example we initialize a grid of permanent magnets for
optimization in a separate script. If coils are needed, 
we also solve a FOCUS like Stage II coil optimisation problem
for a vacuum field, so the target is just zero.

Several pre-set permanent magnet grids are available for the
LandremanPaul QA and QH plasma surfaces, the MUSE surface, 
and the NCSX C09R00 half-Tesla surface.

"""

import os
import sys
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.field.magneticfieldclasses import DipoleField, ToroidalField
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import ScaledCurrent, Current, Coil, coils_via_symmetries
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from permanent_magnet_helpers import *
import time

t_start = time.time()

config_flag, res_flag, initialization_run, dr, coff, poff, surface_flag, input_name, nphi, ntheta, pms_name, is_premade_famus_grid, cylindrical_flag = read_input()

# Don't save in home directory on NERSC -- save on SCRATCH
scratch_path = '/global/cscratch1/sd/akaptano/' 

# Make the output directory
OUT_DIR = scratch_path + config_flag + "_nphi{0:d}_ntheta{1:d}_dr{2:.2e}_coff{3:.2e}_poff{4:.2e}/".format(nphi, ntheta, dr, coff, poff)
print("Output directory = ", OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
class_filename = "PM_optimizer_" + config_flag

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
t1 = time.time()
if surface_flag == 'focus': 
    s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta) 
elif surface_flag == 'wout': 
    s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta) 
else:
    s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta) 
t2 = time.time()
print("Done loading in plasma boundary surface, t = ", t2 - t1)

# File for the desired TF coils
t1 = time.time()

# Don't have NCSX TF coils, just the Bn field on the surface 
# so have to treat the NCSX example separately.
quadpoints_phi = np.linspace(0, 1, 2 * nphi, endpoint=True)
qphi = len(quadpoints_phi)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
if config_flag != 'ncsx':
    t1 = time.time()

    # initialize the coils
    base_curves, curves, coils = initialize_coils(config_flag, TEST_DIR, OUT_DIR)

    # Set up BiotSavart fields
    bs = BiotSavart(coils)

    # Calculate average, approximate on-axis B field strength
    calculate_on_axis_B(bs, s)

    t2 = time.time()
    print("Done setting up biot savart, ", t2 - t1, " s")

    # Make higher resolution surface for plotting Bnormal
    if surface_flag == 'focus': 
        s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    elif surface_flag == 'wout': 
        s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    else:
        s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

    # Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "biot_savart_init", extra_data=pointData)

    # If BiotSavart not yet optimized, optimize it
    if 'muse' not in config_flag: 
        s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, config_flag)

        # check after-optimization average on-axis magnetic field strength
        calculate_on_axis_B(bs, s)

    # Save optimized BiotSavart object
    bs.set_points(s.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
    biotsavart_json_str = bs.save(filename=OUT_DIR + 'BiotSavart.json')

    # Plot Bnormal on plasma surface from optimized BiotSavart coils
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_plot = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    f_B_sf = SquaredFlux(s_plot, bs).J()
    print('BiotSavart f_B = ', f_B_sf)
    pointData = {"B_N": Bnormal_plot[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "biot_savart_opt", extra_data=pointData)
else:
    # Set up the contribution to Bnormal from a purely toroidal field.
    # Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
    net_poloidal_current_Amperes = 3.7713e+6
    mu0 = 4 * np.pi * (1e-7)
    RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
    print('B0 of toroidal field = ', RB)
    tf = ToroidalField(R0=1, B0=RB)

    # Check average on-axis magnetic field strength
    calculate_on_axis_B(tf, s)

    # Calculate Bnormal
    tf.set_points(s.gamma().reshape((-1, 3)))
    Bnormal = np.sum(tf.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

    # Plot Bnormal on plasma surface from optimized BiotSavart coils
    s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    tf.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_plot = np.sum(tf.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
    pointData = {"B_N": Bnormal_plot[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "biot_savart_init", extra_data=pointData)

# Finally, initialize the permanent magnet class
t1 = time.time()
pm_opt = PermanentMagnetOptimizer(
    s, is_premade_famus_grid=is_premade_famus_grid, coil_offset=coff, 
    dr=dr, plasma_offset=poff, Bn=Bnormal, 
    filename=surface_filename, surface_flag=surface_flag, out_dir=OUT_DIR,
    cylindrical_flag=cylindrical_flag, pms_name=pms_name,
)
t2 = time.time()
print('Done initializing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')

# If using a pre-made FAMUS grid of permanent magnet
# locations, save the FAMUS grid and FAMUS solution.
if is_premade_famus_grid:
    t1 = time.time()
    read_FAMUS_grid(pms_name, pm_opt, s, s_plot, Bnormal, Bnormal_plot, OUT_DIR)
    t2 = time.time()
    print('Saving FAMUS solution took ', t2 - t1, ' s')

    # Plot the REGCOIL_PM solution for the NCSX example
    if config_flag == 'ncsx':
        read_regcoil_pm('../../tests/test_files/regcoil_pm_ncsx.nc', surface_filename, OUT_DIR)

# check nothing got mistranslated by plotting the initial dipole guess
t1 = time.time()
pm_opt.m = pm_opt.m0
pm_opt.m_proxy = pm_opt.m0
b_dipole_initial = DipoleField(pm_opt)
b_dipole_initial.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole_initial._toVTK(OUT_DIR + "Dipole_Fields_initial")

# Plot total Bnormal after dipoles are initialized to their initial guess
Nnorms = np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1)))
Ngrid = pm_opt.nphi * pm_opt.ntheta
Bnormal_init = np.sum(b_dipole_initial.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_Am = ((pm_opt.A_obj.dot(pm_opt.m0)) * np.sqrt(Ngrid / Nnorms)).reshape(nphi, ntheta)
print(Bnormal_init, Bnormal_plot, Bnormal_Am)
pointData = {"B_N": Bnormal_init[:, :, None]}
s_plot.to_vtk(OUT_DIR + "Bnormal_dipoles_init_guess", extra_data=pointData)
pointData = {"B_N": Bnormal_Am[:, :, None]}
s.to_vtk(OUT_DIR + "Bnormal_dipoles_init_guess_Am", extra_data=pointData)
pointData = {"B_N": (Bnormal_init + Bnormal_plot)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "Bnormal_init_guess", extra_data=pointData)
f_B_init = SquaredFlux(s, b_dipole_initial, -Bnormal).J()

f_B_Am = np.linalg.norm(pm_opt.A_obj.dot(pm_opt.m0) - pm_opt.b_obj, ord=2) ** 2 / 2.0
print('f_B (total with initial SIMSOPT dipoles) = ', f_B_init)
print('f_B (A * m with initial SIMSOPT dipoles) = ', f_B_Am)

t2 = time.time()
print('Done setting up the Dipole Field class')
print('Process took t = ', t2 - t1, ' s')

t1 = time.time()
# Save PM class object to file for optimization
file_out = open(OUT_DIR + class_filename + ".pickle", "wb")

# SurfaceRZFourier objects not pickle-able, so set to None
# Presumably can fix this with Bharats new json functionality
pm_opt.plasma_boundary = None
pm_opt.rz_inner_surface = None
pm_opt.rz_outer_surface = None
print(file_out)
pickle.dump(pm_opt, file_out)
t2 = time.time()
print('Pickling took ', t2 - t1, ' s')
t_end = time.time()
print('In total, script took ', t_end - t_start, ' s')

# Show the generated figures 
plt.show()
