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
from scipy.optimize import minimize
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, Coil, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, \
    MeanSquaredCurvature, LpCurveCurvature
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from permanent_magnet_helpers import *
import time

t_start = time.time()
# Determine which plasma equilibrium is being used
print("Usage requires a configuration flag chosen from: qa(_nonplanar), qh(_nonplanar), muse, ncsx, and a flag specifying high or low resolution")
if len(sys.argv) < 3:
    print(
        "Error! You must specify at least 1 arguments: "
        "the configuration flag and the resolution flag"
    )
    exit(1)
config_flag = str(sys.argv[1])
if config_flag not in ['qa', 'qa_nonplanar', 'qh', 'qh_nonplanar', 'muse', 'ncsx']:
    print(
        "Error! The configuration flag must specify one of "
        "the pre-set plasma equilibria: qa, qa_nonplanar, "
        "qh, qh_nonplanar, muse, or ncsx. "
    )
    exit(1)
res_flag = str(sys.argv[2])
if res_flag not in ['low', 'high']:
    print(
        "Error! The resolution flag must specify one of "
        "low or high."
    )
    exit(1)
print('Config flag = ', config_flag, ', Resolution flag = ', res_flag)

# Pre-set parameters for each configuration
FOCUS = False
cylindrical_flag = True
is_premade_ncsx = False
if res_flag == 'high':
    nphi = 32
    ntheta = 32
else:
    nphi = 8
    ntheta = 8
if config_flag == 'muse':
    dr = 0.01
    coff = 0.04
    poff = 0.05
    FOCUS = True
    input_name = config_flag 
elif 'qa' in config_flag or 'qh' in config_flag:
    dr = 0.02
    coff = 0.05
    poff = 0.1
    input_name = 'LandremanPaul2021_' + config_flag[:2].upper()
elif config_flag == 'ncsx':
    dr = 0.02
    coff = 0.02
    poff = 0.1
    FOCUS = True
    input_name = 'NCSX_c09r00_halfTeslaTF' 
    coil_name = 'input.NCSX_c09r00_halfTeslaTF_Bn'

# Don't save in home directory on NERSC -- save on SCRATCH
scratch_path = '/global/cscratch1/sd/akaptano/' 

# Make the output directory
OUT_DIR = scratch_path + config_flag + "_nphi{0:d}_ntheta{1:d}_dr{2:.2e}_coff{3:.2e}_poff{4:.2e}/".format(nphi, ntheta, dr, coff, poff)
os.makedirs(OUT_DIR, exist_ok=True)
class_filename = "PM_optimizer_" + config_flag

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / ('input.' + input_name)

# Initialize the boundary magnetic surface:
t1 = time.time()
if config_flag == 'muse' or config_flag == 'ncsx':
    s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta) 
else:
    s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta) 
t2 = time.time()
print("Done loading in plasma boundary surface, t = ", t2 - t1)


# File for the desired TF coils
t1 = time.time()
if config_flag == 'muse':
    # Load in pre-optimized coils
    TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
    coils_filename = TEST_DIR / (config_flag + '_tf_coils.focus')
    base_curves, base_currents, ncoils = read_focus_coils(coils_filename)
    coils = []
    for i in range(ncoils):
        coils.append(Coil(base_curves[i], base_currents[i]))
elif config_flag == 'qa':
    # generate planar TF coils
    ncoils = 6
    R0 = 1.0
    R1 = 0.5
    order = 5

    # Create the initial coils:
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order)
    base_currents = [Current(1e5) for i in range(ncoils)]

    # fix one of the currents:
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    base_currents[0].fix_all()

    # fix all the coil shapes so only the currents are optimized 
    for i in range(ncoils):
        base_curves[i].fix_all()
elif config_flag == 'qa_nonplanar':
    # generate planar TF coils
    ncoils = 1
    R0 = 1.0
    R1 = 0.5
    order = 5

    # Create the initial coils:
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order)
    base_currents = [Current(6e5) for i in range(ncoils)]

    # fix one of the currents but shape can vary:
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    base_currents[0].fix_all()

# Don't have NCSX TF coils, just the Bn field on the surface
if config_flag != 'ncsx':
    # Set up BiotSavart fields
    t1 = time.time()
    bs = BiotSavart(coils)
    bspoints = np.zeros((nphi, 3))
    R0 = s.get_rc(0, 0)
    for i in range(nphi):
        bspoints[i] = np.array([R0 * np.cos(s.quadpoints_phi[i]), R0 * np.sin(s.quadpoints_phi[i]), 0.0]) 
    bs.set_points(bspoints)
    B0 = np.linalg.norm(bs.B(), axis=-1)
    B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
    surface_area = s.area()
    bnormalization = B0avg * surface_area
    print("Bmag at R = ", R0, ", Z = 0: ", B0) 
    print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg) 
    bs.set_points(s.gamma().reshape((-1, 3)))
    t2 = time.time()
    print("Done setting up biot savart, ", t2 - t1, " s")

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_init")
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
    t2 = time.time()
    print("Done loading in coils", t2 - t1, " s")

    # If BiotSavart not yet optimized, optimize it
    if 'qa' in config_flag or 'qh' in config_flag:
        quadpoints_phi = np.linspace(0, 1, 2 * nphi, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
        if config_flag == 'muse' or config_flag == 'ncsx':
            s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        else:
            s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot)
        bs.set_points(bspoints)
        B0 = np.linalg.norm(bs.B(), axis=-1)
        B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
        surface_area = s.area()
        bnormalization = B0avg * surface_area
        print("After coil opt: Bmag at R = ", R0, ", Z = 0: ", B0) 
        print("After coil opt: toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg) 
        bs.set_points(s.gamma().reshape((-1, 3)))

    # Save optimized BiotSavart object
    biotsavart_json_str = bs.save(filename=OUT_DIR + 'BiotSavart.json')
    Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
else:
    Bnormal = load_ncsx_coil_data(s, coil_name)
    is_premade_ncsx = True

# permanent magnet setup 
t1 = time.time()
pm_opt = PermanentMagnetOptimizer(
    s, is_premade_ncsx=is_premade_ncsx, coil_offset=coff, dr=dr, plasma_offset=poff,
    Bn=Bnormal, 
    filename=surface_filename, FOCUS=FOCUS, out_dir=OUT_DIR,
    cylindrical_flag=cylindrical_flag,
)
t2 = time.time()
print('Done initializing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')

# to check nothing got mistranslated
t1 = time.time()
pm_opt.m = pm_opt.m0
pm_opt.m_proxy = pm_opt.m0
b_dipole_initial = DipoleField(pm_opt)
b_dipole_initial.set_points(s.gamma().reshape((-1, 3)))
b_dipole_initial._toVTK(OUT_DIR + "Dipole_Fields_initial")
pm_opt._plot_final_dipoles()
t2 = time.time()
print('Done setting up the Dipole Field class')
print('Process took t = ', t2 - t1, ' s')

# Save PM class object to file for reuse
t1 = time.time()
file_out = open(OUT_DIR + class_filename + ".pickle", "wb")
# SurfaceRZFourier objects not pickle-able, so set to None
pm_opt.plasma_boundary = None
pm_opt.rz_inner_surface = None
pm_opt.rz_outer_surface = None
pickle.dump(pm_opt, file_out)
t2 = time.time()
print('Pickling took ', t2 - t1, ' s')
t_end = time.time()
print('In total, script took ', t_end - t_start, ' s')

plt.show()
