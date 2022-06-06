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

# Determine which plasma equilibrium is being used
print("Usage requires a configuration flag chosen from: qa(_nonplanar), qh(_nonplanar), muse(_famus), ncsx, and a flag specifying high or low resolution")
if len(sys.argv) < 3:
    print(
        "Error! You must specify at least 2 arguments: "
        "the configuration flag and the resolution flag"
    )
    exit(1)
config_flag = str(sys.argv[1])
if config_flag not in ['qa', 'qa_nonplanar', 'QH', 'qh', 'qh_nonplanar', 'muse', 'muse_famus', 'ncsx']:
    print(
        "Error! The configuration flag must specify one of "
        "the pre-set plasma equilibria: qa, qa_nonplanar, "
        "qh, qh_nonplanar, muse, muse_famus, or ncsx. "
    )
    exit(1)
res_flag = str(sys.argv[2])
if res_flag not in ['low', 'medium', 'high']:
    print(
        "Error! The resolution flag must specify one of "
        "low or high."
    )
    exit(1)
print('Config flag = ', config_flag, ', Resolution flag = ', res_flag)
if len(sys.argv) > 3:
    cylindrical_flag = (str(sys.argv[3]) == 'True')
    print(
        "Assuming that the third command-line argument is a boolean "
        "for using cylindrical coordinates or not."
    )
else:
    cylindrical_flag = False

# Pre-set parameters for each configuration
surface_flag = 'vmec'
pms_name = None
is_premade_famus_grid = False
cylindrical_vol_flag = False
if res_flag == 'high':
    nphi = 64
    ntheta = 64
elif res_flag == 'medium':
    nphi = 16
    ntheta = 16
else:
    nphi = 8
    ntheta = 8
if config_flag == 'muse':
    dr = 0.01
    coff = 0.1
    poff = 0.02
    surface_flag = 'focus'
    input_name = 'input.' + config_flag 
elif 'qa' in config_flag:
    dr = 0.01
    coff = 0.06
    poff = 0.04
    input_name = 'input.LandremanPaul2021_' + config_flag[:2].upper()
elif 'qh' in config_flag:
    dr = 0.01
    coff = 0.06
    poff = 0.04
    input_name = 'wout_LandremanPaul_' + config_flag[:2].upper() + '_variant.nc'
    surface_flag = 'wout'
elif 'QH' in config_flag:
    dr = 0.4
    coff = 2.4
    poff = 1.6
    input_name = 'wout_LandremanPaul2021_' + config_flag[:2].upper() + '_reactorScale_lowres_reference.nc'
    surface_flag = 'wout'
    cylindrical_vol_flag = True
elif config_flag == 'ncsx':
    dr = 0.02
    coff = 0.02
    poff = 0.1
    surface_flag = 'wout'
    #input_name = 'input.NCSX_c09r00_halfTeslaTF'
    input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
    coil_name = 'input.NCSX_c09r00_halfTeslaTF_Bn'
    pms_name = 'init_orient_pm_nonorm_5E4_q4_dp.focus'    
    is_premade_famus_grid = True
if config_flag == 'muse_famus':
    dr = 0.01
    coff = 0.1
    poff = 0.02
    surface_flag = 'focus'
    input_name = 'input.muse'
    pms_name = 'zot80.focus'    
    is_premade_famus_grid = True

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
if 'muse' in config_flag:
    # Load in pre-optimized coils
    TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
    coils_filename = TEST_DIR / 'muse_tf_coils.focus'
    base_curves, base_currents, ncoils = read_focus_coils(coils_filename)
    coils = []
    for i in range(ncoils):
        coils.append(Coil(base_curves[i], base_currents[i]))
    base_currents[0].fix_all()

    # fix all the coil shapes
    for i in range(ncoils):
        base_curves[i].fix_all()
elif config_flag == 'qh':
    # generate planar TF coils
    ncoils = 4
    R0 = 1.0
    R1 = 0.6
    order = 5

    # qh needs to be scaled to 0.1 T on-axis magnetic field strength
    from simsopt.mhd.vmec import Vmec
    vmec_file = 'wout_LandremanPaul2021_QH.nc'
    total_current = Vmec(vmec_file).external_current() / (2 * s.nfp) / 8.75
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
    base_currents = [ScaledCurrent(Current(total_current / ncoils * 1e-5), 1e5) for _ in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    # fix all the coil shapes so only the currents are optimized 
    for i in range(ncoils):
        base_curves[i].fix_all()
elif config_flag == 'qa':
    # generate planar TF coils
    ncoils = 4
    R0 = 1.0
    R1 = 0.6
    order = 5

    # qa needs to be scaled to 0.1 T on-axis magnetic field strength
    from simsopt.mhd.vmec import Vmec
    vmec_file = 'wout_LandremanPaul2021_QA.nc'
    total_current = Vmec(vmec_file).external_current() / (2 * s.nfp) / 8
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
    base_currents = [ScaledCurrent(Current(total_current / ncoils * 1e-5), 1e5) for _ in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    # fix all the coil shapes so only the currents are optimized 
    for i in range(ncoils):
        base_curves[i].fix_all()
elif 'QH' in config_flag:
    # generate planar TF coils
    ncoils = 4
    R0 = s.get_rc(0, 0)
    R1 = 10
    order = 5

    #base_currents = [Current(1.2e7) for i in range(ncoils)]
    #base_currents = [Current(6e7) for i in range(ncoils)]

    # QH needs to be scaled to 5.7 T on-axis magnetic field strength
    from simsopt.mhd.vmec import Vmec
    total_current = Vmec(surface_filename).external_current() / (2 * s.nfp) / 0.9
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
    base_currents = [ScaledCurrent(Current(total_current / ncoils * 1e-5), 1e5) for _ in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    # fix all the coil shapes so only the currents are optimized 
    for i in range(ncoils):
        base_curves[i].fix_all()
elif config_flag == 'qa_nonplanar' or config_flag == 'qh_nonplanar':
    # generate single TF coil but then allow for some shape optimization
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
# so have to treat the NCSX example separately.
#quadpoints_phi = np.linspace(0, 1, nphi, endpoint=False)
quadpoints_phi = np.linspace(0, 1, 2 * nphi, endpoint=True)
qphi = len(quadpoints_phi)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
#quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=False)
if config_flag != 'ncsx':
    t1 = time.time()
    bs = BiotSavart(coils)  # Set up BiotSavart fields
    bspoints = np.zeros((nphi, 3))

    # Check average on-axis magnetic field strength
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
    t2 = time.time()
    print("Done setting up biot savart, ", t2 - t1, " s")

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_init")
    t2 = time.time()
    print("Done loading in coils", t2 - t1, " s")

    # Make higher resolution surface for plotting Bnormal
    if surface_flag == 'focus': 
        s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    elif surface_flag == 'wout': 
        s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    else:
        s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

    ######## TEMPORARY for checking things
    #s_plot = s

    # Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "biot_savart_init", extra_data=pointData)

    # If BiotSavart not yet optimized, optimize it
    if 'muse' not in config_flag: 
        s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, config_flag)
        # check after-optimization average on-axis magnetic field strength
        bs.set_points(bspoints)
        B0 = np.linalg.norm(bs.B(), axis=-1)
        B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
        surface_area = s.area()
        bnormalization = B0avg * surface_area
        print("After coil opt: Bmag at R = ", R0, ", Z = 0: ", B0) 
        print("After coil opt: toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg) 

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
    R0 = s.get_rc(0, 0)
    bspoints = np.zeros((nphi, 3))
    for i in range(nphi):
        bspoints[i] = np.array([R0 * np.cos(s.quadpoints_phi[i]), R0 * np.sin(s.quadpoints_phi[i]), 0.0]) 
    tf.set_points(bspoints)
    B0 = np.linalg.norm(tf.B(), axis=-1)
    B0avg = np.mean(np.linalg.norm(tf.B(), axis=-1))
    surface_area = s.area()
    bnormalization = B0avg * surface_area
    print("Bmag at R = ", R0, ", Z = 0: ", B0) 
    print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg) 

    tf.set_points(s.gamma().reshape((-1, 3)))
    Bnormal = np.sum(tf.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

    #Bnormal = load_ncsx_coil_data(s, coil_name)

    # Plot Bnormal on plasma surface from optimized BiotSavart coils
    s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    tf.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_plot = np.sum(tf.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
    #s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    #Bnormal_plot = load_ncsx_coil_data(s_plot, coil_name)
    pointData = {"B_N": Bnormal_plot[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "biot_savart_init", extra_data=pointData)

# permanent magnet setup 
t1 = time.time()
pm_opt = PermanentMagnetOptimizer(
    s, is_premade_famus_grid=is_premade_famus_grid, coil_offset=coff, 
    dr=dr, plasma_offset=poff, Bn=Bnormal, 
    filename=surface_filename, surface_flag=surface_flag, out_dir=OUT_DIR,
    cylindrical_flag=cylindrical_flag, pms_name=pms_name,
    cylindrical_vol_flag=cylindrical_vol_flag
)
t2 = time.time()
print('Done initializing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')


# If using a pre-made FAMUS grid of permanent magnet
# locations, save the FAMUS grid and FAMUS solution.
if is_premade_famus_grid:
    t1 = time.time()
    famus_file = '../../tests/test_files/' + pms_name
    # FAMUS files are for the half-period surface 
    ox, oy, oz, Ic, m0, p, mp, mt = np.loadtxt(
        famus_file, skiprows=3, 
        usecols=[3, 4, 5, 6, 7, 8, 10, 11], 
        delimiter=',', unpack=True
    )
    print('Number of FAMUS dipoles = ', len(ox))
    nonzero_inds = (Ic == 1.0)
    print(nonzero_inds)
    ox = ox[nonzero_inds]
    oy = oy[nonzero_inds]
    oz = oz[nonzero_inds]
    m0 = m0[nonzero_inds]
    p = p[nonzero_inds]
    mp = mp[nonzero_inds]
    mt = mt[nonzero_inds]
    print('Number of FAMUS dipoles (with ports) = ', len(ox))

    phi = np.arctan2(oy, ox)

    # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
    momentq = np.loadtxt(famus_file, skiprows=1, max_rows=1, usecols=[1]) 
    rho = p ** momentq
    plt.figure()
    plt.hist(abs(rho), bins=np.linspace(0, 1, 30), log=True)
    plt.grid(True)
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(OUT_DIR + 'm_FAMUS_histogram.png')
    mm = rho * m0
    mu0 = 4 * np.pi * 1e-7
    Bmax = 1.4 
    print('FAMUS effective volume = ', np.sum(abs(mm)) * mu0 * 2 * s.nfp / Bmax)

    # Convert from spherical to cartesian vectors
    mx = mm * np.sin(mt) * np.cos(mp) 
    my = mm * np.sin(mt) * np.sin(mp) 
    mz = mm * np.cos(mt)
    m_FAMUS = np.ravel((np.array([mx, my, mz]).T))
    pm_opt.cylindrical_flag = False

    ### TEMPORARY -- setting pm_opt initial guess to the FAMUS solution
    pm_opt.m0 = m_FAMUS

    # Set pm_opt m values to the FAMUS solution so we can use the
    # plotting routines from the class object. 
    pm_opt.m = m_FAMUS 
    pm_opt.m_proxy = m_FAMUS
    pm_opt.m_maxima = m0 
    b_dipole_FAMUS = DipoleField(pm_opt)
    b_dipole_FAMUS.set_points(s.gamma().reshape((-1, 3)))
    b_dipole_FAMUS._toVTK(OUT_DIR + "Dipole_Fields_FAMUS")
    Bnormal_FAMUS = np.sum(b_dipole_FAMUS.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=-1)
    # print(Bnormal_FAMUS, Bnormal)
    #grid_fac = pm_opt.dphi * pm_opt.dtheta
    #print(pm_opt.dphi, pm_opt.dtheta, grid_fac, grid_fac / (4 * np.pi ** 2))
    f_B_famus = SquaredFlux(s, b_dipole_FAMUS).J()
    f_B_sf = SquaredFlux(s, b_dipole_FAMUS, -Bnormal).J()  # * grid_fac / (4 * np.pi ** 2)
    print('f_B (only the FAMUS dipoles) = ', f_B_famus)
    print('f_B (FAMUS total) = ', f_B_sf)

    # Plot Bnormal from optimized Bnormal dipoles
    b_dipole_FAMUS.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_FAMUS = np.sum(b_dipole_FAMUS.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    pointData = {"B_N": Bnormal_FAMUS[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "Bnormal_opt_FAMUS", extra_data=pointData)

    # Plot total Bnormal from optimized Bnormal dipoles + coils
    pointData = {"B_N": (Bnormal_FAMUS + Bnormal_plot)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "Bnormal_total_FAMUS", extra_data=pointData)
    t2 = time.time()
    print('Saving FAMUS solution took ', t2 - t1, ' s')

    # Plot the REGCOIL_PM solution for the NCSX example
    if config_flag == 'ncsx':
        read_regcoil_pm('../../tests/test_files/regcoil_pm_ncsx.nc', surface_filename, OUT_DIR)

    # check results with Coilpy
    from coilpy import Dipole 
    dipole_coilpy = Dipole()
    dipole_coilpy = dipole_coilpy.open(famus_file, verbose=True)
    dipole_coilpy.full_period(nfp=3)
    B_coilpy = np.zeros((2 * nphi, ntheta, 3))
    s_points = s_plot.gamma()
    for i in range(2 * nphi):
        for j in range(ntheta):
            B_coilpy[i, j, :] = dipole_coilpy.bfield(s_points[i, j, :])

    Bn_coilpy = np.sum(B_coilpy * s_plot.unitnormal(), axis=2)
    pointData = {"B_N": Bn_coilpy[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "Bnormal_opt_Coilpy", extra_data=pointData)

    # Plot total Bnormal from optimized Bnormal dipoles + coils
    pointData = {"B_N": (Bn_coilpy + Bnormal_plot)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "Bnormal_total_Coilpy", extra_data=pointData)

    pm_opt.cylindrical_flag = cylindrical_flag


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
# plt.show()
