#!/usr/bin/env python
r"""
In this example we optimize the permanent magnets from a 
pre-computed permanent magnet grid, plasma surface, and BiotSavart
field. 

This is the companion script with permanent_magnets_initialization.py. 

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
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.field.magneticfieldclasses import InterpolatedField, DipoleField
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from simsopt._core.optimizable import Optimizable
from permanent_magnet_helpers import *
import time

t_start = time.time()

# Determine which plasma equilibrium is being used
print("Usage requires a configuration flag chosen from: qa(_nonplanar), QH, qh(_nonplanar), muse(_famus), ncsx, and a flag specifying high or low resolution")
if len(sys.argv) < 4:
    print(
        "Error! You must specify at least 3 arguments: "
        "the configuration flag, resolution flag, final run flag "
        "(whether to run time-intensive processes like QFMs, Poincare "
        "plots, VMEC, etc.), and (optionally) the L0 and L2 regularizer"
        " strengths."
    )
    exit(1)
config_flag = str(sys.argv[1])
if config_flag not in ['qa', 'qa_nonplanar', 'QH', 'qh', 'qh_nonplanar', 'muse', 'muse_famus', 'ncsx']:
    print(
        "Error! The configuration flag must specify one of "
        "the pre-set plasma equilibria: qa, qa_nonplanar, "
        "QH, qh, qh_nonplanar, muse, muse_famus, or ncsx. "
    )
    exit(1)
res_flag = str(sys.argv[2])
if res_flag not in ['low', 'medium', 'high']:
    print(
        "Error! The resolution flag must specify one of "
        "low or high."
    )
    exit(1)
final_run = (str(sys.argv[3]) == 'True')
print('Config flag = ', config_flag, ', Resolution flag = ', res_flag, ', Final run =', final_run)

# Check for L2 and L0 regularization
if len(sys.argv) >= 5:
    reg_l0 = float(sys.argv[4])
    if not np.isclose(reg_l0, 0.0, atol=1e-16):
        nu = 1e2
    else:
        nu = 1e100
else:
    reg_l0 = 0.0  # default is no L0 norm
    nu = 1e100
if len(sys.argv) >= 6:
    reg_l2 = float(sys.argv[5])
else:
    reg_l2 = 1e-8

# Pre-set parameters for each configuration
surface_flag = 'vmec'
cylindrical_flag = False
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
    poff = 0.05
    surface_flag = 'focus'
    input_name = 'input.' + config_flag 
if config_flag == 'muse_famus':
    dr = 0.01
    coff = 0.1
    poff = 0.02
    surface_flag = 'focus'
    input_name = 'input.muse'
    pms_name = 'zot80.focus'
elif 'QH' in config_flag:
    dr = 0.4
    coff = 2.4
    poff = 1.6
    input_name = 'wout_LandremanPaul2021_' + config_flag[:2].upper() + '_reactorScale_lowres_reference.nc'
    surface_flag = 'wout'
elif 'qa' in config_flag or 'qh' in config_flag:
    dr = 0.01
    coff = 0.16
    poff = 0.04
    if 'qa' in config_flag:
        input_name = 'input.LandremanPaul2021_' + config_flag[:2].upper()
    else:
        input_name = 'wout_LandremanPaul_' + config_flag[:2].upper() + '_variant.nc'
        surface_flag = 'wout'
elif config_flag == 'ncsx':
    dr = 0.02
    coff = 0.02
    poff = 0.1
    surface_flag = 'focus'
    input_name = 'input.NCSX_c09r00_halfTeslaTF' 
    coil_name = 'input.NCSX_c09r00_halfTeslaTF_Bn'

if final_run:
    from mpi4py import MPI
    from simsopt.field.tracing import SurfaceClassifier, \
        particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data, \
        IterationStoppingCriterion
    from simsopt.util.mpi import MpiPartition
    from simsopt.mhd.vmec import Vmec
    from simsopt.geo.qfmsurface import QfmSurface
    from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
    mpi = MpiPartition(ngroups=4)
    comm = MPI.COMM_WORLD
    # Number of iterations to perform:
    ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
    nfieldlines = 40 if ci else 40
    tmax_fl = 30000 if ci else 50000
    degree = 2 if ci else 4
else:
    comm = None

t1 = time.time()

# Read in permanent magnet class object that has already been initialized
# using the companion script.
class_filename = "PM_optimizer_" + config_flag
scratch_path = '/global/cscratch1/sd/akaptano/'
IN_DIR = scratch_path + config_flag + "_nphi{0:d}_ntheta{1:d}_dr{2:.2e}_coff{3:.2e}_poff{4:.2e}/".format(nphi, ntheta, dr, coff, poff)
pickle_name = IN_DIR + class_filename + ".pickle"
pm_opt = pickle.load(open(pickle_name, "rb", -1))

# Check that you loaded the correct file with the same parameters
assert (dr == pm_opt.dr)
assert (nphi == pm_opt.nphi)
assert (ntheta == pm_opt.ntheta)
assert (coff == pm_opt.coil_offset)
assert (poff == pm_opt.plasma_offset)

# Make a directory for the optimization output
OUT_DIR = IN_DIR + "output_regl2{5:.2e}_regl0{6:.2e}_nu{7:.2e}/".format(nphi, ntheta, dr, coff, poff, reg_l2, reg_l0, nu)
os.makedirs(OUT_DIR, exist_ok=True)
pm_opt.out_dir = OUT_DIR 
t2 = time.time()
print('Loading pickle file and other initialization took ', t2 - t1, ' s')

# Read in the correct plasma equilibrium file
t1 = time.time()
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
if surface_flag == 'focus':
    s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
elif surface_flag == 'wout':
    s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
else:
    s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
t2 = time.time()
print("Done loading in plasma boundary surface, t = ", t2 - t1)

# Read in the Bnormal or BiotSavart fields from any coils 
if config_flag != 'ncsx':
    bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
    bs.set_points(s.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
else:
    Bnormal = pm_opt.Bn

# Set the pm_opt plasma boundary 
pm_opt.plasma_boundary = s
print('Done initializing the permanent magnet object')

# Set some hyperparameters for the optimization
t1 = time.time()
if reg_l0 > 0:
    max_iter_MwPGP = 100
else:
    max_iter_MwPGP = 1000
max_iter_RS = 50
epsilon = 1e-3

# Optimize the permanent magnets
#m0_max = np.ravel(np.array([pm_opt.m_maxima, np.zeros(pm_opt.ndipoles), np.zeros(pm_opt.ndipoles)]).T)
MwPGP_history, RS_history, m_history, dipoles = pm_opt._optimize(
    max_iter_MwPGP=max_iter_MwPGP, epsilon=epsilon,
    reg_l2=reg_l2, reg_l0=reg_l0, nu=nu, max_iter_RS=max_iter_RS,
    #m0=m0_max
)
t2 = time.time()
print('Done optimizing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')

# Print effective permanent magnet volume
M_max = 1.4 / (4 * np.pi * 1e-7)
dipoles = dipoles.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Plot the sparse and less sparse solutions from SIMSOPT 
t1 = time.time()
m_copy = np.copy(pm_opt.m)
pm_opt.m = pm_opt.m_proxy
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields_Sparse")
pm_opt.m = m_copy
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields")
pm_opt._plot_final_dipoles()
t2 = time.time()
print('Done setting up the Dipole Field class')
print('Process took t = ', t2 - t1, ' s')

# Print optimized metrics
t1 = time.time()
dphi = (pm_opt.phi[1] - pm_opt.phi[0]) * 2 * np.pi
dtheta = (pm_opt.theta[1] - pm_opt.theta[0]) * 2 * np.pi
print("Average Bn without the PMs = ", 
      np.mean(np.abs(Bnormal * dphi * dtheta)))
print("Total Bn without the PMs = ", 
      np.sum((pm_opt.b_obj) ** 2) / 2.0)
print("Total Bn without the coils = ", 
      np.sum((pm_opt.A_obj @ pm_opt.m) ** 2) / 2.0)
# grid_fac = np.sqrt(pm_opt.dphi * pm_opt.dtheta)
print("Total Bn = ", 
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

# Compute metrics with permanent magnet results
Nnorms = np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1)))
Ngrid = pm_opt.nphi * pm_opt.ntheta
ave_Bn_proxy = np.mean(np.abs(pm_opt.A_obj.dot(pm_opt.m_proxy) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms)) / (2 * pm_opt.nphi * pm_opt.ntheta)
Bn_Am = (pm_opt.A_obj.dot(pm_opt.m)) * np.sqrt(Ngrid / Nnorms) 
Bn_opt = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms) 
ave_Bn = np.mean(np.abs(Bn_opt) / (2 * pm_opt.nphi * pm_opt.ntheta))
print('<B * n> with the optimized permanent magnets = {0:.8e}'.format(ave_Bn)) 
print('<B * n> with the sparsified permanent magnets = {0:.8e}'.format(ave_Bn_proxy)) 

Bnormal_dipoles = np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
print(Bnormal)
print(Bnormal_dipoles)
print(Bn_Am)
print("Average Bn with the PMs = ", 
      np.mean(np.abs(Bnormal_total) / (2 * pm_opt.nphi * pm_opt.ntheta)))
#grid_fac = np.sqrt(pm_opt.dphi * pm_opt.dtheta)
print('F_B INITIAL = ', SquaredFlux(s, b_dipole, -Bnormal).J())  # * grid_fac ** 2) 
print('F_B INITIAL * 2 * nfp = ', 2 * s.nfp * SquaredFlux(pm_opt.plasma_boundary, b_dipole, -pm_opt.Bn).J())  # * grid_fac ** 2) 

dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(dipoles_m[:, 0] ** 2 + dipoles_m[:, 1] ** 2 + dipoles_m[:, 2] ** 2) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)
dipoles = np.ravel(dipoles)
print('Dipole field setup done')

make_plots = True 
if make_plots:

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.semilogy(MwPGP_history)
    plt.grid(True)
    plt.savefig(OUT_DIR + 'objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    #plt.hist(abs(dipoles) / np.ravel(np.outer(pm_opt.m_maxima, np.ones(3))), bins=np.linspace(0, 1, 30), log=True)
    #plt.hist(abs(pm_opt.m) / np.ravel(np.outer(pm_opt.m_maxima, np.ones(3))), bins=np.linspace(0, 1, 30), log=True)
    x_multi = [abs(dipoles), abs(pm_opt.m)]
    if pm_opt.is_premade_famus_grid:
        famus_file = '../../tests/test_files/' + pm_opt.pms_name
        m0, p = np.loadtxt(
            famus_file, skiprows=3, 
            usecols=[7, 8], 
            delimiter=',', unpack=True
        )
        # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
        momentq = np.loadtxt(famus_file, skiprows=1, max_rows=1, usecols=[1]) 
        rho = p ** momentq
        #plt.hist(abs(rho), bins=np.linspace(0, 1, 30), log=True)
        x_multi = [abs(dipoles), abs(pm_opt.m), abs(rho)]
    plt.hist(x_multi, bins=np.linspace(0, 1, 15), log=True, histtype='bar')
    plt.grid(True)
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(OUT_DIR + 'm_histograms.png')
    print('Done optimizing the permanent magnets')
t2 = time.time()
print("Done printing and plotting, ", t2 - t1, " s")

#if surface_flag == 'focus':
#    s = SurfaceRZFourier.from_focus(surface_filename, range="full torus", nphi=nphi, ntheta=ntheta)
#elif surface_flag == 'wout':
#    s = SurfaceRZFourier.from_wout(surface_filename, range="full torus", nphi=nphi, ntheta=ntheta)
#else:
#    s = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", nphi=nphi, ntheta=ntheta)

if final_run:
    # run Poincare plots
    t1 = time.time()
    n = 16
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2 * np.pi / s.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    t1 = time.time()
    if config_flag != 'ncsx':
        bsh = InterpolatedField(
            bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=s.stellsym
        )
    else:
        bsh = InterpolatedField(
            b_dipole, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=s.stellsym
        )
    # bsh.to_vtk('dipole_fields')
    trace_fieldlines(bsh, 'bsh_PMs', config_flag)
    t2 = time.time()
    print('Done with Poincare plots with the permanent magnets, t = ', t2 - t1)

    # Make the QFM surfaces
    t1 = time.time()
    # need to call set_points again here for the combined field
    Bfield = Optimizable.from_file(IN_DIR + 'BiotSavart.json') + DipoleField(pm_opt)
    Bfield_tf = Optimizable.from_file(IN_DIR + 'BiotSavart.json') + DipoleField(pm_opt)
    Bfield.set_points(s.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s, Bfield, Bfield_tf)
    t2 = time.time()
    print("Making the QFM took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()
    equil = Vmec(surface_filename, mpi)
    equil.boundary = qfm_surf
    equil._boundary = qfm_surf
    equil.need_to_run_code = True
    equil.run()
    t2 = time.time()
    print("VMEC took ", t2 - t1, " s")

if comm is None or comm.rank == 0:
    # double the plasma surface resolution for the vtk plots
    t1 = time.time()
    quadpoints_phi = np.linspace(0, 1, 2 * s.nfp * nphi + 1, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta + 1, endpoint=True)

    if surface_flag == 'focus':
        s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    elif surface_flag == 'wout':
        s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    else:
        s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

    if config_flag == 'ncsx':
        Bnormal = load_ncsx_coil_data(s_plot, coil_name)
    else:
        bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)

    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_dipoles = np.sum(b_dipole.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
    Bnormal_total = Bnormal + Bnormal_dipoles

    Nnorms = np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1)))
    Ngrid = pm_opt.nphi * pm_opt.ntheta
    Bnormal_Am = ((pm_opt.A_obj.dot(pm_opt.m0)) * np.sqrt(Ngrid / Nnorms)).reshape(nphi, ntheta)
    print(Bnormal, Bnormal_dipoles, Bnormal_Am)

    # For plotting Bn on the full torus surface at the end with just the dipole fields
    pointData = {"B_N": Bnormal[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "biot_savart_opt", extra_data=pointData)
    pointData = {"B_N": Bnormal_dipoles[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "only_pms_opt", extra_data=pointData)
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "pms_opt", extra_data=pointData)
    t2 = time.time()
    print('Done saving final vtk files, ', t2 - t1, " s")

    # Print optimized f_B and other metrics
    f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()  # * grid_fac ** 2 
    print('f_B = ', f_B_sf)
    B_max = 1.4
    mu0 = 4 * np.pi * 1e-7
    total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * s.stellsym * mu0 / B_max
    total_volume_sparse = np.sum(np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * s.stellsym * mu0 / B_max
    print('Total volume for m and m_proxy = ', total_volume, total_volume_sparse)
    pm_opt.m = pm_opt.m_proxy
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    f_B_sp = SquaredFlux(s_plot, b_dipole, -Bnormal).J()  # * grid_fac ** 2 
    print('f_B_sparse = ', f_B_sp)
    dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
    num_nonzero_sparse = np.count_nonzero(dipoles[:, 0] ** 2 + dipoles[:, 1] ** 2 + dipoles[:, 2] ** 2) / pm_opt.ndipoles * 100
    np.savetxt(OUT_DIR + 'final_stats.txt', [f_B_sf, f_B_sp, num_nonzero, num_nonzero_sparse, total_volume, total_volume_sparse])

    # Show the figures
    # plt.show()

# Save optimized permanent magnet class object
if final_run:
    file_out = open(OUT_DIR + class_filename + "_optimized.pickle", "wb")
    # SurfaceRZFourier objects not pickle-able, so set to None
    pm_opt.plasma_boundary = None
    pm_opt.rz_inner_surface = None
    pm_opt.rz_outer_surface = None
    pickle.dump(pm_opt, file_out)
