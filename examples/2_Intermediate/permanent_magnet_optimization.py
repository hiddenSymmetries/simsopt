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
from simsopt.field.magneticfieldclasses import InterpolatedField, DipoleField, ToroidalField
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from simsopt._core.optimizable import Optimizable
from permanent_magnet_helpers import *
import time

t_start = time.time()

# Read in all the required parameters
config_flag, res_flag, final_run, reg_l2, epsilon, max_iter_MwPGP, min_fb, reg_l0, nu, max_iter_RS, dr, coff, poff, surface_flag, input_name, nphi, ntheta, pms_name, is_premade_famus_grid = read_input()

# Load in MPI, VMEC, etc. if doing a final run 
# to generate a VMEC wout file which can be 
# used to plot symmetry-breaking bmn, the flux
# surfaces, epsilon_eff, etc. 
if final_run:
    from mpi4py import MPI
    from simsopt.util.mpi import MpiPartition
    from simsopt.mhd.vmec import Vmec
    mpi = MpiPartition(ngroups=4)
    comm = MPI.COMM_WORLD
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
print("Cylindrical coordinates are being used = ", pm_opt.cylindrical_flag)

# Check that you loaded the correct file with the same parameters
assert (dr == pm_opt.dr)
assert (nphi == pm_opt.nphi)
assert (ntheta == pm_opt.ntheta)
assert (coff == pm_opt.coil_offset)
assert (poff == pm_opt.plasma_offset)

# Make a subdirectory for the optimization output
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
    max_iter_MwPGP = 340

# Optimize the permanent magnets
#m0 = np.ravel(np.array([pm_opt.m_maxima, np.zeros(pm_opt.ndipoles), np.zeros(pm_opt.ndipoles)]).T)
#m0 = np.ravel(np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T) / np.sqrt(4)
m0 = np.ravel(np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T) / np.sqrt(3)
#m0 = np.ravel((np.random.rand(pm_opt.ndipoles, 3) - 0.5) * 2 * np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T / np.sqrt(12))
#m0 = np.zeros(pm_opt.m0.shape)
MwPGP_history, RS_history, m_history, dipoles = pm_opt._optimize(
    max_iter_MwPGP=max_iter_MwPGP, epsilon=epsilon, min_fb=min_fb,
    reg_l2=reg_l2, reg_l0=reg_l0, nu=nu, max_iter_RS=max_iter_RS,
    m0=m0
)
t2 = time.time()
print('Done optimizing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')

# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
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
print("Total Bn = ", 
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
print("Total Bn (sparse) = ", 
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m_proxy - pm_opt.b_obj) ** 2))

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
#print(Bnormal)
#print(Bnormal_dipoles)
#print(Bn_Am)
print("Average Bn with the PMs = ", 
      np.mean(np.abs(Bnormal_total) / (2 * pm_opt.nphi * pm_opt.ntheta)))
print('F_B INITIAL = ', SquaredFlux(s, b_dipole, -Bnormal).J())
print('F_B INITIAL * 2 * nfp = ', 2 * s.nfp * SquaredFlux(pm_opt.plasma_boundary, b_dipole, -pm_opt.Bn).J())

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

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(RS_history)
    plt.yscale('log')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(RS_history)
    plt.yscale('log')
    plt.grid(True)
    if len(RS_history) > 10:
        plt.xlim(10, len(RS_history))
        plt.ylim(RS_history[9], RS_history[-1])
    plt.savefig(OUT_DIR + 'RS_objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    m0_abs = np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima
    mproxy_abs = np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima
    x_multi = [m0_abs, mproxy_abs]
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
        x_multi = [m0_abs, mproxy_abs, abs(rho)]
    plt.hist(x_multi, bins=np.linspace(0, 1, 20), log=True, histtype='bar')
    plt.grid(True)
    plt.legend(['m', 'w', 'FAMUS'])
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(OUT_DIR + 'm_histograms.png')
    print('Done optimizing the permanent magnets')
t2 = time.time()
print("Done printing and plotting, ", t2 - t1, " s")

if comm is None or comm.rank == 0:
    # double the plasma surface resolution for the vtk plots
    t1 = time.time()
    qphi = 2 * s.nfp * nphi + 1
    qtheta = ntheta + 1
    endpoint = True
    quadpoints_phi = np.linspace(0, 1, qphi, endpoint=endpoint) 
    quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=endpoint)
    srange = 'full torus' 

    if surface_flag == 'focus':
        s_plot = SurfaceRZFourier.from_focus(surface_filename, range=srange, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    elif surface_flag == 'wout':
        s_plot = SurfaceRZFourier.from_wout(surface_filename, range=srange, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    else:
        s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range=srange, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

    if config_flag == 'ncsx':
        # Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
        net_poloidal_current_Amperes = 3.7713e+6
        mu0 = 4 * np.pi * (1e-7)
        RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
        tf = ToroidalField(R0=1, B0=RB)
        tf.set_points(s_plot.gamma().reshape((-1, 3)))
        Bnormal = np.sum(tf.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
    else:
        bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)

    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_dipoles = np.sum(b_dipole.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
    Bnormal_total = Bnormal + Bnormal_dipoles

    Nnorms = np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1)))
    Ngrid = pm_opt.nphi * pm_opt.ntheta
    Bnormal_Am = ((pm_opt.A_obj.dot(pm_opt.m)) * np.sqrt(Ngrid / Nnorms)).reshape(nphi, ntheta)
    #print(Bnormal, Bnormal_dipoles, Bnormal_Am)

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
    f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
    print('f_B = ', f_B_sf)
    B_max = 1.465
    mu0 = 4 * np.pi * 1e-7
    total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * s.stellsym * mu0 / B_max
    total_volume_sparse = np.sum(np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * s.stellsym * mu0 / B_max
    print('Total volume for m and m_proxy = ', total_volume, total_volume_sparse)
    pm_opt.m = pm_opt.m_proxy
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    f_B_sp = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
    print('f_B_sparse = ', f_B_sp)
    dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
    num_nonzero_sparse = np.count_nonzero(dipoles[:, 0] ** 2 + dipoles[:, 1] ** 2 + dipoles[:, 2] ** 2) / pm_opt.ndipoles * 100
    np.savetxt(OUT_DIR + 'final_stats.txt', [f_B_sf, f_B_sp, num_nonzero, num_nonzero_sparse, total_volume, total_volume_sparse])

    # Show the figures
    # plt.show()

if final_run:
    # run Poincare plots
    t1 = time.time()
    n = 16
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 2
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
    try:
        trace_fieldlines(bsh, 'bsh_PMs', config_flag, s_plot, comm)
    except SystemError:
        print('Poincare plot failed.')

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
    try:
        equil = Vmec(surface_filename, mpi)
        equil.boundary = qfm_surf
        equil._boundary = qfm_surf
        equil.need_to_run_code = True
        equil.run()
    except RuntimeError:
        print('VMEC cannot be initialized from a wout file for some reason.')

    t2 = time.time()
    print("VMEC took ", t2 - t1, " s")

    # Save optimized permanent magnet class object
    file_out = open(OUT_DIR + class_filename + "_optimized.pickle", "wb")
    # SurfaceRZFourier objects not pickle-able, so set to None
    pm_opt.plasma_boundary = None
    pm_opt.rz_inner_surface = None
    pm_opt.rz_outer_surface = None
    pickle.dump(pm_opt, file_out)

t_end = time.time()
print('Total time = ', t_end - t_start)
