#!/usr/bin/env python
r"""
In this example we optimize the permanent magnets from a 
pre-computed permanent magnet grid, plasma surface, and BiotSavart
field. 

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
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule, DipoleField
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from simsopt._core.optimizable import Optimizable
import time

t_start = time.time()
# Determine which plasma equilibrium is being used
print("Usage requires a configuration flag chosen from: qa(_nonplanar), qh(_nonplanar), muse, ncsx, and a flag specifying high or low resolution")
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
final_run = (str(sys.argv[3]) == 'True')
print('Config flag = ', config_flag, ', Resolution flag = ', res_flag, ', Final run =', final_run)
if len(sys.argv) >= 5:
    reg_l0 = float(sys.argv[4])
    if not np.isclose(reg_l0, 0.0):
        nu = 1
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
FOCUS = False
cylindrical_flag = True
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
class_filename = "PM_optimizer_" + config_flag

# Don't save in home directory on NERSC -- save on SCRATCH
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

OUT_DIR = IN_DIR + "output_regl2{5:.2e}_regl0{6:.2e}_nu{7:.2e}/".format(nphi, ntheta, dr, coff, poff, reg_l2, reg_l0, nu)
os.makedirs(OUT_DIR, exist_ok=True)
t2 = time.time()
print('Loading pickle file and other initialization took ', t2 - t1, ' s')

t1 = time.time()
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / ('input.' + input_name)
if config_flag == 'muse':
    s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
else:
    s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
t2 = time.time()
print("Done loading in plasma boundary surface, t = ", t2 - t1)
bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
bs.set_points(s.gamma().reshape((-1, 3)))

# Set the pm_opt plasma boundary 
pm_opt.plasma_boundary = s

print('Done initializing the permanent magnet object')
t1 = time.time()
max_iter_MwPGP = 500
max_iter_RS = 20
epsilon = 1e-2
MwPGP_history, RS_history, m_history, dipoles = pm_opt._optimize(
    max_iter_MwPGP=max_iter_MwPGP, epsilon=epsilon,
    reg_l2=reg_l2, reg_l0=reg_l0, nu=nu, max_iter_RS=max_iter_RS
)
t2 = time.time()
print('Done optimizing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')
M_max = 1.4 / (4 * np.pi * 1e-7)
dipoles = dipoles.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# recompute normal error using the dipole field and bs field
# to check nothing got mistranslated
t1 = time.time()
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields")
pm_opt._plot_final_dipoles()

t2 = time.time()
print('Done setting up the Dipole Field class')
print('Process took t = ', t2 - t1, ' s')

# b_dipole._toVTK("Dipole_Fields_surf", dim=())

t1 = time.time()
dphi = (pm_opt.phi[1] - pm_opt.phi[0]) * 2 * np.pi
dtheta = (pm_opt.theta[1] - pm_opt.theta[0]) * 2 * np.pi
print("Average Bn without the PMs = ", 
      np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal() * dphi * dtheta, axis=2))))
print("Average Bn with the PMs = ", 
      np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal() * dphi * dtheta, axis=2))))

print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", np.count_nonzero(dipoles[:, 0] ** 2 + dipoles[:, 1] ** 2 + dipoles[:, 2] ** 2) / pm_opt.ndipoles)
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
    plt.hist(abs(dipoles) / np.ravel(np.outer(pm_opt.m_maxima, np.ones(3))), bins=np.linspace(0, 1, 30), log=True)
    plt.grid(True)
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(OUT_DIR + 'm_histogram.png')
    print('Done optimizing the permanent magnets')
t2 = time.time()
print("Done printing and plotting, ", t2 - t1, " s")

if config_flag == 'muse':
    s = SurfaceRZFourier.from_focus(surface_filename, range="full torus", nphi=nphi, ntheta=ntheta)
else:
    s = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", nphi=nphi, ntheta=ntheta)

# Makes a Vmec file for the MUSE boundary -- only needed to do it once
#filename = '../../tests/test_files/input.LandremanPaul2021_QA'  # _lowres
#equil = Vmec(filename, mpi)
#equil.boundary = s 
#equil.run()


def trace_fieldlines(bfield, label): 
    t1 = time.time()
    R0 = np.linspace(0.2, 0.4, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-15, comm=comm,
        phis=phis, stopping_criteria=[IterationStoppingCriterion(200000)])
    t2 = time.time()
    # print(fieldlines_phi_hits, np.shape(fieldlines_phi_hits))
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm is None or comm.rank == 0:
        # particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)


def make_qfm(s, Bfield, Bfield_tf):
    constraint_weight = 1e0

    # First optimize at fixed volume

    qfm = QfmResidual(s, Bfield)
    qfm.J()

    vol = Volume(s)
    vol_target = vol.J()

    qfm_surface = QfmSurface(Bfield, s, vol, vol_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Now optimize at fixed toroidal flux
    tf = ToroidalFlux(s, Bfield_tf)
    tf_target = tf.J()

    qfm_surface = QfmSurface(Bfield, s, tf, tf_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                             constraint_weight=constraint_weight)
    print(f"||tf constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Check that volume is not changed
    print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

    # Now optimize at fixed area

    ar = Area(s)
    ar_target = ar.J()

    qfm_surface = QfmSurface(Bfield, s, ar, ar_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                             constraint_weight=constraint_weight)
    print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Check that volume is not changed
    print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")
    # s.plot()
    return qfm_surface.surface 


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
    bsh = InterpolatedField(
        bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=s.stellsym
    )
    # bsh.to_vtk('dipole_fields')
    trace_fieldlines(bsh, 'bsh_PMs')
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
    filename = '../../tests/test_files/input.LandremanPaul2021_QA'
    equil = Vmec(filename, mpi)
    equil.boundary = qfm_surf
    equil.run()
    t2 = time.time()
    print("VMEC took ", t2 - t1, " s")

if comm is None or comm.rank == 0:
    # double the plasma surface resolution for the vtk plots
    t1 = time.time()
    nphi = 2 * nphi
    ntheta = ntheta
    quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)

    if config_flag == 'muse':
        s = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    else:
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

    bs.set_points(s.gamma().reshape((-1, 3)))
    b_dipole.set_points(s.gamma().reshape((-1, 3)))
    # For plotting Bn on the full torus surface at the end with just the dipole fields
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "biot_savart_opt", extra_data=pointData)
    pointData = {"B_N": np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "only_pms_opt", extra_data=pointData)
    pointData = {"B_N": np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "pms_opt", extra_data=pointData)
    t2 = time.time()
    print('Done saving final vtk files, ', t2 - t1, " s")
    plt.show()
