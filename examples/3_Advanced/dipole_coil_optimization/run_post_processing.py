from simsopt.mhd.vmec import Vmec
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.util import proc0_print
from simsopt.util import comm_world
from simsopt._core import Optimizable
import time
import numpy as np
from pathlib import Path
import sys
from simsopt import load
from simsopt.util.permanent_magnet_helper_functions import make_qfm
from simsopt.geo import (
    SurfaceRZFourier)

mpi = MpiPartition(ngroups=1)
comm = comm_world
print(
    'Script requires specifying one command line arguments -- '
    'the configuration name (QA, QH, QASH) and the link to the biotsavart.json '
    'is assumed to be in the dipole_coils_<config_name> directory.'
)
nphi = 256
ntheta = 256
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
nfieldlines = 30
tmax_fl = 40000
Z0 = np.zeros(nfieldlines)
if str(sys.argv[1]) == 'QA':
    input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
    R0 = np.linspace(12.25, 13.2, nfieldlines)
elif str(sys.argv[1]) == 'QH':
    input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
    R0 = np.linspace(16.9, 17.8, nfieldlines)
elif str(sys.argv[1]) == 'QASH':  # Only works if QFM surface is relatively unchanged
    input_name = 'wout_schuett_henneberg_nfp2_QA.nc'
    R0 = np.linspace(4.5, 6.75, nfieldlines)
filename = TEST_DIR / input_name

try:
    s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
except:
    s = SurfaceRZFourier.from_wout(filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
# for loop over all the directories for all the coils
num_directories = 10
feasible_coils = []
for i in range(num_directories):
    Bfield = Optimizable.from_file('dipole_coils_' + str(sys.argv[1]) + '/biot_savart_optimized.json')
    coils = Bfield.coils
    # coils_to_vtk(coils, 'coils.vtk')
    # CoilSurfaceDistance has been removed; skipping distance check
    # Jcs = CoilSurfaceDistance(coils, s)
    # if Jcs.minimum_distance() < 0.02:
    #     print("Coils are too close to the surface")
    #     continue
    # else:
    #     # add it to the list of feasible coils
    feasible_coils.append(Bfield)


Bfield.set_points(s.gamma().reshape((-1, 3)))
if str(sys.argv[1]) != 'QASH':
    BdotN = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                           ) / np.mean(Bfield.AbsB())
    Bn_plasma = None
else:
    Bn_plasma = np.array(load('dipole_coils_QASH/B_external_normal_extended.json')['B_external_normal_extended'])
    pointData = {
        "B_N1": (np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))[:, :, None],
        "B_N2": (Bn_plasma)[:, :, None],
        "B_N": (np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2) - Bn_plasma)[:, :, None],
        "B_N / B": ((np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2
                            ) - Bn_plasma) / np.linalg.norm(Bfield.B().reshape(nphi, ntheta, 3), axis=-1))[:, :, None]}
    s.to_vtk('dipole_coils_QASH/surf_check', extra_data=pointData)
    print(np.shape(Bn_plasma), np.shape(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2) - Bn_plasma))
    BdotN_over_B = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2) - Bn_plasma)
                           ) / np.mean(Bfield.AbsB())
print(BdotN, BdotN_over_B)

# # Make the QFM surfaces
qfm_surf = make_qfm(s, Bfield, Bn_plasma)
qfm_surf = qfm_surf.surface

# Run VMEC with new QFM surface
vmec_input = str(filename)  # "../../../tests/test_files/input.LandremanPaul2021_QA_reactorScale_lowres"
equil = Vmec(vmec_input, mpi)
equil.boundary = qfm_surf
equil.run()

# Configure quasisymmetry objective:
if str(sys.argv[1]) == 'QH':
    qs = QuasisymmetryRatioResidual(equil,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|
else:
    qs = QuasisymmetryRatioResidual(equil,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=0)  # (M, N) you want in |B|

proc0_print("Quasisymmetry objective before optimization:", qs.total())

# Need to load a wout file to get the proper iota profile!
from matplotlib import pyplot as plt
plt.figure()
plt.grid()
psi_s = np.linspace(0, len(equil.wout.iotas[1::]) * equil.ds, len(equil.wout.iotas[1::]))
plt.plot(psi_s, equil.wout.iotas[1::], 'rx')
plt.ylabel(r'rotational transform $\iota$')
plt.xlabel('Normalized toroidal flux s')
plt.show()

from simsopt.field.magneticfieldclasses import InterpolatedField


def skip(rs, phis, zs):
    # Reused function from examples/1_Simple/fieldline_tracing_QA.py
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
    proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip


n = 20
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / s.nfp, n * 2)
zrange = (0, np.max(zs), n // 2)
degree = 4  # 2 is sufficient sometimes
Bfield.set_points(s.gamma().reshape((-1, 3)))
bsh = InterpolatedField(
    Bfield, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=s.stellsym, skip=skip
)
bsh.set_points(s.gamma().reshape((-1, 3)))
from simsopt.field.tracing import compute_fieldlines, \
    plot_poincare_data, \
    SurfaceClassifier, \
    LevelsetStoppingCriterion
from simsopt.util import proc0_print

phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]
print(rrange, zrange, phirange)
print(R0, Z0)

t1 = time.time()
# compute the fieldlines from the initial locations specified above
sc_fieldline = SurfaceClassifier(s, h=0.02, p=2)

fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
    bsh, R0, Z0, tmax=tmax_fl, tol=1e-10, comm=comm,
    phis=phis,
    stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
t2 = time.time()
proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)

# make the poincare plots
if comm is None or comm.rank == 0:
    plot_poincare_data(fieldlines_phi_hits, phis, 'poincare_fieldline.png', dpi=300, surf=s, aspect='auto')
