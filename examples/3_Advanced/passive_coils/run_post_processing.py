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
from simsopt.util.permanent_magnet_helper_functions import make_qfm
from simsopt.geo import (
    SurfaceRZFourier)
from simsopt import load
from simsopt.field import PSCArray

mpi = MpiPartition(ngroups=8)
comm = comm_world
print(
    'Script requires specifying two command line arguments -- '
    'the configuration name (QA, QH, QASH, CSX) and a link to the biotsavart.json '
    'file containing the coil solution.'
)
nphi = 128
ntheta = 64
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
nfieldlines = 40
tmax_fl = 40000
Z0 = np.zeros(nfieldlines)
aa = 0.05
if str(sys.argv[1]) == 'QA':
    input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
    input_dir = "passive_coils_QA/"
    R0 = np.linspace(12.25, 13.2, nfieldlines)
elif str(sys.argv[1]) == 'QH':
    input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
    input_dir = "passive_coils_QH/"
    R0 = np.linspace(16.9, 17.8, nfieldlines)
elif str(sys.argv[1]) == 'QASH':
    input_name = 'input.schuetthenneberg_nfp2'
    input_dir = "passive_coils_QASH/"
    R0 = np.linspace(4.5, 6.75, nfieldlines)
elif str(sys.argv[1]) == 'CSX':
    input_name = 'wout_csx_wps_5.0.nc'
    input_dir = "passive_coils_CSX/"
    R0 = np.linspace(0.32, 0.415, nfieldlines)
    aa = 0.03
filename = TEST_DIR / input_name
coils = load(input_dir + "psc_coils_continuation.json")
coils_TF = load(input_dir + "TF_coils_continuation.json")
if str(sys.argv[1]) != 'CSX':
    s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
else:
    s = SurfaceRZFourier.from_wout(filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
curves = [c.curve for c in coils]
base_curves = curves[:len(curves) // (s.nfp * 2)]
base_coils = coils[:len(coils) // (s.nfp * 2)]
curves_TF = [c.curve for c in coils_TF]
ncoils = len(base_curves)
a_list = np.ones(len(base_curves)) * aa
b_list = np.ones(len(base_curves)) * aa
eval_points = s.gamma().reshape(-1, 3)
psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)
Bfield = psc_array.biot_savart_total
Bfield.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
BdotN_over_B = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                       ) / np.mean(Bfield.AbsB())
print(BdotN, BdotN_over_B)

# # Make the QFM surfaces
qfm_surf = make_qfm(s, Bfield)
qfm_surf = qfm_surf.surface

# VMEC does NOT like the CSX plasma because it's very compact
# Also probably should be including the plasma current contributions
# Although Antoine Baillod has had the same issues.
if str(sys.argv[1]) != 'CSX':
    vmec_input = str(filename)
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

    # Configure quasisymmetry objective:
    if str(sys.argv[1]) == 'QH':
        helicity_n = -1
    else:
        helicity_n = 0

    qs = QuasisymmetryRatioResidual(equil,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=helicity_n)  # (M, N) you want in |B|
    proc0_print("Quasisymmetry objective before optimization:", qs.total())

    # Need to load a wout file to get the proper iota profile!
    from matplotlib import pyplot as plt
    plt.figure()
    plt.grid()
    psi_s = np.linspace(0, len(equil.wout.iotas[1::]) * equil.ds, len(equil.wout.iotas[1::]))
    plt.plot(psi_s, equil.wout.iotas[1::], 'rx')
    plt.ylabel(r'rotational transform $\iota$')
    plt.xlabel('Normalized toroidal flux s')

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
    plot_poincare_data(fieldlines_phi_hits, phis, 'poincare_fieldline' + str(sys.argv[1]) + '.png', dpi=300, surf=s, aspect='auto')

plt.show()
