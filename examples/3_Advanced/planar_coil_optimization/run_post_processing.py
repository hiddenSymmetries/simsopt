from simsopt.mhd.vmec import Vmec
from simsopt.util.mpi import MpiPartition
from simsopt.util import comm_world
from simsopt._core import Optimizable
import time
import numpy as np
from pathlib import Path
import sys
from simsopt.util.permanent_magnet_helper_functions import make_qfm
from simsopt.geo import (
    SurfaceRZFourier, curves_to_vtk)

mpi = MpiPartition(ngroups=8)
comm = comm_world

nphi = 256
ntheta = 128
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
Bfield = Optimizable.from_file(str(sys.argv[1]))
Bfield.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
BdotN_over_B = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                       ) / np.mean(Bfield.AbsB())
print(BdotN, BdotN_over_B)

# # Make the QFM surfaces
qfm_surf = make_qfm(s, Bfield)
qfm_surf = qfm_surf.surface
# qfm_surf.save(filename='QA_qfm.json')
# Bfield.set_points(qfm_surf.gamma().reshape((-1, 3)))
# Bn = np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * qfm_surf.unitnormal(), axis=2)[:, :, None]
# pointData = {"B_N": Bn, "B_N / B": Bn / np.linalg.norm(Bfield.B().reshape(nphi, ntheta, 3), axis=-1)[:, :, None]}
# qfm_surf.to_vtk('qfm_surf', extra_data=pointData)
# qfm_surf.plot()

from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.util import MpiPartition, proc0_print

# Run VMEC with new QFM surface
vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA_reactorScale_lowres"
equil = Vmec(vmec_input, mpi)
equil.boundary = qfm_surf
equil.run()

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(equil,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|

proc0_print("Quasisymmetry objective before optimization:", qs.total())

from simsopt.field.magneticfieldclasses import InterpolatedField


def skip(rs, phis, zs):
    # The RegularGrindInterpolant3D class allows us to specify a function that
    # is used in order to figure out which cells to be skipped.  Internally,
    # the class will evaluate this function on the nodes of the regular mesh,
    # and if *all* of the eight corners are outside the domain, then the cell
    # is skipped.  Since the surface may be curved in a way that for some
    # cells, all mesh nodes are outside the surface, but the surface still
    # intersects with a cell, we need to have a bit of buffer in the signed
    # distance (essentially blowing up the surface a bit), to avoid ignoring
    # cells that shouldn't be ignored
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
    proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip


# out_dir = Path(out_dir)
n = 20
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / s.nfp, n * 2)
zrange = (0, np.max(zs), n // 2)
print(zrange, rrange, phirange)
degree = 4  # 2 is sufficient sometimes
Bfield.set_points(s.gamma().reshape((-1, 3)))
bsh = InterpolatedField(
    Bfield, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=s.stellsym, skip=skip
)
bsh.set_points(s.gamma().reshape((-1, 3)))
from simsopt.field.tracing import compute_fieldlines, \
    plot_poincare_data, \
    IterationStoppingCriterion, SurfaceClassifier, \
    LevelsetStoppingCriterion
from simsopt.util import proc0_print


# set fieldline tracer parameters
nfieldlines = 20
tmax_fl = 20000

R0 = np.linspace(12.25, 13.2, nfieldlines)
Z0 = np.zeros(nfieldlines)
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
