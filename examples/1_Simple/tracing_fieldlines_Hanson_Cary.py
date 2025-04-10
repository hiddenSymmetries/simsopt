#!/usr/bin/env python

"""
In this example, we reproduce the Poincare plots from figures 1-2 in the paper

Elimination of stochasticity in stellarators
James D. Hanson and John R. Cary
Physics of Fluids 27, 767 (1984)
http://dx.doi.org/10.1063/1.864692

This example takes advantage of MPI if you launch it with multiple
processes (e.g. by mpirun -n or srun), but it also works on a single
process.
"""

import time
import os
import logging
import numpy as np

from simsopt.configs import get_Cary_Hanson_field
from simsopt.geo import curves_to_vtk, SurfaceRZFourier
from simsopt.field import (
    compute_fieldlines,
    IterationStoppingCriterion,
    LevelsetStoppingCriterion,
    plot_poincare_data,
    particles_to_vtk,
    InterpolatedField,
    SurfaceClassifier,
)
from simsopt.util import in_github_actions, proc0_print, comm_world

proc0_print("Running 1_Simple/tracing_fieldlines_Hanson_Cary.py")
proc0_print("==================================================")

logging.basicConfig()
logger = logging.getLogger("simsopt.field.tracing")
logger.setLevel(1)

# If we're in the CI, make the run a bit cheaper:
nfieldlines = 3 if in_github_actions else 30
tmax_fl = 1000 if in_github_actions else 5000

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# To avoid tracing field lines that are too close to the coils, tracing will be
# stopped with the lines go outside this surface:
surf = SurfaceRZFourier(mpol=1, ntor=0, nfp=5, stellsym=True)
surf.x = [1.0, 0.3, 0.3]
surf_classifier = SurfaceClassifier(surf, h=0.03, p=2)

for optimized in [False, True]:
    proc0_print(f"Beginning calculations with optimized = {optimized}")
    if optimized:
        optimized_str = "optimized"
    else:
        optimized_str = "nonoptimized"

    coils, field = get_Cary_Hanson_field("1984", optimized=optimized)
    nfp = 5
    curves_to_vtk([c.curve for c in coils], OUT_DIR + "Hanson_Cary_coils_" + optimized_str)

    # Create interpolated field, to significantly speed up field line tracing:
    n = 20
    rrange = (0.7, 1.25, n)
    phirange = (0, 2 * np.pi / nfp, n * 2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, 0.2, n // 2)
    proc0_print("Initializing InterpolatedField")
    degree = 4
    field = InterpolatedField(
        field, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )

    proc0_print("Beginning field line tracing")

    t1 = time.time()
    # Set initial grid of points for field line tracing:
    R0 = np.linspace(0.76, 1.16, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        field,
        R0,
        Z0,
        tmax=tmax_fl,
        tol=1e-10,
        comm=comm_world,
        phis=phis,
        stopping_criteria=[IterationStoppingCriterion(300000), LevelsetStoppingCriterion(surf_classifier.dist)],
    )

    t2 = time.time()
    proc0_print(
        f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys]) // nfieldlines}",
        flush=True,
    )
    if comm_world is None or comm_world.rank == 0:
        particles_to_vtk(
            fieldlines_tys, OUT_DIR + f"fieldlines_Hanson_Cary_{optimized_str}"
        )
        plot_poincare_data(
            fieldlines_phi_hits,
            phis,
            OUT_DIR + f"poincare_Hanson_Cary_{optimized_str}.png",
            marker=".",
            dpi=300,
        )

proc0_print("End of 1_Simple/tracing_fieldlines_Hanson_Cary.py")
proc0_print("=================================================")
