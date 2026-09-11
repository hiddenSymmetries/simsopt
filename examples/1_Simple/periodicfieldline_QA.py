#!/usr/bin/env python3

"""
This example demonstrates how to compute a periodic field line of a coil set
with SIMSOPT.

A periodic field line is a magnetic field line that closes on itself after one
field period, i.e. a fixed point of the field-period return map. The magnetic
axis and the X-points of a stellarator are the most common examples. The
:class:`~simsopt.geo.periodicfieldline.PeriodicFieldLine` object finds such a
curve by solving the field-line residual equation with a Newton iteration,
starting from an initial guess.

Here we use the ``"STAR_Lite-A_low"`` configuration and seed the solver with the
magnetic axis returned by ``get_data``. The magnetic axis is a
:class:`~simsopt.geo.CurveRZFourier`, so we first project it onto a
:class:`~simsopt.geo.CurveXYZFourierSymmetries`, the curve representation the
solver operates on.
"""

import numpy as np

from simsopt.configs import get_data
from simsopt.geo import (CurveRZFourier, CurveXYZFourierSymmetries, CurveLength,
                         PeriodicFieldLine, curves_to_vtk)
from simsopt.field import BiotSavart

# 1. Load the coils and the magnetic axis of the configuration.
base_curves, base_currents, ma, nfp, bs = get_data('STAR_Lite-A_low')
coils = bs.coils

# 2. Build the initial guess for the periodic field line.
#    The solver works with a CurveXYZFourierSymmetries defined over a single
#    field period, theta in [0, 1/nfp). Using 2*order+1 quadpoints gives a
#    square Newton system. Set ntor > 1 if the field line wraps around the
#    magnetic axis several times toroidally before closing (e.g. an X-point).
order = ma.order
quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
axis = CurveXYZFourierSymmetries(quadpoints, order, nfp=nfp, stellsym=True, ntor=1)

# 3. Seed the guess with the magnetic axis. The axis is a CurveRZFourier, whose
#    parameter spans the full torus over [0, 1), so we resample it on the same
#    single-period quadpoints as `axis` before fitting.
ma_seed = CurveRZFourier(quadpoints, ma.order, ma.nfp, ma.stellsym)
ma_seed.x = ma.x
axis.least_squares_fit(ma_seed.gamma())

# 4. Solve for the periodic field line. run_code takes the initial guess for the
#    field line length (the total length of the seed curve is a good choice).
axis_fl = PeriodicFieldLine(BiotSavart(bs.coils), axis)
res = axis_fl.run_code(CurveLength(axis_fl.curve).J())

print(f"Periodic fieldline found (magnetic axis): success={res['success']}, "
      f"iterations={res['iter']}, length={res['length']:.6f} m")

# 5. Save the coils and the periodic field line for visualization (e.g. Paraview).
curves_to_vtk([c.curve for c in coils], "QA_coils")
curves_to_vtk([axis_fl.curve], "QA_periodic_field_line", close=False)
