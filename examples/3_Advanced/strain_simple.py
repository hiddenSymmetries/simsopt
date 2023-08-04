"""
This script performs a stage two coil optimization
with additional terms for torsion and binormal curvature in the cost function.
We use a multifilament approach that initializes the coils using the Frener-Serret Frame
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import CurveLength, CurveCurveDistance, CurveSurfaceDistance
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import ArclengthVariation
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo.strain_optimization_classes import StrainOpt
from simsopt.geo import create_multifilament_grid, ZeroRotation, FramedCurveFrenet, FrameRotation
# from exportcoils import export_coils, import_coils, import_coils_fb, export_coils_fb
from simsopt.configs import get_ncsx_data
import matplotlib.pyplot as plt

curves, currents, ma = get_ncsx_data()
curve = curves[0]

# Set up the winding pack
rot_order = 5  # order of the Fourier expression for the rotation of the filament pack

width = 12

rotation = FrameRotation(curve.quadpoints,rot_order)
framedcurve = FramedCurveFrenet(curve, rotation)

strain = StrainOpt(framedcurve, width=width)

tor = strain.binormal_curvature_strain()

plt.figure()
plt.plot(tor)
plt.show()

# tor = strain.torsional_strain()
tor_frame = framedcurve.frame_torsion()
tor_curve = curve.torsion()
dl = curve.incremental_arclength()

plt.figure()
plt.plot(tor_frame)
plt.plot(tor_curve)
plt.show()
