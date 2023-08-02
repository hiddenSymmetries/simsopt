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
from simsopt.geo.finitebuild import create_multifilament_grid
# from exportcoils import export_coils, import_coils, import_coils_fb, export_coils_fb
from simsopt.configs import get_ncsx_data
import matplotlib.pyplot as plt

curves, currents, ma = get_ncsx_data()
curve = curves[0]

# Set up the winding pack
numfilaments_n = 1  # number of filaments in normal direction
numfilaments_b = 1  # number of filaments in bi-normal direction
gapsize_n = 0.02  # gap between filaments in normal direction
gapsize_b = 0.03  # gap between filaments in bi-normal direction
rot_order = 5  # order of the Fourier expression for the rotation of the filament pack

scale = 1
width = 12

# use sum here to concatenate lists
filaments = create_multifilament_grid(curve, numfilaments_n, numfilaments_b, gapsize_n,
                                     gapsize_b, rotation_order=rot_order, frame='frenet')

strain = StrainOpt(filaments[0], width=width)

tor = strain.binormal_curvature_strain()

plt.figure()
plt.plot(tor)
plt.show()

# tor = strain.torsional_strain()
tor_frame = filaments[0].frame_torsion()
tor_curve = curve.torsion()

plt.figure()
plt.plot(tor_frame)
plt.plot(tor_curve)
plt.show()
