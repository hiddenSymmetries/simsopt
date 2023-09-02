#!/usr/bin/env python

"""
This script performs an optimization of the HTS tape winding angle 
with respect to binormal curvature and torsional strain cost functions. 
The orientation of the tape is defined with respect to the Frenet-Serret Frame
"""

import numpy as np
import os
import numpy as np
from scipy.optimize import minimize
from simsopt.geo import StrainOpt, LPTorsionalStrainPenalty, LPBinormalCurvatureStrainPenalty
from simsopt.geo import FrameRotation, FramedCurveFrenet, CurveXYZFourier
from simsopt.field import load_coils_from_makegrid_file
from simsopt.configs import get_hsx_data
import matplotlib.pyplot as plt

ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 400

curves, currents, ma = get_hsx_data(Nt_coils=10, ppp=10)
curve = curves[1]
scale_factor = 0.1
curve_scaled = CurveXYZFourier(curve.quadpoints, curve.order)
curve_scaled.x = curve.x * scale_factor  # scale coil to magnify the strains
rot_order = 10  # order of the Fourier expression for the rotation of the filament pack
width = 1e-3  # tape width

curve_scaled.fix_all()  # fix curve DOFs -> only optimize winding angle
rotation = FrameRotation(curve_scaled.quadpoints, rot_order)

framedcurve = FramedCurveFrenet(curve_scaled, rotation)

tor_threshold = 0.02  # Threshold for strain parameters
cur_threshold = 0.02

Jtor = LPTorsionalStrainPenalty(framedcurve, p=2, threshold=tor_threshold)
Jbin = LPBinormalCurvatureStrainPenalty(
    framedcurve, p=2, threshold=cur_threshold)

strain = StrainOpt(framedcurve, width)
JF = Jtor + Jbin


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    outstr = f"Max torsional strain={np.max(strain.torsional_strain()):.1e}, Max curvature strain={np.max(strain.binormal_curvature_strain()):.1e}"
    print(outstr)
    return J, grad


f = fun
dofs = JF.x

res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 10, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)
