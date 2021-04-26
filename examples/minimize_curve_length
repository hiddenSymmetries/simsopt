#!/usr/bin/env python3

import numpy as np
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curveobjectives import CurveLength
from simsopt import LeastSquaresProblem
from simsopt import least_squares_serial_solve

"""
Minimize the length of a curve, holding the 0-frequency Fourier mode fixed.
The result should be a circle
"""

# Create a curve:
nquadrature = 100
nfourier = 4
nfp = 5
curve = CurveRZFourier(nquadrature, nfourier, nfp, True)

# Initialize the Fourier amplitudes to some random values
x0 = np.random.rand(curve.num_dofs()) - 0.5
x0[0] = 3.0
curve.set_dofs(x0)
print('Initial curve dofs: ', curve.get_dofs())

# Tell the curve object that the first Fourier mode is fixed, whereas
# all the other dofs are not.
curve.all_fixed(False)
curve.fixed[0] = True

# The length objective is a separate object rather than a function of
# Curve itself.
obj = CurveLength(curve)

print('Initial curve length: ', obj.J())

# Each target function is then equipped with a shift and weight, to
# become a term in a least-squares objective function.
# A list of terms are combined to form a nonlinear-least-squares
# problem.
prob = LeastSquaresProblem([(obj, 0.0, 1.0)])

# At the initial condition, get the Jacobian two ways: analytic
# derivatives and finite differencing. The difference should be small.
fd_jac = prob.dofs.fd_jac()
jac = prob.dofs.jac()
print('finite difference Jacobian:')
print(fd_jac)
print('Analytic Jacobian:')
print(jac)
print('Difference:')
print(fd_jac - jac)

# Solve the minimization problem:
least_squares_serial_solve(prob)

print('At the optimum, x: ', prob.x)
print(' Final curve dofs: ', curve.get_dofs())
print(' Final curve length:    ', obj.J())
print(' Expected final length: ', 2 * np.pi * x0[0])
print(' objective function: ', prob.objective())
