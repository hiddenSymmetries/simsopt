#!/usr/bin/env python3

import numpy as np
from simsopt.geo import CurveRZFourier, CurveLength
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve

"""
Minimize the length of a curve, holding the 0-frequency Fourier mode fixed.
The result should be a circle.
"""
print("Running 1_Simple/minimize_curve_length.py")
print("==============================================")
# Create a curve:
nquadrature = 100
nfourier = 4
nfp = 5  # exploit rotational symmetry and only consider a fifth of the curve
curve = CurveRZFourier(nquadrature, nfourier, nfp, True)

# Initialize the Fourier amplitudes to some random values
x0 = np.random.rand(curve.dof_size) - 0.5
x0[0] = 3.0
curve.x = x0
print('Initial curve dofs: ', curve.x)

# Tell the curve object that the first Fourier mode is fixed, whereas
# all the other dofs are not.
curve.fix(0)

# The length objective is a separate object rather than a function of
# Curve itself.
obj = CurveLength(curve)

print('Initial curve length: ', obj.J())

# Each target function is then equipped with a shift and weight, to
# become a term in a least-squares objective function.
# A list of terms are combined to form a nonlinear-least-squares
# problem.
prob = LeastSquaresProblem.from_tuples([(obj.J, 0.0, 1.0)])

# At the initial condition, get the Jacobian two ways: analytic
# derivatives and finite differencing. The difference should be small.
# fd_jac = prob.dofs.fd_jac()
# jac = prob.dofs.jac()
# print('finite difference Jacobian:')
# print(fd_jac)
# print('Analytic Jacobian:')
# print(jac)
# print('Difference:')
# print(fd_jac - jac)

# Solve the minimization problem:
least_squares_serial_solve(prob)

print('At the optimum, x: ', prob.x)
print(' Final curve dofs: ', curve.local_full_x)
print(' Final curve length:    ', obj.J())
print(' Expected final length: ', 2 * np.pi * x0[0])
print(' objective function: ', prob.objective())
print("End of 1_Simple/minimize_curve_length.py")
print("==============================================")
