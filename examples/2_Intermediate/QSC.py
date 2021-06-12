#!/usr/bin/env python3

import logging
import numpy as np
from qsc import Qsc
from simsopt import make_optimizable
from simsopt import LeastSquaresProblem
from simsopt import least_squares_serial_solve

"""
Optimize an axis shape and the first-order shape of the flux surface
at first order near the magnetic axis for a target iota and low elongation
using the Stellarator Quasisymmetry Construction code
https://github.com/landreman/pyQSC
"""

#logging.basicConfig(level=logging.INFO)

stel = make_optimizable(Qsc(rc=[1, 0.045], zs=[0, 0.045], etabar=0.9, nfp=3, nphi=31))
print('Initial dofs: ', stel.get_dofs())
print('Names of the dofs: ', stel.names)

# Decide which degrees of freedom to optimize
stel.all_fixed()
stel.set_fixed('rc(1)', False)
stel.set_fixed('zs(1)', False)
stel.set_fixed('etabar', False)

# Each target function is then equipped with a shift and weight, to
# become a term in a least-squares objective function
term1 = (stel, 'iota', -0.5, 1.0)
term2 = (stel, 'max_elongation', 0.0, 0.0001)
# Note the weight on elongation must be much smaller than the weight on iota!

# A list of terms are combined to form a nonlinear-least-squares
# problem.
prob = LeastSquaresProblem([term1, term2])

print('Before optimization:')
print(' Global state vector: ', prob.x)
print(' QSC dofs: ', stel.get_dofs())
print(' Iota: ', stel.iota)
print(' Max elongation: ', stel.max_elongation)
print(' objective function: ', prob.objective())

# Solve the minimization problem:
least_squares_serial_solve(prob)

print('At the optimum:')
print(' Global state vector: ', prob.x)
print(' QSC dofs: ', stel.get_dofs())
print(' Iota: ', stel.iota)
print(' Max elongation: ', stel.max_elongation)
print(' objective function: ', prob.objective())
