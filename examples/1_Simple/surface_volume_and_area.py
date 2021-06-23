#!/usr/bin/env python3

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt import LeastSquaresProblem
from simsopt import least_squares_serial_solve
"""
Optimize the minor radius and elongation of an axisymmetric torus to
obtain a desired volume and area.
"""

desired_volume = 0.6
desired_area = 8.0

# Start with a default surface, which is axisymmetric with major
# radius 1 and minor radius 0.1.
surf = SurfaceRZFourier()

# Parameters are all non-fixed by default, meaning they will be
# optimized.  You can choose to exclude any subset of the variables
# from the space of independent variables by setting their 'fixed'
# property to True.
surf.set_fixed('rc(0,0)')

# Each target function is then equipped with a shift and weight, to
# become a term in a least-squares objective function
term1 = (surf.volume, desired_volume, 1)
term2 = (surf.area, desired_area, 1)

# A list of terms are combined to form a nonlinear-least-squares
# problem.
prob = LeastSquaresProblem([term1, term2])

# Solve the minimization problem:
least_squares_serial_solve(prob)

print("At the optimum,")
print(" rc(m=1,n=0) = ", surf.get_rc(1, 0))
print(" zs(m=1,n=0) = ", surf.get_zs(1, 0))
print(" volume = ", surf.volume())
print(" area = ", surf.area())
print(" objective function = ", prob.objective())
