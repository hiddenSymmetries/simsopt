#!/usr/bin/env python3

from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve
from simsopt import load, save
"""
Optimize the minor radius and elongation of an axisymmetric torus to
obtain a desired volume and area.
"""

print("Running 1_Simple/surf_vol_area.py")
print("=======================================")
desired_volume = 0.6
desired_area = 8.0

# Start with a default surface, which is axisymmetric with major
# radius 1 and minor radius 0.1.
surf = SurfaceRZFourier()

# Parameters are all non-fixed by default, meaning they will be
# optimized.  You can choose to exclude any subset of the variables
# from the space of independent variables by setting their 'fixed'
# property to True.
surf.fix('rc(0,0)')

# LeastSquaresProble can be initialized in couple of ways. 
prob1 = LeastSquaresProblem(funcs_in=[surf.area, surf.volume],
                            goals=[desired_area, desired_volume],
                            weights=[1, 1])


# Optimize the problem using the defaults
least_squares_serial_solve(prob1)
print("At the first optimum")
print(" rc(m=1,n=0) = ", surf.get_rc(1, 0))
print(" zs(m=1,n=0) = ", surf.get_zs(1, 0))
print(" volume = ", surf.volume())
print(" area = ", surf.area())
print(" objective function = ", prob1.objective())
print(" -------------------------\n\n")

# Save the optimized surface
surf.save("surf_fw.json", indent=2)

# Load the saved surface, and redo optimization using central difference scheme
surf1 = load("surf_fw.json")

desired_volume1 = 0.8
desired_area1 = 9.0   # These are different from previous values

# An alternative initialization of LSP
prob2 = LeastSquaresProblem.from_tuples([(surf1.area, desired_area1, 1),
                                         (surf1.volume, desired_volume1, 1)])
least_squares_serial_solve(prob2, diff_method="centered")

print("\n\nAt the second optimum")
print(" rc(m=1,n=0) = ", surf1.get_rc(1, 0))
print(" zs(m=1,n=0) = ", surf1.get_zs(1, 0))
print(" volume = ", surf1.volume())
print(" area = ", surf1.area())
print(" objective function = ", prob2.objective())
save(surf1, "surf_centered.json", indent=2)
print(" -------------------------\n\n")
print("End of 1_Simple/surf_vol_area.py")
print("=======================================")
