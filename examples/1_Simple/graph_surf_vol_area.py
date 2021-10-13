#!/usr/bin/env python3

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.graph_least_squares import LeastSquaresProblem
from simsopt.solve.graph_serial import least_squares_serial_solve
"""
Optimize the minor radius and elongation of an axisymmetric torus to
obtain a desired volume and area using the graph method.
"""

print("Running 1_Simple/graph_surf_vol_area.py")
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

# Approach 1
surf3 = SurfaceRZFourier()
surf3.fix('rc(0,0)')
prob3 = LeastSquaresProblem(funcs_in=[surf3.area, surf3.volume],
                            goals=[desired_area, desired_volume],
                            weights=[1, 1])
least_squares_serial_solve(prob3)
print("At the optimum using approach 3,")
print(" rc(m=1,n=0) = ", surf3.get_rc(1, 0))
print(" zs(m=1,n=0) = ", surf3.get_zs(1, 0))
print(" volume = ", surf3.volume())
print(" area = ", surf3.area())
print(" objective function = ", prob3.objective())
print(" -------------------------\n\n")

# Approach 2
surf4 = SurfaceRZFourier()
surf4.fix('rc(0,0)')
prob4 = LeastSquaresProblem.from_tuples([(surf4.area, desired_area, 1),
                                         (surf4.volume, desired_volume, 1)])
print(prob4)
least_squares_serial_solve(prob4)
print("At the optimum using approach 3,")
print(" rc(m=1,n=0) = ", surf4.get_rc(1, 0))
print(" zs(m=1,n=0) = ", surf4.get_zs(1, 0))
print(" volume = ", surf4.volume())
print(" area = ", surf4.area())
print(" objective function = ", prob4.objective())
print(" -------------------------\n\n")


print("End of 1_Simple/graph_surf_vol_area.py")
print("=======================================")
