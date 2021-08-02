#!/usr/bin/env python3

from simsopt.geo.graph_surface import SurfaceRZFourier
from simsopt.objectives.graph_least_squares import LeastSquaresProblem
from simsopt.solve.graph_serial import least_squares_serial_solve
"""
Optimize the minor radius and elongation of an axisymmetric torus to
obtain a desired volume and area using the graph method.
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
surf.fix('rc(0,0)')
surf.get_return_fn_names()

# Approach 1

prob1 = LeastSquaresProblem(depends_on=surf,
                            goals=[desired_area, desired_volume],
                            weights=[1, 1])
least_squares_serial_solve(prob1)

print("At the optimum using approach 1,")
print(" rc(m=1,n=0) = ", surf.get_rc(1, 0))
print(" zs(m=1,n=0) = ", surf.get_zs(1, 0))
print(" volume = ", surf.volume())
print(" area = ", surf.area())
print(" objective function = ", prob1.objective())
print(" -------------------------\n\n")


# Approach 2

surf2 = SurfaceRZFourier()
surf2.fix('rc(0,0)')
prob2 = LeastSquaresProblem(depends_on=surf2,
                            opt_return_fns=['area', 'volume'],
                            goals=[desired_area, desired_volume],
                            weights=[1, 1])
least_squares_serial_solve(prob2)
print("At the optimum using approach 2,")
print(" rc(m=1,n=0) = ", surf2.get_rc(1, 0))
print(" zs(m=1,n=0) = ", surf2.get_zs(1, 0))
print(" volume = ", surf2.volume())
print(" area = ", surf2.area())
print(" objective function = ", prob2.objective())
print(" -------------------------\n\n")


# Approach 3
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

# Approach 4
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

# Each target function is then equipped with a shift and weight, to
# become a term in a least-squares objective function
#term1 = (surf.volume, desired_volume, 1)
#term2 = (surf.area,   desired_area,   1)

# A list of terms are combined to form a nonlinear-least-squares
# problem.
#prob = LeastSquaresProblem([term1, term2])

# Solve the minimization problem:
#least_squares_serial_solve(prob)

#print("At the optimum,")
#print(" rc(m=1,n=0) = ", surf.get_rc(1, 0))
#print(" zs(m=1,n=0) = ", surf.get_zs(1, 0))
#print(" volume = ", surf.volume())
#print(" area = ", surf.area())
#print(" objective function = ", prob.objective())
