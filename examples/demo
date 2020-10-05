#!/usr/bin/env python3

import sys
sys.path.append('..')
from simsopt import *
import logging

"""                                                                           
Minimize f(x,y,z) = ((x-1)/1)^2 + ((y-2)/2)^2 + ((z-3)/3)^2.                  
The optimum is at (x,y,z)=(1,2,3), and f=0 at this point.                     
"""

# To print out diagnostic information along the way, uncomment this
# next line:
#logging.basicConfig(level=logging.INFO)

# Define some Target objects that depend on Parameter objects. In the
# future these functions would involve codes like VMEC, but for now we
# just use the functions f(x) = x.
iden1 = Identity()
iden2 = Identity()
iden3 = Identity()

# Parameters are all not fixed by default, meaning they will not be
# optimized.  You can choose to exclude any subset of the parameters
# from the space of independent variables by setting their 'fixed'
# property to True.
#iden1.fixed[0] = True
#iden2.fixed[0] = True
#iden3.fixed[0] = True

# Each Target is then equipped with a shift and weight, to become a
# term in a least-squares objective function
term1 = LeastSquaresTerm(iden1.J, 1, 1)
term2 = LeastSquaresTerm(iden2.J, 2, 2)
term3 = LeastSquaresTerm(iden3.J, 3, 3)

# A list of terms are combined to form a nonlinear-least-squares problem.
prob = LeastSquaresProblem([term1, term2, term3])

# Solve the minimization problem:
prob.solve()

print("An optimum was found at x=", iden1.x, ", y=", iden2.x, \
          ", z=", iden3.x)
print("The minimum value of the objective function is ", prob.objective())
