#!/usr/bin/env python

import os
import logging
import numpy as np
from simsopt.util import MpiPartition, log
from simsopt.mhd import Spec, Residue
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve

"""
In this example, we show how the shape of a boundary magnetic
surface can be adjusted to eliminate magnetic islands inside it,
considering a vacuum field. For this example we will use the SPEC code
with a single radial domain. The geometry comes from a quasi-helically
symmetric configuration developed at the University of Wisconsin. We
will eliminate the islands by minimizing an objective function
involving Greene's residue for several O-points and X-points, similar
to the approach of Hanson and Cary (1984).
"""
print("Running 2_Intermediate/eliminate_magnetic_islands.py")
print("====================================================")

log()

mpi = MpiPartition()
mpi.write()

# Initialze a Spec object from a standard SPEC input file:
s = Spec(os.path.join(os.path.dirname(__file__), 'inputs', 'QH-residues.sp'),
         mpi=mpi)

# Expand number of Fourier modes to include larger poloidal mode numbers:
s.boundary.change_resolution(6, s.boundary.ntor)
# To make this example run relatively quickly, we will optimize in a
# small parameter space. Here we pick out just 2 Fourier modes to vary
# in the optimization:
s.boundary.fix_all()
s.boundary.unfix('zs(6,1)')
s.boundary.unfix('zs(6,2)')

# The main resonant surface is iota = p / q:
p = -8
q = 7
# Guess for radial location of the island chain:
s_guess = 0.9

residue1 = Residue(s, p, q, s_guess=s_guess)
residue2 = Residue(s, p, q, s_guess=s_guess, theta=np.pi)

initial_r1 = residue1.J()                                                                    
initial_r2 = residue2.J()                                                                    
logging.info(f"Initial residues: {initial_r1}, {initial_r2}")
#exit(0)

# There is another island chain we'd like to control at iota = -12/11:
p = -12
q = 11
s_guess = -0.1

residue3 = Residue(s, p, q, s_guess=s_guess)
residue4 = Residue(s, p, q, s_guess=s_guess, theta=np.pi)

# Objective function is \sum_j residue_j ** 2
prob = LeastSquaresProblem.from_tuples([(residue1.J, 0, 1),
                                        (residue2.J, 0, 1),
                                        (residue3.J, 0, 1),
                                        (residue4.J, 0, 1)])

# Solve the optimization problem:
least_squares_mpi_solve(prob, mpi=mpi, grad=True)

final_r1 = residue1.J()                                                                    
final_r2 = residue2.J()                                                                    
expected_solution = np.array([1.1076171888771095e-03, 4.5277618989828059e-04])
if mpi.proc0_world:
    logging.info(f"Final state vector: zs(6,1)={prob.x[0]}, zs(6,2)={prob.x[1]}")
    logging.info(f"Expected state vector: {expected_solution}")
    logging.info(f"Difference from expected solution: {prob.x - expected_solution}")
    logging.info(f"Final residues: {final_r1}, {final_r2}")

np.testing.assert_allclose(prob.x, expected_solution, rtol=1e-2)

print("End of 2_Intermediate/eliminate_magnetic_islands.py")
print("===================================================")
