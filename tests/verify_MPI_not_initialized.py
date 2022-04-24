#!/usr/bin/env python

"""
Some HPC systems only allow MPI to be initialized when jobs are
submitted through slurm, not on login nodes.  To make the non-MPI
parts of simsopt usable on login nodes, then, we want to be sure that
simsopt does not initialize MPI (via mpi4py) unless classes are
imported that require MPI, such as MpiPartition or Vmec. This script
checks to be sure that importing the top-level simsopt module and
importing a simsopt geometry class do not initialize MPI.

We must do this check in an isolated script rather than in the set of
unit tests, because otherwise one of the unit tests that uses MPI
would initialize MPI.
"""

import sys
import simsopt
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

assert "mpi4py.MPI" not in sys.modules, \
    "Importing simsopt should not initialize MPI"

print("Verified that importing simsopt does not initialize MPI")
