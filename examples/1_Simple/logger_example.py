#!/usr/bin/env python3

import logging
from simsopt.util.log import initialize_logging

"""
Example file for transparently logging both MPI and serial jobs
"""

# Serial logging
initialize_logging(filename='serial.log')
print("Running 1_Simple/logger_example.py")
print("==================================")
for i in range(2):
    logging.info("Hello (times %i) from serial job" % (i+1))

# MPI logging
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    comm = None
    print("MPI not found")

if comm is not None:
    initialize_logging(mpi=True, filename='mpi.log')
    for i in range(2):
        logging.warning("Hello (times %i) from mpi job" % (i+1))
print("End of 1_Simple/logger_example.py")
print("==================================")
