from .mpi import *

import warnings

warnings.warn("Importing of simsopt.solve.graph_mpi module is deprecated. "
              "Instead import simsopt.solve.mpi module",
              DeprecationWarning, stacklevel=2)