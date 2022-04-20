from .mpi import *

import warnings

warnings.warn("Import of graph_mpi module deprecated."
              " Instead use mpi module",
              DeprecationWarning, stacklevel=2)
