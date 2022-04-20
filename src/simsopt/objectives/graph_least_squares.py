from .least_squares import *

import warnings

warnings.warn("Import of graph_least_squares module deprecated."
              " Instead use least_squares module",
              DeprecationWarning, stacklevel=2)