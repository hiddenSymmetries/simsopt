from .least_squares import *

import warnings

warnings.warn("Importing of simsopt.objectives.graph_least_squares is deprecated. "
              "Instead import simsopt.objectives.least_squares module.", 
              DeprecationWarning, stacklevel=2)
