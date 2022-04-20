from .functions import *

import warnings

warnings.warn("Importing of simsopt.objectives.graph_functions module is deprecated. "
              "Instead import simsopt.objectives.functions module",
              DeprecationWarning, stacklevel=2)