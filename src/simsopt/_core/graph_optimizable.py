from .optimizable import *

import warnings

warnings.warn("Importing of simsopt._core.graph_optimizable is deprecated. "
              "Instead import simsopt._core.optimizable module.", 
              DeprecationWarning, stacklevel=2)
