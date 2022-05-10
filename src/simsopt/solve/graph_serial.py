from .serial import *

import warnings

warnings.warn("Importing of simsopt.solve.graph_serial is deprecated. "
              "Instead import simsopt.solve.serial module.", 
              DeprecationWarning, stacklevel=2)
