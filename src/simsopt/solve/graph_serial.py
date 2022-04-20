from .serial import *

import warnings

warnings.warn("Import of graph_serial module deprecated."
              " Instead use serial module",
              DeprecationWarning, stacklevel=2)