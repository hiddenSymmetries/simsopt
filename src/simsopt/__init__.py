# ===================ATTENTION=================================================
# Don't abuse this file by importing all variables from all modules to top-level.
# Import only the important classes that should be at top-level.
# Follow the same logic in the sub-packages.
# ===================END ATTENTION=============================================

# Two ways of achieving the above-mentioned objective
# Use "from xyz import XYZ" style
# Define __all__ dunder at module and subpackage level. Then you could do
# "from xyz import *".  If xyz[.py] contains __all__ = ['XYZ'], only XYZ is
# imported

from ._core import make_optimizable, load, save

# VERSION info
from ._version import version as __version__

# Expose XSIMD depedency in simsoptpp
from simsoptpp import using_xsimd as __built_with_xsimd__
