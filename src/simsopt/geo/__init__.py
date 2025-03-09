import jax
jax.config.update("jax_enable_x64", True)
from .config import *

from .curve import *
from .curvehelical import *
from .curverzfourier import *
from .curvexyzfourier import *
from .curvealongz import *
from .curvexyzfouriersymmetries import *
from .curveperturbed import *
from .curveobjectives import *
from .curveplanarfourier import *
from .framedcurve import *
from .finitebuild import *
from .plotting import *

from .boozersurface import *
from .qfmsurface import *
from .surface import *
from .surfacegarabedian import *
from .surfacehenneberg import *
from .surfaceobjectives import *
from .surfacerzfourier import *
from .surfacexyzfourier import *
from .surfacexyztensorfourier import *
from .strain_optimization import *

from .permanent_magnet_grid import *

__all__ = (curve.__all__ + curvehelical.__all__ +
           curverzfourier.__all__ + curvexyzfourier.__all__ +
           curvexyzfouriersymmetries.__all__ +
           curveperturbed.__all__ + curveobjectives.__all__ +
           curveplanarfourier.__all__ +
           finitebuild.__all__ + plotting.__all__ +
           boozersurface.__all__ + qfmsurface.__all__ +
           surface.__all__ +
           surfacegarabedian.__all__ + surfacehenneberg.__all__ +
           surfacerzfourier.__all__ + surfacexyzfourier.__all__ +
           surfacexyztensorfourier.__all__ + surfaceobjectives.__all__ +
           strain_optimization.__all__ + framedcurve.__all__ + 
           permanent_magnet_grid.__all__)
