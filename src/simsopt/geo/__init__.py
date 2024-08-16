import jax
jax.config.update("jax_enable_x64", True)
from .surface import *
from .surfaceobjectives import *
from .surfacerzfourier import *
from .plotting import*

__all__ = (plotting.__all__ +
           surface.__all__ +
           surfacerzfourier.__all__ +
           surfaceobjectives.__all__)
