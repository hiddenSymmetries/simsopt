import os

import jax
from jax import jit as jaxjit

from .config import parameters

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    jax.config.update("jax_platform_name", "cpu")
if "JAX_ENABLE_X64" not in os.environ:
    jax.config.update("jax_enable_x64", True)


def native_jax_device():
    """Device for the native length mean and incremental-arclength norm VJP.

    These paths use explicit ``device_put``/``device_get`` on the host CPU when
    available. For C++ curves, CurveLength values and gradients stay host-owned
    under ``jax.transfer_guard("disallow")`` even with a default CUDA backend.
    JAX-backed native curves (JaxCurve subclasses such as CurveHelical) still
    transfer implicitly in their coefficient VJP; this guarantee does not
    extend to those paths. With CUDA alone (``JAX_PLATFORMS=cuda``), the length
    kernels use the default device through the same explicit transfers.
    """
    platforms = jax.config.jax_platforms
    if platforms and "cpu" not in platforms.split(","):
        return jax.devices()[0]
    return jax.devices("cpu")[0]


def jit(fun, **args):
    if parameters['jit']:
        return jaxjit(fun, **args)
    else:
        return fun
