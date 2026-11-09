"""Fresh-process child: the platforms ``set_backend`` starts and leaves running.

Usage: ``set_backend_platform_startup.py SCENARIO``. The parent exports the
scenario's ``JAX_PLATFORMS`` (or none) before this process starts, because JAX
picks its backends once, at first use.
"""

import os
import sys

import jax
import jax.extend.backend
import jax.numpy as jnp
from simsopt.geo.jit import native_jax_device
from simsopt_jax.backend import set_backend


def _initialized_platforms() -> set[str]:
    return set(jax.extend.backend.backends())


def gpu_at_startup() -> None:
    """Selecting the GPU first starts CUDA as the default and the CPU for native kernels."""
    set_backend("jax", device="gpu")
    assert jax.default_backend() == "gpu"
    assert _initialized_platforms() == {"cuda", "cpu"}, _initialized_platforms()
    assert native_jax_device().platform == "cpu", native_jax_device()
    assert {d.platform for d in jnp.zeros(2).devices()} == {"gpu"}
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"


def cpu_then_gpu_after_initialization() -> None:
    """After initialization, selections move the default device but not the platforms."""
    assert _initialized_platforms() == {"cuda", "cpu"}, _initialized_platforms()
    for device in ("cpu", "gpu"):
        set_backend("jax", device=device)
        assert {d.platform for d in jnp.zeros(2).devices()} == {device}
        assert native_jax_device().platform == "cpu", (device, native_jax_device())
        assert jax.config.values["jax_platforms"] == "cuda,cpu"
        assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"


def explicit_cuda_only() -> None:
    """An explicit ``JAX_PLATFORMS=cuda`` starts CUDA alone; native kernels use it."""
    set_backend("jax", device="gpu")
    assert _initialized_platforms() == {"cuda"}, _initialized_platforms()
    assert native_jax_device().platform == "gpu", native_jax_device()
    assert os.environ["JAX_PLATFORMS"] == "cuda"


def cpu_before_initialization() -> None:
    """Selecting the CPU before initialization keeps CUDA from starting."""
    set_backend("jax", device="cpu")
    assert _initialized_platforms() == {"cpu"}, _initialized_platforms()
    assert jax.default_backend() == "cpu"
    assert native_jax_device().platform == "cpu", native_jax_device()


_SCENARIOS = {
    "gpu_at_startup": gpu_at_startup,
    "cpu_then_gpu_after_initialization": cpu_then_gpu_after_initialization,
    "explicit_cuda_only": explicit_cuda_only,
    "cpu_before_initialization": cpu_before_initialization,
}


if __name__ == "__main__":
    _SCENARIOS[sys.argv[1]]()
