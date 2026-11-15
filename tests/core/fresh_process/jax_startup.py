"""Fresh-process child ``jax_startup.py SCENARIO``: what ``set_backend`` exports before
JAX starts (the parent sets the scenario's ``XLA_FLAGS`` and ``CUDA_VISIBLE_DEVICES``)."""

import os
import sys
import unittest
import warnings

import jax
import jax.extend.backend
from simsopt_jax.backend import get_backend_config, set_backend

_GPU_FLAGS = (
    "--xla_gpu_experimental_enable_fusion_autotuner=false",
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_exclude_nondeterministic_ops=true",
)


def cpu_flags() -> None:
    """The CPU preset joins the caller's flags, which XLA then reads (two host devices)."""
    set_backend("jax", device="cpu")
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    flags = "--xla_force_host_platform_device_count=2 --xla_cpu_opt_preset=FAST_COMPILE"
    assert os.environ["XLA_FLAGS"] == flags, os.environ["XLA_FLAGS"]
    assert set(jax.extend.backend.backends()) == {"cpu"}
    assert len(jax.devices()) == 2, jax.devices()


def gpu_flags() -> None:
    """The GPU flags join the caller's, whose autotune level stays; nothing warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        set_backend("jax", device="gpu")
    flags = " ".join(("--xla_gpu_autotune_level=2", _GPU_FLAGS[0], _GPU_FLAGS[2]))
    assert os.environ["XLA_FLAGS"] == flags, os.environ["XLA_FLAGS"]
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"
    assert jax.default_backend() == "gpu"


def gpu_missing() -> None:
    """Without a GPU the request raises after exporting the GPU flags, keeping the
    configuration; the CPU still starts afterwards."""
    previous = get_backend_config()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "cuda"):
        set_backend("jax", device="gpu")
    assert get_backend_config() is previous
    assert os.environ["XLA_FLAGS"] == " ".join(_GPU_FLAGS), os.environ["XLA_FLAGS"]
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"
    set_backend("jax", device="cpu")
    assert jax.default_backend() == "cpu"


if __name__ == "__main__":
    globals()[sys.argv[1]]()
