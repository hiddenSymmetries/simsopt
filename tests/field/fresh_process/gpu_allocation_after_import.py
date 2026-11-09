"""Fresh-process child: GPU allocator settings apply after importing JAX.

Installing them must not initialize JAX's devices, which only a process that
has not touched them yet can show. A standard XLA variable no simsopt setting
resolves keeps the caller's value, and the memory fraction goes to
``XLA_CLIENT_MEM_FRACTION``, which jaxlib reads for every allocator, whichever
allocator the caller exported.
"""

import os

import jax  # noqa: F401
from simsopt_jax.backend._runtime_policy import _config_from_mode
from simsopt_jax.backend.runtime import (
    _apply_jax_gpu_memory_env,
    _jax_backends_initialized,
)


def main() -> None:
    """Apply the allocator environment and check JAX stays uninitialized."""
    assert not _jax_backends_initialized()
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    config = _config_from_mode(
        "jax_gpu_fast",
        strict=False,
        xla_gpu_preallocate=False,
        xla_gpu_mem_fraction=0.5,
    )
    _apply_jax_gpu_memory_env(config)
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["XLA_CLIENT_MEM_FRACTION"] == "0.5"
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "vmm"
    assert os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
    assert not _jax_backends_initialized()


if __name__ == "__main__":
    main()
