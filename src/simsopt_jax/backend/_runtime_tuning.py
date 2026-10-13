"""Field-kernel chunk tuning for the JAX backend.

Owns the chunk contract plus the pure builder that resolves it from policy and
environment. Process-global caches and locks live in
:mod:`simsopt_jax.backend.runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass

from simsopt_jax.backend._runtime_policy import (
    BackendPolicy,
    _COIL_CHUNK_SIZE_ENV,
    _POINT_CHUNK_SIZE_ENV,
    _QUADRATURE_BLOCK_SIZE_ENV,
    _optional_nonneg_int_env,
)

_FIELD_KERNEL_DEFAULTS = {
    "native_cpu": {"coil_chunk_size": 0, "quadrature_block_size": 0},
    "jax_cpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64},
    "jax_cpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0},
    "jax_gpu_parity": {"coil_chunk_size": 16, "quadrature_block_size": 0},
    "jax_gpu_fast": {"coil_chunk_size": 64, "quadrature_block_size": 64},
}
_POINT_CHUNK_SIZE_BY_POLICY = {
    "host_reference": 0,
    "stable_default": 256,
    "performance_tuned": 1024,
}
_FIELD_KERNEL_ENV_BY_KEY = {
    "coil_chunk_size": _COIL_CHUNK_SIZE_ENV,
    "quadrature_block_size": _QUADRATURE_BLOCK_SIZE_ENV,
    "point_chunk_size": _POINT_CHUNK_SIZE_ENV,
}


@dataclass(frozen=True)
class FieldKernelTuning:
    """Immutable tile sizes for Biot-Savart field evaluation.

    Args:
        mode (str): Canonical backend mode: native_cpu, jax_cpu_fast, jax_cpu_parity,
            jax_gpu_fast or jax_gpu_parity.
        chunk_policy (str): Resolved tiling-policy name; a dense_audit suffix denotes
            disabled tiling.
        coil_chunk_size (int): Coils per tile; zero disables coil tiling.
        quadrature_block_size (int): Quadrature nodes per block; zero disables blocking.
        point_chunk_size (int): Evaluation points per tile; zero disables point tiling.
    """
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int
    point_chunk_size: int


def _point_chunk_size_default(chunk_policy: str) -> int:
    return _POINT_CHUNK_SIZE_BY_POLICY.get(chunk_policy, 0)


def _static_chunk_sizes(mode: str, chunk_policy: str) -> dict[str, int]:
    return {
        "coil_chunk_size": _FIELD_KERNEL_DEFAULTS[mode]["coil_chunk_size"],
        "quadrature_block_size": _FIELD_KERNEL_DEFAULTS[mode]["quadrature_block_size"],
        "point_chunk_size": _point_chunk_size_default(chunk_policy),
    }


def _apply_chunk_env_overrides(chunk_sizes: dict[str, int]) -> dict[str, int]:
    resolved = dict(chunk_sizes)
    for key, env_name in _FIELD_KERNEL_ENV_BY_KEY.items():
        value = _optional_nonneg_int_env(env_name)
        if value is not None:
            resolved[key] = value
    return resolved


def _build_field_kernel_tuning(
    mode: str,
    policy: BackendPolicy,
) -> FieldKernelTuning:
    chunk_sizes = _apply_chunk_env_overrides(
        _static_chunk_sizes(mode, policy.chunk_policy)
    )
    effective_chunk_policy = policy.chunk_policy
    if policy.transfer_guard == "disallow":
        effective_chunk_policy = f"{policy.chunk_policy}_dense_audit"
        chunk_sizes["coil_chunk_size"] = 0
        chunk_sizes["quadrature_block_size"] = 0
        chunk_sizes["point_chunk_size"] = 0
    return FieldKernelTuning(
        mode=mode,
        chunk_policy=effective_chunk_policy,
        coil_chunk_size=chunk_sizes["coil_chunk_size"],
        quadrature_block_size=chunk_sizes["quadrature_block_size"],
        point_chunk_size=chunk_sizes["point_chunk_size"],
    )
