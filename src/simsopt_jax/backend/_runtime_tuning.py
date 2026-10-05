"""Topology and kernel tuning for the JAX backend.

Owns chunk, sharding, and field-kernel contracts plus the
pure builders and device probes that resolve them from policy and environment.
Process-global caches and locks live in :mod:`simsopt_jax.backend.runtime`;
the one exception is the set of sharding notices already logged, which only
this module's sharding builder reads and writes.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass

from typing import cast

import jax

from simsopt_jax.backend._runtime_policy import (
    BackendPolicy,
    _CHUNK_AUTOTUNE_ENV,
    _COIL_CHUNK_SIZE_ENV,
    _GPU_MEMORY_TOTAL_MB_ENV,
    _MIN_COILS_TO_SHARD_ENV,
    _MIN_POINTS_TO_SHARD_ENV,
    _POINT_CHUNK_SIZE_ENV,
    _QUADRATURE_BLOCK_SIZE_ENV,
    _SHARDING_AXIS_ENV,
    _SHARDING_COIL_AXIS_ENV,
    _SHARDING_STRATEGY_ENV,
    _TRUTHY_ENV_VALUES,
    _optional_env_value,
    _optional_nonempty_env,
    _optional_nonneg_int_env,
    _runtime_jax_backend_name,
)

_LOGGER = logging.getLogger(__name__)
_logged_sharding_notices: set[tuple[str, int]] = set()

_VALID_SHARDING_STRATEGIES = (
    "none",
    "points",
    "hybrid",
    "coil_groups",
    "points_coils",
)
_POINT_AXIS_SHARDING_STRATEGIES = frozenset(("points", "hybrid"))
_COIL_AXIS_SHARDING_STRATEGIES = frozenset(("coil_groups", "points_coils"))

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
_MODE_SHARDING_DEFAULTS = {
    "native_cpu": "none",
    "jax_cpu_fast": "none",
    "jax_cpu_parity": "none",
    "jax_gpu_parity": "none",
    "jax_gpu_fast": "hybrid",
}
_DEFAULT_SHARDING_AXIS_NAME = "d"
_DEFAULT_COIL_SHARDING_AXIS_NAME = "coil"
_MIN_POINTS_TO_SHARD_BY_POLICY = {
    "host_reference": 1 << 30,
    "stable_default": 4096,
    "performance_tuned": 2048,
}
_MIN_COILS_TO_SHARD_BY_POLICY = {
    "host_reference": 1 << 30,
    "stable_default": 8,
    "performance_tuned": 4,
}
_AUTOTUNED_CHUNK_SIZES_BY_POLICY = {
    "host_reference": (),
    "stable_default": (
        (
            8192,
            {
                "coil_chunk_size": 8,
                "quadrature_block_size": 0,
                "point_chunk_size": 128,
            },
        ),
        (
            16384,
            {
                "coil_chunk_size": 16,
                "quadrature_block_size": 0,
                "point_chunk_size": 256,
            },
        ),
        (
            32768,
            {
                "coil_chunk_size": 32,
                "quadrature_block_size": 0,
                "point_chunk_size": 512,
            },
        ),
        (
            None,
            {
                "coil_chunk_size": 64,
                "quadrature_block_size": 0,
                "point_chunk_size": 1024,
            },
        ),
    ),
    "performance_tuned": (
        (
            8192,
            {
                "coil_chunk_size": 32,
                "quadrature_block_size": 32,
                "point_chunk_size": 512,
            },
        ),
        (
            16384,
            {
                "coil_chunk_size": 64,
                "quadrature_block_size": 64,
                "point_chunk_size": 1024,
            },
        ),
        (
            32768,
            {
                "coil_chunk_size": 128,
                "quadrature_block_size": 128,
                "point_chunk_size": 2048,
            },
        ),
        (
            None,
            {
                "coil_chunk_size": 256,
                "quadrature_block_size": 256,
                "point_chunk_size": 4096,
            },
        ),
    ),
}
_FIELD_KERNEL_ENV_BY_KEY = {
    "coil_chunk_size": _COIL_CHUNK_SIZE_ENV,
    "quadrature_block_size": _QUADRATURE_BLOCK_SIZE_ENV,
    "point_chunk_size": _POINT_CHUNK_SIZE_ENV,
}


@dataclass(frozen=True)
class FieldKernelTuning:
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int
    point_chunk_size: int


@dataclass(frozen=True)
class ChunkTuning:
    mode: str
    chunk_policy: str
    coil_chunk_size: int
    quadrature_block_size: int
    point_chunk_size: int
    autotuned: bool
    autotune_source: str | None
    gpu_total_memory_mb: int | None


@dataclass(frozen=True)
class ShardingTuning:
    mode: str
    strategy: str
    mesh_axis_name: str
    point_axis_name: str
    coil_axis_name: str
    mesh_axes: tuple[str, ...]
    point_device_count: int
    coil_device_count: int
    reduced_axis_name: str | None
    min_points_to_shard: int
    min_coils_to_shard: int
    device_count: int
    local_device_count: int
    active: bool
    platform: str


def _validate_sharding_strategy(value: str, *, source: str) -> str:
    if value not in _VALID_SHARDING_STRATEGIES:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_SHARDING_STRATEGIES}"
        )
    return value


def _point_chunk_size_default(chunk_policy: str) -> int:
    return _POINT_CHUNK_SIZE_BY_POLICY.get(chunk_policy, 0)


def _resolve_chunk_autotune_enabled(policy: BackendPolicy) -> bool:
    raw_value = _optional_env_value(_CHUNK_AUTOTUNE_ENV)
    if raw_value is not None:
        return raw_value.strip().lower() in _TRUTHY_ENV_VALUES
    return policy.backend == "jax" and policy.jax_platform == "cuda"


def _resolve_sharding_strategy(mode: str, policy: BackendPolicy) -> str:
    del policy
    raw_value = _optional_nonempty_env(_SHARDING_STRATEGY_ENV)
    if raw_value is not None:
        return _validate_sharding_strategy(raw_value, source=_SHARDING_STRATEGY_ENV)
    return _MODE_SHARDING_DEFAULTS[mode]


def _resolve_sharding_axis_name() -> str:
    raw_value = _optional_nonempty_env(_SHARDING_AXIS_ENV)
    if raw_value is None:
        return _DEFAULT_SHARDING_AXIS_NAME
    return raw_value


def _resolve_coil_sharding_axis_name() -> str:
    raw_value = _optional_nonempty_env(_SHARDING_COIL_AXIS_ENV)
    if raw_value is None:
        return _DEFAULT_COIL_SHARDING_AXIS_NAME
    return raw_value


def _resolve_min_coils_to_shard(policy: BackendPolicy) -> int:
    value = _optional_nonneg_int_env(_MIN_COILS_TO_SHARD_ENV)
    if value is not None:
        return value
    return _MIN_COILS_TO_SHARD_BY_POLICY[policy.chunk_policy]


def _factor_device_count_2d(device_count: int) -> tuple[int, int]:
    """Factor ``device_count`` into ``(point_count, coil_count)`` for a 2D mesh.

    Picks the factor pair closest to a square mesh. ``device_count`` must be
    positive; prime device counts yield ``(1, device_count)`` which still
    constitutes a valid 2D mesh per the JAX shard_map contract.
    """
    if device_count <= 0:
        raise ValueError("points_coils sharding requires device_count > 0.")
    best = (1, device_count)
    best_aspect = float(device_count)
    for point_count in range(1, int(device_count**0.5) + 1):
        if device_count % point_count != 0:
            continue
        coil_count = device_count // point_count
        aspect = max(point_count, coil_count) / min(point_count, coil_count)
        if aspect < best_aspect:
            best = (point_count, coil_count)
            best_aspect = aspect
    return best


def _strategy_device_counts(strategy: str, device_count: int) -> tuple[int, int]:
    if strategy == "none":
        return (0, 0)
    if device_count <= 0:
        raise ValueError(f"{strategy} sharding requires device_count > 0.")
    if strategy == "coil_groups":
        return (1, device_count)
    if strategy in _POINT_AXIS_SHARDING_STRATEGIES:
        return (device_count, 1)
    if strategy == "points_coils":
        return _factor_device_count_2d(device_count)
    raise ValueError(f"unsupported sharding strategy {strategy!r}")


def _strategy_mesh_axis_names(
    strategy: str,
    *,
    point_axis_name: str,
    coil_axis_name: str,
) -> tuple[str, ...]:
    if strategy == "coil_groups":
        return (coil_axis_name,)
    if strategy in _POINT_AXIS_SHARDING_STRATEGIES:
        return (point_axis_name,)
    if strategy == "points_coils":
        return (point_axis_name, coil_axis_name)
    return ()


def _strategy_reduced_axis_name(strategy: str, *, coil_axis_name: str) -> str | None:
    if strategy in _COIL_AXIS_SHARDING_STRATEGIES:
        return coil_axis_name
    return None


def _resolve_min_points_to_shard(policy: BackendPolicy) -> int:
    env_value = _optional_nonneg_int_env(_MIN_POINTS_TO_SHARD_ENV)
    if env_value is not None:
        return env_value
    return _MIN_POINTS_TO_SHARD_BY_POLICY.get(policy.chunk_policy, 0)


def _detect_local_jax_device_count(policy: BackendPolicy) -> int:
    backend_name = _runtime_jax_backend_name(policy.jax_platform)
    return len(jax.local_devices(backend=backend_name))


def _visible_cuda_device_selector() -> str | None:
    raw_value = _optional_env_value("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return None
    first = raw_value.split(",", 1)[0].strip()
    if not first or first in {"-1", "none", "NoDevFiles"}:
        return None
    return first


def _parse_visible_cuda_device_index() -> int | None:
    # External-input parse contract: CUDA_VISIBLE_DEVICES env value.
    # The narrow ValueError catch handles non-integer selectors (e.g. UUIDs
    # or whitespace), not runtime errors.
    selector = _visible_cuda_device_selector()
    if selector is None:
        return None
    try:
        value = int(selector)
    except ValueError:
        return None
    return value if value >= 0 else None


def _detect_imported_jax_cuda_device_index() -> int | None:
    jax = sys.modules.get("jax")
    if jax is None:
        return None
    local_devices = getattr(jax, "local_devices", None)
    if not callable(local_devices):
        return None
    try:
        devices = local_devices(backend="gpu")
    except RuntimeError:
        # GPU backend not available on this host: detection returns None.
        return None
    if not devices:
        return None
    device = cast(list[object], devices)[0]
    for attr in ("local_hardware_id", "id"):
        value = getattr(device, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _detect_active_jax_cuda_device_selector() -> int | str | None:
    runtime_index = _detect_imported_jax_cuda_device_index()
    if runtime_index is not None:
        return runtime_index
    return _visible_cuda_device_selector()


def _parse_nvidia_smi_indexed_value_row(raw_row: str) -> tuple[int, float] | None:
    # External-input parse contract: a single nvidia-smi CSV row. The
    # narrow ValueError catch handles malformed rows from the external
    # tool, not runtime errors in this process.
    fields = [field.strip() for field in raw_row.split(",")]
    if len(fields) != 2:
        return None
    try:
        return int(float(fields[0])), float(fields[1])
    except ValueError:
        return None


def _query_gpu_metric_mb_from_nvidia_smi(
    metric_name: str,
    device_selector: int | str | None = None,
) -> float | None:
    command = [
        "nvidia-smi",
        f"--query-gpu=index,{metric_name}",
        "--format=csv,noheader,nounits",
    ]
    if device_selector is not None:
        command.extend(["-i", str(device_selector)])
    # External-tool availability boundary: nvidia-smi may be absent from
    # PATH (FileNotFoundError) or exit non-zero on hosts without an
    # NVIDIA driver (CalledProcessError). Both are expected absence
    # signals, not runtime errors; return None so the caller can continue
    # the ordered GPU-detection chain.
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    for line in lines:
        parsed = _parse_nvidia_smi_indexed_value_row(line)
        if parsed is None:
            continue
        index, value = parsed
        if isinstance(device_selector, int) and index != device_selector:
            continue
        if value >= 0:
            return float(value)
    return None


def _query_gpu_total_memory_mb_from_nvidia_smi(
    device_selector: int | str | None = None,
) -> int | None:
    value = _query_gpu_metric_mb_from_nvidia_smi("memory.total", device_selector)
    if value is None or value <= 0:
        return None
    return int(value)


def _resolve_gpu_total_memory_mb(
    policy: BackendPolicy,
) -> tuple[int | None, str | None]:
    if policy.jax_platform != "cuda":
        return None, None
    env_value = _optional_nonneg_int_env(_GPU_MEMORY_TOTAL_MB_ENV)
    if env_value is not None:
        if env_value == 0:
            raise ValueError(f"{_GPU_MEMORY_TOTAL_MB_ENV} must be > 0 when set")
        return env_value, _GPU_MEMORY_TOTAL_MB_ENV
    device_selector = _detect_active_jax_cuda_device_selector()
    detected = _query_gpu_total_memory_mb_from_nvidia_smi(device_selector)
    if detected is None:
        return None, None
    if device_selector is None:
        return detected, "nvidia-smi"
    return detected, f"nvidia-smi[{device_selector}]"


def _resolve_autotuned_chunk_sizes(
    chunk_policy: str,
    gpu_total_memory_mb: int | None,
) -> dict[str, int] | None:
    if gpu_total_memory_mb is None:
        return None
    buckets = _AUTOTUNED_CHUNK_SIZES_BY_POLICY.get(chunk_policy, ())
    for max_total_mb, sizes in buckets:
        if max_total_mb is None or gpu_total_memory_mb <= max_total_mb:
            return dict(sizes)
    return None


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


def _build_chunk_tuning(
    mode: str,
    policy: BackendPolicy,
) -> ChunkTuning:
    chunk_sizes = _static_chunk_sizes(mode, policy.chunk_policy)
    autotuned = False
    autotune_source = None
    gpu_total_memory_mb = None
    if _resolve_chunk_autotune_enabled(policy):
        gpu_total_memory_mb, autotune_source = _resolve_gpu_total_memory_mb(policy)
        autotuned_chunk_sizes = _resolve_autotuned_chunk_sizes(
            policy.chunk_policy,
            gpu_total_memory_mb,
        )
        if autotuned_chunk_sizes is not None:
            chunk_sizes.update(autotuned_chunk_sizes)
            autotuned = True
    chunk_sizes = _apply_chunk_env_overrides(chunk_sizes)
    effective_chunk_policy = policy.chunk_policy
    if policy.transfer_guard == "disallow":
        effective_chunk_policy = f"{policy.chunk_policy}_dense_audit"
        chunk_sizes["coil_chunk_size"] = 0
        chunk_sizes["quadrature_block_size"] = 0
        chunk_sizes["point_chunk_size"] = 0
    return ChunkTuning(
        mode=mode,
        chunk_policy=effective_chunk_policy,
        coil_chunk_size=chunk_sizes["coil_chunk_size"],
        quadrature_block_size=chunk_sizes["quadrature_block_size"],
        point_chunk_size=chunk_sizes["point_chunk_size"],
        autotuned=autotuned,
        autotune_source=autotune_source,
        gpu_total_memory_mb=gpu_total_memory_mb,
    )


def _build_sharding_tuning(
    mode: str,
    policy: BackendPolicy,
) -> ShardingTuning:
    strategy = _resolve_sharding_strategy(mode, policy)
    if policy.backend != "jax":
        strategy = "none"
        local_device_count = 0
        device_count = 0
    else:
        local_device_count = _detect_local_jax_device_count(policy)
        device_count = local_device_count
    if mode == "jax_gpu_parity" and strategy == "none" and local_device_count > 1:
        if (mode, local_device_count) not in _logged_sharding_notices:
            _logged_sharding_notices.add((mode, local_device_count))
            _LOGGER.info(
                "jax_gpu_parity defaults to single-device execution on %s local "
                "devices; set SIMSOPT_JAX_SHARDING=hybrid to opt into "
                "point sharding.",
                local_device_count,
            )
    point_axis_name = _resolve_sharding_axis_name()
    coil_axis_name = _resolve_coil_sharding_axis_name()
    point_device_count, coil_device_count = _strategy_device_counts(
        strategy,
        device_count,
    )
    return ShardingTuning(
        mode=mode,
        strategy=strategy,
        mesh_axis_name=point_axis_name,
        point_axis_name=point_axis_name,
        coil_axis_name=coil_axis_name,
        mesh_axes=_strategy_mesh_axis_names(
            strategy,
            point_axis_name=point_axis_name,
            coil_axis_name=coil_axis_name,
        ),
        point_device_count=point_device_count,
        coil_device_count=coil_device_count,
        reduced_axis_name=_strategy_reduced_axis_name(
            strategy,
            coil_axis_name=coil_axis_name,
        ),
        min_points_to_shard=_resolve_min_points_to_shard(policy),
        min_coils_to_shard=_resolve_min_coils_to_shard(policy),
        device_count=device_count,
        local_device_count=local_device_count,
        active=strategy != "none" and device_count > 1,
        platform=policy.jax_platform,
    )
