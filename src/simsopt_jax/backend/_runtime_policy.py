"""Device, precision, chunking and memory configuration for JAX field evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypeVar, cast


_ExplicitT = TypeVar("_ExplicitT")
_ResolvedT = TypeVar("_ResolvedT")

PrecisionSelection = Literal["mode_default", "fp64"]
BackendMode = Literal[
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_gpu_fast",
    "jax_gpu_parity",
]
JaxDevice = Literal["cpu", "gpu"]
ExecutionIntent = Literal["fast", "parity"]

_VALID_BACKENDS = ("cpu", "jax")
_VALID_PLATFORMS = ("cpu", "cuda")
_VALID_POLICY_DTYPES = ("float64",)
_VALID_PRECISION_SELECTIONS = ("mode_default", "fp64")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

_BACKEND_ENV = "SIMSOPT_BACKEND"
_PLATFORM_ENV = "SIMSOPT_JAX_PLATFORM"
_MODE_ENV = "SIMSOPT_BACKEND_MODE"
_PRECISION_ENV = "SIMSOPT_PRECISION"
_STRICT_ENV = "SIMSOPT_BACKEND_STRICT"
_DEBUG_ENV = "SIMSOPT_DEBUG"
_DEBUG_NANS_ENV = "SIMSOPT_JAX_DEBUG_NANS"
_DISABLE_JIT_ENV = "SIMSOPT_JAX_DISABLE_JIT"
_TRANSFER_GUARD_ENV = "SIMSOPT_JAX_TRANSFER_GUARD"
_COMPILATION_CACHE_DIR_ENV = "SIMSOPT_JAX_COMPILATION_CACHE_DIR"
_JAX_COMPILATION_CACHE_DIR_ENV = "JAX_COMPILATION_CACHE_DIR"
_COIL_CHUNK_SIZE_ENV = "SIMSOPT_JAX_COIL_CHUNK_SIZE"
_QUADRATURE_BLOCK_SIZE_ENV = "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE"
_POINT_CHUNK_SIZE_ENV = "SIMSOPT_JAX_POINT_CHUNK_SIZE"
_CHUNK_AUTOTUNE_ENV = "SIMSOPT_JAX_CHUNK_AUTOTUNE"
_GPU_MEMORY_TOTAL_MB_ENV = "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB"
_GPU_PREALLOCATE_ENV = "SIMSOPT_JAX_GPU_PREALLOCATE"
_GPU_MEM_FRACTION_ENV = "SIMSOPT_JAX_GPU_MEM_FRACTION"
_GPU_ALLOCATOR_ENV = "SIMSOPT_JAX_GPU_ALLOCATOR"
_TF_GPU_ALLOCATOR_OVERRIDE_ENV = "SIMSOPT_TF_GPU_ALLOCATOR"
_SHARDING_STRATEGY_ENV = "SIMSOPT_JAX_SHARDING"
_SHARDING_AXIS_ENV = "SIMSOPT_JAX_SHARDING_AXIS"
_SHARDING_COIL_AXIS_ENV = "SIMSOPT_JAX_COIL_SHARDING_AXIS"
_MIN_POINTS_TO_SHARD_ENV = "SIMSOPT_JAX_MIN_POINTS_TO_SHARD"
_MIN_COILS_TO_SHARD_ENV = "SIMSOPT_JAX_MIN_COILS_TO_SHARD"
_JAX_PLATFORMS_ENV = "JAX_PLATFORMS"
_XLA_FLAGS_ENV = "XLA_FLAGS"
_XLA_PYTHON_CLIENT_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
_XLA_PYTHON_CLIENT_MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"
_XLA_PYTHON_CLIENT_ALLOCATOR_ENV = "XLA_PYTHON_CLIENT_ALLOCATOR"
_XLA_CLIENT_MEM_FRACTION_ENV = "XLA_CLIENT_MEM_FRACTION"
_TF_GPU_ALLOCATOR_ENV = "TF_GPU_ALLOCATOR"
_VALID_TRANSFER_GUARDS = ("allow", "log", "disallow")
_VALID_GPU_ALLOCATORS = ("platform", "vmm")
_VALID_TF_GPU_ALLOCATORS = ("cuda_malloc_async",)
_SYNCED_RUNTIME_ENV_VALUES = (
    (_MODE_ENV, "mode"),
    (_PRECISION_ENV, "precision"),
    (_STRICT_ENV, "strict"),
    (_DEBUG_NANS_ENV, "debug_nans"),
    (_DISABLE_JIT_ENV, "disable_jit"),
    (_TRANSFER_GUARD_ENV, "transfer_guard"),
    (_COMPILATION_CACHE_DIR_ENV, "compilation_cache_dir"),
    (_GPU_PREALLOCATE_ENV, "xla_gpu_preallocate"),
    (_GPU_MEM_FRACTION_ENV, "xla_gpu_mem_fraction"),
    (_GPU_ALLOCATOR_ENV, "xla_gpu_allocator"),
    (_TF_GPU_ALLOCATOR_OVERRIDE_ENV, "tf_gpu_allocator"),
    (_BACKEND_ENV, "backend"),
    (_PLATFORM_ENV, "jax_platform"),
    (_JAX_PLATFORMS_ENV, "jax_platforms"),
)
VALID_BACKEND_MODES: tuple[BackendMode, ...] = (
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_gpu_fast",
    "jax_gpu_parity",
)

_JAX_EXECUTION_MODES: dict[tuple[JaxDevice, ExecutionIntent], BackendMode] = {
    ("cpu", "fast"): "jax_cpu_fast",
    ("cpu", "parity"): "jax_cpu_parity",
    ("gpu", "fast"): "jax_gpu_fast",
    ("gpu", "parity"): "jax_gpu_parity",
}

_MODE_TO_RUNTIME = {
    "native_cpu": ("cpu", "cpu"),
    "jax_cpu_fast": ("jax", "cpu"),
    "jax_cpu_parity": ("jax", "cpu"),
    "jax_gpu_parity": ("jax", "cuda"),
    "jax_gpu_fast": ("jax", "cuda"),
}

_NO_GPU_MEMORY_DEFAULTS = {
    "xla_gpu_preallocate": None,
    "xla_gpu_mem_fraction": None,
    "xla_gpu_allocator": None,
    "tf_gpu_allocator": None,
}
_GPU_MEMORY_MODE_DEFAULTS = {
    "xla_gpu_preallocate": False,
    "xla_gpu_mem_fraction": None,
    "xla_gpu_allocator": None,
    "tf_gpu_allocator": None,
}

_MODE_POLICY_DEFAULTS = {
    "native_cpu": {
        "parity_mode": False,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "chunk_policy": "host_reference",
        "matmul_precision": "highest",
        **_NO_GPU_MEMORY_DEFAULTS,
    },
    "jax_cpu_fast": {
        "parity_mode": False,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "chunk_policy": "performance_tuned",
        "matmul_precision": "default",
        **_NO_GPU_MEMORY_DEFAULTS,
    },
    "jax_cpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "chunk_policy": "stable_default",
        "matmul_precision": "highest",
        **_NO_GPU_MEMORY_DEFAULTS,
    },
    "jax_gpu_parity": {
        "parity_mode": True,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "chunk_policy": "stable_default",
        "matmul_precision": "highest",
        **_GPU_MEMORY_MODE_DEFAULTS,
    },
    "jax_gpu_fast": {
        "parity_mode": False,
        "requires_x64": True,
        "runtime_dtype": "float64",
        "host_dtype": "float64",
        "chunk_policy": "performance_tuned",
        "matmul_precision": "default",
        **_GPU_MEMORY_MODE_DEFAULTS,
    },
}

_DEFAULT_TRANSFER_GUARD_BY_MODE = {
    "native_cpu": None,
    "jax_cpu_fast": "log",
    "jax_cpu_parity": "log",
    "jax_gpu_parity": "log",
    "jax_gpu_fast": "log",
}


@dataclass(frozen=True)
class BackendConfig:
    mode: BackendMode
    backend: str
    jax_platform: str
    precision: PrecisionSelection = "mode_default"
    strict: bool = False
    debug_nans: bool = False
    disable_jit: bool = False
    transfer_guard: str | None = None
    compilation_cache_dir: str | None = None
    xla_gpu_preallocate: bool | None = None
    xla_gpu_mem_fraction: float | None = None
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None


@dataclass(frozen=True)
class BackendPolicy:
    """Immutable numerical and execution policy for a resolved field backend mode."""

    mode: BackendMode
    backend: str
    jax_platform: str
    parity_mode: bool
    requires_x64: bool
    runtime_dtype: str
    host_dtype: str
    compute_dtype: str
    chunk_policy: str
    matmul_precision: str
    transfer_guard: str | None


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in _TRUTHY_ENV_VALUES


def _parse_bool_value(raw_value: str, *, source: str) -> bool:
    value = raw_value.strip().lower()
    if value in _TRUTHY_ENV_VALUES:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{source}={raw_value!r} must be a boolean value")


def _optional_bool_env(name: str) -> bool | None:
    raw_value = _optional_env_value(name)
    if raw_value is None:
        return None
    return _parse_bool_value(raw_value, source=name)


def _validate_gpu_allocator(
    value: object | None,
    *,
    source: str,
) -> Literal["platform", "vmm"] | None:
    if value in (None, ""):
        return None
    if value == "platform":
        return "platform"
    if value == "vmm":
        return "vmm"
    raise ValueError(
        f"{source}={value!r} is not valid. Accepted: {_VALID_GPU_ALLOCATORS}"
    )


def _validate_tf_gpu_allocator(
    value: object | None,
    *,
    source: str,
) -> Literal["cuda_malloc_async"] | None:
    if value in (None, ""):
        return None
    if value == "cuda_malloc_async":
        return "cuda_malloc_async"
    raise ValueError(
        f"{source}={value!r} is not valid. Accepted: {_VALID_TF_GPU_ALLOCATORS}"
    )


def _validate_backend(value: str, *, source: str) -> str:
    if value not in _VALID_BACKENDS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_BACKENDS}"
        )
    return value


def _validate_platform(value: str, *, source: str) -> str:
    value = value.lower()
    if value not in _VALID_PLATFORMS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_PLATFORMS}"
        )
    return value


def _validate_mode(mode: str) -> BackendMode:
    if mode not in VALID_BACKEND_MODES:
        raise ValueError(
            f"Backend mode {mode!r} is not valid. Accepted: {VALID_BACKEND_MODES}"
        )
    return cast(BackendMode, mode)


@dataclass(frozen=True)
class JaxExecutionProfile:
    """Resolved JAX placement and numerical intent."""

    device: JaxDevice
    intent: ExecutionIntent
    mode: BackendMode


def resolve_jax_execution_profile(
    device: JaxDevice | str,
    intent: ExecutionIntent | str = "fast",
) -> JaxExecutionProfile:
    """Resolve the public orthogonal JAX selector to one canonical mode."""
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device={device!r} is not valid. Accepted: ('cpu', 'gpu')")
    if intent not in ("fast", "parity"):
        raise ValueError(
            f"intent={intent!r} is not valid. Accepted: ('fast', 'parity')"
        )
    resolved_device = cast(JaxDevice, device)
    resolved_intent = cast(ExecutionIntent, intent)
    return JaxExecutionProfile(
        device=resolved_device,
        intent=resolved_intent,
        mode=_JAX_EXECUTION_MODES[(resolved_device, resolved_intent)],
    )


def _validate_precision_selection(
    value: object,
    *,
    source: str,
) -> PrecisionSelection:
    if value not in _VALID_PRECISION_SELECTIONS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_PRECISION_SELECTIONS}"
        )
    return cast(PrecisionSelection, value)


def _validate_transfer_guard(value: str | None, *, source: str) -> str | None:
    if value in (None, ""):
        return None
    if value not in _VALID_TRANSFER_GUARDS:
        raise ValueError(
            f"{source}={value!r} is not valid. Accepted: {_VALID_TRANSFER_GUARDS}"
        )
    return value


def _default_compilation_cache_dir(mode: str) -> str | None:
    resolved_mode = _validate_mode(mode)
    backend, _platform = _MODE_TO_RUNTIME[resolved_mode]
    if backend != "jax":
        return None
    return str(Path.home() / ".cache" / "simsopt-jax-xla")


def _optional_env_value(name: str) -> str | None:
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return None
    return raw_value


def _optional_nonneg_int_env(name: str) -> int | None:
    raw_value = _optional_env_value(name)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name}={raw_value!r} must be >= 0")
    return value


def _optional_nonempty_env(name: str) -> str | None:
    raw_value = _optional_env_value(name)
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if stripped == "":
        return None
    return stripped


def _runtime_jax_platform_value(platform: str) -> str:
    # Single canonical helper for lowering ``BackendConfig.jax_platform`` to the
    # value JAX expects in ``JAX_PLATFORMS`` / ``jax.config["jax_platforms"]``.
    # Currently identity (cpu/cuda already match upstream casing); kept as
    # a single edit site so a future platform whose JAX name diverges from the
    # simsopt mode token can be remapped here without touching call sites.
    return platform


_CUDA_WITH_CPU_FALLBACK_PLATFORMS = "cuda,cpu"


def _runtime_jax_platforms_value(platform: str) -> str:
    if platform != "cuda":
        return _runtime_jax_platform_value(platform)
    requested_platforms = _optional_env_value(_JAX_PLATFORMS_ENV)
    requested_parts = (
        ()
        if requested_platforms is None
        else tuple(part.strip().lower() for part in requested_platforms.split(","))
    )
    if "cuda" in requested_parts and "cpu" in requested_parts:
        return _CUDA_WITH_CPU_FALLBACK_PLATFORMS
    return _runtime_jax_platform_value(platform)


def _runtime_jax_backend_name(platform: str) -> str:
    if platform == "cuda":
        return "gpu"
    return _runtime_jax_platform_value(platform)


def _primary_jax_platform(platforms: str | None) -> str | None:
    """First entry of a ``JAX_PLATFORMS``-style list, if it is a valid platform."""
    if platforms is None:
        return None
    parts = tuple(part.strip().lower() for part in platforms.split(",") if part.strip())
    if not parts:
        return None
    primary_platform = parts[0]
    return primary_platform if primary_platform in _VALID_PLATFORMS else None


def _resolve_kwarg(
    explicit: _ExplicitT | None,
    *,
    parse_explicit: Callable[[_ExplicitT], _ResolvedT],
    env_names: tuple[str, ...],
    parse_env: Callable[[str, str], _ResolvedT],
    read_default: Callable[[], _ResolvedT],
) -> _ResolvedT:
    if explicit is not None:
        return parse_explicit(explicit)
    for env_name in env_names:
        env_value = _optional_env_value(env_name)
        if env_value is not None:
            return parse_env(env_value, env_name)
    return read_default()


def _optional_bool_policy_default(value: object) -> bool | None:
    return None if value is None else bool(value)


def _debug_overlay_enabled() -> bool:
    return bool(_optional_bool_env(_DEBUG_ENV))


def _default_transfer_guard(mode: str) -> str | None:
    return _DEFAULT_TRANSFER_GUARD_BY_MODE[_validate_mode(mode)]


def _validate_mem_fraction_value(value: object, *, source: str) -> float:
    fraction = float(cast(float | str, value))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{source}={value!r} must be in (0, 1]")
    return fraction


def _config_from_mode(
    mode: str,
    *,
    strict: bool,
    precision: PrecisionSelection | None = None,
    debug_nans: bool | None = None,
    disable_jit: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
    xla_gpu_preallocate: bool | None = None,
    xla_gpu_mem_fraction: float | None = None,
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None,
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None,
) -> BackendConfig:
    mode = _validate_mode(mode)
    backend, jax_platform = _MODE_TO_RUNTIME[mode]
    debug_overlay = _debug_overlay_enabled()
    defaults = _get_mode_policy_defaults(mode)
    resolved_precision: PrecisionSelection = _resolve_kwarg(
        precision,
        parse_explicit=lambda value: _validate_precision_selection(
            value,
            source="precision",
        ),
        env_names=(_PRECISION_ENV,),
        parse_env=lambda value, source: _validate_precision_selection(
            value,
            source=source,
        ),
        read_default=lambda: "mode_default",
    )
    if debug_overlay:
        resolved_debug_nans = True
        resolved_disable_jit = True
        resolved_transfer_guard = "disallow"
    else:
        resolved_debug_nans = _resolve_kwarg(
            debug_nans,
            parse_explicit=bool,
            env_names=(_DEBUG_NANS_ENV,),
            parse_env=lambda value, source: value.strip().lower() in _TRUTHY_ENV_VALUES,
            read_default=lambda: False,
        )
        resolved_disable_jit = _resolve_kwarg(
            disable_jit,
            parse_explicit=bool,
            env_names=(_DISABLE_JIT_ENV,),
            parse_env=lambda value, source: _parse_bool_value(value, source=source),
            read_default=lambda: False,
        )
        resolved_transfer_guard = _resolve_kwarg(
            transfer_guard,
            parse_explicit=lambda value: _validate_transfer_guard(
                value,
                source="transfer_guard",
            ),
            env_names=(_TRANSFER_GUARD_ENV,),
            parse_env=lambda value, source: _validate_transfer_guard(
                value,
                source=source,
            ),
            read_default=lambda: _default_transfer_guard(mode),
        )
    resolved_compilation_cache_dir = _resolve_kwarg(
        compilation_cache_dir,
        parse_explicit=lambda value: value or None,
        env_names=(_COMPILATION_CACHE_DIR_ENV, _JAX_COMPILATION_CACHE_DIR_ENV),
        parse_env=lambda value, source: value,
        read_default=lambda: _default_compilation_cache_dir(mode),
    )
    resolved_xla_gpu_preallocate = _resolve_kwarg(
        xla_gpu_preallocate,
        parse_explicit=bool,
        env_names=(_GPU_PREALLOCATE_ENV,),
        parse_env=lambda value, source: _parse_bool_value(value, source=source),
        read_default=lambda: _optional_bool_policy_default(
            defaults["xla_gpu_preallocate"]
        ),
    )
    resolved_xla_gpu_mem_fraction = _resolve_kwarg(
        xla_gpu_mem_fraction,
        parse_explicit=lambda value: _validate_mem_fraction_value(
            value,
            source="xla_gpu_mem_fraction",
        ),
        env_names=(_GPU_MEM_FRACTION_ENV,),
        parse_env=lambda value, source: _validate_mem_fraction_value(
            value,
            source=source,
        ),
        read_default=lambda: _optional_float_policy_default(
            defaults["xla_gpu_mem_fraction"]
        ),
    )
    resolved_xla_gpu_allocator = _resolve_kwarg(
        xla_gpu_allocator,
        parse_explicit=lambda value: _validate_gpu_allocator(
            value,
            source="xla_gpu_allocator",
        ),
        env_names=(_GPU_ALLOCATOR_ENV,),
        parse_env=lambda value, source: _validate_gpu_allocator(
            value,
            source=source,
        ),
        read_default=lambda: _validate_gpu_allocator(
            defaults["xla_gpu_allocator"],
            source=f"{mode}.xla_gpu_allocator",
        ),
    )
    resolved_tf_gpu_allocator = _resolve_kwarg(
        tf_gpu_allocator,
        parse_explicit=lambda value: _validate_tf_gpu_allocator(
            value,
            source="tf_gpu_allocator",
        ),
        env_names=(_TF_GPU_ALLOCATOR_OVERRIDE_ENV,),
        parse_env=lambda value, source: _validate_tf_gpu_allocator(
            value,
            source=source,
        ),
        read_default=lambda: _validate_tf_gpu_allocator(
            defaults["tf_gpu_allocator"],
            source=f"{mode}.tf_gpu_allocator",
        ),
    )
    return BackendConfig(
        mode=mode,
        backend=backend,
        jax_platform=jax_platform,
        precision=resolved_precision,
        strict=bool(strict) or debug_overlay,
        debug_nans=resolved_debug_nans,
        disable_jit=resolved_disable_jit,
        transfer_guard=resolved_transfer_guard,
        compilation_cache_dir=resolved_compilation_cache_dir,
        xla_gpu_preallocate=resolved_xla_gpu_preallocate,
        xla_gpu_mem_fraction=resolved_xla_gpu_mem_fraction,
        xla_gpu_allocator=cast(Literal["platform", "vmm"] | None, resolved_xla_gpu_allocator),
        tf_gpu_allocator=cast(Literal["cuda_malloc_async"] | None, resolved_tf_gpu_allocator),
    )


def _get_mode_policy_defaults(mode: str) -> dict[str, object]:
    return _MODE_POLICY_DEFAULTS[_validate_mode(mode)]


def _validate_policy_dtype(value: object, *, mode: str, field: str) -> str:
    dtype_name = str(value)
    if dtype_name not in _VALID_POLICY_DTYPES:
        raise ValueError(
            f"Backend mode {mode!r} has unsupported {field}={dtype_name!r}. "
            f"Accepted: {_VALID_POLICY_DTYPES}."
        )
    return dtype_name


def _optional_float_policy_default(value: object) -> float | None:
    if value is None:
        return None
    return float(cast(float | str, value))


def _policy_from_config(config: BackendConfig) -> BackendPolicy:
    defaults = _get_mode_policy_defaults(config.mode)
    return BackendPolicy(
        mode=config.mode,
        backend=config.backend,
        jax_platform=config.jax_platform,
        parity_mode=bool(defaults["parity_mode"]),
        requires_x64=bool(defaults["requires_x64"]),
        runtime_dtype=_validate_policy_dtype(
            defaults["runtime_dtype"],
            mode=config.mode,
            field="runtime_dtype",
        ),
        host_dtype=_validate_policy_dtype(
            defaults["host_dtype"],
            mode=config.mode,
            field="host_dtype",
        ),
        compute_dtype="float64",
        chunk_policy=str(defaults["chunk_policy"]),
        matmul_precision=str(defaults["matmul_precision"]),
        transfer_guard=config.transfer_guard,
    )


def _runtime_env_value(attribute_name: str, value: object) -> str:
    if value is None:
        return ""
    if attribute_name in {
        "strict",
        "debug_nans",
        "disable_jit",
        "xla_gpu_preallocate",
    }:
        return "1" if bool(value) else "0"
    if attribute_name == "jax_platforms":
        return _runtime_jax_platforms_value(str(value))
    if attribute_name == "jax_platform":
        return _runtime_jax_platform_value(str(value))
    return str(value)


def _resolve_legacy_platform(backend: str) -> str:
    raw_value = os.environ.get(_PLATFORM_ENV)
    source = _PLATFORM_ENV

    if raw_value is None:
        raw_value = "cuda" if backend == "jax" else "cpu"
        source = "(default)"
    return _validate_platform(raw_value, source=source)


def _mode_from_legacy_env(backend: str, platform: str) -> BackendMode:
    if backend == "cpu":
        return "native_cpu"
    if platform == "cpu":
        return resolve_jax_execution_profile("cpu").mode
    return resolve_jax_execution_profile("gpu").mode
