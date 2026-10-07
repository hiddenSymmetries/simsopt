"""Explicit process configuration for JAX field evaluation.

Call ``set_backend("jax", device="cpu" or "gpu", intent="parity" or "fast")``
before constructing ``BiotSavartJAX``. It resolves environment overrides and
applies them to JAX. Importing the adapter does not initialize this policy.
The equivalent canonical float64 modes remain available.

GPU parity requires a CUDA-enabled JAX installation and
``--xla_gpu_exclude_nondeterministic_ops=true`` in ``XLA_FLAGS`` before JAX
backend initialization. Preserve other launch flags when adding it::

    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_exclude_nondeterministic_ops=true"

Then call ``set_backend("jax", device="gpu", intent="parity")`` before
constructing the adapter or touching JAX devices. It makes CUDA the default JAX
backend, also when native geometry (``simsopt.geo``) was imported first. The
stage-II example accepts ``--device gpu`` with this launch environment.
Exporting ``JAX_PLATFORMS=cuda,cpu`` as well keeps a CPU device next to CUDA,
so the native C++ curve length kernels run on the host; with CUDA alone they
run on the GPU through the same explicit transfers.

Retained settings:

* ``SIMSOPT_JAX_DEBUG_NANS``, ``SIMSOPT_JAX_DISABLE_JIT`` and
  ``SIMSOPT_JAX_TRANSFER_GUARD`` control JAX diagnostics and transfers.
  Implicit transfers are allowed by default; set the guard to ``log`` or
  ``disallow`` to audit them.
  ``SIMSOPT_DEBUG`` enables all diagnostics, disables JIT and disallows transfers.
  Eager JAX indexing can stage scalar indices; explicitly use the boundary
  owner's ``allow_host_transfers`` context when evaluating in that debug mode.
* Runtime, compute and host precision is float64 for every supported mode.
* ``SIMSOPT_JAX_COIL_CHUNK_SIZE``, ``SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE`` and
  ``SIMSOPT_JAX_POINT_CHUNK_SIZE`` override the mode's field tiling. A disallow
  transfer guard selects dense audit kernels (zero chunk sizes).
* ``SIMSOPT_JAX_COMPILATION_CACHE_DIR`` enables persistent compilation caching;
  backend reconfiguration clears registered in-process kernel and placement caches.
* GPU allocation controls are ``SIMSOPT_JAX_GPU_PREALLOCATE``,
  ``SIMSOPT_JAX_GPU_MEM_FRACTION``, ``SIMSOPT_JAX_GPU_ALLOCATOR`` and
  ``SIMSOPT_TF_GPU_ALLOCATOR``. Set these before initializing a GPU backend.
* Field evaluation runs on the single runtime device (the first local device
  of the selected platform); it does not distribute work across devices.

Policy resolution and field-tiling builders live in ``_runtime_policy`` and
``_runtime_tuning``; this module owns process lifecycle and configuration.
"""

from __future__ import annotations

import os
import shlex
import sys
import threading
import warnings
from typing import Callable, Literal

import jax

# Public + private re-exports: callers and tests keep importing from this facade.
from simsopt_jax.backend._runtime_policy import (
    _BACKEND_ENV,
    _JAX_PLATFORMS_ENV,
    _MODE_ENV,
    _STRICT_ENV,
    _SYNCED_RUNTIME_ENV_VALUES,
    _TF_GPU_ALLOCATOR_ENV,
    _TRUTHY_ENV_VALUES,
    _XLA_CLIENT_MEM_FRACTION_ENV,
    _XLA_FLAGS_ENV,
    _XLA_PYTHON_CLIENT_ALLOCATOR_ENV,
    _XLA_PYTHON_CLIENT_MEM_FRACTION_ENV,
    _XLA_PYTHON_CLIENT_PREALLOCATE_ENV,
    VALID_BACKEND_MODES,
    BackendConfig,
    BackendMode,
    BackendPolicy,
    ExecutionIntent,
    JaxDevice,
    JaxExecutionProfile,
    PrecisionSelection,
    _config_from_mode,
    _env_bool,
    _mode_from_legacy_env,
    _optional_env_value,
    _policy_from_config,
    _primary_jax_platform,
    _resolve_legacy_platform,
    _runtime_env_value,
    _runtime_jax_backend_name,
    _runtime_jax_platforms_value,
    _validate_backend,
    _validate_mode,
    resolve_jax_execution_profile,
)
from simsopt_jax.backend._runtime_tuning import (
    FieldKernelTuning,
    _build_field_kernel_tuning,
)


__all__ = [
    "VALID_BACKEND_MODES",
    "BackendConfig",
    "BackendMode",
    "BackendPolicy",
    "ExecutionIntent",
    "FieldKernelTuning",
    "JaxDevice",
    "JaxExecutionProfile",
    "PrecisionSelection",
    "apply_cuda_xla_flag_pins",
    "apply_jax_runtime_config",
    "get_backend_config",
    "get_backend_mode",
    "get_backend_policy",
    "get_compute_dtype",
    "get_field_kernel_tuning",
    "get_runtime_jax_device",
    "invalidate_backend_cache",
    "register_backend_cache_clear",
    "resolve_jax_execution_profile",
    "set_backend",
]

_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_exclude_nondeterministic_ops",)
_STALE_GPU_DETERMINISM_XLA_FLAGS = ("--xla_gpu_deterministic_ops",)
_CPU_OPT_PRESET_FLAG_NAME = "--xla_cpu_opt_preset"
_CPU_OPT_PRESET_FAST_COMPILE = f"{_CPU_OPT_PRESET_FLAG_NAME}=FAST_COMPILE"
_GPU_FUSION_AUTOTUNER_FLAG_NAME = "--xla_gpu_experimental_enable_fusion_autotuner"
_GPU_FUSION_AUTOTUNER_DISABLED = f"{_GPU_FUSION_AUTOTUNER_FLAG_NAME}=false"
_GPU_AUTOTUNE_LEVEL_FLAG_NAME = "--xla_gpu_autotune_level"
_GPU_AUTOTUNE_LEVEL_PINNED = f"{_GPU_AUTOTUNE_LEVEL_FLAG_NAME}=0"


_BackendCacheClearCallbackKey = tuple[str, str]
_backend_runtime_lock = threading.RLock()
_backend_cache_clear_callbacks: dict[
    _BackendCacheClearCallbackKey, Callable[[], None]
] = {}


def _xla_flag_value(token: str, flag_name: str) -> bool | None:
    if token == flag_name:
        return True
    if not token.startswith(f"{flag_name}="):
        return None
    _, raw_value = token.split("=", 1)
    return raw_value.strip().lower() in _TRUTHY_ENV_VALUES


def _split_xla_flag_tokens(xla_flags: str | None) -> tuple[str, ...]:
    # External-input parse contract: tokenize XLA_FLAGS env value via shlex.
    # The narrow ValueError catch is a boundary parser (malformed user input
    # returns an empty tuple), not a runtime error swallow.
    if not xla_flags:
        return ()
    try:
        return tuple(shlex.split(xla_flags))
    except ValueError:
        return ()


def _xla_flags_enable_gpu_determinism(xla_flags: str | None) -> bool:
    effective_values: dict[str, bool] = {}
    for token in _split_xla_flag_tokens(xla_flags):
        for flag_name in _GPU_DETERMINISM_XLA_FLAGS:
            resolved = _xla_flag_value(token, flag_name)
            if resolved is None:
                continue
            effective_values[flag_name] = resolved
            break
    return any(effective_values.values())


def _xla_flags_include_stale_gpu_determinism(xla_flags: str | None) -> bool:
    for token in _split_xla_flag_tokens(xla_flags):
        if any(
            token == flag_name or token.startswith(f"{flag_name}=")
            for flag_name in _STALE_GPU_DETERMINISM_XLA_FLAGS
        ):
            return True
    return False


def _stale_cuda_determinism_message() -> str:
    stale_flags = " or ".join(_STALE_GPU_DETERMINISM_XLA_FLAGS)
    return (
        f"{_XLA_FLAGS_ENV} contains stale CUDA determinism flag {stale_flags}. "
        f"Use {_enabled_gpu_determinism_flags_text()} before initializing or "
        "touching JAX devices."
    )


def _enabled_gpu_determinism_flags_text() -> str:
    return " or ".join(f"{flag_name}=true" for flag_name in _GPU_DETERMINISM_XLA_FLAGS)


def _xla_flags_with_token(xla_flags: str | None, flag_name: str, token: str) -> str:
    """Return ``xla_flags`` with ``token`` appended unless ``flag_name`` is already set.

    Idempotent and non-destructive: existing tokens are preserved verbatim, and
    a caller-provided ``flag_name`` (any value) is respected rather than
    overridden. ``None``/empty input yields just ``token``.
    """
    stripped_xla_flags = "" if xla_flags is None else xla_flags.strip()
    if any(
        existing == flag_name or existing.startswith(f"{flag_name}=")
        for existing in _split_xla_flag_tokens(xla_flags)
    ):
        return xla_flags or ""
    if not stripped_xla_flags:
        return token
    return f"{stripped_xla_flags} {token}"


def _xla_flags_with_cpu_compile_preset(xla_flags: str | None) -> str:
    """Return ``xla_flags`` with the CPU FAST_COMPILE preset appended.

    Idempotent and non-destructive: existing tokens are preserved verbatim, and
    a caller-provided ``--xla_cpu_opt_preset`` (any value) is respected rather
    than overridden. ``None``/empty input yields just the preset token.
    """
    return _xla_flags_with_token(
        xla_flags, _CPU_OPT_PRESET_FLAG_NAME, _CPU_OPT_PRESET_FAST_COMPILE
    )


def _xla_flags_with_gpu_fusion_autotuner_disabled(xla_flags: str | None) -> str:
    """Return ``xla_flags`` with XLA's experimental GPU fusion autotuner disabled.

    Same composition contract as :func:`_xla_flags_with_cpu_compile_preset`: a
    caller-provided ``--xla_gpu_experimental_enable_fusion_autotuner`` (any
    value) is respected, and re-applying is a no-op.
    """
    return _xla_flags_with_token(
        xla_flags, _GPU_FUSION_AUTOTUNER_FLAG_NAME, _GPU_FUSION_AUTOTUNER_DISABLED
    )


def _xla_flags_with_gpu_autotune_level_pinned(xla_flags: str | None) -> str:
    """Return ``xla_flags`` with XLA's GPU GEMM/convolution autotuner at level 0.

    Same composition contract as :func:`_xla_flags_with_cpu_compile_preset`: a
    caller-provided ``--xla_gpu_autotune_level`` (any value) is respected, and
    re-applying is a no-op.
    """
    return _xla_flags_with_token(
        xla_flags, _GPU_AUTOTUNE_LEVEL_FLAG_NAME, _GPU_AUTOTUNE_LEVEL_PINNED
    )


def _resolve_mode(mode: str | None = None) -> str:
    if mode is None:
        return get_backend_mode()
    return _validate_mode(mode)


_cached_backend_policy: BackendPolicy | None = None


def get_backend_policy(mode: str | None = None) -> BackendPolicy:
    """Return the numerical-policy contract for a backend mode."""
    global _cached_backend_policy
    with _backend_runtime_lock:
        if mode is None:
            if _cached_backend_policy is not None:
                return _cached_backend_policy
            policy = _policy_from_config(get_backend_config())
            _cached_backend_policy = policy
            return policy
        resolved_mode = _resolve_mode(mode)
        current_config = get_backend_config()
        config = (
            current_config
            if current_config.mode == resolved_mode
            else _config_from_mode(resolved_mode, strict=False)
        )
        return _policy_from_config(config)


_cached_backend_config: BackendConfig | None = None


def get_backend_config() -> BackendConfig:
    """Return the resolved backend configuration.

    The result is cached after first resolution. Call
    ``invalidate_backend_cache()`` or ``set_backend()`` to clear.
    """
    global _cached_backend_config
    with _backend_runtime_lock:
        if _cached_backend_config is not None:
            return _cached_backend_config

        strict = _env_bool(_STRICT_ENV)
        mode = os.environ.get(_MODE_ENV)
        if mode is not None:
            config = _config_from_mode(mode, strict=strict)
        else:
            backend = _validate_backend(
                os.environ.get(_BACKEND_ENV, "cpu"), source=_BACKEND_ENV
            )
            platform = _resolve_legacy_platform(backend)
            config = _config_from_mode(
                _mode_from_legacy_env(backend, platform),
                strict=strict,
            )

        _cached_backend_config = config
        _apply_cuda_autotuner_env(config)
        return config


def get_backend_mode() -> str:
    """Return the resolved backend mode."""
    return get_backend_config().mode


def get_compute_dtype(mode: str | None = None) -> str:
    """Return the compute dtype name for a backend mode."""
    return get_backend_policy(mode).compute_dtype


_cached_field_kernel_tuning: FieldKernelTuning | None = None


def get_field_kernel_tuning(mode: str | None = None) -> FieldKernelTuning:
    """Return the field-kernel chunk sizes for the resolved mode."""
    global _cached_field_kernel_tuning
    with _backend_runtime_lock:
        if mode is None and _cached_field_kernel_tuning is not None:
            return _cached_field_kernel_tuning
        resolved_mode = _resolve_mode(mode)
        tuning = _build_field_kernel_tuning(
            resolved_mode,
            get_backend_policy(resolved_mode),
        )
        if mode is None:
            _cached_field_kernel_tuning = tuning
        return tuning


def get_runtime_jax_device(mode: str | None = None):
    """Return the first local JAX device for the active runtime policy.

    A JAX policy names its platform. Otherwise the device follows the platforms
    this process's JAX was configured with (``jax.config.jax_platforms``), not
    the current ``JAX_PLATFORMS`` environment: ``set_backend`` rewrites that
    variable for child processes, and an initialized JAX never reads it again.
    Before JAX has its configuration (not imported yet, or its first import
    still running in another thread) the variable is still what it will
    read, so a native process that never touched JAX does not import it here.
    """
    policy = get_backend_policy(mode)
    if policy.backend == "jax":
        platform = policy.jax_platform
    else:
        jax_config = getattr(sys.modules.get("jax"), "config", None)
        platform = _primary_jax_platform(
            _optional_env_value(_JAX_PLATFORMS_ENV)
            if jax_config is None
            else jax_config.jax_platforms
        )
    if platform is None:
        return None

    _apply_compilation_cache_config(jax, get_backend_config())
    backend_name = _runtime_jax_backend_name(platform)
    return jax.local_devices(backend=backend_name)[0]


def _backend_cache_clear_callback_key(
    callback: Callable[[], None],
) -> _BackendCacheClearCallbackKey:
    return (callback.__module__, callback.__qualname__)


def register_backend_cache_clear(callback: Callable[[], None]) -> None:
    """Register a callback that should run whenever backend caches are cleared."""
    with _backend_runtime_lock:
        _backend_cache_clear_callbacks[_backend_cache_clear_callback_key(callback)] = (
            callback
        )


def _run_backend_cache_clear_callbacks() -> None:
    with _backend_runtime_lock:
        callbacks = tuple(_backend_cache_clear_callbacks.values())
    for callback in callbacks:
        callback()


def _reset_backend_runtime_caches() -> None:
    global _cached_backend_policy, _cached_field_kernel_tuning
    global _compilation_cache_applied_dir
    with _backend_runtime_lock:
        _cached_backend_policy = None
        _compilation_cache_applied_dir = None
        _cached_field_kernel_tuning = None
        _run_backend_cache_clear_callbacks()


def invalidate_backend_cache() -> None:
    """Clear the cached backend configuration and derived caches.

    Call this after mutating ``SIMSOPT_*`` environment variables directly
    (outside of ``set_backend()``) so the next ``get_backend_config()`` call
    re-reads the environment.  Test fixtures should call this when they
    manipulate env vars via ``monkeypatch`` or context managers.
    """
    global _cached_backend_config
    with _backend_runtime_lock:
        _cached_backend_config = None
        _reset_backend_runtime_caches()


def _raise_or_warn_runtime_issue(config: BackendConfig, message: str) -> None:
    if config.mode == "jax_gpu_parity" or config.strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _expected_runtime_backend_names(jax_platform: str) -> frozenset[str]:
    if jax_platform == "cuda":
        return frozenset({"cuda", "gpu"})
    return frozenset({jax_platform})


def _validate_initialized_jax_runtime(jax_module, config: BackendConfig) -> None:
    default_backend = getattr(jax_module, "default_backend", None)
    if not callable(default_backend):
        return
    active_backend = str(default_backend())
    expected_backends = _expected_runtime_backend_names(config.jax_platform)
    if active_backend in expected_backends:
        return
    message = (
        f"Requested JAX platform {config.jax_platform!r} for backend mode "
        f"{config.mode!r}, but the active JAX default backend is "
        f"{active_backend!r}. Set backend environment variables before "
        "importing or touching JAX devices."
    )
    _raise_or_warn_runtime_issue(config, message)


def _validate_cuda_parity_determinism_env(
    config: BackendConfig,
    policy: BackendPolicy,
) -> None:
    if config.jax_platform != "cuda":
        return
    xla_flags = os.environ.get(_XLA_FLAGS_ENV)
    if _xla_flags_include_stale_gpu_determinism(xla_flags):
        message = _stale_cuda_determinism_message()
        _raise_or_warn_runtime_issue(config, message)
        return
    if _xla_flags_enable_gpu_determinism(xla_flags):
        return
    message = (
        f"Backend mode {config.mode!r} selects CUDA execution, but "
        f"{_XLA_FLAGS_ENV} does not enable "
        f"{_enabled_gpu_determinism_flags_text()}. Set {_XLA_FLAGS_ENV} before "
        "importing or touching JAX devices, because changing XLA flags after JAX "
        "backend initialization has no effect."
    )
    _raise_or_warn_runtime_issue(config, message)


def _set_runtime_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
        return
    os.environ[name] = value


def _gpu_memory_runtime_env(
    config: BackendConfig,
) -> tuple[tuple[str, str | None], ...]:
    if config.xla_gpu_preallocate is None:
        preallocate = None
    else:
        preallocate = "true" if config.xla_gpu_preallocate else "false"

    if config.xla_gpu_allocator == "vmm":
        python_mem_fraction = None
        client_mem_fraction = (
            None
            if config.xla_gpu_mem_fraction is None
            else str(config.xla_gpu_mem_fraction)
        )
    else:
        python_mem_fraction = (
            None
            if config.xla_gpu_mem_fraction is None
            else str(config.xla_gpu_mem_fraction)
        )
        client_mem_fraction = None

    return (
        (_XLA_PYTHON_CLIENT_PREALLOCATE_ENV, preallocate),
        (_XLA_PYTHON_CLIENT_ALLOCATOR_ENV, config.xla_gpu_allocator),
        (_XLA_PYTHON_CLIENT_MEM_FRACTION_ENV, python_mem_fraction),
        (_XLA_CLIENT_MEM_FRACTION_ENV, client_mem_fraction),
        (_TF_GPU_ALLOCATOR_ENV, config.tf_gpu_allocator),
    )


def _gpu_memory_runtime_env_matches(config: BackendConfig) -> bool:
    for name, expected in _gpu_memory_runtime_env(config):
        actual = os.environ.get(name)
        if expected is None:
            if actual is not None:
                return False
            continue
        if actual != expected:
            return False
    return True


def _assert_jax_not_initialized_for_gpu_memory_config(config: BackendConfig) -> None:
    if config.jax_platform != "cuda" or not _jax_backends_initialized():
        return
    if _gpu_memory_runtime_env_matches(config):
        return
    raise RuntimeError(
        "JAX GPU memory environment variables must be resolved before "
        "backend initialization. Call simsopt_jax.backend.set_backend(...) "
        "before touching JAX devices."
    )


def _apply_jax_gpu_memory_env(config: BackendConfig) -> None:
    if config.jax_platform != "cuda":
        return
    _assert_jax_not_initialized_for_gpu_memory_config(config)
    for name, value in _gpu_memory_runtime_env(config):
        _set_runtime_env(name, value)


def _apply_cpu_compile_preset_env(config: BackendConfig, policy: BackendPolicy) -> None:
    """Pull the FAST_COMPILE CPU preset into ``XLA_FLAGS`` before JAX inits.

    XLA reads ``XLA_FLAGS`` only at backend initialization, so this runs in the
    pre-backend-initialization region of :func:`apply_jax_runtime_config`. Applied to
    non-parity CPU lanes only: ``xla_cpu_opt_preset`` is inert on the CUDA
    backend (whose XLA flags carry the determinism contract), and the preset
    reduces XLA optimization passes -- which can shift CPU reduction order, so
    it is withheld from the bit-exact ``*_parity`` lanes.
    """
    if config.jax_platform == "cuda" or policy.parity_mode:
        return
    _set_runtime_env(
        _XLA_FLAGS_ENV,
        _xla_flags_with_cpu_compile_preset(os.environ.get(_XLA_FLAGS_ENV)),
    )


def _jax_backends_initialized() -> bool:
    # Inspect the already-loaded bridge without initializing any devices.
    xla_bridge = sys.modules.get("jax._src.xla_bridge")
    if xla_bridge is None:
        return False
    return bool(xla_bridge.backends_are_initialized())


def apply_cuda_xla_flag_pins() -> str:
    """Put both CUDA autotuner pins into ``XLA_FLAGS`` and return its value.

    Inert on the CPU backend, so a host that probes devices before it installs
    a backend config (a test session, a notebook) can call this at import time;
    the config-install sites call it through :func:`_apply_cuda_autotuner_env`.
    """
    pinned = _xla_flags_with_gpu_autotune_level_pinned(
        _xla_flags_with_gpu_fusion_autotuner_disabled(os.environ.get(_XLA_FLAGS_ENV))
    )
    _set_runtime_env(_XLA_FLAGS_ENV, pinned)
    return pinned


def _apply_cuda_autotuner_env(config: BackendConfig) -> None:
    """Apply reproducible CUDA compilation defaults before backend initialization.

    Disabling experimental fusion autotuning avoids data-dependent candidate
    kernels during compilation. Autotune level zero fixes library algorithm
    selection across fresh compiles, which keeps chunked Biot-Savart reductions
    reproducible. Explicit caller flags take precedence; both defaults are
    inert on CPU.
    """
    if config.jax_platform != "cuda":
        return
    previous = os.environ.get(_XLA_FLAGS_ENV)
    if apply_cuda_xla_flag_pins() != previous and _jax_backends_initialized():
        warnings.warn(
            "XLA already initialized its backends before the CUDA autotuner pins "
            f"reached {_XLA_FLAGS_ENV}; this process compiles with XLA's default "
            "GPU autotuning, so fresh compiles are not bitwise-reproducible. "
            "Resolve the simsopt backend config (or call "
            "simsopt_jax.backend.runtime.apply_cuda_xla_flag_pins()) before "
            "touching JAX devices.",
            RuntimeWarning,
            stacklevel=3,
        )


_compilation_cache_applied_dir: str | None = None


def _apply_compilation_cache_config(jax, config: BackendConfig) -> None:
    """Point JAX's persistent compilation cache at the resolved directory.

    Called from every runtime entry that already holds ``jax`` for a resolved
    config -- ``apply_jax_runtime_config`` and ``get_runtime_jax_device`` -- so a
    process configured through the environment alone (every example mirror,
    every driver child) reaches the cache before its first compile instead of
    compiling cold forever. Idempotent per resolved directory.
    """
    global _compilation_cache_applied_dir
    if config.backend != "jax" or config.compilation_cache_dir is None:
        return
    with _backend_runtime_lock:
        if _compilation_cache_applied_dir == config.compilation_cache_dir:
            return
        _compilation_cache_applied_dir = config.compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", config.compilation_cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # Follow JAX's documented GPU persistent-cache setting. Wider XLA cache
    # modes can force nvlink through container CUDA toolkits that differ
    # from the NVIDIA libraries bundled with the JAX wheel.
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )


def apply_jax_runtime_config() -> None:
    """Apply the resolved JAX runtime settings to the active process."""
    config = get_backend_config()
    if config.backend != "jax":
        return
    policy = get_backend_policy(config.mode)
    _validate_cuda_parity_determinism_env(config, policy)
    _apply_jax_gpu_memory_env(config)
    _apply_cpu_compile_preset_env(config, policy)
    _apply_cuda_autotuner_env(config)

    jax.config.update(
        "jax_platforms",
        _runtime_jax_platforms_value(config.jax_platform),
    )
    # JAX resolves the default backend from the deprecated ``jax_platform_name``
    # before ``jax_platforms``. ``simsopt.geo`` sets it to ``"cpu"`` when no JAX
    # platform environment variable is set, which would keep the selected
    # platform from becoming the default (or fail when CPU is not listed).
    # Clearing it makes the selected platform, listed first, the default.
    jax.config.update("jax_platform_name", "")
    jax.config.update("jax_enable_x64", policy.requires_x64)
    jax.config.update("jax_default_matmul_precision", policy.matmul_precision)
    jax.config.update("jax_debug_nans", config.debug_nans)
    jax.config.update("jax_disable_jit", config.disable_jit)
    if config.transfer_guard is not None:
        jax.config.update("jax_transfer_guard", config.transfer_guard)
    _apply_compilation_cache_config(jax, config)
    _validate_initialized_jax_runtime(jax, config)


def set_backend(
    mode: BackendMode | Literal["jax"],
    *,
    device: JaxDevice | None = None,
    intent: ExecutionIntent | None = None,
    precision: PrecisionSelection | None = None,
    strict: bool = False,
    debug_nans: bool | None = None,
    disable_jit: bool | None = None,
    transfer_guard: str | None = None,
    compilation_cache_dir: str | None = None,
    xla_gpu_preallocate: bool | None = None,
    xla_gpu_mem_fraction: float | None = None,
    xla_gpu_allocator: Literal["platform", "vmm"] | None = None,
    tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None,
    configure_runtime: bool = True,
) -> BackendConfig:
    """Set the active backend mode for the current process.

    This keeps the legacy env vars in sync so existing scripts and subprocess
    helpers continue to work unchanged. GPU memory keywords resolve the
    pre-backend-initialization JAX/XLA allocator env vars explicitly; env overrides still sit
    between mode defaults and these arguments. Also updates the config cache so
    subsequent ``get_backend_config()`` calls are free.
    """
    global _cached_backend_config
    if mode == "jax":
        if device is None:
            raise ValueError("set_backend('jax') requires device='cpu' or device='gpu'")
        resolved_mode = resolve_jax_execution_profile(
            device,
            "fast" if intent is None else intent,
        ).mode
    else:
        if device is not None or intent is not None:
            raise ValueError(
                "A canonical backend mode cannot be combined with device or intent"
            )
        resolved_mode = _validate_mode(mode)
    config = _config_from_mode(
        resolved_mode,
        strict=bool(strict),
        precision=precision,
        debug_nans=debug_nans,
        disable_jit=disable_jit,
        transfer_guard=transfer_guard,
        compilation_cache_dir=compilation_cache_dir,
        xla_gpu_preallocate=xla_gpu_preallocate,
        xla_gpu_mem_fraction=xla_gpu_mem_fraction,
        xla_gpu_allocator=xla_gpu_allocator,
        tf_gpu_allocator=tf_gpu_allocator,
    )
    with _backend_runtime_lock:
        _cached_backend_config = config
        _reset_backend_runtime_caches()
        for env_name, attribute_name in _SYNCED_RUNTIME_ENV_VALUES:
            config_attribute_name = (
                "jax_platform" if attribute_name == "jax_platforms" else attribute_name
            )
            os.environ[env_name] = _runtime_env_value(
                attribute_name,
                getattr(config, config_attribute_name),
            )
        _apply_cuda_autotuner_env(config)
    if configure_runtime:
        apply_jax_runtime_config()
    return config
