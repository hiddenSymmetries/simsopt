"""Explicit process configuration for JAX field evaluation.

Call ``set_backend("jax", device="cpu" or "gpu", intent="parity" or "fast")``
before constructing ``JaxBiotSavart``. It resolves environment overrides and
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
Selecting CUDA also starts the CPU backend, so the small JAX kernels native
simsopt compiles with ``simsopt.geo.jit.jit`` run on the host; exporting
``JAX_PLATFORMS=cuda`` (without ``cpu``) starts CUDA alone, and they run on
the GPU.

Retained settings:

* ``SIMSOPT_JAX_DEBUG_NANS``, ``SIMSOPT_JAX_DISABLE_JIT`` and
  ``SIMSOPT_JAX_TRANSFER_GUARD`` control JAX diagnostics and transfers.
  Implicit transfers are allowed by default; set the guard to ``log`` or
  ``disallow`` to audit them.
  ``SIMSOPT_DEBUG`` enables all diagnostics, disables JIT and disallows transfers.
  Eager JAX indexing can stage scalar indices; explicitly use the boundary
  owner's ``allow_host_transfers`` context when evaluating in that debug mode.
* Every supported mode runs JAX in float64 (``jax_enable_x64``, which
  importing ``simsopt_jax`` enables and ``set_backend`` enables again) with
  matrix products at the highest precision.
* ``SIMSOPT_JAX_COIL_CHUNK_SIZE``, ``SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE`` and
  ``SIMSOPT_JAX_POINT_CHUNK_SIZE`` override the mode's field tiling. A disallow
  transfer guard selects dense audit kernels (zero chunk sizes).
* ``SIMSOPT_JAX_COMPILATION_CACHE_DIR`` enables persistent compilation caching;
  backend reconfiguration clears registered in-process kernel and placement caches.
* GPU allocation controls are ``SIMSOPT_JAX_GPU_PREALLOCATE``,
  ``SIMSOPT_JAX_GPU_MEM_FRACTION``, ``SIMSOPT_JAX_GPU_ALLOCATOR`` and
  ``SIMSOPT_TF_GPU_ALLOCATOR``. Standard XLA allocator variables none of them
  sets keep the caller's value.
* XLA reads ``JAX_PLATFORMS``, ``XLA_FLAGS`` and the allocator variables only
  when JAX initializes its backends. They are exported only before that;
  afterwards they cannot change this process, so they are left alone rather
  than handed to child processes with settings the parent never ran with, and
  a requested GPU setting that cannot apply warns.
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
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Literal

import jax
import jax.extend.backend

from simsopt_jax._requirements import enable_float64

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
    "apply_cuda_xla_flag_pins",
    "apply_jax_runtime_config",
    "get_backend_config",
    "get_backend_mode",
    "get_backend_policy",
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


def _token_sets_flag(token: str, flag_name: str) -> bool:
    return token == flag_name or token.startswith(f"{flag_name}=")


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
            _token_sets_flag(token, flag_name)
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
        _token_sets_flag(existing, flag_name)
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
    """Return the numerical-policy contract for a backend mode.

    Args:
        mode (str or None): Canonical backend mode; None selects the active
            configuration.

    Returns:
        BackendPolicy object: Resolved immutable numerical and execution
            policy.
    """
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

    Returns:
        BackendConfig object: Cached configuration resolved from the process
            environment or set_backend.
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
    """Return the resolved backend mode.

    Returns:
        str: Canonical mode of the active configuration.
    """
    return get_backend_config().mode


_cached_field_kernel_tuning: FieldKernelTuning | None = None


def get_field_kernel_tuning(mode: str | None = None) -> FieldKernelTuning:
    """Return the field-kernel chunk sizes for the resolved mode.

    Args:
        mode (str or None): Canonical backend mode; None selects the active
            configuration.

    Returns:
        FieldKernelTuning object: Resolved tile sizes after environment and
            transfer-guard overrides.
    """
    global _cached_field_kernel_tuning
    with _backend_runtime_lock:
        if mode is None and _cached_field_kernel_tuning is not None:
            return _cached_field_kernel_tuning
        resolved_mode = _resolve_mode(mode)
        tuning = _build_field_kernel_tuning(get_backend_policy(resolved_mode))
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

    Args:
        mode (str or None): Canonical backend mode; None selects the active
            configuration.

    Returns:
        jax.Device or None: First local device of the resolved platform, or
            None if native policy has no configured JAX platform.
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
    """Register a callback that should run whenever backend caches are cleared.

    Args:
        callback (Callable[[], None]): Cache-clearing function, keyed by module and
            qualified name; registration replaces a callback with the same key.
    """
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

    Derived caches include those cleared by callbacks registered with
    ``register_backend_cache_clear()``. Call this after mutating ``SIMSOPT_*``
    environment variables directly (outside of ``set_backend()``) so the next
    ``get_backend_config()`` call re-reads the environment.  Test fixtures
    should call this when they manipulate env vars via ``monkeypatch`` or
    context managers.
    """
    global _cached_backend_config
    with _backend_runtime_lock:
        _cached_backend_config = None
        _reset_backend_runtime_caches()


def _warn_at_caller(message: str) -> None:
    """Issue a RuntimeWarning located at the first caller outside this module.

    The public entries (``set_backend``, ``apply_jax_runtime_config``, ...)
    reach the warning sites at different depths, so a fixed ``stacklevel``
    would name a line of this module for some of them.

    Args:
        message (str): Warning text.
    """
    stacklevel = next(
        level
        for level, (frame, _) in enumerate(traceback.walk_stack(sys._getframe(1)), start=2)
        if frame.f_code.co_filename != __file__
    )
    warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)


def _raise_or_warn_runtime_issue(config: BackendConfig, message: str) -> None:
    if config.mode == "jax_gpu_parity" or config.strict:
        raise RuntimeError(message)
    _warn_at_caller(message)


def _expected_runtime_backend_names(jax_platform: str) -> frozenset[str]:
    if jax_platform == "cuda":
        return frozenset({"cuda", "gpu"})
    return frozenset({jax_platform})


def _device_platform(device) -> str:
    """Platform of a ``jax_default_device`` value: a Device or a platform name."""
    return device if isinstance(device, str) else device.platform


def _initialized_runtime_platform(
    jax_module, config: BackendConfig, expected_backends: frozenset[str]
) -> str | None:
    """The first initialized backend in ``expected_backends``, changing no setting.

    Listing the platforms initializes JAX's backends if nothing did yet. A
    platform JAX never initialized cannot be selected: that warns and returns
    None, or raises in strict and GPU parity modes.

    Raises:
        RuntimeError: The platform was never initialized, in strict or GPU
            parity mode.
    """
    initialized_platforms = tuple(jax_module.extend.backend.backends())
    platform = next(
        (name for name in initialized_platforms if name in expected_backends), None
    )
    if platform is None:
        message = (
            f"Requested JAX platform {config.jax_platform!r} for backend mode "
            f"{config.mode!r}, but JAX initialized only "
            f"{sorted(initialized_platforms)}. Set backend environment variables "
            "before importing or touching JAX devices."
        )
        _raise_or_warn_runtime_issue(config, message)
    return platform


def _default_device(jax_module, expected_backends: frozenset[str], platform: str):
    """The ``jax_default_device`` value that makes ``platform`` JAX's default.

    ``jax_platforms`` only decides which platforms JAX initializes, so with
    several initialized the requested one becomes the default through
    ``jax_default_device``: a selection that already runs it is kept, a stale
    one is cleared (None) when JAX's own default backend matches, and
    otherwise ``platform``'s first device is selected. Changing no setting, it
    runs among the checks of :func:`apply_jax_runtime_config`, since looking up
    JAX's default backend raises when a deprecated selector
    (``jax_xla_backend``) names a platform JAX never initialized.

    Raises:
        RuntimeError: JAX cannot resolve its default backend.
    """
    selected_device = jax_module.config.jax_default_device
    default_backend = str(jax_module.default_backend())
    effective_backend = (
        default_backend
        if selected_device is None
        else _device_platform(selected_device)
    )
    if effective_backend in expected_backends:
        return selected_device
    if default_backend in expected_backends:
        return None
    return jax_module.devices(platform)[0]


def _validate_cuda_parity_determinism_env(config: BackendConfig) -> None:
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
    """The standard JAX/XLA allocator variables ``config`` resolves.

    Only settings ``config`` resolves (not None) appear, so a caller's own
    ``XLA_PYTHON_CLIENT_*`` value survives when no ``SIMSOPT_*`` override or
    keyword sets it. A resolved memory fraction goes to
    ``XLA_CLIENT_MEM_FRACTION``, which jaxlib reads for every allocator, and
    clears its deprecated alias ``XLA_PYTHON_CLIENT_MEM_FRACTION`` (value
    None), since jaxlib rejects both set at once.
    """
    env: list[tuple[str, str | None]] = []
    if config.xla_gpu_preallocate is not None:
        env.append(
            (
                _XLA_PYTHON_CLIENT_PREALLOCATE_ENV,
                "true" if config.xla_gpu_preallocate else "false",
            )
        )
    if config.xla_gpu_allocator is not None:
        env.append((_XLA_PYTHON_CLIENT_ALLOCATOR_ENV, config.xla_gpu_allocator))
    if config.xla_gpu_mem_fraction is not None:
        env.append((_XLA_CLIENT_MEM_FRACTION_ENV, str(config.xla_gpu_mem_fraction)))
        env.append((_XLA_PYTHON_CLIENT_MEM_FRACTION_ENV, None))
    if config.tf_gpu_allocator is not None:
        env.append((_TF_GPU_ALLOCATOR_ENV, config.tf_gpu_allocator))
    return tuple(env)


def _export_before_backend_init(
    updates: tuple[tuple[str, str | None], ...],
    warning: Callable[[list[str]], str] | None = None,
) -> bool:
    """Export ``(name, value)`` variable updates while XLA can still read them.

    Owns the rule in the module docstring: once JAX initialized its backends
    nothing is exported, and ``warning(names)``, if given, warns about the
    variables whose value differs. Returns whether the updates were exported.
    """
    if not _jax_backends_initialized():
        for name, value in updates:
            _set_runtime_env(name, value)
        return True
    differing = [name for name, value in updates if os.environ.get(name) != value]
    if differing and warning is not None:
        _warn_at_caller(warning(differing))
    return False


def _apply_jax_gpu_memory_env(config: BackendConfig) -> None:
    """Export ``config``'s GPU allocator settings; once JAX initialized, a differing one warns.

    Child processes still resolve the synchronized ``SIMSOPT_*`` variables.
    """
    if config.jax_platform != "cuda":
        return
    _export_before_backend_init(
        _gpu_memory_runtime_env(config),
        lambda differing: (
            "JAX already initialized its backends, so the GPU memory settings "
            f"of backend mode {config.mode!r} ({', '.join(differing)}) do not "
            "apply to this process; set them before touching JAX devices."
        ),
    )


def _apply_cpu_compile_preset_env(config: BackendConfig, policy: BackendPolicy) -> None:
    """Add the FAST_COMPILE CPU preset to ``XLA_FLAGS`` for non-parity CPU lanes.

    ``xla_cpu_opt_preset`` is inert on the CUDA backend (whose XLA flags carry
    the determinism contract), and the preset reduces XLA optimization passes,
    which can shift CPU reduction order, so the bit-exact ``*_parity`` lanes go
    without it. It is a compile-speed default, not a request, so it never warns.
    """
    if config.jax_platform == "cuda" or policy.parity_mode:
        return
    _export_before_backend_init(
        ((_XLA_FLAGS_ENV, _xla_flags_with_cpu_compile_preset(os.environ.get(_XLA_FLAGS_ENV))),)
    )


def _jax_backends_initialized() -> bool:
    # Inspect the already-loaded bridge without initializing any devices.
    xla_bridge = sys.modules.get("jax._src.xla_bridge")
    if xla_bridge is None:
        return False
    return bool(xla_bridge.backends_are_initialized())


# ``set_backend`` exports ``JAX_PLATFORMS`` so child processes start the same
# platforms. The caller's request is the variable's value unless it still holds
# that export, so a later call never reads its own export as the caller's choice.
_requested_jax_platforms_value: str | None = None
_exported_jax_platforms: str | None = None


def _requested_jax_platforms() -> str | None:
    """``JAX_PLATFORMS`` as the caller set it, not as ``set_backend`` exported it."""
    global _requested_jax_platforms_value
    with _backend_runtime_lock:
        current = _optional_env_value(_JAX_PLATFORMS_ENV)
        if _exported_jax_platforms is None or current != _exported_jax_platforms:
            _requested_jax_platforms_value = current
        return _requested_jax_platforms_value


def _startup_jax_platforms(config: BackendConfig) -> str:
    """The platforms JAX should initialize for ``config``."""
    return _runtime_jax_platforms_value(config.jax_platform, _requested_jax_platforms())


def _export_startup_jax_platforms(config: BackendConfig) -> None:
    """Export ``config``'s startup platforms to ``JAX_PLATFORMS`` for child processes.

    After JAX initialized, narrowing it would also misstate the platforms JAX
    runs to ``native_jax_device()``.
    """
    global _exported_jax_platforms
    with _backend_runtime_lock:
        exported = _startup_jax_platforms(config)
        if _export_before_backend_init(((_JAX_PLATFORMS_ENV, exported),)):
            _exported_jax_platforms = exported


def _xla_flags_with_cuda_autotuner_pins(xla_flags: str | None) -> str:
    """Return ``xla_flags`` with both CUDA autotuner pins appended.

    Same composition contract as :func:`_xla_flags_with_cpu_compile_preset`:
    caller-provided autotuner settings are respected, and re-applying is a no-op.
    """
    return _xla_flags_with_gpu_autotune_level_pinned(
        _xla_flags_with_gpu_fusion_autotuner_disabled(xla_flags)
    )


def apply_cuda_xla_flag_pins() -> str:
    """Put both CUDA autotuner pins into ``XLA_FLAGS`` and return its value.

    Inert on the CPU backend, so a host that probes devices before it installs
    a backend config (a test session, a notebook) can call this at import time.
    Unlike :func:`_apply_cuda_autotuner_env`, it exports the pins also after
    JAX initialized its backends, because the caller asked for them.

    Returns:
        str: Updated XLA_FLAGS, preserving explicitly supplied autotuner
            settings.
    """
    pinned = _xla_flags_with_cuda_autotuner_pins(os.environ.get(_XLA_FLAGS_ENV))
    _set_runtime_env(_XLA_FLAGS_ENV, pinned)
    return pinned


def _apply_cuda_autotuner_env(config: BackendConfig) -> None:
    """Add reproducible CUDA compilation defaults to ``XLA_FLAGS``.

    Disabling experimental fusion autotuning avoids data-dependent candidate
    kernels during compilation. Autotune level zero fixes library algorithm
    selection across fresh compiles, which keeps chunked Biot-Savart reductions
    reproducible. Explicit caller flags take precedence; both defaults are
    inert on CPU. Once JAX initialized, a missing pin warns, since this
    process's fresh compiles are not bitwise-reproducible.

    Args:
        config (BackendConfig): Configuration being installed.
    """
    if config.jax_platform != "cuda":
        return
    xla_flags = os.environ.get(_XLA_FLAGS_ENV)
    pinned = _xla_flags_with_cuda_autotuner_pins(xla_flags)
    if pinned == xla_flags:
        return
    _export_before_backend_init(
        ((_XLA_FLAGS_ENV, pinned),),
        lambda _differing: (
            "XLA already initialized its backends without the CUDA autotuner pins in "
            f"{_XLA_FLAGS_ENV}; this process compiles with XLA's default GPU "
            "autotuning, so fresh compiles are not bitwise-reproducible, and "
            f"{_XLA_FLAGS_ENV} is left unchanged. Resolve the simsopt backend config "
            "(or call simsopt_jax.backend.runtime.apply_cuda_xla_flag_pins()) before "
            "touching JAX devices."
        ),
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
    """Apply the resolved JAX runtime settings to the active process.

    Every check that can reject the configuration (the CUDA determinism flags,
    the platform, then the default-backend lookup) runs before a JAX setting
    that a JAX context can hide changes. A rejected call therefore changes no
    such setting, also inside a context such as ``jax.transfer_guard(...)``.
    The exceptions are :data:`_PRE_CHECK_JAX_SETTINGS`, which the checks need
    first, and which ``set_backend`` restores when the call is rejected. Does
    nothing when the resolved backend is native.

    Raises:
        RuntimeError: GPU parity without the determinism ``XLA_FLAGS``, a
            platform JAX never initialized in strict or GPU parity mode, or a
            default backend JAX cannot resolve.
    """
    config = get_backend_config()
    if config.backend != "jax":
        return
    policy = get_backend_policy(config.mode)
    _validate_cuda_parity_determinism_env(config)
    _apply_jax_gpu_memory_env(config)
    _apply_cpu_compile_preset_env(config, policy)
    # The CUDA autotuner pins are applied where the configuration is installed
    # (``get_backend_config`` and ``set_backend``), once per installation.

    if not _jax_backends_initialized():
        jax.config.update("jax_platforms", _startup_jax_platforms(config))
    expected_backends = _expected_runtime_backend_names(config.jax_platform)
    platform = _initialized_runtime_platform(jax, config, expected_backends)
    # JAX resolves the default backend from the deprecated ``jax_platform_name``
    # before ``jax_platforms``. ``simsopt.geo`` sets it to ``"cpu"`` when no JAX
    # platform environment variable is set, which would keep the selected
    # platform from becoming the default (or fail when CPU is not listed).
    # Clearing it makes the selected platform, listed first, the default.
    jax.config.update("jax_platform_name", "")
    default_device = (
        jax.config.jax_default_device
        if platform is None
        else _default_device(jax, expected_backends, platform)
    )
    enable_float64()
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_debug_nans", config.debug_nans)
    jax.config.update("jax_disable_jit", config.disable_jit)
    if config.transfer_guard is not None:
        jax.config.update("jax_transfer_guard", config.transfer_guard)
    _apply_compilation_cache_config(jax, config)
    if default_device is not jax.config.jax_default_device:
        jax.config.update("jax_default_device", default_device)


# JAX settings ``apply_jax_runtime_config`` changes before its last check:
# ``jax_platforms`` (before backends initialize) decides which platforms the
# platform check initializes, and ``jax_platform_name`` must be cleared before
# the default-backend lookup. A rejected ``set_backend`` restores them.
_PRE_CHECK_JAX_SETTINGS = ("jax_platforms", "jax_platform_name")


# Process state a ``set_backend`` call changes, restored when the call is rejected.
_SET_BACKEND_ENV_NAMES = (
    *(env_name for env_name, _ in _SYNCED_RUNTIME_ENV_VALUES),
    _JAX_PLATFORMS_ENV,
    _XLA_FLAGS_ENV,
    _XLA_PYTHON_CLIENT_PREALLOCATE_ENV,
    _XLA_PYTHON_CLIENT_ALLOCATOR_ENV,
    _XLA_PYTHON_CLIENT_MEM_FRACTION_ENV,
    _XLA_CLIENT_MEM_FRACTION_ENV,
    _TF_GPU_ALLOCATOR_ENV,
)


@contextmanager
def _restore_backend_state_on_error() -> Iterator[None]:
    """Hold the runtime lock and undo every change if the body raises.

    Restores the installed configuration, the environment variables
    ``set_backend`` synchronizes or exports, and :data:`_PRE_CHECK_JAX_SETTINGS`,
    so a rejected call leaves the process in the mode it had before. No other
    JAX setting needs restoring: :func:`apply_jax_runtime_config` runs every
    check that can reject before it changes one.
    """
    global _cached_backend_config
    global _requested_jax_platforms_value, _exported_jax_platforms
    with _backend_runtime_lock:
        config = _cached_backend_config
        platforms_state = (_requested_jax_platforms_value, _exported_jax_platforms)
        env = tuple((name, os.environ.get(name)) for name in _SET_BACKEND_ENV_NAMES)
        jax_settings = tuple(
            (name, jax.config.values[name]) for name in _PRE_CHECK_JAX_SETTINGS
        )
        try:
            yield
        except BaseException:
            _cached_backend_config = config
            _requested_jax_platforms_value, _exported_jax_platforms = platforms_state
            _reset_backend_runtime_caches()
            for name, value in env:
                _set_runtime_env(name, value)
            for name, value in jax_settings:
                if jax.config.values[name] != value:
                    jax.config.update(name, value)
            raise


def set_backend(
    mode: BackendMode | Literal["jax"],
    *,
    device: JaxDevice | None = None,
    intent: ExecutionIntent | None = None,
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
    helpers continue to work unchanged. GPU memory keywords resolve the JAX/XLA
    allocator env vars explicitly; env overrides still sit between mode defaults
    and these arguments. These and the CUDA autotuner pins it adds to
    ``XLA_FLAGS`` apply only before JAX initializes its backends; afterwards
    they are not exported, and a GPU setting that cannot apply warns. Also
    updates the config cache so subsequent ``get_backend_config()`` calls are free.
    A call that raises leaves the previous configuration, environment and JAX
    settings in place; JAX backends it initialized (checking the requested platform
    does) stay initialized, so later GPU allocator settings no longer apply.

    Args:
        mode (str): A canonical backend mode, or jax with an explicit device selector.
        device (str or None): Required cpu or gpu selector when mode is jax; omit with a
            canonical mode. Before JAX initializes its backends, it limits JAX to
            that platform, keeping the CPU next to CUDA unless ``JAX_PLATFORMS``
            lists cuda without cpu. In a process that already initialized several
            platforms (for example ``JAX_PLATFORMS=cuda,cpu``), it leaves them
            running and makes the platform's first device JAX's default device; a
            platform JAX never initialized warns.
        intent (str or None): fast or parity when mode is jax; None selects fast. Omit
            with a canonical mode.
        strict (bool): Raise instead of warning when JAX never initialized the
            requested platform.
        debug_nans (bool or None): Enable JAX NaN debugging; None resolves the
            environment, then the mode default. SIMSOPT_DEBUG forces True.
        disable_jit (bool or None): Disable JAX compilation; None resolves the
            environment, then the mode default. SIMSOPT_DEBUG forces True.
        transfer_guard (str or None): allow, log or disallow implicit transfers; None
            resolves the environment, then the mode default. SIMSOPT_DEBUG forces
            disallow.
        compilation_cache_dir (str or None): Persistent compilation-cache directory;
            None resolves the environment, then the mode default.
        xla_gpu_preallocate (bool or None): GPU allocator preallocation; None resolves
            the environment, then the mode default.
        xla_gpu_mem_fraction (float or None): GPU memory fraction in (0, 1]; None
            resolves the environment, then the mode default.
        xla_gpu_allocator (str or None): platform or vmm GPU allocator; None resolves
            the environment, then the mode default.
        tf_gpu_allocator (str or None): cuda_malloc_async allocator selector; None
            resolves the environment, then the mode default.
        configure_runtime (bool): Apply JAX settings immediately; False installs and
            synchronizes the configuration without applying the full runtime settings.

    Returns:
        BackendConfig object: Installed immutable configuration after
            resolving overrides.

    Raises:
        RuntimeError: GPU parity without the determinism ``XLA_FLAGS``, or a
            platform JAX never initialized in strict or GPU parity mode.
        ValueError: An invalid mode, selector or override.
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
        debug_nans=debug_nans,
        disable_jit=disable_jit,
        transfer_guard=transfer_guard,
        compilation_cache_dir=compilation_cache_dir,
        xla_gpu_preallocate=xla_gpu_preallocate,
        xla_gpu_mem_fraction=xla_gpu_mem_fraction,
        xla_gpu_allocator=xla_gpu_allocator,
        tf_gpu_allocator=tf_gpu_allocator,
    )
    with _restore_backend_state_on_error():
        _cached_backend_config = config
        _reset_backend_runtime_caches()
        for env_name, attribute_name in _SYNCED_RUNTIME_ENV_VALUES:
            os.environ[env_name] = _runtime_env_value(getattr(config, attribute_name))
        _export_startup_jax_platforms(config)
        _apply_cuda_autotuner_env(config)
        if configure_runtime:
            apply_jax_runtime_config()
    return config
