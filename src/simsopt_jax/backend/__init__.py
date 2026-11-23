"""Backend selection: ``set_backend("native")`` or ``set_backend("jax", device="cpu" or "gpu")``."""

from .runtime import BackendConfig, get_backend_config, set_backend

__all__ = ("BackendConfig", "get_backend_config", "set_backend")
