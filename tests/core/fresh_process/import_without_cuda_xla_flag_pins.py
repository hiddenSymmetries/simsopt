"""Fresh-process child: import the shared JAX test support with a broken import.

``apply_cuda_xla_flag_pins`` is removed before ``unittest_jax_support``
imports it, so that import must raise ``ImportError`` even though the optional
JAX runtime is supported. The support module import deliberately follows the
removal at module level.
"""

import simsopt_jax.backend.runtime as runtime

del runtime.apply_cuda_xla_flag_pins

import unittest_jax_support  # noqa: F401
