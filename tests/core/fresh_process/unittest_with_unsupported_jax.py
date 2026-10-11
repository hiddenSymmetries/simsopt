"""Fresh-process child: run ``unittest`` with installed JAX and jaxlib reporting 0.9.0.

Usage: ``unittest_with_unsupported_jax.py SELECTOR... [-v]``, the arguments of
``python -m unittest``. Both versions are lowered before any test module is
loaded, so the optional JAX tests see an unsupported runtime, as on a host with
an old JAX installation.
"""

import unittest

import jax
import jaxlib.version

jax.__version__ = "0.9.0"
jaxlib.version.__version__ = "0.9.0"
unittest.main(module=None)
