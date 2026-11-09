"""Fresh-process child: run ``unittest`` with one simsopt_jax submodule blocked.

Usage: ``unittest_with_broken_simsopt_jax.py SELECTOR... [-v]``, the arguments of
``python -m unittest``. A ``None`` entry in ``sys.modules`` makes importing
``simsopt_jax.core.curve_geometry`` raise ``ImportError`` after the JAX requirement
gate has passed, as a broken internal import would.
"""

import sys
import unittest

sys.modules["simsopt_jax.core.curve_geometry"] = None  # pyright: ignore[reportArgumentType]
unittest.main(module=None)
