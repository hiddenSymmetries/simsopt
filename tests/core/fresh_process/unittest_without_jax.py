"""Fresh-process child: run ``unittest`` with ``import jax`` blocked.

Usage: ``unittest_without_jax.py SELECTOR... [-v]``, the arguments of
``python -m unittest``. A ``None`` entry in ``sys.modules`` makes every later
``import jax`` raise ``ImportError: import of jax halted``.
"""

import sys
import unittest

sys.modules["jax"] = None  # pyright: ignore[reportArgumentType]
unittest.main(module=None)
