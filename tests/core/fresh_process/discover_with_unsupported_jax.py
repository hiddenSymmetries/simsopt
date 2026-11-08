"""Fresh-process child: import every test module with JAX and jaxlib reporting 0.9.0.

Usage: ``discover_with_unsupported_jax.py`` from the tests root. ``unittest
discover`` imports each module it finds; no test case is built or run. Every
import error is printed to stderr and makes the exit status nonzero. With an
unsupported JAX the optional JAX tests must still import, so that discovery
skips them. The run also fails unless the version gate rejected the reported
JAX, so a gate that stopped reading these versions cannot pass unexercised.
"""

import sys
import unittest

import jax
import jaxlib.version


class ImportOnlyLoader(unittest.TestLoader):
    """Discovery that imports modules but builds no test cases (some do work in __init__)."""

    def loadTestsFromModule(self, module, *, pattern=None):
        return unittest.TestSuite()


jax.__version__ = "0.9.0"
jaxlib.version.__version__ = "0.9.0"
loader = ImportOnlyLoader()
loader.discover(start_dir=".", top_level_dir=".")
for error in loader.errors:
    print(error, file=sys.stderr)
gate_error = sys.modules["unittest_jax_support"].JAX_IMPORT_ERROR
if gate_error is None or "requires jaxlib >= " not in gate_error:
    print(f"the JAX version gate did not reject 0.9.0: {gate_error!r}", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if loader.errors else 0)
