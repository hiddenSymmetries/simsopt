"""Optional JAX tests remain loadable by every unittest command-line selector."""

from pathlib import Path
import os
import subprocess
import sys
from unittest import TestCase, skipIf

from unittest_jax_support import JAX_IMPORT_ERROR

try:
    import jax
except ImportError:
    jax = None


_TESTS_ROOT = Path(__file__).resolve().parents[1]
_UNSUPPORTED_JAX_RUNNER = """
import jax
import jaxlib
import unittest

jax.__version__ = "0.9.0"
jaxlib.__version__ = "0.9.0"
unittest.main(module=None)
"""
_MISSING_JAX_RUNNER = """
import sys
import unittest

sys.modules["jax"] = None
unittest.main(module=None)
"""
_MODULE = "core.test_state_tokens"
_CLASS = f"{_MODULE}.TestStateTokens"
_METHOD = f"{_CLASS}.test_state_token_factories_are_independent_monotonic_sequences"


@skipIf(jax is None, "JAX is unavailable for the old-runtime simulation")
class TestUnittestJaxSupport(TestCase):
    def test_unavailable_jax_skips_for_every_loader(self):
        """Old or missing JAX skips for discovery and named loader targets."""
        selectors = (
            ("discover", "-t", ".", "-s", "core", "-p", "test_state_tokens.py"),
            (_MODULE,),
            ("core/test_state_tokens.py",),
            (_CLASS,),
            (_METHOD,),
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(_TESTS_ROOT), environment.get("PYTHONPATH", ""))
        )
        cases = (
            (runner, reason, selector)
            for runner, reason in (
                (_UNSUPPORTED_JAX_RUNNER, "simsopt_jax requires"),
                (_MISSING_JAX_RUNNER, "import of jax halted"),
            )
            for selector in selectors
        )
        for runner, reason, selector in cases:
            with self.subTest(reason=reason, selector=selector):
                completed = subprocess.run(
                    (sys.executable, "-c", runner, *selector, "-v"),
                    cwd=_TESTS_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("Ran 1 test", completed.stderr)
                self.assertIn("OK (skipped=1)", completed.stderr)
                self.assertIn(reason, completed.stderr)

    @skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
    def test_supported_jax_does_not_hide_test_import_errors(self):
        """A broken test import still fails when the optional runtime is supported."""
        broken_imports = (
            (
                "make_state_token_factory",
                """
import simsopt_jax.core.state_tokens as state_tokens

del state_tokens.make_state_token_factory
import core.test_state_tokens
""",
            ),
            (
                "apply_cuda_xla_flag_pins",
                """
import simsopt_jax.backend.runtime as runtime

del runtime.apply_cuda_xla_flag_pins
import unittest_jax_support
""",
            ),
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(_TESTS_ROOT), environment.get("PYTHONPATH", ""))
        )
        for symbol, code in broken_imports:
            with self.subTest(symbol=symbol):
                completed = subprocess.run(
                    (sys.executable, "-c", code),
                    cwd=_TESTS_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ImportError", completed.stderr)
                self.assertIn(symbol, completed.stderr)
