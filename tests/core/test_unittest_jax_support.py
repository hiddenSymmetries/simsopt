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
_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")
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
                ("unittest_with_unsupported_jax.py", "simsopt_jax requires"),
                ("unittest_without_jax.py", "import of jax halted"),
            )
            for selector in selectors
        )
        for runner, reason, selector in cases:
            with self.subTest(reason=reason, selector=selector):
                completed = subprocess.run(
                    (sys.executable, str(_FRESH_PROCESS / runner), *selector, "-v"),
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
            ("make_state_token_factory", "import_without_state_token_factory.py"),
            ("apply_cuda_xla_flag_pins", "import_without_cuda_xla_flag_pins.py"),
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(_TESTS_ROOT), environment.get("PYTHONPATH", ""))
        )
        for symbol, script in broken_imports:
            with self.subTest(symbol=symbol):
                completed = subprocess.run(
                    (sys.executable, str(_FRESH_PROCESS / script)),
                    cwd=_TESTS_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ImportError", completed.stderr)
                self.assertIn(symbol, completed.stderr)
