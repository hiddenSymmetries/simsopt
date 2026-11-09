"""Optional JAX tests remain loadable by every unittest command-line selector.

The optional-JAX gate skips only for an unsupported runtime, never for a broken package.
"""

from pathlib import Path
import os
from unittest import TestCase, skipIf

from unittest_jax_support import JAX_IMPORT_ERROR, optional_jax_imports, run_fresh_process

try:
    import jax
except ImportError:
    jax = None

with optional_jax_imports():
    from simsopt_jax._requirements import _require_jax_version


_TESTS_ROOT = Path(__file__).resolve().parents[1]
_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")
_MODULE = "core.test_state_tokens"
_CLASS = f"{_MODULE}.TestStateTokens"
_METHOD = f"{_CLASS}.test_state_token_factories_are_independent_monotonic_sequences"


def _run_child(script: str, *arguments: str):
    """Run ``fresh_process/<script>`` from the tests root, with it on ``PYTHONPATH``."""
    path = os.pathsep.join((str(_TESTS_ROOT), os.environ.get("PYTHONPATH", "")))
    environment = dict(os.environ, PYTHONPATH=path)
    return run_fresh_process(_FRESH_PROCESS / script, *arguments, env=environment, cwd=_TESTS_ROOT)


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
        # Runtime prerequisites can fail before simsopt_jax imports the blocked jax.
        cases = (
            (runner, reason, selector)
            for runner, reason in (
                ("unittest_with_unsupported_jax.py", "simsopt_jax requires"),
                ("unittest_without_jax.py", JAX_IMPORT_ERROR or "import of jax halted"),
            )
            for selector in selectors
        )
        for runner, reason, selector in cases:
            with self.subTest(reason=reason, selector=selector):
                completed = _run_child(runner, *selector, "-v")
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
        for symbol, script in broken_imports:
            with self.subTest(symbol=symbol):
                completed = _run_child(script)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ImportError", completed.stderr)
                self.assertIn(symbol, completed.stderr)

    @skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
    def test_broken_simsopt_jax_import_fails_instead_of_skipping(self):
        """An ImportError inside simsopt_jax on a supported runtime fails the run."""
        completed = _run_child("unittest_with_broken_simsopt_jax.py", _MODULE, "-v")
        self.assertNotEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("import of simsopt_jax.core.curve_geometry halted", completed.stderr)
        self.assertNotIn("skipped", completed.stderr)

    @skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
    def test_unsupported_jax_discovery_loads_every_test_module(self):
        """With an unsupported JAX every test module still imports, so discovery can skip."""
        completed = _run_child("discover_with_unsupported_jax.py")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
    def test_version_gate_accepts_every_0_12_release_and_rejects_0_11(self):
        """The floor is the 0.12 series: four-part and local releases pass, 0.11 fails."""
        for version in ("0.12.0", "0.12.0.1", "0.12.1", "0.12.0.1+cuda12", "0.13.0", "1.0.0"):
            with self.subTest(version=version):
                _require_jax_version("jax", version)
        for version in ("0.11.2", "0.11.9.1", "0.10.0", "0.9.0"):
            with self.subTest(version=version), self.assertRaisesRegex(
                ImportError, "requires jax >= 0.12; found " + version.replace(".", r"\.")
            ):
                _require_jax_version("jax", version)
