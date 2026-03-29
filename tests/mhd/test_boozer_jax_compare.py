import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np

try:
    import vmec_jax  # noqa: F401
    import booz_xform_jax  # noqa: F401
    import booz_xform  # noqa: F401
except Exception as exc:
    raise unittest.SkipTest(f"optional Boozer/JAX stack not installed: {exc}")

try:
    import vmec as vmec_mod  # noqa: F401
except Exception:
    vmec_mod = None

import jax.numpy as jnp

from simsopt.mhd import Vmec, Boozer, Quasisymmetry, VmecJax, BoozerJax, QuasisymmetryJax


@contextmanager
def _temporary_cwd():
    """Run a test inside a temporary working directory."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield tmpdir
        finally:
            os.chdir(cwd)


@unittest.skipIf(vmec_mod is None, "vmec not installed")
def test_qs_jax_vs_vmec():
    with _temporary_cwd():
        filename = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "examples",
            "2_Intermediate",
            "inputs",
            "input.nfp4_QH_warm_start",
        )
        filename = os.path.abspath(filename)

        vmec = Vmec(filename, verbose=False)
        boozer = Boozer(vmec, mpol=8, ntor=8, use_wout_file=True)
        qs = Quasisymmetry(boozer, 0.5, 1, 1)
        qs_vmec = float(np.sum(qs.J() ** 2))

        vmec_j = VmecJax(filename, verbose=False)
        boozer_j = BoozerJax(vmec_j, mpol=8, ntor=8)
        qs_j = QuasisymmetryJax(boozer_j, 0.5, 1, 1)
        x0 = vmec_j.boundary.get_free_params()
        qs_jax = float(np.sum(np.asarray(qs_j.J(jnp.asarray(x0))) ** 2))

        assert np.isfinite(qs_vmec)
        assert np.isfinite(qs_jax)

        ratio = qs_jax / qs_vmec
        assert ratio < 20.0
