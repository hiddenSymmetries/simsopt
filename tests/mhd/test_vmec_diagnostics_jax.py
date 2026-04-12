import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import jax.numpy as jnp

try:
    import vmec_jax as vj  # noqa: F401
except Exception as exc:
    raise unittest.SkipTest(f"vmec_jax not installed: {exc}")

try:
    import vmec as vmec_mod  # noqa: F401
except Exception:
    vmec_mod = None

from simsopt.mhd import (
    Vmec,
    VmecJax,
    QuasisymmetryRatioResidual,
    QuasisymmetryRatioResidualJax,
    quasisymmetry_ratio_residual_from_wout_jax,
)


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


def _input_filename():
    filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "2_Intermediate",
        "inputs",
        "input.nfp4_QH_warm_start",
    )
    return os.path.abspath(filename)


@unittest.skipIf(vmec_mod is None, "vmec not installed")
def test_qs_ratio_residual_jax_matches_classic_formula():
    with _temporary_cwd():
        filename = _input_filename()
        surfaces = [0.25, 0.5, 0.75]

        vmec = Vmec(filename, verbose=False)
        vmec.indata.mpol = 3
        vmec.indata.ntor = 3

        qs_ref = QuasisymmetryRatioResidual(
            vmec,
            surfaces,
            helicity_m=1,
            helicity_n=-1,
            ntheta=17,
            nphi=18,
        )
        residuals_ref = qs_ref.residuals()
        total_ref = qs_ref.total()

        calc = quasisymmetry_ratio_residual_from_wout_jax(
            vmec.wout,
            surfaces=surfaces,
            helicity_m=1,
            helicity_n=-1,
            ntheta=17,
            nphi=18,
        )

        np.testing.assert_allclose(np.asarray(calc["residuals1d"]), np.asarray(residuals_ref), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(calc["total"]), np.asarray(total_ref), rtol=1e-10, atol=1e-12)


def test_qs_ratio_residual_jax_wrapper_matches_wout_path():
    filename = _input_filename()
    vmec = VmecJax(filename, verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(solver="gd", warm_start_iters=0, max_iter=5, grad_tol=1e-2)

    surf = vmec.boundary
    x0 = surf.get_free_params()
    state = vmec._solve_state(jnp.asarray(x0))

    qs = QuasisymmetryRatioResidualJax(vmec, [0.25, 0.5, 0.75], helicity_m=1, helicity_n=-1, ntheta=17, nphi=18)
    total_from_state = qs.total_from_state(state)
    total_from_wout = qs.total_from_wout(vmec.get_wout_from_state(state))

    np.testing.assert_allclose(np.asarray(total_from_state), np.asarray(total_from_wout), rtol=1e-12, atol=1e-12)
