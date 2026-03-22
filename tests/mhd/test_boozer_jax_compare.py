import os

import numpy as np
import pytest

pytest.importorskip("vmec_jax")
pytest.importorskip("booz_xform_jax")
pytest.importorskip("booz_xform")

try:
    import vmec as vmec_mod  # noqa: F401
except Exception:
    vmec_mod = None

import jax.numpy as jnp

from simsopt.mhd import Vmec, Boozer, Quasisymmetry, VmecJax, BoozerJax, QuasisymmetryJax


@pytest.mark.skipif(vmec_mod is None, reason="vmec not installed")
def test_qs_jax_vs_vmec():
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
