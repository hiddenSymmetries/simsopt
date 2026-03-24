import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np

try:
    from vmec_jax.wout import state_from_wout
    from vmec_jax.booz_input import booz_xform_inputs_from_state
    import vmec_jax as vj
    import booz_xform_jax  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise unittest.SkipTest(f"vmec_jax/booz_xform_jax not installed: {exc}")

try:
    import vmec as vmec_mod  # noqa: F401
except Exception:
    vmec_mod = None

from dataclasses import replace

from simsopt.mhd import Vmec


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


@unittest.skipIf(vmec_mod is None or vj is None, "vmec/vmec_jax not installed")
def test_booz_input_uses_equilibrium_iota_for_current_driven_case():
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
        vmec.indata.mpol = 3
        vmec.indata.ntor = 3
        surf = vmec.boundary
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
        surf.fix("rc(0,0)")
        vmec.run()

        cfg, indata = vj.load_config(filename)
        cfg = replace(cfg, mpol=3, ntor=3)
        static = vj.build_static(cfg)
        wout = vj.load_wout(vmec.output_file)
        state = state_from_wout(wout)
        inputs = booz_xform_inputs_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=wout.signgs,
        )

        np.testing.assert_allclose(np.asarray(inputs.iota), np.asarray(wout.iotas)[1:], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(inputs.lmns), np.asarray(wout.lmns)[1:, :], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(inputs.bsubumnc), np.asarray(wout.bsubumnc)[1:, :], rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(np.asarray(inputs.bsubvmnc), np.asarray(wout.bsubvmnc)[1:, :], rtol=5e-3, atol=5e-3)
