"""Reduced stage-II smoke coverage for native objectives with the JAX field."""

from unittest_jax_support import JaxTestCase, optional_jax_imports

import numpy as np
from scipy.optimize import minimize

from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import SquaredFlux

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax.backend import set_backend


class TestStageTwoBiotsavartJax(JaxTestCase):
    def test_native_objective_optimization(self):
        """Two L-BFGS-B iterations keep native objective gradients finite and decrease J."""
        set_backend("jax", device="cpu", intent="parity")
        s = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.3)
        s.set_zs(1, 0, 0.3)
        base_curves = create_equally_spaced_curves(
            2, s.nfp, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32
        )
        base_currents = [Current(1e5) for _ in base_curves]
        base_currents[0].fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, stellsym=True)
        bs = JaxBiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        Jf = SquaredFlux(s, bs)
        JF = Jf + 1e-5 * sum(CurveLength(c) for c in base_curves)

        def fun(dofs):
            JF.x = dofs
            return JF.J(), JF.dJ()

        initial_value, initial_gradient = fun(JF.x)
        self.assertTrue(np.isfinite(initial_value))
        self.assertTrue(np.all(np.isfinite(initial_gradient)))
        res = minimize(
            fun,
            JF.x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2, "maxcor": 10},
        )
        self.assertTrue(np.isfinite(res.fun))
        self.assertLessEqual(res.fun, initial_value)
