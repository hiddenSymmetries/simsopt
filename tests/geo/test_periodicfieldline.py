import unittest

import numpy as np

from simsopt.configs import get_data
from simsopt.geo import CurveRZFourier, CurveXYZFourierSymmetries, CurveLength
from simsopt.geo.periodicfieldline import (
    PeriodicFieldLine, field_line_residual, periodicfieldline_dcoils_dcurrents_vjp)

configurations = ["STAR_Lite-A_low", "STAR_Lite-A_medium", "STAR_Lite-A_high"]


def get_axis_fieldline(name="STAR_Lite-A_low"):
    """
    Build a :class:`PeriodicFieldLine` for configuration ``name``, seeded with
    the magnetic axis. Returns the (unsolved) field line, the Biot-Savart field,
    the seed axis curve and the number of field periods.
    """
    base_curves, base_currents, ma, nfp, bs = get_data(name)
    order = ma.order
    quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
    axis = CurveXYZFourierSymmetries(quadpoints, order, nfp=nfp, stellsym=True, ntor=1)

    # the magnetic axis is a CurveRZFourier spanning the full torus over [0, 1);
    # resample it on the single-period quadpoints of `axis` before fitting.
    ma_seed = CurveRZFourier(quadpoints, ma.order, ma.nfp, ma.stellsym)
    ma_seed.x = ma.x
    axis.least_squares_fit(ma_seed.gamma())
    return PeriodicFieldLine(bs, axis), bs, ma_seed, nfp


class PeriodicFieldLineTests(unittest.TestCase):

    def test_newton_converges(self):
        """The Newton solver finds the periodic field line for every config."""
        for name in configurations:
            with self.subTest(name=name):
                fl, bs, seed, nfp = get_axis_fieldline(name)
                res = fl.run_code(CurveLength(fl.curve).J())

                self.assertTrue(res["success"])
                mask = res["mask"]
                self.assertLess(np.linalg.norm(res["residual"][mask], np.inf), 1e-9)

                # the solved field line is a small correction to the magnetic axis
                dist = np.linalg.norm(fl.curve.gamma() - seed.gamma(), axis=1).max()
                self.assertLess(dist, 0.1)

    def test_arclength_parametrization(self):
        """
        On the solution ``gammadash = length * B/|B|``, so the curve is
        parametrized proportionally to arclength: ``|gammadash|`` is constant and
        the length dof equals the geometric length of the curve.
        """
        fl, bs, seed, nfp = get_axis_fieldline()
        res = fl.run_code(CurveLength(fl.curve).J())

        speed = np.linalg.norm(fl.curve.gammadash(), axis=1)
        self.assertLess(speed.std() / speed.mean(), 1e-6)
        self.assertAlmostEqual(res["length"], CurveLength(fl.curve).J(), places=8)
        self.assertGreater(res["length"], 0)

    def test_penalty_solver_converges(self):
        """The L-BFGS penalty formulation also drives the residual to zero."""
        fl, bs, seed, nfp = get_axis_fieldline()
        resdict = fl.minimize_boozer_penalty_constraints_LBFGS(tol=1e-12, maxiter=2000)

        self.assertTrue(resdict["success"])
        self.assertLess(resdict["fun"], 1e-8)
        r, _, _ = field_line_residual(fl.curve, resdict["length"], bs)
        self.assertLess(np.linalg.norm(r, np.inf), 1e-3)

    def test_residual_jacobian_taylor(self):
        """
        Taylor test of the analytic Jacobian ``dres`` of the field-line residual
        with respect to (curve dofs, length): the finite-difference error must
        converge at second order.
        """
        np.random.seed(1)
        fl, bs, seed, nfp = get_axis_fieldline()
        curve = fl.curve

        x0 = np.concatenate((curve.get_dofs(), [CurveLength(curve).J()]))

        def residual(x):
            curve.set_dofs(x[:-1])
            r, dres, _ = field_line_residual(curve, x[-1], bs)
            return r, dres

        r0, dres0 = residual(x0)
        h = np.random.standard_normal(x0.shape)
        pred = dres0 @ h

        errs = []
        for eps in np.power(2.0, -np.arange(4, 14)):
            r1, _ = residual(x0 + eps*h)
            errs.append(np.linalg.norm((r1 - r0)/eps - pred, np.inf))
        residual(x0)  # restore

        errs = np.array(errs)
        rates = errs[:-1] / errs[1:]
        # the last few refinements should show ~second order convergence
        self.assertTrue(np.all(rates[-3:] > 1.9))

    def test_dcoils_dcurrents_vjp(self):
        """
        The vjp of the residual with respect to the coil/current dofs (through
        the field ``B``) matches finite differences of ``lm . residual``.
        """
        np.random.seed(2)
        fl, bs, seed, nfp = get_axis_fieldline()
        fl.run_code(CurveLength(fl.curve).J())

        mask = fl.res["mask"]
        lm = np.random.standard_normal(int(mask.sum()))
        grad = periodicfieldline_dcoils_dcurrents_vjp(lm, bs, fl)(bs)

        c0 = bs.x.copy()
        length = fl.res["length"]

        def g():
            r, _, _ = field_line_residual(fl.curve, length, bs)
            return lm @ r[mask]

        g0 = g()
        hb = np.random.standard_normal(c0.shape)
        pred = grad @ hb

        rel_errs = []
        for eps in np.power(2.0, -np.arange(8, 22)):
            bs.x = c0 + eps*hb
            rel_errs.append(abs((g() - g0)/eps - pred) / max(abs(pred), 1e-30))
        bs.x = c0  # restore

        self.assertLess(min(rel_errs), 1e-3)

    def test_stellsym(self):
        """
        In the stellarator-symmetric case, the field-line residual obeys

            rx(-tk) = -rx(tk)
            ry(-tk) =  ry(tk)
            rz(-tk) =  rz(tk)

        For nfp=2 one first has to rotate the residual by 2*pi/nfp, but the same
        relation holds.
        """
        fl, bs, seed, nfp = get_axis_fieldline()
        axis = fl.curve

        out = field_line_residual(axis, CurveLength(axis).J(), bs)
        residual = out[0].reshape((-1, 3))

        angle = 2*np.pi/nfp
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        r1 = residual[1]
        r2 = R @ residual[-1]
        self.assertLess(np.abs(r1[0] + r2[0]), 1e-13)
        self.assertLess(np.linalg.norm(r1[1:] - r2[1:]), 1e-13)

    def test_run_code_caching(self):
        """
        ``run_code`` solves once and caches; touching a coil dof invalidates the
        cache via ``recompute_bell``.
        """
        fl, bs, seed, nfp = get_axis_fieldline()
        self.assertTrue(fl.need_to_run_code)

        fl.run_code(CurveLength(fl.curve).J())
        self.assertFalse(fl.need_to_run_code)

        # a subsequent call is a no-op that returns the cached result (None here)
        self.assertIsNone(fl.run_code(1.234))

        # changing the Biot-Savart dofs must trigger a recompute
        x = bs.x.copy()
        x[0] += 1e-3
        bs.x = x
        self.assertTrue(fl.need_to_run_code)

    def test_stellsym_mask(self):
        """The stellarator-symmetry mask has the expected shape and entries."""
        fl, bs, seed, nfp = get_axis_fieldline()
        order = fl.curve.order
        mask = fl.get_stellsym_mask()

        # three residual entries per quadpoint
        self.assertEqual(mask.size, 3*(2*order+1))
        # the masked-out entries are the xc(0) constraint and the redundant
        # second-half rows enforced by stellarator symmetry; the number of kept
        # equations equals the number of unknowns (curve dofs + length)
        self.assertFalse(mask[0])
        self.assertEqual(int(mask.sum()), 3*order + 2)
        self.assertEqual(int(mask.sum()), fl.curve.num_dofs() + 1)


if __name__ == "__main__":
    unittest.main()
