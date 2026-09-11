import contextlib
import io
import unittest

import numpy as np

from simsopt.configs import get_data
from simsopt.field import BiotSavart, Coil
from simsopt.geo import (
    CurveRZFourier, CurveXYZFourier, CurveXYZFourierSymmetries, CurveLength)
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


def solve_full_torus_axis(bs, order, target):
    """
    Solve for the magnetic axis using the fully general curve representation,
    ``nfp=1`` and ``stellsym=False``, seeded with the points ``target``. This
    representation assumes no symmetry at all, so it can also represent an axis
    of a configuration whose symmetry has been broken.
    """
    quadpoints = np.linspace(0, 1, 2*order+1, endpoint=False)
    axis = CurveXYZFourierSymmetries(quadpoints, order, nfp=1, stellsym=False, ntor=1)
    axis.least_squares_fit(target)
    fl = PeriodicFieldLine(bs, axis,
                           options=dict(newton_tol=1e-11, newton_maxiter=30, verbose=False))
    return fl, fl.run_code(CurveLength(axis).J())


def antisymmetric_dofs(curve):
    """
    The ``xs``, ``yc`` and ``zc`` dof blocks of a non-stellarator-symmetric
    :class:`CurveXYZFourierSymmetries`. They vanish if and only if the curve is
    stellarator symmetric.
    """
    order, d = curve.order, curve.x
    return (d[order+1:2*order+1], d[2*order+1:3*order+2], d[4*order+2:5*order+3])


def perturb_one_coil(coils, index=1, dz=0.01):
    """
    Return a copy of ``coils`` in which a single coil is displaced by ``dz``
    along z. The coil is first replaced by an independent
    :class:`CurveXYZFourier`, so that moving it does not move its symmetry
    copies; this breaks both the stellarator and the field-period symmetry.
    """
    original = coils[index].curve
    independent = CurveXYZFourier(original.quadpoints, coils[0].curve.order)
    independent.least_squares_fit(original.gamma())

    x = independent.x.copy()
    x[independent.local_dof_names.index('zc(0)')] += dz
    independent.x = x

    perturbed = list(coils)
    perturbed[index] = Coil(independent, coils[index].current)
    return perturbed


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
        """
        The L-BFGS penalty formulation also drives the residual to zero. With
        ``verbose=True`` it prints a summary, and a second call returns the
        cached result instead of solving again.
        """
        fl, bs, seed, nfp = get_axis_fieldline()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            resdict = fl.minimize_boozer_penalty_constraints_LBFGS(
                tol=1e-12, maxiter=2000, verbose=True)

        self.assertTrue(resdict["success"])
        self.assertLess(resdict["fun"], 1e-8)
        r, _, _ = field_line_residual(fl.curve, resdict["length"], bs)
        self.assertLess(np.linalg.norm(r, np.inf), 1e-3)
        self.assertIn("L-BFGS-B solve", out.getvalue())

        # the solve is cached: calling again hands back the same result
        self.assertFalse(fl.need_to_run_code)
        self.assertIs(fl.minimize_boozer_penalty_constraints_LBFGS(), resdict)

    def test_newton_verbose_default_length_and_caching(self):
        """
        Called directly, the exact-Newton solver defaults the field-line length
        to the length of the seed curve, prints a summary when ``verbose=True``,
        and returns the cached result on a second call.
        """
        fl, bs, seed, nfp = get_axis_fieldline()

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            # no `length` argument: it falls back to CurveLength(curve).J()
            res = fl.solve_residual_equation_exactly_newton(verbose=True)

        self.assertTrue(res["success"])
        self.assertIn("NEWTON solve", out.getvalue())
        self.assertAlmostEqual(res["length"], CurveLength(fl.curve).J(), places=8)

        self.assertFalse(fl.need_to_run_code)
        self.assertIs(fl.solve_residual_equation_exactly_newton(), res)

    def test_dcoils_dcurrents_vjp_nonstellsym(self):
        """
        The coil/current vjp also matches finite differences for a
        non-stellarator-symmetric field line. There the residual carries the
        extra equation pinning the parametrization, which does not depend on
        ``B`` and so has to be dropped from ``dres_dB``.
        """
        np.random.seed(3)
        base_curves, base_currents, ma, nfp, bs0 = get_data("STAR_Lite-A_low")
        order = 2*ma.order
        quadpoints = np.linspace(0, 1, 2*order+1, endpoint=False)
        ma_seed = CurveRZFourier(quadpoints, ma.order, ma.nfp, ma.stellsym)
        ma_seed.x = ma.x

        bs = BiotSavart(perturb_one_coil(bs0.coils))
        fl, res = solve_full_torus_axis(bs, order, ma_seed.gamma())
        self.assertTrue(res["success"])
        self.assertFalse(fl.curve.stellsym)

        mask = fl.res["mask"]
        lm = np.random.standard_normal(int(mask.sum()))
        grad = periodicfieldline_dcoils_dcurrents_vjp(lm, bs, fl)(bs)

        c0 = bs.x.copy()
        length = fl.res["length"]

        def g():
            r, _, _ = field_line_residual(fl.curve, length, bs)
            return lm @ r[mask]

        g0 = g()
        h = np.random.standard_normal(c0.shape)
        pred = grad @ h

        rel_errs = []
        for eps in np.power(2.0, -np.arange(8, 22)):
            bs.x = c0 + eps*h
            rel_errs.append(abs((g() - g0)/eps - pred) / max(abs(pred), 1e-30))
        bs.x = c0  # restore

        self.assertLess(min(rel_errs), 1e-3)

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

    def test_nonstellsym_fieldline(self):
        """
        Displacing a single coil breaks the stellarator symmetry of the field,
        and the periodic field line must then be represented without any
        symmetry (``nfp=1``, ``stellsym=False``).

        Solved in that representation, the unperturbed configuration must still
        return a stellarator-symmetric axis, i.e. its ``xs``, ``yc`` and ``zc``
        dof blocks vanish. Once one coil is moved those blocks become non-zero.
        """
        base_curves, base_currents, ma, nfp, bs = get_data("STAR_Lite-A_low")
        coils = bs.coils

        # an nfp=1 curve needs twice the harmonics of the nfp=2 magnetic axis
        order = 2*ma.order
        quadpoints = np.linspace(0, 1, 2*order+1, endpoint=False)
        ma_seed = CurveRZFourier(quadpoints, ma.order, ma.nfp, ma.stellsym)
        ma_seed.x = ma.x
        target = ma_seed.gamma()

        # unperturbed: the solution comes back stellarator symmetric
        fl0, res0 = solve_full_torus_axis(BiotSavart(coils), order, target)
        self.assertTrue(res0["success"])
        self.assertLess(np.linalg.norm(res0["residual"], np.inf), 1e-9)
        for block in antisymmetric_dofs(fl0.curve):
            self.assertLess(np.linalg.norm(block), 1e-12)

        # one coil displaced by 1 cm: the symmetry is broken
        fl1, res1 = solve_full_torus_axis(
            BiotSavart(perturb_one_coil(coils)), order, target)
        self.assertTrue(res1["success"])
        self.assertLess(np.linalg.norm(res1["residual"], np.inf), 1e-9)
        for block in antisymmetric_dofs(fl1.curve):
            self.assertGreater(np.linalg.norm(block), 1e-5)

        # and the perturbed axis really is a different curve
        self.assertGreater(abs(res1["length"] - res0["length"]), 1e-4)

        # without stellarator symmetry every residual equation is kept,
        # including the extra one pinning the parametrization (y=0)
        mask = fl1.get_stellsym_mask()
        self.assertEqual(mask.size, 3*(2*order+1) + 1)
        self.assertTrue(np.all(mask))
        self.assertEqual(int(mask.sum()), fl1.curve.num_dofs() + 1)

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
