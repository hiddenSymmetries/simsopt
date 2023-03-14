import unittest
import logging
import numpy as np
from simsopt.objectives.functions import Identity, Rosenbrock, Failer, Affine
from simsopt.objectives.constrained import ConstrainedProblem
from simsopt._core.util import ObjectiveFailure

#logging.basicConfig(level=logging.DEBUG)


class ConstrainedProblemTests(unittest.TestCase):

    def test_bound(self):
        # single variable
        iden = Identity()
        prob = ConstrainedProblem(iden.f, lb=-2.0, ub=np.inf)
        np.testing.assert_allclose(prob.lb, -2.0, 1e-14)
        np.testing.assert_allclose(prob.ub, np.inf, 1e-14)
        self.assertTrue(prob.has_bounds == True)
        self.assertTrue(prob.has_lc == False)
        self.assertTrue(prob.has_nlc == False)

        # multivariable, scalar bounds
        rosen = Rosenbrock()
        prob = ConstrainedProblem(rosen.f, lb=-np.inf, ub=17.0)
        np.testing.assert_allclose(prob.lb, -np.inf, 1e-14)
        np.testing.assert_allclose(prob.ub, 17.0, 1e-14)
        self.assertTrue(prob.has_bounds == True)
        self.assertTrue(prob.has_lc == False)
        self.assertTrue(prob.has_nlc == False)

        # multivariable, vector bounds
        rosen = Rosenbrock()
        lb = [-np.inf, 0.0]
        ub = np.array([17.0, 4.0])
        prob = ConstrainedProblem(rosen.f, lb=lb, ub=ub)
        np.testing.assert_allclose(prob.lb, lb, 1e-14)
        np.testing.assert_allclose(prob.ub, ub, 1e-14)
        self.assertTrue(prob.has_bounds == True)
        self.assertTrue(prob.has_lc == False)
        self.assertTrue(prob.has_nlc == False)

        # multivariable, one sided bounds
        rosen = Rosenbrock()
        lb = [-np.inf, 0.0]
        prob = ConstrainedProblem(rosen.f, lb=lb)
        np.testing.assert_allclose(prob.lb, lb, 1e-14)
        self.assertTrue(prob.has_bounds == True)
        self.assertTrue(prob.has_lc == False)
        self.assertTrue(prob.has_nlc == False)

    def test_fails(self):
        # first objective evaluation is a fail
        fail = Failer(nparams=2, nvals=0, fail_index=1)
        prob = ConstrainedProblem(fail.J)
        with self.assertRaises(ObjectiveFailure):
            prob.objective()

        # first constraint evaluation is a fail
        fail = Failer(nparams=2, nvals=0, fail_index=1)
        prob = ConstrainedProblem(fail.J, tuples_nlc=[(fail.J, 0.0, 1.0)])
        with self.assertRaises(ObjectiveFailure):
            prob.nonlinear_constraints()

        # second evaluation is a fail
        fail = Failer(nparams=2, nvals=0, fail_index=2)
        prob = ConstrainedProblem(fail.J)
        prob.objective(prob.x)
        self.assertEqual(prob.objective(prob.x + 1), prob.fail)

        # second evaluation is a fail
        fail = Failer(nparams=2, nvals=0, fail_index=2)
        prob = ConstrainedProblem(fail.J, tuples_nlc=[(fail.J, 0.0, 1.0)])
        prob.nonlinear_constraints(prob.x)
        np.testing.assert_allclose(prob.nonlinear_constraints(prob.x + 1), np.full(2, prob.fail), 1e-12)

    def test_linear(self):

        # multivariate linear constraints with wrong shape
        rosen = Rosenbrock()
        A = np.random.randn(2)
        b = 7
        prob = ConstrainedProblem(rosen.f, tuple_lc=(A, b))
        np.testing.assert_allclose(prob.A_lc, np.atleast_2d(A), 1e-14)
        np.testing.assert_allclose(prob.b_lc, np.atleast_1d(b), 1e-14)
        self.assertTrue(prob.has_bounds == False)
        self.assertTrue(prob.has_lc == True)
        self.assertTrue(prob.has_nlc == False)

        # multivariate linear constraints
        rosen = Rosenbrock()
        A = np.random.randn(3, 2)
        b = np.random.randn(3)
        prob = ConstrainedProblem(rosen.f, tuple_lc=(A, b))
        np.testing.assert_allclose(prob.A_lc, A, 1e-14)
        np.testing.assert_allclose(prob.b_lc, b, 1e-14)
        self.assertTrue(prob.has_bounds == False)
        self.assertTrue(prob.has_lc == True)
        self.assertTrue(prob.has_nlc == False)

    def test_nonlinear(self):

        # scalar valued constraints, vector bounds
        rosen = Rosenbrock()
        lb = np.array([-1.0, 0.0])
        ub = np.array([10.0, np.inf])
        prob = ConstrainedProblem(rosen.f, tuples_nlc=[(rosen.f, lb, ub)])
        np.testing.assert_allclose(prob.lhs_nlc, [lb], 1e-12)
        np.testing.assert_allclose(prob.rhs_nlc, [ub], 1e-12)
        correct_val = np.concatenate([lb - rosen.f(), rosen.f() - ub])
        np.testing.assert_allclose(prob.nonlinear_constraints(), correct_val, 1e-12)
        self.assertTrue(prob.has_nlc == True)

        # vector valued constraints, scalar bounds
        rosen = Rosenbrock()
        aff = Affine(3, 2)
        prob = ConstrainedProblem(rosen.f, tuples_nlc=[(aff.f, -np.inf, 8.0)])
        np.testing.assert_allclose(prob.lhs_nlc, [-np.inf], 1e-12)
        np.testing.assert_allclose(prob.rhs_nlc, [8.0], 1e-12)
        correct_val = aff.A @ aff.x + aff.B - 8.0
        np.testing.assert_allclose(prob.nonlinear_constraints(), correct_val, 1e-12)
        self.assertTrue(prob.has_nlc == True)

        # vector valued constraints, vector bounds
        rosen = Rosenbrock()
        aff = Affine(3, 2)
        lb = np.array([-1.0, 0.0])
        ub = np.array([10.0, np.inf])
        prob = ConstrainedProblem(rosen.f, tuples_nlc=[(aff.f, lb, ub)])
        np.testing.assert_allclose(prob.lhs_nlc, [lb], 1e-12)
        np.testing.assert_allclose(prob.rhs_nlc, [ub], 1e-12)
        correct_val = np.concatenate([lb - (aff.A @ aff.x + aff.B), aff.A @ aff.x + aff.B - ub])
        np.testing.assert_allclose(prob.nonlinear_constraints(), correct_val, 1e-12)
        self.assertTrue(prob.has_nlc == True)

        # multiple constraints
        rosen = Rosenbrock()
        aff = Affine(3, 2)
        lb = np.array([-1.0, 0.0])
        ub = np.array([10.0, np.inf])
        prob = ConstrainedProblem(rosen.f, tuples_nlc=[(aff.f, lb, ub), (rosen.f, 0.0, 8.0)])
        np.testing.assert_allclose(prob.lhs_nlc[0], lb, 1e-12)
        np.testing.assert_allclose(prob.rhs_nlc[0], ub, 1e-12)
        np.testing.assert_allclose(prob.lhs_nlc[1], 0.0, 1e-12)
        np.testing.assert_allclose(prob.rhs_nlc[1], 8.0, 1e-12)
        correct_val = np.concatenate([lb - (aff.A @ aff.x + aff.B), aff.A @ aff.x + aff.B - ub,
                                      [0.0 - rosen.f()], [rosen.f() - 8.0]])
        np.testing.assert_allclose(prob.nonlinear_constraints(), correct_val, 1e-12)
        self.assertTrue(prob.has_nlc == True)

    def test_unconstrained(self):
        rosen = Rosenbrock()
        prob = ConstrainedProblem(rosen.f)
        np.testing.assert_allclose(prob.objective(), rosen.f(), 1e-12)
        self.assertTrue(prob.has_bounds == False)
        self.assertTrue(prob.has_lc == False)
        self.assertTrue(prob.has_nlc == False)


if __name__ == "__main__":
    unittest.main()
