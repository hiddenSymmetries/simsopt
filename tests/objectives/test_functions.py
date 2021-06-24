import unittest
import numpy as np
from simsopt.objectives.functions import Identity, Adder, Rosenbrock, TestObject1, TestObject2
from simsopt._core.optimizable import Target
from simsopt._core.dofs import Dofs


class IdentityTests(unittest.TestCase):
    def test_basic(self):
        iden = Identity()
        self.assertAlmostEqual(iden.J(), 0, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([0.0]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

        x = 3.5
        iden = Identity(x)
        self.assertAlmostEqual(iden.J(), x, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([x]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

        y = -2
        iden.set_dofs([y])
        self.assertAlmostEqual(iden.J(), y, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([y]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

    def test_gradient(self):
        iden = Identity()
        for n in range(1, 10):
            iden.set_dofs([np.random.rand() * 4 - 2])
            # Supply an object to finite_difference():
            fd_grad = Dofs([iden]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, iden.df)
            np.testing.assert_allclose(fd_grad, iden.dJ())
            # Supply a function to finite_difference():
            fd_grad = Dofs([iden.J]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, iden.df)
            np.testing.assert_allclose(fd_grad, iden.dJ())
            # Supply an attribute to finite_difference():
            fd_grad = Dofs([Target(iden, "f")]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, iden.df)
            np.testing.assert_allclose(fd_grad, iden.dJ())


class AdderTests(unittest.TestCase):
    def test_gradient(self):
        for n in range(1, 10):
            a = Adder(n)
            a.set_dofs(np.random.rand(n) * 4 - 2)
            # Supply an object to finite_difference():
            fd_grad = Dofs([a]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, a.df)
            np.testing.assert_allclose(fd_grad, a.dJ())
            # Supply a function to finite_difference():
            fd_grad = Dofs([a.J]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, a.df)
            np.testing.assert_allclose(fd_grad, a.dJ())
            # Supply an attribute to finite_difference():
            fd_grad = Dofs([Target(a, "f")]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, a.df)
            np.testing.assert_allclose(fd_grad, a.dJ())
            #print('diff in adder gradient: ', fd_grad - a.df)


class RosenbrockTests(unittest.TestCase):
    def test_1(self):
        """
        This is the most common use case.
        """
        r = Rosenbrock()
        self.assertAlmostEqual(r.term1(), -1.0, places=13)
        self.assertAlmostEqual(r.term2(), 0.0, places=13)

        # Change the parameters:
        x_new = [3, 2]
        r.set_dofs(x_new)
        np.testing.assert_allclose(x_new, r.get_dofs(), rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(r.term1(), 2.0, places=13)
        self.assertAlmostEqual(r.term2(), 0.7, places=13)

    def test_gradient(self):
        for n in range(1, 10):
            r = Rosenbrock(b=np.random.rand() * 2)  # Note b must be > 0.
            r.set_dofs(np.random.rand(2) * 4 - 2)

            rtol = 1e-6
            atol = 1e-6

            # Test gradient of term1

            # Supply a function to finite_difference():
            fd_grad = Dofs([r.term1]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, r.dterm1prop, rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, r.dterm1(), rtol=rtol, atol=atol)
            # Supply an attribute to finite_difference():
            fd_grad = Dofs([Target(r, "term1prop")]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, r.dterm1prop, rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, r.dterm1(), rtol=rtol, atol=atol)

            # Test gradient of term2

            # Supply a function to finite_difference():
            fd_grad = Dofs([r.term2]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, r.dterm2prop, rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, r.dterm2(), rtol=rtol, atol=atol)
            # Supply an attribute to finite_difference():
            fd_grad = Dofs([Target(r, "term2prop")]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, r.dterm2prop, rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, r.dterm2(), rtol=rtol, atol=atol)
            #print('Diff in term2:', fd_grad - r.dterm2())


class TestObject1Tests(unittest.TestCase):
    def test_gradient(self):
        for n in range(1, 20):
            o = TestObject1(np.random.rand())
            o.adder1.set_dofs(np.random.rand(3) * 4 - 2)
            o.adder2.set_dofs(np.random.rand(2) - 0.5)

            # Randomly fix some of the degrees of freedom
            o.fixed = np.random.rand(1) > 0.5
            o.adder1.fixed = np.random.rand(3) > 0.5
            o.adder2.fixed = np.random.rand(2) > 0.5

            rtol = 1e-4
            atol = 1e-4

            dofs = Dofs([o.J])
            #mask = np.logical_not(dofs.fixed)
            mask = np.logical_not(np.array(dofs.func_fixed[0]))

            # Supply a function to finite_difference():
            fd_grad = Dofs([o.J]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, o.df[mask], rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, o.dJ()[mask], rtol=rtol, atol=atol)
            # Supply an object to finite_difference():
            fd_grad = Dofs([o]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, o.df[mask], rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, o.dJ()[mask], rtol=rtol, atol=atol)
            # Supply an attribute to finite_difference():
            fd_grad = Dofs([Target(o, "f")]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, o.df[mask], rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, o.dJ()[mask], rtol=rtol, atol=atol)

            print('Diff in TestObject1:', fd_grad - o.df[mask])


class TestObject2Tests(unittest.TestCase):
    def test_gradient(self):
        for n in range(1, 20):
            v1 = np.random.rand() * 4 - 2
            v2 = np.random.rand() * 4 - 2
            o = TestObject2(v1, v2)
            o.adder.set_dofs(np.random.rand(2) * 4 - 2)
            o.t.set_dofs([np.random.rand() * 4 - 2])
            o.t.adder1.set_dofs(np.random.rand(3) * 4 - 2)
            o.t.adder2.set_dofs(np.random.rand(2) - 0.5)

            # Randomly fix some of the degrees of freedom
            o.fixed = np.random.rand(2) > 0.5
            o.adder.fixed = np.random.rand(2) > 0.5
            o.t.adder1.fixed = np.random.rand(3) > 0.5
            o.t.adder2.fixed = np.random.rand(2) > 0.5

            rtol = 1e-4
            atol = 1e-4

            dofs = Dofs([o.J])
            mask = np.logical_not(np.array(dofs.func_fixed[0]))

            # Supply a function to finite_difference():
            fd_grad = Dofs([o.J]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, o.df[mask], rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, o.dJ()[mask], rtol=rtol, atol=atol)
            # Supply an object to finite_difference():
            fd_grad = Dofs([o]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, o.df[mask], rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, o.dJ()[mask], rtol=rtol, atol=atol)
            # Supply an attribute to finite_difference():
            fd_grad = Dofs([Target(o, "f")]).fd_jac().flatten()
            np.testing.assert_allclose(fd_grad, o.df[mask], rtol=rtol, atol=atol)
            np.testing.assert_allclose(fd_grad, o.dJ()[mask], rtol=rtol, atol=atol)

            print('Diff in TestObject2:', fd_grad - o.df[mask])


if __name__ == "__main__":
    unittest.main()
