import unittest
import numpy as np
from simsopt.functions import Identity, Adder, Rosenbrock
from simsopt.finite_difference import finite_difference
from simsopt.optimizable import Target

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
            fd_grad = finite_difference(iden)
            np.testing.assert_allclose(fd_grad, iden.df)
            np.testing.assert_allclose(fd_grad, iden.dJ())
            # Supply a function to finite_difference():
            fd_grad = finite_difference(iden.J)
            np.testing.assert_allclose(fd_grad, iden.df)
            np.testing.assert_allclose(fd_grad, iden.dJ())
            # Supply an attribute to finite_difference():
            fd_grad = finite_difference(Target(iden, "f"))
            np.testing.assert_allclose(fd_grad, iden.df)
            np.testing.assert_allclose(fd_grad, iden.dJ())

class AdderTests(unittest.TestCase):
    def test_gradient(self):
        for n in range(1, 10):
            a = Adder(n)
            a.set_dofs(np.random.rand(n) * 4 - 2)
            # Supply an object to finite_difference():
            fd_grad = finite_difference(a)
            np.testing.assert_allclose(fd_grad, a.df)
            np.testing.assert_allclose(fd_grad, a.dJ())
            # Supply a function to finite_difference():
            fd_grad = finite_difference(a.J)
            np.testing.assert_allclose(fd_grad, a.df)
            np.testing.assert_allclose(fd_grad, a.dJ())
            # Supply an attribute to finite_difference():
            fd_grad = finite_difference(Target(a, "f"))
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
        self.assertAlmostEqual(r.term2(),  0.0, places=13)

        # Change the parameters:
        x_new = [3, 2]
        r.set_dofs(x_new)
        np.testing.assert_allclose(x_new, r.get_dofs(), rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(r.term1(), 2.0, places=13)
        self.assertAlmostEqual(r.term2(), 0.7, places=13)

    def test_gradient(self):
        for n in range(1, 10):
            r = Rosenbrock(b=np.random.rand() * 2) # Note b must be > 0.
            r.set_dofs(np.random.rand(2) * 4 - 2)

            # Test gradient of term1
            
            # Supply a function to finite_difference():
            fd_grad = finite_difference(r.term1)
            np.testing.assert_allclose(fd_grad, r.dterm1prop)
            np.testing.assert_allclose(fd_grad, r.dterm1())
            # Supply an attribute to finite_difference():
            fd_grad = finite_difference(Target(r, "term1prop"))
            np.testing.assert_allclose(fd_grad, r.dterm1prop)
            np.testing.assert_allclose(fd_grad, r.dterm1())

            # Test gradient of term2
            
            # Supply a function to finite_difference():
            fd_grad = finite_difference(r.term2)
            np.testing.assert_allclose(fd_grad, r.dterm2prop)
            np.testing.assert_allclose(fd_grad, r.dterm2())
            # Supply an attribute to finite_difference():
            fd_grad = finite_difference(Target(r, "term2prop"))
            np.testing.assert_allclose(fd_grad, r.dterm2prop)
            np.testing.assert_allclose(fd_grad, r.dterm2())
            #print('Diff in term2:', fd_grad - r.dterm2())
            
if __name__ == "__main__":
    unittest.main()
