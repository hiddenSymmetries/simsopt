import unittest
import numpy as np
from simsopt.dofs import get_owners, Dofs
from simsopt.functions import Identity, Adder, TestObject2, Rosenbrock
from simsopt.optimizable import Target

class GetOwnersTests(unittest.TestCase):
    def test_no_dependents(self):
        """
        For an object that does not depend on anything, just return the
        original object.
        """
        obj = object()
        self.assertEqual(get_owners(obj), [obj])

        iden = Identity()
        self.assertEqual(get_owners(iden), [iden])

    def test_depth_1(self):
        """
        Check cases in which the original object depends on 1 or more others.
        """
        o1 = Identity()
        o2 = Identity()
        o1.o2 = o2
        o1.depends_on = ["o2"]
        self.assertEqual(get_owners(o1), [o1, o2])

        o3 = object()
        o1.depends_on = ["o3", "o2"]
        o1.o3 = o3
        self.assertEqual(get_owners(o1), [o1, o3, o2])
        
    def test_depth_2(self):
        """
        Check cases in which the original object depends on another, which
        depends on another.
        """
        o1 = Identity()
        o2 = Identity()
        o3 = object()
        o1.depends_on = ["o2"]
        o2.depends_on = ["o3"]
        o1.o2 = o2
        o2.o3 = o3
        self.assertEqual(get_owners(o1), [o1, o2, o3])

    def test_circular2(self):
        """
        Verify that a circular dependency among 2 objects is detected.
        """
        o1 = Identity()
        o2 = Identity()
        o1.depends_on = ["o2"]
        o2.depends_on = ["o1"]
        o1.o2 = o2
        o2.o1 = o1
        with self.assertRaises(RuntimeError):
            get_owners(o1)

    def test_circular3(self):
        """
        Verify that a circular dependency among 3 objects is detected.
        """
        o1 = Identity()
        o2 = Identity()
        o3 = Identity()
        o1.depends_on = ["o2"]
        o2.depends_on = ["o3"]
        o3.depends_on = ["o1"]
        o1.o2 = o2
        o2.o3 = o3
        o3.o1 = o1
        with self.assertRaises(RuntimeError):
            get_owners(o1)

    def test_circular4(self):
        """
        Verify that a circular dependency among 4 objects is detected.
        """
        o1 = Identity()
        o2 = Identity()
        o3 = Identity()
        o4 = Identity()
        o1.depends_on = ["o2"]
        o2.depends_on = ["o3"]
        o3.depends_on = ["o4"]
        o4.depends_on = ["o1"]
        o1.o2 = o2
        o2.o3 = o3
        o3.o4 = o4
        o4.o1 = o1
        with self.assertRaises(RuntimeError):
            get_owners(o1)

class DofsTests(unittest.TestCase):
    def test_no_dependents(self):
        """
        Tests for an object that does not depend on other objects.
        """
        obj = Adder(4)
        obj.set_dofs([101, 102, 103, 104])
        dofs = Dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 103, 104])
        self.assertEqual(dofs.all_owners, [obj])
        self.assertEqual(dofs.dof_owners, [obj, obj, obj, obj])
        np.testing.assert_allclose(dofs.indices, [0, 1, 2, 3])

        obj.fixed = [True, False, True, False]
        dofs = Dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [102, 104])
        self.assertEqual(dofs.all_owners, [obj])
        self.assertEqual(dofs.dof_owners, [obj, obj])
        np.testing.assert_allclose(dofs.indices, [1, 3])

        obj.fixed[0] = False
        dofs = Dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 104])
        self.assertEqual(dofs.all_owners, [obj])
        self.assertEqual(dofs.dof_owners, [obj, obj, obj])
        np.testing.assert_allclose(dofs.indices, [0, 1, 3])

    def test_no_fixed(self):
        """
        Test behavior when there is no 'fixed' attribute.
        """
        obj = Adder(4)
        del obj.fixed
        self.assertFalse(hasattr(obj, 'fixed'))
        obj.set_dofs([101, 102, 103, 104])
        dofs = Dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 103, 104])
        self.assertEqual(dofs.all_owners, [obj])
        self.assertEqual(dofs.dof_owners, [obj, obj, obj, obj])
        np.testing.assert_allclose(dofs.indices, [0, 1, 2, 3])


    def test_with_dependents(self):
        """
        Test the case in which the original object depends on another object.
        """
        o1 = Adder(3)
        o2 = Adder(4)
        o1.set_dofs([10, 11, 12])
        o2.set_dofs([101, 102, 103, 104])
        o1.depends_on = ["o2"]
        o1.o2 = o2
        dofs = Dofs([o1.J])
        np.testing.assert_allclose(dofs.x, [10, 11, 12, 101, 102, 103, 104])
        self.assertEqual(dofs.all_owners, [o1, o2])
        self.assertEqual(dofs.dof_owners, [o1, o1, o1, o2, o2, o2, o2])
        np.testing.assert_allclose(dofs.indices, [0, 1, 2, 0, 1, 2, 3])

        o1.fixed = [True, False, True]
        o2.fixed = [False, False, True, True]
        del o1.depends_on
        o2.depends_on = ["o1"]
        o2.o1 = o1
        dofs = Dofs([o2.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 11])
        self.assertEqual(dofs.all_owners, [o2, o1])
        self.assertEqual(dofs.dof_owners, [o2, o2, o1])
        np.testing.assert_allclose(dofs.indices, [0, 1, 1])

    def test_Jacobian(self):
        for n in range(1, 20):
            v1 = np.random.rand() * 4 - 2
            v2 = np.random.rand() * 4 - 2
            o = TestObject2(v1, v2)
            o.adder.set_dofs(np.random.rand(2) * 4 - 2)
            o.t.set_dofs([np.random.rand() * 4 - 2])
            o.t.adder1.set_dofs(np.random.rand(3) * 4 - 2)
            o.t.adder2.set_dofs(np.random.rand(2) * 4 - 2)
            r = Rosenbrock()
            r.set_dofs(np.random.rand(2) * 3 - 1.5)

            # Randomly fix some of the degrees of freedom
            o.fixed = np.random.rand(2) > 0.5
            o.adder.fixed = np.random.rand(2) > 0.5
            o.t.adder1.fixed = np.random.rand(3) > 0.5
            o.t.adder2.fixed = np.random.rand(2) > 0.5
            r.fixed = np.random.rand(2) > 0.5

            rtol = 1e-4
            atol = 1e-4

            for j in range(4):
                # Try different sets of the objects:
                if j==0:
                    dofs = Dofs([o.J, r.term2, o.t.J])
                elif j==1:
                    dofs = Dofs([r.term2, r.term1])
                elif j==2:
                    dofs = Dofs([r.term2, Target(o.t, 'f'), r.term1, Target(o, 'f')])
                elif j==3:
                    dofs = Dofs([o])

                jac = dofs.jac()
                fd_jac = dofs.fd_jac()
                np.testing.assert_allclose(jac, fd_jac, rtol=rtol, atol=atol)
                
                print('Diff in Jacobians:', jac - fd_jac)

if __name__ == "__main__":
    unittest.main()
