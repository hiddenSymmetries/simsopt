import unittest
import numpy as np
from simsopt.collect_dofs import get_owners, collect_dofs
from simsopt.util import Identity, Adder

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
        o1.depends_on = [o2]
        self.assertEqual(get_owners(o1), [o1, o2])

        o3 = object()
        o1.depends_on = [o3, o2]
        self.assertEqual(get_owners(o1), [o1, o3, o2])
        
    def test_depth_2(self):
        """
        Check cases in which the original object depends on another, which
        depends on another.
        """
        o1 = Identity()
        o2 = Identity()
        o3 = object()
        o1.depends_on = [o2]
        o2.depends_on = [o3]
        self.assertEqual(get_owners(o1), [o1, o2, o3])

    def test_circular2(self):
        """
        Verify that a circular dependency among 2 objects is detected.
        """
        o1 = Identity()
        o2 = Identity()
        o1.depends_on = [o2]
        o2.depends_on = [o1]
        with self.assertRaises(RuntimeError):
            get_owners(o1)

    def test_circular3(self):
        """
        Verify that a circular dependency among 3 objects is detected.
        """
        o1 = Identity()
        o2 = Identity()
        o3 = Identity()
        o1.depends_on = [o2]
        o2.depends_on = [o3]
        o3.depends_on = [o1]
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
        o1.depends_on = [o2]
        o2.depends_on = [o3]
        o3.depends_on = [o4]
        o4.depends_on = [o1]
        with self.assertRaises(RuntimeError):
            get_owners(o1)

class CollectDofsTests(unittest.TestCase):
    def test_no_dependents(self):
        """
        Tests for an object that does not depend on other objects.
        """
        obj = Adder(4)
        obj.set_dofs([101, 102, 103, 104])
        dofs = collect_dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 103, 104])
        self.assertEqual(dofs.owners, [obj, obj, obj, obj])
        np.testing.assert_allclose(dofs.indices, [0, 1, 2, 3])

        obj.fixed = [True, False, True, False]
        dofs = collect_dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [102, 104])
        self.assertEqual(dofs.owners, [obj, obj])
        np.testing.assert_allclose(dofs.indices, [1, 3])

        obj.fixed[0] = False
        dofs = collect_dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 104])
        self.assertEqual(dofs.owners, [obj, obj, obj])
        np.testing.assert_allclose(dofs.indices, [0, 1, 3])

    def test_no_fixed(self):
        """
        Test behavior when there is no 'fixed' attribute.
        """
        obj = Adder(4)
        del obj.fixed
        self.assertFalse(hasattr(obj, 'fixed'))
        obj.set_dofs([101, 102, 103, 104])
        dofs = collect_dofs([obj.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 103, 104])
        self.assertEqual(dofs.owners, [obj, obj, obj, obj])
        np.testing.assert_allclose(dofs.indices, [0, 1, 2, 3])


    def test_with_dependents(self):
        """
        Test the case in which the original object depends on another object.
        """
        o1 = Adder(3)
        o2 = Adder(4)
        o1.set_dofs([10, 11, 12])
        o2.set_dofs([101, 102, 103, 104])
        o1.depends_on = [o2]
        dofs = collect_dofs([o1.J])
        np.testing.assert_allclose(dofs.x, [10, 11, 12, 101, 102, 103, 104])
        self.assertEqual(dofs.owners, [o1, o1, o1, o2, o2, o2, o2])
        np.testing.assert_allclose(dofs.indices, [0, 1, 2, 0, 1, 2, 3])

        o1.fixed = [True, False, True]
        o2.fixed = [False, False, True, True]
        del o1.depends_on
        o2.depends_on = [o1]
        dofs = collect_dofs([o2.J])
        np.testing.assert_allclose(dofs.x, [101, 102, 11])
        self.assertEqual(dofs.owners, [o2, o2, o1])
        np.testing.assert_allclose(dofs.indices, [0, 1, 1])

if __name__ == "__main__":
    unittest.main()
