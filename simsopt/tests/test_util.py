import unittest
import numpy as np
from simsopt.util import *

class IsboolTests(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(isbool(True))
        self.assertTrue(isbool(False))
        self.assertFalse(isbool("howdy"))
        a = np.array([True, False])
        b = np.array([1,2,3])
        self.assertTrue(isbool(a[0]))
        self.assertFalse(isbool(b[0]))

class IsnumberTests(unittest.TestCase):
    def test_basic(self):
        # Try some arguments that ARE numbers:
        self.assertTrue(isnumber(5))
        self.assertTrue(isnumber(5.0))
        a = np.array([1])
        b = np.array([1.0])
        self.assertTrue(isnumber(a[0]))
        self.assertTrue(isnumber(b[0]))

        # Try some arguments that are NOT numbers:
        self.assertFalse(isnumber("foo"))
        self.assertFalse(isnumber(object))
        self.assertFalse(isnumber([1,2,3]))

class IdentityTests(unittest.TestCase):
    def test_basic(self):
        iden = Identity()
        self.assertAlmostEqual(iden.f(), 0, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([0.0]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

        x = 3.5
        iden = Identity(x)
        self.assertAlmostEqual(iden.f(), x, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([x]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))

        y = -2
        iden.set_dofs([y])
        self.assertAlmostEqual(iden.f(), y, places=13)
        np.testing.assert_allclose(iden.get_dofs(), np.array([y]))
        np.testing.assert_allclose(iden.fixed, np.array([False]))
        
