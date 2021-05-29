import unittest
import numpy as np
from simsopt._core.util import *


class IsboolTests(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(isbool(True))
        self.assertTrue(isbool(False))
        self.assertFalse(isbool("howdy"))
        a = np.array([True, False])
        b = np.array([1, 2, 3])
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
        self.assertFalse(isnumber([1, 2, 3]))


class UniqueTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(unique([]), [])
        self.assertEqual(unique([5]), [5])
        self.assertEqual(unique([5, 5]), [5])
        self.assertEqual(unique([1, -3, 7, 2]), [1, -3, 7, 2])
        self.assertEqual(unique([1, -3, 7, 2, 1, -3, 7]), [1, -3, 7, 2])
