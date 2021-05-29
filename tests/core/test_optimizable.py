import unittest
import numpy as np
from simsopt._core.optimizable import Optimizable
from simsopt.objectives.functions import Adder


class OptimizableTests(unittest.TestCase):
    def test_index(self):
        """
        Test Optimizable.index()
        """
        o = Optimizable()
        o.names = ['foo', 'bar']
        self.assertEqual(o.index('foo'), 0)
        self.assertEqual(o.index('bar'), 1)
        # If the string does not match any name, raise an exception
        with self.assertRaises(ValueError):
            o.index('zig')

    def test_get_set(self):
        """
        Test Optimizable.set() and Optimizable.get()
        """
        o = Adder(4)
        o.names = ['foo', 'bar', 'gee', 'whiz']
        o.set('gee', 42)
        self.assertEqual(o.get('gee'), 42)
        o.set('foo', -12)
        self.assertEqual(o.get('foo'), -12)
        np.testing.assert_allclose(o.get_dofs(), [-12, 0, 42, 0])

    def test_get_set_fixed(self):
        """
        Test Optimizable.set_fixed() and Optimizable.get_fixed()
        """
        o = Adder(5)
        o.names = ['foo', 'bar', 'gee', 'whiz', 'arf']
        self.assertFalse(o.get_fixed('gee'))
        o.set_fixed('gee')
        self.assertTrue(o.get_fixed('gee'))
        o.set_fixed('gee', False)
        self.assertFalse(o.get_fixed('gee'))


if __name__ == "__main__":
    unittest.main()
