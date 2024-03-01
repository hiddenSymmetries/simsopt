import unittest

import numpy as np

from simsopt._core.util import isnumber, isbool, unique, \
    ObjectiveFailure, finite_difference_steps, nested_lists_to_array


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


class FiniteDifferenceStepsTests(unittest.TestCase):
    def test_abs_step(self):
        """
        Test situations in which abs_step should dominate.
        """
        N = 5
        x = np.zeros(N)
        abs_step = 1.0e-5
        np.testing.assert_allclose(finite_difference_steps(x,
                                                           abs_step=abs_step,
                                                           rel_step=0),
                                   np.full(N, abs_step))

        x = np.array([0.22703548, -0.09694371, 0.36863702,
                      -0.46309711, 0.164472, -0.08652083])
        np.testing.assert_allclose(finite_difference_steps(x,
                                                           abs_step=abs_step,
                                                           rel_step=0),
                                   np.full(6, abs_step))

    def test_rel_step(self):
        """
        Test situations in which rel_step should dominate.
        """
        x = np.array([0.1165368, -0.92130271, 1.09012617, 0.71410455,
                      -0.77997291, -0.69077027, -1.51686311])

        rel_step = 1.0e-5
        np.testing.assert_allclose(finite_difference_steps(x,
                                                           abs_step=0,
                                                           rel_step=rel_step),
                                   np.abs(x) * rel_step)

    def test_mixed(self):
        """
        Test situations in which rel_step should dominate for some elements
        while abs_step should dominate for others.
        """
        x = np.array([0.1165368, -0.92130271, 1.09012617, 0.71410455,
                      -0.77997291, -0.69077027, -1.51686311])

        x0 = np.array([0.08, 0.092130271, 0.109012617, 0.08,
                       0.08, 0.08, 0.151686311])

        np.testing.assert_allclose(finite_difference_steps(x,
                                                           abs_step=0.08,
                                                           rel_step=0.1),
                                   x0)

        x = np.array([0, 1.21923073, 0.85723015, -0.08222701,
                      0, -0.9645991, 0, 0])

        x0 = np.array([1e-7, 1.21923073e-3, 0.85723015e-3, 0.08222701e-3,
                       1e-7, 0.9645991e-3, 1e-7, 1e-7])

        np.testing.assert_allclose(finite_difference_steps(x,
                                                           abs_step=1.0e-7,
                                                           rel_step=1.0e-3),
                                   x0)

    def test_zero_step(self):
        """
        Make sure we handle the case of a step of size 0 arising when
        abs_step is 0 and an element of x is 0.
        """
        x = np.array([0.22703548, -0.09694371, 0.36863702,
                      -0.46309711, 0, -0.08652083])

        with self.assertRaises(ValueError):
            finite_difference_steps(x, abs_step=0, rel_step=1e-3)
        with self.assertRaises(ValueError):
            finite_difference_steps(x, abs_step=0, rel_step=0)

        x = np.zeros(1)
        with self.assertRaises(ValueError):
            finite_difference_steps(x, abs_step=0, rel_step=1e-3)
        with self.assertRaises(ValueError):
            finite_difference_steps(x, abs_step=0, rel_step=0)


class NestedListsToArrayTests(unittest.TestCase):
    def test_nested_lists_to_array(self):
        """
        Test the utility function used to convert 2D data in a fortran
        namelist extracted by f90nml to a 2D numpy array.
        """
        list_of_lists = [[42]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 0, 0],
                         [1, 2, 3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[None, 42], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[0, 42, 0],
                         [1, 2, 3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42, 43, 44], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 43, 44],
                         [1, 2, 3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42, 43, 44, 45], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 43, 44, 45],
                         [1, 2, 3, 0]])
        np.testing.assert_allclose(arr1, arr2)


if __name__ == "__main__":
    unittest.main()
