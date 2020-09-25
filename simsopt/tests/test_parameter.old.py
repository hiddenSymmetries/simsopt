import unittest
import numpy as np
from mattopt.parameter import *

class ParameterTests(unittest.TestCase):
    def test_init1(self):
        """
        This is the most common use case.
        """
        v = 7.2 # Any value will do here.
        p = Parameter(v, name="phiedge")
        self.assertEqual(p.val, v)
        self.assertEqual(p.fixed, True)
        self.assertEqual(p.min, np.NINF)
        self.assertEqual(p.max, np.Inf)
        self.assertEqual(p.__repr__(), "phiedge=7.2 (fixed=True, min=-inf, max=inf)")

    def test_init2(self):
        """
        Try an int type and including finite bounds.
        """
        v = -1 # Any value will do here.
        p = Parameter(v, fixed=False, min=-5, max=10)
        self.assertEqual(p.val, v)
        self.assertEqual(p.fixed, False)
        self.assertEqual(p.min, -5)
        self.assertEqual(p.max, 10)
        self.assertEqual(p.__repr__(), "-1 (fixed=False, min=-5, max=10)")

    def test_init3(self):
        """
        Try initializing with no initial val.
        """
        p = Parameter()
        self.assertEqual(p.val, 0.0)
        self.assertEqual(p.fixed, True)
        self.assertEqual(p.min, np.NINF)
        self.assertEqual(p.max, np.Inf)

    def test_init_validation(self):
        """
        If min <= val <= max is not satisfied, the constructor should
        raise an exception.
        """
        with self.assertRaises(ValueError):
            p = Parameter(1, min=2, max=3)

        with self.assertRaises(ValueError):
            p = Parameter(1, min=-10, max=-5)

    def test_fixed_validation(self):
        """
        We should not be able to change the "fixed" attribute to
        anything other than a bool.
        """

        p = Parameter(1)
        p.fixed = True
        self.assertTrue(p.fixed)
        p.fixed = False
        self.assertFalse(p.fixed)

        with self.assertRaises(ValueError):
            p.fixed = 1

        with self.assertRaises(ValueError):
            p.fixed = 1.0

    def test_validation(self):
        """
        Check validation when we change val, min, or max after the
        Parameter is created.
        """
        p = Parameter(1, min=-10, max=10)
        p.val = 3
        self.assertEqual(p.val, 3)
        p.min = -5
        self.assertEqual(p.min, -5)
        p.max = 5
        self.assertEqual(p.max, 5)
        
        with self.assertRaises(ValueError):
            p.val = -20
        with self.assertRaises(ValueError):
            p.val = 20
        self.assertEqual(p.val, 3)
            
        with self.assertRaises(ValueError):
            p.min = 10
        self.assertEqual(p.min, -5)

        with self.assertRaises(ValueError):
            p.max = -10
        self.assertEqual(p.max, 5)


    def listener1(self):
        self.listener1_called = True

    def listener2(self):
        self.listener2_called = True

    def test_listeners(self):
        """
        Test the _listeners attribute.
        """

        # If we do not specify any listeners, _listeners should be an
        # empty set.
        p = Parameter()
        self.assertEqual(p._listeners, set())

        p = Parameter(3.0, fixed=False, min=-1, max=10)
        self.assertEqual(p._listeners, set())

        # Try specifying 1 listener:
        p = Parameter(7, self.listener1)
        self.assertEqual(p._listeners, {self.listener1})

        p = Parameter(7, {self.listener1})
        self.assertEqual(p._listeners, {self.listener1})

        p = Parameter(7, listener=self.listener1)
        self.assertEqual(p._listeners, {self.listener1})

        # Try specifying >1 listener:
        p = Parameter(7, {self.listener1, self.listener2})
        self.assertEqual(p._listeners, {self.listener1, self.listener2})

        p = Parameter(7, listener={self.listener1, self.listener2})
        self.assertEqual(p._listeners, {self.listener1, self.listener2})

        # If we specify something that is not callable and not a set,
        # an exception should be raised:
        with self.assertRaises(ValueError):
            p = Parameter(7, 5)
        with self.assertRaises(ValueError):
            p = Parameter(7.0, listener=True)

        # If we specify a set constaining anything that is not
        # callable, an exception should be raised:
        with self.assertRaises(ValueError):
            p = Parameter(7, {5})
        with self.assertRaises(ValueError):
            p = Parameter(7, listener={5})
        with self.assertRaises(ValueError):
            p = Parameter(7, {5, self.listener1})
        with self.assertRaises(ValueError):
            p = Parameter(7, listener={self.listener1, 5, self.listener2})

        # When val is changed, the listeners should be called.
        # Try a case with 1 listener:
        self.listener1_called = False
        p = Parameter(7, self.listener1)
        p.val = 1
        self.assertTrue(self.listener1_called)

        # Try a case with 2 listeners:
        self.listener1_called = False
        self.listener2_called = False
        p = Parameter(listener={self.listener1, self.listener2})
        p.val = 1
        self.assertTrue(self.listener1_called)
        self.assertTrue(self.listener2_called)

class ParameterArrayTests(unittest.TestCase):
    def test_init1(self):
        """
        This is the most common use case.
        """
        d1 = 2
        d2 = 3
        v = np.ones((d1,d2))
        p = ParameterArray(v)
        self.assertEqual(p.fixed.shape, v.shape)
        self.assertEqual(p.min.shape, v.shape)
        self.assertEqual(p.max.shape, v.shape)
        for x in np.nditer(p.fixed):
            self.assertTrue(x)
        for x in np.nditer(p.min):
            self.assertEqual(x, np.NINF)
        for x in np.nditer(p.max):
            self.assertEqual(x, np.Inf)

        p.fixed[1,2] = False
        self.assertFalse(p.fixed[1,2])
        p.min[1,1] = -5
        self.assertEqual(p.min[1,1], -5)
        p.max[0,1] = 10
        self.assertEqual(p.max[0,1], 10)

    def test_init_singles(self):
        """
        Provide single values rather than arrays for the constructor.
        """
        d1 = 4
        d2 = 2
        v = np.zeros((d1,d2))
        p = ParameterArray(v, fixed=False, min=-10, max=20)
        self.assertEqual(p.fixed.shape, v.shape)
        self.assertEqual(p.min.shape, v.shape)
        self.assertEqual(p.max.shape, v.shape)
        for x in np.nditer(p.fixed):
            self.assertFalse(x)
        for x in np.nditer(p.min):
            self.assertEqual(x, -10)
        for x in np.nditer(p.max):
            self.assertEqual(x, 20)

    def test_init_arrays(self):
        """
        Provide all 4 arrays for the constructor.
        """
        d1 = 4
        d2 = 2
        v = np.zeros((d1,d2))
        f = np.full((d1,d2), False)
        mymin = np.full((d1,d2), -10)
        mymax = np.full((d1,d2), 20)
        p = ParameterArray(v, fixed=f, min=mymin, max=mymax)
        self.assertEqual(p.fixed.shape, v.shape)
        self.assertEqual(p.min.shape, v.shape)
        self.assertEqual(p.max.shape, v.shape)
        for x in np.nditer(p.fixed):
            self.assertFalse(x)
        for x in np.nditer(p.min):
            self.assertEqual(x, -10)
        for x in np.nditer(p.max):
            self.assertEqual(x, 20)

    def test_init_exceptions(self):
        """
        Test some cases in which an exception should be thrown by the
        constructor.
        """
        v = np.zeros((4, 2))

        with self.assertRaises(ValueError):
            # fixed must be None or a bool or ndarray of the proper size
            p = ParameterArray(v, fixed=7)

        with self.assertRaises(ValueError):
            # min must be None or an int, float, or ndarray of the proper size
            p = ParameterArray(v, min=[1,1])

        with self.assertRaises(ValueError):
            # max must be None or an int, float, or ndarray of the proper size
            p = ParameterArray(v, max=(1,1))

    def test_setters(self):
        """
        Use the setters in a way that should work fine.
        """
        shape = (2,3)
        v = np.zeros(shape)
        p = ParameterArray(v)

        # Try setting to arrays:
        p.val = np.full(shape, -0.5)
        p.fixed = np.full(shape, False)
        p.min = np.full(shape, -5)
        p.max = np.full(shape, 5)
        for x in np.nditer(p.val):
            self.assertEqual(x, -0.5)
        for x in np.nditer(p.fixed):
            self.assertFalse(x)
        for x in np.nditer(p.min):
            self.assertEqual(x, -5)
        for x in np.nditer(p.max):
            self.assertEqual(x, 5)

        # Try setting to single numbers:
        p.val = 2
        p.fixed = True
        p.min = -20
        p.max = 30
        for x in np.nditer(p.val):
            self.assertEqual(x, 2)
        for x in np.nditer(p.fixed):
            self.assertTrue(x)
        for x in np.nditer(p.min):
            self.assertEqual(x, -20)
        for x in np.nditer(p.max):
            self.assertEqual(x, 30)

    def test_setter_exceptions(self):
        """
        Try some things that should cause the setters to raise
        exceptions.
        """
        shape = (2,3)
        v = np.zeros(shape)
        p = ParameterArray(v)

        # val must be an int or float or ndarray of the proper size
        with self.assertRaises(ValueError):
            p.val = [1,2]
        with self.assertRaises(ValueError):
            p.val = (9,2,3)
        with self.assertRaises(ValueError):
            p.val = np.array([1])
        with self.assertRaises(ValueError):
            p.val = np.ones((2,2))

        # fixed must be a bool or ndarray of the proper size
        with self.assertRaises(ValueError):
            p.fixed = 7
        with self.assertRaises(ValueError):
            p.fixed = 7.0
        with self.assertRaises(ValueError):
            p.fixed = np.array([1])
        with self.assertRaises(ValueError):
            p.fixed = np.ones((2,2))

        # min must be an int or float or ndarray of the proper size
        with self.assertRaises(ValueError):
            p.min = [1,2]
        with self.assertRaises(ValueError):
            p.min = (9,2,3)
        with self.assertRaises(ValueError):
            p.min = np.array([1])
        with self.assertRaises(ValueError):
            p.min = np.ones((2,2))

        # max must be an int or float or ndarray of the proper size
        with self.assertRaises(ValueError):
            p.max = [1,2]
        with self.assertRaises(ValueError):
            p.max = (9,2,3)
        with self.assertRaises(ValueError):
            p.max = np.array([1])
        with self.assertRaises(ValueError):
            p.max = np.ones((2,2))

if __name__ == "__main__":
    unittest.main()
