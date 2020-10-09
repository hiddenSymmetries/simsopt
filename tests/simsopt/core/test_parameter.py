import unittest
import numpy as np
from simsopt.core.parameter import *

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


    def observer1(self):
        self.observer1_called = True

    def observer2(self):
        self.observer2_called = True

    def test_observers(self):
        """
        Test the _observers attribute.
        """

        # If we do not specify any observers, _observers should be an
        # empty set.
        p = Parameter()
        self.assertEqual(p._observers, set())

        p = Parameter(3.0, fixed=False, min=-1, max=10)
        self.assertEqual(p._observers, set())

        # Try specifying 1 observer:
        p = Parameter(7, self.observer1)
        self.assertEqual(p._observers, {self.observer1})

        p = Parameter(7, {self.observer1})
        self.assertEqual(p._observers, {self.observer1})

        p = Parameter(7, observers=self.observer1)
        self.assertEqual(p._observers, {self.observer1})

        # Try specifying >1 observer:
        p = Parameter(7, {self.observer1, self.observer2})
        self.assertEqual(p._observers, {self.observer1, self.observer2})

        p = Parameter(7, observers={self.observer1, self.observer2})
        self.assertEqual(p._observers, {self.observer1, self.observer2})

        # If we specify something that is not callable and not a set,
        # an exception should be raised:
        with self.assertRaises(ValueError):
            p = Parameter(7, 5)
        with self.assertRaises(ValueError):
            p = Parameter(7.0, observers=True)

        # If we specify a set constaining anything that is not
        # callable, an exception should be raised:
        with self.assertRaises(ValueError):
            p = Parameter(7, {5})
        with self.assertRaises(ValueError):
            p = Parameter(7, observers={5})
        with self.assertRaises(ValueError):
            p = Parameter(7, {5, self.observer1})
        with self.assertRaises(ValueError):
            p = Parameter(7, observers={self.observer1, 5, self.observer2})

        # When val is changed, the observers should be called.
        # Try a case with 1 observer:
        self.observer1_called = False
        p = Parameter(7, self.observer1)
        p.val = 1
        self.assertTrue(self.observer1_called)

        # Try a case with 2 observers:
        self.observer1_called = False
        self.observer2_called = False
        p = Parameter(observers={self.observer1, self.observer2})
        p.val = 1
        self.assertTrue(self.observer1_called)
        self.assertTrue(self.observer2_called)

class ParameterArrayTests(unittest.TestCase):
    def myfunc(self):
        """
        Function for testing observers.
        """
        return 0

    def myfunc2(self):
        """
        Another function for testing observers.
        """
        return 1

    def test_basic(self):
        """
        This is the most common use case.
        """
        myshape = (4,3)
        v = np.ones(myshape)
        p = ParameterArray(v)
        self.assertEqual(p.shape, myshape)
        self.assertEqual(p.data.shape, myshape)
        # For some reason a simpler iteration of p.data does not work
        it = np.nditer(p.data, flags=['multi_index','refs_ok'])
        for x in it:
            y = p.data[it.multi_index]
            self.assertAlmostEqual(y.val, 1.0, places=13)
            self.assertTrue(y.fixed)
            self.assertEqual(y.min, np.NINF)
            self.assertEqual(y.max, np.Inf)
            self.assertEqual(y.observers, set())

        # Try modifying some individual elements:
        p.data[1,2].fixed = False
        self.assertFalse(p.data[1,2].fixed)
        p.data[1,1].min = -5
        self.assertEqual(p.data[1,1].min, -5)
        p.data[0,1].max = 10
        self.assertEqual(p.data[0,1].max, 10)

    def test_set(self):
        myshape = (2,3)
        v = np.ones(myshape)
        p = ParameterArray(v)

        # Test setting with single values:
        p.set_val(3.14)
        p.set_fixed(True)
        p.set_min(-5)
        p.set_max(10)
        p.set_name("RBC")
        p.set_observers(self.myfunc)
        it = np.nditer(p.data, flags=['multi_index','refs_ok'])
        for y in it:
            x = p.data[it.multi_index]
            self.assertAlmostEqual(x.val, 3.14, places=13)
            self.assertTrue(x.fixed)
            self.assertEqual(x.min, -5)
            self.assertEqual(x.max, 10)
            self.assertEqual(x.name, "RBC")
            self.assertEqual(x.observers, {self.myfunc})
        p.set_fixed(False)
        it = np.nditer(p.data, flags=['multi_index','refs_ok'])
        for y in it:
            x = p.data[it.multi_index]
            self.assertFalse(x.fixed)

        # Test setting with numpy arrays:
        p.set_val(np.full(myshape, 2.7))
        p.set_fixed(np.full(myshape, False))
        p.set_min(np.full(myshape, -100))
        p.set_max(np.full(myshape, 200))
        p.set_name(np.full(myshape, "ZBS"))
        p.set_observers(np.full(myshape, {self.myfunc2}))
        it = np.nditer(p.data, flags=['multi_index','refs_ok'])
        for y in it:
            x = p.data[it.multi_index]
            self.assertAlmostEqual(x.val, 2.7, places=13)
            self.assertFalse(x.fixed)
            self.assertEqual(x.min, -100)
            self.assertEqual(x.max, 200)
            self.assertEqual(x.name, "ZBS")
            self.assertEqual(x.observers, {self.myfunc2})

        # Test setting with non-numpy arrays:
        v = [[1,2,3],[4,5,6]]
        p.set_val(v)
        p.set_min(v)
        p.set_max(v)
        p.set_name(v)
        for j in range(2):
            for k in range(3):
                self.assertEqual(p.data[j,k].val, v[j][k])
                self.assertEqual(p.data[j,k].min, v[j][k])
                self.assertEqual(p.data[j,k].max, v[j][k])
                self.assertEqual(p.data[j,k].name, v[j][k])

    def test_get_variables(self):
        """
        Verify that the get_variables method extracts a set with the
        elements for which fixed=False.
        """
        vals = [1, 2, -1, -1, 5, -1, -1]
        fixeds = [True, True, False, False, True, False, False]
        p = ParameterArray(vals, fixed=fixeds)
        vars = p.get_variables()
        self.assertIs(type(vars), set)
        self.assertEqual(len(vars), 4)
        for x in vars:
            self.assertEqual(x.val, -1)

        vals = [[-1, 2], [3, -1], [-1, 6]]
        fixeds = [[False, True], [True, False], [False, True]]
        p = ParameterArray(vals, fixed=fixeds)
        vars = p.get_variables()
        self.assertIs(type(vars), set)
        self.assertEqual(len(vars), 3)
        for x in vars:
            self.assertEqual(x.val, -1)

    def test_init_singles(self):
        """
        Provide single values rather than arrays for the constructor.
        """
        v = np.full((15,), 6)
        p = ParameterArray(v, fixed=False, min=-10, max=20, name="AM", \
                               observers=self.myfunc)
        self.assertEqual(p.shape, v.shape)
        self.assertEqual(p.data.shape, v.shape)
        it = np.nditer(p.data, flags=['multi_index','refs_ok'])
        for x in it:
            y = p.data[it.multi_index]
            self.assertFalse(y.fixed)
            self.assertEqual(y.min, -10)
            self.assertEqual(y.max, 20)
            self.assertEqual(y.name, "AM")
            self.assertEqual(y.observers, {self.myfunc})

    def test_init_arrays(self):
        """
        Provide all 4 arrays for the constructor.
        """
        myshape = (4, 2)
        v = np.full(myshape, 42)
        f = np.full(myshape, False)
        mymin = np.full(myshape, -10)
        mymax = np.full(myshape, 70)
        mynames = np.full(myshape, "RBS")
        obs = np.full(myshape, {self.myfunc2})
        p = ParameterArray(v, fixed=f, min=mymin, max=mymax, name=mynames, \
                               observers=obs)
        self.assertEqual(p.shape, v.shape)
        self.assertEqual(p.data.shape, v.shape)
        it = np.nditer(p.data, flags=['multi_index','refs_ok'])
        for x in it:
            y = p.data[it.multi_index]
            self.assertEqual(y.val, 42)
            self.assertFalse(y.fixed)
            self.assertEqual(y.min, -10)
            self.assertEqual(y.max, 70)
            self.assertEqual(y.name, "RBS")
            self.assertEqual(y.observers, {self.myfunc2})

    def test_init_exceptions(self):
        """
        Test some cases in which an exception should be thrown by the
        constructor.
        """
        v = np.zeros((4, 2))

        with self.assertRaises(ValueError):
            # fixed must be None or a bool or ndarray of the proper size
            p = ParameterArray(v, fixed=7)

        #with self.assertRaises(ValueError):
        #    # min must be None or an int, float, or ndarray of the proper size
        #    p = ParameterArray(v, min=[1, 1])

        #with self.assertRaises(ValueError):
        #    # max must be None or an int, float, or ndarray of the proper size
        #    p = ParameterArray(v, max=(1, 1))

    def test_setter_exceptions(self):
        """
        Try some things that should cause the setters to raise
        exceptions.
        """
        myshape = (2, 3)
        v = np.zeros(myshape)
        p = ParameterArray(v)

        # fixed must be a bool or ndarray of the proper size
        with self.assertRaises(ValueError):
            p.set_fixed(7)
        with self.assertRaises(ValueError):
            p.set_fixed(7.0)
        with self.assertRaises(ValueError):
            p.set_fixed(np.array([1]))
        with self.assertRaises(ValueError):
            p.set_fixed(np.ones((2, 2)))

    def test_set_data(self):
        # data should be read-only
        v = (4,3)
        pa = ParameterArray(np.ones(v))
        p = Parameter()
        with self.assertRaises(AttributeError):
            pa.data = np.array([p])

    def test_from_array(self):
        p1 = Parameter(1.0)
        p2 = Parameter(2.0)
        arr = np.array([p1,p2])
        pa = ParameterArray.from_array(arr)
        self.assertAlmostEqual(pa.data[0].val, 1.0, places=13)
        self.assertAlmostEqual(pa.data[1].val, 2.0, places=13)

if __name__ == "__main__":
    unittest.main()
