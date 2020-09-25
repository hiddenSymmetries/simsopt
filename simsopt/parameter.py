#!/usr/bin/env python3

"""
This module contains classes related to parameters in simsopt.  A
Parameter is a value that has the potential to be varied in an
optimization.  Sometimes however the value may also be held fixed. If
the value is varied, there may be box constraints, also known as bound
constraints, i.e. upper and lower bounds.

This module contains the Parameter class, which stores a single value,
and the ParameterArray class, which stores arbitrary-dimension arrays
of Parameters.
"""

import numpy as np

def isbool(val):
    """
    Test whether val is any boolean type, either the native python
    bool or numpy's bool_.
    """
    return isinstance(val, bool) or isinstance(val, np.bool_)

def isnumber(val):
    """
    Test whether val is any kind of number, including both native
    python types or numpy types.
    """
    return isinstance(val, int) or isinstance(val, float) or \
        isinstance(val, np.int_) or isinstance(val, np.float)

class Parameter:
    """
    This class represents a value that has the potential to be varied
    in an optimization, though sometime it may also be held
    fixed. This class has private variables _val, _fixed, _min, and
    _max. For each of these variables there is a public "property":
    val, fixed, min, and max. By using the @property decorator it is
    possible to do some validation any time a user attempts to change
    the attributes.

    The instance variables val, min, and max can be any type, not just
    float. This is important because we may want parameters that have
    type int, bool, complex, or something more exotic.
    """
    def __init__(self, val=0.0, observers=None, fixed=True, min=np.NINF, \
                     max=np.Inf, name=None):
        """
        Constructor. observer can be None, or a single callable, or a
        set of callables.
        """
        self._val = val
        self._fixed = fixed
        self._min = min
        self._max = max
        self.verify_bounds()
        self.name = name
        # Initialize _observers to be a set of all observers
        if observers is None:
            self._observers = set()
        elif callable(observers):
            self._observers = {observers}
        elif type(observers) is set:
            for s in observers:
                if not callable(s):
                    raise ValueError("observers must be None, a callable, or " \
                                         + "a set of callable objects.")
            self._observers = observers
        else:
            raise ValueError("observers must be None, a callable, or a set " \
                                 + "of callable objects.")

    # When "val", "min", or "max" is altered by a user, we should
    # check that val is indeed in between min and max.

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, newval):
        self.verify_bounds(val=newval)
        self._val = newval
        # Update all objects that observe this Parameter:
        for observers in self._observers:
            observers()

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, newmin):
        self.verify_bounds(min=newmin)
        self._min = newmin

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, newmax):
        self.verify_bounds(max=newmax)
        self._max = newmax

    # When "fixed" is changed, we do not need to verify the bounds,
    # but we do want to ensure that "fixed" has type bool.
    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        if not isbool(value):
            raise ValueError(
                "fixed attribute of a Parameter must have type bool.")
        self._fixed = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, newname):
        # At some point we may want to force name to be a str, but for
        # now no validation is done.
        self._name = newname

    @property
    def observers(self):
        return self._observers

    @observers.setter
    def observers(self, newobservers):
        errmsg = 'observers must be a set of callables'
        if type(newobservers) is not set:
            raise ValueError(errmsg)
        for x in newobservers:
            if not callable(x):
                raise ValueError(errmsg)
        self._observers = newobservers

    def __repr__(self):
        """
        Print the object in an informative way.
        """
        if self.name is None:
            namestr = ""
        else:
            namestr = str(self.name) + "="

        return namestr + str(self._val) + ' (fixed=' + str(self._fixed) \
            + ', min=' + str(self._min) + ', max=' + str(self._max) + ')'

    def verify_bounds(self, val=None, min=None, max=None):
        """
        Check that the value, lower bound, and upper bound are
        consistent. If no arguments are supplied, the method checks
        the private variables of this instance. The method can also
        check potential new values for val, min, or max, via optional
        arguments.
        """
        if val is None:
            val = self._val
        if min is None:
            min = self._min
        if max is None:
            max = self._max

        if min > max:
            raise ValueError("Parameter has min > max. " +
                               "min = " + str(min) +
                               ", max = " + str(max))
        if val < min:
            raise ValueError("Parameter has val < min. " +
                               "val = " + str(val) +
                               ", min = " + str(min))
        if val > max:
            raise ValueError("Parameter has val > max. " +
                               "val = " + str(val) +
                               ", max = " + str(max))


class ParameterArrayOld:
    """
    This class stores arrays of parameters. However instead of storing
    a list or numpy.ndarray in which the elements are from the class
    Parameter, here instead we have a Parameter-like class in which
    the elements have type numpy.ndarray. The reason is that we want
    to allow ndarray slicing syntax, e.g.
    p = ParameterArray(numpy.zeros(10, 20))
    p.fixed[2:3, 5:10] = True

    As with a Parameter, a ParameterArray instance has 4 private
    variables: _val, _fixed, _min, and _max. For each of these
    variables there is a public "property": val, fixed, min, and
    max. By using the @property decorator it is possible to do some
    validation any time a user attempts to change the attributes.

    Presently there is no checking that min <= val <= max or that
    fixed has type bool. For the first of these, I don't know how to
    automatically validate when a user changes specific ndarray
    elements.

    Presently it is not possible to resize or reshape a
    ParameterArray. This is awkward because it would require changing
    the shape/size of all 4 member arrays (val, fixed, min, max). If
    you tried changing one of those 4 arrays at a time, it would not
    possible to distinguish correct usage from incorrectly changing
    the size of one array without changing the other 3.
    """
    def __init__(self, val, fixed=None, min=None, max=None):
        if not isinstance(val, np.ndarray):
            raise ValueError("val must have type numpy.ndarray")
        self._val = val

        # Handle the attribute "fixed"
        if fixed is None:
            # Create an array of the same shape as "val" populated
            # with "true" entries
            self._fixed = np.full(self._val.shape, True)
        elif isbool(fixed):
            # A single bool was provided. Copy it to all entries.
            self._fixed = np.full(self._val.shape, fixed)
        elif not isinstance(fixed, np.ndarray):
            raise ValueError( \
                "fixed must either be None or have type bool or numpy.ndarray")
        elif fixed.shape == self._val.shape:
            self._fixed = fixed
        else:
            raise ValueError("fixed has a different shape than val")
        # I should also verify that the type is bool.

        # Handle the attribute "min"
        if min is None:
            # Create an array of the same shape as "val" populated
            # with np.NINF entries
            self._min = np.full(self._val.shape, np.NINF)
        elif isinstance(min, int) or isinstance(min, float):
            # A single number was provided. Copy it to all entries.
            self._min = np.full(self._val.shape, min)
        elif not isinstance(min, np.ndarray):
            raise ValueError( \
                "min must either be None or have type int, " \
                + "float, or numpy.ndarray")
        elif min.shape == self._val.shape:
            self._min = min
        else:
            raise ValueError("min has a different shape than val")
        # I should also verify that the type is float.

        # Handle the attribute "max"
        if max is None:
            # Create an array of the same shape as "val" populated
            # with np.Inf entries
            self._max = np.full(self._val.shape, np.Inf)
        elif isinstance(max, int) or isinstance(max, float):
            # A single number was provided. Copy it to all entries.
            self._max = np.full(self._val.shape, max)
        elif not isinstance(max, np.ndarray):
            raise ValueError( \
                "max must either be None or have type int, " \
                + "float, or numpy.ndarray")
        elif max.shape == self._val.shape:
            self._max = max
        else:
            raise ValueError("max has a different shape than val")
        # I should also verify that the type is float.

    def __repr__(self): 
        """
        Print the object in an informative way.
        """
        s = "--- ParameterArray" + str(self._val.shape) + " ---"
        s += "\nval:\n" + self._val.__repr__()
        s += "\nfixed:\n" + self._fixed.__repr__()
        s += "\nmin:\n" + self._min.__repr__()
        s += "\nmax:\n" + self._max.__repr__()
        return s

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, newval):
        if isinstance(newval, int) or isinstance(newval, float):
            newval = np.full(self._val.shape, newval)
        elif not isinstance(newval, np.ndarray):
            raise ValueError("val must have type numpy.ndarray.")
        elif newval.shape != self._val.shape:
            raise ValueError("Shape and size of the val array cannot be " \
              "changed. old shape=" + str(self._val.shape) + " new=" \
              + str(newval.shape))
        #self.verify_bounds(val=newval)
        self._val = newval

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, newfixed):
        if isbool(newfixed):
            newfixed = np.full(self._val.shape, newfixed)
        elif not isinstance(newfixed, np.ndarray):
            raise ValueError("fixed must have type numpy.ndarray.")
        elif newfixed.shape != self._fixed.shape:
            raise ValueError("Shape and size of the fixed array cannot be " \
              "changed. old shape=" + str(self._fixed.shape) + " new=" \
              + str(newfixed.shape))
        #self.verify_bounds(fixed=newfixed)
        self._fixed = newfixed

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, newmin):
        if isinstance(newmin, int) or isinstance(newmin, float):
            newmin = np.full(self._val.shape, newmin)
        elif not isinstance(newmin, np.ndarray):
            raise ValueError("min must have type numpy.ndarray.")
        elif newmin.shape != self._min.shape:
            raise ValueError("Shape and size of the min array cannot be " \
              "changed. old shape=" + str(self._min.shape) + " new=" \
              + str(newmin.shape))
        #self.verify_bounds(min=newmin)
        self._min = newmin

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, newmax):
        if isinstance(newmax, int) or isinstance(newmax, float):
            newmax = np.full(self._val.shape, newmax)
        elif not isinstance(newmax, np.ndarray):
            raise ValueError("max must have type numpy.ndarray.")
        elif newmax.shape != self._max.shape:
            raise ValueError("Shape and size of the max array cannot be " \
              "changed. old shape=" + str(self._max.shape) + " new=" \
              + str(newmax.shape))
        #self.verify_bounds(max=newmax)
        self._max = newmax

class ParameterArray:
    """
    A ParameterArray is essentially a numpy.ndarray with elements of
    type Parameter.  This class provides some additional methods for
    convenience, such as methods for setting or getting the attributes
    of the individual Parameters, and getting the set of all non-fixed
    elements.

    If type(name) is not a numpy.ndarray of the same shape as val,
    whatever name is supplied will be replicated for all Parameters.
    """
    def __init__(self, val=np.array([0.0]), observers=None, fixed=True, \
                     min=np.NINF, max=np.Inf, name=None):
        """
        Constructor. For any argument that is None or a single value,
        it will be converted to a numpy ndarray of the same size as
        val.
        """
        try:
            val = np.array(val)
        except:
            raise ValueError("val must be convertible to numpy.ndarray")

        # Handle min
        if not isinstance(min, np.ndarray):
            min = np.full(val.shape, min)
        if min.shape != val.shape:
            raise ValueError( \
                "Shape of min does not match shape of val.")
 
        # Handle max
        if not isinstance(max, np.ndarray):
            max = np.full(val.shape, max)
        if max.shape != val.shape:
            raise ValueError( \
                "Shape of max does not match shape of val.")

        # Handle name
        if type(name) is not np.ndarray:
            name = np.full(val.shape, name)
        if name.shape != val.shape:
            raise ValueError("Shape of name does not match shape of val.")

        # Verify fixed
        errstr = "fixed must be None, a bool, or " \
            + "convertable to a numpy ndarray of bools."
        if fixed is None or isbool(fixed):
            fixed = np.full(val.shape, fixed)
        else:
            try:
                fixed = np.array(fixed)
            except:
                raise ValueError(errstr)
            if fixed.shape != val.shape:
                raise ValueError( \
                    "Shape of fixed does not match shape of val.")
            # Ensure the first element has the proper type
            x = fixed.flat[0]
            if not isbool(x):
                raise ValueError(errstr + " type=" + str(type(x)))

        # Verify observers
        errstr = "observers must be none, a callable, or " \
            + "convertable to a numpy ndarray of callables."
        if observers is None:
            # Set observers to the empty set.
            observers = np.full(val.shape, set())
        elif callable(observers):
            observers = np.full(val.shape, {observers})
        elif type(observers) is set:
            # Verify each element in the set is callable
            for x in observers:
                if not callable(x):
                    raise ValueError( \
                    "Each element in the observers set must be callable.")
            observers = np.full(val.shape, observers)
        else:
            try:
                observers = np.array(observers)
            except:
                raise ValueError(errstr)
            if observers.shape != val.shape:
                raise ValueError( \
                    "Shape of observers does not match shape of val.")
            # Ensure every element is a set of callables:
            it = np.nditer(observers, flags=['multi_index','refs_ok'])
            for x in it:
                y = observers[it.multi_index]
                if type(y) is not set:
                    raise ValueError(errstr)
                for z in y:
                    if not callable(z):
                        raise ValueError(errstr)

        # At this point, val, fixed, min, max, name, and observers
        # should all be ndarrays of the same shape.
        assert(val.shape == fixed.shape)
        assert(val.shape == min.shape)
        assert(val.shape == max.shape)
        assert(val.shape == name.shape)
        assert(val.shape == observers.shape)

        # Now build the _data array. Start with a dummy array, then
        # over-write the elements with the real Parameters.
        temp_param = Parameter()
        self._data = np.full(val.shape, temp_param)
        # See https://numpy.org/devdocs/reference/arrays.nditer.html
        with np.nditer(self._data, flags=['multi_index','refs_ok'], op_flags=['writeonly']) as it:
            for x in it:
                x[...] = Parameter(val=val[it.multi_index], \
                                       fixed=fixed[it.multi_index], \
                                       min=min[it.multi_index], \
                                       max=max[it.multi_index], \
                                       name=name[it.multi_index], \
                                       observers=observers[it.multi_index])

    @property
    def data(self):
        """
        Public method to get the array of Parameters.
        """
        return self._data

    @property
    def shape(self):
        """
        Return the shape of the data array.
        """
        return self._data.shape

    @classmethod
    def from_array(cls, arr):
        """
        Build a ParameterArray given an array of Parameters. The
        argument arr must be convertable to a numpy ndarray and all
        the elements must have type Parameter.
        """
        try:
            arr = np.array(arr)
        except:
            raise RuntimeError('arr must be convertable to numpy.ndarray')
        it = np.nditer(arr, flags=['multi_index','refs_ok'])
        for x in it:
            if type(arr[it.multi_index]) is not Parameter:
                raise ValueError('Elements of arr must have type Parameter')

        # Start with a dummy ParameterArray:
        pa = cls([0.0])
        # Now replace the dummy data with the actual data:
        pa._data = arr
        return pa

    def set_val(self, val):
        """
        Over-write the val attribute of all the Parameters. If val
        does not have type numpy.ndarray, then a ndarray will be used
        in which each element is val. If an ndarray is specified, the
        shape must match that of the original ParameterArray.
        """
        # Handle the case of a single (non-array) argument:
        if not isinstance(val, np.ndarray):
            val = np.full(self._data.shape, val)
        if val.shape != self._data.shape:
            raise ValueError("Shape of val does not match shape of this " \
                                 " ParameterArray")

        # Now write the values
        it = np.nditer(self._data, flags=['multi_index','refs_ok'])
        for x in it:
            self._data[it.multi_index].val = val[it.multi_index]

    def set_min(self, min):
        """
        Over-write the min attribute of all the Parameters. If min
        does not have type numpy.ndarray, then a ndarray will be used
        in which each element is min. If an ndarray is specified, the
        shape must match that of the original ParameterArray.
        """
        # Handle the case of a single (non-array) argument:
        if not isinstance(min, np.ndarray):
            min = np.full(self._data.shape, min)
        if min.shape != self._data.shape:
            raise ValueError("Shape of min does not match shape of this " \
                                 " ParameterArray")

        # Now write the values
        it = np.nditer(self._data, flags=['multi_index','refs_ok'])
        for x in it:
            self._data[it.multi_index].min = min[it.multi_index]

    def set_max(self, max):
        """
        Over-write the max attribute of all the Parameters. If max
        does not have type numpy.ndarray, then a ndarray will be used
        in which each element is max. If an ndarray is specified, the
        shape must match that of the original ParameterArray.
        """
        # Handle the case of a single (non-array) argument:
        if not isinstance(max, np.ndarray):
            max = np.full(self._data.shape, max)
        if max.shape != self._data.shape:
            raise ValueError("Shape of max does not match shape of this " \
                                 " ParameterArray")

        # Now write the values
        it = np.nditer(self._data, flags=['multi_index','refs_ok'])
        for x in it:
            self._data[it.multi_index].max = max[it.multi_index]

    def set_name(self, name):
        """
        Over-write the name attribute of all the Parameters. If name
        does not have type numpy.ndarray, then a ndarray will be used
        in which each element is name. If an ndarray is specified, the
        shape must match that of the original ParameterArray.
        """
        # Handle the case of a single (non-array) argument:
        if not isinstance(name, np.ndarray):
            name = np.full(self._data.shape, name)
        if name.shape != self._data.shape:
            raise ValueError("Shape of name does not match shape of this " \
                                 " ParameterArray")

        # Now write the values
        it = np.nditer(self._data, flags=['multi_index','refs_ok'])
        for x in it:
            self._data[it.multi_index].name = name[it.multi_index]

    def set_fixed(self, fixed):
        """
        Over-write the fixed attribute of all the Parameters. The
        specified fixed can be a bool, or a numpy.ndarray of bools of
        the same shape as the original ParameterArray.
        """
        # Validate:
        errmsg = "fixed must be a bool, or a numpy.ndarray of bools" \
            " of the same shape as the ParameterArray."
        if isbool(fixed):
            fixed = np.full(self._data.shape, fixed)
        elif isinstance(fixed, np.ndarray):
            if fixed.shape != self._data.shape:
                raise ValueError("Shape of fixed does not match shape of this " \
                                     " ParameterArray")
            it = np.nditer(fixed, flags=['multi_index'])
            for x in it:
                if not isbool(fixed[it.multi_index]):
                    raise ValueError(errmsg)
        else:
            raise ValueError(errmsg)

        # Now write the values
        it = np.nditer(self._data, flags=['multi_index','refs_ok'])
        for x in it:
            self._data[it.multi_index].fixed = fixed[it.multi_index]

    def set_observers(self, observers):
        """
        Over-write the observers attribute of all the Parameters. The
        specified observers can be a callable, a set of callables, or
        a numpy.ndarray (of sets of callables) with the same shape as
        the original ParameterArray.
        """
        # Validate:
        errmsg = "observers must be a callable, a set of callables, or a " \
            + "numpy.ndarray of sets of callables of the same shape as the " \
            + "ParameterArray."
        if isinstance(observers, np.ndarray):
            if observers.shape != self._data.shape:
                raise ValueError("Shape of observers does not match shape of" \
                                     " this ParameterArray")
        elif isinstance(observers, set):
            # Make sure every element of the set is callable:
            for x in observers:
                if not callable(x):
                    raise ValueError(errmsg)
            observers = np.full(self._data.shape, observers)
        elif callable(observers):
            observers = np.full(self._data.shape, {observers})
        else:
            raise ValueError(errmsg)

        # Now write the values
        it = np.nditer(self._data, flags=['multi_index','refs_ok'])
        for x in it:
            self._data[it.multi_index].observers = observers[it.multi_index]

    def get_val(self):
        """
        Extract val from each parameter and return the results in a
        numpy.ndarray of the same shape as the original array of
        Parameters.
        """
        # Create an array of the same type as val of the first element:
        arr = np.full(self._data.shape, self._data.flat[0].val)
        # Over-write with the actual vals:
        with np.nditer(arr, flags=['multi_index'], op_flags=['writeonly']) as it:
            for x in it:
                x[...] = self._data[it.multi_index].val
        return arr

    def get_variables(self):
        """
        Return a set containing all the array elements for which the
        'fixed' attribute is False. These are the variables that would
        be used for optimization.
        """

        vars = set()
        it = np.nditer(self._data, flags=['multi_index','refs_ok'])
        for x in it:
            if not self._data[it.multi_index].fixed:
                vars.add(self._data[it.multi_index])

        return vars

        # The next line fails with 
        # AttributeError: 'numpy.ndarray' object has no attribute 'fixed'
        #return {x for x in self._data if not x.fixed}
