Optimizable objects and objective functions
===========================================

Optimizable objects
-------------------

Simsopt is able to optimize any python object that has a
``get_dofs()`` function and a ``set_dofs()`` function.  The overall
objective function can be defined using any function, attribute, or
@property of such an object.  Optimizable objects can depend on other
optimizable objects, so objects can be part of an optimization even if
they do not directly own a function that is part of the overall
objective function.


Specifying functions that go into the objective function
--------------------------------------------------------

Suppose we want to solve a least-squares optimization problem in which
an object ``obj`` is optimized. If ``obj`` has a function ``func()``,
we can use a 3-element tuple or list::

  term1 = (obj.func, goal, weight)

to represent a term ``weight * ((obj.func() - goal) ** 2)`` in the
least-squares objective function.
In this example, ``func()`` could return a scalar, or it could return
a 1D numpy array. In the latter case, ``sum(weight * ((obj.func() -
goal) ** 2))`` would be included in the objective function, and
``goal`` could be either a scalar or a 1D numpy array of the same
length as that returned by ``func()``.

Similarly, we can define additional terms::

  term2 = (obj2.fn, goal2, weight2)


The total least-squares objective function is created using a list or
tuple of all the terms that are to be added together::

  prob = LeastSquaresProblem.from_tuples((term1, term2))

The sequence can include any mixture of terms defined by scalar functions
and by 1D numpy array-valued functions. The problem can be defined
in an alternative way without defining the intermediate names
``term1``, ``term2``, etc::
  
  prob = LeastSquaresProblem([goal1, goal2, goal3],
                             [weight1, weight2, weight3],
                             [obj1.fn1, obj2.fn2, obj3.fn3])

If you prefer, you can specify
``sigma = 1 / sqrt(weight)`` rather than ``weight`` and use the
``LeastSquaresProblem.from_sigma``  as::

  prob = LeastSquaresProblem.from_sigma([goal1, goal2, goal3],
                                        [sigma1, sigma2, sigma3],
                                        [obj1.fn1, obj2.fn2, obj3.fn3])


Degrees of freedom ("dofs")
---------------------------

Any optimizable object can define a function ``get_dofs(self)`` that
returns a 1D numpy array of floats. (Simsopt will not allow integer
optimization.)  Any optimizable object also can have a function
``set_dofs(self, x)`` that accepts ``x``, a 1D numpy array of
floats. Some of the degrees of freedom may be fixed in the
optimization - see below.

It can be convenient to have a descriptive string associated with each
dof, e.g. ``"phiedge"`` or ``"rc(1,0)"``. To supply these names, add a
``names`` attribute to your optimizable object which is a list of
strings, where the length of the list matches the size of the
``get_dofs()`` array. It is not necessary to have a ``names`` list; if
one is not present, simsopt will use as a default ``["x[0]", "x[1]",
...]``.

Each dof is owned by a single object. In other words, if a physical
dof is represented by an array element for ``get_dofs()`` and
``set_dofs()`` of a particular object, there should be no other object
that has that physical degree of freedom an an element in its
``get_dofs()`` and ``set_dofs()`` arrays. If two objects both depend
on a given physical degree of freedom, that information should be
represented using the ``depends_on`` attribute, described below.

In addition to the "local" dofs owned by each object, an important
concept is the set of "global" dofs associated with an optimization
problem. The global dofs are the set of all non-fixed dofs of each
optimized object and all objects upon which they depend. (Dependencies
among objects are discussed in a section below.)  The global dofs
correspond to the state vector passed to the optimization algorithm.

At each function evaluation during an optimization, simsopt will call
the ``recompute_bell()`` function of each object involved in the
optimization problem,  if the local dofs owned
by that object have changed or if the output of an object it depends on
has changed.   It is the responsibility of
each object to implement the ``recompute_bell`` method  to
update expensive calculations.


Helpful functions
-----------------

Simsopt can provide several helpful functions to your optimizable
object. For instance, functions can be provided to help with fixing
certain degrees of freedom, as discussed below. There is also the
method ``index(str)`` which gets the index into the dof array
associated with a dof name ``str``, and the methods ``get(str)`` and
``set(str, val)`` which get and set a dof associated with the name
``str``.

There are two ways to equip your object with these functions. One way
is to inherit from the :obj:``simsopt.Optimizable`` class. Or, if you do
not want your class to depend on simsopt, you can use the function
:func:``simsopt.make_optimizable()`` to your objective function.


Fixing degrees of freedom
-------------------------

Sometimes we may want to vary a particular dof in an optimization, and
other times we may want to hold that same dof fixed.  One example is
the current in a coil. To enable this flexibility, every dof in
simsopt is considered to be either fixed or not.  Only the non-fixed
dofs are included in the optimization. Whether or not a
dof is fixed can be identified by the ``dofs_free_status`` attribute of the
object. The ``dofs_free_status`` attribute is a boolean
numpy array, with each element True or False. You can also query the free/fixed
status of individual status by :meth:``is_free`` or :meth:``is_fixed`` methods by using the array index
of the dof or by using the name of the dof as key.

There are several ways you can manipulate the fixed/free status of the
dofs.  You can set all entries to True or False using the
:meth:``fix_all`` or :meth:``unfix_all`` methods from
:obj:``simsopt.Optimizable``.  You can set individual entries using
the string names or arry indices of each dof via the :meth:``fix`` or
:meth:``unfix`` methods, e.g. :meth:``fix("phiedge")`` or
:meth:``unfix("2")``.


Dependencies
------------

It may happen that one object depends on degrees of freedom owned by a
different object. For instance, suppose we have an object ``v`` which
is an instance of the ``VMEC`` class. Then ``v`` has as an attribute
``boundary`` which is an instance of the ``Surface`` class, describing
the plasma boundary, and v's functions depend on dofs owned by the
``Surface``. Simsopt detects this kind of dependency automatically, so that
if a function of ``v`` is put into the objective function, the dofs of
``boundary`` are also included among the global dofs.

To represent this situation, the ``v`` object at the time of initialization
passes the argument ``depends_on``, which is a list of Optimizable objects, to
the base ``Optimizable`` class. In this specific
example, inside ``VMEC.__init__`` method, call to base class is made as
``super().__init__(..., depends_on=[self.boundary], ...``.


Derivatives
-----------

Simsopt can manage both derivative-free and derivative-based
optimization, automatically detecting whether derivative information
is available.  For now, if derivatives are not available for all
functions going into the objective function, then derivative-free
optimization will be used; cases with a mixture of analytic and
finite-difference derivatives are left for future work.

To supply derivative information, your object must provide a function,
property, or attribute with the same name as the one supplied to the
objective function, but with a ``d`` added in front. For instance, if
you used ``obj.func()`` to form the objective function, the derivative
of ``obj.func()`` must be a function ``obj.dfunc()``. Or, if you used
a property ``obj.prop`` to form the objective function, the derivative
of ``obj.prop`` must be a property ``obj.dprop``. If simsopt detects
that all of these functions/properties/attributes are present, it will
use derivative-based optimization.  If one or more derivative
functions is missing, a derivative-free algorithm will be used.

These derivative functions must each return a 1D numpy array,
containing the derivative of the original scalar function with respect
to all local dofs owned both by the object *and any objects it depends
on*. So if ``obj`` owns 10 dofs, and it depends on an object ``dep``
that owns 5 dofs, ``obj.dfunc()`` should return a 15-element vector.
The 10 dofs owned directly by ``obj`` come first. The order of the
dofs from dependencies is the order specified in the ``depends_on``
list.  Your object is responsible for gathering and manipulating
derivative information from objects it depends on in order to form
this combined gradient vector.

The length of the gradient vector returned by your function is
independent of whether or not any dofs are fixed. However, if a dof is
fixed, the corresponding entry in the gradient vector will not be
used, so you could return 0.0 for that entry in the vector rather than
actually computing the derivative.
