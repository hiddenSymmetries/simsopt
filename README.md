# simsopt (*Sim*ons *S*tellarator *Opt*imizer Code)

![GitHub](https://img.shields.io/github/license/hiddensymmetries/simsopt)
![Codecov](https://img.shields.io/codecov/c/github/hiddensymmetries/simsopt)

# Status

- [x] Optimization using VMEC
- [x] Derivative-free optimization
- [x] Derivative-based optimization
- [ ] Example in which derivatives are available for some functions but not others
- [x] Example using automatic differentiation
- [x] Example that uses simsgeo
- [x] Optimize either RBC/ZBS or Garabedian coefficients
- [x] MPI
- [x] SPEC
- [x] Boozer-coordinate transformation
- [ ] epsilon_effective
- [ ] Standard (non-least-squares) optimization problem
- [ ] Bound constraints
- [ ] Nonlinear constraints

See the working examples in the `examples/` directory, in particular
`gradientBasedSurfaceOptimization.ipynb` and
`stellopt_scenarios_2DOF_vmecOnly_targetIotaAndVolume.ipynb`.

# Optimizable objects

Simsopt is able to optimize any python object that has a `get_dofs()` function and a `set_dofs()` function.
The overall objective function can be defined using any function, attribute, or @property of such an object.
Optimizable objects can depend on other optimizable objects, so
objects can be part of an optimization even if they do not directly own a function that is part of the overall objective function.


## Specifying functions that go into the objective function

Suppose we want to solve a least-squares optimization problem in which
an object `obj` is optimized. If `obj` has a function `func()`, we can
use a 3-element tuple or list

```python
term1 = (obj.func, goal, weight)
```

to represent a term `weight * ((obj.func() - goal) ** 2)` in the least-squares objective function. If
you prefer, you can specify `sigma = 1 / sqrt(weight)` rather than `weight` and use the `LeastSquaresTerm` object, as in 
`term1 = simsopt.LeastSquaresTerm(obj.func, goal, sigma=0.3)`.

In this example, `func()` could return a scalar, or it could return a 1D numpy array. In the latter case,
`sum(weight * ((obj.func() - goal) ** 2))` would be included in the objective function, and `goal` could
be either a scalar or a 1D numpy array of the same length as that returned by `func()`.

The function name `J` is special: if `obj` has a function `J` then we can specify just the object, and the function name `J` will be assumed:

```python
term2 = (obj, goal, weight)
```

Or, if we want the objective function to include a `@property` or attribute that is not a function, we can use a 4-element tuple or list instead of the 3-element form above. In the 4-element syntax, element 1 is the attribute or property name as a string. For instance, if `obj` has a property named `prop`, we would write

```python
term3 = (obj, 'prop', goal, weight)
```

This longer syntax is needed to optimize a property or attribute because `obj.prop` evaluates to a number.
As with the previous examples, `prop` could be either a scalar or a 1D numpy array.

The total least-squares objective function is created using a list or tuple of all the terms that are to be added together:

```python
prob = LeastSquaresProblem([term1, term2, term3])
```

The list can include any mixture of terms defined by scalar functions
and by 1D numpy array-valued functions.


## Degrees of freedom ("dofs")

Any optimizable object must have a function `get_dofs(self)` that returns a 1D numpy array of floats. (Simsopt will not allow integer optimization.)
Any optimizable object must have a function `set_dofs(self, x)` that accepts `x`, a 1D numpy array of floats. Some of the degrees of freedom may be fixed in the optimization - see below.

It can be convenient to have a descriptive string associated with each dof, e.g. `"phiedge"` or `"rc(1,0)"`. To supply these names, add a `names` attribute to your
optimizable object which is a list of strings, where the length of the list matches the size of the `get_dofs()` array. It is not necessary to have a `names` list;
if one is not present, simsopt will use as a default `["x[0]", "x[1]", ...]`.

Each dof is owned by a single object. In other words, if a physical dof is represented by an array element for `get_dofs()` and `set_dofs()` of a particular object,
there should be no other object that has that physical degree of freedom an an element in its  `get_dofs()` and `set_dofs()` arrays. If two objects both depend on a given physical degree of freedom,
that information should be represented using the `depends_on` attribute, described below.

In addition to the "local" dofs owned by each object, an important concept is the set of "global" dofs associated with an optimization problem. The global
dofs are the set of all non-fixed dofs of each optimized object and all objects upon which they depend. (Dependencies among objects are discussed in a section below.)
The global dofs correspond to the state vector passed to the optimization algorithm.

At each function evaluation during an optimization, simsopt will call the `set_dofs()` function of each object involved in the optimization problem.
This will happen even if the local dofs owned by that object have not changed since the previous call to `set_dofs()` (which may happen if global dofs owned by a different object changed instead.)
This choice is made because even if an object's dofs have not changed, an object it depends on may have changed, so it may still need to update.
It is the responsibility of each object to check its dof vector and its dependencies to know when it needs to update expensive calculations.


## Helpful functions

Simsopt can provide several helpful functions to your optimizable object. For instance, functions can be provided to help with fixing certain degrees of freedom, as discussed below. There is also the method `index(str)` which gets the index into the dof array associated with a dof name `str`, and the methods `get(str)` and `set(str, val)` which get and set a dof associated with the name `str`.

There are two ways to equip your object with these functions. One way is to inherit from the `simsopt.Optimizable` class. Or, if you do not want your class to depend on simsopt, you can apply the decorator `simsopt.optimizable()` to your object. Note the first method uses a capital O whereas the latter uses a lowercase o.


## Fixing degrees of freedom

Sometimes we may want to vary a particular dof in an optimization, and other times we may want to hold that same dof fixed.
One example is the current in a coil. To enable this flexibility, every local dof in simsopt is considered to be either fixed or not.
Only the non-fixed dofs are included in the global dofs for an optimization. (So by definition, every global dof is not fixed.)
Whether or not a local dof is fixed is determined by an optional `fixed` attribute of the object that owns the dof. The `fixed` attribute can be either
a list or numpy array, with each element True or False.

There are several ways you can manipulate the `fixed` list/array. You are free to edit the elements by hand, e.g. `fixed[3] = False`.
You can set all entries to True or False using the `all_fixed()` method from `simsopt.Optimizable` or `simsopt.optimizable`.
You can set individual entries using the string names of each dof via the `set_fixed()` method, e.g. `set_fixed("phiedge")` or `set_fixed("rc(0,0)", False)`.


## Dependencies

It may happen that one object depends on degrees of freedom owned by a different object. For instance, suppose we have an object `v` which
is an instance of the `VMEC` class. Then `v` has as an attribute `boundary` which is an instance of the `Surface` class, describing
the plasma boundary, and v's functions depend on dofs owned by the `Surface`. Simsopt needs to detect this kind of dependency so that if a function of `v`
is put into the objective function, the dofs of `boundary` must be included among the global dofs.

To represent this situation, the `v` object must have
an attribute `depends_on`, which is a list of strings. Each string describes the name of the attribute on which the object depends. In
this specific example, `v.depends_on = ["boundary"]`. The elements of `depends_on` are strings rather than specific objects so that
if we assign a new `Surface` object to `v.boundary`, simsopt will be able to automatically identify the new dependency.

If your object does not have a `depends_on` attribute, simsopt will assume it does not depend on any other object.

The order of entries in the `depends_on` list is important for two reasons. First, it specifies the order in which local dofs of dependencies are 
combined into the global dof vector. Second, and more importantly, it gives the order of entries for gradient vectors, if they are supplied. This issue is detailed in the next section.


## Derivatives

Simsopt can manage both derivative-free and derivative-based optimization, automatically detecting whether derivative information is available.
For now, if derivatives are not available for all functions going into the objective function, then derivative-free optimization will be used;
cases with a mixture of analytic and finite-difference derivatives are left for future work.

To supply derivative information, your object must provide a function, property, or attribute with the same name as the one supplied to the objective function,
but with a `d` added in front. For instance, if you used `obj.func()` to form the objective function, the derivative of `obj.func()` must be a function
`obj.dfunc()`. Or, if you used a property `obj.prop` to form the objective function, the derivative of `obj.prop` must
be a property `obj.dprop`. If simsopt detects that all of these functions/properties/attributes are present, it will use derivative-based optimization.
If one or more derivative functions is missing, a derivative-free algorithm will be used.

These derivative functions must each return a 1D numpy array, containing the derivative of the original scalar function with respect to all local dofs
owned both by the object _and any objects it depends on_. So if `obj` owns 10 dofs, and it depends on an object `dep` that owns 5 dofs, `obj.dfunc()` should return a 15-element vector.
The 10 dofs owned directly by `obj` come first. The order of the dofs from dependencies is the order specified in the `depends_on` list.
Your object is responsible for gathering and manipulating derivative information from objects it depends on in order to form this combined gradient vector.

The length of the gradient vector returned by your function is independent of whether or not any dofs are fixed. However, if a dof is fixed, the corresponding entry in the gradient
vector will not be used, so you could return 0.0 for that entry in the vector rather than actually computing the derivative.
