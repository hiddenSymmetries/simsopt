Overview
========

Ways to use simsopt
-------------------

Simsopt is a collection of classes and functions that can be used in
several ways.  One application is to solve optimization problems
involving stellarators, similar to STELLOPT.  You could also define an
objective function using simsopt, but use an optimization library
outside simsopt to solve it.  Or, you could use the simsopt
optimization infrastructure to optimize your own objects, which may or
may not have any connection to stellarators.  Alternatively, you can
use the stellarator-related objects in a script for some purpose other
than optimization, such as plotting how some code output varies as an
input parameter changes, or evaluating the finite-difference gradient
of some code outputs.  Or, you can manipulate the objects
interactively, at the python command line or in a Jupyter notebook.


Input files
-----------

Simsopt does not use input data files to define optimization problems,
in contrast to STELLOPT. Rather, problems are specified using a
python driver script, in which objects are defined and
configured. However, objects related to specific physics codes may use
their own input files. In particular, a :obj:`simsopt.mhd.Vmec` object
can be initialized using a standard VMEC ``input.*`` input file, and a
:obj:`simsopt.mhd.Spec` object can be initialized using a standard
SPEC ``*.sp`` input file.


Optimization stages
-------------------

Recent optimized stellarators have been designed using two stages,
both of which can be performed using simsopt. In the first stage, the
parameter space is the shape of a toroidal boundary flux
surface. Coils are not considered explicitly in this stage.  The
objective function involves surrogates for confinement and stability
in the plasma inside the boundary surface.  In the second optimization
stage, coil shapes are optimized to produce the plasma shape that
resulted from stage 1.  The parameter space for stage 2 represents the
space of coil shapes. The objective function for stage 2 usually
involves several terms.  One term is the deviation between the
magnetic field produced by the coils and the magnetic field desired at
the plasma boundary, given the stage 1 solution. Other terms in the
objective function introduce regularization on the coil shapes, such
as the coil length and/or curvature, and reflect other engineering
considerations such as the distance between coils. In the future, we
aim to introduce alternative optimization strategies in simsopt
besides this two-stage approach, such as combined single-stage
methods.



Optimization
------------

To do optimization using simsopt, there are four basic steps:

1. Define the physical entities in the optimization problem (coils, MHD equilibria, etc.) by creating instances of the relevant simsopt classes.
2. Define the independent variables for the optimization, by choosing which degrees of freedom of these objects are free vs fixed.
3. Define an objective function.
4. Solve the optimization problem that has been defined.

This pattern is evident in the tutorials in this documentation
and in the :simsopt:`examples` directory of the repository.

Some typical objects are a MHD equilibrium represented by the VMEC or
SPEC code, or some electromagnetic coils. To define objective
functions, a variety of additional objects can be defined that depend
on the MHD equilibrium or coils, such as a
:obj:`simsopt.mhd.Boozer` object for Boozer-coordinate
transformation, a :obj:`simsopt.mhd.Residue` object to represent
Greene's residue of a magnetic island, or a
:obj:`simsopt.geo.LpCurveCurvature` penalty on coil
curvature.

More details about setting degrees of freedom and defining
objective functions can be found on the :doc:`optimizable` page.

For the solution step, two functions are provided presently,
:meth:`simsopt.solve.least_squares_serial_solve` and
:meth:`simsopt.solve.least_squares_mpi_solve`.  The first
is simpler, while the second allows MPI-parallelized finite differences
to be used in the optimization.


Modules
-------

Classes and functions in simsopt are organized into several modules:

- :obj:`simsopt.geo` contains several representations of curves and surfaces.
- :obj:`simsopt.field` contains machinery for the Biot-Savart law and other magnetic field representations.
- :obj:`simsopt.mhd` contains interfaces to MHD equilibrium codes and tools for diagnosing their output.
- :obj:`simsopt.objectives` contains tools for some common objective functions.
- :obj:`simsopt.solve` contains wrappers for some optimization algorithms.
- :obj:`simsopt.util` contains other utility functions.
- :obj:`simsopt._core` defines the ``Optimizable`` class and other tools used internally in simsopt.
