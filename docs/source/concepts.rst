Concepts
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
in contrast to ``STELLOPT``. Rather, problems are specified using a
python driver script, in which objects are defined and
configured. However, objects related to specific physics codes may use
their own input files. In particular, a :obj:`simsopt.mhd.vmec.Vmec` object
can be initialized using a standard VMEC ``input.*`` input file, and a
:obj:`simsopt.mhd.spec.Spec` object can be initialized using a standard
SPEC ``*.sp`` input file.


Optimization
------------

To do optimization using simsopt, there are four basic steps:

1. Create some objects that will participate in the optimization.
2. Define the independent variables for the optimization, by choosing which degrees of freedom of the objects are free vs fixed.
3. Define an objective function.
4. Solve the optimization problem that has been defined.

This pattern is evident in the examples in this documentation
and in the ``examples`` directory of the repository.

Some typical objects are a MHD equilibrium represented by the VMEC or
SPEC code, or some electromagnetic coils. To define objective
functions, a variety of additional objects can be defined that depend
on the MHD equilibrium or coils, such as a
:obj:`simsopt.mhd.boozer.Boozer` object for Boozer-coordinate
transformation, a :obj:`simsopt.mhd.spec.Residue` object to represent
Greene's residue of a magnetic island, or a
:obj:`simsopt.geo.objectives.LpCurveCurvature` penalty on coil
curvature.

More details about setting degrees of freedom and defining
objective functions can be found on the :doc:`problems` page.

For the solution step, two functions are provided presently,
:meth:`simsopt.solve.serial.least_squares_serial_solve` and
:meth:`simsopt.solve.mpi.least_squares_mpi_solve`.  The first
is simpler, while the second allows MPI-parallelized finite differences
to be used in the optimization.



.. _mpi:

MPI Partitions and worker groups
--------------------------------

To use MPI parallelization with simsopt, you must install the
`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ package.

MPI simsopt programs can generally be launched using ``mpirun`` or
``mpiexec``, e.g.

.. code-block::

   mpiexec -n 32 ./name_of_your_python_driver_script

where 32 can be replaced by the number of processors you want to use.
There may be specific instructions to run MPI programs on your HPC system,
so consult the documentation for your system.
   
Given a set of :math:`M` MPI processes, an optimization problem can be
parallelized in different ways.  At one extreme, all :math:`M`
processes could work together on each evaluation of the objective
function, with the optimization algorithm itself only initiating a
single evaluation of the objective function :math:`f` at a time.  At
the other extreme, the optimization algorithm could request :math:`M`
evaluations of :math:`f` all at once, with each evaluation performed
by a single processor.  This type of parallelization can be useful for
instance when finite differences are used to evaluate gradients.  Or,
both types of parallelization could be used at the same time. To allow
all of these types of parallelization, simsopt uses the concepts of
*worker groups* and *MPI partitions*.

A *worker group* is a set of MPI processes that works together to
evaluate the objective function.  An MPI partition is a division of
the total set of :math:`M` MPI processors into one or more worker
groups.  If there are :math:`W` worker groups, each group has
approximately :math:`M/W` processors (approximate because one may need to
round up or down.)  Simsopt has a class
:obj:`simsopt.util.mpi.MpiPartition` to manage this division of
processors into worker groups.  Each MPI-aware simsopt object, such as
:obj:`simsopt.mhd.vmec.Vmec`, keeps an instance of :obj:`~simsopt.util.mpi.MpiPartition`
which can be set from its constructor.  The number of worker
groups can be set by an argument to :obj:`~simsopt.util.mpi.MpiPartition`.
For instance, to tell a Vmec object to use four worker groups, one could write

.. code-block::

   from simsopt.mhd import Vmec
   from simsopt.util.mpi import MpiPartition
   
   mpi = MpiPartition(4)
   equil = Vmec("input.li383_low_res", mpi=mpi)

The same :obj:`~simsopt.util.mpi.MpiPartition` instance should be passed to the solver::

  # ... code to define an optimization problem "prob" ...
  
  from simsopt.solve.mpi import least_squares_mpi_solve
  
  least_squares_mpi_solve(prob, mpi, grad=True)

Many optimization algorithms that do not use derivatives do not
support concurrent evaluations of the objective.  In this case, the
number of worker groups should be :math:`W=1`.  Any algorithm that
uses derivatives, such as Levenberg-Marquardt, can take advantage of
multiple worker groups to evaluate derivatives by finite
differences. If the number of parameters (i.e. independent variables)
is :math:`N`, you ideally want to set :math:`W=N+1` when using 1-sided
finite differences, and set :math:`W=2N+1` when using centered
differences.  These ideal values are not required however -
``simsopt`` will evaluate finite difference derivatives for any value
of :math:`W`, and results should be exactly independent of :math:`W`.
Other derivative-free algorithms intrinsically support
parallelization, such as HOPSPACK, though no such algorithm is
available in ``simsopt`` yet.

An MPI partition is associated with three MPI communicators, "world",
"groups", and "leaders". The "world" communicator
represents all :math:`M` MPI processors available to the program. (Normally
this communicator is the same as ``MPI_COMM_WORLD``, but it could be a
subset thereof if you wish.)  The "groups" communicator also
contains the same :math:`M` processors, but grouped into colors, with
a different color representing each worker group. Therefore
operations such as ``MPI_Send`` and ``MPI_Bcast`` on this communicator
exchange data only within one worker group.  This "groups"
communicator is therefore the one that must be used for evaluation of
the objective function.  Finally, the "leaders" communicator
only includes the :math:`W` processors that have rank 0 in the
"groups" communicator.  This communicator is used for
communicating data within a parallel optimization *algorithm* (as
opposed to within a parallelized objective function).

Given an instance of :obj:`simsopt.util.mpi.MpiPartition` named
``mpi``, the number of worker groups is available as ``mpi.ngroups``,
and the index of a given processor's group is ``mpi.group``.  The
communicators are available as ``mpi.comm_world``,
``mpi.comm_groups``, and ``mpi.comm_leaders``.  The number of
processors within the communicators can be determined from
``mpi.nprocs_world``, ``mpi.nprocs_groups``, and
``mpi.nprocs_leaders``.  The rank of the present processor within the
communicators is available as ``mpi.rank_world``, ``mpi.rank_groups``,
and ``mpi.rank_leaders``.  To determine if the present processor has
rank 0 in a communicator, you can use the variables
``mpi.proc0_world`` or ``mpi.proc0_groups``, which have type ``bool``.

Geometric Objects
-----------------

Simsopt contains implementations of two commonly used geometric objects: curves and surfaces.

Curves
~~~~~~

A :obj:`simsopt.geo.curve.Curve` is modelled as a function :math:`\Gamma:[0, 1] \to \mathbb{R}^3`.
A curve object stores a list of :math:`n_\phi` quadrature points :math:`\{\phi_1, \ldots, \phi_{n_\phi}\} \subset [0, 1]` and return all information about the curve, at these quadrature points.

- ``Curve.gamma()``: returns a ``(n_phi, 3)`` array containing :math:`\Gamma(\phi_i)` for :math:`i\in\{1, \ldots, n_\phi\}`, i.e. returns a list of XYZ coordinates along the curve.
- ``Curve.gammadash()``: returns a ``(n_phi, 3)`` array containing :math:`\Gamma'(\phi_i)` for :math:`i\in\{1, \ldots, n_\phi\}`, i.e. returns the tangent along the curve.
- ``Curve.kappa()``: returns a ``(n_phi, 1)`` array containing the curvature :math:`\kappa` of the curve at the quadrature points.
- ``Curve.torsion()``: returns a ``(n_phi, 1)`` array containing the torsion :math:`\tau` of the curve at the quadrature points.

The different curve classes, such as :obj:`simsopt.geo.curverzfourier.CurveRZFourier` and :obj:`simsopt.geo.curvexyzfourier.CurveXYZFourier` differ in the way curves are discretized.
Each of these take an array of ``dofs`` (e.g. Fourier coefficients) and turn these into a function :math:`\Gamma:[0, 1] \to \mathbb{R}^3`.
These dofs can be set via ``Curve.set_dofs(dofs)`` and ``dofs = Curve.get_dofs()``, where ``n_dofs = dofs.size``.
Changing dofs will change the shape of the curve. Simsopt is able to compute derivatives of all relevant quantities with respect to the discretisation parameters.
For example, to compute the derivative of the coordinates of the curve at quadrature points, one calls ``Curve.dgamma_by_dcoeff()``.
One obtains a numpy array of shape ``(n_phi, 3, n_dofs)``, containing the derivative of the position at every quadrature point with respect to every degree of freedom of the curve.
In the same way one can compute the derivative of quantities such as curvature (via ``Curve.dkappa_by_dcoeff()``) or torsion (via ``Curve.dtorsion_by_dcoeff()``).

A number of quantities are implemented in :obj:`simsopt.geo.curveobjectives` and are computed on a :obj:`simsopt.geo.curve.Curve`:

- ``CurveLength``: computes the length of the ``Curve``.
- ``LpCurveCurvature``: computes a penalty based on the :math:`L_p` norm of the curvature on a curve.
- ``LpCurveTorsion``: computes a penalty based on the :math:`L_p` norm of the torsion on a curve.
- ``MinimumDistance``: computes a penalty term on the minimum distance between a set of curves.

The value of the quantity and its derivative with respect to the curve dofs can be obtained by calling e.g., ``CurveLength.J()`` and ``CurveLength.dJ()``.

Surfaces
~~~~~~~~

The second large class of geometric objects are surfaces.
A :obj:`simsopt.geo.surface.Surface` is modelled as a function :math:`\Gamma:[0, 1] \times [0, 1] \to \mathbb{R}^3` and is evaluated at quadrature points :math:`\{\phi_1, \ldots, \phi_{n_\phi}\}\times\{\theta_1, \ldots, \theta_{n_\theta}\}`.

The usage is similar to that of the :obj:`~simsopt.geo.curve.Curve` class:

- ``Surface.gamma()``: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`, i.e. returns a list of XYZ coordinates on the surface.
- ``Surface.gammadash1()``: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial_\phi \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- ``Surface.gammadash2()``: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial_\theta \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- ``Surface.normal()``: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial_\phi \Gamma(\phi_i, \theta_j)\times \partial_\theta \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- ``Surface.area()``: returns the surface area.
- ``Surface.volume()``: returns the volume enclosed by the surface.

A number of quantities are implemented in :obj:`simsopt.geo.surfaceobjectives` and are computed on a :obj:`simsopt.geo.surface.Surface`:

- ``ToroidalFlux``: computes the flux through a toroidal cross section of a ``Surface``.

The value of the quantity and its derivative with respect to the surface dofs can be obtained by calling e.g., ``ToroidalFlux.J()`` and ``ToroidalFlux.dJ_dsurfacecoefficients()``.


Caching
~~~~~~~~

The quantities that Simsopt can compute for curves and surfaces often depend on each other.
For example, the curvature or torsion of a curve both rely on ``Curve.gammadash()``; to avoid repeated calculation 
geometric objects contain a cache that is automatically managed.
If a quantity for the curve is requested, the cache is checked to see whether it was already computed.
This cache can be cleared manually by calling ``Curve.invalidate_cache()``.
This function is called everytime ``Curve.set_dofs()`` is called (and the shape of the curve changes).

