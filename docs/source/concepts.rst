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


Magnetic Field Classes
-----------------

Simsopt contains several magnetic field classes available to be called directly. Any field can be summed with any other field and/or multiplied by a constant parameter.

BiotSavart
~~~~~~

The :obj:`simsopt.field.biotsavart.BiotSavart` class initializes a magnetic field vector induced by a list of closed curves :math:`\Gamma_k` with electric currents :math:`I_k`. The field is given by

.. math::

  B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi

where :math:`\mu_0=4\pi 10^{-7}` is the vacuum permitivity.
As input, it takes an of closed curves and the corresponding currents.

ToroidalField
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.ToroidalField` class initializes a toroidal magnetic field vector acording to the formula :math:`\mathbf B = B_0 \frac{R_0}{R} \mathbf e_\phi`, where :math:`R_0` and :math:`B_0` are input scalar quantities, with :math:`R_0` representing the major radius of the magnetic axis and :math:`B_0` the magnetic field at :math:`R_0`. :math:`R` is the radial coordinate of the cylindrical coordinate system :math:`(R,Z,\phi)`, so that :math:`B` has the expected :math:`1/R` dependence. Given a set of points :math:`(x,y,z)`, :math:`R` is calculated as :math:`R=\sqrt{x^2+y^2}`. The vector :math:`\mathbf e_\phi` is a unit vector pointing in the direction of increasing :math:`\phi`, with :math:`\phi` the standard azimuthal angle of the cylindrical coordinate system :math:`(R,Z,\phi)`. Given a set of points :math:`(x,y,z)`, :math:`\mathbf e_\phi` is calculated as :math:`\mathbf e_\phi=-\sin \phi \mathbf e_x+\cos \phi \mathbf e_y` with :math:`\phi=\arctan(y/x)`. 

PoloidalField
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.PoloidalField` class initializes a poloidal magnetic field vector acording to the formula :math:`\mathbf B = B_0 \frac{r}{q R_0} \mathbf e_\theta`, where :math:`R_0, q` and :math:`B_0` are input scalar quantities. :math:`R_0` represents the major radius of the magnetic axis, :math:`B_0` the magnetic field at :math:`r=R_0 q` and :math:`q` the safety factor associated with the sum of a poloidal magnetic field and a toroidal magnetic field with major radius :math:`R_0` and magnetic field on-axis :math:`B_0`. :math:`r` is the radial coordinate of the simple toroidal coordinate system :math:`(r,\phi,\theta)`. Given a set of points :math:`(x,y,z)`, :math:`r` is calculated as :math:`r=\sqrt{(\sqrt{x^2+y^2}-R_0)^2+z^2}`. The vector :math:`\mathbf e_\theta` is a unit vector pointing in the direction of increasing :math:`\theta`, with :math:`\theta` the poloidal angle in the simple toroidal coordinate system :math:`(r,\phi,\theta)`. Given a set of points :math:`(x,y,z)`, :math:`\mathbf e_\theta` is calculated as :math:`\mathbf e_\theta=-\sin \theta \cos \phi \mathbf e_x+\sin \theta \sin \phi \mathbf e_y+\cos \theta \mathbf e_z` with :math:`\phi=\arctan(y/x)` and :math:`\theta=\arctan(z/(\sqrt{x^2+y^2}-R_0))`.

ScalarPotentialRZMagneticField
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.ScalarPotentialRZMagneticField` class initializes a vacuum magnetic field :math:`\mathbf B = \nabla \Phi` defined via a scalar potential :math:`\Phi` in cylindrical coordinates :math:`(R,Z,\phi)`. The field :math:`\Phi` is given as an analytical expression and ``simsopt`` performed the necessary partial derivatives in order find :math:`\mathbf B` and its derivatives. Example: the function 
.. code-block::
   ScalarPotentialRZMagneticField("2*phi")
initializes a toroidal magnetic field :math:`\mathbf B = \nabla (2\phi)=2/R \mathbf e_\phi`.
Note: this functions needs the library ``sympy`` for the analytical derivatives.

CircularCoil
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.CircularCoil` class initializes a magnetic field created by a single circular coil. It takes four input quantities: :math:`a`, the radius of the coil, :math:`\mathbf c=[c_x,c_y,c_z]`, the center of the coil, :math:`I`, the current flowing through the coil and :math:`\mathbf n`, the normal vector to the plane of the coil centered at the coil radius, which could be specified either with its three cartesian components :math:`\mathbf n=[n_x,n_y,n_z]` or as :math:`\mathbf n=[\theta,\phi]` with the spherical angles :math:`\theta` and :math:`\phi`.

The magnetic field is calculated analitically using the following expressions (`reference <https://ntrs.nasa.gov/citations/20010038494>`_)

- :math:`B_x=\frac{\mu_0 I}{2\pi}\frac{x z}{\alpha^2 \beta \rho^2}\left[(a^2+r^2)E(k^2)-\alpha^2 K(k^2)\right]`
- :math:`B_y=\frac{y}{x}B_x`
- :math:`B_z=\frac{\mu_0 I}{2\pi \alpha^2 \beta}\left[(a^2-r^2)E(k^2)+\alpha^2 K(k^2)\right]`

where :math:`\rho^2=x^2+y^2`, :math:`r^2=x^2+y^2+z^2`, :math:`\alpha^2=a^2+r^2-2a\rho`, :math:`\beta^2=a^2+r^2+2 a \rho`, :math:`k^2=1-\alpha^2/\beta^2`.

Dommaschk
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.Dommaschk` class initializes a vacuum magnetic field :math:`\mathbf B = \nabla \Phi` with a representation for the scalar potential :math:`\Phi` as proposed in `W. Dommaschk (1986), Computer Physics Communications 40, 203-218 <https://www.sciencedirect.com/science/article/pii/0010465586901098>`_. It allows to quickly generate magnetic fields with islands with only a small set of scalar quantities. Following the original reference, a toroidal field with :math:`B_0=R_0=1` is already included in the definition. As input parameters, it takes two arrays:

- The first array is an :math:`N\times2` array :math:`[(m_1,n_1),(m_2,n_2),...]` specifying which harmonic coefficients :math:`m` and :math:`n` are non-zero.
- The second array is an :math:`N\times2` array :math:`[(b_1,c_1),(b_2,c_2),...]` with :math:`b_i=b_{m_i,n_i}` and :math:`c_i=c_{m_i,n_i}` the coefficients used in the Dommaschk representation.

Reiman
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.Reiman` initializes the magnetic field model in section 5 of `Reiman and Greenside, Computer Physics Communications 43 (1986) 157—167 <https://www.sciencedirect.com/science/article/pii/0010465586900597>`_. 
It is an analytical magnetic field representation that allows the explicit calculation of the width of the magnetic field islands. It takes as input arguments: :math:`\iota_0`, the unperturbed rotational transform, :math:`\iota_1`, the unperturbed global magnetic shear, :math:`k`, an array of integers with that specifies the Fourier modes used, :math:`\epsilon_k`, an array that specifies the coefficient in front of the Fourier modes, :math:`m_0`, the toroidal symmetry parameter (usually 1).

InterpolatedField
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.InterpolatedField` function takes an existing field and interpolates it on a regular grid in :math:`r,\phi,z`. This resulting interpolant can then be evaluated very quickly.
As input arguments, it takes field: the underlying :mod:`simsopt.field.magneticfield.MagneticField` to be interpolated, degree: the degree of the piecewise polynomial interpolant, rrange: a 3-tuple of the form ``(rmin, rmax, nr)``, phirange: a 3-tuple of the form ``(phimin, phimax, nphi)``, zrange: a 3-tuple of the form ``(zmin, zmax, nz)``, extrapolate: whether to extrapolate the field when evaluate outside the integration domain or to throw an error, nfp: Whether to exploit rotational symmetry, stellsym: Whether to exploit stellarator symmetry. 


Particle Tracer
-----------------

Simsopt is able to follow particles in a magnetic field. The main function to use in this case is ``trace_particles`` and it is able to use two different sets of equations depending on the input parameter ``mode``:

- In the case of ``mode='full'`` it solves

.. math::

  [\ddot x, \ddot y, \ddot z] = \frac{q}{m}  [\dot x, \dot y, \dot z] \times \mathbf B

- In the case of ``mode='gc_vac'`` it solves the guiding center equations under
    the assumption :math:`\nabla p=0`, that is

.. math::

  [\dot x, \dot y, \dot z] &= v_{||}\frac{\mathbf B}{B} + \frac{m}{q|B|^3}  \left(\frac{v_\perp^2}{2} + v_{||}^2\right)  \mathbf B\times \nabla B\\
  \dot v_{||}    &= -\mu  \mathbf B \cdot \nabla B

where :math:`v_\perp^2 = 2\mu B`.
See equations (12) and (13) of `Guiding Center Motion, H.J. de Blank <https://doi.org/10.13182/FST04-A468>`_.

The ``trace_particles`` function, takes as arguments

- ``field``: The magnetic field :math:`B`.
- ``xyz_inits``: A (nparticles, 3) array with the initial positions of the particles.
- ``parallel_speeds``: A (nparticles, ) array containing the speed in direction of the B field for each particle.
- ``tmax``: integration time
- ``mass``: particle mass in kg, defaults to the mass of an alpha particle
- ``charge``: charge in Coulomb, defaults to the charge of an alpha particle
- ``Ekin``: kinetic energy in Joule, defaults to 3.52MeV
- ``tol``: tolerance for the adaptive ode solver
- ``comm``: MPI communicator to parallelize over
- ``phis``: list of angles in [0, 2pi] for which intersection with the plane corresponding to that phi should be computed
- ``stopping_criteria``: list of stopping criteria, mostly used in combination with the ``LevelsetStoppingCriterion`` accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
- mode: how to trace the particles. options are
   `gc`: general guiding center equations,
   `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
   `full`: full orbit calculation (slow!)
- ``forget_exact_path``: return an empty list for the ``res_tys``. To be used when only res_phi_hits is of interest and one wants to reduce memory usage.

As output, it returns 2 elements:
- ``res_tys``:
   A list of numpy arrays (one for each particle) describing the
   solution over time. The numpy array is of shape (ntimesteps, M)
   with M depending on the ``mode``.  Each row contains the time and
   the state.  So for `mode='gc'` and `mode='gc_vac'` the state
   consists of the xyz position and the parallel speed, hence
   each row contains `[t, x, y, z, v_par]`.  For `mode='full'`, the
   state consists of position and velocity vector, i.e. each row
   contains `[t, x, y, z, vx, vy, vz]`.

- ``res_phi_hits``:
   A list of numpy arrays (one for each particle) containing
   information on each time the particle hits one of the phi planes or
   one of the stopping criteria. Each row of the array contains
   `[time] + [idx] + state`, where `idx` tells us which of the `phis`
   or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
   was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
