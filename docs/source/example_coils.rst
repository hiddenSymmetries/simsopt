Coil optimization
=================

Here we work through an example of "stage 2" coil optimization.  In
this approach, we consider the target plasma boundary shape to have
been already chosen in a "stage 1", and the present task is to
optimize the shapes of coils to produce this target field. The
tutorial here is similar to the example
``examples/2_Intermediate/stage_two_optimization.py``.  Note that for
this stage-2 problem, no MHD codes like VMEC or SPEC are used, so you
do not need to have them installed.

The objective function we will minimize is

.. math::

   J = \frac{1}{2} \int |\vec{B}\cdot\vec{n}|^2 ds + \alpha \sum_j L_j  + \beta J_{dist}

The first right-hand-side term is the "quadratic flux", the area
integral over the target plasma surface of the square of the magnetic
field normal to the surface. If the coils exactly produce a flux
surface of the target shape, this term will vanish.  Next, :math:`L_j`
is the length of the :math:`j`-th coil.  The scalar regularization
parameter :math:`\alpha` is chosen to balance a trade-off: large
values will give smooth coils at the cost of inaccuracies in producing
the target field; small values of :math:`\alpha` will give a more
accurate match to the target field at the cost of complicated coils.
Finally, :math:`J_{dist}` is a penalty that prevents coils from
becoming too close.  For its precise definition, see
:obj:`simsopt.geo.curveobjectives.MinimumDistance`.  The constant
:math:`\beta` is selected to balance this minimum distance penalty
against the other objectives.  Analytic derivatives are used for the
optimization.

In this tutorial we will consider vacuum fields, so the magnetic field
due to current in the plasma does not need to be subtracted in the
quadratic flux term. The configuration considered is the "precise QA"
case from `arXiv:2108.03711 <http://arxiv.org/pdf/2108.03711.pdf>`_,
which has two field periods and is quasi-axisymmetric.

The stage-2 optimization problem is automatically parallelized in
simsopt using OpenMP and vectorization, but MPI is not used, so the
``mpi4py`` python package is not needed. This example can be run in a
few seconds on a laptop.

To begin solving this optimization problem in simsopt, we first import
some classes that will be used::

  import numpy as np
  from scipy.optimize import minimize
  from simsopt.geo.surfacerzfourier import SurfaceRZFourier
  from simsopt.objectives.fluxobjective import SquaredFlux
  from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
  from simsopt.field.biotsavart import BiotSavart
  from simsopt.field.coil import Current, coils_via_symmetries
  from simsopt.geo.curveobjectives import CurveLength, MinimumDistance
  from simsopt.geo.plot import plot

The target plasma surface is given in the VMEC input file ``tests/test_files/input.LandremanPaul2021_QA``.
We load the surface using

.. code-block::

  nphi = 32
  ntheta = 32
  filename = "tests/test_files/input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

You can adjust the directory in ``"filename"`` as appropriate for your
system. The surface could also be read in from a VMEC wout file using
:obj:`simsopt.geo.surfacerzfourier.SurfaceRZFourier.from_wout()`.
(VMEC does not need to be installed to initialize a surface from a
VMEC input or output file.)  Note that surface objects carry a grid of
"quadrature points" at which the position vector is evaluated, and in
different circumstances we may want these points to cover different
ranges of the toroidal angle. For this problem with stellarator
symmetry and field-period symmetry, we need only consider half of a
field period in order to evaluate integrals over the entire
surface. For this reason, the ``range`` parameter of the surface is
set to ``"half period"`` here.

Next, we set the initial condition for the coils, which will be equally spaced circles.
Here we will consider a case with four unique coil shapes, each of which is repeated four times due to
stellarator symmetry and two-field-period symmetry, giving a total of 16 coils.
The four unique coil shapes are called the "base coils". Each copy of a coil also carries the same current,
but we will allow the unique coil shapes to have different current from each other,
as is allowed in W7-X. For this tutorial we will consider the coils to be infinitesimally thin filaments.
In simsopt, such a coil is represented with the :obj:`simsopt.field.coil.Coil` class,
which is essentially a curve paired with a current, represented using
:obj:`simsopt.geo.curve.Curve` and :obj:`simsopt.field.coil.Current` respectively.
The initial conditions are set as follows::

  # Number of unique coil shapes:
  ncoils = 4

  # Major radius for the initial circular coils:
  R0 = 1.0
  
  # Minor radius for the initial circular coils:
  R1 = 0.5

  # Number of Fourier modes describing each Cartesian component of each coil:
  order = 5

  base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
  base_currents = [Current(1e5) for i in range(ncoils)]

One detail of optimizing coils for a vacuum configuration is that the
optimizer can "cheat" by making all the currents go to zero, which
makes the quadratic flux vanish. To close this loophole, we can fix
the current of the first base coil::

  base_currents[0].local_fix_all()

(A ``Current`` object only has one degree of freedom, hence we can use
``fix_all()``.)  If you wish, you can fix the currents in all the
coils to force them to have the same value. Now the full set of 16
coils can be obtained using stellarator symmetry and field-period
symmetry::

  coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

It is illuminating to look at the non-fixed degrees of freedom that
each coil depends on. This can be done by printing the ``dof_names``
property::

  >>> print(coil[0].dof_names)

  ['CurveXYZFourier1:xc(0)', 'CurveXYZFourier1:xs(1)', 'CurveXYZFourier1:xc(1)', ...

  >>> print(coil[1].dof_names)

  ['Current2:x0', 'CurveXYZFourier2:xc(0)', 'CurveXYZFourier2:xs(1)', 'CurveXYZFourier2:xc(1)', ...

  >>> print(coil[4].dof_names)

  ['CurveXYZFourier1:xc(0)', 'CurveXYZFourier1:xs(1)', 'CurveXYZFourier1:xc(1)', ...

Notice that the current appears in the list of dofs for ``coil[1]``
but not for ``coil[0]``, since we fixed the current for
``coil[0]``. Also notice that ``coil[4]`` has the same degrees of
freedom (owned by ``CurveXYZFourier1``) as ``coil[0]``, because coils
0 and 4 refer to the same base coil shape.

There are several ways to view the objects we have created so far. One
approach is the function :obj:`simsopt.geo.plot.plot()`, which accepts
a list of Coil, Curve, and/or Surface objects::

  plot(coils + [s], engine="mayavi", close=True)

.. image:: coils_init.png
   :width: 500
	
Instead of ``"mayavi"`` you can select ``"matplotlib"`` or
``"plotly"`` as the graphics engine, although matplotlib has problems
with displaying multiple 3D objects in the proper
order. Alternatively, you can export the objects in VTK format and
open them in Paraview::

  curves = [c.curve for c in coils]
  curves_to_vtk(curves, "curves_init")
  s.to_vtk("surf_init")
  
To evaluate the magnetic field on the target surface, we create
:obj:`simsopt.field.biotsavart.BiotSavart` object based on the coils,
and instruct it to evaluate the field on the surface::

  bs = BiotSavart(coils)
  bs.set_points(s.gamma().reshape((-1, 3)))

(The surface position vector ``gamma()`` returns an array of size
``(nphi, ntheta, 3)``, which we reshaped here to
``(number_of_evaluation_points, 3)`` for the
:obj:`~simsopt.field.biotsavart.BiotSavart` object.)  Let us check the
size of the field normal to the target surface before optimization::

  B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
  print('Initial max B dot n:', np.max(B_dot_n))

The result is 0.19 Tesla. We now define the objective function::

  # Weight on the curve lengths in the objective function:
  ALPHA = 1e-6
  # Threshhold for the coil-to-coil distance penalty in the objective function:
  MIN_DIST = 0.1
  # Weight on the coil-to-coil distance penalty term in the objective function:
  BETA = 10
  
  Jf = SquaredFlux(s, bs)
  Jls = [CurveLength(c) for c in base_curves]
  Jdist = MinimumDistance(curves, MIN_DIST)
  # Scale and add terms to form the total objective function:
  objective = Jf + ALPHA * sum(Jls) + BETA * Jdist

In the last line, we have used the fact that the Optimizable objects
representing the individual terms in the objective can be scaled by a
constant and added.  (This feature applies to Optimizable objects that
have a function ``J()`` returning the objective and, if gradients are
used, a function ``dJ()`` returning the gradient.)

You can check the degrees of freedom that will be varied in the
optimization by printing the ``dof_names`` property of the objective::

  >>> print(objective.dof_names)

  ['Current2:x0', 'Current3:x0', 'Current4:x0', 'CurveXYZFourier1:xc(0)', 'CurveXYZFourier1:xs(1)', ...
   'CurveXYZFourier1:zc(5)', 'CurveXYZFourier2:xc(0)', 'CurveXYZFourier2:xs(1)', ...
   'CurveXYZFourier4:zs(5)', 'CurveXYZFourier4:zc(5)']

As desired, the Fourier amplitudes of all four base coils appear, as
do three of the four currents.  Next, to interface with scipy's
minimization routines, we write a small function::

  def fun(dofs):
    objective.x = dofs
    return objective.J(), objective.dJ()

Note that when the ``dJ()`` method of the objective is called to
compute the gradient, simsopt automatically applies the chain rule to
assemble the derivatives from the various terms in the objective, and
entries in the gradient corresponding to degrees of freedom that are
fixed (such as the current in the first coil) are automatically
removed.  We can now run the optimization using the `L-BFGS-B algorithm
from scipy
<https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_::

  res = minimize(fun, objective.x, jac=True, method='L-BFGS-B',
                 options={'maxiter': 200, 'iprint': 5}, tol=1e-15)
  
The optimization takes a few seconds, and the output will look like

.. code-block:: none
   
   RUNNING THE L-BFGS-B CODE

           * * *

  Machine precision = 2.220D-16
   N =          135     M =           10
   This problem is unconstrained.

  At X0         0 variables are exactly at the bounds

  At iterate    0    f=  3.26880D-02    |proj g|=  5.14674D-02

  At iterate    5    f=  6.61538D-04    |proj g|=  2.13561D-03

  At iterate   10    f=  1.13772D-04    |proj g|=  6.27872D-04

  ...
  At iterate  195    f=  1.81723D-05    |proj g|=  4.18583D-06

  At iterate  200    f=  1.81655D-05    |proj g|=  6.31030D-06

           * * *

  Tit   = total number of iterations
  Tnf   = total number of function evaluations
  Tnint = total number of segments explored during Cauchy searches
  Skip  = number of BFGS updates skipped
  Nact  = number of active bounds at final generalized Cauchy point
  Projg = norm of the final projected gradient
  F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
  135    200    234      1     0     0   6.310D-06   1.817D-05
  F =   1.8165520700970273E-005

  STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 

You can adjust parameters such as the tolerance and number of
iterations. Let us check the final :math:`\vec{B}\cdot\vec{n}` on the surface::

  B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
  print('Final max B dot n:', np.max(B_dot_n))

The final value is 0.0017 Tesla, reduced two orders of magnitude from
the initial state.  As with the initial conditions, you can plot the
optimized coil shapes directly from simsopt using

.. code-block::

  plot(coils + [s], engine="mayavi", close=True)
  
or you can export the objects in VTK format and open them in
Paraview. For this latter option, we can also export the final
:math:`\vec{B}\cdot\vec{n}` on the surface using the following
syntax::

  curves = [c.curve for c in coils]
  curves_to_vtk(curves, "curves_opt")
  s.to_vtk("surf_opt", extra_data={"B_N": B_dot_n[:, :, None]})

.. image:: coils_final.png
   :width: 500
	
The optimized value of the current in coil ``j`` can be obtained using
``coils[j].current.get_value()``. The optimized Fourier coefficients
for coil ``j`` can be obtained from ``coils[j].curve.x``, where the
meaning of each array element can be seen from
``coils[j].curve.dof_names``.  The position vector for coil ``j`` in
Cartesian coordinates can be obtained from ``coils[j].curve.gamma()``.
