Coil optimization
=================

Here we work through an example of "stage 2" coil optimization.  In
this approach, we consider that a surface of constant magnetic flux (plasma
boundary shape) has already been found in what is commonly known as
a "stage 1" optimization, and the present task is to
optimize the shapes of coils to produce this target field.
For this stage-2 problem, no MHD codes are used (example, VMEC or SPEC), so
there is no need to have them installed.

We first describe the basic principles of "stage 2" coil optimization and
how they are used in SIMSOPT, namely, how to set up the optimization problem
in the minimal example ``examples/2_Intermediate/stage_two_optimization.py``.
We then add increasingly more terms to the objective function in order to illustrate
how to perform various extensions of the minimal example, namely:
- Take into account coil perturbations via systematic and statistical
errors using a stochastic optimization method similar to the example in
``examples/2_Intermediate/stage_two_optimization_stochastic.py``.
- Take into account the presence of the plasma and its respective magnetic field
using the virtual casing principle, similar to the example
in ``examples/2_Intermediate/stage_two_optimization_finite_beta.py``.
- Take into account finite coil width using a multifilament approach
similar to the example in ``examples/3_Advanced/stage_two_optimization_finite_build.py``.

.. _simplest_stage2:
Simplest objective function
---------------------------

The first form of the objective function :math:`J` (or cost function)
that will be minimized in order to find coils that yield the desired magnetic field is
the one present in ``examples/2_Intermediate/stage_two_optimization.py`` and given by:

.. math::

  J = (1/2) \int |B dot n|^2 ds
      + LENGTH_WEIGHT * \sum_j L_j
      + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
      + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
      + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

The first term in the right-hand-side term is the "quadratic flux", the area
integral over the target plasma surface of the square of the magnetic
field normal to the surface. If the coils exactly produce a flux
surface of the target shape, this term will vanish. 

Next, :math:`L_j`
is the length of the :math:`j`-th coil.  The scalar regularization
parameter `LENGTH_WEIGHT` is chosen to balance a trade-off: large
values will give smooth coils at the cost of inaccuracies in producing
the target field; small values of `LENGTH_WEIGHT` will give a more
accurate match to the target field at the cost of complicated coils.

Next, `MininumDistancePenalty` is a penalty that prevents coils from
becoming too close.  For its precise definition, see
:obj:`simsopt.geo.curveobjectives.CurveCurveDistance`.  The constant
`DISTANCE_WEIGHT` is selected to balance this minimum distance penalty
against the other objectives.  Analytic derivatives are used for the
optimization.

Finally, the two remaining terms `CurvaturePenalty` and `MeanSquaredCurvaturePenalty`
are regularization terms the prevent the coils from becoming to wiggly.
These try to make the coils as smooth as possible to accomodate
possible engineering constraints.

In this first section we consider vacuum fields only, so the magnetic field
due to current in the plasma does not need to be subtracted in the
quadratic flux term. The configuration considered is the "precise QA"
case from `arXiv:2108.03711 <http://arxiv.org/pdf/2108.03711.pdf>`_,
which has two field periods and is quasi-axisymmetric.

The stage-2 optimization problem is automatically parallelized in
simsopt using OpenMP and vectorization, but MPI is not used, so the
``mpi4py`` python package is not needed. This example can be run in a
few seconds on a laptop.

The target plasma surface is given in the VMEC input file ``tests/test_files/input.LandremanPaul2021_QA``.
We load the surface using::

.. code-block::

  nphi = 32
  ntheta = 32
  filename = "tests/test_files/input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

You can adjust the directory in ``"filename"`` as appropriate for your
system. The surface could also be read in from a VMEC wout file using
:obj:`simsopt.geo.surfacerzfourier.SurfaceRZFourier.from_wout()`.
Note that VMEC does not need to be installed to initialize a surface from a
VMEC input or output file. As surface objects carry a grid of
"quadrature points" at which the position vector is evaluated
we may want these points to cover different
ranges of the toroidal angle. For this problem with stellarator
symmetry and field-period symmetry, we need only consider half of a
field period in order to evaluate integrals over the entire
surface. For this reason, the ``range`` parameter of the surface is
set to ``"half period"`` here. Possible options are ``full torus``, ``field period`` and ``half period``.

After importing the necessary classes (see ``examples/2_Intermediate/stage_two_optimization.py``),
we set the initial condition for the coils, which will be equally spaced circles.
We consider here a case with four unique coil shapes, each of which is repeated four times due to
stellarator symmetry and two-field-period symmetry, giving a total of 16 coils.
The four unique coil shapes are called the "base coils". Each copy of a coil also carries the same current,
but we will allow the unique coil shapes to have different current from each other,
as is allowed on W7-X. For this tutorial we consider the coils to be infinitesimally thin filaments.
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

  base_currents[0].fix_all()

(A ``Current`` object only has one degree of freedom, hence we can use
``fix_all()``.)  If you wish, you can fix the currents in all the
coils to force them to have the same value. Now the full set of 16
coils can be obtained using stellarator symmetry and field-period
symmetry::

  coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

To make sure that the coils class has the non-fixed degrees of freedom that
we specified, we can print the ``dof_names`` property::

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
:obj:`~simsopt.field.biotsavart.BiotSavart` object.) 
To check the size of the field normal to the target surface
before optimization we can run::

  B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
  print('Initial max B dot n:', np.max(B_dot_n))

The result is 0.19 Tesla.

We now define the objective function by stating what are the weights
used and the corresponding terms::

  # Weight on the curve lengths in the objective function. We use the `Weight`
  # class here to later easily adjust the scalar value and rerun the optimization
  # without having to rebuild the objective.
  LENGTH_WEIGHT = Weight(1e-6)

  # Threshold and weight for the coil-to-coil distance penalty in the objective function:
  CC_THRESHOLD = 0.1
  CC_WEIGHT = 1000

  # Threshold and weight for the coil-to-surface distance penalty in the objective function:
  CS_THRESHOLD = 0.3
  CS_WEIGHT = 10

  # Threshold and weight for the curvature penalty in the objective function:
  CURVATURE_THRESHOLD = 5.
  CURVATURE_WEIGHT = 1e-6

  # Threshold and weight for the mean squared curvature penalty in the objective function:
  MSC_THRESHOLD = 5
  MSC_WEIGHT = 1e-6
  
  # Define the individual terms objective function:
  Jf = SquaredFlux(s, bs)
  Jls = [CurveLength(c) for c in base_curves]
  Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
  Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
  Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
  Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

  # Form the total objective function.
  JF = Jf \
      + LENGTH_WEIGHT * sum(Jls) \
      + CC_WEIGHT * Jccdist \
      + CS_WEIGHT * Jcsdist \
      + CURVATURE_WEIGHT * sum(Jcs) \
      + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)

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

Stochastic Optimization
---------------------------

In this example we solve a stochastic version of
the :ref:`first example here <_simplest_stage2>`. As before,
the goal is to find coils that generate a specific target
normal field on a given surface. As we are still considering a vacuum
field the target is just zero.
The target equilibrium is the precise QA configuration of arXiv:2108.03711.
The complete script can be found in ``examples/2_Intermediate/stage_two_optimization_stochastic.py``.

The objective function similar to :ref:`the first example <_simplest_stage2>`
with small modifications::

    J = (1/2) Mean(\int |B dot n|^2 ds)
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + ARCLENGTH_WEIGHT * ArclengthVariation

The first term is altered to be given by the Mean of the flux instead of the
flux itself. In here, the Mean is approximated by a sample average over perturbed coils.
The coil perturbations for each coil are the sum of a 'systematic error' and a
'statistical error'.  The former satisfies rotational and stellarator symmetry,
the latter is independent for each coil.

An extra term term is also added, namely the variation of the arclength along a curve.
The idea is to avoid ill-posedness of curve objectives due to
non-uniqueness of the underlying parametrization. Essentially we want to
achieve constant arclength along the curve. Since we can not expect
perfectly constant arclength along the entire curve, this class has
some support to relax this notion. Consider a partition of the :math:`[0, 1]`
interval into intervals :math:`\{I_i\}_{i=1}^L`, and tenote the average incremental arclength
on interval :math:`I_i` by :math:`\ell_i`. This objective then penalises the variance
.. math::
    J = \mathrm{Var}(\ell_i)
it remains to choose the number of intervals :math:`L` that :math:`[0, 1]` is split into.
If ``nintervals="full"``, then the number of intervals :math:`L` is equal to the number of quadrature
points of the curve. If ``nintervals="partial"``, then the argument is as follows:
A curve in 3d space is defined uniquely by an initial point, an initial
direction, and the arclength, curvature, and torsion along the curve. For a
:mod:`simsopt.geo.curvexyzfourier.CurveXYZFourier`, the intuition is now as
follows: assuming that the curve has order :math:`p`, that means we have
:math:`3*(2p+1)` degrees of freedom in total. Assuming that three each are
required for both the initial position and direction, :math:`6p-3` are left
over for curvature, torsion, and arclength. We want to fix the arclength,
so we can afford :math:`2p-1` constraints, which corresponds to
:math:`L=2p`.


We now define the objective function by stating what are the weights
used and the corresponding terms. Besides the terms in
:ref:`the first example <_simplest_stage2>`, we additionally define::

  # Weight for the arclength variation penalty in the objective function:
  ARCLENGTH_WEIGHT = 1e-2

  # Standard deviation for the coil errors
  SIGMA = 1e-3

  # Length scale for the coil errors
  L = 0.5

  # Number of samples to approximate the mean
  N_SAMPLES = 16

  # Number of samples for out-of-sample evaluation
  N_OOS = 256

  # Objective function for the arclength variation
  Jals = [ArclengthVariation(c) for c in base_curves]

  # Objective function for the coils and its perturbations
  rg = np.random.Generator(PCG64(seed, inc=0))
  sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=1)
  Jfs = []
  curves_pert = []
  for i in range(N_SAMPLES):
      # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
      base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
      coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
      # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
      coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
      curves_pert.append([c.curve for c in coils_pert])
      bs_pert = BiotSavart(coils_pert)
      Jfs.append(SquaredFlux(s, bs_pert))
  Jmpi = MPIObjective(Jfs, comm, needs_splitting=True)

  # Form the total objective function. To do this, we can exploit the
  # fact that Optimizable objects with J() and dJ() functions can be
  # multiplied by scalars and added:
  JF = Jmpi \
      + LENGTH_WEIGHT * sum(Jls) \
      + DISTANCE_WEIGHT * Jdist \
      + CURVATURE_WEIGHT * sum(Jcs) \
      + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
      + ARCLENGTH_WEIGHT * sum(Jals)

As can be seen here, in the stochastic optimization method,
we apply two different types of errors.
The first one is the systematic error which is applied where
a random perturbation and a Gaussian Sampler with a predefined standard deviation
are added to the base curves. The second is a statistical error that is
added to each of the final coils, and is independent between coils.


Finite Beta Optimization
---------------------------

In this example, we solve a finite beta version of
the :ref:`first example here <_simplest_stage2>`.
By finite beta, it is understood that the effect of
the plasma is also taken into accout when calculating
the normal field on a given surface. Therefore, the
target quantity :math:`B_{external}\cdot \mathbf n` is no longer zero
and a virtual casing calculation is used to find its value.
The complete script can be found in ``examples/2_Intermediate/stage_two_finite_beta.py``.

We use an objective function similar to :ref:`the first example <_simplest_stage2>`
with small modifications::

    J = (1/2) \int |(B_{BiotSavart} - B_{External}) dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)

The first term, while similar to the previous examples, it
calculates the external field :math:`B_{external}` using a
virtual casing principle.


  # Resolution for the virtual casing calculation:
  vc_src_nphi = 80
  # (For the virtual casing src_ resolution, only nphi needs to be
  # specified; the theta resolution is computed automatically to
  # minimize anisotropy of the grid.)



  # Once the virtual casing calculation has been run once, the results
  # can be used for many coil optimizations. Therefore here we check to
  # see if the virtual casing output file alreadys exists. If so, load
  # the results, otherwise run the virtual casing calculation and save
  # the results.
  head, tail = os.path.split(vmec_file)
  vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
  print('virtual casing data file:', vc_filename)
  if os.path.isfile(vc_filename):
      print('Loading saved virtual casing result')
      vc = VirtualCasing.load(vc_filename)
  else:
      # Virtual casing must not have been run yet.
      print('Running the virtual casing calculation')
      vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)

  # Initialize the boundary magnetic surface:
  s = SurfaceRZFourier.from_wout(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
  total_current = Vmec(vmec_file).external_current() / (2 * s.nfp)





  # Form the total objective function. To do this, we can exploit the
  # fact that Optimizable objects with J() and dJ() functions can be
  # multiplied by scalars and added:
  JF = Jf \
      + LENGTH_PENALTY * sum(QuadraticPenalty(Jls[i], Jls[i].J()) for i in range(len(base_curves)))
