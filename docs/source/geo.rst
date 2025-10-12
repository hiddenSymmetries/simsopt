Geometric objects
-----------------

Simsopt contains implementations of two types of geometric objects
that are important for stellarators: curves and surfaces. Each are
available in a variety of parameterizations, and several derivatives
are provided.  The curve and surface objects can all be found in the
:obj:`simsopt.geo` module.

.. _curves:

Curves
~~~~~~

Curves are useful for representing electromagnetic coils and the
magnetic axis.  Curves are represented in simsopt with subclasses of
the base class :obj:`simsopt.geo.Curve`.  A simsopt curve is
modelled as a function :math:`\Gamma:[0, 1] \to \mathbb{R}^3`.  Note
that the curve parameter goes up to 1, not to :math:`2\pi`.  Curves in
simsopt are assumed to be periodic in the parameter. A curve object
stores a list of :math:`n_\theta` "quadrature points" :math:`\{\theta_1,
\ldots, \theta_{n_\theta}\} \subset [0, 1]`.  A variety of class methods
return information about the curve at these quadrature points. Some of
the available methods are the following:

- :obj:`Curva.gamma() <simsoptpp.Curve.gamma>`: returns a ``(n_theta, 3)`` array containing :math:`\Gamma(\theta_i)` for :math:`i\in\{1, \ldots, n_\theta\}`, i.e. returns a list of XYZ coordinates along the curve.
- :obj:`Curve.gammadash() <simsoptpp.Curve.gammadash>`: returns a ``(n_theta, 3)`` array containing :math:`\Gamma'(\theta_i)` for :math:`i\in\{1, \ldots, n_\theta\}`, i.e. returns the (non-unit-length) tangent vector along the curve.
- :obj:`Curve.kappa() <simsoptpp.Curve.kappa>`: returns a ``(n_theta, 1)`` array containing the curvature :math:`\kappa` of the curve at the quadrature points.
- :obj:`Curve.torsion() <simsoptpp.Curve.torsion>`: returns a ``(n_theta, 1)`` array containing the torsion :math:`\tau` of the curve at the quadrature points.
- :obj:`Curve.frenet_frame() <simsopt.geo.Curve.frenet_frame>`: returns a 3-element tuple. The leading element is a ``(n_theta, 3)`` array containing the Cartesian components of the unit tangent vector at the quadrature points. Similarly, the remaining two entries of the tuple give the unit normal and binormal vectors.

The different curve classes, such as
:obj:`simsopt.geo.CurveRZFourier` and
:obj:`simsopt.geo.CurveXYZFourier` differ in the way
curves are discretized.  Each of these take an array of "dofs"
(parameters, e.g. Fourier coefficients) and turn these into a function
:math:`\Gamma:[0, 1] \to \mathbb{R}^3`.  These dofs can be queried and
set via the ``.x`` or ``.full_x`` properties; the former gives just
the non-fixed dofs, whereas the latter gives all dofs including those
that are fixed. You can also set an individual dof using its string
name.  For example, for a
:obj:`simsopt.geo.CurveXYZFourier` object ``c``, the
zero-frequency Fourier mode of the y Cartesian component can be set to
5.0 using ``c.set("yc(0)", 5.0)``.  Changing the dofs will change the
shape of the curve. Simsopt is able to compute derivatives of all
relevant quantities with respect to the discretization parameters.
For example, to compute the derivative of the coordinates of the curve
at quadrature points, one calls ``Curve.dgamma_by_dcoeff()``.  One
obtains a numpy array of shape ``(n_theta, 3, n_dofs)``, containing the
derivative of the position at every quadrature point with respect to
every degree of freedom of the curve.  In the same way one can compute
the derivative of quantities such as curvature (via
``Curve.dkappa_by_dcoeff()``) or torsion (via
``Curve.dtorsion_by_dcoeff()``).

A number of objective functions are available in
:obj:`simsopt.geo` and are computed on a
:obj:`simsopt.geo.Curve`:

- :obj:`~simsopt.geo.CurveLength`: computes the length of a curve.
- :obj:`~simsopt.geo.LpCurveCurvature`: computes a penalty based on the :math:`L_p` norm of the curvature on a curve.
- :obj:`~simsopt.geo.LpCurveTorsion`: computes a penalty based on the :math:`L_p` norm of the torsion on a curve.
- :obj:`~simsopt.geo.CurveCurveDistance`: computes a penalty term on the minimum distance between a set of curves.
- :obj:`~simsopt.geo.CurveSurfaceDistance`: computes a penalty term on the minimum distance between a set of curves and a surface.

The value of the quantity and its derivative with respect to the curve
dofs can be obtained by calling e.g., ``CurveLength.J()`` and
``CurveLength.dJ()``.

.. _surfaces:

Surfaces
~~~~~~~~

Surfaces are used to represent flux surfaces, particularly for the
boundary of MHD equilibria, and for the target surface in stage-2 coil
optimization.  Surfaces are represented in simsopt using subclasses of
the base class :obj:`simsopt.geo.Surface`.  A surface is
modelled in simsopt as a function :math:`\Gamma:[0, 1] \times [0, 1]
\to \mathbb{R}^3` and is evaluated at quadrature points
:math:`\{\phi_1, \ldots, \phi_{n_\phi}\}\times\{\theta_1, \ldots,
\theta_{n_\theta}\}`.  Here, :math:`\phi` is the toroidal angle and
:math:`\theta` is the poloidal angle. Note that :math:`\phi` and
:math:`\theta` go up to 1, not up to :math:`2 \pi`! Surfaces in
simsopt are assumed to be periodic in both angles.

In practice, you almost never use the base
:obj:`~simsopt.geo.Surface` class.  Rather, you typically use
one of the subclasses corresponding to a specific parameterization.
Presently, the available subclasses are
:obj:`~simsopt.geo.SurfaceRZFourier`,
:obj:`~simsopt.geo.SurfaceGarabedian`,
:obj:`~simsopt.geo.SurfaceHenneberg`,
:obj:`~simsopt.geo.SurfaceXYZFourier`,
and
:obj:`~simsopt.geo.SurfaceXYZTensorFourier`.
In many cases you can convert a surface from one type to another by going through
:obj:`~simsopt.geo.SurfaceRZFourier`, as most surface types have
``to_RZFourier()`` and ``from_RZFourier()`` methods.
Note that :obj:`~simsopt.geo.SurfaceRZFourier`
corresponds to the surface parameterization used internally in the VMEC and SPEC codes.
However when using these codes in simsopt, any of the available surface subclasses
can be used to represent the surfaces, and simsopt will automatically handle the conversion
to :obj:`~simsopt.geo.SurfaceRZFourier` when running the code.

The points :math:`\phi_j` and :math:`\theta_j` are used for evaluating
the position vector and its derivatives, for computing integrals, and
for plotting, and there are several available methods to specify these
points. For :math:`\theta_j`,  you can specify
a list or array of points to the ``quadpoints_theta`` keyword argument
when instantiating surface classes.
Similarly, you can pass a list or array to the
``quadpoints_phi`` keyword argument of the constructor for any Surface
subclass to specify the :math:`\phi_j` points directly.  

Often, it is desirable to specify the number of grid points instead of  
supplying ``quadpoints_phi`` and ``quadpoints_theta`` to the Surface classes.
In this case, use the class method
:meth:`~simsopt.geo.surface.Surface.from_nphi_ntheta`. Replace
``Surface`` with the desired subclass when calling the ``from_nphi_ntheta`` 
class method.
For :math:`\theta_j`, you specify a keyword
argument ``ntheta`` to the class method when instantiating a surface
class. This results in a grid of ``ntheta`` uniformly spaced points
between 0 and 1, with no endpoint at 1. 
For the :math:`\phi` coordinate, you
sometimes want points up to 1 (the full torus), sometimes up to
:math:`1/n_{fp}` (one field period), and sometimes up to :math:`1/(2
n_{fp})` (half a field period). These three cases can be selected by
setting the ``range`` keyword argument of the class method to
``"full torus"``, ``"field period"``, or ``"half period"``.
Equivalently, you can set ``range`` to the constants
``S.RANGE_FULL_TORUS``, ``S.RANGE_FIELD_PERIOD``, or
``S.RANGE_HALF_PERIOD``, where ``S`` can be
:obj:`simsopt.geo.Surface` or any of its subclasses.  Note
that the :math:`\phi` grid points begin at 0 for ``"full torus"`` and
``"field period"``, whereas for ``"half period"`` the :math:`\phi`
grid is shifted by half of the grid spacing to preserve spectral
accuracy of integration.  For all three cases, the ``nphi`` keyword
argument can be set to the desired number of :math:`\phi` grid
points. 
For more information about these arguments, see the
:obj:`~simsopt.geo.SurfaceRZFourier` API
documentation.

The methods available to each surface class are similar to those of
the :obj:`~simsopt.geo.Curve` class:

- :obj:`~simsoptpp.Surface.gamma()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`, i.e. returns a list of XYZ coordinates on the surface.
- :obj:`~simsoptpp.Surface.gammadash1()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial_\phi \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- :obj:`~simsoptpp.Surface.gammadash2()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial_\theta \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- :obj:`~simsoptpp.Surface.normal()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial_\phi \Gamma(\phi_i, \theta_j)\times \partial_\theta \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- :obj:`~simsoptpp.Surface.area()`: returns the surface area.
- :obj:`~simsoptpp.Surface.volume()`: returns the volume enclosed by the surface.
- :obj:`~simsoptpp.Surface.gammadash1dash1()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial^2_{\phi,\phi} \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- :obj:`~simsoptpp.Surface.gammadash2dash2()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial^2_{\theta,\theta} \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- :obj:`~simsoptpp.Surface.gammadash1dash2()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`\partial^2_{\theta,\phi} \Gamma(\phi_i, \theta_j)` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- :obj:`~simsoptpp.Surface.first_fund_form()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`[\partial_{\phi} \Gamma(\phi_i, \theta_j) \cdot \partial_{\phi} \Gamma(\phi_i, \theta_j), \partial_{\phi} \Gamma(\phi_i, \theta_j) \cdot \partial_{\theta} \Gamma(\phi_i, \theta_j), \partial_{\theta} \Gamma(\phi_i, \theta_j) \cdot \partial_{\theta} \Gamma(\phi_i, \theta_j)]` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}`.
- :obj:`~simsoptpp.Surface.second_fund_form()`: returns a ``(n_phi, n_theta, 3)`` array containing :math:`[\hat{\textbf{n}}(\phi_i, \theta_j) \cdot \partial^2_{\phi,\phi} \Gamma(\phi_i, \theta_j), \hat{\textbf{n}}(\phi_i, \theta_j) \cdot \partial^2_{\phi,\theta} \Gamma(\phi_i, \theta_j), \hat{\textbf{n}}(\phi_i, \theta_j) \cdot \partial^2_{\theta,\theta} \Gamma(\phi_i, \theta_j)]` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}` where :math:`\hat{\textbf{n}}` is the unit normal.
- :obj:`~simsoptpp.Surface.surface_curvatures()`: returns a ``(n_phi, n_theta, 4)`` array containing :math:`[H(\phi_i, \theta_j),K(\phi_i, \theta_j),\kappa_1(\phi_i, \theta_j),\kappa_2(\phi_i, \theta_j)]` for :math:`i\in\{1, \ldots, n_\phi\}, j\in\{1, \ldots, n_\theta\}` where :math:`H` is the mean curvature, :math:`K` is the Gaussian curvature, and :math:`\kappa_{1,2}` are the principal curvatures with :math:`\kappa_1>\kappa_2`.

A number of objective functions related to surfaces are available in :obj:`simsopt.geo`:

- :obj:`~simsopt.geo.ToroidalFlux`: computes the flux through a toroidal cross section of a :obj:`~simsopt.geo.Surface`.
- :obj:`~simsopt.geo.PrincipalCurvature`: computes a metric which penalizes large values of the principal curvatures of a given :obj:`~simsopt.geo.Surface`.

The value of the quantity and its derivative with respect to the surface dofs can be obtained by calling e.g., :obj:`ToroidalFlux.J() <simsopt.geo.ToroidalFlux.J>` and :obj:`ToroidalFlux.dJ_by_dsurfacecoefficients() <simsopt.geo.ToroidalFlux.dJ_by_dsurfacecoefficients>`.


Caching
~~~~~~~

The quantities that Simsopt can compute for curves and surfaces often
depend on each other.  For example, the curvature or torsion of a
curve both rely on :obj:`Curve.gammadash() <simsoptpp.Curve.gammadash>`; to avoid repeated
calculation, geometric objects contain a cache that is automatically
managed.  If a quantity for the curve is requested, the cache is
checked to see whether it was already computed.  This cache can be
cleared manually by calling :obj:`Curve.invalidate_cache() <simsoptpp.Curve.invalidate_cache>`.  This
function is called every time values are assigned to ``Curve.x``
(meaning the shape of the curve changes).

Graphics
~~~~~~~~

Some basic graphics functions are provided for curve and surface
objects.  To plot a single curve or surface, you can call the
``.plot()`` function of the object.  Presently, three graphics engines
are supported: matplotlib, mayavi, and plotly.  You can select the
plotting engine by passing the ``engine`` keyword argument, e.g. if
``c`` is a Curve object you can call ``c.plot(engine="mayavi")``. You
can use the ``close`` argument to control whether segments are drawn
between the last quadrature point and the first. For these and other
options, see the API documentation for
:func:`Curve.plot() <simsopt.geo.Curve.plot>` and
:func:`Surface.plot() <simsopt.geo.Surface.plot>`.

If you have multiple curve and/or surface objects, a convenient way to
plot them together on the same axes is the function
:func:`simsopt.geo.plot()`, which accepts a list of objects as
its argument. Any keywords passed to this function are passed to the
``.plot()`` methods of the individual objects, so you may wish to pass
keywords such as ``engine`` or ``close``.  Alternatively, you can also
use the ``ax`` and ``show`` arguments of the ``.plot()`` methods for
individual curve and surface objects to put them on shared axes.

It is also possible to export curve and surface objects in VTK format,
so they can be viewed in Paraview.  This functionality requires the
python package ``pyevtk``, which can be installed via ``pip install
pyevtk``. A list of curve objects can be exported using the function
:func:`simsopt.geo.curves_to_vtk()`. To export a VTK file for a
surface, call the ``.to_vtk(filename)`` function of the object.  See
:func:`Surface.to_vtk() <simsopt.geo.Surface.to_vtk>` for more details.
