Geometric Objects
-----------------

Simsopt contains implementations of two commonly used geometric objects: curves and surfaces.

Curves
~~~~~~

A :obj:`simsopt.geo.curve.Curve` is modelled as a function :math:`\Gamma:[0, 1] \to \mathbb{R}^3`.
A curve object stores a list of :math:`n_\phi` quadrature points :math:`\{\phi_1, \ldots, \phi_{n_\phi}\} \subset [0, 1]` and returns all information about the curve, at these quadrature points.

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

.. _surfaces:

Surfaces
~~~~~~~~

The second large class of geometric objects are surfaces.  A
:obj:`simsopt.geo.surface.Surface` is modelled as a function
:math:`\Gamma:[0, 1] \times [0, 1] \to \mathbb{R}^3` and is evaluated
at quadrature points :math:`\{\phi_1, \ldots,
\phi_{n_\phi}\}\times\{\theta_1, \ldots, \theta_{n_\theta}\}`.  Here,
:math:`\phi` is the toroidal angle and :math:`\theta` is the poloidal
angle. Note that :math:`\phi` and :math:`\theta` go up to 1, not up to
:math:`2 \pi`!

In practice, you almost never use the base
:obj:`~simsopt.geo.surface.Surface` class.  Rather, you typically use
one of the subclasses corresponding to a specific parameterization.
Presently, the available subclasses are
:obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`,
:obj:`~simsopt.geo.surfacegarabedian.SurfaceGarabedian`,
:obj:`~simsopt.geo.surfacehenneberg.SurfaceHenneberg`,
:obj:`~simsopt.geo.surfacexyzfourier.SurfaceXYZFourier`,
and
:obj:`~simsopt.geo.surfacexyztensorfourier.SurfaceXYZTensorFourier`.
In many cases you can convert a surface from one type to another by going through
:obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`, as most surface types have
``to_RZFourier()`` and ``from_RZFourier()`` methods.
Note that :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`
corresponds to the surface parameterization used internally in the VMEC and SPEC codes.
However when using these codes in simsopt, any of the available surface subclasses
can be used to represent the surfaces, and simsopt will automatically handle the conversion
to :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier` when running the code.

The points :math:`\phi_j` and :math:`\theta_j` are used for evaluating
the position vector and its derivatives, for computing integrals, and
for plotting, and there are several available methods to specify these
points.  For :math:`\theta_j`, you typically specify a keyword
argument ``ntheta`` to the constructor when instantiating a surface
class. This results in a grid of ``ntheta`` uniformly spaced points
between 0 and 1, with no endpoint at 1. Alternatively, you can specify
a list or array of points to the ``quadpoints_theta`` keyword argument
when instantiating a surface class, specifying the :math:`\theta_j`
directly.  If both ``ntheta`` and ``quadpoints_theta`` are specified,
an exception will be raised.  For the :math:`\phi` coordinate, you
sometimes want points up to 1 (the full torus), sometimes up to
:math:`1/n_{fp}` (one field period), and sometimes up to :math:`1/(2
n_{fp})` (half a field period). These three cases can be selected by
setting the ``range`` keyword argument of the surface subclasses to
``"full torus"``, ``"field period"``, or ``"half period"``.
Equivalently, you can set ``range`` to the constants
``S.RANGE_FULL_TORUS``, ``S.RANGE_FIELD_PERIOD``, or
``S.RANGE_HALF_PERIOD``, where ``S`` can be
:obj:`simsopt.geo.surface.Surface` or any of its subclasses.  For all
of these cases, the ``nphi`` keyword argument can be set to the
desired number of :math:`\phi` grid points. Alternatively, you can
pass a list or array to the ``quadpoints_phi`` keyword argument of the
constructor for any Surface subclass to specify the :math:`\phi_j`
points directly.  An exception will be raised if both ``nphi`` and
``quadpoints_phi`` are specified.  For more information about these
arguments, see the
:obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier` API
documentation.

The methods available to each surface class are similar to those of
the :obj:`~simsopt.geo.curve.Curve` class:

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
~~~~~~~

The quantities that Simsopt can compute for curves and surfaces often depend on each other.
For example, the curvature or torsion of a curve both rely on ``Curve.gammadash()``; to avoid repeated calculation 
geometric objects contain a cache that is automatically managed.
If a quantity for the curve is requested, the cache is checked to see whether it was already computed.
This cache can be cleared manually by calling ``Curve.invalidate_cache()``.
This function is called everytime ``Curve.set_dofs()`` is called (and the shape of the curve changes).
