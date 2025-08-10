Magnetic fields
---------------

Simsopt provides several representations of magnetic fields, including
the Biot-Savart field associated with coils, as well as others.
Magnetic fields are represented as subclasses of the base class
:obj:`simsopt.field.MagneticField`.  The various field
types can be scaled and summed together; for instance you can add a
purely toroidal field to the field from coils.  Each field object has
an associated set of evaluation points.  At these points, you can
query the field B, the vector potential A, and their derivatives with
respect to position or the field parameters.


Field types
^^^^^^^^^^^

Coils and BiotSavart
~~~~~~~~~~~~~~~~~~~~

In simsopt, a filamentary coil is represented by the class
:obj:`simsopt.field.Coil`. A :obj:`~simsopt.field.Coil` is a pairing of a
:obj:`~simsopt.geo.Curve` with a current magnitude. The latter
is represented by a :obj:`simsopt.field.Current` object.  For
information about Curve objects see :ref:`the page on geometric
objects <curves>`. If you wish for several filamentary curves to carry
the same current, you can reuse a :obj:`~simsopt.field.Current`
object in multiple :obj:`~simsopt.field.Coil` objects. 

Once :obj:`~simsopt.field.Coil` objects are defined, the magnetic
field they produce can be evaluated using the class
:obj:`simsopt.field.BiotSavart`. This class provides an
implementation of the Biot-Savart law

.. math::

  B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi

where :math:`\mu_0=4\pi \times 10^{-7}` is the vacuum permitivity,
:math:`\Gamma_k` is the position vector of coil :math:`k`, and :math:`I_k`
indicates the electric currents.

Example::

  import numpy as np
  from simsopt.geo import CurveXYZFourier
  from simsopt.field import Current, Coil
  from simsopt.field import BiotSavart

  # Most of the classes defined in the submodules are imported one level 
  # higher automatically. So instead of simsopt.geo.curve.CurveXYZFourier,
  # we use simsopt.geo.CurveXYZFourier. Similar is the case for classes
  # such as Current, Coil etc., defined in the field module.

  curve = CurveXYZFourier(100, 1)  # 100 = Number of quadrature points, 1 = max Fourier mode number
  curve.x = [0, 0, 1., 0., 1., 0., 0., 0., 0.]  # Set Fourier amplitudes
  coil = Coil(curve, Current(1.0e4))  # 10 kAmpere-turns
  field = BiotSavart([coil])  # Multiple coils can be included in the list 
  field.set_points(np.array([[0.5, 0.5, 0.1], [0.1, 0.1, -0.3]]))
  print(field.B())

For a more complex example of a
:obj:`~simsopt.field.BiotSavart` object used in coil
optimization, see
:simsopt_file:`examples/2_Intermediate/stage_two_optimization.py`.

ToroidalField
~~~~~~~~~~~~~

The :obj:`simsopt.field.ToroidalField` class
represents a purely toroidal magnetic field. The field is given by
:math:`\mathbf B = B_0 \frac{R_0}{R} \mathbf e_\phi`, where
:math:`R_0` and :math:`B_0` are input scalar quantities, with
:math:`R_0` representing a reference major radius, :math:`B_0` the
magnetic field at :math:`R_0`, and :math:`R` is the radial coordinate
of the cylindrical coordinate system :math:`(R,Z,\phi)`.  The vector
:math:`\mathbf e_\phi` is a unit vector pointing in the direction of
increasing :math:`\phi`, with :math:`\phi` the standard azimuthal
angle. Given Cartesian coordinates :math:`(x,y,z)`, :math:`\mathbf e_\phi`
is calculated as :math:`\mathbf e_\phi=-\sin \phi \mathbf e_x+\cos
\phi \mathbf e_y` with :math:`\phi=\arctan(y/x)`.

PoloidalField
~~~~~~~~~~~~~

The :obj:`simsopt.field.PoloidalField` class
represents a poloidal magnetic field acording to the formula
:math:`\mathbf B = B_0 \frac{r}{q R_0} \mathbf e_\theta`, where
:math:`R_0, q` and :math:`B_0` are input scalar
quantities. :math:`R_0` represents the major radius of the magnetic
axis, :math:`B_0` the magnetic field at :math:`r=R_0 q` and :math:`q`
the safety factor associated with the sum of a poloidal magnetic field
and a toroidal magnetic field with major radius :math:`R_0` and
magnetic field on-axis :math:`B_0`. :math:`r` is the radial coordinate
of the simple toroidal coordinate system
:math:`(r,\phi,\theta)`. Given a set of points :math:`(x,y,z)`,
:math:`r` is calculated as
:math:`r=\sqrt{(\sqrt{x^2+y^2}-R_0)^2+z^2}`. The vector :math:`\mathbf
e_\theta` is a unit vector pointing in the direction of increasing
:math:`\theta`, with :math:`\theta` the poloidal angle in the simple
toroidal coordinate system :math:`(r,\phi,\theta)`. Given a set of
points :math:`(x,y,z)`, :math:`\mathbf e_\theta` is calculated as
:math:`\mathbf e_\theta=-\sin \theta \cos \phi \mathbf e_x+\sin \theta
\sin \phi \mathbf e_y+\cos \theta \mathbf e_z` with
:math:`\phi=\arctan(y/x)` and
:math:`\theta=\arctan(z/(\sqrt{x^2+y^2}-R_0))`.

ScalarPotentialRZMagneticField
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The
:obj:`simsopt.field.ScalarPotentialRZMagneticField`
class initializes a vacuum magnetic field :math:`\mathbf B = \nabla
\Phi` defined via a scalar potential :math:`\Phi` in cylindrical
coordinates :math:`(R,Z,\phi)`. The field :math:`\Phi` is specified as
an analytical expression via a string argument. Simsopt performs the
necessary partial derivatives in order find :math:`\mathbf B` and its
derivatives. For example, the object
``ScalarPotentialRZMagneticField("2*phi")`` represents a toroidal
magnetic field :math:`\mathbf B = \nabla (2\phi)=2/R \mathbf e_\phi`.
Note: this functions needs the library ``sympy`` for the analytical
derivatives.

CircularCoil
~~~~~~~~~~~~

The :obj:`simsopt.field.CircularCoil` class
represents a magnetic field created by a single circular coil. It
takes four input quantities: :math:`a`, the radius of the coil,
:math:`\mathbf c=[c_x,c_y,c_z]`, the center of the coil, :math:`I`,
the current flowing through the coil and :math:`\mathbf n`, the normal
vector to the plane of the coil centered at the coil radius, which
could be specified either with its three Cartesian components
:math:`\mathbf n=[n_x,n_y,n_z]` or as :math:`\mathbf n=[\theta,\phi]`
with the spherical angles :math:`\theta` and :math:`\phi`.

The magnetic field is calculated analitically using the following
expressions (`reference
<https://ntrs.nasa.gov/citations/20010038494>`_)

- :math:`B_x=\frac{\mu_0 I}{2\pi}\frac{x z}{\alpha^2 \beta \rho^2}\left[(a^2+r^2)E(k^2)-\alpha^2 K(k^2)\right]`
- :math:`B_y=\frac{y}{x}B_x`
- :math:`B_z=\frac{\mu_0 I}{2\pi \alpha^2 \beta}\left[(a^2-r^2)E(k^2)+\alpha^2 K(k^2)\right]`

where :math:`\rho^2=x^2+y^2`, :math:`r^2=x^2+y^2+z^2`, :math:`\alpha^2=a^2+r^2-2a\rho`, :math:`\beta^2=a^2+r^2+2 a \rho`, :math:`k^2=1-\alpha^2/\beta^2`.

Dommaschk
~~~~~~~~~

The :obj:`simsopt.field.Dommaschk` class
represents a vacuum magnetic field :math:`\mathbf B = \nabla \Phi`
with basis functions for the scalar potential :math:`\Phi` described
in `W. Dommaschk (1986), Computer Physics Communications 40, 203-218
<https://www.sciencedirect.com/science/article/pii/0010465586901098>`_. This
representation provides explicit analytic formulae for vacuum fields
with a mixture of flux surfaces, islands, and chaos. Following the
original reference, a toroidal field with :math:`B_0=R_0=1` is already
included in the definition. As input parameters, it takes two arrays:

- The first array is an :math:`N\times2` array :math:`[(m_1,n_1),(m_2,n_2),...]` specifying which harmonic coefficients :math:`m` and :math:`n` are non-zero.
- The second array is an :math:`N\times2` array :math:`[(b_1,c_1),(b_2,c_2),...]` with :math:`b_i=b_{m_i,n_i}` and :math:`c_i=c_{m_i,n_i}` the coefficients used in the Dommaschk representation.

Reiman
~~~~~~

The :obj:`simsopt.field.Reiman` provides the
magnetic field model in section 5 of `Reiman and Greenside, Computer
Physics Communications 43 (1986) 157—167
<https://www.sciencedirect.com/science/article/pii/0010465586900597>`_.
It is an analytical magnetic field representation that allows the
explicit calculation of the width of the magnetic field islands.

MirrorModel
~~~~~~~~~~~

The class :obj:`simsopt.field.MirrorModel` provides an analytic model field for
magnetic mirrors, used in https://arxiv.org/abs/2305.06372 to study
the experiment WHAM. See the documentation of :obj:`~simsopt.field.MirrorModel`
for more details.

InterpolatedField
~~~~~~~~~~~~~~~~~

The :obj:`simsopt.field.InterpolatedField`
function takes an existing field and interpolates it on a regular grid
in :math:`r,\phi,z`. This resulting interpolant can then be evaluated
very quickly. This is useful for efficiently tracing field lines and
particle trajectories.

Scaling and summing fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

Magnetic field objects can be added together, either by using the
``+`` operator, or by creating an instance of the class
:obj:`simsopt.field.MagneticFieldSum`. (The ``+``
operator creates the latter.)

Magnetic fields can also be scaled by a constant. This can be accomplished either using the ``*`` operator,
or by creating an instance of the class
:obj:`simsopt.field.MagneticFieldMultiply`. (The ``*``
operator creates the latter.)

Example::

   from simsopt.field import ToroidalField, CircularCoil
   
   field1 = CircularCoil(I=1.e7, r0=1.)
   field2 = ToroidalField(R0=1., B0=1.)
   total_field = field1 + 2.5 * field2

Common operations
^^^^^^^^^^^^^^^^^

Magnetic field objects have a large number of functions available. Before evaluating the field, you must
set the evaluation points. This can be done using either Cartesian or cylindrical coordinates.
Let ``m`` be a :obj:`~simsopt.field.MagneticField` object, and suppose there are ``n`` points
at which you wish to evaluate the field.

- :func:`m.set_points_cart() <simsopt.field.MagneticField.set_points_cart>` takes a numpy array of size ``(n, 3)`` with the Cartesian coordinates ``(x, y, z)`` of the points.
- :func:`m.set_points_cyl() <simsopt.field.MagneticField.set_points_cyl>` takes a numpy array of size ``(n, 3)`` with the cylindrical coordinates ``(r, phi, z)`` of the points.
- :func:`m.set_points() <simsopt.field.MagneticField.set_points>` is shorthand for ``m.set_points_cart()``.
- :func:`m.get_points_cart() <simsoptpp.MagneticField.get_points_cart>` returns a numpy array of size ``(n, 3)`` with the Cartesian coordinates ``(x, y, z)`` of the points.
- :func:`m.get_points_cyl() <simsoptpp.MagneticField.get_points_cyl>` returns a numpy array of size ``(n, 3)`` with the cylindrical coordinates ``(r, phi, z)`` of the points.

A variety of functions are available to return the magnetic field
:math:`B`, vector potential :math:`A`, and their gradients.  The most
commonly used ones are the following:

- :func:`m.B() <simsoptpp.MagneticField.B>` returns an array of size ``(n, 3)`` with the Cartesian coordinates of :math:`B`.
- :func:`m.B_cyl() <simsoptpp.MagneticField.B_cyl>` returns an array of size ``(n, 3)`` with the cylindrical ``(r, phi, z)`` coordinates of :math:`B`.
- :func:`m.A() <simsoptpp.MagneticField.A>` returns an array of size ``(n, 3)`` with the Cartesian coordinates of :math:`A`.
- :func:`m.AbsB() <simsoptpp.MagneticField.AbsB>` returns an array of size ``(n, 1)`` with the field magnitude :math:`|B|`.
- :func:`m.dB_by_dX() <simsoptpp.MagneticField.dB_by_dX>` returns an array of size ``(n, 3, 3)`` with the Cartesian coordinates of :math:`\nabla B`. Denoting the indices
  by :math:`(i,j,l)`, the result contains  :math:`\partial_j B_l(x_i)`.
- :func:`m.d2B_by_dXdX() <simsoptpp.MagneticField.d2B_by_dXdX>` returns an array of size ``(n, 3, 3, 3)`` with the Cartesian coordinates of :math:`\nabla\nabla B`. Denoting the indices
  by :math:`(i,j,k,l)`, the result contains  :math:`\partial_k \partial_j B_l(x_i)`.
- :func:`m.dA_by_dX() <simsoptpp.MagneticField.dA_by_dX>` returns an array of size ``(n, 3, 3)`` with the Cartesian coordinates of :math:`\nabla A`. Denoting the indices
  by :math:`(i,j,l)`, the result contains  :math:`\partial_j A_l(x_i)`.
- :func:`m.d2A_by_dXdX() <simsoptpp.MagneticField.d2A_by_dXdX>` returns an array of size ``(n, 3, 3, 3)`` with the Cartesian coordinates of :math:`\nabla\nabla A`. Denoting the indices
  by :math:`(i,j,k,l)`, the result contains  :math:`\partial_k \partial_j A_l(x_i)`.
- :func:`m.GradAbsB() <simsoptpp.MagneticField.GradAbsB>` returns an array of size ``(n, 3)`` with the Cartesian components of :math:`\nabla |B|`.

Example:

.. code-block::

   import numpy as np
   from simsopt.field import CircularCoil
   
   field = CircularCoil(I=1.e7, r0=1.)
   points = np.array([[0.5, 0.5, 0.1], [0.1, 0.1, -0.3]])
   field.set_points(points)
   print(field.B())
   print(field.dB_by_dX())

