Magnetic Fields
---------------

Simsopt contains several magnetic field classes available to be called
directly. Any field can be summed with any other field and/or
multiplied by a constant parameter. To get the magnetic field (or its
derivatives) at a set of points, first, an instance of that particular
magnetic field is created, then all its properties are evaluated
internally at those points and, finally, those properties can be
outputed. Below is an example that prints the components of a magnetic
field and its derivatives of a sum of a circular coil in the xy-plane
with current ``I=1.e7`` and a radius ``r0=1`` and a toroidal field
with a magnetic field ``B0=1`` at major radius ``R0=1``. This field is
evaluated at the set of ``points=[[0.5, 0.5, 0.1],[0.1, 0.1, -0.3]]``.

.. code-block::

   from simsopt.field.magneticfieldclasses import ToroidalField, CircularCoil
   
   Bfield1 = CircularCoil(I=1.e7, r0=1.)
   Bfield2 = ToroidalField(R0=1., B0=1.)
   Bfield = Bfield1 + Bfield2
   points=[[0.5, 0.5, 0.1], [0.1, 0.1, -0.3]]
   Bfield.set_points(points)
   print(Bfield.B())
   print(Bfield.dB_by_dX())

Below is a similar example where, instead of calculating the magnetic field using analytical functions from the circular coil class, it is calculated using the BiotSavart class

.. code-block::

   from simsopt.field.magneticfieldclasses import ToroidalField
   from simsopt.field.biotsavart import BiotSavart
   from simsopt.geo.curvexyzfourier import CurveXYZFourier

   coil = CurveXYZFourier(300, 1)
   coil.set_dofs([0, 0, 1., 0., 1., 0., 0., 0., 0.])
   Bfield1 = BiotSavart([coil], [1.e7])
   Bfield2 = ToroidalField(R0=1., B0=1.)
   Bfield = Bfield1 + Bfield2
   points=[[0.5, 0.5, 0.1], [0.1, 0.1, -0.3]]
   Bfield.set_points(points)
   print(Bfield.B())
   print(Bfield.dB_by_dX())


BiotSavart
~~~~~~~~~~

The :obj:`simsopt.field.biotsavart.BiotSavart` class initializes a magnetic field vector induced by a list of closed curves :math:`\Gamma_k` with electric currents :math:`I_k`. The field is given by

.. math::

  B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi

where :math:`\mu_0=4\pi 10^{-7}` is the vacuum permitivity.
As input, it takes a list of closed curves and the corresponding currents.

ToroidalField
~~~~~~~~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.ToroidalField` class initializes a toroidal magnetic field vector acording to the formula :math:`\mathbf B = B_0 \frac{R_0}{R} \mathbf e_\phi`, where :math:`R_0` and :math:`B_0` are input scalar quantities, with :math:`R_0` representing the major radius of the magnetic axis and :math:`B_0` the magnetic field at :math:`R_0`. :math:`R` is the radial coordinate of the cylindrical coordinate system :math:`(R,Z,\phi)`, so that :math:`B` has the expected :math:`1/R` dependence. Given a set of points :math:`(x,y,z)`, :math:`R` is calculated as :math:`R=\sqrt{x^2+y^2}`. The vector :math:`\mathbf e_\phi` is a unit vector pointing in the direction of increasing :math:`\phi`, with :math:`\phi` the standard azimuthal angle of the cylindrical coordinate system :math:`(R,Z,\phi)`. Given a set of points :math:`(x,y,z)`, :math:`\mathbf e_\phi` is calculated as :math:`\mathbf e_\phi=-\sin \phi \mathbf e_x+\cos \phi \mathbf e_y` with :math:`\phi=\arctan(y/x)`. 

PoloidalField
~~~~~~~~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.PoloidalField` class initializes a poloidal magnetic field vector acording to the formula :math:`\mathbf B = B_0 \frac{r}{q R_0} \mathbf e_\theta`, where :math:`R_0, q` and :math:`B_0` are input scalar quantities. :math:`R_0` represents the major radius of the magnetic axis, :math:`B_0` the magnetic field at :math:`r=R_0 q` and :math:`q` the safety factor associated with the sum of a poloidal magnetic field and a toroidal magnetic field with major radius :math:`R_0` and magnetic field on-axis :math:`B_0`. :math:`r` is the radial coordinate of the simple toroidal coordinate system :math:`(r,\phi,\theta)`. Given a set of points :math:`(x,y,z)`, :math:`r` is calculated as :math:`r=\sqrt{(\sqrt{x^2+y^2}-R_0)^2+z^2}`. The vector :math:`\mathbf e_\theta` is a unit vector pointing in the direction of increasing :math:`\theta`, with :math:`\theta` the poloidal angle in the simple toroidal coordinate system :math:`(r,\phi,\theta)`. Given a set of points :math:`(x,y,z)`, :math:`\mathbf e_\theta` is calculated as :math:`\mathbf e_\theta=-\sin \theta \cos \phi \mathbf e_x+\sin \theta \sin \phi \mathbf e_y+\cos \theta \mathbf e_z` with :math:`\phi=\arctan(y/x)` and :math:`\theta=\arctan(z/(\sqrt{x^2+y^2}-R_0))`.

ScalarPotentialRZMagneticField
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.ScalarPotentialRZMagneticField` class initializes a vacuum magnetic field :math:`\mathbf B = \nabla \Phi` defined via a scalar potential :math:`\Phi` in cylindrical coordinates :math:`(R,Z,\phi)`. The field :math:`\Phi` is given as an analytical expression and ``simsopt`` performed the necessary partial derivatives in order find :math:`\mathbf B` and its derivatives. Example: the function 

.. code-block::

   ScalarPotentialRZMagneticField("2*phi")

initializes a toroidal magnetic field :math:`\mathbf B = \nabla (2\phi)=2/R \mathbf e_\phi`.
Note: this functions needs the library ``sympy`` for the analytical derivatives.

CircularCoil
~~~~~~~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.CircularCoil` class initializes a magnetic field created by a single circular coil. It takes four input quantities: :math:`a`, the radius of the coil, :math:`\mathbf c=[c_x,c_y,c_z]`, the center of the coil, :math:`I`, the current flowing through the coil and :math:`\mathbf n`, the normal vector to the plane of the coil centered at the coil radius, which could be specified either with its three cartesian components :math:`\mathbf n=[n_x,n_y,n_z]` or as :math:`\mathbf n=[\theta,\phi]` with the spherical angles :math:`\theta` and :math:`\phi`.

The magnetic field is calculated analitically using the following expressions (`reference <https://ntrs.nasa.gov/citations/20010038494>`_)

- :math:`B_x=\frac{\mu_0 I}{2\pi}\frac{x z}{\alpha^2 \beta \rho^2}\left[(a^2+r^2)E(k^2)-\alpha^2 K(k^2)\right]`
- :math:`B_y=\frac{y}{x}B_x`
- :math:`B_z=\frac{\mu_0 I}{2\pi \alpha^2 \beta}\left[(a^2-r^2)E(k^2)+\alpha^2 K(k^2)\right]`

where :math:`\rho^2=x^2+y^2`, :math:`r^2=x^2+y^2+z^2`, :math:`\alpha^2=a^2+r^2-2a\rho`, :math:`\beta^2=a^2+r^2+2 a \rho`, :math:`k^2=1-\alpha^2/\beta^2`.

Dommaschk
~~~~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.Dommaschk` class initializes a vacuum magnetic field :math:`\mathbf B = \nabla \Phi` with a representation for the scalar potential :math:`\Phi` as proposed in `W. Dommaschk (1986), Computer Physics Communications 40, 203-218 <https://www.sciencedirect.com/science/article/pii/0010465586901098>`_. It allows to quickly generate magnetic fields with islands with only a small set of scalar quantities. Following the original reference, a toroidal field with :math:`B_0=R_0=1` is already included in the definition. As input parameters, it takes two arrays:

- The first array is an :math:`N\times2` array :math:`[(m_1,n_1),(m_2,n_2),...]` specifying which harmonic coefficients :math:`m` and :math:`n` are non-zero.
- The second array is an :math:`N\times2` array :math:`[(b_1,c_1),(b_2,c_2),...]` with :math:`b_i=b_{m_i,n_i}` and :math:`c_i=c_{m_i,n_i}` the coefficients used in the Dommaschk representation.

Reiman
~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.Reiman` initializes the magnetic field model in section 5 of `Reiman and Greenside, Computer Physics Communications 43 (1986) 157—167 <https://www.sciencedirect.com/science/article/pii/0010465586900597>`_. 
It is an analytical magnetic field representation that allows the explicit calculation of the width of the magnetic field islands. It takes as input arguments: :math:`\iota_0`, the unperturbed rotational transform, :math:`\iota_1`, the unperturbed global magnetic shear, :math:`k`, an array of integers with that specifies the Fourier modes used, :math:`\epsilon_k`, an array that specifies the coefficient in front of the Fourier modes, :math:`m_0`, the toroidal symmetry parameter (usually 1).

InterpolatedField
~~~~~~~~~~~~~~~~~

The :obj:`simsopt.field.magneticfieldclasses.InterpolatedField` function takes an existing field and interpolates it on a regular grid in :math:`r,\phi,z`. This resulting interpolant can then be evaluated very quickly.
As input arguments, it takes field: the underlying :mod:`simsopt.field.magneticfield.MagneticField` to be interpolated, degree: the degree of the piecewise polynomial interpolant, rrange: a 3-tuple of the form ``(rmin, rmax, nr)``, phirange: a 3-tuple of the form ``(phimin, phimax, nphi)``, zrange: a 3-tuple of the form ``(zmin, zmax, nz)``, extrapolate: whether to extrapolate the field when evaluate outside the integration domain or to throw an error, nfp: Whether to exploit rotational symmetry, stellsym: Whether to exploit stellarator symmetry. 

