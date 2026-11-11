"""
Elliptical planar coils posed in cylindrical / toroidal geometry (JAX).

This module implements :class:`CurvePlanarEllipticalCylindrical`, a closed curve whose
**intrinsic shape** is a fixed ellipse (semi-axes ``a``, ``b``), embedded and moved using
**cylindrical coordinates** ``(R, phi, Z)`` and three rotation angles. It complements
:class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`:

- **CurvePlanarFourier** (C++ / simsoptpp): the curve in its own plane is a **Fourier
  series in polar angle** :math:`r(\\varphi)`, so the in-plane silhouette is flexible
  (many DOFs). The plane is placed in 3D via a **unit quaternion** (avoids gimbal lock)
  and a **Cartesian** center ``(X, Y, Z)``.

- **CurvePlanarEllipticalCylindrical** (this file): the in-plane shape is **not**
  Fourier-decomposed; it is **exactly an ellipse** parameterized by ``l \\in [0,1)``
  (toroidal-like parameter). Placement uses **major radius** ``R0``, **toroidal angle**
  ``phi``, **vertical shift** ``Z0``, plus rotations ``(r_rotation, phi_rotation,
  z_rotation)`` applied in the construction in ``gamma_pure``. There are **6 scalar
  DOFs** total, so this class is suited to **rigid** elliptical loops (e.g. coil
  centers with fixed cross-section), not arbitrary star-shaped planar boundaries.

**Equivalence with :class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`**

* **Shape (in the coil plane):** The ellipse’s polar radius
  :math:`\\rho(\\theta)=ab/\\sqrt{(b\\cos\\theta)^2+(a\\sin\\theta)^2}` is smooth and
  :math:`2\\pi`-periodic, so it admits a convergent Fourier series in :math:`\\theta`.
  A **high-order** ``CurvePlanarFourier`` can therefore **approximate** this ellipse as
  closely as desired. For **finite** Fourier order :math:`M`, the map
  :math:`\\theta\\mapsto\\rho(\\theta)` is generally **not** a trigonometric polynomial of
  degree :math:`M`, so there is **no** finite set of Fourier coefficients that matches the
  ellipse **exactly** everywhere—only a limiting approximation as :math:`M\\to\\infty`.
  Here, the ellipse is **closed-form** with **no** shape DOFs (only fixed ``a``, ``b``).

* **Pose in 3D:** Both representations use **six** continuous degrees of freedom for
  rigid placement of a **fixed** planar curve (three for position, three for orientation).
  ``CurvePlanarFourier`` uses a **quaternion** plus **Cartesian** ``(X,Y,Z)``; this
  module uses **cylindrical** offsets and three **Euler-like** angles in
  :func:`rotations`. Any given embedded curve could be matched in principle by solving
  for quaternion and center, but the **construction** is different: ``gamma_pure`` uses a
  specific sequence (lift, fixed matrix order, ``dr``, cylindrical shift) that does not
  coincide with “Fourier planar curve + quaternion about center,” so DOF labels are not
  interchangeable without reparameterization.

**What this representation adds**

* **Exact** ellipse geometry and **minimal** state—six scalars for pose only—when the
  modeling choice is “rigid ellipse of known ``a``, ``b``.”
* **Interpretable** major radius, toroidal angle, and height for tokamak/stellarator-style
  coil layouts, instead of quaternion + lab ``(X,Y,Z)``.
* A **single** JAX-native implementation (:class:`CurvePlanarEllipticalCylindrical`) for
  autodiff through the explicit map, without truncating the in-plane boundary.

**Trade-offs:** No in-plane **shape** optimization (unlike Fourier modes); Euler angles can
exhibit chart singularities where quaternions remain smooth; use ``CurvePlanarFourier`` when
the boundary is not an ellipse or must be Fourier-parametrized.

The helper functions below implement the ellipse in a canonical frame, conversion between
Cartesian and cylindrical coordinates, and the composed map used in :func:`gamma_pure`.
"""

import jax.numpy as jnp
import numpy as np
from .curve import JaxCurve

__all__ = ['CurvePlanarEllipticalCylindrical', 'create_equally_spaced_cylindrical_curves',
           'xyz_cyl', 'rotations', 'convert_to_cyl', 'cylindrical_shift', 'cyl_to_cart',
           'gamma_pure']


def r_ellipse(a, b, l):
    """Radial distance from the origin to a point on an ellipse in polar form.

    For parameter ``l``, the ellipse (semi-major ``a``, semi-minor ``b``) is expressed
    with polar angle :math:`\\theta = 2\\pi l` in the plane of the ellipse; this returns
    the corresponding :math:`r(\\theta)` so that
    :math:`(r\\cos\\theta,\\, r\\sin\\theta)` lies on the ellipse.

    Args:
        a: Semi-axis (used along the ``x`` direction in :func:`xyz_cyl`).
        b: Semi-axis (used along the ``z`` direction in :func:`xyz_cyl`).
        l: Parameter values in :math:`[0, 1)`, same shape as used for quadrature points.

    Returns:
        jax.Array: Radial radius, broadcast-compatible with ``l``.
    """
    return a*b / jnp.sqrt((b*jnp.cos(2.*np.pi*l))**2 + (a*jnp.sin(2.*np.pi*l))**2)


def xyz_cyl(a, b, l):
    """
    Build one closed ellipse lying in the ``x``-``z`` plane (``y = 0``), centered at the origin.

    The parameter ``l`` runs from 0 to 1 (typically ``endpoint=False`` quadrature); the
    geometric angle is :math:`2\\pi l`. The ellipse is aligned so that ``a`` is along
    ``x`` and ``b`` along ``z`` in this canonical frame, before any cylindrical placement
    or rotation in :func:`gamma_pure`.

    Args:
        a: Semi-axis along ``x`` for the canonical ellipse.
        b: Semi-axis along ``z`` for the canonical ellipse.
        l: 1D array of parameters in :math:`[0,1)` of length ``nl``.

    Returns:
        jax.Array: Shape ``(nl, 3)`` Cartesian points ``(x, y, z)`` with ``y = 0``.
    """
    # l is equally spaced from 0 to 1 - 2pi*l is an equally spaced array in theta from 0 to 2pi
    nl = l.size
    out = jnp.zeros((nl, 3))
    out = out.at[:, 0].set(r_ellipse(a, b, l)*jnp.cos(2.*np.pi*l))  # x
    out = out.at[:, 1].set(0.0)  # y
    out = out.at[:, 2].set(r_ellipse(a, b, l)*jnp.sin(2.*np.pi*l))  # z
    return out


def rotations(curve, a, b, alpha_r, alpha_phi, alpha_z, dr):
    """Apply a rigid motion: shift, Euler-like rotations, and major-radius offset.

    The input ``curve`` is expected in the same canonical frame as :func:`xyz_cyl`
    (ellipse in the ``x``-``z`` plane). The implementation:

    1. Translates along ``z`` by ``+b`` so the bottom of the ellipse touches ``z = 0``.
    2. Applies rotation matrices ``z_rot`` (:math:`\\alpha_z` about ``z``), ``y_rot``
       (:math:`\\alpha_phi` about ``y``), ``x_rot`` (:math:`\\alpha_r` about ``x``),
       in the order ``curve @ y_rot @ x_rot @ z_rot`` (row vectors on the left).
    3. Shifts toroidal major radius by ``dr`` along ``x``.
    4. Translates ``z`` by ``-b`` to undo the initial lift.

    Together with :func:`convert_to_cyl` and :func:`cylindrical_shift`, this orients the
    ellipse before it is expressed at ``(R0, phi, Z0)`` in :func:`gamma_pure`.

    Args:
        curve: Array of shape ``(N, 3)`` Cartesian points.
        a: Semi-axis ``a`` (passed for API symmetry; not used in the current formula).
        b: Semi-axis along ``z`` in the canonical ellipse; sets the vertical offset steps.
        alpha_r: Rotation angle about the local ``x`` axis (radians).
        alpha_phi: Rotation angle about the local ``y`` axis (radians).
        alpha_z: Rotation angle about the local ``z`` axis (radians).
        dr: Additional shift along ``x`` after rotation (major-radius-like offset).

    Returns:
        jax.Array: Transformed points, shape ``(N, 3)``.
    """
    z_rot = jnp.asarray(
        [[jnp.cos(alpha_z), -jnp.sin(alpha_z), 0],
         [jnp.sin(alpha_z), jnp.cos(alpha_z), 0],
         [0, 0, 1]])

    y_rot = jnp.asarray(
        [[jnp.cos(alpha_phi), 0, jnp.sin(alpha_phi)],
         [0, 1, 0],
         [-jnp.sin(alpha_phi), 0, jnp.cos(alpha_phi)]])

    x_rot = jnp.asarray(
        [[1, 0, 0],
         [0, jnp.cos(alpha_r), -jnp.sin(alpha_r)],
         [0, jnp.sin(alpha_r), jnp.cos(alpha_r)]])

    out = curve

    # I think this is doing the z rotation, so use b
    out = out.at[:, 2].set(out[:, 2] + b)  # move bottom of coil up to z=0
    out = out @ y_rot @ x_rot @ z_rot  # apply rotation in this frame
    out = out.at[:, 0].set(out[:, 0] + dr)  # dr is the major radius shift
    out = out.at[:, 2].set(out[:, 2] - b)  # move bottom of coils back down

    return out


def convert_to_cyl(a):
    """Map Cartesian ``(x, y, z)`` to cylindrical ``(R, phi, z)``.

    Uses the standard right-handed convention ``R = \\sqrt{x^2 + y^2}``,
    ``phi = \\mathrm{atan2}(y, x)``, and the same axial ``z``.

    Args:
        a: Array of shape ``(N, 3)`` with columns ``x, y, z``.

    Returns:
        jax.Array: Same shape as ``a``, columns ``(R, phi, z)``.
    """
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(jnp.sqrt(a[:, 0]**2 + a[:, 1]**2))  # R = X^2 + Y^2
    out = out.at[:, 1].set(jnp.arctan2(a[:, 1], a[:, 0]))  # phi = arctan(Y/X)
    out = out.at[:, 2].set(a[:, 2])  # z = z
    return out


def cylindrical_shift(a, dphi, dz):
    """Add increments to toroidal angle and height in cylindrical coordinates.

    ``R`` is unchanged; ``phi`` and ``z`` are shifted by ``dphi`` and ``dz``. This is
    how :func:`gamma_pure` applies the pose degrees of freedom ``phi`` and ``Z0``
    (stored in the first coefficient block together with ``R0``).

    Args:
        a: Cylindrical points ``(R, phi, z)``, shape ``(N, 3)``.
        dphi: Increment added to ``phi`` (scalar broadcast to all rows).
        dz: Increment added to ``z`` (scalar broadcast to all rows).

    Returns:
        jax.Array: Shifted cylindrical coordinates, shape ``(N, 3)``.
    """
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(a[:, 0])
    out = out.at[:, 1].set(a[:, 1]+dphi)
    out = out.at[:, 2].set(a[:, 2]+dz)
    return out


def cyl_to_cart(a):
    """Convert cylindrical ``(R, phi, z)`` to Cartesian ``(x, y, z)``.

    Args:
        a: Array of shape ``(N, 3)`` with columns ``R, phi, z``.

    Returns:
        jax.Array: Cartesian coordinates, shape ``(N, 3)``.
    """
    out = jnp.zeros(a.shape)
    out = out.at[:, 0].set(a[:, 0] * jnp.cos(a[:, 1]))  # x = R*cos(phi)
    out = out.at[:, 1].set(a[:, 0] * jnp.sin(a[:, 1]))  # y = R*sin(phi)
    out = out.at[:, 2].set(a[:, 2])  # z = z
    return out


def gamma_pure(dofs, points, a, b):
    """Evaluate the curve :math:`\\gamma` at quadrature ``points`` for DOF vector ``dofs``.

    Pipeline (matches the order of operations in code):

    1. Build the canonical ellipse with :func:`xyz_cyl` (fixed ``a``, ``b``).
    2. Apply :func:`rotations` using angles ``(r_rotation, phi_rotation, z_rotation)``
       from ``dofs[3:6]`` and radial offset ``dofs[0]`` (``R0``) as ``dr``.
    3. Convert to cylindrical with :func:`convert_to_cyl`.
    4. Shift by ``(phi, Z0)`` from ``dofs[1]`` and ``dofs[2]`` via
       :func:`cylindrical_shift`.
    5. Map back to Cartesian with :func:`cyl_to_cart`.

    Unlike :class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`, there is **no**
    Fourier spectrum for the in-plane radius; shape freedom is only through the fixed
    ellipse semi-axes ``a`` and ``b`` supplied to the class constructor.

    Args:
        dofs: Length-6 vector ``[R0, phi, Z0, r_rotation, phi_rotation, z_rotation]``.
        points: 1D quadrature parameters in :math:`[0,1)` (same convention as other JAX curves).
        a: Semi-axis of the ellipse (major in the canonical ``x`` direction in ``xyz_cyl``).
        b: Semi-axis of the ellipse (minor in the canonical ``z`` direction in ``xyz_cyl``).

    Returns:
        jax.Array: Shape ``(N, 3)`` world-frame Cartesian coordinates of the coil curve.
    """
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    g1 = xyz_cyl(a, b, points)  # generate elliptical parameterization
    g2 = rotations(g1, a, b, ypr[0], ypr[1], ypr[2], xyz[0])  # rotate and translate in R
    g3 = convert_to_cyl(g2)  # convert to R, phi, Z coordinates
    g4 = cylindrical_shift(g3, xyz[1], xyz[2])  # shift in phi and Z
    final_gamma = cyl_to_cart(g4)  # convert back to cartesian
    return final_gamma


class CurvePlanarEllipticalCylindrical(JaxCurve):
    r"""
    Closed **elliptical** curve placed in 3D using **cylindrical** pose and local rotations.

    **Geometry**

    The intrinsic curve is an ellipse with fixed semi-axes ``a`` and ``b`` (constructor
    arguments, not DOFs). In the canonical frame used by :func:`gamma_pure`, the ellipse
    lies in the :math:`x`–:math:`z` plane with :math:`y = 0`; ``a`` is along :math:`x`
    and ``b`` along :math:`z`. The curve is closed as the parameter runs over
    :math:`l \in [0,1)`.

    **Mathematical definition**

    Let :math:`l \in [0,1)` be the normalized curve parameter (quadrature points) and
    :math:`\theta = 2\pi l`. In the canonical frame, the ellipse is written in polar
    form with radius

    .. math::

       \rho(\theta) = \frac{ab}{\sqrt{(b\cos\theta)^2 + (a\sin\theta)^2}}.

    The corresponding Cartesian points (``y = 0``) are

    .. math::

       \mathbf{r}^{(0)}(\theta) =
       \begin{pmatrix}
          \rho(\theta)\cos\theta \\[4pt]
          0 \\[4pt]
          \rho(\theta)\sin\theta
       \end{pmatrix}.

    Denote the three rotation angles by :math:`\alpha_r,\alpha_\phi,\alpha_z`
    (degrees of freedom ``r_rotation``, ``phi_rotation``, ``z_rotation``) and the
    offsets by :math:`d_r` (degree of freedom ``R0``), :math:`\phi_{\mathrm{tor}}`
    (``phi``), and :math:`Z_{\mathrm{shift}}` (``Z0``). Using **row-vector** convention
    as in the implementation, :func:`rotations` maps :math:`\mathbf{r}^{(0)}` to
    :math:`\mathbf{r}^{(1)}` via

    .. math::

       \widetilde{\mathbf{r}} &= \mathbf{r}^{(0)} + (0,\,0,\,b), \\
       \widehat{\mathbf{r}} &= \widetilde{\mathbf{r}}\,
          \mathbf{R}_y(\alpha_\phi)\,\mathbf{R}_x(\alpha_r)\,\mathbf{R}_z(\alpha_z), \\
       \mathbf{r}^{(1)} &= \widehat{\mathbf{r}} + (d_r,\,0,\,-b),

    where :math:`\mathbf{R}_x`, :math:`\mathbf{R}_y`, :math:`\mathbf{R}_z` are the
    standard elementary rotation matrices about the :math:`x`, :math:`y`, and :math:`z`
    axes (same ordering as in the code).

    Cylindrical coordinates :math:`(R,\Phi,Z)` of :math:`\mathbf{r}^{(1)}` are

    .. math::

       R &= \sqrt{x^2 + y^2}, \\
       \Phi &= \operatorname{atan2}(y,\,x), \\
       Z &= z,

    with :math:`(x,y,z)` the components of :math:`\mathbf{r}^{(1)}`. The pose shifts
    :math:`\phi_{\mathrm{tor}}` and :math:`Z_{\mathrm{shift}}` act in this cylindrical
    frame as

    .. math::

       (R,\,\Phi,\,Z) \;\longmapsto\; (R,\,\Phi + \phi_{\mathrm{tor}},\,Z + Z_{\mathrm{shift}}).

    The final world-frame curve :math:`\boldsymbol{\gamma}(l)` is the Cartesian image,

    .. math::

       \boldsymbol{\gamma}(l) =
       \begin{pmatrix}
          R\cos\bigl(\Phi + \phi_{\mathrm{tor}}\bigr) \\
          R\sin\bigl(\Phi + \phi_{\mathrm{tor}}\bigr) \\
          Z + Z_{\mathrm{shift}}
       \end{pmatrix},

    with :math:`(R,\Phi,Z)` computed from :math:`\mathbf{r}^{(1)}(\theta(l))` as above.
    This is exactly the composition implemented in :func:`gamma_pure`.

    **Degrees of freedom (6 scalars)**

    The optimizable vector is::

        [R0, phi, Z0, r_rotation, phi_rotation, z_rotation]

    * ``R0``, ``phi``, ``Z0``: cylindrical placement after the local rotations described in
      :func:`rotations`—major-radius offset, toroidal angle shift, and vertical shift (see
      :func:`gamma_pure` for the exact composition).
    * ``r_rotation``, ``phi_rotation``, ``z_rotation``: angles (radians) applied in the
      ``rotations`` step, nominally about local axes associated with the coil frame.

    **Contrast with :class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier`**

    * **In-plane shape:** ``CurvePlanarFourier`` uses a **Fourier series** for the polar
      radius :math:`r(\varphi)` in the plane (many DOFs). Here the in-plane shape is a
      **fixed ellipse** parameterized only by constructor semi-axes ``a`` and ``b`` (no
      Fourier shape DOFs).
    * **3D placement:** ``CurvePlanarFourier`` uses a **quaternion** and a **Cartesian**
      center ``(X, Y, Z)``. This class uses **cylindrical** pose ``(R0, phi, Z0)`` plus three
      rotation angles in :func:`rotations` (see :func:`gamma_pure`).
    * **Typical use:** General optimized planar boundaries vs **rigid** elliptical loops
      (e.g. coil candidates with fixed elliptical cross-section).

    This class subclasses :class:`~simsopt.geo.curve.JaxCurve`; derivatives use JAX AD
    through :func:`gamma_pure`.

    Args:
        quadpoints: Integer count (uniform ``[0,1)`` points) or 1D array of quadrature
            parameters, same convention as other JAX curves.
        a: Semi-axis of the ellipse in the canonical ``x`` direction (see :func:`xyz_cyl`).
        b: Semi-axis of the ellipse in the canonical ``z`` direction (see :func:`xyz_cyl`).
        dofs: Optional length-6 initial DOF vector; if omitted, coefficients start at zero.
    """

    def __init__(self, quadpoints, a, b, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        def pure(dofs, points): return gamma_pure(dofs, points, a, b)

        self.coefficients = [np.zeros((3,)), np.zeros((3,))]
        self.a = a
        self.b = b
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=CurvePlanarEllipticalCylindrical.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=CurvePlanarEllipticalCylindrical.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """Return number of scalar DOFs (always 6: three pose, three rotation)."""
        return 3 + 3

    def get_dofs(self):
        """Return the DOF vector ``[R0, phi, Z0, r_rotation, phi_rotation, z_rotation]``."""
        return np.concatenate([self.coefficients[0], self.coefficients[1]])

    def set_dofs_impl(self, dofs):
        """Write ``dofs`` into internal coefficient blocks (pose and rotation)."""
        self.coefficients[0][:] = dofs[0:3]  # R0, phi, Z0
        self.coefficients[1][:] = dofs[3:6]  # r_rotation, phi_rotation, z_rotation

    def _make_names(self):
        """Return human-readable DOF names for serialization and optimization."""
        rtpz_name = ['R0', 'phi', 'Z0']
        angle_name = ['r_rotation', 'phi_rotation', 'z_rotation']
        return rtpz_name + angle_name


def create_equally_spaced_cylindrical_curves(ncurves, nfp, stellsym, R0, a, b, numquadpoints=32):
    """Build several identical ellipses at equally spaced toroidal angles.

    Each curve is a :class:`CurvePlanarEllipticalCylindrical` with the same ``a``, ``b``,
    and ``numquadpoints``. The toroidal angle ``phi`` of each coil is set to

    .. math::

        \\phi_i = \\frac{(i + \\tfrac{1}{2})\\, 2\\pi}{(1 + \\mathbb{1}_{\\mathrm{stellsym}})\\, \\mathrm{nfp}\\, \\mathrm{ncurves}},

    where :math:`(1 + \\mathbb{1}_{\\mathrm{stellsym}})` is ``1`` or ``2`` as in the
    implementation ``(1 + int(stellsym))``. This matches the usual stellarator-symmetry
    spacing used elsewhere (cf. :func:`~simsopt.geo.curve.create_equally_spaced_curves`),
    but builds **elliptical** :class:`CurvePlanarEllipticalCylindrical` instances rather
    than :class:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier` coils.

    All other DOFs are zero: ``Z0`` and the three rotation angles start at 0.

    Args:
        ncurves: Number of curves per field-period sector (see formula above).
        nfp: Number of field periods.
        stellsym: If True, the denominator includes a factor of ``2`` (implemented as
            ``1 + int(stellsym)``) in the toroidal spacing formula.
        R0: Major radius ``R0`` assigned to every curve.
        a: Ellipse semi-axis ``a`` (constructor argument, not a DOF).
        b: Ellipse semi-axis ``b`` (constructor argument, not a DOF).
        numquadpoints: Quadrature count passed to each curve constructor.

    Returns:
        list[CurvePlanarEllipticalCylindrical]: One curve per index ``i`` in ``range(ncurves)``.
    """
    curves = []
    for i in range(ncurves):
        curve = CurvePlanarEllipticalCylindrical(numquadpoints, a, b)
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ncurves)
        curve.set('R0', R0)
        curve.set('phi', angle)
        curve.set('Z0', 0)
        curve.set('r_rotation', 0)
        curve.set('phi_rotation', 0)
        curve.set('z_rotation', 0)
        curves.append(curve)
    return curves
