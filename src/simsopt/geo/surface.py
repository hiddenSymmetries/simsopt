import abc

import numpy as np

try:
    from pyevtk.hl import gridToVTK
except ImportError:
    gridToVTK = None

try:
    from ground.base import get_context
except ImportError:
    get_context = None

try:
    from bentley_ottmann.planar import contour_self_intersects
except ImportError:
    contour_self_intersects = None

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.dev import SimsoptRequires
from .plotting import fix_matplotlib_3d
from .._core.json import GSONable

__all__ = ['Surface', 'signed_distance_from_surface', 'SurfaceClassifier', 'SurfaceScaled', 'best_nphi_over_ntheta']


class Surface(Optimizable):
    r"""
    ``Surface`` is a base class for various representations of toroidal
    surfaces in simsopt.

    A ``Surface`` is modelled as a function
    :math:`\Gamma:[0, 1] \times [0, 1] \to R^3` and is evaluated at
    quadrature points :math:`\{\phi_1, \ldots, \phi_{n_\phi}\}\times\{\theta_1, \ldots, \theta_{n_\theta}\}`.
    """

    # Options for the 'range' parameter for setting quadpoints_phi:
    RANGE_FULL_TORUS = "full torus"
    RANGE_FIELD_PERIOD = "field period"
    RANGE_HALF_PERIOD = "half period"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_nphi_ntheta(cls, nphi=61, ntheta=62, range="full torus", nfp=1,
                         **kwargs):
        r"""
        Initializes surface classes from the specified number of grid
        points along toroidal, :math:`\phi`, and poloidal, :math:`\theta`,
        directions. Additional parameters required for surface initialization
        could be supplied as keyword arguments.

        Args:
            nphi: Number of grid points :math:`\phi_j` in the toroidal angle
              :math:`\phi`.
            ntheta: Number of grid points :math:`\theta_i` in the poloidal angle
              :math:`\theta`.
            range: Toroidal extent of the :math:`\phi` grid.
              Set to ``"full torus"`` (or equivalently ``SurfaceRZFourier.RANGE_FULL_TORUS``)
              to generate quadrature points up to 1 (with no point at 1).
              Set to ``"field period"`` (or equivalently ``SurfaceRZFourier.RANGE_FIELD_PERIOD``)
              to generate points up to :math:`1/n_{fp}` (with no point at :math:`1/n_{fp}`).
              Set to ``"half period"`` (or equivalently ``SurfaceRZFourier.RANGE_HALF_PERIOD``)
              to generate points up to :math:`1/(2 n_{fp})`, with all grid points shifted by half
              of the grid spacing in order to provide spectral convergence of integrals.
            nfp: The number of field periods.
            kwargs: Additional arguments to initialize the surface classes. Look
              at the docstrings of the specific class you are interested in.

        """
        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
            nphi, ntheta, nfp=nfp, range=range)
        return cls(quadpoints_phi=quadpoints_phi,
                   quadpoints_theta=quadpoints_theta, nfp=nfp, **kwargs)

    def get_quadpoints(nphi=None,
                       ntheta=None,
                       range=None,
                       nfp=1):
        r"""
        Sets the theta and phi grid points for Surface subclasses.
        It is typically called in when constructing Surface subclasses.

        For more information about the arguments ``nphi``, ``ntheta``,
        ``range``, ``quadpoints_phi``, and ``quadpoints_theta``, see the
        general documentation on :ref:`surfaces`.

        Args:
            nfp: The number of field periods.
            nphi: Number of grid points :math:`\phi_j` in the toroidal angle :math:`\phi`.
            ntheta: Number of grid points :math:`\theta_j` in the toroidal angle :math:`\theta`.
            range: Toroidal extent of the :math:`\phi` grid.
              Set to ``"full torus"`` (or equivalently ``Surface.RANGE_FULL_TORUS``)
              to generate points up to 1 (with no point at 1).
              Set to ``"field period"`` (or equivalently ``Surface.RANGE_FIELD_PERIOD``)
              to generate points up to :math:`1/n_{fp}` (with no point at :math:`1/n_{fp}`).
              Set to ``"half period"`` (or equivalently ``Surface.RANGE_HALF_PERIOD``)
              to generate points up to :math:`1/(2 n_{fp})`, with all grid points shifted by half
              of the grid spacing in order to provide spectral convergence of integrals.

        Returns:
            Tuple containing

            - **quadpoints_phi**: List of grid points :math:`\phi_j`.
            - **quadpoints_theta**: List of grid points :math:`\theta_j`.
        """
        return (Surface.get_phi_quadpoints(nphi=nphi, range=range, nfp=nfp),
                Surface.get_theta_quadpoints(ntheta=ntheta))

    def get_theta_quadpoints(ntheta=None):
        r"""
        Sets the theta grid points for Surface subclasses.

        Args:
            ntheta: Number of grid points :math:`\theta_j` in the toroidal angle :math:`\theta`.

        Returns:
            List of grid points :math:`\theta_j`.
        """
        # Handle theta:
        if ntheta is None:
            ntheta = 62
        return list(np.linspace(0.0, 1.0, ntheta, endpoint=False))

    def get_phi_quadpoints(nphi=None, range=None, nfp=1):
        r"""
        Sets the phi grid points for Surface subclasses.

        Args:
            nphi: Number of grid points :math:`\phi_j` in the toroidal angle :math:`\phi`.
            range: Toroidal extent of the :math:`\phi` grid.
              Set to ``"full torus"`` (or equivalently ``Surface.RANGE_FULL_TORUS``)
              to generate points up to 1 (with no point at 1).
              Set to ``"field period"`` (or equivalently ``Surface.RANGE_FIELD_PERIOD``)
              to generate points up to :math:`1/n_{fp}` (with no point at :math:`1/n_{fp}`).
              Set to ``"half period"`` (or equivalently ``Surface.RANGE_HALF_PERIOD``)
              to generate points up to :math:`1/(2 n_{fp})`, with all grid points shifted by half
              of the grid spacing in order to provide spectral convergence of integrals.
            nfp: The number of field periods.

        Returns:
            List of grid points :math:`\phi_j`.
        """

        if range is None:
            range = Surface.RANGE_FULL_TORUS
        assert range in (Surface.RANGE_FULL_TORUS, Surface.RANGE_HALF_PERIOD,
                         Surface.RANGE_FIELD_PERIOD)
        if range == Surface.RANGE_FULL_TORUS:
            div = 1
        else:
            div = nfp
        if range == Surface.RANGE_HALF_PERIOD:
            end_val = 0.5
        else:
            end_val = 1.0

        if nphi is None:
            nphi = 61
        quadpoints_phi = np.linspace(0.0, end_val / div, nphi, endpoint=False)
        # Shift by half of the grid spacing:
        if range == Surface.RANGE_HALF_PERIOD:
            dphi = quadpoints_phi[1] - quadpoints_phi[0]
            quadpoints_phi += 0.5 * dphi

        return list(quadpoints_phi)

    def plot(self, engine="matplotlib", ax=None, show=True, close=False, axis_equal=True,
             plot_normal=False, plot_derivative=False, wireframe=True, **kwargs):
        """
        Plot the surface in 3D using matplotlib/mayavi/plotly.

        Args:
            engine: Selects the graphics engine. Currently supported options are ``"matplotlib"`` (default),
              ``"mayavi"``, and ``"plotly"``.
            ax: The figure/axis to be plotted on. This argument is useful when plotting multiple
              objects on the same axes. If equal to the default ``None``, a new axis will be created.
            show: Whether to call the ``show()`` function of the graphics engine.
              Should be set to ``False`` if more objects will be plotted on the same axes.
            close: Whether to close the seams in the surface where the angles jump back to 0.
            axis_equal: For matplotlib, whether to adjust the scales of the x, y, and z axes so
              distances in each direction appear equal.
            plot_normal: Whether to plot the surface normal vectors. Only implemented for mayavi.
            plot_derivative: Whether to plot the surface derivatives. Only implemented for mayavi.
            wireframe: Whether to plot the wireframe in Mayavi.
            kwargs: Any additional arguments to pass to the plotting function, like ``color='r'``.
        Note: the ``ax`` and ``show`` parameters can be used to plot more than one surface:

        .. code-block:: python

            ax = surface1.plot(show=False)
            ax = surface2.plot(ax=ax, show=False)
            surface3.plot(ax=ax, show=True)

        Returns:
            An axis which could be passed to a further call to the graphics engine
            so multiple objects are shown together.
        """
        gamma = self.gamma()

        if plot_derivative:
            dg1 = 0.05 * self.gammadash1()
            dg2 = 0.05 * self.gammadash2()
        else:
            # No need to calculate derivatives.
            dg1 = np.array([[[1.0]]])
            dg2 = np.array([[[1.0]]])

        if plot_normal:
            normal = 0.005 * self.normal()
        else:
            # No need to calculate the normal
            normal = np.array([[[1.0]]])

        if close:
            # Always close in theta:
            gamma = np.concatenate((gamma, gamma[:, :1, :]), axis=1)
            dg1 = np.concatenate((dg1, dg1[:, :1, :]), axis=1)
            dg2 = np.concatenate((dg2, dg2[:, :1, :]), axis=1)
            normal = np.concatenate((normal, normal[:, :1, :]), axis=1)

            # Only close in phi if range == 'full torus':
            dphi = self.quadpoints_phi[1] - self.quadpoints_phi[0]
            if 1 - self.quadpoints_phi[-1] < 1.1 * dphi:
                gamma = np.concatenate((gamma, gamma[:1, :, :]), axis=0)
                dg1 = np.concatenate((dg1, dg1[:1, :, :]), axis=0)
                dg2 = np.concatenate((dg2, dg2[:1, :, :]), axis=0)
                normal = np.concatenate((normal, normal[:1, :, :]), axis=0)

        if engine == "matplotlib":
            # plot in matplotlib.pyplot
            import matplotlib.pyplot as plt

            if ax is None or ax.name != "3d":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], **kwargs)
            if axis_equal:
                fix_matplotlib_3d(ax)
            if show:
                plt.show()

        elif engine == "mayavi":
            # plot 3D surface in mayavi.mlab
            from mayavi import mlab

            mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], **kwargs)
            if wireframe:
                mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], representation='wireframe', color=(0, 0, 0),
                          opacity=0.5)

            if plot_derivative:
                mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], dg1[:, :, 0], dg1[:, :, 1], dg1[:, :, 2])
                mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], dg2[:, :, 0], dg2[:, :, 1], dg2[:, :, 2])
            if plot_normal:
                mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], normal[:, :, 0], normal[:, :, 1],
                              normal[:, :, 2])
            if show:
                mlab.show()

        elif engine == "plotly":
            # plot in plotly
            import plotly.graph_objects as go

            if "color" in list(kwargs.keys()):
                color = kwargs["color"]
                del kwargs["color"]
                kwargs["colorscale"] = [[0, color], [1, color]]
            # for plotly, ax is actually the figure
            if ax is None:
                ax = go.Figure()
            ax.add_trace(go.Surface(x=gamma[:, :, 0], y=gamma[:, :, 1], z=gamma[:, :, 2], **kwargs))
            ax.update_layout(scene_aspectmode="data")
            if show:
                ax.show()
        else:
            raise ValueError("Invalid engine option! Please use one of {matplotlib, mayavi, plotly}.")
        return ax

    @SimsoptRequires(gridToVTK is not None, "to_vtk method requires pyevtk module")
    def to_vtk(self, filename, extra_data=None):
        """
        Export the surface to a VTK format file, which can be read with
        Paraview. This function requires the ``pyevtk`` python
        package, which can be installed using ``pip install pyevtk``.

        Args:
            filename: Name of the file to write
            extra_data: An optional data field on the surface, which can be associated with a colormap in Paraview.
        """
        g = self.gamma()
        ntor = g.shape[0]
        npol = g.shape[1]
        x = self.gamma()[:, :, 0].reshape((1, ntor, npol)).copy()
        y = self.gamma()[:, :, 1].reshape((1, ntor, npol)).copy()
        z = self.gamma()[:, :, 2].reshape((1, ntor, npol)).copy()
        n = self.normal().reshape((1, ntor, npol, 3))
        dphi = self.gammadash1().reshape((1, ntor, npol, 3))
        dtheta = self.gammadash2().reshape((1, ntor, npol, 3))
        contig = np.ascontiguousarray
        pointData = {
            "dphi x dtheta": (contig(n[..., 0]), contig(n[..., 1]), contig(n[..., 2])),
            "dphi": (contig(dphi[..., 0]), contig(dphi[..., 1]), contig(dphi[..., 2])),
            "dtheta": (contig(dtheta[..., 0]), contig(dtheta[..., 1]), contig(dtheta[..., 2])),
        }
        if extra_data is not None:
            pointData = {**pointData, **extra_data}

        gridToVTK(str(filename), x, y, z, pointData=pointData)

    @abc.abstractmethod
    def to_RZFourier(self):
        """
        Return a :obj:`simsopt.geo.surfacerzfourier.SurfaceRZFourier` instance
        corresponding to the shape of this surface. All subclasses should
        implement this abstract method.
        """
        raise NotImplementedError
    
    def cross_section(self, phi, thetas=None):
        """
        This function takes in a cylindrical angle :math:`\phi` and returns the cross
        section of the surface in that plane evaluated at `thetas`. This is
        done using the method of bisection.
        This function takes in a cylindrical angle :math:`\phi` and returns
        the cross section of the surface in that plane evaluated at `thetas`.
        This is done using the method of bisection.

        This function assumes that the surface intersection with the plane is a
        single curve.
        """

        # phi is assumed to be between [-pi, pi], so if it does not lie on that interval
        # we shift it by multiples of 2pi until it does
        phi = phi - np.sign(phi) * np.floor(np.abs(phi) / (2 * np.pi)) * (2. * np.pi)
        if phi > np.pi:
            phi = phi - 2. * np.pi
        if phi < -np.pi:
            phi = phi + 2. * np.pi

        # varphi are the search intervals on which we look for the cross section in
        # at constant cylindrical phi
        # The cross section is sampled at a number of points (theta_resolution) poloidally.
        varphi = np.asarray([0., 0.5, 1.0])

        if thetas is None:
            theta = np.asarray(self.quadpoints_theta)
        elif isinstance(thetas, np.ndarray):
            theta = thetas
        elif isinstance(thetas, int):
            theta = np.linspace(0, 1, thetas, endpoint=False)
        else:
            raise NotImplementedError('Need to pass int or 1d np.array to thetas')

        varphigrid, thetagrid = np.meshgrid(varphi, theta)
        varphigrid = varphigrid.T
        thetagrid = thetagrid.T

        # sample the surface at the varphi and theta points
        gamma = np.zeros((varphigrid.shape[0], varphigrid.shape[1], 3))
        self.gamma_lin(gamma, varphigrid.flatten(), thetagrid.flatten())

        # compute the cylindrical phi coordinate of each sampled point on the surface
        cyl_phi = np.arctan2(gamma[:, :, 1], gamma[:, :, 0])

        # reorder varphi, theta with respect to increasing cylindrical phi
        idx = np.argsort(cyl_phi, axis=0)
        cyl_phi = np.take_along_axis(cyl_phi, idx, axis=0)
        varphigrid = np.take_along_axis(varphigrid, idx, axis=0)

        # In case the target cylindrical angle "phi" lies above the first row or below the last row,
        # we must concatenate the lower row above the top row and the top row below the lower row.
        # This is allowable since the data in the matrices are periodic
        cyl_phi = np.concatenate((cyl_phi[-1, :][None, :] - 2. * np.pi, cyl_phi, cyl_phi[0, :][None, :] + 2. * np.pi),
                                 axis=0)
        varphigrid = np.concatenate((varphigrid[-1, :][None, :] - 1., varphigrid, varphigrid[0, :][None, :] + 1.),
                                    axis=0)

        # ensure that varphi does not have massive jumps.
        diff = varphigrid[1:] - varphigrid[:-1]
        pinc = np.abs(diff + 1) < np.abs(diff)
        minc = np.abs(diff - 1) < np.abs(diff)
        inc = pinc.astype(int) - minc.astype(int)
        prefix_sum = np.cumsum(inc, axis=0)
        varphigrid[1:] = varphigrid[1:] + prefix_sum

        # find the subintervals in varphi on which the desired cross section lies.
        # if idx_right == 0, then the subinterval must be idx_left = 0 and idx_right = 1
        idx_right = np.argmax(phi <= cyl_phi, axis=0)
        idx_right = np.where(idx_right == 0, 1, idx_right)
        idx_left = idx_right - 1

        varphi_left = varphigrid[idx_left, np.arange(idx_left.size)]
        varphi_right = varphigrid[idx_right, np.arange(idx_right.size)]
        cyl_phi_left = cyl_phi[idx_left, np.arange(idx_left.size)]
        cyl_phi_right = cyl_phi[idx_right, np.arange(idx_right.size)]

        # this function converts varphi to cylindrical phi, ensuring that the returned angle
        # lies between left_bound and right_bound.
        def varphi2phi(varphi_in, left_bound, right_bound):
            gamma = np.zeros((varphi_in.size, 3))
            self.gamma_lin(gamma, varphi_in, theta)
            phi = np.arctan2(gamma[:, 1], gamma[:, 0])
            pinc = (phi < left_bound).astype(int)
            minc = (phi > right_bound).astype(int)
            phi = phi + 2. * np.pi * (pinc - minc)
            return phi

        def bisection(phia, a, phic, c):
            err = 1.
            while err > 1e-13:
                b = (a + c) / 2.
                phib = varphi2phi(b, phia, phic)

                flag = (phib - phi) * (phic - phi) > 0
                # if flag is true,  then root lies on interval [a,b)
                # if flag is false, then root lies on interval [b,c]
                phia = np.where(flag, phia, phib)
                phic = np.where(flag, phib, phic)
                a = np.where(flag, a, b)
                c = np.where(flag, b, c)
                err = np.max(np.abs(a - c))
            b = (a + c) / 2.
            return b

        # bisect cyl_phi to compute the cross section
        sol = bisection(cyl_phi_left, varphi_left, cyl_phi_right, varphi_right)
        cross_section = np.zeros((sol.size, 3))
        self.gamma_lin(cross_section, sol, theta)
        return cross_section
    
    @SimsoptRequires(get_context is not None, "is_self_intersecting requires ground package")
    @SimsoptRequires(contour_self_intersects is not None, "is_self_intersecting requires the bentley_ottmann package")
    def is_self_intersecting(self, angle=0., thetas=None):
        r"""
        This function computes a cross section of self at the input cylindrical angle.  Then,
        approximating the cross section as a piecewise linear polyline, the Bentley-Ottmann algorithm
        is used to check for self-intersecting segments of the cross section.  NOTE: if this function returns False,
        the surface may still be self-intersecting away from angle.

        Args:
            angle: the cylindrical angle at which we would like to check whether the surface is self-intersecting.  Note that a
                   surface might not be self-intersecting at a given angle, but may be self-intersecting elsewhere.  To be certain
                   that the surface is not self-intersecting, it is recommended to run this check at multiple angles.  Also note
                   that angle is assumed to be in radians, and not divided by 2*pi.
            thetas: the number of uniformly spaced points to compute poloidally in a cross section.  If None, then there will be
                    surface.quadpoints_theta.size uniformly space points in the cross section.
        Returns:
            True if surface is self-intersecting at angle, else False.
        """

        cs = self.cross_section(angle, thetas=thetas)
        R = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
        Z = cs[:, 2]
    
        context = get_context()
        Point, Contour = context.point_cls, context.contour_cls
        contour = Contour([Point(R[i], Z[i]) for i in range(cs.shape[0])])
        return contour_self_intersects(contour)

    def aspect_ratio(self):
        r"""
        Note: cylindrical coordinates are :math:`(R, \phi, Z)`, where
        :math:`\phi \in [-\pi,\pi)` and the angles that parametrize the
        surface are :math:`(\varphi, \theta) \in [0,1)^2`
        For a given surface, this function computes its aspect ratio using
        the VMEC definition:

        .. math::
            AR = R_{\text{major}} / R_{\text{minor}}

        where

        .. math::
            R_{\text{minor}} &= \sqrt{ \overline{A} / \pi } \\
            R_{\text{major}} &= \frac{V}{2 \pi^2  R_{\text{minor}}^2}

        and :math:`V` is the volume enclosed by the surface, and
        :math:`\overline{A}` is the average cross sectional area.

        """

        R_minor = self.minor_radius()
        R_major = self.major_radius()
        AR = R_major/R_minor
        return AR

    def daspect_ratio_by_dcoeff(self):
        """
        Return the derivative of the aspect ratio with respect to the surface coefficients
        """

        R_minor = self.minor_radius()
        R_major = self.major_radius()
        dAR_ds = (self.dmajor_radius_by_dcoeff()*R_minor - self.dminor_radius_by_dcoeff() * R_major)/R_minor**2
        return dAR_ds

    def d2aspect_ratio_by_dcoeff_dcoeff(self):
        """
        Return the derivative of the aspect ratio with respect to the surface coefficients
        """
        r = self.minor_radius()
        dr_ds = self.dminor_radius_by_dcoeff()[:, None]
        dr_dt = self.dminor_radius_by_dcoeff()[None, :]
        d2r_dsdt = self.d2minor_radius_by_dcoeff_dcoeff()
        R = self.major_radius()
        dR_ds = self.dmajor_radius_by_dcoeff()[:, None]
        dR_dt = self.dmajor_radius_by_dcoeff()[None, :]
        d2R_dsdt = self.d2major_radius_by_dcoeff_dcoeff()

        return (2*R*dr_dt*dr_ds)/r**3 - (dR_dt*dr_ds)/r**2 - (dr_dt*dR_ds)/r**2 - (R*d2r_dsdt)/r**2 + d2R_dsdt/r

    def minor_radius(self):
        r"""
        Return the minor radius of the surface using the formula

        .. math::
            R_{\text{minor}} &= \sqrt{ \overline{A} / \pi }

        where :math:`\overline{A}` is the average cross sectional area.

        """

        R_minor = np.sqrt(self.mean_cross_sectional_area() / np.pi)
        return R_minor

    def dminor_radius_by_dcoeff(self):
        """
        Return the derivative of the minor radius wrt surface coefficients

        """

        return (0.5/np.pi)*self.dmean_cross_sectional_area_by_dcoeff()/np.sqrt(self.mean_cross_sectional_area() / np.pi)

    def d2minor_radius_by_dcoeff_dcoeff(self):
        """
        Return the second derivative of the minor radius wrt surface coefficients

        """
        A = self.mean_cross_sectional_area()
        dA_ds = self.dmean_cross_sectional_area_by_dcoeff()[:, None]
        dA_dt = self.dmean_cross_sectional_area_by_dcoeff()[None, :]
        d2A_ds2 = self.d2mean_cross_sectional_area_by_dcoeff_dcoeff()
        return (-(dA_dt*dA_ds) + 2*A*d2A_ds2)/(4*np.sqrt(np.pi)*A**(3/2))

    def major_radius(self):
        r"""
        Return the major radius of the surface using the formula

        .. math::
            R_{\text{major}} = \frac{V}{2 \pi^2  R_{\text{minor}}^2}

        where :math:`\overline{A}` is the average cross sectional area,
        and :math:`R_{\text{minor}}` is the minor radius of the surface.

        """

        R_minor = self.minor_radius()
        R_major = np.abs(self.volume()) / (2. * np.pi**2 * R_minor**2)
        return R_major

    def dmajor_radius_by_dcoeff(self):
        """
        Return the derivative of the major radius wrt surface coefficients
        """

        mean_area = self.mean_cross_sectional_area()
        dmean_area_ds = self.dmean_cross_sectional_area_by_dcoeff()

        dR_major_ds = (-self.volume() * dmean_area_ds + self.dvolume_by_dcoeff() * mean_area) / mean_area**2
        return dR_major_ds * np.sign(self.volume()) / (2. * np.pi)

    def d2major_radius_by_dcoeff_dcoeff(self):
        """
        Return the second derivative of the major radius wrt surface coefficients
        """
        V = self.volume()
        dV_ds = self.dvolume_by_dcoeff()[:, None]
        dV_dt = self.dvolume_by_dcoeff()[None, :]
        d2V_dsdt = self.d2volume_by_dcoeffdcoeff()
        r = self.minor_radius()
        dr_ds = self.dminor_radius_by_dcoeff()[:, None]
        dr_dt = self.dminor_radius_by_dcoeff()[None, :]
        d2r_dsdt = self.d2minor_radius_by_dcoeff_dcoeff()

        return ((6*V*dr_dt*dr_ds)/r**4 - (2*dV_dt*dr_ds)/r**3 - \
                (2*dr_dt*dV_ds)/r**3 - (2*V*d2r_dsdt)/r**3 + \
                d2V_dsdt/r**2) * np.sign(V)/(2*np.pi**2)

    def mean_cross_sectional_area(self):
        r"""
        Note: cylindrical coordinates are :math:`(R, \phi, Z)`, where
        :math:`\phi \in [-\pi,\pi)` and the angles that parametrize the
        surface are :math:`(\varphi, \theta) \in [0,1)^2`.
        The mean cross sectional area is given by the integral

        .. math::
            \overline{A} = \frac{1}{2\pi} \int_{S_{\phi}} ~dS ~d\phi

        where :math:`S_\phi` is the cross section of the surface at the
        cylindrical angle :math:`\phi`.
        Note that :math:`\int_{S_\phi} ~dS` can be rewritten as a line integral

        .. math::
            \int_{S_\phi}~dS &= \int_{S_\phi} ~dR dZ \\
            &= \int_{\partial S_\phi}  [R,0] \cdot \mathbf n/\|\mathbf n\| ~dl \\
            &= \int^1_{0} R \frac{\partial Z}{\partial \theta}~d\theta

        where :math:`\mathbf n = [n_R, n_Z] = [\partial Z/\partial \theta, -\partial R/\partial \theta]`
        is the outward pointing normal.
        Consider the surface in cylindrical coordinates terms of its angles
        :math:`[R(\varphi,\theta), \phi(\varphi,\theta), Z(\varphi,\theta)]`.
        The boundary of the cross section :math:`\partial S_\phi` is given
        by the points :math:`\theta\rightarrow[R(\varphi(\phi,\theta),\theta),\phi,
        Z(\varphi(\phi,\theta),\theta)]` for fixed :math:`\phi`. The cross
        sectional area of :math:`S_\phi` becomes

        .. math::
            \int^{1}_{0} R(\varphi(\phi,\theta),\theta)
            \frac{\partial}{\partial \theta}[Z(\varphi(\phi,\theta),\theta)] ~d\theta

        Now, substituting this into the formula for the mean cross sectional
        area, we have

        .. math::
            \overline{A} = \frac{1}{2\pi}\int^{\pi}_{-\pi}\int^{1}_{0} R(\varphi(\phi,\theta),\theta)
                \frac{\partial}{\partial \theta}[Z(\varphi(\phi,\theta),\theta)] ~d\theta ~d\phi

        Instead of integrating over cylindrical :math:`\phi`, let's complete
        the change of variables and integrate over :math:`\varphi` using the
        mapping:

        .. math::
            [\phi,\theta] \leftarrow [\text{atan2}(y(\varphi,\theta), x(\varphi,\theta)), \theta]

        After the change of variables, the integral becomes:

        .. math::
            \overline{A} = \frac{1}{2\pi}\int^{1}_{0}\int^{1}_{0} R(\varphi,\theta) \left[\frac{\partial Z}{\partial \varphi}
            \frac{\partial \varphi}{d \theta} + \frac{\partial Z}{\partial \theta} \right] \text{det} J ~d\theta ~d\varphi

        where :math:`\text{det}J` is the determinant of the mapping's Jacobian.

        """

        xyz = self.gamma()
        x2y2 = xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2
        dgamma1 = self.gammadash1()
        dgamma2 = self.gammadash2()

        # compute the average cross sectional area
        J = np.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
        J[:, :, 0, 0] = (xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0]) / x2y2
        J[:, :, 0, 1] = (xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0]) / x2y2
        J[:, :, 1, 0] = 0.
        J[:, :, 1, 1] = 1.

        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)

        dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
        mean_cross_sectional_area = np.abs(np.mean(np.sqrt(x2y2) * dZ_dtheta * detJ))/(2 * np.pi)
        return mean_cross_sectional_area

    def dmean_cross_sectional_area_by_dcoeff(self):
        """
        Return the derivative of the mean cross sectional area wrt surface coefficients
        """

        g = self.gamma()
        g1 = self.gammadash1()
        g2 = self.gammadash2()

        dg_ds = self.dgamma_by_dcoeff()
        dg1_ds = self.dgammadash1_by_dcoeff()
        dg2_ds = self.dgammadash2_by_dcoeff()

        x = g[:, :, 0, None]
        y = g[:, :, 1, None]

        dx_ds = dg_ds[:, :, 0, :]
        dy_ds = dg_ds[:, :, 1, :]

        r = np.sqrt(x**2+y**2)
        dr_ds = (x*dx_ds+y*dy_ds)/r

        xvarphi = g1[:, :, 0, None]
        yvarphi = g1[:, :, 1, None]
        zvarphi = g1[:, :, 2, None]

        xtheta = g2[:, :, 0, None]
        ytheta = g2[:, :, 1, None]
        ztheta = g2[:, :, 2, None]

        dxvarphi_ds = dg1_ds[:, :, 0, :]
        dyvarphi_ds = dg1_ds[:, :, 1, :]
        dzvarphi_ds = dg1_ds[:, :, 2, :]

        dxtheta_ds = dg2_ds[:, :, 0, :]
        dytheta_ds = dg2_ds[:, :, 1, :]
        dztheta_ds = dg2_ds[:, :, 2, :]

        mean_area = np.mean((1/r) * (ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta)))/(2.*np.pi)
        dmean_area_ds = np.mean((1/(r**2))*((xvarphi * y * ztheta - xtheta * y * zvarphi + x * (-yvarphi * ztheta + ytheta * zvarphi)) * dr_ds + r * (-zvarphi * (ytheta * dx_ds - y * dxtheta_ds - xtheta * dy_ds + x * dytheta_ds) + ztheta * (yvarphi * dx_ds - y * dxvarphi_ds - xvarphi * dy_ds + x * dyvarphi_ds) + (-xvarphi * y + x * yvarphi) * dztheta_ds + (xtheta * y - x * ytheta) * dzvarphi_ds)), axis=(0, 1))
        return np.sign(mean_area) * dmean_area_ds/(2*np.pi)

    def d2mean_cross_sectional_area_by_dcoeff_dcoeff(self):
        """
        Return the second derivative of the mean cross sectional area wrt surface coefficients
        """

        g = self.gamma()
        g1 = self.gammadash1()
        g2 = self.gammadash2()

        dg_ds = self.dgamma_by_dcoeff()
        dg1_ds = self.dgammadash1_by_dcoeff()
        dg2_ds = self.dgammadash2_by_dcoeff()

        x = g[:, :, 0, None, None]
        y = g[:, :, 1, None, None]

        dx_ds = dg_ds[:, :, 0, :, None]
        dy_ds = dg_ds[:, :, 1, :, None]

        dx_dt = dg_ds[:, :, 0, None, :]
        dy_dt = dg_ds[:, :, 1, None, :]

        r = np.sqrt(x**2+y**2)
        dr_ds = (x*dx_ds+y*dy_ds)/r
        dr_dt = (x*dx_dt+y*dy_dt)/r
        dr2_dsdt = -((2*x*dx_dt + 2*y*dy_dt)*(2*x*dx_ds + 2*y*dy_ds))/(4*(x**2 + y**2)**(3/2)) + (2*dx_dt*dx_ds + 2*dy_dt*dy_ds)/(2*r)
        
        xvarphi = g1[:, :, 0, None, None]
        yvarphi = g1[:, :, 1, None, None]
        zvarphi = g1[:, :, 2, None, None]

        xtheta = g2[:, :, 0, None, None]
        ytheta = g2[:, :, 1, None, None]
        ztheta = g2[:, :, 2, None, None]

        dxvarphi_ds = dg1_ds[:, :, 0, :, None]
        dyvarphi_ds = dg1_ds[:, :, 1, :, None]
        dzvarphi_ds = dg1_ds[:, :, 2, :, None]

        dxtheta_ds = dg2_ds[:, :, 0, :, None]
        dytheta_ds = dg2_ds[:, :, 1, :, None]
        dztheta_ds = dg2_ds[:, :, 2, :, None]

        dxvarphi_dt = dg1_ds[:, :, 0, None, :]
        dyvarphi_dt = dg1_ds[:, :, 1, None, :]
        dzvarphi_dt = dg1_ds[:, :, 2, None, :]

        dxtheta_dt = dg2_ds[:, :, 0, None, :]
        dytheta_dt = dg2_ds[:, :, 1, None, :]
        dztheta_dt = dg2_ds[:, :, 2, None, :]

        mean_area = np.mean((1/r) * (ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta)))/(2.*np.pi)
        d2mean_area_ds2 = np.sign(mean_area)*np.mean((2*(-(xvarphi*y*ztheta) + xtheta*y*zvarphi + x*(yvarphi*ztheta - \
                                                     ytheta*zvarphi))*dr_dt*dr_ds - r*((yvarphi*ztheta - \
                                                     ytheta*zvarphi)*dx_dt + y*zvarphi*dxtheta_dt - y*ztheta*dxvarphi_dt - \
                                                     xvarphi*ztheta*dy_dt + xtheta*zvarphi*dy_dt - xvarphi*y*dztheta_dt + \
                                                     xtheta*y*dzvarphi_dt + x*(-(zvarphi*dytheta_dt) + ztheta*dyvarphi_dt \
                                                     + yvarphi*dztheta_dt - ytheta*dzvarphi_dt))*dr_ds - \
                                                     r*dr_dt*((yvarphi*ztheta - ytheta*zvarphi)*dx_ds + \
                                                     y*zvarphi*dxtheta_ds - y*ztheta*dxvarphi_ds - xvarphi*ztheta*dy_ds + \
                                                     xtheta*zvarphi*dy_ds - xvarphi*y*dztheta_ds + xtheta*y*dzvarphi_ds + \
                                                     x*(-(zvarphi*dytheta_ds) + ztheta*dyvarphi_ds + yvarphi*dztheta_ds - \
                                                     ytheta*dzvarphi_ds)) + r**2*((-(zvarphi*dytheta_dt) + \
                                                     ztheta*dyvarphi_dt + yvarphi*dztheta_dt - ytheta*dzvarphi_dt)*dx_ds + \
                                                     zvarphi*dy_dt*dxtheta_ds + y*dzvarphi_dt*dxtheta_ds - \
                                                     ztheta*dy_dt*dxvarphi_ds - y*dztheta_dt*dxvarphi_ds + \
                                                     zvarphi*dxtheta_dt*dy_ds - ztheta*dxvarphi_dt*dy_ds - \
                                                     xvarphi*dztheta_dt*dy_ds + xtheta*dzvarphi_dt*dy_ds - \
                                                     y*dxvarphi_dt*dztheta_ds - xvarphi*dy_dt*dztheta_ds + \
                                                     y*dxtheta_dt*dzvarphi_ds + xtheta*dy_dt*dzvarphi_ds + \
                                                     dx_dt*(-(zvarphi*dytheta_ds) + ztheta*dyvarphi_ds + \
                                                     yvarphi*dztheta_ds - ytheta*dzvarphi_ds) + \
                                                     x*(-(dzvarphi_dt*dytheta_ds) + dztheta_dt*dyvarphi_ds + \
                                                     dyvarphi_dt*dztheta_ds - dytheta_dt*dzvarphi_ds)) + \
                                                     r*(xvarphi*y*ztheta - xtheta*y*zvarphi + x*(-(yvarphi*ztheta) + \
                                                     ytheta*zvarphi))*dr2_dsdt)/r**3, axis=(0, 1))/(2*np.pi) # noqa
        return d2mean_area_ds2

    def arclength_poloidal_angle(self):
        """
        Computes poloidal angle based on arclenth along magnetic surface at
        constant phi. The resulting angle is in the range [0,1]. This is required
        for evaluating the adjoint shape gradient for free-boundary calculations.

        Returns:
            2d array of shape ``(numquadpoints_phi, numquadpoints_theta)``
            containing the arclength poloidal angle
        """
        gamma = self.gamma()
        X = gamma[:, :, 0]
        Y = gamma[:, :, 1]
        Z = gamma[:, :, 2]
        R = np.sqrt(X ** 2 + Y ** 2)

        theta_arclength = np.zeros_like(gamma[:, :, 0])
        nphi = len(theta_arclength[:, 0])
        ntheta = len(theta_arclength[0, :])
        for iphi in range(nphi):
            for itheta in range(1, ntheta):
                dr = np.sqrt((R[iphi, itheta] - R[iphi, itheta - 1]) ** 2
                             + (Z[iphi, itheta] - Z[iphi, itheta - 1]) ** 2)
                theta_arclength[iphi, itheta] = \
                    theta_arclength[iphi, itheta - 1] + dr
            dr = np.sqrt((R[iphi, 0] - R[iphi, -1]) ** 2
                         + (Z[iphi, 0] - Z[iphi, -1]) ** 2)
            L = theta_arclength[iphi, -1] + dr
            theta_arclength[iphi, :] = theta_arclength[iphi, :] / L
        return theta_arclength

    def interpolate_on_arclength_grid(self, function, theta_evaluate):
        """
        Interpolate function onto the theta_evaluate grid in the arclength
        poloidal angle. This is required for evaluating the adjoint shape gradient
        for free-boundary calculations.

        Returns:
            function_interpolated: 2d array (numquadpoints_phi,numquadpoints_theta)
                defining interpolated function on arclength angle along curve
                at constant phi
        """
        from scipy import interpolate

        theta_arclength = self.arclength_poloidal_angle()
        function_interpolated = np.zeros_like(function)
        nphi = len(theta_arclength[:, 0])
        for iphi in range(nphi):
            f = interpolate.InterpolatedUnivariateSpline(
                theta_arclength[iphi, :], function[iphi, :])
            function_interpolated[iphi, :] = f(theta_evaluate[iphi, :])

        return function_interpolated


def signed_distance_from_surface(xyz, surface):
    """
    Compute the signed distances from points ``xyz`` to a surface.  The sign is
    positive for points inside the volume surrounded by the surface.
    """
    gammas = surface.gamma().reshape((-1, 3))
    from scipy.spatial import KDTree
    tree = KDTree(gammas)
    _, mins = tree.query(xyz, k=1)  # find closest points on the surface

    n = surface.unitnormal().reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]

    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = np.sum((xyz-gammamins) * nmins, axis=1)

    a_point_in_the_surface = np.mean(surface.gamma()[0, :, :], axis=0)
    sign_of_interiorpoint = np.sign(np.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))

    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists


class SurfaceClassifier():
    r"""
    Takes in a toroidal surface and constructs an interpolant of the signed distance function
    :math:`f:R^3\to R` that is positive inside the volume contained by the surface,
    (approximately) zero on the surface, and negative outisde the volume contained by the surface.
    """

    def __init__(self, surface, p=1, h=0.05):
        """
        Args:
            surface: the surface to contruct the distance from.
            p: degree of the interpolant
            h: grid resolution of the interpolant
        """
        gammas = surface.gamma()
        r = np.linalg.norm(gammas[:, :, :2], axis=2)
        z = gammas[:, :, 2]
        rmin = max(np.min(r) - 0.1, 0.)
        rmax = np.max(r) + 0.1
        zmin = np.min(z) - 0.1
        zmax = np.max(z) + 0.1

        self.zrange = (zmin, zmax)
        self.rrange = (rmin, rmax)

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*np.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)

        def fbatch(rs, phis, zs):
            xyz = np.zeros((len(rs), 3))
            xyz[:, 0] = rs * np.cos(phis)
            xyz[:, 1] = rs * np.sin(phis)
            xyz[:, 2] = zs
            return list(signed_distance_from_surface(xyz, surface))

        rule = sopp.UniformInterpolationRule(p)
        self.dist = sopp.RegularGridInterpolant3D(
            rule, [rmin, rmax, nr], [0., 2*np.pi, nphi], [zmin, zmax, nz], 1, True)
        self.dist.interpolate_batch(fbatch)

    def evaluate_xyz(self, xyz):
        rphiz = np.zeros_like(xyz)
        rphiz[:, 0] = np.linalg.norm(xyz[:, :2], axis=1)
        rphiz[:, 1] = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2*np.pi)
        rphiz[:, 2] = xyz[:, 2]
        # initialize to -1 since the regular grid interpolant will just keep
        # that value when evaluated outside of bounds
        d = -np.ones((xyz.shape[0], 1))
        self.dist.evaluate_batch(rphiz, d)
        return d

    def evaluate_rphiz(self, rphiz):
        # initialize to -1 since the regular grid interpolant will just keep
        # that value when evaluated outside of bounds
        d = -np.ones((rphiz.shape[0], 1))
        self.dist.evaluate_batch(rphiz, d)
        return d

    @SimsoptRequires(gridToVTK is not None,
                     "to_vtk method requires pyevtk module")
    def to_vtk(self, filename, h=0.01):

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*np.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)
        rs = np.linspace(self.rrange[0], self.rrange[1], nr)
        phis = np.linspace(0, 2*np.pi, nphi)
        zs = np.linspace(self.zrange[0], self.zrange[1], nz)

        R, Phi, Z = np.meshgrid(rs, phis, zs)
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        Z = Z

        RPhiZ = np.zeros((R.size, 3))
        RPhiZ[:, 0] = R.flatten()
        RPhiZ[:, 1] = Phi.flatten()
        RPhiZ[:, 2] = Z.flatten()
        vals = -np.ones((R.size, 1))
        self.dist.evaluate_batch(RPhiZ, vals)
        vals = vals.reshape(R.shape)
        gridToVTK(filename, X, Y, Z, pointData={"levelset": vals})


class SurfaceScaled(Optimizable):
    """
    Allows you to take any Surface class and scale the dofs. This is
    useful for stage-1 optimization.
    """

    def __init__(self, surf, scale_factors):
        self.surf = surf
        self.scale_factors = scale_factors
        super().__init__(x0=surf.x / scale_factors, names=surf.local_dof_names)

    def recompute_bell(self, parent=None):
        self.surf.local_full_x = self.local_full_x * self.scale_factors

    def to_RZFourier(self):
        return self.surf.to_RZFourier()

    def update_fixed(self):
        """
        Copy the fixed status from self.surf to self.
        """
        for j, is_free in enumerate(self.surf.local_dofs_free_status):
            if is_free:
                self.unfix(j)
            else:
                self.fix(j)

    def as_dict(self, serial_objs_dict) -> dict:
        return GSONable.as_dict(self, serial_objs_dict=serial_objs_dict)


def best_nphi_over_ntheta(surf):
    """
    Given a surface, estimate the ratio of ``nphi / ntheta`` that
    minimizes the mesh anisotropy. This is useful for improving speed
    and accuracy of the virtual casing calculation. The result refers
    to the number of grid points in ``phi`` covering the full torus,
    not just one field period or half a field period. The input
    surface need not have ``range=="full torus"`` however; any
    ``range`` will work.

    The result of this function will depend somewhat on the quadrature
    points of the input surface, but the dependence should be weak.

    Args:
        surf: A surface object.

    Returns:
        float with the best ratio ``nphi / ntheta``.
    """
    gammadash1 = np.linalg.norm(surf.gammadash1(), axis=2)
    gammadash2 = np.linalg.norm(surf.gammadash2(), axis=2)
    ratio = gammadash1 / gammadash2
    return np.sqrt(np.max(ratio) / np.max(1 / ratio))

