import numpy as np

from .._core.optimizable import Optimizable


class Surface(Optimizable):
    r"""
    ``Surface`` is a base class for various representations of toroidal
    surfaces in simsopt.

    A ``Surface`` is modelled as a function :math:`\Gamma:[0, 1] \times [0, 1] \to R^3` and is evaluated at quadrature points :math:`\{\phi_1, \ldots, \phi_{n_\phi}\}\times\{\theta_1, \ldots, \theta_{n_\theta}\}`.

    """

    def __init__(self):
        Optimizable.__init__(self)
        self.dependencies = []
        self.fixed = np.full(len(self.get_dofs()), False)

    def plot(self, ax=None, show=True, plot_normal=False, plot_derivative=False, scalars=None, wireframe=True):
        """
        Plot the surface using mayavi. 
        Note: the `ax` and `show` parameter can be used to plot more than one surface:

        .. code-block::

            ax = surface1.plot(show=False)
            ax = surface2.plot(ax=ax, show=False)
            surface3.plot(ax=ax, show=True)


        """
        gamma = self.gamma()

        from mayavi import mlab
        mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], scalars=scalars)
        if wireframe:
            mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], representation='wireframe', color=(0, 0, 0), opacity=0.5)

        if plot_derivative:
            dg1 = 0.05 * self.gammadash1()
            dg2 = 0.05 * self.gammadash2()
            mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], dg1[:, :, 0], dg1[:, :, 1], dg1[:, :, 2])
            mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], dg2[:, :, 0], dg2[:, :, 1], dg2[:, :, 2])
        if plot_normal:
            n = 0.005 * self.normal()
            mlab.quiver3d(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], n[:, :, 0], n[:, :, 1], n[:, :, 2])
        if show:
            mlab.show()

    def __repr__(self):
        return "Surface " + str(hex(id(self)))

    def to_RZFourier(self):
        """
        Return a :obj:`simsopt.geo.surfacerzfourier.SurfaceRZFourier` instance corresponding to the shape of this
        surface.  All subclasses should implement this abstract
        method.
        """
        raise NotImplementedError

    def cross_section(self, phi, thetas=None):
        """
        This function takes in a cylindrical angle :math:`\phi` and returns the cross
        section of the surface in that plane evaluated at `thetas`. This is
        done using the method of bisection.

        This function assumes that the surface intersection with the plane is a
        single curve.
        """

        # phi is assumed to be between [-pi, pi], so if it does not lie on that interval
        # we shift it by multiples of 2pi until it does
        phi = phi - np.sign(phi) * np.floor(np.abs(phi) / (2*np.pi)) * (2. * np.pi)
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
        self.gamma_impl(gamma, varphi, theta)

        # compute the cylindrical phi coordinate of each sampled point on the surface
        cyl_phi = np.arctan2(gamma[:, :, 1], gamma[:, :, 0])

        # reorder varphi, theta with respect to increasing cylindrical phi
        idx = np.argsort(cyl_phi, axis=0)
        cyl_phi = np.take_along_axis(cyl_phi, idx, axis=0)
        varphigrid = np.take_along_axis(varphigrid, idx, axis=0)

        # In case the target cylindrical angle "phi" lies above the first row or below the last row,
        # we must concatenate the lower row above the top row and the top row below the lower row.
        # This is allowable since the data in the matrices are periodic
        cyl_phi = np.concatenate((cyl_phi[-1, :][None, :]-2.*np.pi, cyl_phi, cyl_phi[0, :][None, :]+2.*np.pi), axis=0)
        varphigrid = np.concatenate((varphigrid[-1, :][None, :]-1., varphigrid, varphigrid[0, :][None, :]+1.), axis=0)

        # ensure that varphi does not have massive jumps.
        diff = varphigrid[1:]-varphigrid[:-1]
        pinc = np.abs(diff+1) < np.abs(diff)
        minc = np.abs(diff-1) < np.abs(diff)
        inc = pinc.astype(int) - minc.astype(int)
        prefix_sum = np.cumsum(inc, axis=0)
        varphigrid[1:] = varphigrid[1:] + prefix_sum

        # find the subintervals in varphi on which the desired cross section lies.
        # if idx_right == 0, then the subinterval must be idx_left = 0 and idx_right = 1
        idx_right = np.argmax(phi <= cyl_phi, axis=0)
        idx_right = np.where(idx_right == 0, 1, idx_right)
        idx_left = idx_right-1 

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
            phi = phi + 2.*np.pi * (pinc - minc)
            return phi

        def bisection(phia, a, phic, c):
            err = 1.
            while err > 1e-13:
                b = (a + c)/2.
                phib = varphi2phi(b, phia, phic)

                flag = (phib - phi) * (phic - phi) > 0
                # if flag is true,  then root lies on interval [a,b)
                # if flag is false, then root lies on interval [b,c]
                phia = np.where(flag, phia, phib)
                phic = np.where(flag, phib, phic)
                a = np.where(flag, a, b)
                c = np.where(flag, b, c)
                err = np.max(np.abs(a-c))
            b = (a + c)/2.
            return b          
        # bisect cyl_phi to compute the cross section
        sol = bisection(cyl_phi_left, varphi_left, cyl_phi_right, varphi_right)
        cross_section = np.zeros((sol.size, 3))
        self.gamma_lin(cross_section, sol, theta) 
        return cross_section

    def aspect_ratio(self):
        r"""
        Note: cylindrical coordinates are :math:`(R, \phi, Z)`, where :math:`\phi \in [-\pi,\pi)`
        and the angles that parametrize the surface are :math:`(\varphi, \theta) \in [0,1)^2`
        For a given surface, this function computes its aspect ratio using the VMEC
        definition:

        .. math::
            AR = R_{\text{major}} / R_{\text{minor}}

        where 

        .. math::
            R_{\text{minor}} &= \sqrt{ \overline{A} / \pi } \\
            R_{\text{major}} &= \frac{V}{2 \pi^2  R_{\text{minor}}^2} 

        and :math:`V` is the volume enclosed by the surface, and :math:`\overline{A}` is the
        average cross sectional area.
        The main difficult part of this calculation is the mean cross sectional
        area.  This is given by the integral

        .. math::
            \overline{A} = \frac{1}{2\pi} \int_{S_{\phi}} ~dS ~d\phi

        where :math:`S_\phi` is the cross section of the surface at the cylindrical angle :math:`\phi`.
        Note that :math:`\int_{S_\phi} ~dS` can be rewritten as a line integral 

        .. math::
            \int_{S_\phi}~dS &= \int_{S_\phi} ~dR dZ \\ 
            &= \int_{\partial S_\phi}  [R,0] \cdot \mathbf n/\|\mathbf n\| ~dl \\ 
            &= \int^1_{0} R \frac{\partial Z}{\partial \theta}~d\theta

        where :math:`\mathbf n = [n_R, n_Z] = [\partial Z/\partial \theta, -\partial R/\partial \theta]` is the outward pointing normal.

        Consider the surface in cylindrical coordinates terms of its angles :math:`[R(\varphi,\theta), 
        \phi(\varphi,\theta), Z(\varphi,\theta)]`.  The boundary of the cross section 
        :math:`\partial S_\phi` is given by the points :math:`\theta\rightarrow[R(\varphi(\phi,\theta),\theta),\phi, 
        Z(\varphi(\phi,\theta),\theta)]` for fixed :math:`\phi`.  The cross sectional area of :math:`S_\phi` becomes

        .. math::
            \int^{1}_{0} R(\varphi(\phi,\theta),\theta)
            \frac{\partial}{\partial \theta}[Z(\varphi(\phi,\theta),\theta)] ~d\theta

        Now, substituting this into the formula for the mean cross sectional area, we have

        .. math::
            \overline{A} = \frac{1}{2\pi}\int^{\pi}_{-\pi}\int^{1}_{0} R(\varphi(\phi,\theta),\theta)
                \frac{\partial}{\partial \theta}[Z(\varphi(\phi,\theta),\theta)] ~d\theta ~d\phi

        Instead of integrating over cylindrical :math:`\phi`, let's complete the change of variables and
        integrate over :math:`\varphi` using the mapping:

        .. math::
            [\phi,\theta] \leftarrow [\text{atan2}(y(\varphi,\theta), x(\varphi,\theta)), \theta]

        After the change of variables, the integral becomes:

        .. math::
            \overline{A} = \frac{1}{2\pi}\int^{1}_{0}\int^{1}_{0} R(\varphi,\theta) \left[\frac{\partial Z}{\partial \varphi} 
            \frac{\partial \varphi}{d \theta} + \frac{\partial Z}{\partial \theta} \right] \text{det} J ~d\theta ~d\varphi

        where :math:`\text{det}J` is the determinant of the mapping's Jacobian.

        """

        xyz = self.gamma()
        x2y2 = xyz[:, :, 0]**2 + xyz[:, :, 1]**2
        dgamma1 = self.gammadash1()
        dgamma2 = self.gammadash2()

        # compute the average cross sectional area
        J = np.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
        J[:, :, 0, 0] = (xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0])/x2y2
        J[:, :, 0, 1] = (xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0])/x2y2
        J[:, :, 1, 0] = 0.
        J[:, :, 1, 1] = 1.

        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)

        dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
        mean_cross_sectional_area = np.abs(np.mean(np.sqrt(x2y2) * dZ_dtheta * detJ))/(2 * np.pi) 

        R_minor = np.sqrt(mean_cross_sectional_area / np.pi)
        R_major = np.abs(self.volume()) / (2. * np.pi**2 * R_minor**2)

        AR = R_major/R_minor
        return AR
