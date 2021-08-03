from math import sin, cos

import numpy as np
from jax import vjp, jacfwd, jvp
from .jit import jit
import jax.numpy as jnp
from monty.dev import requires

import matplotlib.pyplot as plt

try:
    from myavi import mlab
except ImportError:
    mlab = None

import simsoptpp as sopp
from .._core.graph_optimizable import Optimizable


@jit
def incremental_arclength_pure(d1gamma):
    """
    This function is used in a Python+Jax implementation of the curve arc length formula.
    """

    return jnp.linalg.norm(d1gamma, axis=1)


incremental_arclength_vjp = jit(lambda d1gamma, v: vjp(lambda d1g: incremental_arclength_pure(d1g), d1gamma)[1](v)[0])


@jit
def kappa_pure(d1gamma, d2gamma):
    """
    This function is used in a Python+Jax implementation of formula for curvature.
    """

    return jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)/jnp.linalg.norm(d1gamma, axis=1)**3


kappavjp0 = jit(lambda d1gamma, d2gamma, v: vjp(lambda d1g: kappa_pure(d1g, d2gamma), d1gamma)[1](v)[0])
kappavjp1 = jit(lambda d1gamma, d2gamma, v: vjp(lambda d2g: kappa_pure(d1gamma, d2g), d2gamma)[1](v)[0])
kappagrad0 = jit(lambda d1gamma, d2gamma: jacfwd(lambda d1g: kappa_pure(d1g, d2gamma))(d1gamma))
kappagrad1 = jit(lambda d1gamma, d2gamma: jacfwd(lambda d2g: kappa_pure(d1gamma, d2g))(d2gamma))


@jit
def torsion_pure(d1gamma, d2gamma, d3gamma):
    """
    This function is used in a Python+Jax implementation of formula for torsion.
    """

    return jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1)**2, axis=1)


torsionvjp0 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d1g: torsion_pure(d1g, d2gamma, d3gamma), d1gamma)[1](v)[0])
torsionvjp1 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d2g: torsion_pure(d1gamma, d2g, d3gamma), d2gamma)[1](v)[0])
torsionvjp2 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d3g: torsion_pure(d1gamma, d2gamma, d3g), d3gamma)[1](v)[0])


class Curve(Optimizable):
    """
    Curve  is a base class for various representations of curves in SIMSOPT
    using the graph based Optimizable framework with external handling of DOFS
    as well.
    """

    def __init__(self, **kwargs):
        Optimizable.__init__(self, **kwargs)

    def recompute_bell(self, parent=None):
        """
        For derivative classes of Curve, all of which also subclass
        from C++ Curve class, call invalidate_cache which is implemented
        in C++ side.
        """
        self.invalidate_cache()

    def plot(self, ax=None, show=True, plot_derivative=False, closed_loop=False, color=None, linestyle=None, axis_equal=True):
        """
        Plot the curve using :mod:`matplotlib.pyplot`

        Args:
            ax: the axis object to plot this one. useful when plotting multiple
                curves in the same plot. defaults to ``None`` and creates a new
                axis.
            show: whether to call ``plt.show()`` at the end. should be set to
                  false if more objects are plotted on top.
            plot_derivative: whether to plot the tangent of the curve too
            closed_loop: whether to connect the first and last point on the
                         curve. can lead to surprising results when only quadrature points
                         on a part of the curve are considered, e.g. when exploting
                         rotational symmetry.
            color: color of the curve, passed to the ``color=`` kwarg of pyplot
            linestyle: linestyle of the curve, passed to the ``linestyle=`` kwarg of pyplot
            axis_equal: whether all three dimensions should be scaled equally.
                        this is actually broken in matplotlib, so we add a workaround that
                        at least does the right think for a single curve. For
                        multiple curves in the same plot, this will not give
                        perfectly equal scaling.

        Returns: a axis which could be passed to a further call to
                 ``Curve.plot`` so that multiple curve are shown together.
        """

        gamma = self.gamma()
        gammadash = self.gammadash()
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        def rep(data):
            if closed_loop:
                return np.concatenate((data, [data[0]]))
            else:
                return data
        X = rep(gamma[:, 0])
        Y = rep(gamma[:, 1])
        Z = rep(gamma[:, 2])
        ax.plot(X, Y, Z, color=color, linestyle=linestyle)
        if plot_derivative:
            ax.quiver(rep(gamma[:, 0]), rep(gamma[:, 1]), rep(gamma[:, 2]), 0.1 * rep(gammadash[:, 0]),
                      0.1 * rep(gammadash[:, 1]), 0.1 * rep(gammadash[:, 2]), arrow_length_ratio=0.1, color="r")
        if axis_equal:  # trick from
            # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
            # to force the axis to be equal, since set_aspect('equal') doesn't work in 3d.

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')
        if show:
            plt.show()
        return ax

    @requires(mlab is not None, "plot_mayavi requires mayavi")
    def plot_mayavi(self, show=True):
        """
        Plot the curve using :mod:`mayavi.mlab` rather than :mod:`matplotlib.pyplot`.
        """

        g = self.gamma()
        mlab.plot3d(g[:, 0], g[:, 1], g[:, 2])
        if show:
            mlab.show()

    def dincremental_arclength_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \|\Gamma'\|}{\partial \mathbf{c}}

        where :math:`\|\Gamma'\|` is the incremental arclength, :math:`\Gamma'` is the tangent 
        to the curve and :math:`\mathbf{c}` are the curve dofs.
        """

        return self.dgammadash_by_dcoeff_vjp(
            incremental_arclength_vjp(self.gammadash(), v))

    def kappa_impl(self, kappa):
        r"""
        This function implements the curvature, :math:`\kappa(\varphi)`.
        """
        kappa[:] = np.asarray(kappa_pure(
            self.gammadash(), self.gammadashdash()))

    def dkappa_by_dcoeff_impl(self, dkappa_by_dcoeff):
        r"""
        This function computes the derivative of the curvature with respect to the curve coefficients.

        .. math::
            \frac{\partial \kappa}{\partial \mathbf c}

        where :math:`\mathbf c` are the curve dofs, :math:`\kappa` is the curvature.
        """

        dgamma_by_dphi = self.gammadash()
        dgamma_by_dphidphi = self.gammadashdash()
        dgamma_by_dphidcoeff = self.dgammadash_by_dcoeff()
        dgamma_by_dphidphidcoeff = self.dgammadashdash_by_dcoeff()
        num_coeff = dgamma_by_dphidcoeff.shape[2]

        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        numerator = np.cross(dgamma_by_dphi, dgamma_by_dphidphi)
        denominator = self.incremental_arclength()
        dkappa_by_dcoeff[:, :] = (1 / (denominator**3*norm(numerator)))[:, None] * np.sum(numerator[:, :, None] * (
            np.cross(dgamma_by_dphidcoeff[:, :, :], dgamma_by_dphidphi[:, :, None], axis=1) +
            np.cross(dgamma_by_dphi[:, :, None], dgamma_by_dphidphidcoeff[:, :, :], axis=1)), axis=1) \
            - (norm(numerator) * 3 / denominator**5)[:, None] * np.sum(dgamma_by_dphi[:, :, None] * dgamma_by_dphidcoeff[:, :, :], axis=1)

    def torsion_impl(self, torsion):
        r"""
        This function returns the torsion, :math:`\tau`, of a curve.
        """
        torsion[:] = torsion_pure(self.gammadash(), self.gammadashdash(),
                                  self.gammadashdashdash())

    def dtorsion_by_dcoeff_impl(self, dtorsion_by_dcoeff):
        r"""
        This function returns the derivative of torsion with respect to the curve dofs.

        .. math::
            \frac{\partial \tau}{\partial \mathbf c}

        where :math:`\mathbf c` are the curve dofs, and :math:`\tau` is the torsion.
        """
        d1gamma = self.gammadash()
        d2gamma = self.gammadashdash()
        d3gamma = self.gammadashdashdash()
        d1gammadcoeff = self.dgammadash_by_dcoeff()
        d2gammadcoeff = self.dgammadashdash_by_dcoeff()
        d3gammadcoeff = self.dgammadashdashdash_by_dcoeff()
        dtorsion_by_dcoeff[:, :] = (
            np.sum(np.cross(d1gamma, d2gamma, axis=1)[:, :, None] * d3gammadcoeff, axis=1)
            + np.sum((np.cross(d1gammadcoeff, d2gamma[:, :, None], axis=1) + np.cross(d1gamma[:, :, None], d2gammadcoeff, axis=1)) * d3gamma[:, :, None], axis=1)
        )/np.sum(np.cross(d1gamma, d2gamma, axis=1)**2, axis=1)[:, None]
        dtorsion_by_dcoeff[:, :] -= np.sum(np.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1)[:, None] * np.sum(2 * np.cross(d1gamma, d2gamma, axis=1)[:, :, None] * (np.cross(d1gammadcoeff, d2gamma[:, :, None], axis=1) + np.cross(d1gamma[:, :, None], d2gammadcoeff, axis=1)), axis=1)/np.sum(np.cross(d1gamma, d2gamma, axis=1)**2, axis=1)[:, None]**2

    def dkappa_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}} 

        where :math:`\mathbf c` are the curve dofs and :math:`\kappa` is the curvature.
        """

        return self.dgammadash_by_dcoeff_vjp(kappavjp0(self.gammadash(), self.gammadashdash(), v)) \
            + self.dgammadashdash_by_dcoeff_vjp(kappavjp1(self.gammadash(), self.gammadashdash(), v))

    def dtorsion_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \tau}{\partial \mathbf{c}} 

        where :math:`\mathbf c` are the curve dofs, and :math:`\tau` is the torsion.
        """

        return self.dgammadash_by_dcoeff_vjp(torsionvjp0(self.gammadash(), self.gammadashdash(), self.gammadashdashdash(), v)) \
            + self.dgammadashdash_by_dcoeff_vjp(torsionvjp1(self.gammadash(), self.gammadashdash(), self.gammadashdashdash(), v)) \
            + self.dgammadashdashdash_by_dcoeff_vjp(torsionvjp2(self.gammadash(), self.gammadashdash(), self.gammadashdashdash(), v))

    def frenet_frame(self):
        r"""
        This function returns the Frenet frame, :math:`(\mathbf{t}, \mathbf{n}, \mathbf{b})`,
        associated to the curve.
        """

        gammadash = self.gammadash()
        gammadashdash = self.gammadashdash()
        l = self.incremental_arclength()
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        N = len(self.quadpoints)
        t, n, b = (np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)))
        t[:, :] = (1./l[:, None]) * gammadash

        tdash = (1./l[:, None])**2 * (l[:, None] * gammadashdash
                                      - (inner(gammadash, gammadashdash)/l)[:, None] * gammadash
                                      )
        kappa = self.kappa
        n[:, :] = (1./norm(tdash))[:, None] * tdash
        b[:, :] = np.cross(t, n, axis=1)
        return t, n, b

    def kappadash(self):
        r"""
        This function returns :math:`\kappa'(\phi)`, where :math:`\kappa` is the curvature.
        """
        dkappa_by_dphi = np.zeros((len(self.quadpoints), ))
        dgamma = self.gammadash()
        d2gamma = self.gammadashdash()
        d3gamma = self.gammadashdashdash()
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        cross = lambda a, b: np.cross(a, b, axis=1)
        dkappa_by_dphi[:] = inner(cross(dgamma, d2gamma), cross(dgamma, d3gamma))/(norm(cross(dgamma, d2gamma)) * norm(dgamma)**3) \
            - 3 * inner(dgamma, d2gamma) * norm(cross(dgamma, d2gamma))/norm(dgamma)**5
        return dkappa_by_dphi

    def dfrenet_frame_by_dcoeff(self):
        r"""
        This function returns the derivative of the curve's Frenet frame, 

        .. math::
            \left(\frac{\partial \mathbf{t}}{\partial \mathbf{c}}, \frac{\partial \mathbf{n}}{\partial \mathbf{c}}, \frac{\partial \mathbf{b}}{\partial \mathbf{c}}\right),

        with respect to the curve dofs, where :math:`(\mathbf t, \mathbf n, \mathbf b)` correspond to the Frenet frame, and :math:`\mathbf c` are the curve dofs.
        """
        dgamma_by_dphi = self.gammadash()
        d2gamma_by_dphidphi = self.gammadashdash()
        d2gamma_by_dphidcoeff = self.dgammadash_by_dcoeff()
        d3gamma_by_dphidphidcoeff = self.dgammadashdash_by_dcoeff()

        l = self.incremental_arclength()
        dl_by_dcoeff = self.dincremental_arclength_by_dcoeff()

        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        inner2 = lambda a, b: np.sum(a*b, axis=2)

        N = len(self.quadpoints)
        dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff = (np.zeros((N, 3, self.num_dofs())), np.zeros((N, 3, self.num_dofs())), np.zeros((N, 3, self.num_dofs())))
        t, n, b = self.frenet_frame()

        dt_by_dcoeff[:, :, :] = -(dl_by_dcoeff[:, None, :]/l[:, None, None]**2) * dgamma_by_dphi[:, :, None] \
            + d2gamma_by_dphidcoeff / l[:, None, None]

        tdash = (1./l[:, None])**2 * (
            l[:, None] * d2gamma_by_dphidphi
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/l)[:, None] * dgamma_by_dphi
        )

        dtdash_by_dcoeff = (-2 * dl_by_dcoeff[:, None, :] / l[:, None, None]**3) * (l[:, None] * d2gamma_by_dphidphi - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/l)[:, None] * dgamma_by_dphi)[:, :, None] \
            + (1./l[:, None, None])**2 * (
                dl_by_dcoeff[:, None, :] * d2gamma_by_dphidphi[:, :, None] + l[:, None, None] * d3gamma_by_dphidphidcoeff
                - (inner(d2gamma_by_dphidcoeff, d2gamma_by_dphidphi[:, :, None])[:, None, :]/l[:, None, None]) * dgamma_by_dphi[:, :, None]
                - (inner(dgamma_by_dphi[:, :, None], d3gamma_by_dphidphidcoeff)[:, None, :]/l[:, None, None]) * dgamma_by_dphi[:, :, None]
                + (inner(dgamma_by_dphi, d2gamma_by_dphidphi)[:, None, None] * dl_by_dcoeff[:, None, :]/l[:, None, None]**2) * dgamma_by_dphi[:, :, None]
                - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/l)[:, None, None] * d2gamma_by_dphidcoeff
        )
        dn_by_dcoeff[:, :, :] = (1./norm(tdash))[:, None, None] * dtdash_by_dcoeff \
            - (inner(tdash[:, :, None], dtdash_by_dcoeff)[:, None, :]/inner(tdash, tdash)[:, None, None]**1.5) * tdash[:, :, None]

        db_by_dcoeff[:, :, :] = np.cross(dt_by_dcoeff, n[:, :, None], axis=1) + np.cross(t[:, :, None], dn_by_dcoeff, axis=1)
        return dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff

    def dkappadash_by_dcoeff(self):
        r"""
        This function returns 

        .. math::
            \frac{\partial \kappa'(\phi)}{\partial \mathbf{c}}.

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\kappa` is the curvature.
        """

        dkappadash_by_dcoeff = np.zeros((len(self.quadpoints), self.num_dofs()))
        dgamma = self.gammadash()
        d2gamma = self.gammadashdash()
        d3gamma = self.gammadashdashdash()

        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        cross = lambda a, b: np.cross(a, b, axis=1)
        d1_dot_d2 = inner(dgamma, d2gamma)
        d1_x_d2 = cross(dgamma, d2gamma)
        d1_x_d3 = cross(dgamma, d3gamma)
        normdgamma = norm(dgamma)
        norm_d1_x_d2 = norm(d1_x_d2)
        dgamma_dcoeff_ = self.dgammadash_by_dcoeff()
        d2gamma_dcoeff_ = self.dgammadashdash_by_dcoeff()
        d3gamma_dcoeff_ = self.dgammadashdashdash_by_dcoeff()
        for i in range(self.num_dofs()):
            dgamma_dcoeff = dgamma_dcoeff_[:, :, i]
            d2gamma_dcoeff = d2gamma_dcoeff_[:, :, i]
            d3gamma_dcoeff = d3gamma_dcoeff_[:, :, i]

            d1coeff_x_d2 = cross(dgamma_dcoeff, d2gamma)
            d1coeff_dot_d2 = inner(dgamma_dcoeff, d2gamma)
            d1coeff_x_d3 = cross(dgamma_dcoeff, d3gamma)
            d1_x_d2coeff = cross(dgamma, d2gamma_dcoeff)
            d1_dot_d2coeff = inner(dgamma, d2gamma_dcoeff)
            d1_dot_d1coeff = inner(dgamma, dgamma_dcoeff)
            d1_x_d3coeff = cross(dgamma, d3gamma_dcoeff)

            dkappadash_by_dcoeff[:, i] = (
                +inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d3)
                + inner(d1_x_d2, d1coeff_x_d3 + d1_x_d3coeff)
            )/(norm_d1_x_d2 * normdgamma**3) \
                - inner(d1_x_d2, d1_x_d3) * (
                    (
                        inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d2)/(norm_d1_x_d2**3 * normdgamma**3)
                        + 3 * inner(dgamma, dgamma_dcoeff)/(norm_d1_x_d2 * normdgamma**5)
                    )
            ) \
                - 3 * (
                    + (d1coeff_dot_d2 + d1_dot_d2coeff) * norm_d1_x_d2/normdgamma**5
                    + d1_dot_d2 * inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d2)/(norm_d1_x_d2 * normdgamma**5)
                    - 5 * d1_dot_d2 * norm_d1_x_d2 * d1_dot_d1coeff/normdgamma**7
            )
        return dkappadash_by_dcoeff


class JaxCurve(sopp.Curve, Curve):
    def __init__(self, quadpoints, gamma_pure, **kwargs):
        if isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.Curve.__init__(self, quadpoints)
        if "external_dof_setter" not in kwargs:
            kwargs["external_dof_setter"] = sopp.Curve.set_dofs_impl
        # We are not doing the same search for x0
        Curve.__init__(self, **kwargs)
        self.gamma_pure = gamma_pure
        points = np.asarray(self.quadpoints)
        ones = jnp.ones_like(points)

        self.gamma_jax = jit(lambda dofs: self.gamma_pure(dofs, points))
        self.gamma_impl_jax = jit(lambda dofs, p: self.gamma_pure(dofs, p))
        self.dgamma_by_dcoeff_jax = jit(jacfwd(self.gamma_jax))
        self.dgamma_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gamma_jax, x)[1](v)[0])

        self.gammadash_pure = lambda x, q: jvp(lambda p: self.gamma_pure(x, p), (q,), (ones,))[1]
        self.gammadash_jax = jit(lambda x: self.gammadash_pure(x, points))
        self.dgammadash_by_dcoeff_jax = jit(jacfwd(self.gammadash_jax))
        self.dgammadash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadash_jax, x)[1](v)[0])

        self.gammadashdash_pure = lambda x, q: jvp(lambda p: self.gammadash_pure(x, p), (q,), (ones,))[1]
        self.gammadashdash_jax = jit(lambda x: self.gammadashdash_pure(x, points))
        self.dgammadashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdash_jax))
        self.dgammadashdash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadashdash_jax, x)[1](v)[0])

        self.gammadashdashdash_pure = lambda x, q: jvp(lambda p: self.gammadashdash_pure(x, p), (q,), (ones,))[1]
        self.gammadashdashdash_jax = jit(lambda x: self.gammadashdashdash_pure(x, points))
        self.dgammadashdashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdashdash_jax))
        self.dgammadashdashdash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadashdashdash_jax, x)[1](v)[0])

        self.dkappa_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), x)[1](v)[0])

        self.dtorsion_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v)[0])

    def set_dofs(self, dofs):
        self.local_x = dofs
        sopp.Curve.set_dofs(self, dofs)

    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve :math:`\Gamma`.
        """

        gamma[:, :] = self.gamma_impl_jax(self.get_dofs(), quadpoints)

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        dgamma_by_dcoeff[:, :, :] = self.dgamma_by_dcoeff_jax(self.get_dofs())

    def dgamma_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        return self.dgamma_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadash_impl(self, gammadash):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadash[:, :] = self.gammadash_jax(self.get_dofs())

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        dgammadash_by_dcoeff[:, :, :] = self.dgammadash_by_dcoeff_jax(self.get_dofs())

    def dgammadash_by_dcoeff_vjp(self, v):
        r"""
        This function returns 

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        return self.dgammadash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadashdash_impl(self, gammadashdash):
        r"""
        This function returns :math:`\Gamma''(\varphi)`, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadashdash[:, :] = self.gammadashdash_jax(self.get_dofs())

    def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        dgammadashdash_by_dcoeff[:, :, :] = self.dgammadashdash_by_dcoeff_jax(self.get_dofs())

    def dgammadashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma''}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadashdashdash_impl(self, gammadashdashdash):
        r"""
        This function returns :math:`\Gamma'''(\varphi)`, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadashdashdash[:, :] = self.gammadashdashdash_jax(self.get_dofs())

    def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        dgammadashdashdash_by_dcoeff[:, :, :] = self.dgammadashdashdash_by_dcoeff_jax(self.get_dofs())

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma'''}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dkappa_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

        """
        return self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dtorsion_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \tau}{\partial \mathbf{c}} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

        """

        return self.dtorsion_by_dcoeff_vjp_jax(self.get_dofs(), v)


class RotatedCurve(sopp.Curve, Curve):
    """
    RotatedCurve inherits from the Curve base class.  It takes an input
    a Curve, rotates it by ``theta``, and optionally completes a
    reflection when ``flip=True``.
    """

    def __init__(self, curve, theta, flip):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve],
                       external_dof_setter=sopp.Curve.set_dofs)
        self.rotmat = np.asarray(
            [[cos(theta), -sin(theta), 0],
             [sin(theta), cos(theta), 0],
             [0, 0, 1]]).T
        if flip:
            self.rotmat = self.rotmat @ np.asarray(
                [[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]])
        self.rotmatT = self.rotmat.T

    def get_dofs(self):
        """
        RotatedCurve does not have any dofs of its own.
        This function returns null array
        """
        return np.array([])

    def set_dofs_impl(self, d):
        """
        RotatedCurve does not have any dofs of its own.
        This function does nothing.
        """
        pass

    def num_dofs(self):
        """
        This function returns the number of dofs associated to the curve.
        """
        return self.curve.num_dofs()

    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve, :math:`\Gamma`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        self.curve.gamma_impl(gamma, quadpoints)
        gamma[:] = gamma @ self.rotmat

    def gammadash_impl(self, gammadash):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        gammadash[:] = self.curve.gammadash() @ self.rotmat

    def gammadashdash_impl(self, gammadashdash):
        r"""
        This function returns :math:`\Gamma''(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        gammadashdash[:] = self.curve.gammadashdash() @ self.rotmat

    def gammadashdashdash_impl(self, gammadashdashdash):
        r"""
        This function returns :math:`\Gamma'''(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        gammadashdashdash[:] = self.curve.gammadashdashdash() @ self.rotmat

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        dgamma_by_dcoeff[:] = self.rotmatT @ self.curve.dgamma_by_dcoeff()

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.
        """

        dgammadash_by_dcoeff[:] = self.rotmatT @ self.curve.dgammadash_by_dcoeff()

    def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        dgammadashdash_by_dcoeff[:] = self.rotmatT @ self.curve.dgammadashdash_by_dcoeff()

    def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        dgammadashdashdash_by_dcoeff[:] = self.rotmatT @ self.curve.dgammadashdashdash_by_dcoeff()

    def dgamma_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        return self.curve.dgamma_by_dcoeff_vjp(v @ self.rotmat.T)

    def dgammadash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma'}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        return self.curve.dgammadash_by_dcoeff_vjp(v @ self.rotmat.T)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma''}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        return self.curve.dgammadashdash_by_dcoeff_vjp(v @ self.rotmat.T)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma'''}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v @ self.rotmat.T)


def curves_to_vtk(curves, filename):
    from pyevtk.hl import polyLinesToVTK
    x = np.concatenate([c.gamma()[:, 0] for c in curves])
    y = np.concatenate([c.gamma()[:, 1] for c in curves])
    z = np.concatenate([c.gamma()[:, 2] for c in curves])
    ppl = np.asarray([c.gamma().shape[0] for c in curves])
    data = np.concatenate([i*np.ones((curves[i].gamma().shape[0], )) for i in range(len(curves))])
    polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})
