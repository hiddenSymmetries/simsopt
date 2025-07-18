from math import sin, cos

import numpy as np
from jax import vjp, jacfwd, jvp
import jax.numpy as jnp

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative

from .jit import jit
from .plotting import fix_matplotlib_3d

__all__ = ['Curve', 'JaxCurve', 'RotatedCurve', 'curves_to_vtk', 'create_equally_spaced_curves',
           'create_equally_spaced_planar_curves', 'create_planar_curves_between_two_toroidal_surfaces']

@jit
def centroid_pure(gamma, gammadash):
    """
    This pure function is used in a Python+Jax implementation of formula for centroid.

    .. math::
        \mathbf{c} = \frac{1}{L} \int_0^L \mathbf{\gamma}(l) dl

    where :math:`\gamma` is the position vector on the curve.
    """
    arclength = jnp.linalg.norm(gammadash, axis=-1)
    centroid = jnp.sum(gamma * arclength[:, None], axis=0) / jnp.sum(arclength)
    return centroid

@jit
def incremental_arclength_pure(d1gamma):
    """
    This function is used in a Python+Jax implementation of the curve arc length formula.

    .. math::
        \text{incremental arclength} = \|\mathbf{\gammadash}(\phi)\| d\phi

    where :math:`\mathbf{\gammadash}(\phi)` is the derivative of the 
    position vector to the curve.
    """
    return jnp.linalg.norm(d1gamma, axis=1)


incremental_arclength_vjp = jit(lambda d1gamma, v: vjp(lambda d1g: incremental_arclength_pure(d1g), d1gamma)[1](v)[0])


@jit
def kappa_pure(d1gamma, d2gamma):
    """
    This function is used in a Python+Jax implementation of formula for curvature.

    .. math::
        \kappa(\phi) = \frac{\|\mathbf{\gammadash} \times \mathbf{\gammadashdash}\|}{\|\mathbf{\gammadash}\|^3}

    where :math:`\mathbf{\gammadash}` is the tangent vector to the curve and 
    :math:`\mathbf{\gammadashdash}` is the derivative of the tangent vector.
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

    .. math::
        \tau(\phi) = \frac{\mathbf{\gammadash} \times \mathbf{\gammadashdash} \cdot \mathbf{\gammadashdashdash}}{\|\mathbf{\gammadash} \times \mathbf{\gammadashdash}\|^2}

    where :math:`\mathbf{\gammadash}` is the tangent vector to the curve, 
    :math:`\mathbf{\gammadashdash}` is the derivative of the tangent vector, and 
    :math:`\mathbf{\gammadashdashdash}` is the derivative of the derivative of the tangent vector.
    """
    return jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1)**2, axis=1)


torsionvjp0 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d1g: torsion_pure(d1g, d2gamma, d3gamma), d1gamma)[1](v)[0])
torsionvjp1 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d2g: torsion_pure(d1gamma, d2g, d3gamma), d2gamma)[1](v)[0])
torsionvjp2 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d3g: torsion_pure(d1gamma, d2gamma, d3g), d3gamma)[1](v)[0])


@jit
def frenet_frame_pure(gammadash, gammadashdash, incremental_arclength):
    r"""
    This function returns the Frenet frame, :math:`(\mathbf{t}, \mathbf{n}, \mathbf{b})`,
    associated to the curve.

    .. math::
        \mathbf{t} = \frac{1}{l} \mathbf{\gammadash}

    where :math:`l` is the the derivative of arclength with respect 
    to the curve parameter. t = gammadash / |gammadash|.

    .. math::
        \mathbf{n} = \frac{1}{\|\mathbf{tdash}\|}\mathbf{tdash}

    .. math::
        \mathbf{b} = \mathbf{t} \times \mathbf{n}
    """
    def norm(a): return jnp.linalg.norm(a, axis=1)
    def inner(a, b): return jnp.sum(a*b, axis=1)
    N = jnp.shape(gammadash)[0]
    t, n, b = (jnp.zeros((N, 3)), jnp.zeros((N, 3)), jnp.zeros((N, 3)))
    t = (1./incremental_arclength[:, None]) * gammadash

    tdash = (1./incremental_arclength[:, None])**2 * (incremental_arclength[:, None] * gammadashdash
                                    - (inner(gammadash, gammadashdash)/incremental_arclength)[:, None] * gammadash
                                    )
    n = (1./norm(tdash))[:, None] * tdash
    b = jnp.cross(t, n, axis=1)
    return t, n, b

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

    def plot(self, engine="matplotlib", ax=None, show=True, plot_derivative=False, close=False, axis_equal=True, **kwargs):
        """
        Plot the curve in 3D using ``matplotlib.pyplot``, ``mayavi``, or ``plotly``.

        Args:
            engine: The graphics engine to use. Available settings are ``"matplotlib"``, ``"mayavi"``, and ``"plotly"``.
            ax: The axis object on which to plot. This argument is useful when plotting multiple
              objects on the same axes. If equal to the default ``None``, a new axis will be created.
            show: Whether to call the ``show()`` function of the graphics engine. Should be set to
              ``False`` if more objects will be plotted on the same axes.
            plot_derivative: Whether to plot the tangent of the curve too. Not implemented for plotly.
            close: Whether to connect the first and last point on the
              curve. Can lead to surprising results when only quadrature points
              on a part of the curve are considered, e.g. when exploting rotational symmetry.
            axis_equal: For matplotlib, whether all three dimensions should be scaled equally.
            kwargs: Any additional arguments to pass to the plotting function, like ``color='r'``.

        Returns:
            An axis which could be passed to a further call to the graphics engine
            so multiple objects are shown together.
        """

        def rep(data):
            if close:
                return np.concatenate((data, [data[0]]))
            else:
                return data

        x = rep(self.gamma()[:, 0])
        y = rep(self.gamma()[:, 1])
        z = rep(self.gamma()[:, 2])
        if plot_derivative:
            xt = rep(self.gammadash()[:, 0])
            yt = rep(self.gammadash()[:, 1])
            zt = rep(self.gammadash()[:, 2])

        if engine == "matplotlib":
            # plot in matplotlib.pyplot
            import matplotlib.pyplot as plt

            if ax is None or ax.name != "3d":
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
            ax.plot(x, y, z, **kwargs)
            if plot_derivative:
                ax.quiver(x, y, z, 0.1 * xt, 0.1 * yt, 0.1 * zt, arrow_length_ratio=0.1, color="r")
            if axis_equal:
                fix_matplotlib_3d(ax)
            if show:
                plt.show()

        elif engine == "mayavi":
            # plot 3D curve in mayavi.mlab
            from mayavi import mlab

            mlab.plot3d(x, y, z, **kwargs)
            if plot_derivative:
                mlab.quiver3d(x, y, z, 0.1*xt, 0.1*yt, 0.1*zt)
            if show:
                mlab.show()

        elif engine == "plotly":
            import plotly.graph_objects as go

            if "color" in list(kwargs.keys()):
                color = kwargs["color"]
                del kwargs["color"]
            else:
                color = "blue"
            kwargs.setdefault("line", go.scatter3d.Line(color=color, width=4))
            if ax is None:
                ax = go.Figure()
            ax.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z, mode="lines", **kwargs
                )
            )
            ax.update_layout(scene_aspectmode="data")
            if show:
                ax.show()
        else:
            raise ValueError("Invalid engine option! Please use one of {matplotlib, mayavi, plotly}.")
        return ax

    def dgamma_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgamma_by_dcoeff_vjp_impl(v)})

    def dgammadash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadash_by_dcoeff_vjp_impl(v)})

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadashdash_by_dcoeff_vjp_impl(v)})

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadashdashdash_by_dcoeff_vjp_impl(v)})

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

        def norm(a): return np.linalg.norm(a, axis=1)
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
        return frenet_frame_pure(self.gammadash(), self.gammadashdash(), self.incremental_arclength())

    def kappadash(self):
        r"""
        This function returns :math:`\kappa'(\phi)`, where :math:`\kappa` is the curvature.
        """
        dkappa_by_dphi = np.zeros((len(self.quadpoints), ))
        dgamma = self.gammadash()
        d2gamma = self.gammadashdash()
        d3gamma = self.gammadashdashdash()
        def norm(a): return np.linalg.norm(a, axis=1)
        def inner(a, b): return np.sum(a*b, axis=1)
        def cross(a, b): return np.cross(a, b, axis=1)
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

        def norm(a): return np.linalg.norm(a, axis=1)
        def inner(a, b): return np.sum(a*b, axis=1)

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

        def norm(a): return np.linalg.norm(a, axis=1)
        def inner(a, b): return np.sum(a*b, axis=1)
        def cross(a, b): return np.cross(a, b, axis=1)
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

    def centroid(self):
        """ 
        Compute the centroid of the curve

        .. math::
            \mathbf{c} = \frac{1}{L} \int_0^L \mathbf{\gamma}(l) dl

        where :math:`\gamma` is the position on the curve. Note that this function was once called
        `center` but this conflicts with the center property of the C++ CurvePlanarFourier
        implementation.
        """
        return centroid_pure(self.gamma(), self.gammadash())

class JaxCurve(sopp.Curve, Curve):
    """
    A class for curves defined by a pure function.

    Args:
        quadpoints (array): Array of quadrature points.
        gamma_pure (function): Pure function for the curve.
        **kwargs: Additional keyword arguments.
    """
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

        self.incremental_arclength_jax = jit(lambda x: incremental_arclength_pure(self.gammadash_jax(x)))
        self.dkappa_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), x)[1](v)[0])
        self.dtorsion_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v)[0])

    def set_dofs(self, dofs):
        """
        This function sets the dofs of the curve.
        """
        self.local_x = dofs
        sopp.Curve.set_dofs(self, dofs)

    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve :math:`\Gamma`.
        """
        gamma[:, :] = self.gamma_impl_jax(self.get_dofs(), quadpoints)

    def incremental_arclength_impl(self):
        """
        This function returns the incremental arclength of the curve.
        """
        return self.incremental_arclength_jax(self.get_dofs())

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        dgamma_by_dcoeff[:, :, :] = self.dgamma_by_dcoeff_jax(self.get_dofs())

    def dgamma_by_dcoeff_vjp_impl(self, v):
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

    def dgammadash_by_dcoeff_vjp_impl(self, v):
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

    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
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

    def dgammadashdashdash_by_dcoeff_vjp_impl(self, v):
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
        return Derivative({self: self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), v)})

    def dtorsion_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \tau}{\partial \mathbf{c}} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

        """

        return Derivative({self: self.dtorsion_by_dcoeff_vjp_jax(self.get_dofs(), v)})


class RotatedCurve(sopp.Curve, Curve):
    """
    RotatedCurve inherits from the Curve base class.  It takes an
    input a Curve, rotates it about the ``z`` axis by a toroidal angle
    ``phi``, and optionally completes a reflection when ``flip=True``.
    """

    def __init__(self, curve, phi, flip):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self._phi = phi
        self.rotmat = np.asarray(
            [[cos(phi), -sin(phi), 0],
             [sin(phi), cos(phi), 0],
             [0, 0, 1]]).T
        if flip:
            self.rotmat = self.rotmat @ np.asarray(
                [[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]])
        self.rotmatT = self.rotmat.T.copy()

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

        if len(quadpoints) == len(self.curve.quadpoints) \
                and np.sum((quadpoints-self.curve.quadpoints)**2) < 1e-15:
            gamma[:] = self.curve.gamma() @ self.rotmat
        else:
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
        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma'}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma''}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma'''}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)

    @property
    def flip(self):
        return True if self.rotmat[2][2] == -1 else False


def curves_to_vtk(curves, filename, close=False, extra_data=None):
    """
    Export a list of Curve objects in VTK format, so they can be
    viewed using Paraview. This function requires the python package ``pyevtk``,
    which can be installed using ``pip install pyevtk``.

    Args:
        curves: A python list of Curve objects.
        filename: Name of the file to write.
        close: Whether to draw the segment from the last quadrature point back to the first.
    """
    from pyevtk.hl import polyLinesToVTK

    def wrap(data):
        return np.concatenate([data, [data[0]]])

    if close:
        x = np.concatenate([wrap(c.gamma()[:, 0]) for c in curves])
        y = np.concatenate([wrap(c.gamma()[:, 1]) for c in curves])
        z = np.concatenate([wrap(c.gamma()[:, 2]) for c in curves])
        ppl = np.asarray([c.gamma().shape[0]+1 for c in curves])
    else:
        x = np.concatenate([c.gamma()[:, 0] for c in curves])
        y = np.concatenate([c.gamma()[:, 1] for c in curves])
        z = np.concatenate([c.gamma()[:, 2] for c in curves])
        ppl = np.asarray([c.gamma().shape[0] for c in curves])
    data = np.concatenate([i*np.ones((ppl[i], )) for i in range(len(curves))])
    pointData = {'idx': data}
    if extra_data is not None:
        pointData = {**pointData, **extra_data}

    polyLinesToVTK(str(filename), x, y, z, pointsPerLine=ppl, pointData=pointData)


def _setup_uniform_grid_in_bounding_box(s_outer, Nx, Ny, Nz, Nmin_factor=2.01):
    """
    Generate a uniform 3D grid of points where a set of circular coils 
    will be initialized to have their centers. The coils are uniformly
    spaced on the Cartesian grid, although the grid may have different spacing 
    in the x, y, and z directions, and it is appropriately initialized to
    respect the discrete symmetries of the plasma.

    The grid is defined by the inner and outermost points of a toroidal surface s_outer.
    Filtering on this grid is done to avoid coil overlap and respect stellarator and 
    field-period symmetries.

    This function is typically used to initialize candidate coil center locations for planar 
    coil optimization. 
    It computes a uniform grid in the bounding box from the min and max points of the toroidal surface 
    s_outer (typically generated using s.extend_via_normal() or similar function). Then it:
    1. Generates a uniform grid for a set of circular coils by:
        (a) X = np.linspace(dx / 2.0 + x_min, x_max - dx / 2.0, Nx, endpoint=True)
        (b) Y = np.linspace(dy / 2.0 + y_min, y_max - dy / 2.0, Ny, endpoint=True)
        (c) Z = np.linspace(-z_max, z_max, Nz, endpoint=True)
        (d) Computes the radius R of the coils by taking the minimum spacing of the grid and dividing by Nmin_factor:
            dx = X[1] - X[0]
            dy = Y[1] - Y[0]
            dz = Z[1] - Z[0]
            Nmin = min(dx, min(dy, dz))
            R = Nmin / Nmin_factor
            - As long as Nmin_factor > 2, then the coils cannot overlap.
        (e) Removes points too close to the unique sector [0, pi / nfp] (or [0, 2pi / nfp] 
            for stellsym = False) to avoid overlap after symmetry operations. To guarantee that 
            the symmetrized coils do not overlap, we remove points according to the following logic:
            - Compute the coil curve in the x-y plane.
            - Compute the angle of every point on the coil curve.
            - Remove points where the angle is greater than phi0 or less than 0.
            - This guarantees that the symmetrized coils do not overlap.

    Parameters
    ----------
    s_outer : Surface
        The outer toroidal surface (for grid bounding box). Assumed to have the same 
        discrete symmetries as the plasma surface.
    Nx : int
        Number of grid points in the x direction.
    Ny : int
        Number of grid points in the y direction.
    Nz : int
        Number of grid points in the z direction.
    Nmin_factor : float, optional
        Factor to set minimum coil spacing (default: 2.01). The coil radius is set to Nmin / Nmin_factor, 
        where Nmin is the minimum grid spacing. So as long as Nmin_factor > 2, then the coils 
        (which are initialized as circles of radius R) will not overlap.

    Returns
    -------
    xyz_uniform : ndarray, shape (N, 3)
        Array of candidate coil center points in 3D, filtered for symmetry and spacing.
    R : float
        The coil radius used for spacing.
    """
    import warnings
    if Nmin_factor <= 2.0:
        warnings.warn('Nmin_factor should be greater than 2.0 to avoid coil overlap.')

    # Get (X, Y, Z) coordinates of the two boundaries
    nfp = s_outer.nfp
    xyz_outer = s_outer.gamma().reshape(-1, 3)
    x_outer = xyz_outer[:, 0]
    y_outer = xyz_outer[:, 1]
    z_outer = xyz_outer[:, 2]
    x_max = np.max(x_outer)
    x_min = np.min(x_outer)
    y_max = np.max(y_outer)
    y_min = 0
    z_max = np.max(z_outer)
    z_min = np.min(z_outer)
    z_max = min(z_max, abs(z_min))  # Note min here! 
    
    # Initialize uniform grid
    if nfp != 1:
        x_min = 0.0
    
    dx = (x_max - x_min) / (Nx - 1)  # x \in [x_min, x_max], x_min = 0.0 if nfp != 1
    dy = (y_max) / (Ny - 1)  # y \in [0, y_max]
    # Z-grid spacing should be symmetric around z = 0 to be able 
    # to properly impose stellarator symmetry
    dz = 2 * z_max / (Nz - 1)  # z \in [-z_max, z_max]

    # Shift by dx / 2.0 to the right and dy / 2.0 to the top to continue to have 
    # dx and dy spacing between points on either side of a symmetry plane. 
    X = np.linspace(
        dx / 2.0 + x_min, x_max - dx / 2.0,
        Nx, endpoint=True
    )
    Y = np.linspace(
        dy / 2.0 + y_min, y_max - dy / 2.0,
        Ny, endpoint=True
    )
    Z = np.linspace(-z_max, z_max, Nz, endpoint=True)

    # Now recompute the grid spacing (for setting the coil radius R)
    # since we have shifted the end points of the grid.
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    dz = Z[1] - Z[0]
    Nmin = min(dx, min(dy, dz))

    # Coils are now spaced so that every coil of radius R is at least 2R away from the next coil'
    R = Nmin / Nmin_factor
    print('Major radius of the coils is R = ', R)

    # Make 3D mesh
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
    xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)

    # Now need to chop off points close to the unique sector [0, (2)pi / nfp]
    # to avoid overlap after discrete symmetry operations.
    if s_outer.stellsym:
        phi0 = np.pi / nfp
    else:
        phi0 = 2 * np.pi / nfp

    # Plan is to generate a circular coil of radius R centered at the coil center in the x-y plane
    # and then remove points on this circle that have phi > phi0 or phi < 0.
    nt = 100
    t = np.linspace(0, 2 * np.pi, nt)
    circle_xy = np.zeros((nt, Nx * Ny * Nz, 2))
    circle_xy[:, :, 0] = R * np.outer(np.cos(t), np.ones(Nx * Ny * Nz)) + np.outer(np.ones(nt), xyz_uniform[:, 0])
    circle_xy[:, :, 1] = R * np.outer(np.sin(t), np.ones(Nx * Ny * Nz)) + np.outer(np.ones(nt), xyz_uniform[:, 1])

    # Remove points where the angle is greater than phi0 or less than 0
    phi = np.arctan2(circle_xy[:, :, 1], circle_xy[:, :, 0])
    remove_inds = np.logical_or(phi >= phi0, phi <= 0)
    intersection_inds = np.any(remove_inds, axis=0)
    xyz_uniform = xyz_uniform[~intersection_inds, :]
    return xyz_uniform, R


def create_planar_curves_between_two_toroidal_surfaces(
    s, s_inner, s_outer, Nx=10, Ny=10, Nz=10, order=1,
    use_jax_curve=False, numquadpoints=None,
    Nmin_factor=2.01,
):
    """
    Create a list of planar curves between two toroidal surfaces. The curves are initialized as 
    circular coils of radius R and then the coils are rotated and flipped to satisfy stellarator 
    symmetry. They are originally initialized on a uniform Cartesian grid and then filtered to 
    only include points that are between the two toroidal surfaces.

    Args:
        s : Surface
            The plasma surface object (used for nfp and geometry info).
        s_inner : Surface
            The inner toroidal surface (for grid bounding box). Typically generated
            from s.extend_via_normal().
        s_outer : Surface
            The outer toroidal surface (for grid bounding box). Typically generated
            from s.extend_via_normal() and should extend out further than s_inner.
        Nx : int, optional
            Number of uniform grid points in the x direction to initialize a uniform grid.
        Ny : int, optional
            Number of uniform grid points in the y direction to initialize a uniform grid.
        Nz : int, optional
            Number of uniform grid points in the z direction to initialize a uniform grid.
        order : int, optional
            Order of the Fourier series in the planar curve representation.
        use_jax_curve : bool, optional
            Whether to use JaxCurvePlanarFourier instead of CurvePlanarFourier.
        numquadpoints : int, optional
            Number of quadrature points to use.
        Nmin_factor : float, optional
            Factor to set minimum coil spacing (default: 2.01). The coil radius is set to Nmin / Nmin_factor, 
            where Nmin is the minimum grid spacing. So as long as Nmin_factor > 2, then the coils 
            (which are initialized as circles of radius R) will not overlap.

    Returns:
        curves : list
            List of CurvePlanarFourier or JaxCurvePlanarFourier objects.
        all_curves : list
    """
    from simsopt.geo import CurvePlanarFourier, JaxCurvePlanarFourier
    from simsopt.field import apply_symmetries_to_curves

    nfp = s.nfp
    stellsym = s.stellsym
    normal_inner = s_inner.unitnormal().reshape(-1, 3)
    xyz_inner = s_inner.gamma().reshape(-1, 3)
    normal_outer = s_outer.unitnormal().reshape(-1, 3)
    xyz_outer = s_outer.gamma().reshape(-1, 3)

    # Now guarantees that circular coils of radius R on this grid do not overlap 
    xyz_uniform, R = _setup_uniform_grid_in_bounding_box(s_outer, Nx, Ny, Nz, Nmin_factor=Nmin_factor)
    
    # Have the uniform grid, now need to loop through and eliminate any points that are 
    # not actually between the two toroidal surfaces.
    contig = np.ascontiguousarray
    grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
        contig(normal_inner),
        contig(normal_outer),
        contig(xyz_uniform),
        contig(xyz_inner),
        contig(xyz_outer)
    )
    inds = np.ravel(np.logical_not(np.all(grid_xyz == 0.0, axis=-1)))
    grid_xyz = np.array(grid_xyz[inds, :], dtype=float)
    ncoils = grid_xyz.shape[0]
    if numquadpoints is None:
        nquad = (order + 1)*40
    else:
        nquad = numquadpoints
    if use_jax_curve:
        curves = [JaxCurvePlanarFourier(nquad, order) for i in range(ncoils)]
    else:
        curves = [CurvePlanarFourier(nquad, order) for i in range(ncoils)]

    # Initialize a bunch of circular coils with same normal vector
    for ic in range(ncoils):
        alpha2 = (np.random.rand(1) * np.pi - np.pi / 2.0)[0]
        delta2 = (np.random.rand(1) * np.pi)[0]
        calpha2 = np.cos(alpha2)
        salpha2 = np.sin(alpha2)
        cdelta2 = np.cos(delta2)
        sdelta2 = np.sin(delta2)
        dofs = np.zeros(2 * order + 8)
        dofs[0] = R
        for j in range(1, 2 * order + 1):
            dofs[j] = 0.0
        # Conversion from Euler angles in 3-2-1 body sequence to quaternions:
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        dofs[2 * order + 1] = calpha2 * cdelta2
        dofs[2 * order + 2] = salpha2 * cdelta2
        dofs[2 * order + 3] = calpha2 * sdelta2
        dofs[2 * order + 4] = -salpha2 * sdelta2
        # Now specify the center
        dofs[2 * order + 5:2 * order + 8] = grid_xyz[ic, :]
        curves[ic].set_dofs(dofs)
        curves[ic].x = curves[ic].x  # need to do this to transfer data to C++?
    all_curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    return curves, all_curves


def create_equally_spaced_curves(ncurves, nfp, stellsym, R0=1.0, R1=0.5, order=6, numquadpoints=None, use_jax_curve=False):
    """
    Create ``ncurves`` curves of type
    :obj:`~simsopt.geo.curvexyzfourier.CurveXYZFourier` of order
    ``order`` that will result in circular equally spaced coils (major
    radius ``R0`` and minor radius ``R1``) after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries`.

    Usage example: create 4 base curves, which are then rotated 3 times and
    flipped for stellarator symmetry:

    .. code-block::

        base_curves = create_equally_spaced_curves(4, 3, stellsym=True)
        base_currents = [Current(1e5) for c in base_curves]
        coils = coils_via_symmetries(base_curves, base_currents, 3, stellsym=True)

    Args:
        ncurves : int
            Number of curves to create.
        nfp : int
            Field period symmetry of the plasma.
        stellsym : bool
            Whether the plasma has stellarator symmetry.
        R0 : float, optional, default=1.0
            Major radius of the coils.
        R1 : float, optional, default=0.5
            Minor radius of the coils.
        order : int, optional, default=6
            Order of the Fourier series in the planar curve representation.
        numquadpoints : int, optional, default=None
            Number of quadrature points to use.
        use_jax_curve : bool, optional, default=False
            Whether to use JaxCurvePlanarFourier instead of CurvePlanarFourier.

    Returns:
        curves : list
            List of CurvePlanarFourier or JaxCurvePlanarFourier objects.
    """
    from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
    if numquadpoints is None:
        numquadpoints = 15 * order
    if use_jax_curve:
        curvefunc = JaxCurveXYZFourier
    else:
        curvefunc = CurveXYZFourier
    curves = []
    for i in range(ncurves):
        curve = curvefunc(numquadpoints, order)
        angle = (i + 0.5) * (2 * np.pi) / ((1 + int(stellsym)) * nfp * ncurves)
        curve.set("xc(0)", cos(angle) * R0)
        curve.set("xc(1)", cos(angle) * R1)
        curve.set("yc(0)", sin(angle) * R0)
        curve.set("yc(1)", sin(angle) * R1)
        # The the next line, the minus sign is for consistency with
        # Vmec.external_current(), so the coils create a toroidal field of the
        # proper sign and free-boundary equilibrium works following stage-2 optimization.
        curve.set("zs(1)", -R1)
        curve.x = curve.x  # need to do this to transfer data to C++
        curves.append(curve)
    return curves


def create_equally_spaced_planar_curves(
        ncurves, nfp, stellsym, R0=1.0, R1=0.5, 
        order=6, numquadpoints=None, use_jax_curve=False):
    """
    Create ``ncurves`` curves of type
    :obj:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier` of order
    ``order`` that will result in circular equally spaced coils (major
    radius ``R0`` and minor radius ``R1``) after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries`.

    Args:
        ncurves : int
            Number of curves to create.
        nfp : int
            Field period symmetry of the plasma.
        stellsym : bool
            Whether the plasma has stellarator symmetry.
        R0 : float, optional, default=1.0
            Major radius of the coils.
        R1 : float, optional, default=0.5
            Minor radius of the coils.
        order : int, optional, default=6
            Order of the Fourier series in the planar curve representation.
        numquadpoints : int, optional, default=None
            Number of quadrature points to use.
        use_jax_curve : bool, optional, default=False
            Whether to use JaxCurvePlanarFourier instead of CurvePlanarFourier.

    Returns:
        curves : list
            List of CurvePlanarFourier or JaxCurvePlanarFourier objects.
    """
    from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
    if numquadpoints is None:
        numquadpoints = 15 * order
    if use_jax_curve:
        curvefunc = JaxCurvePlanarFourier
    else:
        curvefunc = CurvePlanarFourier
    curves = []
    for k in range(ncurves):
        angle = (k + 0.5) * (2 * np.pi) / ((1 + int(stellsym)) * nfp * ncurves)
        curve = curvefunc(numquadpoints, order)
        rcCoeffs = np.zeros(order+1)
        rcCoeffs[0] = R1
        rsCoeffs = np.zeros(order)
        center = [R0 * cos(angle), R0 * sin(angle), 0]
        rotation = [1, -cos(angle), -sin(angle), 0]
        dofs = np.zeros(len(curve.get_dofs()))

        j = 0
        for i in rcCoeffs:
            dofs[j] = i
            j += 1
        for i in rsCoeffs:
            dofs[j] = i
            j += 1
        for i in rotation:
            dofs[j] = i
            j += 1
        for i in center:
            dofs[j] = i
            j += 1

        curve.set_dofs(dofs)
        curves.append(curve)
    return curves
