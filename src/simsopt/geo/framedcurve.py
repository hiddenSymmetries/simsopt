import numpy as np
import jax.numpy as jnp
from jax import vjp, jvp
import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative
from .curve import Curve
from .jit import jit

__all__ = ['FramedCurve', 'FramedCurveFrenet', 'FramedCurveCentroid',
           'FrameRotation', 'ZeroRotation', 'FramedCurve']


class FramedCurve(sopp.Curve, Curve):

    def __init__(self, curve, rotation=None):
        """
        A FramedCurve defines an orthonormal basis around a Curve, 
        where one basis is taken to be the tangent along the Curve. 
        The frame is defined with respect to a reference frame,
        either centroid or frenet. A rotation angle defines the rotation 
        with respect to this reference frame. 
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        deps = [curve]
        if rotation is not None:
            deps.append(rotation)
        if rotation is None:
            rotation = ZeroRotation(curve.quadpoints)
        self.rotation = rotation
        Curve.__init__(self, depends_on=deps)


class FramedCurveFrenet(FramedCurve):
    r"""
    Given a curve, one defines a reference frame using the Frenet normal and
    binormal vectors:

    tangent = dr/dl

    normal = (dtangent/dl)/||dtangent/dl||

    binormal = tangent x normal 

    In addition, we specify an angle along the curve that 
    defines the rotation with respect to this reference frame. 
    """

    def __init__(self, curve, rotation=None):
        FramedCurve.__init__(self, curve, rotation)

        self.binorm = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash: binormal_curvature_pure_frenet(
            gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash))
        self.binormgrad_vjp0 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(g, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash), gamma)[1](v)[0])
        self.binormgrad_vjp1 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, g, gammadashdash, gammadashdashdash, alpha, alphadash), gammadash)[1](v)[0])
        self.binormgrad_vjp2 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, gammadash, g, gammadashdashdash, alpha, alphadash), gammadashdash)[1](v)[0])
        self.binormgrad_vjp3 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, gammadash, gammadashdash, g, alpha, alphadash), gammadashdashdash)[1](v)[0])
        self.binormgrad_vjp4 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, gammadash, gammadashdash, gammadashdashdash, g, alphadash), alpha)[1](v)[0])
        self.binormgrad_vjp5 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, g), alphadash)[1](v)[0])

        self.torsion = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash: torsion_pure_frenet(
            gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash))
        self.torsiongrad_vjp0 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(g, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash), gamma)[1](v)[0])
        self.torsiongrad_vjp1 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, g, gammadashdash, gammadashdashdash, alpha, alphadash), gammadash)[1](v)[0])
        self.torsiongrad_vjp2 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, gammadash, g, gammadashdashdash, alpha, alphadash), gammadashdash)[1](v)[0])
        self.torsiongrad_vjp3 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, gammadash, gammadashdash, g, alpha, alphadash), gammadashdashdash)[1](v)[0])
        self.torsiongrad_vjp4 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, gammadash, gammadashdash, gammadashdashdash, g, alphadash), alpha)[1](v)[0])
        self.torsiongrad_vjp5 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, g), alphadash)[1](v)[0])

    def rotated_frame(self):
        return rotated_frenet_frame(self.curve.gamma(), self.curve.gammadash(), self.curve.gammadashdash(), self.rotation.alpha(self.curve.quadpoints))

    def rotated_frame_dash(self):
        return rotated_frenet_frame_dash(
            self.curve.gamma(), self.curve.gammadash(), self.curve.gammadashdash(), self.curve.gammadashdashdash(),
            self.rotation.alpha(self.curve.quadpoints), self.rotation.alphadash(self.curve.quadpoints)
        )

    def frame_torsion(self):
        """Exports frame torsion along a curve"""
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        d3gamma = self.curve.gammadashdashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)
        return self.torsion(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash)

    def frame_binormal_curvature(self):
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        d3gamma = self.curve.gammadashdashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)
        return self.binorm(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash)

    def dframe_torsion_by_dcoeff_vjp(self, v):
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        d3gamma = self.curve.gammadashdashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)

        grad0 = self.torsiongrad_vjp0(gamma, d1gamma, d2gamma,
                                      d3gamma, alpha, alphadash, v)
        grad1 = self.torsiongrad_vjp1(gamma, d1gamma, d2gamma,
                                      d3gamma, alpha, alphadash, v)
        grad2 = self.torsiongrad_vjp2(gamma, d1gamma, d2gamma,
                                      d3gamma, alpha, alphadash, v)
        grad3 = self.torsiongrad_vjp3(gamma, d1gamma, d2gamma,
                                      d3gamma, alpha, alphadash, v)
        grad4 = self.torsiongrad_vjp4(gamma, d1gamma, d2gamma,
                                      d3gamma, alpha, alphadash, v)
        grad5 = self.torsiongrad_vjp5(gamma, d1gamma, d2gamma,
                                      d3gamma, alpha, alphadash, v)

        return self.curve.dgamma_by_dcoeff_vjp(grad0) \
            + self.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2) \
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(grad3) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4) \
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)

    def dframe_binormal_curvature_by_dcoeff_vjp(self, v):
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        d3gamma = self.curve.gammadashdashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)

        grad0 = self.binormgrad_vjp0(gamma, d1gamma, d2gamma,
                                     d3gamma, alpha, alphadash, v)
        grad1 = self.binormgrad_vjp1(gamma, d1gamma, d2gamma,
                                     d3gamma, alpha, alphadash, v)
        grad2 = self.binormgrad_vjp2(gamma, d1gamma, d2gamma,
                                     d3gamma, alpha, alphadash, v)
        grad3 = self.binormgrad_vjp3(gamma, d1gamma, d2gamma,
                                     d3gamma, alpha, alphadash, v)
        grad4 = self.binormgrad_vjp4(gamma, d1gamma, d2gamma,
                                     d3gamma, alpha, alphadash, v)
        grad5 = self.binormgrad_vjp5(gamma, d1gamma, d2gamma,
                                     d3gamma, alpha, alphadash, v)

        return self.curve.dgamma_by_dcoeff_vjp(grad0) \
            + self.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2) \
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(grad3) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4) \
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)

    def rotated_frame_dcoeff_vjp(self, v, dn, db, arg=0):
        assert arg in [0, 1, 2, 3]
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        gdd = self.curve.gammadashdash()
        a = self.rotation.alpha(self.curve.quadpoints)
        zero = np.zeros_like(v)
        if arg == 0:
            return rotated_frenet_frame_dcoeff_vjp0(
                g, gd, gdd, a, (zero, dn*v, db*v))
        elif arg == 1:
            return rotated_frenet_frame_dcoeff_vjp1(
                g, gd, gdd, a, (zero, dn*v, db*v))
        elif arg == 2:
            return rotated_frenet_frame_dcoeff_vjp2(
                g, gd, gdd, a, (zero, dn*v, db*v))
        elif arg == 3:
            return rotated_frenet_frame_dcoeff_vjp3(
                g, gd, gdd, a, (zero, dn*v, db*v))

    def rotated_frame_dash_dcoeff_vjp(self, v, dn, db, arg=0):
        assert arg in [0, 1, 2, 3, 4, 5]
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        gdd = self.curve.gammadashdash()
        gddd = self.curve.gammadashdashdash()
        a = self.rotation.alpha(self.curve.quadpoints)
        ad = self.rotation.alphadash(self.curve.quadpoints)
        zero = np.zeros_like(v)
        if arg == 0:
            return rotated_frenet_frame_dash_dcoeff_vjp0(
                g, gd, gdd, gddd, a, ad, (zero, dn*v, db*v))
        if arg == 1:
            return rotated_frenet_frame_dash_dcoeff_vjp1(
                g, gd, gdd, gddd, a, ad, (zero, dn*v, db*v))
        if arg == 2:
            return rotated_frenet_frame_dash_dcoeff_vjp2(
                g, gd, gdd, gddd, a, ad, (zero, dn*v, db*v))
        if arg == 3:
            return rotated_frenet_frame_dash_dcoeff_vjp3(
                g, gd, gdd, gddd, a, ad, (zero, dn*v, db*v))
        if arg == 4:
            return rotated_frenet_frame_dash_dcoeff_vjp4(
                g, gd, gdd, gddd, a, ad, (zero, dn*v, db*v))
        if arg == 5:
            return rotated_frenet_frame_dash_dcoeff_vjp5(
                g, gd, gdd, gddd, a, ad, (zero, dn*v, db*v))


class FramedCurveCentroid(FramedCurve):
    """
    Implementation of the centroid frame introduced in
    Singh et al, "Optimization of finite-build stellarator coils",
    Journal of Plasma Physics 86 (2020),
    doi:10.1017/S0022377820000756. 
    Given a curve, one defines a reference frame using the normal and
    binormal vector based on the centoid of the coil. In addition, we specify an 
    angle along the curve that defines the rotation with respect to this 
    reference frame. 

    The idea is explained well in Figure 1 in the reference above.
    """

    def __init__(self, curve, rotation=None):
        FramedCurve.__init__(self, curve, rotation)

        self.torsion = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash: torsion_pure_centroid(
            gamma, gammadash, gammadashdash, alpha, alphadash))
        self.torsiongrad_vjp0 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(g, gammadash, gammadashdash, alpha, alphadash), gamma)[1](v)[0])
        self.torsiongrad_vjp1 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, g, gammadashdash, alpha, alphadash), gammadash)[1](v)[0])
        self.torsiongrad_vjp2 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, gammadash, g, alpha, alphadash), gammadashdash)[1](v)[0])
        self.torsiongrad_vjp4 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, gammadash, gammadashdash, g, alphadash), alpha)[1](v)[0])
        self.torsiongrad_vjp5 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.torsion(gamma, gammadash, gammadashdash, alpha, g), alphadash)[1](v)[0])

        self.binorm = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash: binormal_curvature_pure_centroid(
            gamma, gammadash, gammadashdash, alpha, alphadash))
        self.binormgrad_vjp0 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(g, gammadash, gammadashdash, alpha, alphadash), gamma)[1](v)[0])
        self.binormgrad_vjp1 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, g, gammadashdash, alpha, alphadash), gammadash)[1](v)[0])
        self.binormgrad_vjp2 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, gammadash, g, alpha, alphadash), gammadashdash)[1](v)[0])
        self.binormgrad_vjp4 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, gammadash, gammadashdash, g, alphadash), alpha)[1](v)[0])
        self.binormgrad_vjp5 = jit(lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
            lambda g: self.binorm(gamma, gammadash, gammadashdash, alpha, g), alphadash)[1](v)[0])

    def frame_torsion(self):
        """Exports frame torsion along a curve"""
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)
        return self.torsion(gamma, d1gamma, d2gamma, alpha, alphadash)

    def frame_binormal_curvature(self):
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)
        return self.binorm(gamma, d1gamma, d2gamma, alpha, alphadash)

    def rotated_frame(self):
        return rotated_centroid_frame(self.curve.gamma(), self.curve.gammadash(),
                                      self.rotation.alpha(self.curve.quadpoints))

    def rotated_frame_dash(self):
        return rotated_centroid_frame_dash(
            self.curve.gamma(), self.curve.gammadash(), self.curve.gammadashdash(),
            self.rotation.alpha(self.curve.quadpoints), self.rotation.alphadash(self.curve.quadpoints)
        )

    def rotated_frame_dcoeff_vjp(self, v, dn, db, arg=0):
        assert arg in [0, 1, 2, 3]
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        a = self.rotation.alpha(self.curve.quadpoints)
        zero = np.zeros_like(v)
        if arg == 0:
            return rotated_centroid_frame_dcoeff_vjp0(
                g, gd, a, (zero, dn*v, db*v))
        if arg == 1:
            return rotated_centroid_frame_dcoeff_vjp1(
                g, gd, a, (zero, dn*v, db*v))
        if arg == 2:
            return None
        if arg == 3:
            return rotated_centroid_frame_dcoeff_vjp3(
                g, gd, a, (zero, dn*v, db*v))

    def rotated_frame_dash_dcoeff_vjp(self, v, dn, db, arg=0):
        assert arg in [0, 1, 2, 3, 4, 5]
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        gdd = self.curve.gammadashdash()
        a = self.rotation.alpha(self.curve.quadpoints)
        ad = self.rotation.alphadash(self.curve.quadpoints)
        zero = np.zeros_like(v)
        if arg == 0:
            return rotated_centroid_frame_dash_dcoeff_vjp0(
                g, gd, gdd, a, ad, (zero, dn*v, db*v))
        if arg == 1:
            return rotated_centroid_frame_dash_dcoeff_vjp1(
                g, gd, gdd, a, ad, (zero, dn*v, db*v))
        if arg == 2:
            return rotated_centroid_frame_dash_dcoeff_vjp2(
                g, gd, gdd, a, ad, (zero, dn*v, db*v))
        if arg == 3:
            return None
        if arg == 4:
            return rotated_centroid_frame_dash_dcoeff_vjp4(
                g, gd, gdd, a, ad, (zero, dn*v, db*v))
        if arg == 5:
            return rotated_centroid_frame_dash_dcoeff_vjp5(
                g, gd, gdd, a, ad, (zero, dn*v, db*v))

    def dframe_binormal_curvature_by_dcoeff_vjp(self, v):
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)

        grad0 = self.binormgrad_vjp0(gamma, d1gamma, d2gamma,
                                     alpha, alphadash, v)
        grad1 = self.binormgrad_vjp1(gamma, d1gamma, d2gamma,
                                     alpha, alphadash, v)
        grad2 = self.binormgrad_vjp2(gamma, d1gamma, d2gamma,
                                     alpha, alphadash, v)
        grad4 = self.binormgrad_vjp4(gamma, d1gamma, d2gamma,
                                     alpha, alphadash, v)
        grad5 = self.binormgrad_vjp5(gamma, d1gamma, d2gamma,
                                     alpha, alphadash, v)

        return self.curve.dgamma_by_dcoeff_vjp(grad0) \
            + self.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4) \
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)

    def dframe_torsion_by_dcoeff_vjp(self, v):
        gamma = self.curve.gamma()
        d1gamma = self.curve.gammadash()
        d2gamma = self.curve.gammadashdash()
        alpha = self.rotation.alpha(self.curve.quadpoints)
        alphadash = self.rotation.alphadash(self.curve.quadpoints)

        grad0 = self.torsiongrad_vjp0(gamma, d1gamma, d2gamma,
                                      alpha, alphadash, v)
        grad1 = self.torsiongrad_vjp1(gamma, d1gamma, d2gamma,
                                      alpha, alphadash, v)
        grad2 = self.torsiongrad_vjp2(gamma, d1gamma, d2gamma,
                                      alpha, alphadash, v)
        grad4 = self.torsiongrad_vjp4(gamma, d1gamma, d2gamma,
                                      alpha, alphadash, v)
        grad5 = self.torsiongrad_vjp5(gamma, d1gamma, d2gamma,
                                      alpha, alphadash, v)

        return self.curve.dgamma_by_dcoeff_vjp(grad0) \
            + self.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4) \
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)


class FrameRotation(Optimizable):

    def __init__(self, quadpoints, order, scale=1., dofs=None):
        """
        Defines the rotation angle with respect to a reference orthonormal 
        frame (either frenet or centroid). For example, can be used to 
        define the rotation of a multifilament pack; alpha in Figure 1 of
        Singh et al, "Optimization of finite-build stellarator coils",
        Journal of Plasma Physics 86 (2020),
        doi:10.1017/S0022377820000756
        """
        self.order = order
        if dofs is None:
            super().__init__(x0=np.zeros((2*order+1, )))
        else:
            super().__init__(dofs=dofs)
        self.quadpoints = quadpoints
        self.scale = scale
        self.jac = rotation_dcoeff(quadpoints, order)
        self.jacdash = rotationdash_dcoeff(quadpoints, order)
        self.jax_alpha = jit(lambda dofs, points: jaxrotation_pure(dofs, points, self.order))
        self.jax_alphadash = jit(lambda dofs, points: jaxrotationdash_pure(dofs, points, self.order))

    def alpha(self, quadpoints):
        return self.scale * self.jax_alpha(self._dofs.full_x, quadpoints)

    def alphadash(self, quadpoints):
        return self.scale * self.jax_alphadash(self._dofs.full_x, quadpoints)

    def dalpha_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({self: self.scale * sopp.vjp(v, self.jac)})

    def dalphadash_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({self: self.scale * sopp.vjp(v, self.jacdash)})


class ZeroRotation(Optimizable):

    def __init__(self, quadpoints):
        """
        Dummy class that just returns zero for the rotation angle. Equivalent to using

        .. code-block:: python

            rot = FrameRotation(...)
            rot.fix_all()

        """
        super().__init__()
        self.zero = np.zeros((quadpoints.size, ))

    def alpha(self, quadpoints):
        return self.zero

    def alphadash(self, quadpoints):
        return self.zero

    def dalpha_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({})

    def dalphadash_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({})


@jit
def rotated_centroid_frame(gamma, gammadash, alpha):
    t = gammadash
    t *= 1./jnp.linalg.norm(gammadash, axis=1)[:, None]
    R = jnp.mean(gamma, axis=0)  # centroid
    delta = gamma - R[None, :]
    n = delta - jnp.sum(delta * t, axis=1)[:, None] * t
    n *= 1./jnp.linalg.norm(n, axis=1)[:, None]
    b = jnp.cross(t, n, axis=1)

    # now rotate the frame by alpha
    nn = jnp.cos(alpha)[:, None] * n - jnp.sin(alpha)[:, None] * b
    bb = jnp.sin(alpha)[:, None] * n + jnp.cos(alpha)[:, None] * b
    return t, nn, bb


rotated_centroid_frame_dash = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash: jvp(rotated_centroid_frame,
                                                                  (gamma, gammadash, alpha),
                                                                  (gammadash, gammadashdash, alphadash))[1])

rotated_centroid_frame_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda g: rotated_centroid_frame(g, gammadash, alpha), gamma)[1](v)[0])

rotated_centroid_frame_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda gd: rotated_centroid_frame(gamma, gd, alpha), gammadash)[1](v)[0])

rotated_centroid_frame_dcoeff_vjp3 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda a: rotated_centroid_frame(gamma, gammadash, a), alpha)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda g: rotated_centroid_frame_dash(g, gammadash, gammadashdash, alpha, alphadash), gamma)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda gd: rotated_centroid_frame_dash(gamma, gd, gammadashdash, alpha, alphadash), gammadash)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda gdd: rotated_centroid_frame_dash(gamma, gammadash, gdd, alpha, alphadash), gammadashdash)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp4 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda a: rotated_centroid_frame_dash(gamma, gammadash, gammadashdash, a, alphadash), alpha)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp5 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda ad: rotated_centroid_frame_dash(gamma, gammadash, gammadashdash, alpha, ad), alphadash)[1](v)[0])


@jit
def rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha):
    """Frenet frame of a curve rotated by a angle that varies along the coil path"""

    N = gamma.shape[0]
    t, n, b = (np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)))
    t = gammadash
    t *= 1./jnp.linalg.norm(gammadash, axis=1)[:, None]

    tdash = (1./jnp.linalg.norm(gammadash, axis=1)[:, None])**2 * (jnp.linalg.norm(gammadash, axis=1)[:, None] * gammadashdash
                                                                   - (inner(gammadash, gammadashdash)/jnp.linalg.norm(gammadash, axis=1))[:, None] * gammadash)

    n = tdash
    n *= 1/jnp.linalg.norm(tdash, axis=1)[:, None]
    b = jnp.cross(t, n, axis=1)
    # now rotate the frame by alpha
    nn = jnp.cos(alpha)[:, None] * n - jnp.sin(alpha)[:, None] * b
    bb = jnp.sin(alpha)[:, None] * n + jnp.cos(alpha)[:, None] * b

    return t, nn, bb


rotated_frenet_frame_dash = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash: jvp(rotated_frenet_frame,
                                                                                     (gamma, gammadash,
                                                                                      gammadashdash, alpha),
                                                                                     (gammadash, gammadashdash, gammadashdashdash, alphadash))[1])

rotated_frenet_frame_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda g: rotated_frenet_frame(g, gammadash, gammadashdash, alpha), gamma)[1](v)[0])

rotated_frenet_frame_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda gd: rotated_frenet_frame(gamma, gd, gammadashdash, alpha), gammadash)[1](v)[0])

rotated_frenet_frame_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda gdd: rotated_frenet_frame(gamma, gammadash, gdd, alpha), gammadashdash)[1](v)[0])

rotated_frenet_frame_dcoeff_vjp3 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, v: vjp(
        lambda a: rotated_frenet_frame(gamma, gammadash, gammadashdash, a), alpha)[1](v)[0])

rotated_frenet_frame_dash_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda g: rotated_frenet_frame_dash(g, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash), gamma)[1](v)[0])

rotated_frenet_frame_dash_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda gd: rotated_frenet_frame_dash(gamma, gd, gammadashdash, gammadashdashdash, alpha, alphadash), gammadash)[1](v)[0])

rotated_frenet_frame_dash_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda gdd: rotated_frenet_frame_dash(gamma, gammadash, gdd, gammadashdashdash, alpha, alphadash), gammadashdash)[1](v)[0])

rotated_frenet_frame_dash_dcoeff_vjp3 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda gddd: rotated_frenet_frame_dash(gamma, gammadash, gammadashdash, gddd, alpha, alphadash), gammadashdashdash)[1](v)[0])

rotated_frenet_frame_dash_dcoeff_vjp4 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda a: rotated_frenet_frame_dash(gamma, gammadash, gammadashdash, gammadashdashdash, a, alphadash), alpha)[1](v)[0])

rotated_frenet_frame_dash_dcoeff_vjp5 = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, v: vjp(
        lambda ad: rotated_frenet_frame_dash(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, ad), alphadash)[1](v)[0])


def jaxrotation_pure(dofs, points, order):
    rotation = jnp.zeros((len(points), ))
    rotation += dofs[0]
    for j in range(1, order+1):
        rotation += dofs[2*j-1] * jnp.sin(2*np.pi*j*points)
        rotation += dofs[2*j] * jnp.cos(2*np.pi*j*points)
    return rotation


def jaxrotationdash_pure(dofs, points, order):
    rotation = jnp.zeros((len(points), ))
    for j in range(1, order+1):
        rotation += dofs[2*j-1] * 2*np.pi*j*jnp.cos(2*np.pi*j*points)
        rotation -= dofs[2*j] * 2*np.pi*j*jnp.sin(2*np.pi*j*points)
    return rotation


def rotation_dcoeff(points, order):
    jac = np.zeros((len(points), 2*order+1))
    jac[:, 0] = 1
    for j in range(1, order+1):
        jac[:, 2*j-1] = np.sin(2*np.pi*j*points)
        jac[:, 2*j+0] = np.cos(2*np.pi*j*points)
    return jac


def rotationdash_dcoeff(points, order):
    jac = np.zeros((len(points), 2*order+1))
    for j in range(1, order+1):
        jac[:, 2*j-1] = +2*np.pi*j*np.cos(2*np.pi*j*points)
        jac[:, 2*j+0] = -2*np.pi*j*np.sin(2*np.pi*j*points)
    return jac


def inner(a, b):
    """Inner product for arrays of shape (N, 3)"""
    return np.sum(a*b, axis=1)


def torsion_pure_frenet(gamma, gammadash, gammadashdash, gammadashdashdash,
                        alpha, alphadash):
    """Torsion function for export/evaulate coil sets"""

    _, _, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    _, ndash, _ = rotated_frenet_frame_dash(
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash)

    ndash *= 1/jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(ndash, b)


def binormal_curvature_pure_frenet(gamma, gammadash, gammadashdash, gammadashdashdash,
                                   alpha, alphadash):
    """Binormal curvature function for export/evaulate coil sets."""

    _, _, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    tdash, _, _ = rotated_frenet_frame_dash(
        gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash)

    tdash *= 1/jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(tdash, b)


def torsion_pure_centroid(gamma, gammadash, gammadashdash,
                          alpha, alphadash):
    """Torsion function for export/evaulate coil sets"""

    _, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
    _, ndash, _ = rotated_centroid_frame_dash(
        gamma, gammadash, gammadashdash, alpha, alphadash)

    ndash *= 1/jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(ndash, b)


def binormal_curvature_pure_centroid(gamma, gammadash, gammadashdash,
                                     alpha, alphadash):
    """Binormal curvature function for export/evaulate coil sets."""

    _, _, b = rotated_centroid_frame(gamma, gammadash, alpha)
    tdash, _, _ = rotated_centroid_frame_dash(
        gamma, gammadash, gammadashdash, alpha, alphadash)

    tdash *= 1/jnp.linalg.norm(gammadash, axis=1)[:, None]
    return inner(tdash, b)
