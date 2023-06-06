import numpy as np
import jax.numpy as jnp
from jax import vjp, jvp, grad, debug
from simsopt.geo.jit import jit
from simsopt.geo import ZeroRotation, FilamentRotation, Curve
import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative
from simsopt._core.derivative import derivative_dec
from jax.tree_util import Partial


def normal_curvature_pure(n, tdash):
    
    inner = lambda a, b: np.sum(a*b, axis=1)
    normal_curvature = inner( tdash, n )    
    return normal_curvature

def torsion_pure(ndash, b):
    inner = lambda a, b: np.sum(a*b, axis=1)
    torsion = inner(ndash, b)
    return torsion

torsion2vjp0 = jit(lambda ndash, b, v: vjp(lambda nd: torsion_pure(nd, b), ndash)[1](v)[0])
torsion2vjp1 = jit(lambda ndash, b, v: vjp(lambda bi: torsion_pure(ndash, bi), b)[1](v)[0])


def binormal_curvature_pure(tdash, b):
    inner = lambda a, b: np.sum(a*b, axis=1)
    binormal_curvature = inner(tdash, b)
    return binormal_curvature
 
def anti_twist_pure(n, ndash):
    inner = lambda a, b: np.sum(a*b, axis=1)
    #func = lambda a: np.arccos(inner(a, np.roll(a, 1, axis=0)))
    return inner(n, ndash) * jnp.heaviside(inner(n, ndash), 1)

def torsion_export_pure(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash,  width=12, scale=1):
    inner = lambda a, b: np.sum(a*b, axis=1)

    gamma =  jnp.multiply(gamma, scale)
    #gammadash =  jnp.multiply(gammadash, scale)
    gammadashdash =  jnp.multiply(gammadashdash, 1/scale)
    gammadashdashdash =  jnp.multiply(gammadashdashdash, 1/scale**2)

    t, n, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    tdash, ndash, bdash = rotated_frenet_frame_dash(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash)

    curvature = normal_curvature_pure(n, tdash)
    torsion = torsion_pure(ndash, b)
    binormal_curvature = binormal_curvature_pure(tdash, b)
    anti_twist = anti_twist_pure(n, ndash)
    w = 0#100
    # 12mm tape limits
    t_lim = 12.9
    b_lim = 0.66
    t_lim *= 12 / width
    b_lim *= 12 / width

    result = torsion/t_lim
    return result

def binormal_curvature_export_pure(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash,  width=12, scale=1):
    inner = lambda a, b: np.sum(a*b, axis=1)

    gamma =  jnp.multiply(gamma, scale)
    #gammadash =  jnp.multiply(gammadash, scale)
    gammadashdash =  jnp.multiply(gammadashdash, 1/scale)
    gammadashdashdash =  jnp.multiply(gammadashdashdash, 1/scale**2)

    t, n, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    tdash, ndash, bdash = rotated_frenet_frame_dash(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash)

    curvature = normal_curvature_pure(n, tdash)
    torsion = torsion_pure(ndash, b)
    binormal_curvature = binormal_curvature_pure(tdash, b)
    anti_twist = anti_twist_pure(n, ndash)
    w = 0#100
    # 12mm tape limits
    t_lim = 12.9
    b_lim = 0.66
    t_lim *= 12 / width
    b_lim *= 12 / width
    

    result = binormal_curvature/b_lim
    return result

# Optimization function, limits based on 0.2% strain
#@Partial(jit, static_argnums=(6,))
@jit
def strain_opt_pure(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash,  width=3, scale=1):
    tdash1, ndash1, bdash1 = rotated_frenet_frame_dash(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash)
    inner = lambda a, b: jnp.sum(a*b, axis=1)
    gamma =  jnp.multiply(gamma, scale)
    #gammadash =  jnp.multiply(gammadash, scale)
    gammadashdash =  jnp.multiply(gammadashdash, 1/scale)
    gammadashdashdash =  jnp.multiply(gammadashdashdash, 1/scale**2)
    t, n, b = rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha)
    tdash, ndash, bdash = rotated_frenet_frame_dash(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash)

    curvature = normal_curvature_pure(n, tdash)
    torsion = torsion_pure(ndash, b)
    binormal_curvature = binormal_curvature_pure(tdash, b)
    torsion1=torsion_pure(ndash1, b)

    anti_twist = anti_twist_pure(n, ndash)
    w = 0
    # 12mm tape limits
    t_lim = 12.9
    b_lim = 0.66
    t_lim *= 12 / width
    b_lim *= 12 / width

    result = jnp.sum(jnp.power(torsion/t_lim, 4)) + jnp.sum(jnp.power(binormal_curvature/b_lim, 4)) + w * jnp.sum(anti_twist)
    return result


class strain_opt(Optimizable):

    def __init__(self, curve, width=3, scale=1):
        self.curve = curve
        self.width = width
        self.scale = scale
        #self.J_jax = jit(lambda torsion, binormal_curvature: strain_opt_pure(torsion, binormal_curvature, width))
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: strain_opt_pure(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.thisgrad3 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.thisgrad4 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.thisgrad5 = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: grad(self.J_jax, argnums=5)(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.torsion = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: torsion_export_pure(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        self.binormal_curvature = jit(lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale: binormal_curvature_export_pure(gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash, width, scale))
        super().__init__(depends_on=[curve])

    def torsion_export(self):

        gamma = self.curve.curve.gamma() 
        d1gamma = self.curve.curve.gammadash() 
        d2gamma = self.curve.curve.gammadashdash() 
        d3gamma = self.curve.curve.gammadashdashdash() 
        alpha = self.curve.rotation.alpha(self.curve.quadpoints)
        alphadash = self.curve.rotation.alphadash(self.curve.quadpoints)
        scale = self.scale
        width = self.width
        return self.torsion(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)
    
    def binormal_curvature_export(self):
        width = self.width
        scale = self.scale
        gamma = self.curve.curve.gamma()
        d1gamma = self.curve.curve.gammadash()
        d2gamma = self.curve.curve.gammadashdash()
        d3gamma = self.curve.curve.gammadashdashdash()
        alpha = self.curve.rotation.alpha(self.curve.quadpoints)
        alphadash = self.curve.rotation.alphadash(self.curve.quadpoints)
        

        return self.binormal_curvature(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)

    def c_axis_angle(self, B):
        gamma = self.curve.curve.gamma()
        d1gamma = self.curve.curve.gammadash()
        d2gamma = self.curve.curve.gammadashdash()
        alpha = self.curve.rotation.alpha(self.curve.quadpoints)

        return self.c_axis(gamma, d1gamma, d2gamma, alpha, B)

    def J(self):
        """
        This returns the value of the quantity.
        """
        gamma = self.curve.curve.gamma()
        d1gamma = self.curve.curve.gammadash()
        d2gamma = self.curve.curve.gammadashdash()
        d3gamma = self.curve.curve.gammadashdashdash()
        alpha = self.curve.rotation.alpha(self.curve.quadpoints)
        alphadash = self.curve.rotation.alphadash(self.curve.quadpoints)
        width = self.width
        scale = self.scale
        
        return self.J_jax(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        gamma = self.curve.curve.gamma()
        d1gamma = self.curve.curve.gammadash()
        d2gamma = self.curve.curve.gammadashdash()
        d3gamma = self.curve.curve.gammadashdashdash()
        alpha = self.curve.rotation.alpha(self.curve.quadpoints)
        alphadash = self.curve.rotation.alphadash(self.curve.quadpoints)
        width = self.width
        scale = self.scale

        grad0 = self.thisgrad0(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)
        grad1 = self.thisgrad1(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)
        grad2 = self.thisgrad2(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)
        grad3 = self.thisgrad3(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)
        grad4 = self.thisgrad4(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)
        grad5 = self.thisgrad5(gamma, d1gamma, d2gamma, d3gamma, alpha, alphadash, width, scale)


        return self.curve.curve.dgamma_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.curve.curve.dgammadashdash_by_dcoeff_vjp(grad2) + self.curve.curve.dgammadashdashdash_by_dcoeff_vjp(grad3) \
            + self.curve.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, grad4) + self.curve.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, grad5)

    return_fn_map = {'J': J, 'dJ': dJ}

@jit
def rotated_frenet_frame(gamma, gammadash, gammadashdash, alpha):

    norm = lambda a: jnp.linalg.norm(a, axis=1)
    inner = lambda a, b: jnp.sum(a*b, axis=1)
    N = gamma.shape[0]
    t, n, b = (np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)))
    t = gammadash#(1./l[:, None]) * gammadash
    t *= 1./jnp.linalg.norm(gammadash, axis=1)[:, None]

    tdash = (1./jnp.linalg.norm(gammadash, axis=1)[:, None])**2 * (jnp.linalg.norm(gammadash, axis=1)[:, None] * gammadashdash
                                    - (inner(gammadash, gammadashdash)/jnp.linalg.norm(gammadash, axis=1))[:, None] * gammadash)

    n = tdash#(1./norm(tdash))[:, None] * tdash
    n *= 1/jnp.linalg.norm(tdash, axis=1)[:, None]
    b = jnp.cross(t, n, axis=1)
    # now rotate the frame by alpha
    nn = jnp.cos(alpha)[:, None] * n - jnp.sin(alpha)[:, None] * b
    bb = jnp.sin(alpha)[:, None] * n + jnp.cos(alpha)[:, None] * b

    return t, nn, bb


rotated_frenet_frame_dash = jit(
    lambda gamma, gammadash, gammadashdash, gammadashdashdash, alpha, alphadash: jvp(rotated_frenet_frame,
                                                                  (gamma, gammadash, gammadashdash, alpha),
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

def create_multifilament_grid_frenet(curve, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b, rotation_order=None, rotation_scaling=None):
    """
    Create a regular grid of ``numfilaments_n * numfilaments_b`` many
    filaments to approximate a finite-build coil.

    Note that "normal" and "binormal" in the function arguments here
    refer not to the Frenet frame but rather to the "coil centroid
    frame" defined by Singh et al., before rotation.

    Args:
        curve: The underlying curve.
        numfilaments_n: number of filaments in normal direction.
        numfilaments_b: number of filaments in bi-normal direction.
        gapsize_n: gap between filaments in normal direction.
        gapsize_b: gap between filaments in bi-normal direction.
        rotation_order: Fourier order (maximum mode number) to use in the expression for the rotation
                        of the filament pack. ``None`` means that the rotation is not optimized.
        rotation_scaling: scaling for the rotation degrees of freedom. good
                           scaling improves the convergence of first order optimization
                           algorithms. If ``None``, then the default of ``1 / max(gapsize_n, gapsize_b)``
                           is used.
    """
    if numfilaments_n % 2 == 1:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n//2
    else:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n/2 + 0.5
    shifts_n = shifts_n * gapsize_n
    if numfilaments_b % 2 == 1:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b//2
    else:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b/2 + 0.5
    shifts_b = shifts_b * gapsize_b

    if rotation_scaling is None:
        rotation_scaling = 1/max(gapsize_n, gapsize_b)
    if rotation_order is None:
        rotation = ZeroRotation(curve.quadpoints)
    else:
        rotation = FilamentRotation(curve.quadpoints, rotation_order, scale=rotation_scaling)
    filaments = []
    for i in range(numfilaments_n):
        for j in range(numfilaments_b):
            filaments.append(CurveFilament_frenet(curve, shifts_n[i], shifts_b[j], rotation))
    return filaments


class CurveFilament_frenet(sopp.Curve, Curve):

    def __init__(self, curve, dn, db, rotation=None):
        """
        Implementation as the frenet frame. Given a curve, one defines a normal and
        binormal vector and then creates a grid of curves by shifting along the
        normal and binormal vector. In addition, we specify an angle along the
        curve that allows us to optimise for the rotation of the winding pack.


        Args:
            curve: the underlying curve
            dn: how far to move in normal direction
            db: how far to move in binormal direction
            rotation: angle along the curve to rotate the frame.
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        deps = [curve]
        if rotation is not None:
            deps.append(rotation)
        Curve.__init__(self, depends_on=deps)
        self.curve = curve
        self.dn = dn
        self.db = db
        if rotation is None:
            rotation = ZeroRotation(curve.quadpoints)
        self.rotation = rotation

        # My own stuff
        self.t, self.n, self.b = rotated_frenet_frame(curve.gamma(), curve.gammadash(), curve.gammadashdash(), rotation.alpha(curve.quadpoints))
        self.tdash, self.ndash, self.bdash = rotated_frenet_frame_dash(curve.gamma(), curve.gammadash(), curve.gammadashdash(), curve.gammadashdashdash(), rotation.alpha(curve.quadpoints), rotation.alphadash(curve.quadpoints))

#        self.dnormal_curvature_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.ndash, self.b), x)[1](v)[0])
#        self.dbinormal_curvature_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.ndash, self.b), x)[1](v)[0])
#        self.dtorsion2_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.ndash, self.b), x)[1](v)[0])
    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        c = self.curve
        t, n, b = rotated_frenet_frame(c.gamma(), c.gammadash(), c.gammadashdash(), self.rotation.alpha(c.quadpoints))
        gamma[:] = self.curve.gamma() + self.dn * n + self.db * b

    def gammadash_impl(self, gammadash):
        c = self.curve
        td, nd, bd = rotated_frenet_frame_dash(
            c.gamma(), c.gammadash(), c.gammadashdash(), c.gammadashdashdash(),
            self.rotation.alpha(c.quadpoints), self.rotation.alphadash(c.quadpoints)
        )
        gammadash[:] = self.curve.gammadash() + self.dn * nd + self.db * bd

    def dgamma_by_dcoeff_vjp(self, v):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        gdd = self.curve.gammadashdash()
        a = self.rotation.alpha(self.curve.quadpoints)
        zero = np.zeros_like(v)
        vg = rotated_frenet_frame_dcoeff_vjp0(g, gd, gdd, a, (zero, self.dn*v, self.db*v))
        vgd = rotated_frenet_frame_dcoeff_vjp1(g, gd, gdd, a, (zero, self.dn*v, self.db*v))
        vgdd = rotated_frenet_frame_dcoeff_vjp2(g, gd, gdd, a, (zero, self.dn*v, self.db*v))
        va = rotated_frenet_frame_dcoeff_vjp3(g, gd, gdd, a, (zero, self.dn*v, self.db*v))
        return self.curve.dgamma_by_dcoeff_vjp(v + vg) \
            + self.curve.dgammadash_by_dcoeff_vjp(vgd) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(vgdd) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, va)

    def dgammadash_by_dcoeff_vjp(self, v):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        gdd = self.curve.gammadashdash()
        gddd = self.curve.gammadashdashdash()
        a = self.rotation.alpha(self.curve.quadpoints)
        ad = self.rotation.alphadash(self.curve.quadpoints)
        zero = np.zeros_like(v)

        vg = rotated_frenet_frame_dash_dcoeff_vjp0(g, gd, gdd, gddd, a, ad, (zero, self.dn*v, self.db*v))
        vgd = rotated_frenet_frame_dash_dcoeff_vjp1(g, gd, gdd, gddd, a, ad, (zero, self.dn*v, self.db*v))
        vgdd = rotated_frenet_frame_dash_dcoeff_vjp2(g, gd, gdd, gddd, a, ad, (zero, self.dn*v, self.db*v))
        vgddd = rotated_frenet_frame_dash_dcoeff_vjp3(g, gd, gdd, gddd, a, ad, (zero, self.dn*v, self.db*v))
        va = rotated_frenet_frame_dash_dcoeff_vjp4(g, gd, gdd, gddd, a, ad, (zero, self.dn*v, self.db*v))
        vad = rotated_frenet_frame_dash_dcoeff_vjp5(g, gd, gdd, gddd, a, ad, (zero, self.dn*v, self.db*v))
        return self.curve.dgamma_by_dcoeff_vjp(vg) \
            + self.curve.dgammadash_by_dcoeff_vjp(v+vgd) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(vgdd) \
            + self.curve.dgammadashdashdash_by_dcoeff_vjp(vgddd) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, va) \
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, vad)
    


    def drotated_frenet_frame_by_dcoeff(self):
        r"""
        This function returns the derivative of the curve's Frenet frame, 

        .. math::
            \left(\frac{\partial \mathbf{t}}{\partial \mathbf{c}}, \frac{\partial \mathbf{n}}{\partial \mathbf{c}}, \frac{\partial \mathbf{b}}{\partial \mathbf{c}}\right),

        with respect to the curve dofs, where :math:`(\mathbf t, \mathbf n, \mathbf b)` correspond to the Frenet frame, and :math:`\mathbf c` are the curve dofs.
        """
        c = self.curve
        gamma = c.gamma()
        dgamma_by_dphi = c.gammadash()
        d2gamma_by_dphidphi = c.gammadashdash()
        d3gamma_by_dphiphiphi = c.gammadashdashdash()
        d2gamma_by_dphidcoeff = c.dgammadash_by_dcoeff()
        d3gamma_by_dphidphidcoeff = c.dgammadashdash_by_dcoeff()
        d4gamma_bydphiphiphicoeff = c.dgammadashdashdash_by_dcoeff()

        alpha = self.rotation.alpha(c.quadpoints)

        l = c.incremental_arclength()
        dl_by_dcoeff = c.dincremental_arclength_by_dcoeff()

        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        inner2 = lambda a, b: np.sum(a*b, axis=2)
        outer = lambda a, b: np.einsum('ij, ik -> ijk', a, b)
        outer2 = lambda a,b: np.einsum('ijk, ijl -> ijl' , a, b)
        N = len(self.quadpoints)
        unity = np.repeat(np.identity(3)[None,:], N, axis=0)

        dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff = (np.zeros((N, 3, c.num_dofs())), np.zeros((N, 3, c.num_dofs())), np.zeros((N, 3, c.num_dofs())))
        t, n, b = rotated_frenet_frame(gamma, dgamma_by_dphi, d2gamma_by_dphidphi, alpha)
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



        # Implement derivative of the normal vector and its derivative w.r.t. the DOFs of the curve
        ndash = d3gamma_by_dphiphiphi / norm(d2gamma_by_dphidphi)[:, None]  - ( d2gamma_by_dphidphi * inner(d3gamma_by_dphiphiphi, d2gamma_by_dphidphi)[:, None] ) / norm(d2gamma_by_dphidphi)[:,None]**3
        dndash_by_dcoeff = outer2(((-outer(d3gamma_by_dphiphiphi, d2gamma_by_dphidphi) )/ norm(d2gamma_by_dphidphi)[:,None,None] - 1 / norm(d2gamma_by_dphidphi)[:,None,None] * unity \
                        - outer(d2gamma_by_dphidphi, d3gamma_by_dphiphiphi) / norm(d3gamma_by_dphiphiphi)[:,None,None]**3 \
                        + 3 * inner(d2gamma_by_dphidphi, d3gamma_by_dphiphiphi)[:,None,None] * outer(d2gamma_by_dphidphi, d3gamma_by_dphiphiphi) / norm(d2gamma_by_dphidphi)[:,None,None]**5), d3gamma_by_dphidphidcoeff ) \
                        + outer2((1 / norm(d2gamma_by_dphidphi)[:,None,None] * unity - outer(d2gamma_by_dphidphi, d2gamma_by_dphidphi) / norm(d2gamma_by_dphidphi)[:,None,None]),  d4gamma_bydphiphiphicoeff)




        return dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff, tdash, ndash, dtdash_by_dcoeff, dndash_by_dcoeff
