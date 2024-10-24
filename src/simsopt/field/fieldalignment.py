"""Implements optimization of the angle between the magnetic field and the ReBCO c-axis."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt.field import BiotSavart
from simsopt._core.derivative import Derivative
from simsopt.field.selffield import regularization_rect, B_regularized_pure
from simsopt.geo.framedcurve import rotated_centroid_frame
from simsopt.geo.jit import jit
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

__all__ = ['CriticalCurrentOpt']

def vector_matrix_product(matrix, vector):
    """Product between matrix and a vector with one free axis (e.g. along a coil)
    
    Args:
      - matrix: array of size ??
      - vector: array of size ??

    Output:
      - array of shape ??
    """
    #TODO: complete documentation with help of P. Huslage
    return jnp.einsum('ijk, ik -> ij', matrix, vector)


def inner(vector_1, vector_2):
    """Dot Product between two vectors with one free axis (e.g. along a coil)
    
    Args:
      - vector_1: first vector of shape (N,3)
      - vector_2: second vector of shape (N,3)

    Output:
      - array of shape (N,)
    """
    return jnp.sum(vector_1*vector_2, axis=1)


def rotation(angle, axis, vector):
    """Rotates a 3D vector around an axis vector
    
    Args:
      - angle: ??
      - axis: ??
      - vector: Vector to be rotated, shape (N,3)

    Output:
      - array of shape ??
    """
    #TODO: complete documentation with help of P. Huslage

    n_quad = angle.shape[0]

    rotation_matrix = jnp.array([[[jnp.cos(angle[i])+axis[i, 0]**2*(1-jnp.cos(angle[i])),
                                  axis[i, 0]*axis[i, 1] *
                                  (1-jnp.cos(angle[i])) -
                                  axis[i, 2]*jnp.sin(angle[i]),
                                  axis[i, 0]*axis[i, 2]*(1-jnp.cos(angle[i])) + axis[i, 1]*jnp.sin(angle[i])],

                                 [axis[i, 1]*axis[i, 0]*(1-jnp.cos(angle[i])) + axis[i, 2]*jnp.sin(angle[i]),
                                  jnp.cos(angle[i]) + axis[i, 1]**2 *
                                  (1-jnp.cos(angle[i])),
                                  axis[i, 1]*axis[i, 2]*(1-jnp.cos(angle[i]))-axis[i, 0]*jnp.sin(angle[i])],

                                 [axis[i, 0]*axis[i, 2]*(1-jnp.cos(angle[i])) - axis[i, 1]*jnp.sin(angle[i]),
                                  axis[i, 1]*axis[i, 2] *
                                  (1-jnp.cos(angle[i])) +
                                  axis[i, 0]*jnp.sin(angle[i]),
                                  jnp.cos(angle[i]) + axis[i, 2]**2*(1-jnp.cos(angle[i]))]] for i in range(n_quad)])

    return vector_matrix_product(rotation_matrix, vector)


def c_axis_pure(t, b):
    """Implements THEVA-tape-like c-axis
    
    Args:
      - t: tape tangent vector, shape (N,3)
      - b: tape binormal vector, shape (N,3)

    Output:
      - array of shape (N,3)
    """
    n_quad = t.shape[0]
    rotation_degrees = 60*jnp.ones(n_quad)
    rotation_radians = jnp.radians(rotation_degrees)
    c_axis = rotation(rotation_radians, t, b)

    return c_axis


def c_axis_angle_pure(coil, B):
    """Angle between the magnetic field and the c-axis of the REBCO Tape
    
    Args:
      - coil: Instance of simsopt.field.coil.Coil
      - B: Magnetic field evaluated along coil.curve. Array of shape (N,3), where N is coil.curve.quadpoints.size

    Output:
      - ??
    """
    #TODO: complete documentation with P.Huslage help
    t, _, b = coil.curve.rotated_frame()
    b_norm = jnp.einsum("ij, i->ij", B, 1/jnp.linalg.norm(B, axis=1))

    c_axis = c_axis_pure(t, b)
    angle = jnp.arccos(inner(c_axis, b_norm))

    return angle


def critical_current_pure(gamma, gammadash, alpha, Bself, Bext, model=None):
    """Evaluate the critical current along a coil, based on a critical current model provided by the user. If no critical current model is provided, the reduced Kim-like model (doi:10.1088/0953-2048/24/6/065005) is used.
    
    Args:
      - gamma: Position of the curve in cartesian coordinates, shape (N,3)
      - gammadash: Derivative of gamma w.r.t the curve arclength, shape (N,3)
      - alpha: HTS frame rotation angle, shape (N,)
      - Bself: Self-field from the coil, evaluated at the position given by gamma. Shape (N,3)
      - Bext: External field, generated for example by other coils. Evaluated at the position given by gamma. Shape (N,3).
      - model: function, taking as argument (B_parallel, B_perp) and returning the critical current. Should be Jaxable.

    Returns:
      - critical current along the curve. Shape (N,)
    """
    tangent, normal, binormal = rotated_centroid_frame(
        gamma, gammadash, alpha
        )

    # Evaluate field
    field = Bself + Bext
    #B_proj = field #- inner(field, tangent)[:, None] * tangent # No need to remove the tangent part of the field; (t,n,b) is an orthogonal coordinate system
    B_perp = inner(field, normal)
    B_par = inner(field, binormal)

    # Fit parameters for reduced Kim-like model of the critical current (doi:10.1088/0953-2048/24/6/065005)
    if model is None:
        xi = -0.7
        k = 0.3
        B_0 = 42.6e-3
        Ic_0 = 1  # 1.3e11,

        model = lambda B_par, B_perp: Ic_0*(1 + jnp.sqrt(k**2 * B_par**2 + B_perp**2) / B_0)**xi

    #TODO: make a file with different critical current models
    return model(B_par, B_perp)


def critical_current_obj_pure(gamma, gammadash, alpha, Bself, Bext, p=-2.0, model=None, threshold=0.5):
    """Evaluates the critical current objective, given by
    
    .. math::
        J = \left(\frac{1}{N}\sum_{k=1}^N I_{c,k}^(1/p)\right)^p,
    
    where :math:`N` is the number of quadrature points :math:`\mathbf{x}_k` along the curve, :math:`I_{c,k}` is the critical current evaluated at :math:`\mathbf{x}_k`, and :math:`p` is a parameter, with :math:`p>1`.

    This is constructed to return the minimum critical current along the curve, approximated by a p-norm.

    Args:
      - gamma: Position of the curve in cartesian coordinates, shape (N,3)
      - gammadash: Derivative of gamma w.r.t the curve arclength, shape (N,3)
      - alpha: HTS frame rotation angle, shape (N,)
      - Bself: Self-field from the coil, evaluated at the position given by gamma. Shape (N,3)
      - Bext: External field, generated for example by other coils. Evaluated at the position given by gamma. Shape (N,3).
      - p: float, larger than 1. Default value is 10.
      - model: function, taking as argument (B_parallel, B_perp) and returning the critical current. Should be Jaxable. Default is None (i.e. use an analytical model).

    Output:
      - Objective value, float.
    """
    Ic = critical_current_pure(
        gamma, gammadash, alpha, Bself, Bext, model
    )

    # TODO: clarify this thing.
    # Step 1: Maximize critical current for one winding.    
    # Step 1: P-norm approx of the critical current minimum

    #arc_length = jnp.linalg.norm(gammadash, axis=1)
    #return jnp.mean(arc_length * jnp.maximum(threshold-Ic,0)**2)

    return -jnp.sum(Ic**p)**(1./p)
    #return jnp.mean(jnp.maximum(threshold-Ic, 0)**p * arc_length)

    # Step 2: Penalize current above critical current, given temperature and number of winds
    # Ic is the current in the coil
    # Ic0 is the critical current along the coil
    #return jnp.sum(jnp.maximum(Ic-Ic0, 0)**2)


class CriticalCurrentOpt(Optimizable):
    """Optimizable class to optimize the critical on a ReBCO coil
    
    Args:
      - self_field: instance of simsopt.field.selffield.SelfField.
      - frame: HTS frame associated to self_field._curve.
      - biotsavart_ext: biotsavart object to evaluate external field.
      - model: critical current model, default is None
      - p: float, larger than 1. Used to approximate critical current minimum with a p-norm.
    """

    def __init__(self, self_field, frame, biotsavart_ext, model=None, p=4, threshold=0.5):
        self.self_field = self_field
        self._curve = self_field._curve #shorthand
        self._current = self_field._current #shorthand
        self.frame = frame
        self.B_ext = biotsavart_ext
        self.B_ext.set_points(self.self_field.get_points_cart())
        self.model = model
        self.p = p

        self.critical_current_jax = lambda gamma, gammadash, alpha, Bself, Bext: critical_current_pure(gamma, gammadash, alpha, Bself, Bext, model=self.model)

        self.J_jax = jit(lambda gamma, gammadash, alpha, Bself, Bext: critical_current_obj_pure(gamma, gammadash, alpha, Bself, Bext, p=self.p, model=self.model, threshold=threshold))

        self.thisgrad0 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=0)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad1 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=1)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad2 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=2)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad3 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=3)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad4 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=4)(gamma, gammadash, alpha, Bself, Bext))

        super().__init__(depends_on=[self_field, frame, biotsavart_ext])

    def critical_current(self):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        alpha = self.frame.rotation.alpha(self._curve.quadpoints)
        Bself = self.self_field.B()
        Bext = self.B_ext.B()
        return self.critical_current_jax(gamma, gammadash, alpha, Bself, Bext)

    def J(self):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        alpha = self.frame.rotation.alpha(self._curve.quadpoints)
        Bself = self.self_field.B()
        Bext = self.B_ext.B()

        return self.J_jax(gamma, gammadash, alpha, Bself, Bext)

    @derivative_dec
    def dJ(self):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        alpha = self.frame.rotation.alpha(self._curve.quadpoints)
        Bself = self.self_field.B()
        Bext = self.B_ext.B()

        grad0 = self.thisgrad0(gamma, gammadash, alpha, Bself, Bext)
        grad1 = self.thisgrad1(gamma, gammadash, alpha, Bself, Bext)
        grad2 = self.thisgrad2(gamma, gammadash, alpha, Bself, Bext)
        grad3 = self.thisgrad3(gamma, gammadash, alpha, Bself, Bext)
        grad4 = self.thisgrad4(gamma, gammadash, alpha, Bself, Bext)

        out = self._curve.dgamma_by_dcoeff_vjp(grad0) \
            + self._curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.frame.rotation.dalpha_by_dcoeff_vjp(self._curve.quadpoints, grad2) \
            + self.self_field.B_vjp(grad3) \
            + self.B_ext.B_vjp(grad4)

        return out
    
    return_fn_map = {'J': J, 'dJ': dJ}
