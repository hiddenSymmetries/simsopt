import jax.numpy as jnp
from math import pi, sin, cos
import numpy as np
from .curve import Curve
from .curvexyzfourier import CurveXYZFourier
import simsoptpp as sopp
from simsopt._core.optimizable import DOFs, Optimizable
from .._core.derivative import Derivative
from jax import grad, jit, vjp

__all__ = ['WindowpaneCurve']

def shift_pure( v, xyz ):
    for ii in range(0,3):
        v = v.at[:,ii].add(xyz[ii])
    return v

class Position( Optimizable ):
    def __init__(self, gamma, x, y, z):
        dofs = np.array([x, y, z])
        self._gamma = gamma
        Optimizable.__init__(self, x0=dofs, names=self._make_names())

        self.fun = lambda dofs: shift_pure( jnp.array(self._gamma), jnp.array(dofs) ) 
        self.jac = lambda dofs, v: vjp(self.fun, jnp.array(dofs))[1](v)[0]
    
    def set_gamma(self, gamma):
        self._gamma = gamma

    def _make_names(self):
        return ['x', 'y', 'z']
    
    def set_dofs(self, dofs):
        self.local_x = dofs

    def shift(self):
        return self.fun(self.local_x)
    
    def vjp(self, v): 
        return Derivative({self: self.vjp_impl(v)})
    
    def vjp_impl(self, v):
        return self.jac(self.local_x, v)


def rotate_pure( v, ypr ):        
    yaw = ypr[0]
    pitch = ypr[1]
    roll = ypr[2]

    Myaw = jnp.asarray(
        [[jnp.cos(yaw), -jnp.sin(yaw), 0],
        [jnp.sin(yaw), jnp.cos(yaw), 0],
        [0, 0, 1]]
    )
    Mpitch = jnp.asarray(
        [[jnp.cos(pitch), 0, jnp.sin(pitch)],
        [0, 1, 0],
        [-jnp.sin(pitch), 0, jnp.cos(pitch)]]
    )
    Mroll = jnp.asarray(
        [[1, 0, 0],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll), jnp.cos(roll)]]
    )

    return v @ Myaw @ Mpitch @ Mroll

class Orientation( Optimizable ):
    def __init__(self, gamma, yaw, pitch, roll):
        dofs = np.array([yaw, pitch, roll])
        self._gamma = gamma
        Optimizable.__init__(self, x0=dofs, names=self._make_names())

        self.fun = jit(lambda dofs: rotate_pure( jnp.array(self._gamma), jnp.array(dofs) ) )
        self.jac = jit(lambda dofs, v: vjp(self.fun, jnp.array(dofs))[1](v)[0] )
    
    def set_gamma(self, gamma):
        self._gamma = gamma

    def _make_names(self):
        return ['yaw', 'pitch', 'roll']
    
    def set_dofs(self, dofs):
        self.local_x = dofs
    
    def rotate_array(self):
        return self.fun(self.local_x)
    
    def vjp(self, v):
        return Derivative({self: self.vjp_impl(v)})

    def vjp_impl(self, v):
        return self.jac(self.local_x, v)

class WindowpaneCurve( sopp.Curve, Curve ):
    """
    WindowpaneCurve inherits from the Curve base class. It takes as 
    input a Curve, which is assumed to be centered at the origin
    (xc(0)=yc(0)=zc(0)). The base curve is assumed to be fixed, only
    its orientation can change.

    It is then shifted along a vector (x,z)=(Rc,Zc), and rotated 
    with respect to the yaw, pitch and roll angle.
    """
    def __init__(self, curve, xc, yc, zc, yaw, pitch, roll ):
        self.curve = curve

        sopp.Curve.__init__(self, curve.quadpoints)
        self.position = Position( self.curve.gamma(), xc, yc, zc ) # get rid of that?
        self.orientation = Orientation( self.curve.gamma(), yaw, pitch, roll )
        Curve.__init__(self, depends_on=[curve, self.position, self.orientation])

    def test_curve(self):
        for c in ['xc(0)', 'yc(0)', 'zc(0)']:
            if self.curve.is_free(c):
                raise ValueError(f'Curve {c} should be fixed')
            if self.curve.get(c)!=0:
                raise ValueError(f'Curve should be centered at origin, but {c} is not zero')
   
    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve, :math:`\Gamma`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        self.test_curve()
        if len(quadpoints) == len(self.curve.quadpoints) \
                and np.sum((quadpoints-self.curve.quadpoints)**2) < 1e-15:
            self.orientation.set_gamma( self.curve.gamma() )
            gamma[:] = self.orientation.rotate_array()
            self.position.set_gamma( gamma )
            gamma[:] = self.position.shift()
        else:
            self.curve.gamma_impl(gamma, quadpoints)
            self.orientation.set_gamma( gamma )
            gamma[:] = self.orientation.rotate_array()
            self.position.set_gamma( gamma )
            gamma[:] = self.position.shift()
    
    def gammadash_impl(self, gammadash):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        self.orientation.set_gamma( self.curve.gammadash() )
        gammadash[:] = self.orientation.rotate_array()

    def gammadashdash_impl(self, gammadashdash):
        r"""
        This function returns :math:`\Gamma''(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        self.orientation.set_gamma( self.curve.gammadashdash() )
        gammadashdash[:] = self.orientation.rotate_array( )
    
    def gammadashdashdash_impl(self, gammadashdashdash):
        r"""
        This function returns :math:`\Gamma'''(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        self.orientation.set_gamma( self.curve.gammadashdashdash() )
        gammadashdashdash[:] = self.orientation.rotate_array( )

    # def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
    #     r"""
    #     This function returns

    #     .. math::
    #         \frac{\partial \Gamma}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
    #     coordinates of the curve.

    #     """
    #     dgamma_by_dcoeff[:] = self.orientation.rotate_array( self.curve.dgamma_by_dcoeff() ) + self.position.vjp( dgamma_by_dcoeff ) + self.orientation.vjp( dgamma_by_dcoeff )
        
    # def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
    #     dgammadash_by_dcoeff[:] = self.orientation.rotate_array( self.curve.dgammadash_by_dcoeff() ) + self.position.vjp( dgammadash_by_dcoeff ) + self.orientation.vjp( dgammadash_by_dcoeff )
        
    # def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
    #     dgammadashdash_by_dcoeff[:] = self.orientation.rotate_array( self.curve.dgammadashdash_by_dcoeff() ) + self.position.vjp( dgammadashdash_by_dcoeff ) + self.orientation.vjp( dgammadashdash_by_dcoeff )

    # def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
    #     dgammadashdashdash_by_dcoeff[:] = self.orientation.rotate_array( self.curve.dgammadash_by_dcoeff ) + self.position.vjp( dgammadash_by_dcoeff ) + self.orientation.vjp( dgammadash_by_dcoeff )

    def dgamma_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        dgdcurve =  self.curve.dgamma_by_dcoeff_vjp( v )
        self.orientation.set_gamma( self.curve.gamma() )
        dgdorientation = self.orientation.vjp( v )
        newgamma = self.orientation.rotate_array()
        self.position.set_gamma( newgamma )
        dgdposition = self.position.vjp( v )
        return dgdcurve + dgdposition + dgdorientation
    
    def dgammadash_by_dcoeff_vjp(self, v):
        dgdcurve =  self.curve.dgammadash_by_dcoeff_vjp( v )
        self.orientation.set_gamma( self.curve.gamma() )
        dgdorientation = self.orientation.vjp( v )
        return dgdcurve + dgdorientation
    
    def dgammadashdash_by_dcoeff_vjp(self, v):
        dgdcurve =  self.curve.dgammadashdash_by_dcoeff_vjp( v )
        self.orientation.set_gamma( self.curve.gamma() )
        dgdorientation = self.orientation.vjp( v )
        return dgdcurve + dgdorientation
    
    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        dgdcurve =  self.curve.dgammadashdashdash_by_dcoeff_vjp( v )
        self.orientation.set_gamma( self.curve.gamma() )
        dgdorientation = self.orientation.vjp( v )
        return dgdcurve + dgdorientation
    

