from math import sin, cos

import numpy as np
from jax import vjp, jacfwd, jvp, hessian, grad
import jax.numpy as jnp

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative
from .surfacerzfourier import SurfaceRZFourier
from.surfacexyzfourier import SurfaceXYZFourier
from .surfacexyztensorfourier import SurfaceXYZTensorFourier
from .curve import Curve


from .jit import jit
from .._core.derivative import derivative_dec
from .plotting import fix_matplotlib_3d

__all__ = ['CurveCWSFourierCPP']

def gamma_2d(cdofs, qpts, order, G:int=0, H:int=0):
    """Given some dofs, return curve position in 2D cartesian coordinate
    
    Args:
     - cdofs: Input dofs. Array of size 2*(2*order+1)
     - qpts: quadrature points. Array of floats from 0 to 1, of size N.
     - order: Maximum Fourier series order.

    Returns:
     - phi: Array of size N x 1.
     - theta: Array of size N x 1.
    """
    # Unpack dofs
    phic = cdofs[:order+1]
    phis = cdofs[order+1:2*order+1]
    thetac   = cdofs[2*order+1:3*order+2]
    thetas   = cdofs[3*order+2:]

    # Construct theta and phi arrays
    theta = jnp.zeros((qpts.size,))
    phi = jnp.zeros((qpts.size,))

    ll = qpts*2.0*jnp.pi
    for ii in range(order+1):
        theta = theta + thetac[ii] * jnp.cos(ii*ll)
        phi   = phi   + phic[ii]   * jnp.cos(ii*ll)

    for ii in range(order):
        theta = theta + thetas[ii] * jnp.sin((ii+1)*ll)
        phi   = phi   + phis[ii]   * jnp.sin((ii+1)*ll)

    # Add secular terms
    theta = theta + G * qpts
    phi = phi + H * qpts

    # Prepare output
    out = jnp.zeros((qpts.size, 2))
    out = out.at[:,0].set(phi)
    out = out.at[:,1].set(theta)
    return out

def vjp_contraction_1d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum('ij,i->j',mat,v)

def vjp_contraction_2d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum('ijk,ij->k',mat,v)


class CurveCWSFourierCPP( Curve, sopp.Curve ):
    def __init__(self, quadpoints, order, surf, G=0, H=0, **kwargs):
        # Curve order. Number of Fourier harmonics for phi and theta
        self.order = order
        self.G = G
        self.H = H
        
        # Modes are order as phic, phis, thetac, thetas
        self.modes = [np.zeros((order+1,)), np.zeros((order,)), np.zeros((order+1,)), np.zeros((order,))]

        #self.quadpoints = quadpoints
        self.surf = surf

        # Initialize C++ class and Curve class   
        sopp.Curve.__init__(self, quadpoints)
        Curve.__init__(self, x0=self.get_dofs(), depends_on=[], names=self._make_names(), external_dof_setter=CurveCWSFourierCPP.set_dofs_impl, **kwargs)      

        self.numquadpoints = self.quadpoints.size

        # useful functions
        points = np.asarray(self.quadpoints)
        ones = jnp.ones_like(points)

        ## gamma
        self.gamma_2d_pure = jit(lambda cdofs, qpts: gamma_2d(cdofs, qpts, self.order, self.G, self.H)) 
        self.gamma_2d_jax = jit(lambda cdofs: self.gamma_2d_pure(cdofs, self.quadpoints)) 
        self.dgamma_2d_by_dcoeff_jax = jit(lambda cdofs: jacfwd(self.gamma_2d_jax)(cdofs))
        self.dgamma_2d_by_dcoeff_vjp = jit(lambda cdofs, v: vjp(self.gamma_2d_jax, cdofs)[1](v)[0])

        ## gammadash
        self.gammadash_2d_pure = jit(lambda cdofs, q: jvp(lambda qpts: self.gamma_2d_pure(cdofs, qpts), (q,), (ones,))[1])
        self.gammadash_2d_jax = jit(lambda cdofs: self.gammadash_2d_pure(cdofs, self.quadpoints))
        self.dgammadash_2d_by_dcoeff_jax = jit(lambda cdofs: jacfwd(self.gammadash_2d_jax)(cdofs))
        self.dgammadash_2d_by_dcoeff_vjp = jit(lambda cdofs, v: vjp(self.gammadash_2d_jax, cdofs)[1](v)[0])

        ## gammadashdash
        self.gammadashdash_2d_pure = jit(lambda cdofs, q: jvp(lambda qpts: self.gammadash_2d_pure(cdofs, qpts), (q,), (ones,))[1])
        self.gammadashdash_2d_jax = jit(lambda cdofs: self.gammadashdash_2d_pure(cdofs, self.quadpoints))
        self.dgammadashdash_2d_by_dcoeff_jax = jit(lambda cdofs: jacfwd(self.gammadashdash_2d_jax)(cdofs))
        self.dgammadashdash_2d_by_dcoeff_vjp = jit(lambda cdofs, v: vjp(self.gammadashdash_2d_jax, cdofs)[1](v)[0])

        # Curvature, torsion - no need

        # Normal to the surface along the curve... and its derivatives


    def set_dofs(self, dofs):
        self.local_x = dofs
        sopp.Curve.set_dofs(self, dofs)

    def num_dofs(self):
        return 2*(self.order+1) + 2*self.order
    
    def get_dofs(self):
        return np.concatenate(self.modes)

    def set_dofs_impl(self, dofs):
        self.modes[0] = dofs[0:self.order+1]
        self.modes[1] = dofs[self.order+1:2*self.order+1]
        self.modes[2] = dofs[2*self.order+1:3*self.order+2]
        self.modes[3] = dofs[3*self.order+2:4*self.order+2]

    def _make_names(self):
        dofs_name = []
        for mode in ['phic', 'phis', 'thetac', 'thetas']:
            for ii in range(self.order+1):
                if mode=='phis' and ii==0:
                    continue

                if mode=='thetas' and ii==0:
                    continue

                dofs_name.append(f'{mode}({ii})')

        return dofs_name

    # =========================================================================
    # GAMMA
    # -----
    def gamma_2d(self):
        cdofs = self.get_dofs()
        return self.gamma_2d_jax(cdofs)
    
    def gamma_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:,:] =  self.gamma_2d_pure(cdofs, quadpoints)
    
    def gamma(self):
        g2 = self.gamma_2d()
        out = np.zeros((self.numquadpoints,3))
        self.surf.gamma_lin(out, g2[:,0], g2[:,1])
        return out
    
    def gamma_impl(self, gamma, quadpoints):
        g2 = np.zeros((quadpoints.size,2))
        self.gamma_2d_impl(g2, quadpoints)
        self.surf.gamma_lin(gamma, g2[:,0], g2[:,1])

    def dgamma_2d_by_dcoeff(self):
        cdofs = self.get_dofs()
        return self.dgamma_2d_by_dcoeff_jax(cdofs)

    def dgamma_by_dcoeff(self):
        g2 = self.gamma_2d()
        dsurf_dphi = np.zeros((self.numquadpoints,3)) # shape nqpts x 3
        dsurf_dtheta = np.zeros((self.numquadpoints,3)) # shape nqpts x 3
        self.surf.gammadash1_lin(dsurf_dphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash2_lin(dsurf_dtheta, g2[:,0], g2[:,1])

        dg2_by_dcoeff = self.dgamma_2d_by_dcoeff() # shape nqpts x 2 x ndofs
        dphi_by_dcoeff = dg2_by_dcoeff[:,0,:] # shape nqpts x ndofs
        dtheta_by_dcoeff = dg2_by_dcoeff[:,1,:] # shape nqpts x ndofs

        # Evaluate dgamma_by_dcoeff, size nqpts x 3 x ndofs
        return np.einsum('ij,ik->ijk', dsurf_dphi, dphi_by_dcoeff) + np.einsum('ij,ik->ijk', dsurf_dtheta, dtheta_by_dcoeff)

    def dgamma_by_dcoeff_impl(self, v):
        v[:,:,:] = self.dgamma_by_dcoeff()

    def dgamma_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgamma_by_dcoeff_vjp_impl(v)})
        
    def dgamma_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgamma_by_dcoeff(), v)

    #=========================================================================
    # GAMMADASH
    # ---------
    def gammadash_2d(self):
        cdofs = self.get_dofs()
        return self.gammadash_2d_jax(cdofs)
    
    def gammadash_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:,:] =  self.gammadash_2d_pure(cdofs, quadpoints)

    def dgammadash_2d_by_dcoeff(self):
        cdofs = self.get_dofs()
        return self.dgammadash_2d_by_dcoeff_jax(cdofs)
    
    def gammadash(self):
        g2 = self.gamma_2d()
        dsurf_dphi = np.zeros((self.numquadpoints,3)) # shape nqpts x 3
        dsurf_dtheta = np.zeros((self.numquadpoints,3)) # shape nqpts x 3
        self.surf.gammadash1_lin(dsurf_dphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash2_lin(dsurf_dtheta, g2[:,0], g2[:,1])

        g2dash = self.gammadash_2d()
        phidash = g2dash[:,0] # shape nqpts
        thetadash = g2dash[:,1] # shape nqpts

        # Evaluate dgamma_by_dcoeff, size nqpts x 3
        return np.einsum('ij,i->ij', dsurf_dphi, phidash) + np.einsum('ij,i->ij', dsurf_dtheta, thetadash)
    
    def gammadash_impl(self, gammadash):
        gammadash[:,:] = self.gammadash()

    def dgammadash_by_dcoeff(self):# dgammadash by dcoeff
        g2 = self.gamma_2d()
        dsurf_dphi = np.zeros((self.numquadpoints,3)) 
        dsurf_dtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dphidphi = np.zeros((self.numquadpoints,3))
        dsurf_dphidtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dthetadtheta = np.zeros((self.numquadpoints,3)) 
        self.surf.gammadash1_lin(dsurf_dphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash2_lin(dsurf_dtheta, g2[:,0], g2[:,1])
        self.surf.gammadash1dash1_lin(dsurf_dphidphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash1dash2_lin(dsurf_dphidtheta, g2[:,0], g2[:,1])
        self.surf.gammadash2dash2_lin(dsurf_dthetadtheta, g2[:,0], g2[:,1]) 

        g2dash = self.gammadash_2d()
        phidash = g2dash[:,0]
        thetadash = g2dash[:,1]

        dg2_by_dcoef = self.dgamma_2d_by_dcoeff()
        dphi_by_dcoef = dg2_by_dcoef[:,0,:]
        dtheta_by_dcoef = dg2_by_dcoef[:,1,:]

        dg2dash_by_dcoeff = self.dgammadash_2d_by_dcoeff()
        dphidash_by_dcoeff = dg2dash_by_dcoeff[:,0,:] # shape nqpts x ndofs
        dthetadash_by_dcoeff = dg2dash_by_dcoeff[:,1,:] # shape nqpts x ndofs

        # Evaluate dgamma_by_dcoeff, size nqpts x 3 x ndofs
        return np.einsum('ij,ik->ijk', dsurf_dphi, dphidash_by_dcoeff) \
             + np.einsum('ij,ik->ijk', dsurf_dtheta, dthetadash_by_dcoeff) \
             + np.einsum('ij,i,ik->ijk', dsurf_dphidphi, phidash, dphi_by_dcoef) \
             + np.einsum('ij,i,ik->ijk', dsurf_dphidtheta, phidash, dtheta_by_dcoef) \
             + np.einsum('ij,i,ik->ijk', dsurf_dphidtheta, thetadash, dphi_by_dcoef) \
             + np.einsum('ij,i,ik->ijk', dsurf_dthetadtheta, thetadash, dtheta_by_dcoef)

    def dgammadash_by_dcoeff_impl(self, v):
        v[:,:,:] = self.dgammadash_by_dcoeff()

    def dgammadash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadash_by_dcoeff_vjp_impl(v)})
        
    def dgammadash_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgammadash_by_dcoeff(), v)

 
    #=========================================================================
    # GAMMADASHDASH
    # -------------
    def gammadashdash_2d(self):
        cdofs = self.get_dofs()
        return self.gammadashdash_2d_jax(cdofs)
    
    def gammadashdash_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:,:] =  self.gammadashdash_2d_pure(cdofs, quadpoints)
    
    def gammadashdash(self):
        g2 = self.gamma_2d()
        dsurf_dphi = np.zeros((self.numquadpoints,3)) 
        dsurf_dtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dphidphi = np.zeros((self.numquadpoints,3))
        dsurf_dphidtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dthetadtheta = np.zeros((self.numquadpoints,3)) 
        self.surf.gammadash1_lin(dsurf_dphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash2_lin(dsurf_dtheta, g2[:,0], g2[:,1])
        self.surf.gammadash1dash1_lin(dsurf_dphidphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash1dash2_lin(dsurf_dphidtheta, g2[:,0], g2[:,1])
        self.surf.gammadash2dash2_lin(dsurf_dthetadtheta, g2[:,0], g2[:,1]) 

        g2dash = self.gammadash_2d()
        phidash = g2dash[:,0] # self.numquadpoints
        thetadash = g2dash[:,1] # self.numquadpoints

        g2dashdash = self.gammadashdash_2d()
        phidashdash = g2dashdash[:,0] # self.numquadpoints
        thetadashdash = g2dashdash[:,1] # self.numquadpoints

        return np.einsum('ij,i->ij', dsurf_dphidphi, phidash**2) \
             + np.einsum('ij,i->ij', dsurf_dthetadtheta, thetadash**2) \
             + 2 * np.einsum('ij,i,i->ij', dsurf_dphidtheta, phidash, thetadash) \
             + np.einsum('ij,i->ij', dsurf_dphi, phidashdash) \
             + np.einsum('ij,i->ij', dsurf_dtheta, thetadashdash)
        
    
    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:,:] = self.gammadashdash()

    def dgammadashdash_by_dcoeff(self):
        # This is ugly, but I don't know how to make it better!
        g2 = self.gamma_2d()

        ## First order derivative
        dsurf_dphi = np.zeros((self.numquadpoints,3)) 
        dsurf_dtheta = np.zeros((self.numquadpoints,3)) 
        self.surf.gammadash1_lin(dsurf_dphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash2_lin(dsurf_dtheta, g2[:,0], g2[:,1])

        ## Second order derivative
        dsurf_dphidphi = np.zeros((self.numquadpoints,3))
        dsurf_dphidtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dthetadtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dphidphidphi = np.zeros((self.numquadpoints,3))
        self.surf.gammadash1dash1_lin(dsurf_dphidphi, g2[:,0], g2[:,1]) 
        self.surf.gammadash1dash2_lin(dsurf_dphidtheta, g2[:,0], g2[:,1])
        self.surf.gammadash2dash2_lin(dsurf_dthetadtheta, g2[:,0], g2[:,1]) 

        ## Third order derivative
        dsurf_dphidphidtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dphidthetadtheta = np.zeros((self.numquadpoints,3)) 
        dsurf_dthetadthetadtheta = np.zeros((self.numquadpoints,3)) 
        self.surf.gammadash1dash1dash1_lin(dsurf_dphidphidphi, g2[:,0], g2[:,1])
        self.surf.gammadash1dash1dash2_lin(dsurf_dphidphidtheta, g2[:,0], g2[:,1]) 
        self.surf.gammadash1dash2dash2_lin(dsurf_dphidthetadtheta, g2[:,0], g2[:,1])
        self.surf.gammadash2dash2dash2_lin(dsurf_dthetadthetadtheta, g2[:,0], g2[:,1]) 

        cdofs = self.get_dofs()
        dg2_by_dcoef = self.dgamma_2d_by_dcoeff_jax(cdofs)
        dphi_by_dcoef = dg2_by_dcoef[:,0,:]
        dtheta_by_dcoef = dg2_by_dcoef[:,1,:]

        g2dash = self.gammadash_2d_jax(cdofs)
        phidash = g2dash[:,0] # self.numquadpoints
        thetadash = g2dash[:,1] # self.numquadpoints

        g2dashdash = self.gammadashdash_2d_jax(cdofs)
        phidashdash = g2dashdash[:,0] # self.numquadpoints
        thetadashdash = g2dashdash[:,1] # self.numquadpoints

        dg2dash_by_dcoeff = self.dgammadash_2d_by_dcoeff_jax(cdofs)
        dphidash_by_dcoeff = dg2dash_by_dcoeff[:,0] # self.numquadpoints
        dthetadash_by_dcoeff = dg2dash_by_dcoeff[:,1] # self.numquadpoints

        dg2dashdash_by_dcoeff = self.dgammadashdash_2d_by_dcoeff_jax(cdofs)
        dphidashdash_by_dcoeff = dg2dashdash_by_dcoeff[:,0] # self.numquadpoints
        dthetadashdash_by_dcoeff = dg2dashdash_by_dcoeff[:,1] # self.numquadpoints

        # l1-l6 denotes lines in my hand-written notes...
        l1 = np.einsum('ij,ik,i->ijk', dsurf_dthetadthetadtheta, dtheta_by_dcoef, thetadash**2) \
           + np.einsum('ij,ik,i,i->ijk', dsurf_dphidthetadtheta, dtheta_by_dcoef, thetadash, phidash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dthetadtheta, dthetadash_by_dcoeff, thetadash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dthetadtheta, dtheta_by_dcoef, thetadashdash)
        
        l2 = np.einsum('ij,ik,i,i->ijk', dsurf_dphidthetadtheta, dtheta_by_dcoef, thetadash, phidash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidphidtheta, dtheta_by_dcoef, phidash**2) \
           + np.einsum('ij,ik,i->ijk',dsurf_dphidtheta, dthetadash_by_dcoeff,phidash) \
           + np.einsum('ij,ik,i->ijk',dsurf_dphidtheta,dtheta_by_dcoef,phidashdash)
        
        l3 = np.einsum('ij,ik,i->ijk',dsurf_dthetadtheta,dthetadash_by_dcoeff,thetadash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidtheta, dthetadash_by_dcoeff, phidash) \
           + np.einsum('ij,ik->ijk', dsurf_dtheta, dthetadashdash_by_dcoeff)
        
        l4 = np.einsum('ij,ik,i,i->ijk', dsurf_dphidphidtheta, dphi_by_dcoef, thetadash, phidash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidphidphi, dphi_by_dcoef, phidash**2) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidphi, dphidash_by_dcoeff, phidash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidphi, dphi_by_dcoef, phidashdash)
        
        l5 = np.einsum('ij,ik,i->ijk', dsurf_dphidthetadtheta, dphi_by_dcoef, thetadash**2) \
           + np.einsum('ij,ik,i,i->ijk', dsurf_dphidphidtheta, dphi_by_dcoef, phidash, thetadash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidtheta, dphidash_by_dcoeff, thetadash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidtheta, dphi_by_dcoef, thetadashdash)
        
        l6 = np.einsum('ij,ik,i->ijk', dsurf_dphidtheta, dphidash_by_dcoeff, thetadash) \
           + np.einsum('ij,ik,i->ijk', dsurf_dphidphi, dphidash_by_dcoeff, phidash) \
           + np.einsum('ij,ik->ijk', dsurf_dphi, dphidashdash_by_dcoeff)

        return l1 + l2 + l3 + l4 + l5 + l6

    def dgammadashdash_by_dcoeff_impl(self, v):
        v[:,:,:] = self.dgammadashdash_by_dcoeff()

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadashdash_by_dcoeff_vjp_impl(v)})
        
    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgammadashdash_by_dcoeff(), v)
    
    #=========================================================================
    # NORMAL
    # ------
    def unit_normal(self):
        return self.unit_normal_impl(self.get_dofs())
    
    def unit_normal_impl(self, cdofs):
        g2 = self.gamma_2d_jax(cdofs)
        dxdtheta = np.zeros((self.numquadpoints,3))
        dxdphi = np.zeros((self.numquadpoints,3))
        self.surf.gammadash1_lin(dxdphi, g2[:,0], g2[:,1])
        self.surf.gammadash2_lin(dxdtheta, g2[:,0], g2[:,1])

        normal = np.cross(dxdphi, dxdtheta)
        unit_normal = normal / np.linalg.norm(normal, axis=1 )[:,None]
        return unit_normal


    def dunit_normal_by_dcoeff(self):
        g2 = self.gamma_2d()
        dxdtheta = np.zeros((self.numquadpoints,3))
        dxdphi = np.zeros((self.numquadpoints,3))
        self.surf.gammadash1_lin(dxdphi, g2[:,0], g2[:,1])
        self.surf.gammadash2_lin(dxdtheta, g2[:,0], g2[:,1])

        normal = np.cross(dxdphi, dxdtheta)
        normal_norm = np.linalg.norm( normal, axis=1 )

        dg2_by_dcoeff = self.dgamma_2d_by_dcoeff()
        dxdthetadtheta = np.zeros((self.numquadpoints,3))
        dxdphidtheta = np.zeros((self.numquadpoints,3))
        dxdphidphi = np.zeros((self.numquadpoints,3))
        self.surf.gammadash1dash1_lin(dxdphidphi, g2[:,0], g2[:,1])
        self.surf.gammadash1dash2_lin(dxdphidtheta, g2[:,0], g2[:,1])
        self.surf.gammadash2dash2_lin(dxdthetadtheta, g2[:,0], g2[:,1])

        p0 = np.cross(dxdphi, dxdphidtheta) + np.cross(dxdphidphi, dxdtheta)
        p1 = np.cross(dxdphi, dxdthetadtheta) + np.cross(dxdphidtheta, dxdtheta)
        dnormal_by_dcoeff = np.einsum('ik,ij->ijk', dg2_by_dcoeff[:,0,:], p0) \
                          + np.einsum('ik,ij->ijk', dg2_by_dcoeff[:,1,:], p1)


        t1 = np.einsum('ijk,i->ijk', dnormal_by_dcoeff, 1./normal_norm) # this has shape (nqpts,3,ndofs)
        prod = np.einsum('ij,ijk->ik', normal, dnormal_by_dcoeff)
        t2 = np.einsum('ik,ij,i->ijk', prod, normal, normal_norm**(-3))
        dunit_normal_by_dcoef =  t1 - t2
        return dunit_normal_by_dcoef

    def zfactor(self):
        return self.unit_normal()[:,2]

    def dzfactor_by_dcoeff(self):
        return self.dunit_normal_by_dcoeff()[:,2,:]
        
    def dzfactor_by_dcoeff_vjp(self,v):
        return Derivative({self: vjp_contraction_1d(self.dzfactor_by_dcoeff(), v)})

    def rfactor(self):
        g2 = self.gamma_2d()
        unit_normal = self.unit_normal()

        # Now project in the radial direction...
        return unit_normal[:,0]*np.cos(g2[:,0]) + unit_normal[:,1]*np.sin(g2[:,0])
    
    def drfactor_by_dcoeff(self):
        g2 = self.gamma_2d()
        dg2_by_dcoef = self.dgamma_2d_by_dcoeff()
        unit_normal = self.unit_normal()
        dunit_normal_by_dcoef = self.dunit_normal_by_dcoeff()

        # Now project in the radial direction...
        return dunit_normal_by_dcoef[:,0,:]*np.cos(g2[:,0,None]) + dunit_normal_by_dcoef[:,1,:]*np.sin(g2[:,0,None]) + dg2_by_dcoef[:,0,:]*(-unit_normal[:,0,None]*np.sin(g2[:,0,None]) + unit_normal[:,1,None]*np.cos(g2[:,0,None]))

    def drfactor_by_dcoeff_vjp(self, v):
        return Derivative({self: vjp_contraction_1d(self.drfactor_by_dcoeff(), v)})
