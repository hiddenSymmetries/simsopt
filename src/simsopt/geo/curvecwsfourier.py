import numpy as np
import jax.numpy as jnp
from jax import vjp, jacfwd, jvp

from .jit import jit
import simsoptpp as sopp
from .._core.json import GSONDecoder
from .._core.derivative import Derivative
from .curve import Curve, JaxCurve, kappa_pure, torsion_pure
from .surfacerzfourier import SurfaceRZFourier
from .._core.optimizable import Optimizable

__all__ = ['CurveCWSFourier', 'CurveCWSFourier2']

def gamma_curve_on_surface(modes, qpts, order, surf_dofs, mpol, ntor, nfp):
    # Unpack dofs
    phic = modes[:order+1]
    phis = modes[order+1:2*order+1]
    thetac   = modes[2*order+1:3*order+2]
    thetas   = modes[3*order+2:]

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


    # Construct curve on surface
    r = jnp.zeros((qpts.size,))
    z = jnp.zeros((qpts.size,))

    nmn = ntor+1 + mpol*(2*ntor+1)
    rc = surf_dofs[:nmn]
    zs = surf_dofs[nmn:]

    th = theta * 2.0 * jnp.pi
    ph = phi   * 2.0 * jnp.pi * nfp
    
    #ph = (1.0+phi) * jnp.pi / (2.0*nfp)

    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<0:
                continue
            counter = counter+1
            r = r + rc[counter] * jnp.cos(mm*th - nn*ph)


    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<=0:
                continue
            counter = counter+1
            z = z + zs[counter] * jnp.sin(mm*th - nn*ph)
            
    gamma = jnp.zeros((qpts.size, 3))
    gamma = gamma.at[:,0].set( r * jnp.cos( phi*jnp.pi*2 ) )
    gamma = gamma.at[:,1].set( r * jnp.sin( phi*jnp.pi*2 ) )
    gamma = gamma.at[:,2].set( z                 )

    return gamma

def normal(modes, qpts, order, surf_dofs, mpol, ntor, nfp):

    # Unpack dofs
    phic = modes[:order+1]
    phis = modes[order+1:2*order+1]
    thetac   = modes[2*order+1:3*order+2]
    thetas   = modes[3*order+2:]

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


    # Construct normal on surface
    r = jnp.zeros((qpts.size,))
    drdt = jnp.zeros((qpts.size,))
    drdp = jnp.zeros((qpts.size,))
    dzdt = jnp.zeros((qpts.size,))
    dzdp = jnp.zeros((qpts.size,))

    nmn = ntor+1 + mpol*(2*ntor+1)
    rc = surf_dofs[:nmn]
    zs = surf_dofs[nmn:]

    th = theta * 2.0 * jnp.pi
    ph = phi   * 2.0 * jnp.pi * nfp

    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<0:
                continue
            counter = counter+1 
            r = r + rc[counter] * jnp.cos(mm*th - nn*ph)
            drdt = drdt - mm * rc[counter] * jnp.sin(mm*th - nn*ph)
            drdp = drdp + nn * nfp * rc[counter] * jnp.sin(mm*th - nn*ph)

    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<=0:
                continue
            counter = counter+1 
            dzdt = dzdt + mm * zs[counter] * jnp.cos(mm*th - nn*ph)
            dzdp = dzdp - nn * nfp * zs[counter] * jnp.cos(mm*th - nn*ph)

    n = jnp.zeros((qpts.size, 3))
    nnorm = jnp.sqrt(r**2 * (drdt**2 + dzdt**2) + (drdp*dzdt-drdt*dzdp)**2)
    
    n = n.at[:,0].set(  r*dzdt / nnorm )
    n = n.at[:,1].set( -(drdp*dzdt-drdt*dzdp) / nnorm )
    n = n.at[:,2].set( -r*drdt / nnorm )
    
    return n


def nfactor(modes, qpts, order, surf_dofs, mpol, ntor, nfp, direction='z'):
    if direction=='z':
        return normal(modes, qpts, order, surf_dofs, mpol, ntor, nfp)[:,2]
    elif direction=='r':
        return normal(modes, qpts, order, surf_dofs, mpol, ntor, nfp)[:,0]

    

class CurveCWSFourier( JaxCurve ):
    def __init__(self, quadpoints, order, surf:SurfaceRZFourier, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = jnp.linspace(0, 1, quadpoints, endpoint=False)
        
        # Curve order. Number of Fourier harmonics for phi and theta
        self.order = order

        # Set surface
        if not surf.stellsym:
            raise ValueError('Not implemented for non-stellarator symmetric surfaces')

        if not isinstance(surf, SurfaceRZFourier):
            raise ValueError('This implementation assumes the surface to be a SurfaceRZFourier object')
        self.surf = surf
        
        # Modes are order as phic, phis, thetac, thetas
        self.modes = [np.zeros((order+1,)), np.zeros((order,)), np.zeros((order+1,)), np.zeros((order,))]

        # Define pure function
        pure = jit(lambda dofs, points: gamma_curve_on_surface(dofs, points, self.order, self.surf.x, self.surf.mpol, self.surf.ntor, self.surf.nfp))


        # Call JaxCurve constructor
        if dofs is None:
            super().__init__(
                quadpoints, 
                pure, 
                x0=np.concatenate(self.modes),
                external_dof_setter=CurveCWSFourier.set_dofs_impl,
                names=self._make_names()
                )
        else:
            super().__init__(
                quadpoints, 
                pure, 
                dofs=dofs,
                external_dof_setter=CurveCWSFourier.set_dofs_impl,
                names=self._make_names()
                )

        self.snz = jit(lambda dofs: nfactor(dofs, self.quadpoints, self.order, self.surf.x, self.surf.mpol, self.surf.ntor, self.surf.nfp, direction='z'))
        self.snr = jit(lambda dofs: nfactor(dofs, self.quadpoints, self.order, self.surf.x, self.surf.mpol, self.surf.ntor, self.surf.nfp, direction='r'))

        self.dsnz_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.snz, x)[1](v)[0])
        self.dsnr_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.snr, x)[1](v)[0])

    def zfactor(self):
        return self.snz(self.get_dofs())

    def dzfactor_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dsnz_by_dcoeff_vjp_jax(self.get_dofs(), v)})

    def rfactor(self):
        return self.snr(self.get_dofs())

    def drfactor_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dsnr_by_dcoeff_vjp_jax(self.get_dofs(), v)})

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





def gamma_curve_on_surface2(dofs, surf_dofs, qpts, order, mpol, ntor, nfp):
    # Unpack dofs
    phic = dofs[:order+1]
    phis = dofs[order+1:2*order+1]
    thetac   = dofs[2*order+1:3*order+2]
    thetas   = dofs[3*order+2:4*order+2]

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


    # Construct curve on surface
    r = jnp.zeros((qpts.size,))
    z = jnp.zeros((qpts.size,))

    nmn = ntor+1 + mpol*(2*ntor+1)
    rc = surf_dofs[:nmn]
    zs = surf_dofs[nmn:]

    th = theta * 2.0 * jnp.pi
    ph = phi   * 2.0 * jnp.pi * nfp
    
    #ph = (1.0+phi) * jnp.pi / (2.0*nfp)

    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<0:
                continue
            counter = counter+1
            r = r + rc[counter] * jnp.cos(mm*th - nn*ph)


    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<=0:
                continue
            counter = counter+1
            z = z + zs[counter] * jnp.sin(mm*th - nn*ph)
            
    gamma = jnp.zeros((qpts.size, 3))
    gamma = gamma.at[:,0].set( r * jnp.cos( phi*jnp.pi*2 ) )
    gamma = gamma.at[:,1].set( r * jnp.sin( phi*jnp.pi*2 ) )
    gamma = gamma.at[:,2].set( z                 )

    return gamma


class CurveCWSFourier2( Optimizable ):
    def __init__(self, curve2d, surf):   
        
        self.curve = curve2d
        self.surf = surf

        points = self.curve.quadpoints
        ones = jnp.ones_like(points)

        # We are not doing the same search for x0
        super().__init__(depends_on=[self.surf, self.curve2d])

        self.gamma_pure = jit(lambda gamma2d, surf_dofs, points: gamma_curve_on_surface2(gamma2d, surf_dofs, points, self.order, self.surf.mpol, self.surf.ntor, self.surf.nfp))

        # GAMMA
        self.gamma_jax = jit(lambda gamma2d, surf_dofs: self.gamma_pure(gamma2d,  surf_dofs, points))
        self.gamma_impl_jax = jit(lambda gamma2d, surf_dofs, p: self.gamma_pure(gamma2d, surf_dofs, p))

        # GAMMA DERIVATIVES
        self.dgamma_by_dcurve_jax = jit(lambda gamma2d, surf_dofs: jacfwd(self.gamma_jax, argnums=0)(gamma2d, surf_dofs))
        self.dgamma_by_dsurf_jax = jit(lambda gamma2d, surf_dofs: jacfwd(self.gamma_jax, argnums=1)(gamma2d, surf_dofs))
        self.dgamma_by_dcurve_vjp_jax = jit(lambda gamma2d, surf_dofs, v: vjp(self.gamma_jax, gamma2d, surf_dofs)[1](v)[0])
        self.dgamma_by_dsurf_vjp_jax = jit(lambda gamma2d, surf_dofs, v: vjp(self.gamma_jax, dofs, surf_dofs)[1](v)[1])

        # GAMMADASH
        self.gammadash_pure = lambda dofs, surf_dofs, q: jvp(lambda p: self.gamma_pure(dofs, surf_dofs, p), (q,), (ones,))[1]
        self.gammadash_jax = jit(lambda dofs, surf_dofs: self.gammadash_pure(dofs, surf_dofs, points))

        # GAMMADASH DERIVATIVES
        self.dgammadash_by_dcurve_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadash_jax, argnums=0)(dofs, surf_dofs))
        self.dgammadash_by_dsurf_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadash_jax, argnums=1)(dofs, surf_dofs))
        self.dgammadash_by_dcurve_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadash_jax, dofs, surf_dofs)[1](v)[0])
        self.dgammadash_by_dsurf_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadash_jax, dofs, surf_dofs)[1](v)[1])

        # GAMMADASHDASH
        self.gammadashdash_pure = lambda dofs, surf_dofs, q: jvp(lambda p: self.gammadash_pure(dofs, surf_dofs, p), (q,), (ones,))[1]
        self.gammadashdash_jax = jit(lambda dofs, surf_dofs: self.gammadashdash_pure(dofs, surf_dofs, points))

        # GAMMADASH DERIVATIVES
        self.dgammadashdash_by_dcurve_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdash_jax, argnums=0)(dofs, surf_dofs))
        self.dgammadashdash_by_dsurf_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdash_jax, argnums=1)(dofs, surf_dofs))
        self.dgammadashdash_by_dcurve_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdash_jax, dofs, surf_dofs)[1](v)[0])
        self.dgammadashdash_by_dsurf_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdash_jax, dofs, surf_dofs)[1](v)[1])

        # GAMMADASHDASHDASH
        self.gammadashdashdash_pure = lambda dofs, surf_dofs, q: jvp(lambda p: self.gammadashdash_pure(dofs, surf_dofs, p), (q,), (ones,))[1]
        self.gammadashdashdash_jax = jit(lambda dofs, surf_dofs: self.gammadashdashdash_pure(dofs, surf_dofs, points))

        # GAMMADASHDASH DERIVATIVES
        self.dgammadashdashdash_by_dcurve_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdashdash_jax, argnums=0)(dofs, surf_dofs))
        self.dgammadashdashdash_by_dsurf_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdashdash_jax, argnums=1)(dofs, surf_dofs))
        self.dgammadashdashdash_by_dcurve_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdashdash_jax, dofs, surf_dofs)[1](v)[0])
        self.dgammadashdashdash_by_dsurf_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdashdash_jax, dofs, surf_dofs)[1](v)[1])

        # CURVATURE
        self.dkappa_by_dcurve_vjp_jax = jit(lambda dofs, curve_dofs, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), dofs, curve_dofs)[1](v)[0])
        self.dkappa_by_dsurf_vjp_jax = jit(lambda dofs, curve_dofs, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), dofs, curve_dofs)[1](v)[1])

        # TORSION
        self.dtorsion_by_dcurve_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v)[0])
        self.dtorsion_by_dsurf_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v)[1])

    def num_dofs(self):
        return 2*(self.order+1) + 2*self.order #+ self.ndof_surf
    
    def get_dofs(self):
        return np.concatenate(self.modes)
        #return np.append(np.concatenate(self.modes), self.surf.get_dofs())

    def set_dofs_impl(self, dofs):
        c = 0
        self.modes[0] = dofs[c:c+self.order+1]

        c = c + self.order+1
        self.modes[1] = dofs[c:c+self.order]

        c = c + self.order
        self.modes[2] = dofs[c:c+self.order+1]

        c = c + self.order+1
        self.modes[3] = dofs[c:c+self.order]

        # c = c + self.order
        # self.surf.set_dofs( dofs[self.ndof_curve:] )

    def _make_names(self):
        dofs_name = []
        for mode in ['phic', 'phis', 'thetac', 'thetas']:
            for ii in range(self.order+1):
                if mode=='phis' and ii==0:
                    continue

                if mode=='thetas' and ii==0:
                    continue

                dofs_name.append(f'{mode}({ii})')

        return dofs_name #+ self.surf.full_dof_names


    # GAMMA AND DERIVATIVES
    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve :math:`\Gamma`.
        """
        gamma[:, :] = self.gamma_impl_jax(self.get_dofs(), self.surf.get_dofs(), quadpoints)


    # dgamma dcoef
    def dgamma_by_dcoeff_impl(self, d):
        d[:, :, :] = self.dgamma_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs()) + self.dgamma_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())
        #self.dgamma_by_dcurve_impl(d)
        #self.dgamma_by_dsurf_impl(d)
    
    def dgamma_by_dcurve_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgamma_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs())

    def dgamma_by_dsurf_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgamma_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())

    # dgamma dcoef vjp
    def dgamma_by_dcoeff_vjp_impl(self, v):   
        return self.dgamma_by_dcurve_vjp_impl(v) + self.dgamma_by_dsurf_vjp_impl(v)
    
    def dgamma_by_dcurve_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgamma_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    
    def dgamma_by_dsurf_vjp(self, v):
        return Derivative({self: self.dgamma_by_dsurf_vjp_impl(v)})

    def dgamma_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma}{\partial \mathbf c} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgamma_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    

    #GAMMADASH AND DERIVATIVES
    def gammadash_impl(self, gammadash):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        gammadash[:, :] = self.gammadash_jax(self.get_dofs(), self.surf.get_dofs())

    # dgammadash dcoef
    def dgammadash_by_dcoeff_impl(self, d):
        self.dgammadash_by_dcurve_impl(d)

    def dgammadash_by_dcurve_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgammadash_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs())

    def dgammadash_by_dsurf_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgammadash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())

    # dgammadash vjp
    def dgammadash_by_dcoeff_vjp_impl(self, v):
        return self.dgammadash_by_dcurve_vjp_impl(v)

    def dgammadash_by_dcurve_vjp_impl(self, v):
        r"""
        This function returns 

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgammadash_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    def dgammadash_by_dsurf_vjp(self, v):
        return Derivative({self: self.dgammadash_by_dsurf_vjp_impl(v)})

    def dgammadash_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns 

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgammadash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    

 
    #GAMMADASHDASH AND DERIVATIVES
    def gammadashdash_impl(self, d):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :] = self.gammadashdash_jax(self.get_dofs(), self.surf.get_dofs())

    # dgammadashdash dcoef
    def dgammadashdash_by_dcoeff_impl(self, d):
        self.dgammadashdash_by_dcurve_impl(d)

    def dgammadashdash_by_dcurve_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgammadashdash_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs())

    def dgammadashdash_by_dsurf_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgammadashdash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())

    # dgammadashdash vjp
    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
        return self.dgammadashdash_by_dcurve_vjp_impl(v)

    def dgammadashdash_by_dcurve_vjp_impl(self, v):
        r"""
        This function returns 

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgammadashdash_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    def dgammadashdash_by_dsurf_vjp(self, v):
        return Derivative({self: self.dgammadashdash_by_dsurf_vjp_impl(v)})

    def dgammadashdash_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns 

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgammadashdash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    

    #GAMMADASHDASHDASH AND DERIVATIVES
    def gammadashdashdash_impl(self, d):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :] = self.gammadashdashdash_jax(self.get_dofs(), self.surf.get_dofs())

    # dgammadashdashdash dcoef
    def dgammadashdashdash_by_dcoeff_impl(self, d):
        self.dgammadashdashdash_by_dcurve_impl(d)

    def dgammadashdashdash_by_dcurve_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgammadashdashdash_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs())

    def dgammadashdashdash_by_dsurf_impl(self, d):
        r"""
        This function returns 

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        d[:, :, :] = self.dgammadashdashdash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())

    # dgammadashdash vjp
    def dgammadashdashdash_by_dcoeff_vjp_impl(self, v):
        return self.dgammadashdashdash_by_dcurve_vjp_impl(v)

    def dgammadashdashdash_by_dcurve_vjp_impl(self, v):
        r"""
        This function returns 

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgammadashdashdash_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    def dgammadashdashdash_by_dsurf_vjp(self, v):
        return Derivative({self: self.dgammadashdashdash_by_dsurf_vjp_impl(v)})

    def dgammadashdashdash_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns 

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        return self.dgammadashdashdash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    

    # CURVATURE AND DERIVATIVES
    def dkappa_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

        """
        return self.dkappa_by_dcurve_vjp(v)
    
    def dkappa_by_dcurve_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

        """
        return Derivative({self: self.dkappa_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})
    
    def dkappa_by_dsurf_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

        """
        return Derivative({self: self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})

    # TORSION AND DERIVATIVES
    def dtorsion_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dtorsion_by_dcurve_vjp(self.get_dofs(), self.surf.get_dofs(), v)})
    
    def dtorsion_by_dcurve_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \tau}{\partial \mathbf{c}} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

        """
        return Derivative({self: self.dtorsion_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})
    
    def dtorsion_by_dsurf_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \tau}{\partial \mathbf{c}} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

        """
        return Derivative({self: self.dtorsion_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})
