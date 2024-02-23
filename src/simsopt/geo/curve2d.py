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


from .plotting import fix_matplotlib_3d

__all__ = ['Curve2D', 'CurveCWSFourier']

def gamma_2d(modes, qpts, order):
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
            
    gamma = jnp.zeros((qpts.size, 3))
    gamma = gamma.at[:,0].set( phi   )
    gamma = gamma.at[:,1].set( theta )
    gamma = gamma.at[:,2].set( 0.0   )

    return gamma

def zfactor(modes, qpts, order, surf_dofs, mpol, ntor, nfp, k=10):
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

    
    return -r*drdt / jnp.sqrt(r**2 * (drdt**2 + dzdt**2) + (drdp*dzdt-drdt*dzdp)**2)

class Curve2D( JaxCurve ):
    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = jnp.linspace(0, 1, quadpoints, endpoint=False)
        
        # Curve order. Number of Fourier harmonics for phi and theta
        self.order = order
        
        # Modes are order as phic, phis, thetac, thetas
        self.modes = [np.zeros((order+1,)), np.zeros((order,)), np.zeros((order+1,)), np.zeros((order,))]

        # Define pure function
        pure = jit(lambda dofs, points: gamma_2d(dofs, points, self.order))

        self.dgamma_by_dpoint_vjp_jax = jit(lambda x, v: vjp(lambda d: pure(x, d), self.quadpoints)[1](v)[0])

        # Call JaxCurve constructor
        if dofs is None:
            super().__init__(
                quadpoints, 
                pure, 
                x0=np.concatenate(self.modes),
                external_dof_setter=Curve2D.set_dofs_impl,
                names=self._make_names()
                )
        else:
            super().__init__(
                quadpoints, 
                pure, 
                dofs=dofs,
                external_dof_setter=Curve2D.set_dofs_impl,
                names=self._make_names()
            )

    def dgamma_by_dpoint_vjp(self,v):
        return Derivative({self: self.dgamma_by_dpoint_vjp_jax(self.get_dofs(), v)})

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




def gamma_curve_on_surface(gamma2d, surf_dofs, qpts, mpol, ntor, nfp):
    phi = gamma2d[:,0]
    theta = gamma2d[:,1]

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

class CurveCWSFourier( Optimizable ):
    def __init__(self, curve2d, surf):   
        
        self.curve = curve2d
        self.surf = surf

        points = self.curve.quadpoints
        self.quadpoints = self.curve.quadpoints
        ones = jnp.ones_like(points)

        # We are not doing the same search for x0
        super().__init__(depends_on=[self.surf, self.curve])

        self.gamma_pure = jit(lambda gamma2d, surf_dofs, points: gamma_curve_on_surface(gamma2d, surf_dofs, points, self.surf.mpol, self.surf.ntor, self.surf.nfp))

        # GAMMA
        self.gamma_jax = jit(lambda gamma2d, surf_dofs: self.gamma_pure(gamma2d,  surf_dofs, points))
        self.gamma_impl_jax = jit(lambda gamma2d, surf_dofs, p: self.gamma_pure(gamma2d, surf_dofs, p))

        # GAMMA DERIVATIVES
        self.dgamma_by_dcurve_jax = jit(lambda gamma2d, surf_dofs: jacfwd(self.gamma_jax, argnums=0)(gamma2d, surf_dofs))
        self.dgamma_by_dsurf_jax = jit(lambda gamma2d, surf_dofs: jacfwd(self.gamma_jax, argnums=1)(gamma2d, surf_dofs))
        self.dgamma_by_dcurve_vjp_jax = jit(lambda gamma2d, surf_dofs, v: vjp(self.gamma_jax, gamma2d, surf_dofs)[1](v)[0])
        self.dgamma_by_dsurf_vjp_jax = jit(lambda gamma2d, surf_dofs, v: vjp(self.gamma_jax, gamma2d, surf_dofs)[1](v)[1])


        self.gammadash_pure = lambda g2, sdofs, q: jvp(lambda p: self.gamma_pure(g2, sdofs, p), (q,), (self.curve.gammadash(),))[1]
        self.gammadash_jax = jit(lambda g2, sdofs: self.gammadash_pure(g2, sdofs, points))

        # GAMMADASH



        # self.gammadash_pure = lambda dofs, surf_dofs, q: jvp(lambda p: self.gamma_pure(dofs, surf_dofs, p), (q,), (ones,))[1]
        #self.gammadash_jax = jit(lambda dofs, surf_dofs: self.gammadash_pure(dofs, surf_dofs, points))

        # # GAMMADASH DERIVATIVES
        # self.dgammadash_by_dcurve_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadash_jax, argnums=0)(dofs, surf_dofs))
        # self.dgammadash_by_dsurf_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadash_jax, argnums=1)(dofs, surf_dofs))
        # self.dgammadash_by_dcurve_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadash_jax, dofs, surf_dofs)[1](v)[0])
        # self.dgammadash_by_dsurf_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadash_jax, dofs, surf_dofs)[1](v)[1])

        # # GAMMADASHDASH
        # self.gammadashdash_pure = lambda dofs, surf_dofs, q: jvp(lambda p: self.gammadash_pure(dofs, surf_dofs, p), (q,), (ones,))[1]
        # self.gammadashdash_jax = jit(lambda dofs, surf_dofs: self.gammadashdash_pure(dofs, surf_dofs, points))

        # # GAMMADASH DERIVATIVES
        # self.dgammadashdash_by_dcurve_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdash_jax, argnums=0)(dofs, surf_dofs))
        # self.dgammadashdash_by_dsurf_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdash_jax, argnums=1)(dofs, surf_dofs))
        # self.dgammadashdash_by_dcurve_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdash_jax, dofs, surf_dofs)[1](v)[0])
        # self.dgammadashdash_by_dsurf_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdash_jax, dofs, surf_dofs)[1](v)[1])

        # # GAMMADASHDASHDASH
        # self.gammadashdashdash_pure = lambda dofs, surf_dofs, q: jvp(lambda p: self.gammadashdash_pure(dofs, surf_dofs, p), (q,), (ones,))[1]
        # self.gammadashdashdash_jax = jit(lambda dofs, surf_dofs: self.gammadashdashdash_pure(dofs, surf_dofs, points))

        # # GAMMADASHDASH DERIVATIVES
        # self.dgammadashdashdash_by_dcurve_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdashdash_jax, argnums=0)(dofs, surf_dofs))
        # self.dgammadashdashdash_by_dsurf_jax = jit(lambda dofs, surf_dofs: jacfwd(self.gammadashdashdash_jax, argnums=1)(dofs, surf_dofs))
        # self.dgammadashdashdash_by_dcurve_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdashdash_jax, dofs, surf_dofs)[1](v)[0])
        # self.dgammadashdashdash_by_dsurf_vjp_jax = jit(lambda dofs, surf_dofs, v: vjp(self.gammadashdashdash_jax, dofs, surf_dofs)[1](v)[1])

        # # CURVATURE
        # self.dkappa_by_dcurve_vjp_jax = jit(lambda dofs, curve_dofs, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), dofs, curve_dofs)[1](v)[0])
        # self.dkappa_by_dsurf_vjp_jax = jit(lambda dofs, curve_dofs, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), dofs, curve_dofs)[1](v)[1])

        # # TORSION
        # self.dtorsion_by_dcurve_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v)[0])
        # self.dtorsion_by_dsurf_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v)[1])


    # def gamma_impl(self, gamma, quadpoints):
    #     r"""
    #     This function returns the x,y,z coordinates of the curve :math:`\Gamma`.
    #     """
    #     gamma[:, :] = self.gamma_impl_jax(self.curve.gamma(), self.surf.get_dofs(), quadpoints)


    # GAMMA AND DERIVATIVES
    def gamma(self):
        gamma2d = self.curve.gamma()
        surf_dofs = self.surf.get_dofs()
        return self.gamma_jax(gamma2d, surf_dofs)

    def gammadash(self):
        gamma2d = self.curve.gamma()
        surf_dofs = self.surf.get_dofs()
        grad0 = self.dgamma_by_dcurve_jax(gamma2d, surf_dofs)

        #return self.curve.dgamma_by_dpoint_vjp(grad0)
        return self.gammadash_jax(gamma2d, surf_dofs)
        

    # def gammadash(self):
    #     gamma2d = self.curve.gamma()
    #     surf_dofs = self.surf.get_dofs()
    #     return self.gammada(gamma2d, surf_dofs)
        


    # dgamma dcoef vjp
    def dgamma_by_dcoeff_vjp_impl(self, v):   
        g2 = self.curve.gamma()
        grad0 = self.dgamma_by_dcurve_jax(g2, self.surf.get_dofs())
        grad1 = Derivative({self.surface: self.dgamma_by_dsurf_jax(g2, self.surf.get_dofs())})
        return self.curve.dgamma_by_dcoeff_vjp(grad0) + grad1
    

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
    
    # def dgamma_by_dcurve_vjp_impl(self, v):
    #     r"""
    #     This function returns the vector Jacobian product

    #     .. math::
    #         v^T  \frac{\partial \Gamma}{\partial \mathbf c} 

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgamma_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    
    # def dgamma_by_dsurf_vjp(self, v):
    #     return Derivative({self: self.dgamma_by_dsurf_vjp_impl(v)})

    # def dgamma_by_dsurf_vjp_impl(self, v):
    #     r"""
    #     This function returns the vector Jacobian product

    #     .. math::
    #         v^T  \frac{\partial \Gamma}{\partial \mathbf c} 

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgamma_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    

    # #GAMMADASH AND DERIVATIVES
    # def gammadash_impl(self, gammadash):
    #     r"""
    #     This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     gammadash[:, :] = self.gammadash_jax(self.get_dofs(), self.surf.get_dofs())

    # # dgammadash dcoef
    # def dgammadash_by_dcoeff_impl(self, d):
    #     self.dgammadash_by_dcurve_impl(d)

    # def dgammadash_by_dcurve_impl(self, d):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :, :] = self.dgammadash_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs())

    # def dgammadash_by_dsurf_impl(self, d):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :, :] = self.dgammadash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())

    # # dgammadash vjp
    # def dgammadash_by_dcoeff_vjp_impl(self, v):
    #     return self.dgammadash_by_dcurve_vjp_impl(v)

    # def dgammadash_by_dcurve_vjp_impl(self, v):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgammadash_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    # def dgammadash_by_dsurf_vjp(self, v):
    #     return Derivative({self: self.dgammadash_by_dsurf_vjp_impl(v)})

    # def dgammadash_by_dsurf_vjp_impl(self, v):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgammadash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    

 
    # #GAMMADASHDASH AND DERIVATIVES
    # def gammadashdash_impl(self, d):
    #     r"""
    #     This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :] = self.gammadashdash_jax(self.get_dofs(), self.surf.get_dofs())

    # # dgammadashdash dcoef
    # def dgammadashdash_by_dcoeff_impl(self, d):
    #     self.dgammadashdash_by_dcurve_impl(d)

    # def dgammadashdash_by_dcurve_impl(self, d):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :, :] = self.dgammadashdash_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs())

    # def dgammadashdash_by_dsurf_impl(self, d):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :, :] = self.dgammadashdash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())

    # # dgammadashdash vjp
    # def dgammadashdash_by_dcoeff_vjp_impl(self, v):
    #     return self.dgammadashdash_by_dcurve_vjp_impl(v)

    # def dgammadashdash_by_dcurve_vjp_impl(self, v):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgammadashdash_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    # def dgammadashdash_by_dsurf_vjp(self, v):
    #     return Derivative({self: self.dgammadashdash_by_dsurf_vjp_impl(v)})

    # def dgammadashdash_by_dsurf_vjp_impl(self, v):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgammadashdash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    

    # #GAMMADASHDASHDASH AND DERIVATIVES
    # def gammadashdashdash_impl(self, d):
    #     r"""
    #     This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :] = self.gammadashdashdash_jax(self.get_dofs(), self.surf.get_dofs())

    # # dgammadashdashdash dcoef
    # def dgammadashdashdash_by_dcoeff_impl(self, d):
    #     self.dgammadashdashdash_by_dcurve_impl(d)

    # def dgammadashdashdash_by_dcurve_impl(self, d):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :, :] = self.dgammadashdashdash_by_dcurve_jax(self.get_dofs(), self.surf.get_dofs())

    # def dgammadashdashdash_by_dsurf_impl(self, d):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     d[:, :, :] = self.dgammadashdashdash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())

    # # dgammadashdash vjp
    # def dgammadashdashdash_by_dcoeff_vjp_impl(self, v):
    #     return self.dgammadashdashdash_by_dcurve_vjp_impl(v)

    # def dgammadashdashdash_by_dcurve_vjp_impl(self, v):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgammadashdashdash_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)

    # def dgammadashdashdash_by_dsurf_vjp(self, v):
    #     return Derivative({self: self.dgammadashdashdash_by_dsurf_vjp_impl(v)})

    # def dgammadashdashdash_by_dsurf_vjp_impl(self, v):
    #     r"""
    #     This function returns 

    #     .. math::
    #         \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
    #     of the curve.
    #     """
    #     return self.dgammadashdashdash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
    

    # # CURVATURE AND DERIVATIVES
    # def dkappa_by_dcoeff_vjp(self, v):
    #     r"""
    #     This function returns the vector Jacobian product

    #     .. math::
    #         v^T \frac{\partial \kappa}{\partial \mathbf{c}}

    #     where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

    #     """
    #     return self.dkappa_by_dcurve_vjp(v)
    
    # def dkappa_by_dcurve_vjp(self, v):
    #     r"""
    #     This function returns the vector Jacobian product

    #     .. math::
    #         v^T \frac{\partial \kappa}{\partial \mathbf{c}}

    #     where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

    #     """
    #     return Derivative({self: self.dkappa_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})
    
    # def dkappa_by_dsurf_vjp(self, v):
    #     r"""
    #     This function returns the vector Jacobian product

    #     .. math::
    #         v^T \frac{\partial \kappa}{\partial \mathbf{c}}

    #     where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

    #     """
    #     return Derivative({self: self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})

    # # TORSION AND DERIVATIVES
    # def dtorsion_by_dcoeff_vjp(self, v):
    #     return Derivative({self: self.dtorsion_by_dcurve_vjp(self.get_dofs(), self.surf.get_dofs(), v)})
    
    # def dtorsion_by_dcurve_vjp(self, v):
    #     r"""
    #     This function returns the vector Jacobian product

    #     .. math::
    #         v^T \frac{\partial \tau}{\partial \mathbf{c}} 

    #     where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

    #     """
    #     return Derivative({self: self.dtorsion_by_dcurve_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})
    
    # def dtorsion_by_dsurf_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \tau}{\partial \mathbf{c}} 

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

        """
        return Derivative({self: self.dtorsion_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)})
