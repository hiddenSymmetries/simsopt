import numpy as np
from jax import grad, hessian, jacfwd, jacrev
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .curve import Curve, CurveCWSFourier
from .curveobjectives import ArclengthVariation

from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative

from scipy.linalg import lu
from scipy.optimize import minimize

__all__ = ['PortSize', 'ProjectedEnclosedArea', 'ProjectedCurveCurveDistance', 'ProjectedCurveConvexity', 'DirectedFacingPort', 'CurveInPortPenalty']


class PortSize(Optimizable):
    def __init__(self, port, curves, curve_port_distance_threshold, direction, solver='jax', port_coil_distance_weight=1e2, port_arc_penalty_weight=1e-2, port_forward_facing_weight=1e2):
        if not isinstance(port, CurveCWSFourier):
            raise ValueError("Port should be a CurveCWSFourier instance")
        if not isinstance(curves, list):
            raise ValueError("Curves should be a list")
        if len(curves)<1:
            raise ValueError("Should have at least one curve")
        if any([not isinstance(c, Curve) for c in curves]):
            raise ValueError("Curves elements should be instances of curves")
        if not isinstance(curve_port_distance_threshold, (int, float)):
            raise ValueError("Threshold value for curve to port distance should be an integer or a float")
        if not ((direction=='radial') or (direction=='vertical')):
            raise ValueError("Only radial and vertical directions are implemented")
        if not ((solver=='jax') or (solver=='explicit')):
            raise ValueError("Solver should be either 'jax' or 'explicit'")
        
        self.port = port    
        self.curves = curves
        self.solver = solver
        self.curve_port_distance_threshold = curve_port_distance_threshold
        self.direction = direction

        self.port_arc_penalty_weight = port_arc_penalty_weight
        self.port_coil_distance_weight = port_coil_distance_weight
        self.port_forward_facing_weight = port_forward_facing_weight

        # Define objective
        if self.direction == 'radial':
            projection='zphi'
        elif self.direction == 'vertical':
            projection='xy'

        self.Jxyarea = ProjectedEnclosedArea( self.port, projection=projection )
        self.Jccxydist = ProjectedCurveCurveDistance( self.curves, self.port, self.curve_port_distance_threshold, projection=projection )
        self.Jarc = ArclengthVariation( self.port )
        self.Jufp = DirectedFacingPort( self.port, projection=projection)

        # self.Jport = -1*self.Jxyarea \
        #     + self.port_coil_distance_weight * self.Jccxydist \
        #     + self.port_forward_facing_weight * self.Jufp 
            #+ self.port_arc_penalty_weight * self.Jarc \

        # Initialize parent
        # Does NOT depend on port! Port is specified by this inner
        # optimization.
        super().__init__(depends_on=self.curves)

        # Some cache boolean
        self.need_to_run_code = True

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True
         
    def objective(self, dofs, hessian_bool):
        self.port.x = dofs # We only optimize the port dofs

        if hessian_bool: # Don't include the arclength penalty here
            J = -1*self.Jxyarea.J() \
                + self.port_coil_distance_weight * self.Jccxydist.J() \
                + self.port_forward_facing_weight * self.Jufp.J() 

            dJ = -1*self.Jxyarea.dJ(partials=True).data[self.port] \
                + self.port_coil_distance_weight * self.Jccxydist.dJ(partials=True).data[self.port] \
                + self.port_forward_facing_weight * self.Jufp.dJ(partials=True).data[self.port]
            
            hess = -1*self.Jxyarea.ddJ_ddport() \
                + self.port_coil_distance_weight * self.Jccxydist.ddJ_ddport() \
                + self.port_forward_facing_weight * self.Jufp.ddJ_ddport() 
            
            return J, dJ, hess
        
        else:
            J = -1*self.Jxyarea.J() \
                + self.port_coil_distance_weight * self.Jccxydist.J() \
                + self.port_forward_facing_weight * self.Jufp.J() 
                #+ self.port_arc_penalty_weight * self.Jarc.J() \

            dJ = -1*self.Jxyarea.dJ(partials=True).data[self.port] \
                + self.port_coil_distance_weight * self.Jccxydist.dJ(partials=True).data[self.port] \
                + self.port_forward_facing_weight * self.Jufp.dJ(partials=True).data[self.port]
                #+ self.port_arc_penalty_weight * self.Jarc.dJ(partials=True).data[self.port] \


            return J, dJ

    def explicit_solve(self, verbose=True):
        if not self.need_to_run_code:
            return
        
        # Pre-conditioning. Move port s.t. constraints port-coil distance is satisfied.
        # This is necessary, otherwise results of optimization below will not satisfy constraints
        counter = 0
        while self.Jccxydist.J()>0:
            print("Port intersect coils... reducing toroidal extend of port")
            for oo in range(1, self.port.order+1):
                for par in ['c','s']:
                    key = f'phi{par}({oo})'
                    self.port.set(key, self.port.get(key)*(0.9**oo))
                    key = f'theta{par}({oo})'
                    self.port.set(key, self.port.get(key)*(0.9**oo))

            if counter==20:
                raise ValueError('Initial port does not satisfy constraints')


        # First, L-BFGS solve
        fun = lambda x: self.objective(x, hessian_bool=False)
        dofs = self.port.x
        maxiter = 1000
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-08})

        res_bfgs = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res, "success": res.success
        }

        if verbose:
            print(f"L-BFGS-B solve - {res_bfgs['success']}  iter={res_bfgs['iter']}, ||grad||_inf = {np.linalg.norm(res_bfgs['gradient'], ord=np.inf):.3e}", flush=True)

        # Then, Newton step
        fun = lambda x: self.objective(x, hessian_bool=True)
        x = res.x
        maxiter = 10
        stab = 0.
        tol = 1e-12
        J, dJ, d2J = fun(x)
        norm = np.linalg.norm(d2J)
        i = 0
        while i < maxiter and norm > tol:
            d2J += stab*np.identity(d2J.shape[0])
            dx = np.linalg.solve(d2J, dJ)
            if norm < 1e-9:
                dx += np.linalg.solve(d2J, dJ - d2J@dx)
            x = x - dx
            J, dJ, d2J = fun(x)
            norm = np.linalg.norm(dJ)
            i = i+1

        P, L, U = lu(d2J)
        res_newton = {
            "residual": J, "jacobian": dJ, "hessian": d2J, "iter": i, "success": norm <= tol, "G": None, 
            "PLU" : (P, L, U), "vjp": None
        }

        if verbose:
            print(f"NEWTON solve - {res_newton['success']}  iter={res_newton['iter']}, ||grad||_inf = {np.linalg.norm(res_newton['jacobian'], ord=np.inf):.3e}", flush=True)

        self.need_to_run_code = False
        
        return res_bfgs, res_newton
    
    def J(self):
        self.explicit_solve()
        return self.Jxyarea.J()
    
    @derivative_dec
    def dJ(self):
        self.explicit_solve()
        U1 = self.Jxyarea.ddJ_ddport()
        U2 = self.port_coil_distance_weight * self.Jccxydist.ddJ_ddport()
        U3 = self.port_forward_facing_weight * self.Jufp.ddJ_ddport()
        U = -U1 + U2 + U3

        x = np.linalg.solve( U, self.Jxyarea.dJ() )

        d = Derivative()
        for c in self.curves:
            V1 = self.Jxyarea.ddJ_dportdcoil(c)
            V2 = self.port_coil_distance_weight * self.Jccxydist.ddJ_dportdcoil(c)
            V3 = self.port_forward_facing_weight * self.Jufp.ddJ_dportdcoil(c)
            V = -V1 + V2 + V3

            dAdxc = np.matmul(V, x)
            d += Derivative({c: dAdxc})

        return d


    def plot(self):
        _, ax = plt.subplots()

        # Plot surface
        if self.direction=='radial':
            x0 = np.mean(self.port.gamma(),axis=0)
            gproj = project(self.port.surf.gamma().reshape((-1,3)), x0)
            gport = project(self.port.gamma(), x0)
        elif self.direction=='vertical':
            # Rerorganizing indices to be compatible with the projection in the case of radial access. Not very elegant, but I don't won't to reimplement the project function to order dimensions differently
            gproj = self.port.surf.gamma()[:,:,[2,0,1]]
            gport = self.port.gamma()[:,[2,0,1]]
        ind = np.argsort(gproj[:,0])
        ax.scatter(gproj[ind,1],gproj[ind,2],c=gproj[ind,0],s=10)

        # Plot port
        ax.fill(gport[:,1], gport[:,2], color='r', alpha=.8)

        # Plot coils
        for c in self.curves:
            if self.direction=='radial':
                g = project(c.gamma(), x0)
            elif self.direction=='vertical':
                g = c.gamma()[:,[2,0,1]]
            zcurves = g[:,0]

            ind = np.where( zcurves>0 )[0]
            g = g[ind,:]

            ax.scatter(g[:,1], g[:,2], color='k', marker='o', s=15)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')


def PolyArea(x,y):
    # Showlace formula, using https://stackoverflow.com/a/30408825
    shifty = jnp.roll(y,1)
    shiftx = jnp.roll(x,1)
    return 0.5*jnp.abs(jnp.dot(x,shifty)-jnp.dot(y,shiftx))

def xy_enclosed_area( gamma ):
    """
    Returns enclosed area in XY plane
    """

    x = gamma[:,0]
    y = gamma[:,1]
    return PolyArea(x,y)

def project(x, gamma):
    centroid = jnp.mean(gamma.reshape((-1,3)), axis=0)
    phic = jnp.arctan2(centroid[1], centroid[0])
    unit_normal = jnp.array([jnp.cos(phic), jnp.sin(phic), jnp.zeros(phic.shape)]) #er
    unit_tangent = jnp.array([-jnp.sin(phic), jnp.cos(phic), jnp.zeros(phic.shape)]) #ephi
    unit_z = jnp.array([jnp.zeros(phic.shape),jnp.zeros(phic.shape),jnp.ones(phic.shape)]) #ez

    #                 r           phi         z
    M = jnp.array([unit_normal,unit_tangent,unit_z]).transpose()
    invM = jnp.linalg.inv(M)
    
    return jnp.einsum('ij,...j->...i',invM,x-centroid)  # return phi, r, z coords

def zphi_enclosed_area(gamma): 
    """
    Returns enclosed area in Z-phi plane
    """
    local_project = lambda x: project(x, gamma)

    gcyl = local_project(gamma)

    x = gcyl[:,1]
    y = gcyl[:,2]

    return PolyArea(x,y)

class ProjectedEnclosedArea( Optimizable ):
    def __init__(self, curve, projection='xy'):
        self.curve = curve
        self.projection = projection

        if self.projection=='xy':
            self.J_jax = jit(lambda gamma: xy_enclosed_area(gamma))
        elif self.projection=='zphi':
            self.J_jax = jit(lambda gamma: zphi_enclosed_area(gamma))
            
        self.thisgrad0 = jit(lambda gamma: grad(self.J_jax, argnums=0)(gamma))
        self.hessian = jit(lambda gamma: hessian(self.J_jax, argnums=0)(gamma))

        super().__init__(depends_on=[curve])

    def J(self):
        gamma = self.curve.gamma()
        return self.J_jax(gamma)
    
    @derivative_dec
    def dJ(self):
        gamma = self.curve.gamma()
        grad0 = self.thisgrad0(gamma)

        dcurve = self.curve.dgamma_by_dcoeff_vjp(grad0)
        return dcurve
    
    def ddJ_ddport(self):
        gamma = self.curve.gamma()
        hess = self.hessian(gamma) # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        dgdx = self.curve.dgamma_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs

        grad0 = self.thisgrad0(gamma) # this is dJ/dgamma, size npts x 3
        curve_hessian = self.curve.gamma_hessian() # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs

        return np.einsum('ijkl,ijm,kln->mn', hess, dgdx, dgdx) + np.einsum('ij,ijkl->kl',grad0,curve_hessian) # this should be size ndofs x ndofs
    
    def ddJ_dportdcoil(self, curve=None):
        return 0 #Does not depend on coils



def upward_facing_pure( nznorm ):
    return jnp.sum(jnp.maximum(-nznorm, 0)**2)

class DirectedFacingPort(Optimizable):
    def __init__(self, curve, projection='xy'):
        self.curve = curve
        self.projection = projection

        self.J_jax = jit(lambda nz: upward_facing_pure(nz))
        self.thisgrad = jit(lambda nz: grad(self.J_jax)(nz))
        self.hessian = jit(lambda nz: hessian(self.J_jax)(nz) )

        super().__init__(depends_on=[curve])

    def J(self):
        if self.projection=='xy':
            n=self.curve.zfactor()
        elif self.projection=='zphi':
            n=self.curve.rfactor()
        return self.J_jax(n)
    
    @derivative_dec
    def dJ(self):
        if self.projection=='xy':
            f = self.curve.dzfactor_by_dcoeff_vjp
            n = self.curve.zfactor()
        elif self.projection=='zphi':
            f = self.curve.drfactor_by_dcoeff_vjp
            n = self.curve.rfactor()

        return f(self.thisgrad(n))
    
    def ddJ_ddport(self):
        if self.projection=='xy':
            dgdx = self.curve.dzfactor_by_dcoeff()
            curve_hessian = self.curve.zfactor_hessian()
            n = self.curve.zfactor()
        elif self.projection=='zphi':
            dgdx = self.curve.drfactor_by_dcoeff()
            curve_hessian = self.curve.rfactor_hessian()
            n = self.curve.rfactor()
            
        hess = self.hessian(n) # this is d^2J/dgamma_i dgamma_j. Size npts x npts 
        grad0 = self.thisgrad(n) # this is dJ/dgamma, size npts 

        return np.einsum('ij,il,jm->lm', hess, dgdx, dgdx) + np.einsum('i,ilm->lm',grad0,curve_hessian) # this should be size ndofs x ndofs

    def ddJ_dportdcoil(self, curve=None):
        return 0 # does not depend on coils



def winding_number_2d_pure(g2, g2dash, pts):
    x = pts[:,None,0]-g2[None,:,0]
    y = pts[:,None,1]-g2[None,:,1]
    rsquared = x**2 + y**2

    # icurve, iport
    integrand = x/rsquared * g2dash[None,:,1] - y/rsquared * g2dash[None,:,0]
    return (np.mean(integrand, axis=1) / (2*np.pi))

def winding_number_2d_squared(g2, g2dash, pts):
    x = pts[:,None,0]-g2[None,:,0]
    y = pts[:,None,1]-g2[None,:,1]
    rsquared = x**2 + y**2

    # icurve, iport
    integrand = x/rsquared * g2dash[None,:,1] - y/rsquared * g2dash[None,:,0]
    return (jnp.mean(integrand, axis=1) / (2*jnp.pi))**2

def curve_inside_port_penalty_pure(gamma_port, gammadash_port, projection, gamma_curve, gammadash_curve, threshold=0.1):
    # Project
    if projection=='zphi':
        local_project = lambda x: project(x, gamma_port)
    elif projection=='xy':
        local_project = lambda x: x[:,[2,0,1]]

    gport_2d = local_project(gamma_port)
    gportdash_2d = local_project(gammadash_port)
    gamma_curve_2d = local_project(gamma_curve)
    gammadash_curve_2d = local_project(gammadash_curve)
    gammadash_curve_2d = gammadash_curve_2d.at[:,0].set(1E-14)
    dlcurve = jnp.linalg.norm(gammadash_curve_2d, axis=1) 

    # For each point of the curve, evaluate the winding number weighted by wether they are behind or in front of the port
    weights = jnp.maximum(gamma_curve_2d[:,0], 0)**2
    winding_numbers = winding_number_2d_squared(
        gport_2d[:,1:3], gportdash_2d[:,1:3], gamma_curve_2d[:,1:3]
    )

    # Return integral along curve of winding numbers, with some threshold
    return jnp.sum( dlcurve * weights * jnp.maximum(winding_numbers - threshold, 0)**2 )

def count_inside_points(gamma_port, gammadash_port, projection, gamma_curve, threshold=0.1):
    # Project
    if projection=='zphi':
        local_project = lambda x: project(x, gamma_port)
    elif projection=='xy':
        local_project = lambda x: x[:,[2,0,1]]

    gport_2d = local_project(gamma_port)
    gportdash_2d = local_project(gammadash_port)
    gamma_curve_2d = local_project(gamma_curve)

    # For each point of the curve, evaluate the winding number weighted by wether they are behind or in front of the port
    winding_numbers = winding_number_2d_pure(
        gport_2d[:,1:3], gportdash_2d[:,1:3], gamma_curve_2d[:,1:3]
    )

    # Find closest point, evaluate if curve is in front or behind port
    dists = np.linalg.norm(gport_2d[:,None,:]-gamma_curve_2d[None,:,:], axis=2)
    min_dists_index = np.argmin(dists, axis=0)
    test = gamma_curve_2d[:,0]>gport_2d[min_dists_index,0]


    # Return integral along curve of winding numbers, with some threshold
    return np.sum( [((np.abs(w)>threshold) and (t)) for w, t in zip(winding_numbers, test)] )

class CurveInPortPenalty(Optimizable):
    def __init__(self, port, curves, threshold, projection='zphi'):
        self.port = port
        self.curves = curves
        self.threshold = threshold
        self.projection = projection

        self.J_jax = jit(lambda gp, l1, gc, l2: curve_inside_port_penalty_pure(gp, l1, self.projection, gc, l2, self.threshold))
        self.thisgrad0 = jit(lambda gp, l1, gc, l2: grad(self.J_jax, argnums=0)(gp, l1, gc, l2))
        self.thisgrad1 = jit(lambda gp, l1, gc, l2: grad(self.J_jax, argnums=1)(gp, l1, gc, l2))
        self.thisgrad2 = jit(lambda gp, l1, gc, l2: grad(self.J_jax, argnums=2)(gp, l1, gc, l2))
        self.thisgrad3 = jit(lambda gp, l1, gc, l2: grad(self.J_jax, argnums=3)(gp, l1, gc, l2))

        self.ddJdgpdgp = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=0)(gp,l1,gc,l2))
        self.ddJdgpdl1 = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=1)(gp,l1,gc,l2))
        self.ddJdgpdgc = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=2)(gp,l1,gc,l2))
        self.ddJdgpdl2 = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=3)(gp,l1,gc,l2))
        self.ddJdl1dl1 = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=1), argnums=1)(gp,l1,gc,l2))
        self.ddJdl1dgc = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=1), argnums=2)(gp,l1,gc,l2))
        self.ddJdl1dl2 = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=1), argnums=3)(gp,l1,gc,l2))
        self.ddJdgcdgc = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=2), argnums=2)(gp,l1,gc,l2))
        self.ddJdgcdl2 = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=2), argnums=3)(gp,l1,gc,l2))
        self.ddJdl2dl2 = jit(lambda gp, l1, gc, l2: jacfwd(jacrev(self.J_jax, argnums=3), argnums=3)(gp,l1,gc,l2))

        super().__init__(depends_on=curves+[port])

    def count_inside_points(self):
        gp = self.port.gamma()
        l1 = self.port.gammadash()

        N = 0
        for c in self.curves:
            gc = c.gamma()
            N += count_inside_points(
                gp, l1, self.projection, gc, threshold=self.threshold
            )

        return N



    def J(self):
        gp = self.port.gamma()
        l1 = self.port.gammadash()

        out = 0
        for c in self.curves:
            gc = c.gamma()
            l2 = c.gammadash()
            out += self.J_jax(gp, l1, gc, l2)

        return out
    
    @derivative_dec
    def dJ(self):
        cc = self.curves + [self.port]

        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in cc]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in cc]  
        
        gp = self.port.gamma()
        l1 = self.port.gammadash()
        for i, c in enumerate(self.curves):
            gc = c.gamma()
            l2 = c.gammadash()

            # derivatives w.r.t port
            dgamma_by_dcoeff_vjp_vecs[-1] += self.thisgrad0(gp, l1, gc, l2)
            dgammadash_by_dcoeff_vjp_vecs[-1] += self.thisgrad1(gp, l1, gc, l2)

            # derivatives w.r.t curves
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad2(gp, l1, gc, l2)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad3(gp, l1, gc, l2)
        
        
        res = [
            self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i])\
          + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))
        ]
        res.append(
            self.port.dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[-1]) + self.port.dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[-1])
        )
        return sum(res)
    























def min_xy_distance(gamma1, gamma2):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula in xy plane.
    """
    g1cyl = gamma1[:, [2,0,1]]
    g2cyl = gamma2[:, [2,0,1]]

    g1 = gamma1[:,:2]
    g2 = gamma2[:,:2]

    dists = np.linalg.norm(g1[:,None,:]-g2[None,:,:], axis=2)
    f = np.maximum(g2cyl[None,:,0]-g1cyl[:,None,0], 0) / (g2cyl[None,:,0]-g1cyl[:,None,0])

    if (f==0).all():
        return np.nan
    else:
        return -np.min(-dists*f)

def cc_xy_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula in xy plane.
    """
    g1 = gamma1[:,:2]
    g2 = gamma2[:,:2]

    # Set gammadah in z-direction to almost 0. We cannot set it to zero, otherwise derivatives would be NaN when gammadash is purely along z
    l1 = l1.at[:,2].set( 1E-14 )
    l2 = l2.at[:,2].set( 1E-14 )

    # This is 0 for all points below (i.e z2<z1) their comparison point on the curve
    f = jnp.maximum(gamma2[None,:,2]-gamma1[:,None,2], 0)**2

    dists = jnp.sqrt(jnp.sum((g1[:, None, :] - g2[None, :, :])**2, axis=2))

    l1norm = jnp.linalg.norm(l1, axis=1) 
    l2norm = jnp.linalg.norm(l2, axis=1) 

    alen = l1norm[:, None] * l2norm[None, :]
    return jnp.sum(alen * f * jnp.maximum(minimum_distance-dists, 0)**2)

def min_zphi_distance(gamma1, gamma2):
    local_project = lambda x: project(x, gamma1)

    g1cyl = local_project(gamma1)
    g2cyl = local_project(gamma2)
    g1 = g1cyl[:,1:]
    g2 = g2cyl[:,1:]

    dists = np.linalg.norm(g1[:,None,:]-g2[None,:,:], axis=2)
    mask = g2cyl[None,:,0]>g1cyl[:,None,0]

    if mask.any():
        return -np.min(-dists*mask)
    else:
        return np.nan



def cc_zphi_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    local_project = lambda x: project(x, gamma1)

    g1cyl = local_project(gamma1)
    g2cyl = local_project(gamma2)

    l1cyl = local_project(l1)
    l2cyl = local_project(l2)

    g1 = g1cyl[:,1:]
    g2 = g2cyl[:,1:]

    # Set gammadah in z-direction to almost 0. We cannot set it to zero, otherwise derivatives would be NaN when gammadash is purely along z
    l1cyl = l1cyl.at[:,0].set( 1E-14 )
    l2cyl = l2cyl.at[:,0].set( 1E-14 )

    # This is 0 for all points below (i.e z2<z1) their comparison point on the curve
    f = jnp.maximum(g2cyl[None,:,0]-g1cyl[:,None,0], 0)**2

    dists = jnp.sqrt(jnp.sum((g1[:, None, :] - g2[None, :, :])**2, axis=2))

    l1norm = jnp.linalg.norm(l1cyl, axis=1) 
    l2norm = jnp.linalg.norm(l2cyl, axis=1) 

    alen = l1norm[:, None] * l2norm[None, :]
    return jnp.sum(alen * f * jnp.maximum(minimum_distance-dists, 0)**2)/(g1.shape[0])

class ProjectedCurveCurveDistance( Optimizable ):
    def __init__(self, base_curves, curve, minimum_distance=0, projection='xy'):
        self.base_curves = base_curves
        self.curve = curve

        self.projection=projection

        if self.projection=='xy':
            self.J_jax = jit(lambda gamma1, l1, gamma2, l2: cc_xy_distance_pure(gamma1, l1, gamma2, l2, minimum_distance))
        elif self.projection=='zphi':
            self.J_jax = jit(lambda gamma1, l1, gamma2, l2: cc_zphi_distance_pure(gamma1, l1, gamma2, l2, minimum_distance))

        self.thisgrad0 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2))
        self.thisgrad1 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2))
        self.thisgrad2 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2))
        self.thisgrad3 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2))

        self.ddJdg1dg1 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=0)(g1,l1,g2,l2))
        self.ddJdg1dl1 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=1)(g1,l1,g2,l2))
        self.ddJdg1dg2 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=2)(g1,l1,g2,l2))
        self.ddJdg1dl2 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=0), argnums=3)(g1,l1,g2,l2))
        self.ddJdl1dl1 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=1), argnums=1)(g1,l1,g2,l2))
        self.ddJdl1dg2 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=1), argnums=2)(g1,l1,g2,l2))
        self.ddJdl1dl2 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=1), argnums=3)(g1,l1,g2,l2))
        self.ddJdg2dg2 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=2), argnums=2)(g1,l1,g2,l2))
        self.ddJdg2dl2 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=2), argnums=3)(g1,l1,g2,l2))
        self.ddJdl2dl2 = jit(lambda g1, l1, g2, l2: jacfwd(jacrev(self.J_jax, argnums=3), argnums=3)(g1,l1,g2,l2))

        self.num_basecurves = len(base_curves)
        super().__init__(depends_on=base_curves + [curve])

    def shortest_distance(self):
        res = np.zeros((len(self.base_curves),))
        gamma = self.curve.gamma()
        for ii, c in enumerate(self.base_curves):
            gamma2 = c.gamma()
            if self.projection=='xy':
                res[ii] = min_xy_distance(gamma, gamma2)
            elif self.projection=='zphi':
                res[ii] = min_zphi_distance(gamma, gamma2)

        if all([np.isnan(k) for k in res]):
            raise RuntimeError("No curves in front of port")
        else:
            return np.min(res[~np.isnan(res)])

    def J(self):
        res = 0

        gamma = self.curve.gamma()
        l = self.curve.gammadash()
        for c in self.base_curves:
            gamma2 = c.gamma()
            l2 = c.gammadash()

            res += self.J_jax(gamma, l, gamma2, l2)

        return res
    
    @derivative_dec
    def dJ(self):
        cc = self.base_curves + [self.curve]

        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in cc]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in cc]

        gamma1 = self.curve.gamma()
        l1 = self.curve.gammadash()
        for i, c in enumerate(self.base_curves):
            gamma2 = c.gamma()
            l2 = c.gammadash()
            dgamma_by_dcoeff_vjp_vecs[-1] += self.thisgrad0(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[-1] += self.thisgrad1(gamma1, l1, gamma2, l2)
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad2(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad3(gamma1, l1, gamma2, l2)

        res = [self.base_curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.base_curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.base_curves))]
        res.append(
            self.curve.dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[-1]) + self.curve.dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[-1])
        )
        return sum(res)

    def ddJ_ddport(self):
        g1 = self.curve.gamma()
        l1 = self.curve.gammadash()
        dg1dx = self.curve.dgamma_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        dl1dx = self.curve.dgammadash_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        gamma_hessian = self.curve.gamma_hessian() # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs
        gammadash_hessian = self.curve.gammadash_hessian() # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs

        ndofs_port = self.curve.num_dofs()
        res = np.zeros((ndofs_port,ndofs_port))
        for c in self.base_curves:
            g2 = c.gamma()
            l2 = c.gammadash()
            hess00 = self.ddJdg1dg1(g1, l1, g2, l2) 
            hess01 = self.ddJdg1dl1(g1, l1, g2, l2) 
            hess11 = self.ddJdl1dl1(g1, l1, g2, l2) 

            grad0 = self.thisgrad0(g1, l1, g2, l2) # this is dJ/dgamma, size npts x 3
            grad1 = self.thisgrad1(g1, l1, g2, l2) # this is dJ/dgammadash, size npts x 3

            res += np.einsum('ijkl,ijm,kln->mn', hess00, dg1dx, dg1dx) \
                +  np.einsum('ijkl,ijm,kln->mn', hess11, dl1dx, dl1dx)\
                +  np.einsum('ij,ijkl->kl',grad0,gamma_hessian) \
                +  np.einsum('ij,ijkl->kl',grad1,gammadash_hessian) \
                +  np.einsum('kilj,kim,ljn->mn', hess01, dg1dx, dl1dx) \
                +  np.einsum('kilj,kin,ljm->mn', hess01, dg1dx, dl1dx)

        return res

    def ddJ_dportdcoil(self, curve=None):
        g1 = self.curve.gamma()
        l1 = self.curve.gammadash()
        dg1dx = self.curve.dgamma_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        dl1dx = self.curve.dgammadash_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        ndofs_c1 = self.curve.num_dofs()
        ndofs_bc = [c.num_dofs() for c in self.base_curves]
        res = np.zeros((sum(ndofs_bc),ndofs_c1))
        hess = []
        for k in self.unique_dof_lineage:
            if not isinstance(k,Curve): # only consider coils
                continue
            if k is self.curve: # skip dep on port
                continue

            if (not curve is None):
                if k is not curve:
                    continue 

            if np.any(k.dofs_free_status):

                shape = (k.local_dof_size,self.curve.num_dofs())
                res = np.zeros(shape)
                
                if k.local_dof_size>0:
                    for opt in k.dofs.dep_opts():
                        g2 = opt.gamma()
                        l2 = opt.gammadash()
                        dg2dx = opt.dgamma_by_dcoeff()
                        dl2dx = opt.dgammadash_by_dcoeff()

                        hg1g2 = self.ddJdg1dg2(g1,l1,g2,l2)
                        hl1g2 = self.ddJdl1dg2(g1,l1,g2,l2)
                        hg1l2 = self.ddJdg1dl2(g1,l1,g2,l2)
                        hl1l2 = self.ddJdl1dl2(g1,l1,g2,l2)

                        a = np.einsum('ijkl,ijm,kln->nm', hg1g2, dg1dx, dg2dx)
                        b = np.einsum('ijkl,ijm,kln->nm', hl1g2, dl1dx, dg2dx)
                        c = np.einsum('ijkl,ijm,kln->nm', hg1l2, dg1dx, dl2dx)
                        d = np.einsum('ijkl,ijm,kln->nm', hl1l2, dl1dx, dl2dx)

                        res[:,:] += a + b + c + d
                
                hess.append(res)

        return np.concatenate(hess)
        


def xy_convexity( pts, g, gd, gdd ):
    gd = gd.at[:,2].set( 1E-12 ) # not zero to avoid singularities
    gdd = gdd.at[:,2].set( 1E-12 ) # not zero to avoid singularities

    # First we evaluate the 2D curvature
    kappa = (gd[:,0]*gdd[:,1] - gd[:,1]*gdd[:,0]) / (gd[:,0]**2 + gd[:,1]**2)**(3./2.)

    integral_of_kappa = jnp.trapz(jnp.abs(kappa)*jnp.linalg.norm(gd,axis=1), pts)

    # Allow 5% margin of error for numerical integration error
    return jnp.max( jnp.array([integral_of_kappa - 1.05*2.0*jnp.pi]), 0 )**2

def zphi_convexity( pts, g, gd, gdd):
    local_project = lambda x: project(x, g)

    gd_projected = local_project(gd)
    gdd_projected = local_project(gdd)

    gd_projected = gd_projected.at[:,0].set( 1E-12 ) # not zero to avoid singularities
    gdd_projected = gdd_projected.at[:,0].set( 1E-12 ) # not zero to avoid singularities

    # First we evaluate the 2D curvature
    kappa = (gd_projected[:,1]*gdd_projected[:,2] - gd_projected[:,2]*gdd_projected[:,1]) / (gd_projected[:,1]**2 + gd_projected[:,2]**2)**(3./2.)

    integral_of_kappa = jnp.trapz(jnp.abs(kappa)*jnp.linalg.norm(gd_projected,axis=1), pts)

    # Allow 5% margin of error for numerical integration error
    return jnp.max( jnp.array([integral_of_kappa - 1.05*2.0*jnp.pi]), 0 )**2

class ProjectedCurveConvexity( Optimizable ):
    def __init__(self, curve, projection='xy'):
        self.curve = curve
        self.projection = projection

        if self.projection == 'xy':
            self.J_jax = jit(lambda gamma, gammadash, gammadashdash: xy_convexity(self.curve.quadpoints, gamma, gammadash, gammadashdash))
        elif self.projection =='zphi':
            self.J_jax = jit(lambda gamma, gammadash, gammadashdash: zphi_convexity(self.curve.quadpoints, gamma, gammadash, gammadashdash))
        
        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash: grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash: grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash: grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash))

        super().__init__(depends_on=[curve])

    def J(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        gammadashdash = self.curve.gammadashdash()
        return self.J_jax(gamma, gammadash, gammadashdash)


    @derivative_dec
    def dJ(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        gammadashdash = self.curve.gammadashdash()

        grad0 = self.thisgrad0(gamma, gammadash, gammadashdash)
        grad1 = self.thisgrad1(gamma, gammadash, gammadashdash)
        grad2 = self.thisgrad2(gamma, gammadash, gammadashdash)

        return self.curve.dgamma_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1) + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
    
    def ddJ_ddport(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        gammadashdash = self.curve.gammadashdash()
        hess0 = hessian(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash) # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        hess1 = hessian(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash) # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        hess2 = hessian(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash) # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        dgdx = self.curve.dgamma_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        dgdxdash = self.curve.dgammadash_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        dgdxdashdash = self.curve.dgammadashdash_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs

        grad0 = self.thisgrad0(gamma, gammadash, gammadashdash) # this is dJ/dgamma, size npts x 3
        grad1 = self.thisgrad1(gamma, gammadash, gammadashdash) # this is dJ/dgamma, size npts x 3
        grad2 = self.thisgrad2(gamma, gammadash, gammadashdash) # this is dJ/dgamma, size npts x 3
        gamma_hessian = self.curve.gamma_hessian() # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs
        gammadash_hessian = self.curve.gammadash_hessian() # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs
        gammadashdash_hessian = self.curve.gammadashdash_hessian() # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs

        # Contribution for derivatives w.r.t gamma
        ddJ_dpport_1 = np.einsum('ijkl,ijm,kln->mn', hess0, dgdx, dgdx) + np.einsum('ij,ijkl->kl',grad0,gamma_hessian) # this should be size ndofs x ndofs
        ddJ_dpport_2 = np.einsum('ijkl,ijm,kln->mn', hess1, dgdxdash, dgdx) + np.einsum('ij,ijkl->kl',grad1,gammadash_hessian) # this should be size ndofs x ndofs
        ddJ_dpport_3 = np.einsum('ijkl,ijm,kln->mn', hess2, dgdxdashdash, dgdx) + np.einsum('ij,ijkl->kl',grad2,gammadashdash_hessian) # this should be size ndofs x ndofs

        raise NotImplementedError("Missing mixed terms")

        return ddJ_dpport_1 + ddJ_dpport_2 + ddJ_dpport_3

    def ddJ_dportdcoil(self, curve=None):
        return 0 #Does not depend on coils

