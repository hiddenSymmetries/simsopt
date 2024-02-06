import numpy as np
from jax import grad
import jax.numpy as jnp
from .hull import hull2D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec

__all__ = ['PortSize', 'Access']



def coil_normal_distance_pure(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot, p):
    """Find the maximum port size given surface and coils
    
    Heaviside, min and max functions are approximated with differentiable functions so that this function can be used in an optimization.

    Args:
        - gamma_curves: np.array of size (ncurves, npts, 3). List of points defining the coils
        - gamma_surf: np.array of size (nphi, ntheta, 3). List of points on the surface
        - unit_normal: np.array of size (nphi, ntheta, 3). Unit vector normal to gamma_surf
        - unit_tangent: np.array of size (nphi, ntheta, 3). Unit vector tangent to gamma_surf
        - phi_ind: List of phi-indices for points on gamma_surf that are candidate for a port access. Shape is (np,)
        - theta_ind: List of theta-indices for points on gamma_surf that are candidates for a port access. Shape is (np,)
        - max_sizes:: Maximum size for port at position iphi, itheta. Shape should be (nphi,ntheta)
        - ftop: list of polynomials to fit the upper envelop of the surface projected on the plane tangent to gamma_surf at iphi, itheta. Should be of shape (nphi, ntheta, deg+1)
        - fbot: list of polynomials to fit the lower envelop of the surface projected on the plane tangent to gamma_surf at iphi, itheta. Should be of shape (nphi, ntheta, deg+1)
        - p: coefficient for p-norm
    Output:
        - rp: largest circular port size on gamma_surf.
    """
    hv = lambda x: logistic(x, k=4)
    min = lambda x: np.sum(x**(-p))**(-1./p)
    max = lambda x: np.sum(x**( p))**( 1./p)

    return coil_normal_distance(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot, hv, min, max, False)

def coil_normal_distance(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot, hv, min, max, return_pos ):
    """ Return largest circular port access size, with normal access

    For each quadrature point on the surface, evaluate the smallest distance
    from the coils to the line normal to the surface at this point. Return then
    largest of these minimum radii.

    Args:
        - gamma_curves: np.array of size (ncurves, npts, 3). List of points defining the coils
        - gamma_surf: np.array of size (nphi, ntheta, 3). List of points on the surface
        - unit_normal: np.array of size (nphi, ntheta, 3). Unit vector normal to gamma_surf
        - unit_tangent: np.array of size (nphi, ntheta, 3). Unit vector tangent to gamma_surf
        - phi_ind: List of phi-indices for points on gamma_surf that are candidate for a port access. Shape is (np,)
        - theta_ind: List of theta-indices for points on gamma_surf that are candidates for a port access. Shape is (np,)
        - max_sizes:: Maximum size for port at position iphi, itheta. Shape should be (nphi,ntheta)
        - ftop: list of polynomials to fit the upper envelop of the surface projected on the plane tangent to gamma_surf at iphi, itheta. Should be of shape (nphi, ntheta, deg+1)
        - fbot: list of polynomials to fit the lower envelop of the surface projected on the plane tangent to gamma_surf at iphi, itheta. Should be of shape (nphi, ntheta, deg+1)
        - hv: Heaviside function, or a differentiable approximation
        - min: Minimum function, or a differentiable approximation 
        - max: Maximum function, or a differentiable approximation 
        - return_pos: Flag to control output

    Output:
        - rp: Maximum circular port size on gamma_surf.
        - pos: (iphi, itheta) index of the maximum port size position. Only returned if return_pos is True.
    """
    nn = phi_ind.size
    port_sizes = jnp.zeros((nn,))

    for counter, (itheta, iphi) in enumerate(zip(theta_ind,phi_ind)):
        upper_envelop = lambda x: jnp.polyval(ftop[iphi,itheta], x)
        lower_envelop = lambda x: jnp.polyval(fbot[iphi,itheta], x)

        min_dist = find_port_size(gamma_curves, gamma_surf, iphi, itheta, unit_normal[iphi,itheta], unit_tangent[iphi,itheta], hv, max_sizes[iphi,itheta], upper_envelop, lower_envelop, min)

        port_sizes = port_sizes.at[counter].set(min_dist)

    if return_pos:
        ind = np.argmax(port_sizes)
        pos = (phi_ind[ind], theta_ind[ind])
        return max(port_sizes), pos
    else:
        return max(port_sizes)

def find_port_size(gamma_curves, gamma_surf, iphi, itheta, unit_normal, unit_tangent, hv, max_size, ftop, fbot, min):    
    """"Find the maximum port size to access a toroidal volume from a given position
    
    Args:
        - gamma_curves: Collection of points in cartesian coordinates that represents all coils. Shape should be (ncurve, npts, 3)
        - gamma_surf: Collection of points in cartesian coordinates that represents the boundary of the toroidal volume. Shape should be (nphi, ntheta, 3)
        - iphi: toroidal index of the access point on gamma_surf. Should satisfy 0<=iphi<=nphi
        - itheta: Poloidal index of the access point on gamma_surf. Should satisfy 0<=itheta<=ntheta
        - unit_normal: unit vector normal to gamma_surf at gamma_surf(iphi,itheta), pointing outwards. Should have shape (3,)
        - unit_tangent: unit vector tangent to gamma_surf at gamma_surf(iphi,itheta), pointing in the toroidal direction. Should be obtained by calling surface.dgammadphi. Should have shape (3,)
        - hv: function that should either be the heaviside function, or a differentiable approximation of the heaviside function (required for optimization)
        - max_size: maximum port size
        - ftop: function that return the upper envelop
        - fbot: same as ftop, but for the lower envelop.
        - min: function to evaluate the minimum of a np.array. Use np.min if you don't need differentiability, instead consider the p-norm.
    Returns:
        - rp: float, port size at position gamma_surf(iphi,itheta)
    """
    # Evaluate position on surface
    xsurf = gamma_surf[iphi,itheta]
    # Construct coordinate associated to xsurf, and build projection operatore
    bnorm = jnp.cross(unit_normal, unit_tangent)
    M = jnp.array([unit_normal,unit_tangent,bnorm]).transpose()
    invM = jnp.linalg.inv(M)
    def project(x):
        return jnp.einsum('ij,...j->...i',invM,x-xsurf[...,:])

    nphi, _, _ = gamma_surf.shape
    nphi = int(nphi/3)
    phi_start = iphi-int(nphi/2)
    phi_end = iphi+int(nphi/2)+1
    gamma_surf = gamma_surf[phi_start:phi_end]

    gamma_coil_proj = project(gamma_curves)
    ncurves, npts, _ = gamma_curves.shape


    distances = jnp.zeros((1+ncurves*npts,))
    distances = distances.at[0].set( max_size )
    counter = 1
    
    # Consider all curve pts
    x = gamma_coil_proj.reshape((-1,3))

    # Evaluate the upper and lower envelop for this toroidal plane            
    tn = ftop(x[:,1])
    bn = fbot(x[:,1])

    # Read coordinate of the point and of the upper and lower envelop in the (n,t,b) coordinate system
    dd = x[:,1]**2 + x[:,2]**2

    # All logical tests are replaced by an approximated Heaviside function to make the port size evaluation differentiable. 
    # here hv(max_size-dd) checks that the point is not further than the maximum port size, hv(-cb)*hv(cn-bn) checks if the point is in front of the surface, and below the point xsurf, and hv(cb)*hv(cn-tn) checks if the point is in front of the surface, and above the point xsurf
    # test is 1 only if the point is closer than max_size, and if the point is in front of the surface.
    test = hv(max_size-dd) * (hv(-x[:,2])*hv(x[:,0]-bn) + hv(x[:,2])*hv(x[:,0]-tn))

    # We shift test and multiply it by a large number, so that if the point does not satisfy test, we associate to it a large distance. Here we chose a multiplication factor of 1E2 - larger number make the target function more stiff
    distances = distances.at[1:].set( (-1e2*(test-1) +1) * dd )
    counter += 1

    # Return port size
    return jnp.sqrt(min(distances))

def heaviside(x):
    """Just the regular Heaviside function
    """
    f = jnp.zeros(x.shape)
    f = f.at[jnp.where(x>0)].set(1.0)
    f = f.at[jnp.where(x==0)].set(0.5) 
    return f
  
def logistic(x, k=10):
    """Differentiable approximation to the Heaviside function
    
    Args:
        - x: function argument
        - k: if k large, this function approaches the Heaviside function.

    Output:
        - y: np.array of the same shape as x.
    """
    return 0.5 + 0.5*jnp.tanh(k*x)

class PortSize(Optimizable):
    r"""
    Evaluate the largest port size given some coils and a boundary. The porte size is defined as the radius of the largest straight cylinder that can fit within the coils and connect to the boundary along its normal direction

    Boundary should span 3/2 field periods, and curves should be all coils spanning the same 3/2 field periods.
    """
    def __init__(self, curves, surf, p=4):
        self.curves = curves

        # Every surface dependent quantity is only evaluated once for speed
        self.update_surface(surf)

        # I don't know how to get derivatives of unit_normal wrt boundary dofs; for now this class only depends on curves. Should not be used in any kind of combined approach, excepted if finite differences are implemented
        # Also, if the port size is made dependent on the surface, the objective function might not be differentiable anymore. 
        super().__init__(depends_on=curves)

        self.J_jax=lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot: coil_normal_distance_pure(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot, p)
        self.thisgrad0 =lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot: grad(self.J_jax, argnums=0)(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot)
        #self.thisgrad1 = jit(lambda gamma_curves, gamma_surf, unit_normal: grad(self.J_jax, argnums=1)(gamma_curves, gamma_surf, unit_normal))
        #self.thisgrad2 = jit(lambda gamma_curves, gamma_surf, unit_normal: grad(self.J_jax, argnums=2)(gamma_curves, gamma_surf, unit_normal))

    def update_surface(self, surf):
        """Update all surface dependent quantities.
        
        This should be used only to set the problem, and not in an optimization loop (this class assumes the boundary dofs to be fixed)

        Args:
            - surf: Surface object describing a toroidal volume to be accessed with a port.

        Exceptions:
            - ValueError: if the number of toroidal quadrature points of the surface is not a multiple of 3.
        """
        self.boundary = surf
        R = surf.major_radius()
        Nfp = surf.nfp
        # Only consider points on the outboard side
        self.gamma_boundary = self.boundary.gamma()
        nphi, ntheta, _ = self.gamma_boundary.shape
        if np.mod(nphi,3)!=0:
            raise ValueError('Boundary should have quadpoints_phi be a multiple of 3')
        nphi = int(nphi/3)
        dmax = 2*np.pi*R/Nfp * 1/nphi

        # self.phi_ind and self.theta_ind are arrays that store which point on the surface are to be considered when evaluating the port access. We set phi_ind such that all points are in the central half field period of the surface, and theta_ind such that only outboard points are considered.
        phi_ind = jnp.array([], dtype=int)
        theta_ind = jnp.array([], dtype=int)
        for iphi in range(nphi,2*nphi): #only consider one half field period
            for itheta in range(ntheta):
                Rsurf_square = self.gamma_boundary[iphi,itheta,0]**2+self.gamma_boundary[iphi,itheta,1]**2
                R0 = jnp.sum(self.gamma_boundary[iphi,:,0]**2 + self.gamma_boundary[iphi,:,1]**2) / ntheta
                if Rsurf_square>=R0:
                    phi_ind = jnp.append(phi_ind, iphi)
                    theta_ind = jnp.append(theta_ind, itheta)
        self.phi_ind = phi_ind
        self.theta_ind = theta_ind

        # Pre-compute the unit normal and unit tangent on each point of the surface
        self.unit_normal = self.boundary.unitnormal()
        tangent = self.boundary.dgammadphi()
        norm = np.linalg.norm(tangent, axis=2)
        self.unit_tangent = tangent / norm[:,:,None]

        # Prepare all max port size and prepare polynomial fits
        # For each point x=(phi,theta) on the surface, the maximal port size is defined as the max radius of a circle centered at x that fits within the surface projected on the plane tangent to the surface at x
        # Furthermore, to evaluate whether a coil is in front, above, behind, or below the surface, we need to fit the upper and lower "envelop" of the prohected surface
        self.max_port_size = np.zeros((3*nphi,ntheta))
        deg = 5 # Degree of polynomial fit
        # self.bot_fits = np.zeros((3*nphi,ntheta,3,deg+1)) # upper envelop fit
        # self.top_fits = np.zeros((3*nphi,ntheta,3,deg+1)) # lower envelop fit
        self.bot_fits = np.zeros((3*nphi,ntheta,deg+1))
        self.top_fits = np.zeros((3*nphi,ntheta,deg+1))
        to_pop = []
        for ii, (iphi, itheta) in enumerate(zip(self.phi_ind, self.theta_ind)):
            # Construct coordinate system with normal, and tangent vectors. 
            bnorm = jnp.cross(self.unit_normal[iphi,itheta], self.unit_tangent[iphi,itheta])
            
            # Build operator to project any point (x,y,z) on the new coordinate system (n,t,b)
            M = jnp.array([self.unit_normal[iphi,itheta], self.unit_tangent[iphi,itheta],bnorm]).transpose()
            invM = jnp.linalg.inv(M)
            xsurf = self.gamma_boundary[iphi,itheta]

            # Only consider points on a half field period centered around x.
            # These indices are different from phi_ind!
            phi_start = iphi-int(nphi/2)
            phi_end = iphi+int(nphi/2)+1
            surf = self.gamma_boundary[phi_start:phi_end,:,:]
            gamma_surf_proj = np.einsum('ij,lmj->lmi', invM, surf - xsurf[None,None,:]) # This is the section of the surface projected on the tangent plane (i.e. in coord system (n,t,b)) 

            xproj = gamma_surf_proj.reshape((-1,3))
            xflat = xproj[:,1:]
            try:
                hull = hull2D( xflat, surf )
                uenv = hull.envelop(dmax, 'upper')
                lenv = hull.envelop(dmax, 'lower')
            except BaseException:
                print(f'Popping point (phi,theta)={(iphi,itheta)}')
                to_pop.append(ii)
                continue

            xyztop = xproj[uenv,:]
            xyzbot = xproj[lenv,:]

            # Evaluate closest point from the envelop. This sets the maximum port size
            tt = np.append(xyztop,xyzbot,axis=0)
            self.max_port_size[iphi,itheta] = np.min(tt[:,1]**2 + tt[:,2]**2)

            # Interpolate lower and upper envelop - this is a linear interpolation, does it keep the C1 property? I don't think so. Smooth interpolation would be better
            #self.bot_fits[iphi,itheta] = lambda x: np.interp(x, xyzbot[:,1], xyzbot[:,2] )
            #self.top_fits[iphi,itheta] = lambda x: np.interp(x, xyztop[:,1], xyztop[:,2] )
            self.bot_fits[iphi,itheta] = np.polyfit(xyzbot[:,1], xyzbot[:,2], deg)
            self.top_fits[iphi,itheta] = np.polyfit(xyztop[:,1], xyztop[:,2], deg)
        
        if len(to_pop)>0:
            to_pop = np.array(to_pop)
            self.phi_ind = np.delete(self.phi_ind, to_pop)
            self.theta_ind = np.delete(self.theta_ind, to_pop)

    def J(self):
        gamma_curves = jnp.array([curve.gamma() for curve in self.curves])

        return self.J_jax(gamma_curves, self.gamma_boundary, self.unit_normal, self.unit_tangent, self.phi_ind, self.theta_ind, self.max_port_size, self.top_fits, self.bot_fits)
    
    @derivative_dec
    def dJ(self):
        gamma_curves = jnp.array([curve.gamma() for curve in self.curves])

        grad0 = self.thisgrad0(gamma_curves, self.gamma_boundary, self.unit_normal, self.unit_tangent, self.phi_ind, self.theta_ind, self.max_port_size, self.top_fits, self.bot_fits)
        #grad1 = self.thisgrad1(gamma_curves, gamma_boundary, unit_normal)
        #grad2 = self.thisgrad2(gamma_curves, gamma_boundary, unit_normal)

        return np.sum([curve.dgamma_by_dcoeff_vjp(gg) for curve, gg in zip(self.curves, grad0)])
    
    def find_max_port_size_and_position(self, smooth=False):
        gamma_curves = np.array([curve.gamma() for curve in self.curves])
        gamma_surf = self.gamma_boundary
        unit_normal = self.unit_normal
        unit_tangent = self.unit_tangent
        phi_ind = self.phi_ind
        theta_ind = self.theta_ind
        max_sizes = self.max_port_size
        ftop = self.top_fits
        fbot = self.bot_fits

        if smooth:
            hv = logistic
            p =10
            min = lambda x: np.sum(x**(-p))**(-1./p)
            max = lambda x: np.sum(x**( p))**( 1./p)
        else:
            hv = heaviside
            min = np.min
            max = np.max

        rp, pos =  coil_normal_distance(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot, hv, min, max, True)
        return rp, pos
    
    def get_port_size_at_position(self, iphi, itheta):
        if iphi not in self.phi_ind:
            raise ValueError('Toroidal plane is excluded')
        if itheta not in self.theta_ind:
            raise ValueError('Point is not on the outboard side of the surface')
        gamma_curves = np.stack([curve.gamma() for curve in self.curves])
        gamma_boundary = self.gamma_boundary
        unit_normal = self.unit_normal[iphi,itheta]
        unit_tangent = self.unit_tangent[iphi,itheta]
        msize = self.max_port_size[iphi,itheta]
        ftop = self.top_fits[iphi,itheta]
        fbot = self.bot_fits[iphi,itheta]

        lower_envelop = lambda x: np.polyval(fbot, x)
        upper_envelop = lambda x: np.polyval(ftop, x)

        min = lambda x: np.sum(x**(-10))**(-1./10)

        return find_port_size(gamma_curves, gamma_boundary, iphi, itheta, unit_normal, unit_tangent, heaviside, msize, upper_envelop, lower_envelop, np.min)
    
    def plot(self, iphi, itheta, rp=None, ax=None):

        if rp is None:
            rp = self.get_port_size_at_position( iphi, itheta )

        gamma_bnd = self.gamma_boundary
        unit_normal = self.unit_normal[iphi,itheta]
        unit_tangent = self.unit_tangent[iphi,itheta]

        bnorm = jnp.cross(unit_normal, unit_tangent)
        M = jnp.array([unit_normal,unit_tangent,bnorm]).transpose()
        invM = jnp.linalg.inv(M)
        def project(x):
            return jnp.einsum('ij,...j->...i',invM,x-gamma_bnd[iphi,itheta,:])
        
        nphi, _, _ = gamma_bnd.shape
        nphi = int(nphi/3)
        phi_start = iphi-int(nphi/2)
        phi_end = iphi+int(nphi/2)+1
        gamma_surf = gamma_bnd[phi_start:phi_end,:,:]

        gamma_surf_proj = project(gamma_surf).reshape((-1,3))

        ftop = self.top_fits[iphi,itheta]
        fbot = self.bot_fits[iphi,itheta]

        lower_envelop = lambda x: np.polyval(fbot, x)
        upper_envelop = lambda x: np.polyval(ftop, x)

        if ax is None:
            fig, ax = plt.subplots()

        plt.scatter(gamma_surf_proj[:,1], gamma_surf_proj[:,2], s=1, color='k')
        xmin = np.min(gamma_surf_proj[:,1])
        xmax = np.max(gamma_surf_proj[:,1])
        xplot = np.linspace(xmin,xmax,128)
        plt.plot(xplot, lower_envelop(xplot), linewidth=3, color='g')
        plt.plot(xplot, upper_envelop(xplot), linewidth=3, color='r')

        color = cm.rainbow(np.linspace(0, 1, len(self.curves)))
        for c, col in zip(self.curves, color):
            gamma_proj = project(c.gamma())
            npts, _ = gamma_proj.shape
            ind_above = []
            ind_below = []
            ind_front = []
            for ii in range(npts):
                if gamma_proj[ii,2]>upper_envelop(gamma_proj[ii,1]):
                    ind_above.append(ii)
                elif gamma_proj[ii,2]<lower_envelop(gamma_proj[ii,1]):
                    ind_below.append(ii)
                elif gamma_proj[ii,0]>=0:
                    ind_front.append(ii)

            plt.scatter(gamma_proj[ind_below,1], gamma_proj[ind_below,2], s=5, marker='^', c=col)
            plt.scatter(gamma_proj[ind_above,1], gamma_proj[ind_above,2], s=5, marker='o', c=col)
            plt.scatter(gamma_proj[ind_front,1], gamma_proj[ind_front,2], s=5, marker='v', c=col)

        l = np.linspace(0,2*np.pi,64)
        ax.plot(rp*np.cos(l), rp*np.sin(l), 'k-', linewidth=3)
        ax.set_aspect('equal')
         

#=======================================================
# Promote accessibility
def promote_accessibility( gamma_curves, gamma_surf, unit_normal, unit_tangent, pfactor ):  
    nphi, ntheta, _ = gamma_surf.shape
    distances = jnp.zeros((nphi*ntheta,))
    counter = -1
    for ip in range(nphi):
        for it in range(ntheta):
            counter += 1

            # Evaluate position on surface
            xsurf = gamma_surf[ip, it]
            R = jnp.sqrt(gamma_surf[ip,:,0]**2 + gamma_surf[ip,:,1]**2 )
            R0 = jnp.mean( R )

            un = unit_normal[ip, it]
            ut = unit_tangent[ip, it]
            # Construct coordinate associated to xsurf, and build projection operator
            ub = jnp.cross(un, ut)
            M = jnp.array([un,ut,ub]).transpose()
            invM = jnp.linalg.inv(M)
            
            # Consider all curve pts
            gamma_coil_proj = jnp.einsum('ij,...j->...i',invM,gamma_curves-xsurf[...,:])
            x = gamma_coil_proj.reshape((-1,3))
            npts, _ = x.shape

            # Find minimal distance from pt to coil, weighted by Heaviside fct.
            dd = logistic( R[it]-R0, k=2*pfactor ) * logistic( x[:,0] ) * ( x[:,1]**2 + x[:,2]**2 )
            distances = distances.at[counter].set( jnp.sum(dd**pfactor) / npts )

    return jnp.mean( distances )**(1./pfactor)

class Access(Optimizable):
    def __init__(self, curves, surf, p=4):
        self.curves = curves
        self.boundary = surf
        self.p = p

        # I don't know how to get derivatives of unit_normal wrt boundary dofs; for now this class only depends on curves. Should not be used in any kind of combined approach, excepted if finite differences are implemented
        # Also, if the port size is made dependent on the surface, the objective function might not be differentiable anymore. 
        #super().__init__(depends_on=curves + [surf])
        super().__init__(depends_on=curves)

        self.J_jax=lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, p: promote_accessibility(gamma_curves, gamma_surf, unit_normal, unit_tangent, p)
        self.thisgrad0 =lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, p: grad(self.J_jax, argnums=0)(gamma_curves, gamma_surf, unit_normal, unit_tangent, p)
        self.thisgrad1 = lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, p: grad(self.J_jax, argnums=1)(gamma_curves, gamma_surf, unit_normal, unit_tangent, p)
        self.thisgrad2 = lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, p: grad(self.J_jax, argnums=2)(gamma_curves, gamma_surf, unit_normal, unit_tangent, p)
        self.thisgrad3 = lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, p: grad(self.J_jax, argnums=3)(gamma_curves, gamma_surf, unit_normal, unit_tangent, p)


    def J(self):
        gamma_curves = jnp.array([curve.gamma() for curve in self.curves])
        gamma_boundary = self.boundary.gamma()
        unit_normal = self.boundary.unitnormal()
        tangent = self.boundary.dgammadphi()
        norm = np.linalg.norm(tangent, axis=2)
        unit_tangent = tangent / norm[:,:,None]

        return self.J_jax(gamma_curves, gamma_boundary, unit_normal, unit_tangent, self.p)
    
    @derivative_dec
    def dJ(self):
        gamma_curves = jnp.array([curve.gamma() for curve in self.curves])
        gamma_boundary = self.boundary.gamma()
        unit_normal = self.boundary.unitnormal()
        tangent = self.boundary.dgammadphi()
        norm = np.linalg.norm(tangent, axis=2)
        unit_tangent = tangent / norm[:,:,None]

        grad0 = self.thisgrad0(gamma_curves, gamma_boundary, unit_normal, unit_tangent, self.p)
        # grad1 = self.thisgrad1(gamma_curves, gamma_boundary, unit_normal, unit_tangent, self.p)
        # grad2 = self.thisgrad2(gamma_curves, gamma_boundary, unit_normal, unit_tangent, self.p)
        # grad3 = self.thisgrad3(gamma_curves, gamma_boundary, unit_normal, unit_tangent, self.p)

        return np.sum([curve.dgamma_by_dcoeff_vjp(gg) for curve, gg in zip(self.curves, grad0)])
         

