import numpy as np
from jax import grad, hessian, jacfwd, jacrev
import jax.numpy as jnp
from .hull import hull2D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative

__all__ = ['PortSize', 'VerticalPortDiscrete', 'ProjectedEnclosedArea', 'ProjectedCurveCurveDistance', 'ProjectedCurveConvexity', 'DirectedFacingPort']



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
         





def vertical_access(gamma_surf, gamma_curves, phi_ind, theta_ind, dmax):
    
    xflat = gamma_surf.reshape((-1,3))

    xcurves = gamma_curves[:,0]
    ycurves = gamma_curves[:,1]
    zcurves = gamma_curves[:,2]

    rports = []
    for iphi, itheta, dd in zip(phi_ind, theta_ind, dmax):
        xx = gamma_surf[iphi,itheta,0]
        yy = gamma_surf[iphi,itheta,1]

        surf_dist_to_pt = np.sqrt((xx-xflat[:,0])**2 + (yy-xflat[:,1])**2)
        ind = np.where(surf_dist_to_pt<=dd)
        mean_z = np.mean( xflat[ind,2] )

        ind = np.where( zcurves>mean_z )[0]

        min_dist_to_curve = np.min(
            np.sqrt((xx-xcurves[ind])**2 + (yy-ycurves[ind])**2)
        )

        rports.append( np.min( [min_dist_to_curve, dd] ) )

    ii = np.argmax( rports )
    return rports[ii], phi_ind[ii], theta_ind[ii], ii

class VerticalPortDiscrete( Optimizable ):
    def __init__(self, surface, curves=None ):
        self.surface = surface
        self.curves = curves

        self.update_surface()

        super().__init__(depends_on=curves)

    def update_surface(self):
        self.gamma_surf = self.surface.gamma()
        nphi, ntheta, _ = self.gamma_surf.shape

        phi_ind = []
        theta_ind = []
        dmax = []
        for iphi in range(nphi):
            for itheta in range(ntheta):

                x = self.gamma_surf[iphi,itheta,0]
                y = self.gamma_surf[iphi,itheta,1]
                z = self.gamma_surf[iphi,itheta,2]
        
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2( y, x )

                cs = self.surface.cross_section( phi )
                Rcs = np.sqrt(cs[:,0]**2 + cs[:,1]**2)
                irmin = np.argmin(Rcs)
                irmax = np.argmax(Rcs)

                izmax = np.argmax(cs[:,2])

                r = np.sqrt(x**2 + y**2)
                if r>=Rcs[izmax]:
                    if z>cs[irmax,2]:
                        phi_ind.append( iphi )
                        theta_ind.append( itheta )
                        
                        dmax.append(np.min([
                            np.sqrt((x-cs[irmin,0])**2+(y-cs[irmin,1])**2),
                            np.sqrt((x-cs[irmax,0])**2+(y-cs[irmax,1])**2)
                        ]))

                if r<Rcs[izmax]:
                    if z>cs[irmin,2]:
                        phi_ind.append( iphi )
                        theta_ind.append( itheta )

                        dmax.append(np.min([
                            np.sqrt((x-cs[irmin,0])**2+(y-cs[irmin,1])**2),
                            np.sqrt((x-cs[irmax,0])**2+(y-cs[irmax,1])**2)
                        ]))


        self.phi_ind = np.asarray( phi_ind )
        self.theta_ind = np.asarray( theta_ind )
        self.dmax = np.asarray( dmax )


    def find_max_port_size_and_position(self):

        gamma_curves = np.stack([curve.gamma() for curve in self.curves]).reshape((-1,3))

        return vertical_access( self.gamma_surf, gamma_curves, self.phi_ind, self.theta_ind, self.dmax )


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
    
    def ddJ_dportdcoil(self):
        return 0 #Does not depend on coils



def upward_facing_pure( nznorm ):
    return jnp.sum(jnp.maximum(-nznorm, 0)**2)

class DirectedFacingPort(Optimizable):
    def __init__(self, curve, projection='z'):
        self.curve = curve
        self.projection = projection

        self.J_jax = lambda nz: upward_facing_pure(nz)
        self.thisgrad = lambda nz: grad(self.J_jax)(nz)
        self.hessian = lambda nz: hessian(self.J_jax)(nz) 

        super().__init__(depends_on=[curve])

    def J(self):
        if self.projection=='z':
            n=self.curve.zfactor()
        elif self.projection=='r':
            n=self.curve.rfactor()
        return self.J_jax(n)
    
    @derivative_dec
    def dJ(self):
        if self.projection=='z':
            f = self.curve.dzfactor_by_dcoeff_vjp
            n = self.curve.zfactor()
        elif self.projection=='r':
            f = self.curve.drfactor_by_dcoeff_vjp
            n = self.curve.rfactor()

        return f(self.thisgrad(n))
    
    def ddJ_ddport(self):
        if self.projection=='z':
            dgdx = self.curve.dzfactor_by_dcoeff()
            curve_hessian = self.curve.zfactor_hessian()
            n = self.curve.zfactor()
        elif self.projection=='r':
            dgdx = self.curve.drfactor_by_dcoeff()
            curve_hessian = self.curve.rfactor_hessian()
            n = self.curve.rfactor()
            
        hess = self.hessian(n) # this is d^2J/dgamma_i dgamma_j. Size npts x npts 
        grad0 = self.thisgrad(n) # this is dJ/dgamma, size npts 

        return np.einsum('ij,il,jm->lm', hess, dgdx, dgdx) + np.einsum('i,ilm->lm',grad0,curve_hessian) # this should be size ndofs x ndofs

    def ddJ_dportdcoil(self):
        return 0 # does not depend on coils


def min_xy_distance(gamma1, gamma2):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula in xy plane.
    """
    g1 = gamma1[:,:2]
    g2 = gamma2[:,:2]

    ind = np.where(gamma2[:,2]>0)[0]
    if len(ind)==0: # Then the curve is behind the port - ignore this
        return np.nan
    else:
        dists = np.sqrt(np.sum((g1[:, None, :] - g2[None, ind, :])**2, axis=2))
        return np.min(dists)

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

    g1 = local_project(gamma1)
    g2 = local_project(gamma2)
    ind = np.where(g2[:,1]>0)[0]

    g1 = g1.at[:,1].set(0)
    g2 = g2.at[:,1].set(0)

    if len(ind)==0:
        return np.nan
    else:
        dists = np.sqrt(np.sum((g1[:, None, :] - g2[None, ind, :])**2, axis=2))
        return np.min(dists)

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

    def ddJ_dportdcoil(self):
        g1 = self.curve.gamma()
        l1 = self.curve.gammadash()
        dg1dx = self.curve.dgamma_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        dl1dx = self.curve.dgammadash_by_dcoeff() # this is dgamma/dx, size npts x 3 x ndofs
        ndofs_c1 = self.curve.num_dofs()
        ndofs_bc = [c.num_dofs() for c in self.base_curves]
        res = np.zeros((sum(ndofs_bc),ndofs_c1))
        counter=0
        for ii, c in enumerate(self.base_curves):
            g2 = c.gamma()
            l2 = c.gammadash()
            dg2dx = c.dgamma_by_dcoeff()
            dl2dx = c.dgammadash_by_dcoeff()

            hg1g2 = self.ddJdg1dg2(g1,l1,g2,l2)
            hl1g2 = self.ddJdl1dg2(g1,l1,g2,l2)
            hg1l2 = self.ddJdg1dl2(g1,l1,g2,l2)
            hl1l2 = self.ddJdl1dl2(g1,l1,g2,l2)

            res[counter:counter+ndofs_bc[ii],:] = \
                   np.einsum('ijkl,ijm,kln->nm', hg1g2, dg1dx, dg2dx) \
                +  np.einsum('ijkl,ijm,kln->nm', hl1g2, dl1dx, dg2dx) \
                +  np.einsum('ijkl,ijm,kln->nm', hg1l2, dg1dx, dl2dx) \
                +  np.einsum('ijkl,ijm,kln->nm', hl1l2, dl1dx, dl2dx)
            counter += ndofs_bc[ii]

        # /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\
        # This is considering all coils independent from one another... Forgot to apply symmetry!
        # /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\

        return res
        



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

    def ddJ_dportdcoil(self):
        return 0 #Does not depend on coils

