from deprecated import deprecated

from .._core.util import ObjectiveFailure
import numpy as np
from jax import grad
import jax.numpy as jnp
from jax.lax import dynamic_slice

from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp

__all__ = ['CurveLength', 'LpCurveCurvature', 'LpCurveTorsion',
           'CurveCurveDistance', 'CurveSurfaceDistance', 'ArclengthVariation',
           'MeanSquaredCurvature', 'LinkingNumber', 'CurveCylinderDistance', 'PortSize']


@jit
def curve_length_pure(l):
    """
    This function is used in a Python+Jax implementation of the curve length formula.
    """
    return jnp.mean(l)


class CurveLength(Optimizable):
    r"""
    CurveLength is a class that computes the length of a curve, i.e.

    .. math::
        J = \int_{\text{curve}}~dl.

    """

    def __init__(self, curve):
        self.curve = curve
        self.thisgrad = jit(lambda l: grad(curve_length_pure)(l))
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return curve_length_pure(self.curve.incremental_arclength())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """

        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    """
    This function is used in a Python+Jax implementation of the curvature penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)


class LpCurveCurvature(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the curve's curvature, and penalizes where the local curve curvature exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\kappa - \kappa_0, 0)^p ~dl

    where :math:`\kappa_0` is a threshold curvature, given by the argument ``threshold``.
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.kappa(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


def coil_normal_distance_pure(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot):
    p = 4
    hv = lambda x: logistic(x, k=4)
    min = lambda x: np.sum(x**(-p))**(-1./p)
    max = lambda x: np.sum(x**( p))**( 1./p)

    return coil_normal_distance(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot, hv, min, max, False)

def coil_normal_distance(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot, hv, min, max, return_pos ):
    """ Return largest circular port access size, with normal access

    For each quadrature point on the surface, evaluate the smallest distance
    from the coils to the line normal to the surface at this point. Return then
    largest of these minimum radii.

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
    test = hv(max_size-dd) * (hv(-x[:,2])*hv(x[:,0]-bn)+hv(x[:,2])*hv(x[:,0]-tn))

    # we shift test and multiply it by a large number, so that if the point does not satisfy test, we associate to it a large distance
    distances = distances.at[1:].set( (-1e2*(test-1) +1) * dd )
    counter += 1

    # p-norm to approximate minimum. p should be a parameter?
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
    def __init__(self, curves, surf):
        self.curves = curves

        # Every surface dependent quantity is only evaluated once for speed
        self.update_surface(surf)

        # I don't know how to get derivatives of unit_normal wrt boundary dofs; for now this class only depends on curves. Should not be used in any kind of combined approach, excepted if finite differences are implemented
        # Also, if the port size is made dependent on the surface, the objective function might not be differentiable anymore. 
        super().__init__(depends_on=curves)

        self.J_jax=lambda gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot: coil_normal_distance_pure(gamma_curves, gamma_surf, unit_normal, unit_tangent, phi_ind, theta_ind, max_sizes, ftop, fbot)
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
        # Only consider points on the outboard side
        self.gamma_boundary = self.boundary.gamma()
        nphi, ntheta, _ = self.gamma_boundary.shape
        if np.mod(nphi,3)!=0:
            raise ValueError('BOundary should have quadpoints_phi be a multiple of 3')
        nphi = int(nphi/3)

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
        for iphi, itheta in zip(self.phi_ind, self.theta_ind):
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
            new_nphi = phi_end-phi_start
            surf = self.gamma_boundary[phi_start:phi_end,:,:]
            gamma_surf_proj = np.einsum('ij,lmj->lmi', invM, surf - xsurf[None,None,:]) # This is the section of the surface projected on the tangent plane (i.e. in coord system (n,t,b)) 

            # Now find the envelop. This is an approximation, as for very twisted surfaces, toroidal planes can overlap in the (t,b) plane, and taking the max/min b-coordinate of each toroidal plane does not necessarily envelop the full projected surface. 
            xyzbot = np.zeros((new_nphi,3))
            xyztop = np.zeros((new_nphi,3))
            for jj in range(new_nphi):
                itop = np.argmax(gamma_surf_proj[jj,:,2]) 
                ibot = np.argmin(gamma_surf_proj[jj,:,2]) 
                xyzbot[jj,:] = gamma_surf_proj[jj,ibot,:]
                xyztop[jj,:] = gamma_surf_proj[jj,itop,:]

            # Evaluate closest point from the envelop. This sets the maximum port size
            tt = np.append(xyztop,xyzbot,axis=0)
            self.max_port_size[iphi,itheta] = np.min(tt[:,1]**2 + tt[:,2]**2)

            # Interpolate lower and upper envelop - this is a linear interpolation, does it keep the C1 property? I don't think so. Smooth interpolation would be better
            #self.bot_fits[iphi,itheta] = lambda x: np.interp(x, xyzbot[:,1], xyzbot[:,2] )
            #self.top_fits[iphi,itheta] = lambda x: np.interp(x, xyztop[:,1], xyztop[:,2] )
            self.bot_fits[iphi,itheta] = np.polyfit(xyzbot[:,1], xyzbot[:,2], deg)
            self.top_fits[iphi,itheta] = np.polyfit(xyztop[:,1], xyztop[:,2], deg)


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
         



@jit
def Lp_torsion_pure(torsion, gammadash, p, threshold):
    """
    This function is used in a Python+Jax implementation of the formula for the torsion penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(jnp.abs(torsion)-threshold, 0)**p * arc_length)


class LpCurveTorsion(Optimizable):
    r"""
    LpCurveTorsion is a class that computes a penalty term based on the :math:`L_p` norm
    of the curve's torsion:

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(|\tau|-\tau_0, 0)^p ~dl.

    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.thisgrad1 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula.
    """
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1)[:, None] * jnp.linalg.norm(l2, axis=1)[None, :]
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])

class CurveCurveDistance(Optimizable):
    r"""
    CurveCurveDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} d_{i,j}

    where 

    .. math::
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coils :math:`i` and :math:`j`, respectively.
    :math:`d_\min` is a desired threshold minimum intercoil distance.  This penalty term is zero when the points on coil :math:`i` and 
    coil :math:`j` lie more than :math:`d_\min` away from one another, for :math:`i, j \in \{1, \cdots, \text{num_coils}\}`

    If num_basecurves is passed, then the code only computes the distance to
    the first `num_basecurves` many curves, which is useful when the coils
    satisfy symmetries that can be exploited.

    """

    def __init__(self, curves, minimum_distance, num_basecurves=None):
        self.curves = curves
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gamma1, l1, gamma2, l2: cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance))
        self.thisgrad0 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2))
        self.thisgrad1 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2))
        self.thisgrad2 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2))
        self.thisgrad3 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2))
        self.candidates = None
        self.num_basecurves = num_basecurves or len(curves)
        super().__init__(depends_on=curves)

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
                [c.gamma() for c in self.curves], self.minimum_distance, self.num_basecurves)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), self.curves[j].gamma())) for i, j in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        return min([np.min(cdist(self.curves[i].gamma(), self.curves[j].gamma())) for i in range(len(self.curves)) for j in range(i)])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            res += self.J_jax(gamma1, l1, gamma2, l2)

        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]

        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gamma1, l1, gamma2, l2)
            dgamma_by_dcoeff_vjp_vecs[j] += self.thisgrad2(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[j] += self.thisgrad3(gamma1, l1, gamma2, l2)

        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}


def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(ns, axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)


class CurveSurfaceDistance(Optimizable):
    r"""
    CurveSurfaceDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} d_{i}

    where

    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. :math:`d_\min` is a desired threshold
    minimum coil-to-surface distance.  This penalty term is zero when the
    points on all coils :math:`i` and on the surface lie more than
    :math:`d_\min` away from one another.

    """

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gammac, lc, gammas, ns: cs_distance_pure(gammac, lc, gammas, ns, minimum_distance))
        self.thisgrad0 = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=0)(gammac, lc, gammas, ns))
        self.thisgrad1 = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=1)(gammac, lc, gammas, ns))
        self.candidates = None
        super().__init__(depends_on=curves)  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves], [self.surface.gamma().reshape((-1, 3))], self.minimum_distance)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i, _ in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i in range(len(self.curves))])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammas, ns)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        gammas = self.surface.gamma().reshape((-1, 3))

        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gammac, lc, gammas, ns)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gammac, lc, gammas, ns)
        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_arclengthvariation_pure(l, mat):
    """
    This function is used in a Python+Jax implementation of the curve arclength variation.
    """
    return jnp.var(mat @ l)


class ArclengthVariation(Optimizable):

    def __init__(self, curve, nintervals="full"):
        r"""
        This class penalizes variation of the arclength along a curve.
        The idea of this class is to avoid ill-posedness of curve objectives due to
        non-uniqueness of the underlying parametrization. Essentially we want to
        achieve constant arclength along the curve. Since we can not expect
        perfectly constant arclength along the entire curve, this class has
        some support to relax this notion. Consider a partition of the :math:`[0, 1]`
        interval into intervals :math:`\{I_i\}_{i=1}^L`, and tenote the average incremental arclength
        on interval :math:`I_i` by :math:`\ell_i`. This objective then penalises the variance

        .. math::
            J = \mathrm{Var}(\ell_i)

        it remains to choose the number of intervals :math:`L` that :math:`[0, 1]` is split into.
        If ``nintervals="full"``, then the number of intervals :math:`L` is equal to the number of quadrature
        points of the curve. If ``nintervals="partial"``, then the argument is as follows:

        A curve in 3d space is defined uniquely by an initial point, an initial
        direction, and the arclength, curvature, and torsion along the curve. For a
        :mod:`simsopt.geo.curvexyzfourier.CurveXYZFourier`, the intuition is now as
        follows: assuming that the curve has order :math:`p`, that means we have
        :math:`3*(2p+1)` degrees of freedom in total. Assuming that three each are
        required for both the initial position and direction, :math:`6p-3` are left
        over for curvature, torsion, and arclength. We want to fix the arclength,
        so we can afford :math:`2p-1` constraints, which corresponds to
        :math:`L=2p`.

        Finally, the user can also provide an integer value for `nintervals`
        and thus specify the number of intervals directly.
        """
        super().__init__(depends_on=[curve])

        assert nintervals in ["full", "partial"] \
            or (isinstance(nintervals, int) and 0 < nintervals <= curve.gamma().shape[0])
        self.curve = curve
        nquadpoints = len(curve.quadpoints)
        if nintervals == "full":
            nintervals = curve.gamma().shape[0]
        elif nintervals == "partial":
            from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
            if isinstance(curve, CurveXYZFourier) or isinstance(curve, JaxCurveXYZFourier):
                nintervals = 2*curve.order
            else:
                raise RuntimeError("Please provide a value other than `partial` for `nintervals`. We only have a default for `CurveXYZFourier` and `JaxCurveXYZFourier`.")

        self.nintervals = nintervals
        indices = np.floor(np.linspace(0, nquadpoints, nintervals+1, endpoint=True)).astype(int)
        mat = np.zeros((nintervals, nquadpoints))
        for i in range(nintervals):
            mat[i, indices[i]:indices[i+1]] = 1/(indices[i+1]-indices[i])
        self.mat = mat
        self.thisgrad = jit(lambda l: grad(lambda x: curve_arclengthvariation_pure(x, mat))(l))

    def J(self):
        return float(curve_arclengthvariation_pure(self.curve.incremental_arclength(), self.mat))

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_msc_pure(kappa, gammadash):
    """
    This function is used in a Python+Jax implementation of the mean squared curvature objective.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length)/jnp.mean(arc_length)


class MeanSquaredCurvature(Optimizable):

    def __init__(self, curve):
        r"""
        Compute the mean of the squared curvature of a curve.

        .. math::
            J = (1/L) \int_{\text{curve}} \kappa^2 ~dl

        where :math:`L` is the curve length, :math:`\ell` is the incremental
        arclength, and :math:`\kappa` is the curvature.

        Args:
            curve: the curve of which the curvature should be computed.
        """
        super().__init__(depends_on=[curve])
        self.curve = curve
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=1)(kappa, gammadash))

    def J(self):
        return float(curve_msc_pure(self.curve.kappa(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)


@deprecated("`MinimumDistance` has been deprecated and will be removed. Please use `CurveCurveDistance` instead.")
class MinimumDistance(CurveCurveDistance):
    pass


class LinkingNumber(Optimizable):

    def __init__(self, curves):
        Optimizable.__init__(self, depends_on=curves)
        self.curves = curves
        r"""
        Compute the Linking number of a set of curves (whether the curves 
        are interlocked or not).

        The value is 1 if the are interlocked, 0 if not.
        
        .. math::
            Link(c1,c2) = \frac{1}{4\pi} \oint_{c1}\oint_{c2}\frac{\textbf{R1} - \textbf{R2}}{|\textbf{R1}-\textbf{R2}|^3} (d\textbf{R1} \times d\textbf{R2})
            
        where :math:`c1` is the first curve and :math:`c2` is the second curve, 
        :math:`\textbf{R1}` is the radius vector of the first curve, and 
        :math:`\textbf{R2}` is the radius vector of the second curve

        Args:
            curves: the set of curves on which the linking number should be computed.
        
        """

    def J(self):
        ncoils = len(self.curves)
        linkNum = np.zeros([ncoils + 1, ncoils + 1])
        i = 0
        for c1 in self.curves[:(ncoils + 1)]:
            j = 0
            i = i + 1
            for c2 in self.curves[:(ncoils + 1)]:
                j = j + 1
                if i < j:
                    R1 = c1.gamma()
                    R2 = c2.gamma()
                    dS = c1.quadpoints[1] - c1.quadpoints[0]
                    dT = c2.quadpoints[1] - c1.quadpoints[0]
                    dR1 = c1.gammadash()
                    dR2 = c2.gammadash()

                    integrals = sopp.linkNumber(R1, R2, dR1, dR2) * dS * dT
                    linkNum[i-1][j-1] = 1/(4*np.pi) * (integrals)
        linkNumSum = sum(sum(abs(linkNum)))
        return linkNumSum

    @derivative_dec
    def dJ(self):
        return Derivative({})
    

@jit
def curve_cylinder_distance_pure(g, R0):
    R = jnp.sqrt(g[:,0]**2 + g[:,1]**2)
    npts = g.shape[0]
    return jnp.sqrt(jnp.sum((R-R0)**2)) / npts

@jit
def curve_centroid_cylinder_distance_pure(g, R0):
    c = jnp.mean(g,axis=0)
    Rc = jnp.sqrt(c[0]**2+c[1]**2)
    return Rc-R0

class CurveCylinderDistance(Optimizable):
    def __init__(self, curve, R0, mth='pts'):
        self.curve = curve
        self.R0 = R0
        self.mth = mth

        if mth=='pts':
            self._Jlocal = curve_cylinder_distance_pure
        elif mth=='ctr':
            self._Jlocal = curve_centroid_cylinder_distance_pure
        else:
            raise ValueError('Unknown mth')
        
        self.thisgrad = jit(lambda g, R: grad(self._Jlocal, argnums=0)(g, R))
        super().__init__(depends_on=[curve])

    def J(self):
        return self._Jlocal(self.curve.gamma(), self.R0)
    
    @derivative_dec
    def dJ(self):
        grad0 = self.thisgrad(self.curve.gamma(), self.R0)
        return self.curve.dgamma_by_dcoeff_vjp(grad0)
