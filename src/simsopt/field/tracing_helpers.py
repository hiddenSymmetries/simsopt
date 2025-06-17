import numpy as np
from .._core.util import parallel_loop_bounds

__all__ = ['initialize_position_uniform_surf','initialize_position_profile','initialize_position_uniform_vol']

def initialize_position_uniform_surf(field, nparticles, s, nfp=1, ntheta_max=100, nzeta_max=100, comm=None, seed=None):
    r"""
    Initialize particle positions on a given magnetic surface, s, uniformly with respect to the volume element 
    in Boozer coordinates 

    Args:
        field : The :class:`BoozerMagneticField` instance
        nparticles : Number of particles to initialize
        s: Normalized toroidal flux for surface to initialize particles. 
        nfp: Number of field periods used to initialize particles in range [0, 2*pi/nfp)] (default: 1)
        ns_max : Number of points in s direction for Jacobian computation
        ntheta_max : Number of points in theta direction for Jacobian computation
        nzeta_max : Number of points in zeta direction for Jacobian computation
        comm : MPI communicator (default: None)
        seed: Random seed for reproducibility (default: None, uses random seed from numpy)

    Returns: 
        points: A numpy array of shape (nparticles, 3) containing the initialized particle positions in Boozer 
                coordinates (s, theta, zeta).
    """
    np.random.seed(seed)
    # Compute max value of Jacobian on a grid for normalization 
    theta_grid = np.linspace(0,2*np.pi,ntheta_max,endpoint=False)
    zeta_grid = np.linspace(0,2*np.pi/nfp,nzeta_max,endpoint=False)
    [zeta_grid,theta_grid] = np.meshgrid(zeta_grid,theta_grid)
    points = np.zeros((len(theta_grid.flatten()),3))
    points[:,0] = s
    points[:,1] = theta_grid.flatten()
    points[:,2] = zeta_grid.flatten()

    field.set_points(points)
    G = field.G()
    iota = field.iota()
    I = field.I()
    modB = field.modB()
    J = (G + iota*I)/(modB**2)

    J_max = np.max(J)

    theta_init = []
    zeta_init = []
    s_init = []
    points = np.zeros((1,3))

    first, last = parallel_loop_bounds(comm,nparticles)
    for i in range(first,last):
            while True:
                rand1 = np.random.uniform(0,1,None)
                theta = np.random.uniform(0,2*np.pi,None)
                zeta  = np.random.uniform(0,2*np.pi/nfp,None)
                points[:,0] = s
                points[:,1] = theta
                points[:,2] = zeta
                field.set_points(points)
                J = (field.G()[0,0] + field.iota()[0,0]*field.I()[0,0])/(field.modB()[0,0]**2)

                # Normalize the Jacobian
                prob_norm = J / J_max 

                if (rand1 <= prob_norm):
                    s_init.append(s)
                    theta_init.append(theta)
                    zeta_init.append(zeta)
                    break
 
    # Gather all particle positions across all processes
    if comm is not None: 
        s_init = [i for o in comm.allgather(s_init) for i in o]
        theta_init = [i for o in comm.allgather(theta_init) for i in o]
        zeta_init = [i for o in comm.allgather(zeta_init) for i in o]

    points = np.zeros((nparticles,3))
    points[:,0] = np.asarray(s_init)
    points[:,1] = np.asarray(theta_init)
    points[:,2] = np.asarray(zeta_init)

    return points    

def initialize_position_profile(field, nparticles, profile, nfp=1, ns_max=100, ntheta_max=100, nzeta_max=100, 
                                comm=None, seed=None): 
    r"""
    Initialize particle positions from probability density function :math:`p(s)` define by a profile function of the normalized
    flux, :math:`s`. Typically, this is used for sampling from the fusion birth distribution,

    .. math:: 
        p(s) = n_D(s) n_T(s) \langle \sigma v \rangle(s) 

    where :math:`n_D(s)` and :math:`n_T(s)` are the deuterium and tritium density profiles, respectively, and
    :math:`\langle \sigma v \rangle(s)` is the cross section profile. 
    
    See Bader, A., et al. "Modeling of energetic particle transport in optimized stellarators." Nuclear Fusion 61.11 (2021): 116060.

    Note that :math:`p(s)` need not be normalized: normalization is performed internally by computing the maximum value of the probabiliy on 
    the grid provided by `ns_max`. 
    
    Args:
        field : The :class:`BoozerMagneticField` instance
        nparticles : Number of particles to initialize
        profile: A function that takes a single argument (the normalized flux `s`) and returns the probability profile value at that point.
        nfp: Number of field periods used to initialize particles in range [0, 2*pi/nfp)] (default: 1)
        ns_max : Number of points in s direction for Jacobian computation
        ntheta_max : Number of points in theta direction for Jacobian computation
        nzeta_max : Number of points in zeta direction for Jacobian computation
        comm : MPI communicator (default: None)
        seed: Random seed for reproducibility (default: None, uses random seed from numpy)

    Returns: 
        points: A numpy array of shape (nparticles, 3) containing the initialized particle positions in Boozer 
                coordinates (s, theta, zeta).
    """
    if not callable(profile):
        raise ValueError("The profile must be a callable function that takes a single argument (s) and returns a value.")

    np.random.seed(seed)
    # Compute max value of Jacobian and p on a grid for normalization 
    s_grid = np.linspace(0,1,ns_max)
    theta_grid = np.linspace(0,2*np.pi,ntheta_max,endpoint=False)
    zeta_grid = np.linspace(0,2*np.pi/nfp,nzeta_max,endpoint=False)
    [zeta_grid,theta_grid,s_grid] = np.meshgrid(zeta_grid,theta_grid,s_grid)
    points = np.zeros((len(theta_grid.flatten()),3))
    points[:,0] = s_grid.flatten()
    points[:,1] = theta_grid.flatten()
    points[:,2] = zeta_grid.flatten()

    field.set_points(points)
    G = field.G()
    iota = field.iota()
    I = field.I()
    modB = field.modB()
    J = (G + iota*I)/(modB**2)

    # Compute normalized profile values on the grid 
    profile_values = np.array([profile(s) for s in s_grid.flatten()])
    prob_max = np.max(J[:,0] * profile_values)  # Normalize by the maximum value of J * profile

    theta_init = []
    zeta_init = []
    s_init = []
    points = np.zeros((1,3))

    first, last = parallel_loop_bounds(comm,nparticles)
    for i in range(first,last):
            while True:
                rand1 = np.random.uniform(0,1,None)
                s = np.random.uniform(0,1.0,None)
                theta = np.random.uniform(0,2*np.pi,None)
                zeta  = np.random.uniform(0,2*np.pi/nfp,None)
                points[:,0] = s
                points[:,1] = theta
                points[:,2] = zeta
                field.set_points(points)
                J = (field.G()[0,0] + field.iota()[0,0]*field.I()[0,0])/(field.modB()[0,0]**2)

                # Normalize the probability 
                prob_norm = J * profile(s) / prob_max 

                if (rand1 <= prob_norm):
                    s_init.append(s)
                    theta_init.append(theta)
                    zeta_init.append(zeta)
                    break
 
    # Gather all particle positions across all processes
    if comm is not None: 
        s_init = [i for o in comm.allgather(s_init) for i in o]
        theta_init = [i for o in comm.allgather(theta_init) for i in o]
        zeta_init = [i for o in comm.allgather(zeta_init) for i in o]

    points = np.zeros((nparticles,3))
    points[:,0] = np.asarray(s_init)
    points[:,1] = np.asarray(theta_init)
    points[:,2] = np.asarray(zeta_init)

    return points 

def initialize_position_uniform_vol(field, nparticles, nfp=1, ns_max=100, ntheta_max=100, nzeta_max=100, 
                                comm=None, seed=None): 
    r"""
    Initialize particle positions uniformly in the plasma volume with respect to the volume element in 
    Boozer coordinates.

    Args:
        field : The :class:`BoozerMagneticField` instance
        nparticles : Number of particles to initialize
        nfp: Number of field periods used to initialize particles in range [0, 2*pi/nfp)] (default: 1)
        ns_max : Number of points in s direction for Jacobian computation
        ntheta_max : Number of points in theta direction for Jacobian computation
        nzeta_max : Number of points in zeta direction for Jacobian computation
        comm : MPI communicator (default: None)
        seed: Random seed for reproducibility (default: None, uses random seed from numpy)

    Returns: 
        points: A numpy array of shape (nparticles, 3) containing the initialized particle positions in Boozer 
                coordinates (s, theta, zeta).
    """
    return initialize_position_profile(
        field, nparticles, lambda s: 1.0, nfp=nfp, ns_max=ns_max, ntheta_max=ntheta_max, nzeta_max=nzeta_max, 
        comm=comm, seed=seed
    )

def initialize_passing_map(field,lam,vtotal,sign_vpar,ns_poinc=120,ntheta_poinc=2,comm=None):
    r"""
    Given a :class:`BoozerMagneticField` instance, this function generates initial positions
    for the passing Poincare return map.

    Args:
        field : The :class:`BoozerMagneticField` instance
        lam: Lambda value for the passing map, defined as :math:`\lambda = v_{\perp}^2/(v^2 B)` 
            where :math:`v` is the total velocity. 
        vtotal: Total velocity of the particles
        sign_vpar: Sign of the parallel velocity. 
        ns_poinc: Number of points in s direction for Poincare map (default: 120)
        ntheta_poinc: Number of points in theta direction for Poincare map (default: 2)
        comm: MPI communicator used for evaluation of initial conditions (default: None)
    """
    s = np.linspace(0,1,ns_poinc+1,endpoint=False)[1::]
    thetas = np.linspace(0,2*np.pi,ntheta_poinc)
    s, thetas = np.meshgrid(s, thetas)
    s = s.flatten()
    thetas = thetas.flatten()

    def vpar_func(s,theta):
        point = np.zeros((1,3))
        point[0,0] = s
        point[0,1] = theta
        field.set_points(point)
        modB = field.modB()[0,0]
        # Skip any trapped particles
        if (1 - lam*modB < 0):
            return None
        else:
            return sign_vpar*vtotal*np.sqrt(1 - lam*modB)

    first, last = parallel_loop_bounds(comm, len(s))
    # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
    s_init = []
    thetas_init = []
    vpars_init = []
    for i in range(first,last):
        vpar = vpar_func(s[i],thetas[i])
        if vpar is not None:
            s_init.append(s[i])
            thetas_init.append(thetas[i])
            vpars_init.append(vpar)

    if comm is not None:
        s_init = [i for o in comm.allgather(s_init) for i in o]
        thetas_init = [i for o in comm.allgather(thetas_init) for i in o]
        vpars_init = [i for o in comm.allgather(vpars_init) for i in o]

    return s_init, thetas_init, vpars_init 

