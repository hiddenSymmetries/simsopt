import numpy as np
from .._core.util import parallel_loop_bounds

__all__ = ['initialize_position_uniform']

def initialize_position_uniform(field, nparticles, nfp=1, ns_max=100, ntheta_max=100, nzeta_max=100, 
                                comm=None, seed=None): 
    r"""
    Initialize particle positions uniformly with respect to the volume element in Boozer coordinates.
    The volume element in Boozer coordinates is given by the Jacobian :math:`\sqrt{g} = (G + \iota I)/(|B|^2)`. 

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
    np.random.seed(seed)
    # Compute max value of Jacobian on a grid for normalization 
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
    maxJ = np.max(J)

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
                J = J/maxJ # Normalize

                if (rand1 <= J):
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