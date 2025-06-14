import numpy as np

__all__ = ['compute_loss_fraction','compute_trajectory_cylindrical']

def compute_loss_fraction(res_tys, tmin=1e-7, tmax=1e-2, ntime=1000):
    r"""
    Compute the fraction of particles lost as a function of time. 

    Args:
        res_tys : List of particle trajectories, where each trajectory is a 2D array with shape (nsteps, 5) 
                 containing time and coordinates (t, s, theta, zeta, vpar).
        tmin : Minimum time to consider for loss fraction (default: 1e-7)
        tmax : Maximum time to consider for loss fraction (default: 1e-2)
        ntime : Number of time points to evaluate the loss fraction (default: 1000)
    Returns:
        times : A numpy array of shape (ntime,) containing the time points at which the loss fraction is evaluated.
        loss_frac : A numpy array of shape (ntime,) containing the fraction of particles lost at each time point.
    """
    nparticles = len(res_tys)

    timelost = np.zeros((nparticles,))
    for ip in range(nparticles):
       timelost[ip] = res_tys[ip][-1,0]
        
    times = np.logspace(np.log10(tmin),np.log10(tmax),ntime)

    loss_frac = np.zeros_like(times)
    for it in range(ntime):
        loss_frac[it] = np.count_nonzero(timelost < times[it])/nparticles

    return times, loss_frac

def compute_trajectory_cylindrical(res_ty, field): 
    r"""
    Compute the cylindrical coordinates (R, Z, phi) in a given BoozerMagneticField for each particle trajectory.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the trajectory of a single particle in Boozer coordinates 
                (s, theta, zeta, vpar). Tracing should be performed with forget_exact_path=False to save the trajectory
                information. 
        field : The :class:`BoozerMagneticField` instance used to set the points for the field.
    
    Returns:
        R_traj : A numpy array with shape (nsteps,) containing the radial coordinate R for the particle trajectory. 
        Z_traj : A numpy array with shape (nsteps,) containing the vertical coordinate Z for the particle trajectory. 
        phi_traj : A numpy array with shape (nsteps,) containing the azimuthal angle phi for the particle trajectory. 
    """
    nsteps = len(res_ty[:,0])
    points = np.zeros((nsteps, 3))
    points[:, 0] = res_ty[:, 1]
    points[:, 1] = res_ty[:, 2]
    points[:, 2] = res_ty[:, 3]
    field.set_points(points)

    R_traj = field.R()[:,0] 
    Z_traj = field.Z()[:,0] 
    nu = field.nu()[:,0] 
    phi_traj = res_ty[:, 3] - nu

    return R_traj, phi_traj, Z_traj