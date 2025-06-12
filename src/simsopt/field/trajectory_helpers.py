import numpy as np
import matplotlib.pyplot as plt

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

    timelost = []
    for ip in range(nparticles):
        timelost.append(res_tys[ip][-1,0])
    times = np.logspace(np.log(tmin),np.log(tmax),ntime)

    loss_frac = np.zeros_like(times)
    for it in range(len(times)):
        loss_frac[it] = np.count_nonzero(timelost < times[it])/nparticles

    return times, loss_frac

def compute_trajectory_cylindrical(field, gc_tys): 
    r"""
    Compute the cylindrical coordinates (R, Z, phi) in a given BoozerMagneticField for each particle trajectory.

    Args:
        res_tys : List of particle trajectories, where each trajectory is a 2D array with shape (nsteps, 5) 
                 containing time and coordinates (t, s, theta, zeta, vpar).
        field : The :class:`BoozerMagneticField` instance used to set the points for the field.
    
    Returns:
        R_trajs : A list of numpy arrays, where each array has shape (nsteps,) containing the radial coordinate R for each particle.
        Z_trajs : A list of numpy arrays, where each array has shape (nsteps,) containing the vertical coordinate Z for each particle.
        phi_trajs : A list of numpy arrays, where each array has shape (nsteps,) containing the azimuthal angle phi for each particle.
    """
    nparticles = len(gc_tys)

    R_trajs = []
    Z_trajs = []
    phi_trajs = []
    for ip in range(nparticles):
        nsteps = len(gc_tys[0])
        points = np.zeros((nsteps, 3))
        points[:, 0] = gc_tys[ip][:, 1]
        points[:, 1] = gc_tys[ip][:, 2]
        points[:, 2] = gc_tys[ip][:, 3]
        field.set_points(points)
        R_trajs.append(field.R())
        Z_trajs.append(field.Z())
        nu = field.nu()
        phi_trajs.append(gc_tys[ip][:, 3] - nu)

    return R_trajs, Z_trajs, phi_trajs