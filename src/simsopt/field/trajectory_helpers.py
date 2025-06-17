import numpy as np
from simsopt.field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion, MinToroidalFluxStoppingCriterion

__all__ = ['compute_loss_fraction','compute_trajectory_cylindrical','passing_map']

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

def passing_map(point,field,mass,charge,Ekin,tmax=1e-2,solver_options={}):
    r"""
    Given the coordinates (s,theta,vpar) at zeta = 0, integrates the guiding
    center equations until the trajectory returns to the zeta = 0 plane. 
    We assume that the particle is passing, so if vpar crosses through 0 
    along the trajectory, a RuntimeError is raised. A RuntimeError is also
    raised if the particle leaves the s = 1 surface. The coordinates (s,theta,vpar)
    for the return map are returned.

    Args:
        point: A numpy array of shape (3,) containing the initial coordinates
            (s,theta,vpar).
        field: The :class:`BoozerMagneticField` instance used for integration
        mass: Particle mass.
        charge: Particle charge.
        Ekin: Particle total energy.
        tmax: Maximum time for integration (default: 1e-2 seconds).
        solver_options: A dictionary of options to pass to the ODE solver.
            See :func:`simsopt.field.tracing.trace_particles_boozer` for details.

    Returns:
        point: A numpy array of shape (3,) containing the coordinates
            (s,theta,vpar) when the trajectory returns to the zeta = 0 plane.
    """
    points = np.zeros((1,3))
    points[:,0] = point[0]
    points[:,1] = point[1]
    points[:,2] = 0

    # Set solver options needed for passing map 
    solver_options.setdefault('vpars_stop',True)
    solver_options.setdefault('zetas_stop',True)
    res_tys, res_hits = trace_particles_boozer(field, points, [point[2]], tmax=tmax, mass=mass, charge=charge,
            Ekin=Ekin, zetas=[0], vpars=[0], omegas=[0], stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=False,solver_options=solver_options)
    res_hit = res_hits[0][0,:] # Only check the first hit or stopping criterion

    if (res_hit[1] == 0): # Check that the zetas=[0] plane was hit
        point[0] = res_hit[2]
        point[1] = res_hit[3]
        point[2] = res_hit[5]
        return point
    else: 
        raise RuntimeError('Alternative stopping criterion reached in passing_map.') 

def trajectory_to_vtk(res_ty,field,filename='trajectory'):
    r"""
    Save a single particle trajectory in Cartesian coordinates to a VTK file.
    Requires the pyevtk package to be installed.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the trajectory of a 
                 single particle in Boozer coordinates. 
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        filename : The name of the output VTK file. 
    """
    from pyevtk.hl import polyLinesToVTK
    R_traj, phi_traj, Z_traj = compute_trajectory_cylindrical(res_ty, field)

    X_traj = R_traj * np.cos(phi_traj)
    Y_traj = R_traj * np.sin(phi_traj)

    ppl = np.asarray([len(R_traj)]) # Number of points along trajectory 
    polyLinesToVTK(filename, X_traj, Y_traj, Z_traj, pointsPerLine=ppl)