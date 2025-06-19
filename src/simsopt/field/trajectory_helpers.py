import numpy as np

from ..field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion, MinToroidalFluxStoppingCriterion
from .._core.util import parallel_loop_bounds
from ..util.functions import proc0_print

__all__ = ['compute_loss_fraction','compute_trajectory_cylindrical','PassingPoincare','trajectory_to_vtk']

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
class PassingPoincare:
    """
    Class to compute and store passing Poincare maps and related quantities for a given BoozerMagneticField.
    """

    def __init__(self, field, lam, sign_vpar, mass, charge, Ekin, ns_poinc=120, ntheta_poinc=2, Nmaps=500,
                 comm=None, tmax=1e-2, solver_options={}):
        """
        Initialize and compute the passing Poincare map, evaluated by integrating the guiding
        center equations until the trajectory returns to the zeta = 0 plane. 
        We assume that the particle is passing, so the parallel velocity does not change sign. 

        Args:
            field : The BoozerMagneticField instance used for integration.
            lam : The pitch-angle variable, lambda = v_perp^2/(v^2 B).
            sign_vpar : The sign of the parallel velocity. Should be either +1 or -1.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total energy.
            ns_poinc : Number of initial conditions in s for Poincare plot (default: 120).
            ntheta_poinc : Number of initial conditions in theta for Poincare plot (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver (default: {}).
        """
        if (sign_vpar not in [-1, 1]):
            raise ValueError("sign_vpar should be either -1 or +1")
        
        self.field = field
        self.lam = lam
        self.sign_vpar = sign_vpar
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.ns_poinc = ns_poinc
        self.ntheta_poinc = ntheta_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        self.s_all, self.thetas_all, self.vpars_all, self.t_all = self.compute_passing_map()

    def initialize_passing_map(self):
        r"""
        Given a :class:`BoozerMagneticField` instance, this function generates initial positions
        for the passing Poincare return map. Particles are initialized on the zeta = 0 plane 
        such that the parallel velocity is consistent with the prescribed total energy and pitch-angle 
        variable, :math:`\lambda = v_\perp^2/(v^2 B)`. 

        Returns: 
            s_init : List of initial s coordinates for the Poincare map.
            thetas_init : List of initial theta coordinates for the Poincare map.
            vpars_init : List of initial parallel velocities for the Poincare map.
        """
        s = np.linspace(0,1,self.ns_poinc+1,endpoint=False)[1::]
        thetas = np.linspace(0,2*np.pi,self.ntheta_poinc)
        s, thetas = np.meshgrid(s, thetas)
        s = s.flatten()
        thetas = thetas.flatten()

        vtotal = np.sqrt(2 * self.Ekin / self.mass)  # Total velocity from kinetic energy

        def vpar_func(s,theta):
            point = np.zeros((1,3))
            point[0,0] = s
            point[0,1] = theta
            self.field.set_points(point)
            modB = self.field.modB()[0,0]
            # Skip any trapped particles
            if (1 - self.lam*modB < 0):
                return None
            else:
                return self.sign_vpar*vtotal*np.sqrt(1 - self.lam*modB)

        first, last = parallel_loop_bounds(self.comm, len(s))
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

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            thetas_init = [i for o in self.comm.allgather(thetas_init) for i in o]
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]

        return s_init, thetas_init, vpars_init

    def passing_map(self, point):
        r"""
        Given the coordinates (s,theta,vpar) at zeta = 0, integrates the guiding
        center equations until the trajectory returns to the zeta = 0 plane. 
        We assume that the particle is passing, so if vpar crosses through 0 
        along the trajectory, a RuntimeError is raised. A RuntimeError is also
        raised if the particle leaves the s = 1 surface. The coordinates (s,theta,vpar)
        for the return map are returned.

        Args: 
            point : A numpy array of shape (3,) containing the initial coordinates
                (s,theta,vpar).

        Returns:    
            point : A numpy array of shape (3,) containing the coordinates
                (s,theta,vpar) when the trajectory returns to the zeta = 0 plane.
            time : The time taken to return to the zeta = 0 plane.
        """

        points = np.zeros((1,3))
        points[:,0] = point[0]
        points[:,1] = point[1]
        points[:,2] = 0

        # Set solver options needed for passing map 
        self.solver_options.update({'vpars_stop':True, 'zetas_stop':True})
        res_tys, res_hits = trace_particles_boozer(self.field, points, [point[2]], tmax=self.tmax, mass=self.mass, charge=self.charge,
                Ekin=self.Ekin, zetas=[0], vpars=[0], omegas=[0], stopping_criteria=[MinToroidalFluxStoppingCriterion(0.01),MaxToroidalFluxStoppingCriterion(1.0)],
                forget_exact_path=False,solver_options=self.solver_options)
        res_hit = res_hits[0][0,:] # Only check the first hit or stopping criterion

        if (res_hit[1] == 0): # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = res_hit[3]
            point[2] = res_hit[5]
            time = res_hit[0] 
            return point, time 
        else: 
            raise RuntimeError('Alternative stopping criterion reached in passing_map.') 

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle positions.
        """
        self.s_init, self.thetas_init, self.vpars_init = self.initialize_passing_map()
        Ntrj = len(self.s_init)

        s_all = []
        thetas_all = []
        vpars_all = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first,last):
            tr = [self.s_init[itrj],self.thetas_init[itrj],self.vpars_init[itrj]]
            s_traj = [tr[0]]
            thetas_traj = [tr[1]]
            vpars_traj = [tr[2]]
            t_traj = [0]
            for jj in range(self.Nmaps):
                try:
                    tr, time = self.passing_map(tr)
                    s_traj.append(self.s_init[itrj])
                    thetas_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    t_traj.append(time)
                except RuntimeError:
                    break
            s_all.append(s_traj)
            thetas_all.append(thetas_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            thetas_all = [i for o in self.comm.allgather(thetas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, thetas_all, vpars_all, t_all

    def compute_frequencies(self):
        """
        Compute the passing particle poloidal and toroidal transit frequencies and mean radial position.
        The frequency is only computed if the trajectory has completed at least one full Poincare return map 
        before the trajectory is terminated due to a time limit or stopping criterion. Since the frequency 
        will depend on field-line label and trapping well in a general 3D field, an average is taken over
        all trajectories initialized on the same flux surface and all Poincare return maps along each trajectory.
        Since the frequency profile will flatten in a phase-space island, it is sometimes easier to interpret the
        frequency profile in a field with enforced quasisymmetry (i.e., initialize BoozerRadialInterpolant with N
        prescribed).

        Returns:
            omega_theta : List of poloidal transit frequencies.
            omega_zeta : List of toroidal transit frequencies.
            init_s : List of initial s values for each trajectory.
        """
        if self.solver_options['axis'] != 0:
            raise ValueError('ODE solver must integrate with solver_options["axis"]=0 to compute passing frequencies.')
            
        self.field.set_points(np.array([[1],[0],[0]]).T)
        sign_G = np.sign(self.field.G()[0])

        omega_theta = []
        omega_zeta = []
        init_s = []
        for s_traj, theta_traj, vpar_traj, t_traj in zip(self.s_all, self.thetas_all, self.vpars_all, self.t_all):
            if len(s_traj) < 2: # Need at least one full Poincare return maps to compute frequency
                continue
            delta_theta = np.array(theta_traj[1::]) - np.array(theta_traj[0:-1])
            
            delta_t = np.array(t_traj[1::]) - np.array(t_traj[0:-1])
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G

            # Average over wells along one field line
            freq_theta = np.mean(delta_theta)/np.mean(delta_t)
            freq_zeta = delta_zeta/np.mean(delta_t)

            omega_theta.append(freq_theta)
            omega_zeta.append(freq_zeta)
            init_s.append(np.mean(s_traj))

        omega_theta = np.array(omega_theta)
        omega_zeta = np.array(omega_zeta)
        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_theta_prof = np.zeros((len(s_prof),))
        omega_zeta_prof = np.zeros((len(s_prof),))

        # Average over field-line label 
        for i, s in enumerate(s_prof):
            omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s==s)])
            omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s==s)])

        return omega_theta_prof, omega_zeta_prof, s_prof

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, thetas_all, vpars_all, t_all : Lists of trajectory data.
        """
        return self.s_all, self.thetas_all, self.vpars_all, self.t_all
    
    def plot_poincare(self,ax=None,filename='passing_poincare.pdf'):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only 
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are created.
            filename : Name of the file to save the plot (default: 'passing_poincare.pdf').
        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib
        matplotlib.use('Agg') # Don't use interactive backend
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$s$')
        ax.set_xlim([0,2*np.pi])
        ax.set_ylim([0,1])
        for i in range(len(self.thetas_all)):
            ax.scatter(np.mod(self.thetas_all[i],2*np.pi), self.s_all[i], marker='o',s=0.5,edgecolors='none')
            plt.savefig(filename)

        return ax 

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