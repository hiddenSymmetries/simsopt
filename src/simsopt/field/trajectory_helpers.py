import numpy as np
from warnings import warn

from ..field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion, MinToroidalFluxStoppingCriterion
from .._core.util import parallel_loop_bounds

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
        if len(res_hits[0]) == 0:
            raise RuntimeError('No stopping criterion reached in passing_map.')

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
class TrappedPoincare:
    """
    Class to compute and store trapped Poincare maps and related quantities for a given BoozerMagneticField.
    """

    def __init__(self, field, helicity_M, helicity_N, s_mirror, theta_mirror, zeta_mirror, mass, charge, Ekin, ns_poinc=120, neta_poinc=2, Nmaps=500,
                 comm=None, tmax=1e-2, solver_options={}):
        """
        Initialize and compute the trapped Poincare map, evaluated by integrating the guiding
        center equations from the v_{\|} = 0 plane until the trajectory returns to the v_{\|} = 0 plane. 
        The field strength contours are assumed to have helicity (M,N) in Boozer coordinates. The mapping
        coordinate, eta, is chosen based on the helicity of the field strength contours: if M = 0 (e.g., QP or OP), 
        then eta=theta is used, if N = 0 (e.g., QA/OA, QH/OA), then eta = nfp*zeta is used. 

        The particle mu is selected by providing a single mirror point, (s_mirror, theta_mirror, zeta_mirror), where
        the parallel velocity is zero. The pitch-angle variable, :math:`\lambda = v_\perp^2/(v^2 B)`, is computed and held 
        fixed for all trajectories.

        Args:
            field : The BoozerMagneticField instance used for integration.
            helicity_M : The poloidal helicity of the magnetic field.
            helicity_N : The toroidal helicity of the magnetic field.
            s_mirror : The flux surface where the particle mirrors.
            theta_mirror : The poloidal angle where the particle mirrors.
            zeta_mirror : The toroidal angle where the particle mirrors.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total energy.
            ns_poinc : Number of initial conditions in s for Poincare plot (default: 120).
            neta_poinc : Number of initial conditions in theta for Poincare plot (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver (default: {}).
        """
        self.field = field

        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        # If modB contours close poloidally, then use theta as mapping coordinate 
        if self.helicity_M == 0:
            self.helicity_Mp = 1
            self.helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate 
        else:
            self.helicity_Mp = 0
            self.helicity_Np = self.field.nfp

        self.s_mirror = s_mirror
        self.theta_mirror = theta_mirror
        self.zeta_mirror = zeta_mirror
        field.set_points(np.array([[s_mirror],[theta_mirror],[zeta_mirror]]).T)
        self.modBcrit = field.modB()[0,0] # Magnetic field at mirror point
        self.lam = 1/self.modBcrit # lambda = v_perp^2/(v^2 B) = 1/modBcrit
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.ns_poinc = ns_poinc
        self.neta_poinc = neta_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        denom = self.helicity_Np*self.helicity_M - self.helicity_N*self.helicity_Mp
        self.dtheta_dchi = self.helicity_Np/denom 
        self.dzeta_dchi = self.helicity_Mp/denom

        self.s_all, self.chis_all, self.etas_all, self.t_all = self.compute_trapped_map()

    def chi(self,theta,zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M*theta - self.helicity_N*zeta
    
    def eta(self,theta,zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp*theta - self.helicity_Np*zeta
    
    def chi_eta_to_theta_zeta(self,chi,eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np*self.helicity_M - self.helicity_N*self.helicity_Mp
        theta = (self.helicity_Np*chi - self.helicity_N*eta)/denom  
        zeta = (self.helicity_Mp*chi - self.helicity_M*eta)/denom

        return theta, zeta

    def trapped_map(self,point):
        r"""
        Integrates the gc equations from one mirror point to the next mirror point.
        point contains the [s, theta, zeta] coordinates and returns the same coordinates
        after mapping.

        Args: 
            point : A numpy array of shape (3,) containing the initial coordinates
                (s,theta,zeta).   
        Returns:    
            point : A numpy array of shape (3,) containing the coordinates
                (s,theta,zeta) when the trajectory returns to the vpar = 0 plane.
            time : The time taken to return to the vpar = 0 plane.  
        """
        theta, zeta = self.chi_eta_to_theta_zeta(point[1],point[2])
        points = np.zeros((1,3))
        points[:,0] = point[0]
        points[:,1] = theta
        points[:,2] = zeta

        # Set solver options needed for passing map 
        self.solver_options.setdefault('vpars_stop',True)
        self.solver_options.setdefault('zetas_stop',False)
        res_tys, res_hits = trace_particles_boozer(self.field, points, [0], tmax=self.tmax, mass=self.mass, charge=self.charge,
                Ekin=self.Ekin, zetas=[], vpars=[0], stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                forget_exact_path=False,solver_options=self.solver_options)
        
        if len(res_hits[0]) == 0:
            raise RuntimeError('No stopping criterion reached in trapped_map.')
        
        res_hit = res_hits[0][0,:] # Only check the first hit or stopping criterion

        if (res_hit[1] == 0): # Check that the vpars=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3],res_hit[4])
            point[2] = self.eta(res_hit[3],res_hit[4])
            time = res_hit[0] 
            return point, time 
        else: 
            raise RuntimeError('Alternative stopping criterion reached in passing_map.') 

    def initialize_trapped_map(self):
        r"""
        For the given :class:`BoozerMagneticField` instance, this function generates initial positions
        for the trapped Poincare return map. Particles are initialized on the vpar = 0 plane 
        such that the parallel velocity is consistent with the prescribed total energy and pitch-angle 
        variable, :math:`\lambda = v_\perp^2/(v^2 B)` (i.e., all points are mirror points). 

        Returns: 
            s_init : List of initial s coordinates for the Poincare map.
            thetas_init : List of initial theta coordinates for the Poincare map.       
            zetas_init : List of initial zeta coordinates for the Poincare map.
        """
        # We now compute all of the chi mirror points for each s and eta
        def chi_mirror_func(s,eta):
            from scipy.optimize import root_scalar

            point = np.zeros((1,3))
            point[:,0] = s
            # Peform root solve to find bounce point
            def diffmodB(chi):
                return (modB_func(chi)-self.modBcrit)
            def graddiffmodB(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi,eta)
                point[:,1] = theta
                point[:,2] = zeta
                self.field.set_points(point)
                return self.field.dmodBdtheta()[0,0] * self.dtheta_dchi + self.field.dmodBdzeta()[0,0] * self.dzeta_dchi
            def modB_func(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi,eta)
                point[:,1] = theta
                point[:,2] = zeta
                self.field.set_points(point)
                return self.field.modB()[0,0]
            
            chi_mirror = self.chi(self.theta_mirror,self.zeta_mirror)
            try: 
                sol = root_scalar(diffmodB,fprime=graddiffmodB,x0=chi_mirror,method='toms748',bracket=[0,np.pi])
            except Exception:
                raise RuntimeError(f'Root solve for chi_mirror failed! s = {s}, eta/(2*pi) = {eta/(2*np.pi)}')
            if not sol.converged:
                raise RuntimeError(f'Root solve for chi_mirror did not converge! s = {s}, eta/(2*pi) = {eta/(2*np.pi)}')
            return sol.root

        etas = np.linspace(0,2*np.pi,self.neta_poinc,endpoint=False)
        s = np.linspace(0,1.0,self.ns_poinc+1,endpoint=False)[1::]
        etas2d,s2d = np.meshgrid(etas,s)
        etas2d = etas2d.flatten()
        s2d = s2d.flatten()

        s_init = []
        chis_init = []
        etas_init = []
        first, last = parallel_loop_bounds(self.comm, len(etas2d))
        # For each point, find the mirror point in chi
        for i in range(first,last):
            try:
                chi = chi_mirror_func(s2d[i],etas2d[i])
                s_init.append(s2d[i])
                chis_init.append(chi)
                etas_init.append(etas2d[i])
            except RuntimeError:
                warn(f'Root solve for chi_mirror failed! s = {s2d[i]}, eta/(2*pi) = {etas2d[i]/(2*np.pi)}')

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            etas_init = [i for o in self.comm.allgather(etas_init) for i in o]

        return s_init, chis_init, etas_init 

    def compute_trapped_map(self):
        r"""
        Evaluates the trapped Poincare return map for the initialized particle positions.
        """
        self.s_init, self.chis_init, self.etas_init = self.initialize_trapped_map()
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first,last):
            tr = [self.s_init[itrj],self.chis_init[itrj],self.etas_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            etas_traj = [tr[2]]
            t_traj = [0]
            broken = False 
            for jj in range(self.Nmaps):
                try:
                    # Apply trapped map twice to return to same vpar = 0 plane
                    tr, time = self.trapped_map(tr)
                    tr, time = self.trapped_map(tr)
                    if np.abs(tr[1]-chis_traj[-1]) > 2*np.pi:
                        warn('Barely trapped particle detected in trapped_map.')
                        broken = True 
                        break
                    s_traj.append(tr[0])
                    chis_traj.append(tr[1])
                    etas_traj.append(tr[2])
                    t_traj.append(time)
                except RuntimeError:
                    broken = True 
                    break
            if not broken: 
                s_all.append(s_traj)
                chis_all.append(chis_traj)
                etas_all.append(etas_traj)
                t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, chis_all, etas_all, t_all

    def plot_poincare(self,ax=None,filename='trapped_poincare.pdf'):
        r"""
        Plot the trapped Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are created.
            filename : Name of the file to save the plot (default: 'trapped_poincare.pdf').
        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib
        matplotlib.use('Agg') # Don't use interactive backend
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$s$')
        ax.set_xlim([0,2*np.pi])
        ax.set_ylim([0,1])
        for i in range(len(self.etas_all)):
            ax.scatter(np.mod(self.etas_all[i],2*np.pi), self.s_all[i], marker='o',s=0.5,edgecolors='none')
        plt.savefig(filename)

        return ax 
    
    def compute_frequencies(self):
        """
        Compute the trapped particle eta and bounce frequencies and mean radial position.
        The frequency is only computed if the trajectory has completed at least one full Poincare return map 
        before the trajectory is terminated due to a time limit or stopping criterion. Since the frequency 
        will depend on field-line label and trapping well in a general 3D field, an average is taken over
        all trajectories initialized on the same flux surface and all Poincare return maps along each trajectory.
        Since the frequency profile will flatten in a phase-space island, it is sometimes easier to interpret the
        frequency profile in a field with enforced quasisymmetry (i.e., initialize BoozerRadialInterpolant with N
        prescribed).

        Returns:
            omega_eta : List of mapping angle eta frequencies.
            omega_b : List of bounce frequencies.
            init_s : List of initial s values for each trajectory.
        """
        if self.solver_options['axis'] != 0:
            raise ValueError('ODE solver must integrate with solver_options["axis"]=0 to compute trapped frequencies.')

        omega_eta = []
        omega_b = []
        init_s = []
        for s_traj, chi_traj, eta_traj, t_traj in zip(self.s_all, self.chis_all, self.etas_all, self.t_all):
            if len(s_traj) < 2: # Need at least one full Poincare return maps to compute frequency
                continue
            delta_eta = np.array(eta_traj[1::]) - np.array(eta_traj[0:-1])
            delta_t = np.array(t_traj[1::]) - np.array(t_traj[0:-1])

            # Average over wells along one field line
            omega_eta.append(np.mean(delta_eta)/np.mean(delta_t))
            omega_b.append(2*np.pi/np.mean(delta_t)) # bounce frequency
            init_s.append(np.mean(s_traj))

        omega_eta = np.array(omega_eta)
        omega_b = np.array(omega_b)
        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_eta_prof = np.zeros((len(s_prof),))
        omega_b_prof = np.zeros((len(s_prof),))

        # Average over field-line label 
        for i, s in enumerate(s_prof):
            omega_eta_prof[i] = np.mean(omega_eta[np.where(init_s==s)])
            omega_b_prof[i] = np.mean(omega_b[np.where(init_s==s)])

        return omega_eta_prof, omega_b_prof, s_prof

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