import numpy as np
from warnings import warn

from ..field.tracing import (
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
)
from ..field.boozermagneticfield import ShearAlfvenHarmonic, ShearAlfvenWave
from .._core.util import parallel_loop_bounds
from ..util.functions import proc0_print

__all__ = [
    "compute_loss_fraction",
    "compute_trajectory_cylindrical",
    "PassingPoincare",
    "PassingPerturbedPoincare",
    "trajectory_to_vtk",
]


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
        timelost[ip] = res_tys[ip][-1, 0]

    times = np.logspace(np.log10(tmin), np.log10(tmax), ntime)

    loss_frac = np.zeros_like(times)
    for it in range(ntime):
        loss_frac[it] = np.count_nonzero(timelost < times[it]-1e-15) / nparticles

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
    nsteps = len(res_ty[:, 0])
    points = np.zeros((nsteps, 3))
    points[:, 0] = res_ty[:, 1]
    points[:, 1] = res_ty[:, 2]
    points[:, 2] = res_ty[:, 3]
    field.set_points(points)

    R_traj = field.R()[:, 0]
    Z_traj = field.Z()[:, 0]
    nu = field.nu()[:, 0]
    phi_traj = res_ty[:, 3] - nu

    return R_traj, phi_traj, Z_traj


class PassingPoincare:
    """
    Class to compute and store passing Poincare maps and related quantities for a given BoozerMagneticField.
    """

    def __init__(
        self,
        field,
        lam,
        sign_vpar,
        mass,
        charge,
        Ekin,
        ns_poinc=120,
        ntheta_poinc=2,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options={},
    ):
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
        if sign_vpar not in [-1, 1]:
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

        self.s_all, self.thetas_all, self.vpars_all, self.t_all = (
            self.compute_passing_map()
        )

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
        s = np.linspace(0, 1, self.ns_poinc + 1, endpoint=False)[1::]
        thetas = np.linspace(0, 2 * np.pi, self.ntheta_poinc)
        s, thetas = np.meshgrid(s, thetas)
        s = s.flatten()
        thetas = thetas.flatten()

        vtotal = np.sqrt(
            2 * self.Ekin / self.mass
        )  # Total velocity from kinetic energy

        def vpar_func(s, theta):
            point = np.zeros((1, 3))
            point[0, 0] = s
            point[0, 1] = theta
            self.field.set_points(point)
            modB = self.field.modB()[0, 0]
            # Skip any trapped particles
            if 1 - self.lam * modB < 0:
                return None
            else:
                return self.sign_vpar * vtotal * np.sqrt(1 - self.lam * modB)

        first, last = parallel_loop_bounds(self.comm, len(s))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        s_init = []
        thetas_init = []
        vpars_init = []
        for i in range(first, last):
            vpar = vpar_func(s[i], thetas[i])
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

        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = point[1]
        points[:, 2] = 0
        # Set solver options needed for passing map
        res_tys, res_hits = trace_particles_boozer(
            self.field,
            points,
            [point[2]],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            zetas=[0],
            vpars=[0],
            omega_zetas=[0],
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            zetas_stop=True,
            **self.solver_options,
        )
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        if res_hit[1] == 0:  # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = res_hit[3]
            point[2] = res_hit[5]
            time = res_hit[0]
            return point, time
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

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
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.thetas_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            thetas_traj = [tr[1]]
            vpars_traj = [tr[2]]
            t_traj = [0]
            for jj in range(self.Nmaps):
                try:
                    tr, time = self.passing_map(tr)
                    s_traj.append(tr[0])
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
        if "axis" in self.solver_options:
            if self.solver_options["axis"] != 0:
                raise ValueError(
                    'ODE solver must integrate with solver_options["axis"]=0 to compute passing frequencies.'
                )

        self.field.set_points(np.array([[1], [0], [0]]).T)
        sign_G = np.sign(self.field.G()[0])

        omega_theta = []
        omega_zeta = []
        init_s = []
        for s_traj, theta_traj, vpar_traj, t_traj in zip(
            self.s_all, self.thetas_all, self.vpars_all, self.t_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_theta = np.array(theta_traj[1::]) - np.array(theta_traj[0:-1])

            delta_t = np.array(t_traj[1::]) - np.array(t_traj[0:-1])
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G

            # Average over wells along one field line
            freq_theta = np.mean(delta_theta) / np.mean(delta_t)
            freq_zeta = delta_zeta / np.mean(delta_t)

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
            omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s == s)])
            omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s == s)])

        return omega_theta_prof, omega_zeta_prof, s_prof

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, thetas_all, vpars_all, t_all : Lists of trajectory data.
        """
        return self.s_all, self.thetas_all, self.vpars_all, self.t_all

    def plot_poincare(self, ax=None, filename="passing_poincare.pdf"):
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

        matplotlib.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.thetas_all)):
            ax.scatter(
                np.mod(self.thetas_all[i], 2 * np.pi),
                self.s_all[i],
                marker="o",
                s=0.5,
                edgecolors="none",
            )
            plt.savefig(filename)

        return ax


class TrappedPoincare:
    """
    Class to compute and store trapped Poincare maps and related quantities for a given BoozerMagneticField.
    """

    def __init__(
        self,
        field,
        helicity_M,
        helicity_N,
        s_mirror,
        theta_mirror,
        zeta_mirror,
        mass,
        charge,
        Ekin,
        ns_poinc=120,
        neta_poinc=2,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options={},
    ):
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
        field.set_points(np.array([[s_mirror], [theta_mirror], [zeta_mirror]]).T)
        self.modBcrit = field.modB()[0, 0]  # Magnetic field at mirror point
        self.lam = 1 / self.modBcrit  # lambda = v_perp^2/(v^2 B) = 1/modBcrit
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.ns_poinc = ns_poinc
        self.neta_poinc = neta_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        self.dtheta_dchi = self.helicity_Np / denom
        self.dzeta_dchi = self.helicity_Mp / denom

        self.s_all, self.chis_all, self.etas_all, self.t_all = (
            self.compute_trapped_map()
        )

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def trapped_map(self, point):
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
        theta, zeta = self.chi_eta_to_theta_zeta(point[1], point[2])
        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = theta
        points[:, 2] = zeta

        # Set solver options needed for passing map
        res_tys, res_hits = trace_particles_boozer(
            self.field,
            points,
            [0],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            zetas=[],
            vpars=[0],
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            zetas_stop=False,
            **self.solver_options,
        )

        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in trapped_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        if res_hit[1] == 0:  # Check that the vpars=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3], res_hit[4])
            point[2] = self.eta(res_hit[3], res_hit[4])
            time = res_hit[0]
            return point, time
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

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
        def chi_mirror_func(s, eta):
            from scipy.optimize import root_scalar

            point = np.zeros((1, 3))
            point[:, 0] = s

            # Peform root solve to find bounce point
            def diffmodB(chi):
                return modB_func(chi) - self.modBcrit

            def graddiffmodB(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi, eta)
                point[:, 1] = theta
                point[:, 2] = zeta
                self.field.set_points(point)
                return (
                    self.field.dmodBdtheta()[0, 0] * self.dtheta_dchi
                    + self.field.dmodBdzeta()[0, 0] * self.dzeta_dchi
                )

            def modB_func(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi, eta)
                point[:, 1] = theta
                point[:, 2] = zeta
                self.field.set_points(point)
                return self.field.modB()[0, 0]

            chi_mirror = self.chi(self.theta_mirror, self.zeta_mirror)
            try:
                sol = root_scalar(
                    diffmodB,
                    fprime=graddiffmodB,
                    x0=chi_mirror,
                    method="toms748",
                    bracket=[0, np.pi],
                )
            except Exception:
                raise RuntimeError(
                    f"Root solve for chi_mirror failed! s = {s}, eta/(2*pi) = {eta / (2 * np.pi)}"
                )
            if not sol.converged:
                raise RuntimeError(
                    f"Root solve for chi_mirror did not converge! s = {s}, eta/(2*pi) = {eta / (2 * np.pi)}"
                )
            return sol.root

        etas = np.linspace(0, 2 * np.pi, self.neta_poinc, endpoint=False)
        s = np.linspace(0, 1.0, self.ns_poinc + 1, endpoint=False)[1::]
        etas2d, s2d = np.meshgrid(etas, s)
        etas2d = etas2d.flatten()
        s2d = s2d.flatten()

        s_init = []
        chis_init = []
        etas_init = []
        first, last = parallel_loop_bounds(self.comm, len(etas2d))
        # For each point, find the mirror point in chi
        for i in range(first, last):
            try:
                chi = chi_mirror_func(s2d[i], etas2d[i])
                s_init.append(s2d[i])
                chis_init.append(chi)
                etas_init.append(etas2d[i])
            except RuntimeError:
                warn(
                    f"Root solve for chi_mirror failed! s = {s2d[i]}, eta/(2*pi) = {etas2d[i] / (2 * np.pi)}"
                )

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
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.etas_init[itrj]]
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
                    if np.abs(tr[1] - chis_traj[-1]) > 2 * np.pi:
                        warn("Barely trapped particle detected in trapped_map.")
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

    def plot_poincare(self, ax=None, filename="trapped_poincare.pdf"):
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

        matplotlib.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.etas_all)):
            ax.scatter(
                np.mod(self.etas_all[i], 2 * np.pi),
                self.s_all[i],
                marker="o",
                s=0.5,
                edgecolors="none",
            )
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
        if self.solver_options["axis"] != 0:
            raise ValueError(
                'ODE solver must integrate with solver_options["axis"]=0 to compute trapped frequencies.'
            )

        omega_eta = []
        omega_b = []
        init_s = []
        for s_traj, chi_traj, eta_traj, t_traj in zip(
            self.s_all, self.chis_all, self.etas_all, self.t_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_eta = np.array(eta_traj[1::]) - np.array(eta_traj[0:-1])
            delta_t = np.array(t_traj[1::]) - np.array(t_traj[0:-1])

            # Average over wells along one field line
            omega_eta.append(np.mean(delta_eta) / np.mean(delta_t))
            omega_b.append(2 * np.pi / np.mean(delta_t))  # bounce frequency
            init_s.append(np.mean(s_traj))

        omega_eta = np.array(omega_eta)
        omega_b = np.array(omega_b)
        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_eta_prof = np.zeros((len(s_prof),))
        omega_b_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            omega_eta_prof[i] = np.mean(omega_eta[np.where(init_s == s)])
            omega_b_prof[i] = np.mean(omega_b[np.where(init_s == s)])

        return omega_eta_prof, omega_b_prof, s_prof


def compute_peta(field_or_saw, points, vpar, mass, charge, helicity_M, helicity_N):
    r"""
    Given a ShearAlfvenWave or BoozerMagneticField instance, a point in Boozer coordinates, and particle properties, compute the
    value of the canonical momentum, :math:`p_{\eta}`, which is conserved under the
    perturbed guiding center equations if the field strength has helicity (M,N).

    :math:`p_{\eta} = (M G + N I) \left(\frac{m v_{\|\|}}{B} + q \alpha \right) + q (M \psi - M \psi')`

    If field_or_saw is a BoozerMagneticField instance, then alpha = 0.

    Args:
        field_or_saw : The BoozerMagneticField or ShearAlfvenWave instance.
        points : A numpy array of shape (npoints,4) containing the coordinates (s,theta,zeta,t).
            If field_or_saw is a ShearAlfvenWave, then t is the time coordinate.
            If field_or_saw is a BoozerMagneticField, then t is ignored, and points is allowed to
            have shape (npoints,3) for (s,theta,zeta).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field.
        helicity_N : Toroidal helicity of the magnetic field.

    Returns:
        peta : A numpy array of shape (npoints,) containing the value of the canonical
            momentum :math:`p_{\eta}` at each point.
    """
    if points.shape[1] not in [3, 4]:
        raise ValueError(
            "Points must have shape (npoints, 4) for (s, theta, zeta, t) or (npoints, 3) for (s, theta, zeta)"
        )
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )

    if isinstance(field_or_saw, ShearAlfvenWave):
        field = field_or_saw.B0
        field_or_saw.set_points(points)
        alpha = field_or_saw.alpha()
    else:
        field = field_or_saw
        alpha = 0.0
        if points.shape == 4:
            points = points[:, :3]
        field.set_points(points)

    modB = field.modB()[:, 0]
    G = field.G()[:, 0]
    I = field.I()[:, 0]
    psi = field.psi0 * points[:, 0]
    psip = field.psip()[:, 0]

    # If modB contours close poloidally, then use theta as mapping coordinate
    if helicity_M == 0:
        helicity_Mp = 1
        helicity_Np = 0
    # Otherwise, use zeta as mapping coordinate
    else:
        helicity_Mp = 0
        helicity_Np = -1
    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp
    peta = (
        -(
            (helicity_M * G + helicity_N * I) * (mass * vpar / modB + charge * alpha)
            + charge * (helicity_N * psi - helicity_M * psip)
        )
        / denom
    )
    return peta


def compute_Eprime(saw, points, vpar, mu, mass, charge, helicity_M, helicity_N):
    r"""
    Compute the invariant Eprime for a ShearAlfvenHarmonic instance given points in Boozer coordinates.

    Args:
        saw : An instance of ShearAlfvenHarmonic.
        points : A numpy array of shape (npoints,4) containing the coordinates (s,theta,zeta,t).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mu : Magnetic moment of the particle, vperp^2/(2 B).
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field strength.
        helicity_N : Toroidal helicity of the magnetic field strength.
    """
    if points.shape[1] != 4:
        raise ValueError("Points must have shape (npoints, 4) for (s, theta, zeta, t)")
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )
    if vpar.shape[0] != points.shape[0]:
        raise ValueError("vpar must have the same number of points as points")
    if isinstance(saw, ShearAlfvenHarmonic) is False:
        raise TypeError("Expected saw to be an instance of ShearAlfvenHarmonic")

    # If modB contours close poloidally, then use theta as mapping coordinate
    if helicity_M == 0:
        helicity_Mp = 1
        helicity_Np = 0
    # Otherwise, use zeta as mapping coordinate
    else:
        helicity_Mp = 0
        helicity_Np = -1

    Phim = saw.Phim
    Phin = saw.Phin
    omega = saw.omega

    # Compute the canonical momentum p_eta
    p_eta = compute_peta(saw, points, vpar, mass, charge, helicity_M, helicity_N)

    # Compute the energy E
    modB = saw.B0.modB()[:, 0]
    E = 0.5 * mass * vpar**2 + mass * mu * modB + charge * saw.Phi()[:, 0]

    # Compute the invariant Eprime
    nprime = (Phim * helicity_N - Phin * helicity_M) / (
        helicity_Np * helicity_M - helicity_N * helicity_Mp
    )
    Eprime = nprime * E - omega * p_eta
    return Eprime


class PassingPerturbedPoincare:
    def __init__(
        self,
        saw,
        sign_vpar,
        mass,
        charge,
        helicity_M,
        helicity_N,
        Eprime=None,
        mu=None,
        Ekin=None,
        p0=None,
        lam=None,
        ns_poinc=120,
        nchi_poinc=2,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options={},
    ):
        """
        Initialize the PassingPerturbedPoincare class, which computes the Poincare return map
        for passing particles in a ShearAlfvenHarmonic magnetic field.

        The field strength contours are assumed to have helicity (M,N) in Boozer coordinates such that
        the field strength can be expressed as B(s,chi), where chi = M*theta - N*zeta is the helical angle.
        The mapping coordinate, eta, is chosen based on the helicity of the field strength contours: if M = 0 (e.g., QP or OP),
        then eta=theta is used, if N = 0 (e.g., QA/OA, QH/OA), then eta = zeta is used.

        The ShearAlfvenHarmonic can then be expressed in terms of the mapping coordinates with phase,
        m'*chi - n'*eta + omega * t.

        The map is evaluated by integrating the guiding center equations from the eta - omega/n' * t = 0 plane until the
        trajectory returns to the same plane.

        The map is well-defined (i.e., trajectories don't cross in the (s,chi=M*theta - N*zeta) plane) if
        the unperturbed field is quasisymmetric with helicity (M,N). However, the map can still be computed
        for a non-quasisymmetric field, but the trajectories may cross in the (s,chi) plane.

        The constants of motion are the magnetic moment, mu = vperp^2/(2 B), and the shifted energy,
        Eprime = n' * E - omega * p_eta, E is the total energy, and p_eta is the canonical momentum.

        These constants of motion can be prescribed directly with the Eprime and mu parameters. Alternatively,
        they can be computed from the prescribed pitch-angle variable, lam = vperp^2/(v^2 B), total unperturbed kinetic energy, Ekin,
        and a given point p0 in Boozer coordinates.

        Args:
            saw : An instance of ShearAlfvenHarmonic representing the magnetic field.
            sign_vpar : Sign of the parallel velocity, either -1 or +1.
            mass : Mass of the particle.
            charge : Charge of the particle.
            helicity_M : Poloidal helicity of the magnetic field.
            helicity_N : Toroidal helicity of the magnetic field.
            Eprime: Shifted energy, Eprime = n' * E - omega * p_eta.
            mu: Magnetic moment, mu = vperp^2/(2 B).
            Ekin: Total unperturbed kinetic energy of the particle, used to compute Eprime if not provided.
            p0: Initial point in Boozer coordinates for evaluation of Eprime.
            lam: Pitch angle variable, lambda = vperp^2/(v^2 B), used to compute mu if not provided.
            ns_poinc : Number of initial conditions in s for Poincare plot (default: 120).
            nchi_poinc : Number of initial conditions in chi for Poincare plot (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver (default: {}).
        """
        if not isinstance(saw, ShearAlfvenHarmonic):
            raise TypeError("Expected saw to be an instance of ShearAlfvenHarmonic")
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self.mass = mass
        self.charge = charge
        self.sign_vpar = sign_vpar
        # If modB contours close poloidally, then use theta as mapping coordinate
        if self.helicity_M == 0:
            self.helicity_Mp = 1
            self.helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            self.helicity_Mp = 0
            self.helicity_Np = -1
        self.ns_poinc = ns_poinc
        self.nchi_poinc = nchi_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        self.Phin = saw.Phin
        self.Phim = saw.Phim
        self.omega = saw.omega

        self.nprime = (self.Phim * self.helicity_N - self.Phin * self.helicity_M) / (
            self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        )

        self.omegan = self.omega / self.nprime  # Frequency for Poincare slice

        if Eprime is not None and mu is not None:
            self.Eprime = Eprime
            self.mu = mu
            self.Ekin = None
        elif Ekin is not None and lam is not None and p0 is not None:
            """
            Compute unperturbed values of mu, p_eta, and Eprime from the given parameters.
            """
            v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
            self.mu = 0.5 * lam * v0**2  # mu = vperp^2/(2 B)
            self.Ekin = Ekin  # Total kinetic energy
            saw.B0.set_points(p0)
            modB = saw.B0.modB()[0, 0]
            if 1 - lam * modB < 0:
                raise ValueError(
                    "Invalid parameter p0: 1 - lambda * modB must be non-negative."
                )
            vpar = sign_vpar * v0 * np.sqrt(1 - lam * modB)  # Parallel velocity
            Peta0 = compute_peta(
                saw.B0,
                p0,
                vpar,
                mass,
                charge,
                helicity_M,
                helicity_N,
            )
            self.Eprime = self.nprime * Ekin - self.omega * Peta0
        else:
            raise ValueError(
                "Either Eprime and mu must be provided, or Ekin, lam, and p0 must be provided."
            )

        # Initialize the passing map
        self.s_init, self.chis_init, self.vpars_init = self.initialize_passing_map()
        # If Ekin is not provided, compute it from the initial parallel velocity
        # this is only used for computing maximum time step in the ODE solver
        if self.Ekin is None:
            self.Ekin = 0.5 * self.mass * self.vpars_init[0] ** 2

        self.s_all, self.chis_all, self.etas_all, self.vpars_all, self.t_all = (
            self.compute_passing_map()
        )

    def initialize_passing_map(self):
        """
        Compute vpar given (s,chi) such that Eprime = Eprime0
        """

        def vpar_func_perturbed(s, chi):
            # Choose initial conditions on the eta = 0 plane
            theta, zeta = self.chi_eta_to_theta_zeta(chi, 0)
            point = np.zeros((1, 4))  # initialize with t = 0
            point[0, 0] = s
            point[0, 1] = theta
            point[0, 2] = zeta
            self.saw.set_points(point)
            modB = self.B0.modB()[0, 0]
            G = self.B0.G()[0, 0]
            I = self.B0.I()[0, 0]
            psi = self.B0.psi0 * s
            psip = self.B0.psip()[0, 0]
            Phi = self.saw.Phi()[0, 0]
            alpha = self.saw.alpha()[0, 0]
            denom = (
                self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
            )  # - 1 in QA
            d_peta_d_vpar = (
                -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB))
                / denom
            )  # G m/ modB in QA
            d_E_d_vpar2 = 0.5 * self.mass
            a = self.nprime * d_E_d_vpar2  # Coefficient of vpar^2
            b = -self.omega * d_peta_d_vpar  # Coefficient of vpar
            # Constant term
            c = (
                self.nprime * (self.mass * self.mu * modB + self.charge * Phi)
                + self.omega
                * (
                    (self.helicity_M * G + self.helicity_N * I) * self.charge * alpha
                    + self.charge * (self.helicity_N * psi - self.helicity_M * psip)
                )
                / denom
                - self.Eprime
            )
            if (b**2 - 4 * a * c) < 0:
                raise RuntimeError(
                    "No solution for vpar found! Check the parameters and initial conditions."
                )
            else:
                return (-b + self.sign_vpar * np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        s = np.linspace(0, 1, self.ns_poinc + 1, endpoint=False)[1::]
        chis = np.linspace(0, 2 * np.pi, self.nchi_poinc)
        s, chis = np.meshgrid(s, chis)
        s = s.flatten()
        chis = chis.flatten()

        first, last = parallel_loop_bounds(self.comm, len(s))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        s_init = []
        chis_init = []
        vpars_init = []
        for i in range(first, last):
            try:
                vpar = vpar_func_perturbed(s[i], chis[i])
                s_init.append(s[i])
                chis_init.append(chis[i])
                vpars_init.append(vpar)
                theta, zeta = self.chi_eta_to_theta_zeta(chis[i], 0)
            except RuntimeError:
                continue

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]

        return s_init, chis_init, vpars_init

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def passing_map(self, point, t, eta):
        r"""
        Integrates the GC equations from the provided point on the eta - omega/n' * t plane
        to the next intersection with this plane. An assumption is made that the particle
        is passing, so a RuntimeError is raised if the particle mirrors.

        Since the phase of the ShearAlfvenWave depends on time, the initial time, t, is
        passed as an argument.

        Args:
            point : A numpy array of shape (3,) containing the initial coordinates (s,chi,eta).
            t : Initial time at which the map is evaluated
        Returns:
            point : A numpy array of shape (3,) containing the coordinates (s,chi,eta).
            time : The time at which the trajectory returns to the eta - omega/n' * t plane.
        """
        phase = self.omega * t
        self.saw.phase = phase
        theta, zeta = self.chi_eta_to_theta_zeta(point[1], eta)
        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = theta
        points[:, 2] = zeta

        if self.helicity_M != 0:
            zetas = [zeta]
            thetas = []
            omega_zetas = [self.omegan]
            omega_thetas = []
            zetas_stop = True
            thetas_stop = False
        else:
            zetas = []
            thetas = [theta]
            omega_zetas = []
            omega_thetas = [self.omegan]
            zetas_stop = False
            thetas_stop = True

        res_tys, res_hits = trace_particles_boozer_perturbed(
            self.saw,
            points,
            [point[2]],
            [self.mu],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            thetas=thetas,
            zetas=zetas,
            vpars=[0],
            omega_thetas=omega_thetas,
            omega_zetas=omega_zetas,
            axis=0,
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=True,
            vpars_stop=True,
            zetas_stop=zetas_stop,
            thetas_stop=thetas_stop,
            **self.solver_options,
        )
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        if res_hit[1] == 0:  # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3], res_hit[4])
            point[2] = res_hit[5]
            return point, res_hit[0] + t, self.eta(res_hit[3], res_hit[4])
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle positions.
        """
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        vpars_all = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            eta_traj = [0]
            vpars_traj = [tr[2]]
            t_traj = [0]
            for jj in range(self.Nmaps):
                try:
                    tr, time, eta = self.passing_map(tr, t_traj[-1], eta_traj[-1])
                    s_traj.append(tr[0])
                    chis_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    t_traj.append(time)
                    eta_traj.append(eta)
                except RuntimeError:
                    break
            s_all.append(s_traj)
            chis_all.append(chis_traj)
            etas_all.append(eta_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, chis_all, etas_all, vpars_all, t_all

    def plot_poincare(self, ax=None, filename="passing_poincare.pdf"):
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

        matplotlib.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r"$\chi$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.chis_all)):
            ax.scatter(
                np.mod(self.chis_all[i], 2 * np.pi),
                self.s_all[i],
                marker="o",
                s=0.5,
                edgecolors="none",
            )
            plt.savefig(filename)

        return ax


def trajectory_to_vtk(res_ty, field, filename="trajectory"):
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

    ppl = np.asarray([len(R_traj)])  # Number of points along trajectory
    polyLinesToVTK(filename, X_traj, Y_traj, Z_traj, pointsPerLine=ppl)
