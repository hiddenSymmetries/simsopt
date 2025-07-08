from math import sqrt
from warnings import warn
import numpy as np
import simsoptpp as sopp
from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import BoozerMagneticField, ShearAlfvenWave
from ..util.constants import (
    ALPHA_PARTICLE_MASS,
    ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from ..util.functions import print, proc0_print
from .._core.types import RealArray

__all__ = [
    "MinToroidalFluxStoppingCriterion",
    "MaxToroidalFluxStoppingCriterion",
    "IterationStoppingCriterion",
    "ToroidalTransitStoppingCriterion",
    "StepSizeStoppingCriterion",
    "compute_resonances",
    "compute_poloidal_transits",
    "compute_toroidal_transits",
    "trace_particles_boozer",
]


def trace_particles_boozer_perturbed(
    perturbed_field: ShearAlfvenWave,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    mus: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=None, 
    dt_max=None, 
    tol=1e-9,
    abstol=None,
    reltol=None,
    comm=None,
    thetas=[],
    zetas=[],
    omega_thetas=[],
    omega_zetas=[],
    vpars=[],
    stopping_criteria=[],
    dt_save=1e-6,
    mode=None,
    forget_exact_path=False,
    thetas_stop=False,
    zetas_stop=False,
    vpars_stop=False,
    axis=2,
):
    r"""
    Follow particles in a perturbed field of class :class:`ShearAlfvenWave`. This is modeled after
    :func:`trace_particles`.


    In the case of ``mode='gc_vac'`` we solve the guiding center equations under
    the vacuum assumption, i.e :math:`G =` const. and :math:`I = 0`:

    .. math::

        \dot s = (\alpha_{,\theta} B v_{||} - Phi_{,\theta})/\psi_0 -|B|_{,\theta} m(v_{||}^2/|B| + \mu)/(q \psi_0)

        \dot \theta = \Phi_{,\psi}/\psi_0 + |B|_{,s} m(v_{||}^2/|B| + \mu)/(q \psi_0) + \iota v_{||} |B|/G

        \dot \zeta = v_{||}|B|/G

        \dot v_{||} = -(\iota |B|_{,\theta} + |B|_{,\zeta})\mu |B|/G,

    where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp^2 = 2\mu|B|`.

    In the case of ``mode='gc'`` we solve the general guiding center equations
    for an MHD equilibrium:

    .. math::

        \dot s = (I |B|_{,\zeta} - G |B|_{,\theta})m(v_{||}^2/|B| + \mu)/(\iota D \psi_0)

        \dot \theta = ((G |B|_{,\psi} - K |B|_{,\zeta}) m(v_{||}^2/|B| + \mu) - C v_{||} |B|)/(\iota D)

        \dot \zeta = (F v_{||} |B| - (|B|_{,\psi} I - |B|_{,\theta} K) m(\rho_{||}^2 |B| + \mu) )/(\iota D)

        \dot v_{||} = (C|B|_{,\theta} - F|B|_{,\zeta})\mu |B|/(\iota D)

        C = - m v_{||} K_{,\zeta}/|B|  - q \iota + m v_{||}G'/|B|

        F = - m v_{||} K_{,\theta}/|B| + q + m v_{||}I'/|B|

        D = (F G - C I)/\iota

    where primes indicate differentiation wrt :math:`\psi`. In the case ``mode='gc_noK'``,
    the above equations are used with :math:`K=0`.

    Args:
        perturbed_field: The :class:`ShearAlfvenWave` instance
        stz_inits: A ``(nparticles, 3)`` array with the initial positions of
            the particles in Boozer coordinates :math:`(s,\theta,\zeta)`.
        parallel_speeds: A ``(nparticles, )`` array containing the speed in
            direction of the B field for each particle.
        mus: A ``(nparticles, )`` array containing the magnetic moment for
            each particle, vperp^2/(2*B). Note that unlike the unperturbed tracing, mu must be provided independently from the parallel speed since energy is not conserved.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV. Note that for perturbed tracing, this is simply used to determine the
            maximum timestep, dtmax = (G0/B0)*0.5*pi/vtotal, where G0 and B0 are evaluated at the first initial condition 
            and vtotal = sqrt(2*Ekin/mass). If Ekin is None, then dtmax is computed using data from the first initial condition with 
            vtotal = sqrt(vpar^2 + 2*mu*B0). 
        tol: tolerance for the adaptive ode solver
        abstol: absolute tolerance for adaptive ode solver
        reltol: relative tolerance for adaptive ode solver
        comm: MPI communicator to parallelize over
        thetas: list of angles in [0, 2pi] for which intersection with the plane 
            corresponding to that theta should be computed. Should only be used if axis = 0.
        zetas: list of angles in [0, 2pi] for which intersection with the plane
            corresponding to that zeta should be computed
        omega_thetas: list of frequencies defining a stopping criterion such that
              thetas - omega_thetas*t = 0. Must have the same length as thetas.
              If provided, the solver will stop when the particle hits the plane defined by thetas and omega_thetas.
        omega_zetas: list of frequencies defining a stopping criterion such that
              zetas - omega_zetas*t = 0. Must have the same length as zetas.
              If provided, the solver will stop when the particle hits the plane defined by zetas and omega_zetas.
        vpars: list of parallel velocities defining a stopping criterion such
              that the solver will stop when the particle hits these values.
        stopping_criteria: list of stopping criteria, mostly used in
            combination with the ``LevelsetStoppingCriterion``
            accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        dt_save: Time interval for saving the results. Only use if
            forget_exact_path = False.
        mode: how to trace the particles. Options are
            `gc`: general guiding center equations.
            `gc_vac`: simplified guiding center equations for the case :math:`G` = const.,
            :math:`I = 0`, and :math:`K = 0`.
            `gc_noK`: simplified guiding center equations for the case :math:`K = 0`.
            By default, this is determined from field.field_type
        forget_exact_path: return only the first and last position of each
            particle for the ``res_tys``. To be used when only res_hits is
            of interest or one wants to reduce memory usage.
        thetas_stop: whether to stop if hit provided theta planes.
        zetas_stop: whether to stop if hit provided zeta planes
        vpars_stop: whether to stop if hit provided vpar planes
        axis: Defines handling of coordinate singularity. If 0, tracing is
            performed in Boozer coordinates (s,theta,zeta). If 1, tracing is performed in coordinates (sqrt(s)*cos(theta), sqrt(s)*sin(theta), zeta). 
            If 2, tracing is performed in coordinates (s*cos(theta),s*sin(theta),zeta). Option 2 is recommended.
    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, 5).
            Each row contains the time and the state, `[t, s, theta, zeta, v_par]`.

        - ``res_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the zeta planes or
            one of the stopping criteria. Each row or the array contains `[time] + [idx] + state`, where `idx` tells us which of the hit planes or stopping criteria was hit.
            If `idx>=0` and `idx<len(zetas)`, then the `zetas[idx]` plane was hit. If `len(vpars)+len(zetas)>idx>=len(zetas)`, then the `vpars[idx-len(zetas)]` plane was hit.
            If `idx>=len(vpars)+len(zetas)`, then the `thetas[idx-len(vpars)-len(zetas)]` plane was hit.
            If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit. The state vector is `[s, theta, zeta, v_par, t]`.
    """
    if reltol is None:
        reltol = tol
    if abstol is None:
        abstol = tol
    nparticles = stz_inits.shape[0]
    assert stz_inits.shape[0] == len(parallel_speeds)
    assert len(mus) == len(parallel_speeds)
    speed_par = parallel_speeds
    m = mass
    if Ekin is None: 
        perturbed_field.set_points(np.asarray([[stz_inits[0,0]], [stz_inits[0,1]], [stz_inits[0,2]], [0]]).T)
        B0 = perturbed_field.B0.modB()[0,0]
        speed_total = sqrt(speed_par[0]**2 + 2 * mus[0] * B0)
    else: 
        speed_total = sqrt(2 * Ekin / m)

    if mode is not None:
        mode = mode.lower()
        assert mode in ["gc", "gc_vac", "gc_nok"]
        if "gc_" + perturbed_field.B0.field_type != mode:
            warn(
                f"Prescribed mode is inconsistent with field_type. Proceeding with mode={mode}.",
                RuntimeWarning,
            )
    else:
        mode = "gc_" + perturbed_field.B0.field_type

    res_tys = []
    res_hits = []
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        res_ty, res_hit = sopp.particle_guiding_center_boozer_perturbed_tracing(
            perturbed_field,
            stz_inits[i, :],
            m,
            charge,
            speed_total,
            speed_par[i],
            mus[i],
            tmax,
            abstol,
            reltol,
            vacuum=(mode == "gc_vac"),
            noK=(mode == "gc_nok"),
            thetas=thetas,
            zetas=zetas,
            omega_thetas=omega_thetas,
            omega_zetas=omega_zetas,
            vpars=vpars,
            stopping_criteria=stopping_criteria,
            dt_save=dt_save,
            thetas_stop=thetas_stop,
            zetas_stop=zetas_stop,
            vpars_stop=vpars_stop,
            forget_exact_path=forget_exact_path,
            axis=axis,
        )
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_hits.append(np.asarray(res_hit))
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_hits = [i for o in comm.allgather(res_hits) for i in o]

    return res_tys, res_hits

def trace_particles_boozer(
    field: BoozerMagneticField,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    abstol=None,
    reltol=None,
    comm=None,
    thetas=[],
    zetas=[],
    omega_thetas=[],
    omega_zetas=[],
    vpars=[],
    stopping_criteria=[],
    dt_save=1e-6,
    mode=None,
    forget_exact_path=False,
    thetas_stop=False,
    zetas_stop=False,
    vpars_stop=False,
    axis=None,
    dt=None,
    solveSympl=False,
    roottol=None,
    predictor_step=None,

):
    r"""
    Follow particles in a :class:`BoozerMagneticField`.


    In the case of ``mod='gc_vac'`` we solve the guiding center equations under
    the vacuum assumption, i.e :math:`G =` const. and :math:`I = 0`:

    .. math::

        \dot s = -|B|_{,\theta} m(v_{||}^2/|B| + \mu)/(q \psi_0)

        \dot \theta = |B|_{,s} m(v_{||}^2/|B| + \mu)/(q \psi_0) + \iota v_{||} |B|/G

        \dot \zeta = v_{||}|B|/G

        \dot v_{||} = -(\iota |B|_{,\theta} + |B|_{,\zeta})\mu |B|/G,

    where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp^2 = 2\mu|B|`.

    In the case of ``mode='gc'`` we solve the general guiding center equations
    for an MHD equilibrium:

    .. math::

        \dot s = (I |B|_{,\zeta} - G |B|_{,\theta})m(v_{||}^2/|B| + \mu)/(\iota D \psi_0)

        \dot \theta = ((G |B|_{,\psi} - K |B|_{,\zeta}) m(v_{||}^2/|B| + \mu) - C v_{||} |B|)/(\iota D)

        \dot \zeta = (F v_{||} |B| - (|B|_{,\psi} I - |B|_{,\theta} K) m(\rho_{||}^2 |B| + \mu) )/(\iota D)

        \dot v_{||} = (C|B|_{,\theta} - F|B|_{,\zeta})\mu |B|/(\iota D)

        C = - m v_{||} K_{,\zeta}/|B|  - q \iota + m v_{||}G'/|B|

        F = - m v_{||} K_{,\theta}/|B| + q + m v_{||}I'/|B|

        D = (F G - C I))/\iota

    where primes indicate differentiation wrt :math:`\psi`. In the case ``mod='gc_noK'``,
    the above equations are used with :math:`K=0`.

    Args:
        field: The :class:`BoozerMagneticField` instance
        stz_inits: A ``(nparticles, 3)`` array with the initial positions of
            the particles in Boozer coordinates :math:`(s,\theta,\zeta)`.
        parallel_speeds: A ``(nparticles, )`` array containing the speed in
                         direction of the B field for each particle.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV. Either a scalar
            (same energy for all particles), or``(nparticles, )`` array.
        tol: default tolerance for ode solver when solver-specific tolerances
            are not set
        reltol: relative tolerance for adaptive ode solver (defaults to `tol`). Only used if `solveSympl` is False.
        abstol: absolute tolerance for adaptive ode solver (defaults to `tol`). Only used if `solveSympl` is False.
        comm: MPI communicator to parallelize over
        thetas: list of angles in [0, 2pi] for which intersection with the plane 
            corresponding to that theta should be computed. Should only be used if axis = 0.
        zetas: list of angles in [0, 2pi] for which intersection with the plane
            corresponding to that zeta should be computed
        omega_thetas: list of frequencies defining a stopping criterion such that
              thetas - omega_thetas*t = 0. Must have the same length as thetas.
              If provided, the solver will stop when the particle hits the plane defined by thetas and omega_thetas.
        omega_zetas: list of frequencies defining a stopping criterion such that
              zetas - omega_zetas*t = 0. Must have the same length as zetas.
              If provided, the solver will stop when the particle hits the plane defined by zetas and omega_zetas.
        vpars: list of parallel velocities defining a stopping criterion such
              that the solver will stop when the particle hits these values.
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        dt_save: Time interval for saving the results. Only use if
            forget_exact_path = False.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case
                :math:`G` = const., :math:`I = 0`, and :math:`K = 0`.
            `gc_noK`: simplified guiding center equations for the case :math:`K = 0`.
            By default, this is determined from field.field_type.
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_hits is of
                           interest or one wants to reduce memory usage.
        zetas_stop: whether to stop if hit provided zeta planes
        vpars_stop: whether to stop if hit provided vpar planes
        axis: Defines handling of coordinate singularity. If 0, tracing is performed in Boozer coordinates (s,theta,zeta).
              Only used if `solveSympl` is False.
              If 1, tracing is performed in coordinates (sqrt(s)*cos(theta), sqrt(s)*sin(theta), zeta).
              If 2, tracing is performed in coordinates (s*cos(theta),s*sin(theta),zeta). Option 2 (default) is recommended.
        dt: time step for the symplectic solver. Only used if `solveSympl` is True.
        solveSympl: If True, uses symplectic solver. If False (default), uses RK45 solver with adaptive time step.
        roottol: root solver tolerance for the symplectic solver. Only used if `solveSympl` is True. If None, defaults to `tol`.
        predictor_step: provide better initial guess for the next time step using predictor steps. Defaults to True if `solveSympl` is True.
    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, 5).
            Each row contains the time and the state, `[t, s, theta, zeta, v_par]`.

        - ``res_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the zeta planes or
            one of the stopping criteria. Each row or the array contains `[time] + [idx] + state`, where `idx` tells us which of the hit planes or stopping criteria was hit.
            If `idx>=0` and `idx<len(zetas)`, then the `zetas[idx]` plane was hit. If `len(vpars)+len(zetas)>idx>=len(zetas)`, then the `vpars[idx-len(zetas)]` plane was hit.
            If `idx>=len(vpars)+len(zetas)`, then the `thetas[idx-len(vpars)-len(zetas)]` plane was hit.
            If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit. The state vector is `[s, theta, zeta, v_par]`.
    """
    if zetas_stop and (not len(zetas) and not len(omega_zetas)):
        raise ValueError("No zetas and omega_zetas provided for the zeta stopping criterion")
    if thetas_stop and (not len(thetas) and not len(omega_thetas)):
        raise ValueError("No thetas and omega_thetas provided for the theta stopping criterion")
    if vpars_stop and (not len(vpars)):
        raise ValueError("No vpars provided for the vpar stopping criterion")


    if solveSympl:
        if abstol is not None or reltol is not None:
            warn(
                "Symplectic solver does not use absolute or relative tolerance. "
                "Use dt and roottol to control timestep.",
                RuntimeWarning,
            )
        if (axis is not None and axis != 0):
            warn(
                "Symplectic solver must be run with axis = 0.",
                RuntimeWarning,
            )
            axis = 0
    else:
        if dt is not None or roottol is not None or predictor_step is not None:
            warn(
                "RK45 solver does not use dt, roottol, or predictor_step. "
                "Use abstol and reltol to control the timestep.",
                RuntimeWarning,
            )
    # Set default values for parameters
    if axis is None:
        axis = 2  
    if reltol is None:
        reltol = tol
    if abstol is None:
        abstol = tol
    if roottol is None:
        roottol = tol
    if dt is None:
        dt = 1e-7
    if predictor_step is None:
        predictor_step = True

    nparticles = stz_inits.shape[0]
    assert stz_inits.shape[0] == len(parallel_speeds)
    speed_par = parallel_speeds
    m = mass
    assert np.isscalar(Ekin) or (len(Ekin) == len(parallel_speeds))
    if np.isscalar(Ekin):
        Ekin = Ekin * np.ones((len(parallel_speeds),))
    # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    speed_total = np.sqrt(2 * Ekin / m)

    if mode is not None:
        mode = mode.lower()
        assert mode in ["gc", "gc_vac", "gc_nok"]
        if "gc_" + field.field_type != mode:
            warn(
                f"Prescribed mode is inconsistent with field_type. Proceeding with mode={mode}.",
                RuntimeWarning,
            )
    else:
        mode = "gc_" + field.field_type

    res_tys = []
    res_hits = []
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        res_ty, res_hit = sopp.particle_guiding_center_boozer_tracing(
            field,
            stz_inits[i, :],
            m,
            charge,
            speed_total[i],
            speed_par[i],
            tmax,
            vacuum=(mode == "gc_vac"),
            noK=(mode == "gc_nok"),
            thetas=thetas,
            zetas=zetas,
            omega_thetas=omega_thetas,
            omega_zetas=omega_zetas,
            vpars=vpars,
            stopping_criteria=stopping_criteria,
            dt_save=dt_save,
            forget_exact_path=forget_exact_path,
            thetas_stop=thetas_stop,
            zetas_stop=zetas_stop,
            vpars_stop=vpars_stop,
            axis=axis,
            abstol=abstol,
            reltol=reltol,
            solveSympl=solveSympl,
            predictor_step=predictor_step,
            roottol=roottol,
            dt=dt,
        )
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_hits.append(np.asarray(res_hit))
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_hits = [i for o in comm.allgather(res_hits) for i in o]
    return res_tys, res_hits


def compute_resonances(res_tys, res_hits, delta=1e-2):
    r"""
    Computes resonant particle orbits given the output of :func:`trace_particles_boozer`, ``res_tys`` and
    ``res_hits``, with ``forget_exact_path=False``. Resonance indicates a trajectory which returns to the same position
    at the :math:`\zeta = 0` plane after ``mpol`` poloidal turns and
    ``ntor`` toroidal turns.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``
        res_hits: output of :func:`trace_particles_boozer` with `zetas = [0]`
        delta: the distance tolerance in the poloidal plane used to compute
                a resonant orbit. (defaults to 1e-2)

    Returns:
        resonances: list of 7d arrays containing resonant particle orbits. The
                elements of each array is ``[s0, theta0, zeta0, vpar0, t, mpol, ntor]``. Here ``(s0, theta0, zeta0, vpar0)`` indicates the
                initial position and parallel velocity of the particle, ``t``
                indicates the time of the  resonance, ``mpol`` is the number of
                poloidal turns of the orbit, and ``ntor`` is the number of toroidal turns.
    """
    nparticles = len(res_tys)
    resonances = []
    # Iterate over particles
    for ip in range(nparticles):
        nhits = len(res_hits[ip])
        s0 = res_tys[ip][0, 1]
        theta0 = res_tys[ip][0, 2]
        zeta0 = res_tys[ip][0, 3]
        theta0_mod = theta0 % (2 * np.pi)
        x0 = s0 * np.cos(theta0)
        y0 = s0 * np.sin(theta0)
        vpar0 = res_tys[ip][0, 4]
        for it in range(1, nhits):
            # Check whether phi hit or stopping criteria achieved
            if int(res_hits[ip][it, 1]) >= 0:
                s = res_hits[ip][it, 2]
                theta = res_hits[ip][it, 3]
                zeta = res_hits[ip][it, 4]
                theta_mod = theta % 2 * np.pi
                x = s * np.cos(theta)
                y = s * np.sin(theta)
                dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                t = res_hits[ip][it, 0]
                if dist < delta:
                    proc0_print("Resonance found.")
                    proc0_print(
                        f"theta = {theta_mod}, theta0 = {theta0_mod}, s = {s}, s0 = {s0}"
                    )
                    mpol = np.rint((theta - theta0) / (2 * np.pi))
                    ntor = np.rint((zeta - zeta0) / (2 * np.pi))
                    resonances.append(
                        np.asarray([s0, theta0, zeta0, vpar0, t, mpol, ntor])
                    )
    return resonances


def compute_toroidal_transits(res_tys):
    r"""
    Computes the number of toroidal transits of an orbit.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles_boozer` with ``forget_exact_path=False``.

    Returns:
        ntransits: array with length ``len(res_tys)``. Each element contains the
                number of toroidal transits of the orbit.
    """
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        phi_init = res_tys[ip][0, 3]
        for it in range(1, ntraj):
            phi = res_tys[ip][it, 3]
        if ntraj > 1:
            ntransits[ip] = np.round((phi - phi_init) / (2 * np.pi))
    return ntransits


def compute_poloidal_transits(res_tys, ma=None, flux=True):
    r"""
    Computes the number of poloidal transits of an orbit. For the case of
    particles traced in a :class:`MagneticField` (not a :class:`BoozerMagneticField`),
    the poloidal angle is computed using the arctangent angle in the poloidal plane with
    respect to the coordinate axis, ``ma``,

    .. math::
        \theta = \tan^{-1} \left( \frac{R(\phi)-R_{\mathrm{ma}}(\phi)}{Z(\phi)-Z_{\mathrm{ma}}(\phi)} \right),

    where :math:`(R,\phi,Z)` are the cylindrical coordinates of the trajectory
    and :math:`(R_{\mathrm{ma}}(\phi),Z_{\mathrm{ma}(\phi)})` is the position
    of the coordinate axis.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``.
        ma: an instance of :class:`Curve` representing the coordinate axis with
                respect to which the poloidal angle is computed. If orbit is
                computed in Boozer coordinates, ``ma`` should be ``None``.
        flux: if ``True``, ``res_tys`` represents the position in flux coordinates
                (should be ``True`` if computed from :func:`trace_particles_boozer`).
                If ``True``, ``ma`` is not used.
    Returns:
        ntransits: array with length ``len(res_tys)``. Each element contains the
                number of poloidal transits of the orbit.
    """
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        theta_init = res_tys[ip][0, 2]
        for it in range(1, ntraj):
            theta = res_tys[ip][it, 2]
        if ntraj > 1:
            ntransits[ip] = np.round((theta - theta_init) / (2 * np.pi))
    return ntransits


class MinToroidalFluxStoppingCriterion(sopp.MinToroidalFluxStoppingCriterion):
    """
    Stop the iteration once a particle falls below a critical value of
    ``s``, the normalized toroidal flux. This :class:`StoppingCriterion` is
    important to use when tracing particles in flux coordinates, as the poloidal
    angle becomes ill-defined at the magnetic axis. This should only be used
    when tracing trajectories in a flux coordinate system (i.e., :class:`trace_particles_boozer`).

    Usage:

    .. code-block::

        stopping_criteria=[MinToroidalFluxStopingCriterion(s)]

    where ``s`` is the value of the minimum normalized toroidal flux.
    """

    pass


class MaxToroidalFluxStoppingCriterion(sopp.MaxToroidalFluxStoppingCriterion):
    """
    Stop the iteration once a particle falls above a critical value of
    ``s``, the normalized toroidal flux. This should only be used when tracing
    trajectories in a flux coordinate system (i.e., :class:`trace_particles_boozer`).

    Usage:

    .. code-block::

        stopping_criteria=[MaxToroidalFluxStopingCriterion(s)]

    where ``s`` is the value of the maximum normalized toroidal flux.
    """

    pass


class ToroidalTransitStoppingCriterion(sopp.ToroidalTransitStoppingCriterion):
    """
    Stop the iteration once the maximum number of toroidal transits is reached.

    Usage:

    .. code-block::

        stopping_criteria=[ToroidalTransitStoppingCriterion(ntransits,flux)]

    where ``ntransits`` is the maximum number of toroidal transits and ``flux``
    is a boolean indicating whether tracing is being performed in a flux coordinate system.
    """

    pass


class IterationStoppingCriterion(sopp.IterationStoppingCriterion):
    """
    Stop the iteration once the maximum number of iterations is reached.
    """

    pass


class StepSizeStoppingCriterion(sopp.StepSizeStoppingCriterion):
    """
    Stop the iteration once the step size is too small.
    """

    pass
