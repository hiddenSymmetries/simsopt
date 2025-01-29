import logging
from math import sqrt
from warnings import warn

import numpy as np

import simsoptpp as sopp
from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import BoozerMagneticField
from ..util.constants import (
    ALPHA_PARTICLE_MASS, 
    ALPHA_PARTICLE_CHARGE, 
    FUSION_ALPHA_PARTICLE_ENERGY
)
from .._core.types import RealArray


logger = logging.getLogger(__name__)

__all__ = ['MinToroidalFluxStoppingCriterion', 'MaxToroidalFluxStoppingCriterion',
           'IterationStoppingCriterion', 'ToroidalTransitStoppingCriterion',
           'StepSizeStoppingCriterion', 'compute_resonances',
           'compute_poloidal_transits', 'compute_toroidal_transits',
           'trace_particles_boozer']


def compute_gc_radius(m, vperp, q, absb):
    """
    Computes the gyro radius of a particle in a field with strenght ``absb```,
    that is ``r=m*vperp/(abs(q)*absb)``.
    """
    return m*vperp/(abs(q)*absb)

def trace_particles_boozer_perturbed(
    field: BoozerMagneticField,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    mus: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    abstol=None,
    reltol=None,
    comm=None,
    zetas=[],
    omegas=[],
    vpars=[],
    stopping_criteria=[],
    dt_save=1e-6,
    mode='gc_vac', 
    forget_exact_path=False,
    zetas_stop=False,
    vpars_stop=False,
    Phihat=0,
    omega=0,
    Phim=0,
    Phin=0,
    phase=0,
    axis=0
):
    r"""
    Follow particles in a :class:`BoozerMagneticField`. This is modeled after
    :func:`trace_particles`.


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
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        zetas: list of angles in [0, 2pi] for which intersection with the plane
            corresponding to that zeta should be computed
        stopping_criteria: list of stopping criteria, mostly used in
            combination with the ``LevelsetStoppingCriterion``
            accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. Options are
            `gc`: general guiding center equations.
            `gc_vac`: simplified guiding center equations for the case :math:`G` = const.,
            :math:`I = 0`, and :math:`K = 0`.
            `gc_noK`: simplified guiding center equations for the case :math:`K = 0`.
        forget_exact_path: return only the first and last position of each
            particle for the ``res_tys``. To be used when only res_zeta_hits is of
            interest or one wants to reduce memory usage.

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, M)
            with M depending on the ``mode``.  Each row contains the time and
            the state.  So for `mode='gc'` and `mode='gc_vac'` the state
            consists of the :math:`(s,\theta,\zeta)` position and the parallel speed, hence
            each row contains `[t, s, t, z, v_par]`.

        - ``res_zeta_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the zeta planes or
            one of the stopping criteria. Each row of the array contains
            `[time] + [idx] + state`, where `idx` tells us which of the `zetas`
            or `stopping_criteria` was hit.  If `idx>=0`, then `zetas[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
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
    speed_total = sqrt(2*Ekin/m)
    mode = mode.lower()
    assert mode in ['gc', 'gc_vac', 'gc_nok']

    res_tys = []
    res_zeta_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        res_ty, res_zeta_hit = sopp.particle_guiding_center_boozer_perturbed_tracing(
            field,
            stz_inits[i, :],
            m,
            charge,
            speed_total,
            speed_par[i],
            mus[i],
            tmax,
            abstol,
            reltol,
            vacuum=(mode == 'gc_vac'),
            noK=(mode == 'gc_nok'),
            zetas=zetas,
            omegas=omegas,
            vpars=vpars,
            stopping_criteria=stopping_criteria,
            dt_save=dt_save, 
            zetas_stop=zetas_stop,
            vpars_stop=vpars_stop,
            Phihat=Phihat,
            omega=omega,
            Phim=Phim,
            Phin=Phin,
            phase=phase,
            forget_exact_path=forget_exact_path,
            axis=axis
        )
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_zeta_hits.append(np.asarray(res_zeta_hit))
        dtavg = res_ty[-1][0]/len(res_ty)
        logger.debug(f"{i+1:3d}/{nparticles}, t_final={res_ty[-1][0]}, average timestep {1000*dtavg:.10f}ms")
        if res_ty[-1][0] < tmax - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_zeta_hits = [i for o in comm.allgather(res_zeta_hits) for i in o]

    logger.debug(f'Particles lost {loss_ctr}/{nparticles}={(100*loss_ctr)//nparticles:d}%')
    return res_tys, res_zeta_hits

#TODO options
def trace_particles_boozer(field: BoozerMagneticField, stz_inits: RealArray,
                           parallel_speeds: RealArray, tmax=1e-4,
                           mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                           tol=1e-9, comm=None, zetas=[], omegas=[], vpars=[], stopping_criteria=[], dt_save=1e-6, mode='gc_vac',
                           forget_exact_path=False, solver_options=None):
    r"""
    Follow particles in a :class:`BoozerMagneticField`. This is modeled after
    :func:`trace_particles`.


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
        Ekin: kinetic energy in Joule, defaults to 3.52MeV. Either a scalar (assumed
              same energy for all particles), or ``(nparticles, )`` array. 
        tol: defualt tolerance for ode solver when solver-specific tolerances are not set
        comm: MPI communicator to parallelize over
        zetas: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that zeta should be computed
        TODO omega and vpar 
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`G` = const.,
                           :math:`I = 0`, and :math:`K = 0`.
            `gc_noK`: simplified guiding center equations for the case :math:`K = 0`.
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_zeta_hits is of
                           interest or one wants to reduce memory usage.
        solver_options: solver options
            shared options are
                `zetas_stop`: whether to stop if hit provided zeta planes
                `vpars_stop`: whether to stop if hit provided vpar planes
            default solver (rk45) specific options are
                `axis`: If `True`, tracing is performed in coordinates :math:`x = \sqrt{s}\cos(\theta)`
                           and :math:`y = \sqrt{s}\sin(\theta)` to avoid a coordinate singularity on
                           the magnetic axis. If `False`, tracing is performed in :math:`(s,\theta)`. 
                `reltol`: relative tolerance for adaptive ode solver
                `abstol`: absolute tolerance for adaptive ode solver
            symplectic solver specific options are 
                `solveSympl`: using symplectic solver
                `dt`: time step
                `roottol`: root solver tolerance
                `predictor_step`: provide better initial guess for the next time step 
                    using predictor steps

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, M)
            with M depending on the ``mode``.  Each row contains the time and
            the state.  So for `mode='gc'` and `mode='gc_vac'` the state
            consists of the :math:`(s,\theta,\zeta)` position and the parallel speed, hence
            each row contains `[t, s, t, z, v_par]`.

        - ``res_zeta_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the zeta planes or
            one of the stopping criteria. Each row of the array contains
            `[time] + [idx] + state`, where `idx` tells us which of the `zetas`
            or `stopping_criteria` was hit.  If `idx>=0`, then `zetas[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
    """
    if solver_options:
        options = dict(solver_options)
    else:
        options = dict()

    if options.get('zetas_stop') and (not len(zetas) and not len(omegas)):
        warn("No zetas and omegas provided for the zeta stopping critierion", RuntimeWarning)
    if options.get('vpars_stop') and (not len(vpars)):
        warn("No vpars provided for the vpar stopping criterion", RuntimeWarning)
    
    options['zetas_stop'] = options.pop('zetas_stop', False)

    if options.get('solveSympl'):
        for op in ['axis', 'reltol', 'abstol']:
            if options.get(op):
                warn("Symplectic solver does not use option %s" % op, RuntimeWarning)
    else:
        logger.debug(options)
        for op in ['dt', 'roottol', 'predictor_step']:
            if options.get(op):
                warn("RK45 solver does not use option %s" % op, RuntimeWarning)
                
    options.setdefault('roottol', tol)
    options.setdefault('reltol', tol)
    options.setdefault('abstol', tol)
    options.setdefault('dt', 1e-7)

    nparticles = stz_inits.shape[0]
    assert stz_inits.shape[0] == len(parallel_speeds)
    speed_par = parallel_speeds
    m = mass
    assert np.isscalar(Ekin) or (len(Ekin) == len(parallel_speeds)) 
    if np.isscalar(Ekin):
        Ekin = Ekin * np.ones((len(parallel_speeds),))
    speed_total = np.sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    mode = mode.lower()
    assert mode in ['gc', 'gc_vac', 'gc_nok']

    res_tys = []
    res_zeta_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        res_ty, res_zeta_hit = sopp.particle_guiding_center_boozer_tracing(
            field, stz_inits[i, :],
            m, charge, speed_total[i], speed_par[i], tmax, vacuum=(mode == 'gc_vac'),
            noK=(mode == 'gc_nok'), zetas=zetas, omegas=omegas, stopping_criteria=stopping_criteria, dt_save=dt_save, vpars=vpars, 
            **options)
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_zeta_hits.append(np.asarray(res_zeta_hit))
        dtavg = res_ty[-1][0]/len(res_ty)
        logger.debug(f"{i+1:3d}/{nparticles}, t_final={res_ty[-1][0]}, average timestep {1000*dtavg:.10f}ms")
        if res_ty[-1][0] < tmax - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_zeta_hits = [i for o in comm.allgather(res_zeta_hits) for i in o]
    logger.debug(f'Particles lost {loss_ctr}/{nparticles}={(100*loss_ctr)//nparticles:d}%')
    return res_tys, res_zeta_hits


def compute_resonances(res_tys, res_hits, ma=None, delta=1e-2):
    r"""
    Computes resonant particle orbits given the output of either
    :func:`trace_particles` or :func:`trace_particles_boozer`, ``res_tys`` and
    ``res_phi_hits``/``res_zeta_hits``, with ``forget_exact_path=False``.
    Resonance indicates a trajectory which returns to the same position
    at the :math:`\zeta = 0` plane after ``mpol`` poloidal turns and
    ``ntor`` toroidal turns. For the case of particles traced in a
    :class:`MagneticField` (not a :class:`BoozerMagneticField`), the poloidal
    angle is computed using the arctangent angle in the poloidal plane with
    respect to the coordinate axis, ``ma``,

    .. math::
        \theta = \tan^{-1} \left( \frac{R(\phi)-R_{\mathrm{ma}}(\phi)}{Z(\phi)-Z_{\mathrm{ma}}(\phi)} \right),

    where :math:`(R,\phi,Z)` are the cylindrical coordinates of the trajectory
    and :math:`(R_{\mathrm{ma}}(\phi),Z_{\mathrm{ma}(\phi)})` is the position
    of the coordinate axis.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``
        res_phi_hits: output of :func:`trace_particles` or
                :func:`trace_particles_boozer` with `phis/zetas = [0]`
        ma: an instance of :class:`Curve` representing the coordinate axis with
                respect to which the poloidal angle is computed. If the orbit is
                computed in flux coordinates, ``ma`` should be ``None``. (defaults to None)
        delta: the distance tolerance in the poloidal plane used to compute
                a resonant orbit. (defaults to 1e-2)

    Returns:
        resonances: list of 7d arrays containing resonant particle orbits. The
                elements of each array is ``[s0, theta0, zeta0, vpar0, t, mpol, ntor]``
                if ``ma=None``, and ``[R0, Z0, phi0, vpar0, t, mpol, ntor]`` otherwise.
                Here ``(s0, theta0, zeta0, vpar0)/(R0, Z0, phi0, vpar0)`` indicates the
                initial position and parallel velocity of the particle, ``t``
                indicates the time of the  resonance, ``mpol`` is the number of
                poloidal turns of the orbit, and ``ntor`` is the number of toroidal turns.
    """
    nparticles = len(res_tys)
    resonances = []
    gamma = np.zeros((1, 3))
    # Iterate over particles
    for ip in range(nparticles):
        nhits = len(res_hits[ip])
        s0 = res_tys[ip][0, 1]
        theta0 = res_tys[ip][0, 2]
        zeta0 = res_tys[ip][0, 3]
        theta0_mod = theta0 % (2*np.pi)
        zeta0_mod = zeta0 % (2*np.pi)
        x0 = s0 * np.cos(theta0)
        y0 = s0 * np.sin(theta0)
        vpar0 = res_tys[ip][0, 4]
        for it in range(1, nhits):
            # Check whether phi hit or stopping criteria achieved
            if int(res_phi_hits[ip][it, 1]) >= 0:
                s = res_phi_hits[ip][it, 2]
                theta = res_phi_hits[ip][it, 3]
                zeta = res_phi_hits[ip][it, 4]
                theta_mod = theta % 2*np.pi
                x = s * np.cos(theta)
                y = s * np.sin(theta)
                dist = np.sqrt((x-x0)**2 + (y-y0)**2)
                t = res_phi_hits[ip][it, 0]
                if dist < delta:
                    logger.debug('Resonance found.')
                    logger.debug(f'theta = {theta_mod}, theta0 = {theta0_mod}, s = {s}, s0 = {s0}')
                    mpol = np.rint((theta-theta0)/(2*np.pi))
                    ntor = np.rint((zeta-zeta0)/(2*np.pi))
                    resonances.append(np.asarray([s0, theta0, zeta0, vpar0, t, mpol, ntor]))
    return resonances

def compute_toroidal_transits(res_tys, flux=True):
    r"""
    Computes the number of toroidal transits of an orbit.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``.
        flux: if ``True``, ``res_tys`` represents the position in flux coordinates
                (should be ``True`` if computed from :func:`trace_particles_boozer`)
    Returns:
        ntransits: array with length ``len(res_tys)``. Each element contains the
                number of toroidal transits of the orbit.
    """
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        if flux:
            phi_init = res_tys[ip][0, 3]
        else:
            phi_init = sopp.get_phi(res_tys[ip][0, 1], res_tys[ip][0, 2], np.pi)
        phi_prev = phi_init
        for it in range(1, ntraj):
            if flux:
                phi = res_tys[ip][it, 3]
            else:
                phi = sopp.get_phi(res_tys[ip][it, 1], res_tys[ip][it, 2], phi_prev)
            phi_prev = phi
        if ntraj > 1:
            ntransits[ip] = np.round((phi - phi_init)/(2*np.pi))
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
    gamma = np.zeros((1, 3))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        theta_init = res_tys[ip][0, 2]
        theta_prev = theta_init
        for it in range(1, ntraj):
            theta = res_tys[ip][it, 2]
            theta_prev = theta
        if ntraj > 1:
            ntransits[ip] = np.round((theta - theta_init)/(2*np.pi))
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

class VparStoppingCriterion(sopp.VparStoppingCriterion):
    """
    Stop the iteration once the maximum number of iterations is reached.
    """
    pass

class ZetaStoppingCriterion(sopp.ZetaStoppingCriterion):
    """
    Stop the iteration once the maximum number of iterations is reached.
    """
    pass
