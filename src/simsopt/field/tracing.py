import logging
from math import sqrt

import numpy as np
# from nptyping import NDArray, Float

import simsoptpp as sopp
from .._core.util import parallel_loop_bounds
from ..field.magneticfield import MagneticField
from ..field.boozermagneticfield import BoozerMagneticField
from ..field.sampling import draw_uniform_on_curve, draw_uniform_on_surface
from ..geo.surface import SurfaceClassifier
from ..util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from .._core.types import RealArray


logger = logging.getLogger(__name__)

__all__ = ['SurfaceClassifier', 'LevelsetStoppingCriterion',
           'MinToroidalFluxStoppingCriterion', 'MaxToroidalFluxStoppingCriterion',
           'IterationStoppingCriterion', 'ToroidalTransitStoppingCriterion',
           'compute_fieldlines', 'compute_resonances',
           'compute_poloidal_transits', 'compute_toroidal_transits',
           'trace_particles', 'trace_particles_boozer',
           'trace_particles_starting_on_curve',
           'trace_particles_starting_on_surface',
           'particles_to_vtk', 'plot_poincare_data']


def compute_gc_radius(m, vperp, q, absb):
    """
    Computes the gyro radius of a particle in a field with strenght ``absb```,
    that is ``r=m*vperp/(abs(q)*absb)``.
    """

    return m*vperp/(abs(q)*absb)


def gc_to_fullorbit_initial_guesses(field, xyz_inits, speed_pars, speed_total, m, q, eta=0):
    """
    Takes in guiding center positions ``xyz_inits`` as well as a parallel
    speeds ``speed_pars`` and total velocities ``speed_total`` to compute orbit
    positions for a full orbit calculation that matches the given guiding
    center. The phase angle can be controll via the `eta` parameter
    """

    nparticles = xyz_inits.shape[0]
    xyz_inits_full = np.zeros_like(xyz_inits)
    v_inits = np.zeros((nparticles, 3))
    rgs = np.zeros((nparticles, ))
    field.set_points(xyz_inits)
    Bs = field.B()
    AbsBs = field.AbsB()
    eB = Bs/AbsBs
    for i in range(nparticles):
        p1 = eB[i, :]
        p2 = np.asarray([0, 0, 1])
        p3 = -np.cross(p1, p2)
        p3 *= 1./np.linalg.norm(p3)
        q1 = p1
        q2 = p2 - np.sum(q1*p2)*q1
        q2 = q2/np.sum(q2**2)**0.5
        q3 = p3 - np.sum(q1*p3)*q1 - np.sum(q2*p3)*q2
        q3 = q3/np.sum(q3**2)**0.5
        speed_perp = np.sqrt(speed_total**2 - speed_pars[i]**2)
        rgs[i] = compute_gc_radius(m, speed_perp, q, AbsBs[i, 0])

        xyz_inits_full[i, :] = xyz_inits[i, :] + rgs[i] * np.sin(eta) * q2 + rgs[i] * np.cos(eta) * q3
        vperp = -speed_perp * np.cos(eta) * q2 + speed_perp * np.sin(eta) * q3
        v_inits[i, :] = speed_pars[i] * q1 + vperp
    return xyz_inits_full, v_inits, rgs


def trace_particles_boozer(field: BoozerMagneticField,
                           stz_inits: RealArray,  # NDArray[Float],
                           parallel_speeds: RealArray,  # NDArray[Float],
                           tmax=1e-4,
                           mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                           tol=1e-9, comm=None, zetas=[], stopping_criteria=[], mode='gc_vac', forget_exact_path=False):
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

    nparticles = stz_inits.shape[0]
    assert stz_inits.shape[0] == len(parallel_speeds)
    speed_par = parallel_speeds
    m = mass
    speed_total = sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    mode = mode.lower()
    assert mode in ['gc', 'gc_vac', 'gc_nok']

    res_tys = []
    res_zeta_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        res_ty, res_zeta_hit = sopp.particle_guiding_center_boozer_tracing(
            field, stz_inits[i, :],
            m, charge, speed_total, speed_par[i], tmax, tol, vacuum=(mode == 'gc_vac'),
            noK=(mode == 'gc_nok'), zetas=zetas, stopping_criteria=stopping_criteria)
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


def trace_particles(field: MagneticField,
                    xyz_inits: RealArray,  # NDArray[Float],
                    parallel_speeds: RealArray,  # NDArray[Float],
                    tmax=1e-4,
                    mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                    tol=1e-9, comm=None, phis=[], stopping_criteria=[], mode='gc_vac', forget_exact_path=False,
                    phase_angle=0):
    r"""
    Follow particles in a magnetic field.

    In the case of ``mod='full'`` we solve

    .. math::

        [\ddot x, \ddot y, \ddot z] = \frac{q}{m}  [\dot x, \dot y, \dot z] \times B

    in the case of ``mod='gc_vac'`` we solve the guiding center equations under
    the assumption :math:`\nabla p=0`, that is

    .. math::

        [\dot x, \dot y, \dot z] &= v_{||}\frac{B}{|B|} + \frac{m}{q|B|^3}  (0.5v_\perp^2 + v_{||}^2)  B\times \nabla(|B|)\\
        \dot v_{||}    &= -\mu  (B \cdot \nabla(|B|))

    where :math:`v_\perp = 2\mu|B|`. See equations (12) and (13) of
    [Guiding Center Motion, H.J. de Blank, https://doi.org/10.13182/FST04-A468].

    Args:
        field: The magnetic field :math:`B`.
        xyz_inits: A (nparticles, 3) array with the initial positions of the particles.
        parallel_speeds: A (nparticles, ) array containing the speed in direction of the B field
                         for each particle.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, M)
            with M depending on the ``mode``.  Each row contains the time and
            the state.  So for `mode='gc'` and `mode='gc_vac'` the state
            consists of the xyz position and the parallel speed, hence
            each row contains `[t, x, y, z, v_par]`.  For `mode='full'`, the
            state consists of position and velocity vector, i.e. each row
            contains `[t, x, y, z, vx, vy, vz]`.

        - ``res_phi_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the phi planes or
            one of the stopping criteria. Each row of the array contains
            `[time] + [idx] + state`, where `idx` tells us which of the `phis`
            or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
    """

    nparticles = xyz_inits.shape[0]
    assert xyz_inits.shape[0] == len(parallel_speeds)
    speed_par = parallel_speeds
    mode = mode.lower()
    assert mode in ['gc', 'gc_vac', 'full']
    m = mass
    speed_total = sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)

    if mode == 'full':
        xyz_inits, v_inits, _ = gc_to_fullorbit_initial_guesses(field, xyz_inits, speed_par, speed_total, m, charge, eta=phase_angle)
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        if 'gc' in mode:
            res_ty, res_phi_hit = sopp.particle_guiding_center_tracing(
                field, xyz_inits[i, :],
                m, charge, speed_total, speed_par[i], tmax, tol,
                vacuum=(mode == 'gc_vac'), phis=phis, stopping_criteria=stopping_criteria)
        else:
            res_ty, res_phi_hit = sopp.particle_fullorbit_tracing(
                field, xyz_inits[i, :], v_inits[i, :],
                m, charge, tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_phi_hits.append(np.asarray(res_phi_hit))
        dtavg = res_ty[-1][0]/len(res_ty)
        logger.debug(f"{i+1:3d}/{nparticles}, t_final={res_ty[-1][0]}, average timestep {1000*dtavg:.10f}ms")
        if res_ty[-1][0] < tmax - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_phi_hits = [i for o in comm.allgather(res_phi_hits) for i in o]
    logger.debug(f'Particles lost {loss_ctr}/{nparticles}={(100*loss_ctr)//nparticles:d}%')
    return res_tys, res_phi_hits


def trace_particles_starting_on_curve(curve, field, nparticles, tmax=1e-4,
                                      mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                                      Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                                      tol=1e-9, comm=None, seed=1, umin=-1, umax=+1,
                                      phis=[], stopping_criteria=[], mode='gc_vac', forget_exact_path=False,
                                      phase_angle=0):
    r"""
    Follows particles spawned at random locations on the magnetic axis with random pitch angle.
    See :mod:`simsopt.field.tracing.trace_particles` for the governing equations.

    Args:
        curve: The :mod:`simsopt.geo.curve.Curve` to spawn the particles on. Uses rejection sampling
               to sample points on the curve. *Warning*: assumes that the underlying
               quadrature points on the Curve are uniformly distributed.
        field: The magnetic field :math:`B`.
        nparticles: number of particles to follow.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        seed: random seed
        umin: the parallel speed is defined as  ``v_par = u * speed_total``
              where  ``u`` is drawn uniformly in ``[umin, umax]``
        umax: see ``umin``
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: see :mod:`simsopt.field.tracing.trace_particles`
    """
    m = mass
    speed_total = sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    np.random.seed(seed)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles, ))
    speed_par = us*speed_total
    xyz, _ = draw_uniform_on_curve(curve, nparticles, safetyfactor=10)
    return trace_particles(
        field, xyz, speed_par, tmax=tmax, mass=mass, charge=charge,
        Ekin=Ekin, tol=tol, comm=comm, phis=phis,
        stopping_criteria=stopping_criteria, mode=mode, forget_exact_path=forget_exact_path,
        phase_angle=phase_angle)


def trace_particles_starting_on_surface(surface, field, nparticles, tmax=1e-4,
                                        mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                                        Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                                        tol=1e-9, comm=None, seed=1, umin=-1, umax=+1,
                                        phis=[], stopping_criteria=[], mode='gc_vac', forget_exact_path=False,
                                        phase_angle=0):
    r"""
    Follows particles spawned at random locations on the magnetic axis with random pitch angle.
    See :mod:`simsopt.field.tracing.trace_particles` for the governing equations.

    Args:
        surface: The :mod:`simsopt.geo.surface.Surface` to spawn the particles
                 on. Uses rejection sampling to sample points on the curve. *Warning*:
                 assumes that the underlying quadrature points on the Curve as uniformly
                 distributed.
        field: The magnetic field :math:`B`.
        nparticles: number of particles to follow.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        seed: random seed
        umin: the parallel speed is defined as  ``v_par = u * speed_total``
              where  ``u`` is drawn uniformly in ``[umin, umax]``.
        umax: see ``umin``
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: see :mod:`simsopt.field.tracing.trace_particles`
    """
    m = mass
    speed_total = sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    np.random.seed(seed)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles, ))
    speed_par = us*speed_total
    xyz, _ = draw_uniform_on_surface(surface, nparticles, safetyfactor=10)
    return trace_particles(
        field, xyz, speed_par, tmax=tmax, mass=mass, charge=charge,
        Ekin=Ekin, tol=tol, comm=comm, phis=phis,
        stopping_criteria=stopping_criteria, mode=mode, forget_exact_path=forget_exact_path,
        phase_angle=phase_angle)


def compute_resonances(res_tys, res_phi_hits, ma=None, delta=1e-2):
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
    flux = False
    if ma is None:
        flux = True
    nparticles = len(res_tys)
    resonances = []
    gamma = np.zeros((1, 3))
    # Iterate over particles
    for ip in range(nparticles):
        nhits = len(res_phi_hits[ip])
        if (flux):
            s0 = res_tys[ip][0, 1]
            theta0 = res_tys[ip][0, 2]
            zeta0 = res_tys[ip][0, 3]
            theta0_mod = theta0 % (2*np.pi)
            zeta0_mod = zeta0 % (2*np.pi)
            x0 = s0 * np.cos(theta0)
            y0 = s0 * np.sin(theta0)
        else:
            X0 = res_tys[ip][0, 1]
            Y0 = res_tys[ip][0, 2]
            Z0 = res_tys[ip][0, 3]
            R0 = np.sqrt(X0**2 + Y0**2)
            phi0 = np.arctan2(Y0, X0)
            ma.gamma_impl(gamma, phi0/(2*np.pi))
            R_ma0 = np.sqrt(gamma[0, 0]**2 + gamma[0, 1]**2)
            Z_ma0 = gamma[0, 2]
            theta0 = np.arctan2(Z0-Z_ma0, R0-R_ma0)
        vpar0 = res_tys[ip][0, 4]
        for it in range(1, nhits):
            # Check whether phi hit or stopping criteria achieved
            if int(res_phi_hits[ip][it, 1]) >= 0:
                if (flux):
                    s = res_phi_hits[ip][it, 2]
                    theta = res_phi_hits[ip][it, 3]
                    zeta = res_phi_hits[ip][it, 4]
                    theta_mod = theta % 2*np.pi
                    x = s * np.cos(theta)
                    y = s * np.sin(theta)
                    dist = np.sqrt((x-x0)**2 + (y-y0)**2)
                else:
                    # Check that distance is less than delta
                    X = res_phi_hits[ip][it, 2]
                    Y = res_phi_hits[ip][it, 3]
                    R = np.sqrt(X**2 + Y**2)
                    Z = res_phi_hits[ip][it, 4]
                    dist = np.sqrt((R-R0)**2 + (Z-Z0)**2)
                t = res_phi_hits[ip][it, 0]
                if dist < delta:
                    logger.debug('Resonance found.')
                    if flux:
                        logger.debug(f'theta = {theta_mod}, theta0 = {theta0_mod}, s = {s}, s0 = {s0}')
                        mpol = np.rint((theta-theta0)/(2*np.pi))
                        ntor = np.rint((zeta-zeta0)/(2*np.pi))
                        resonances.append(np.asarray([s0, theta0, zeta0, vpar0, t, mpol, ntor]))
                    else:
                        # Find index of closest point along trajectory
                        indexm = np.argmin(np.abs(res_tys[ip][:, 0]-t))
                        # Compute mpol and ntor for neighboring points as well
                        indexl = indexm-1
                        indexr = indexm+1
                        dtl = np.abs(res_tys[ip][indexl, 0]-t)
                        trajlistl = []
                        trajlistl.append(res_tys[ip][0:indexl+1, :])
                        mpoll = np.abs(compute_poloidal_transits(trajlistl, ma, flux))
                        ntorl = np.abs(compute_toroidal_transits(trajlistl, flux))
                        logger.debug(f'dtl ={dtl}, mpoll = {mpoll}, ntorl = {ntorl}, tl={res_tys[ip][indexl,0]}')
                        logger.debug(f'(R,Z)l = {np.sqrt(res_tys[ip][indexl,1]**2 + res_tys[ip][indexl,2]**2),res_tys[ip][indexl,3]}')

                        trajlistm = []
                        dtm = np.abs(res_tys[ip][indexm, 0]-t)
                        trajlistm.append(res_tys[ip][0:indexm+1, :])
                        mpolm = np.abs(compute_poloidal_transits(trajlistm, ma, flux))
                        ntorm = np.abs(compute_toroidal_transits(trajlistm, flux))
                        logger.debug(f'dtm ={dtm}, mpolm = {mpolm}, ntorm = {ntorm}, tm={res_tys[ip][indexm,0]}')
                        logger.debug(f'(R,Z)m = {np.sqrt(res_tys[ip][indexm,1]**2 + res_tys[ip][indexm,2]**2),res_tys[ip][indexm,3]}')

                        mpolr = 0
                        ntorr = 0
                        if indexr < len(res_tys[ip][:, 0]):
                            trajlistr = []
                            dtr = np.abs(res_tys[ip][indexr, 0]-t)
                            trajlistr.append(res_tys[ip][0:indexr+1, :])
                            # Take maximum over neighboring points to catch near resonances
                            mpolr = np.abs(compute_poloidal_transits(trajlistr, ma, flux))
                            ntorr = np.abs(compute_toroidal_transits(trajlistr, flux))
                            logger.debug(f'dtr ={dtr}, mpolr = {mpolr}, ntorr = {ntorr}, tr={res_tys[ip][indexr,0]}')
                            logger.debug(f'(R,Z)r = {np.sqrt(res_tys[ip][indexr,1]**2 + res_tys[ip][indexr,2]**2),res_tys[ip][indexr,3]}')

                        mpol = np.amax([mpoll, mpolm, mpolr])
                        index_mpol = np.argmax([mpoll, mpolm, mpolr])
                        ntor = np.amax([ntorl, ntorm, ntorr])
                        index_ntor = np.argmax([ntorl, ntorm, ntorr])
                        index = np.amax([index_mpol, index_ntor])
                        resonances.append(np.asarray([R0, Z0, phi0, vpar0, t, mpol, ntor]))
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
    if not flux:
        assert (ma is not None)
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    gamma = np.zeros((1, 3))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        if flux:
            theta_init = res_tys[ip][0, 2]
        else:
            R_init = np.sqrt(res_tys[ip][0, 1]**2 + res_tys[ip][0, 2]**2)
            Z_init = res_tys[ip][0, 3]
            phi_init = np.arctan2(res_tys[ip][0, 2], res_tys[ip][0, 1])
            ma.gamma_impl(gamma, phi_init/(2*np.pi))
            R_ma = np.sqrt(gamma[0, 0]**2 + gamma[0, 1]**2)
            Z_ma = gamma[0, 2]
            theta_init = sopp.get_phi(R_init-R_ma, Z_init-Z_ma, np.pi)
        theta_prev = theta_init
        for it in range(1, ntraj):
            if flux:
                theta = res_tys[ip][it, 2]
            else:
                phi = np.arctan2(res_tys[ip][it, 2], res_tys[ip][it, 1])
                ma.gamma_impl(gamma, phi/(2*np.pi))
                R_ma = np.sqrt(gamma[0, 0]**2 + gamma[0, 1]**2)
                Z_ma = gamma[0, 2]
                R = np.sqrt(res_tys[ip][it, 1]**2 + res_tys[ip][it, 2]**2)
                Z = res_tys[ip][it, 3]
                theta = sopp.get_phi(R-R_ma, Z-Z_ma, theta_prev)
            theta_prev = theta
        if ntraj > 1:
            ntransits[ip] = np.round((theta - theta_init)/(2*np.pi))
    return ntransits


def compute_fieldlines(field, R0, Z0, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None):
    r"""
    Compute magnetic field lines by solving

    .. math::

        [\dot x, \dot y, \dot z] = B(x, y, z)

    Args:
        field: the magnetic field :math:`B`
        R0: list of radial components of initial points
        Z0: list of vertical components of initial points
        tmax: for how long to trace. will do roughly ``|B|*tmax/(2*pi*r0)`` revolutions of the device
        tol: tolerance for the adaptive ode solver
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, 4).
            Each row contains the time and
            the position, i.e.`[t, x, y, z]`.
        - ``res_phi_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the phi planes or
            one of the stopping criteria. Each row of the array contains
            `[time, idx, x, y, z]`, where `idx` tells us which of the `phis`
            or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
    """
    assert len(R0) == len(Z0)
    nlines = len(R0)
    xyz_inits = np.zeros((nlines, 3))
    xyz_inits[:, 0] = np.asarray(R0)
    xyz_inits[:, 2] = np.asarray(Z0)
    res_tys = []
    res_phi_hits = []
    first, last = parallel_loop_bounds(comm, nlines)
    for i in range(first, last):
        res_ty, res_phi_hit = sopp.fieldline_tracing(
            field, xyz_inits[i, :],
            tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(np.asarray(res_phi_hit))
        dtavg = res_ty[-1][0]/len(res_ty)
        logger.debug(f"{i+1:3d}/{nlines}, t_final={res_ty[-1][0]}, average timestep {dtavg:.10f}s")
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_phi_hits = [i for o in comm.allgather(res_phi_hits) for i in o]
    return res_tys, res_phi_hits


def particles_to_vtk(res_tys, filename):
    """
    Export particle tracing or field lines to a vtk file.
    Expects that the xyz positions can be obtained by ``xyz[:, 1:4]``.
    """
    from pyevtk.hl import polyLinesToVTK
    x = np.concatenate([xyz[:, 1] for xyz in res_tys])
    y = np.concatenate([xyz[:, 2] for xyz in res_tys])
    z = np.concatenate([xyz[:, 3] for xyz in res_tys])
    ppl = np.asarray([xyz.shape[0] for xyz in res_tys])
    data = np.concatenate([i*np.ones((res_tys[i].shape[0], )) for i in range(len(res_tys))])
    polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})


class LevelsetStoppingCriterion(sopp.LevelsetStoppingCriterion):
    r"""
    Based on a scalar function :math:`f:R^3\to R`, this criterion checks whether
    :math:`f(x, y, z) < 0` and stops the iteration once this is true.

    The idea is to use this for example with signed distance functions to a surface.
    """

    def __init__(self, classifier):
        assert isinstance(classifier, SurfaceClassifier) \
            or isinstance(classifier, sopp.RegularGridInterpolant3D)
        if isinstance(classifier, SurfaceClassifier):
            sopp.LevelsetStoppingCriterion.__init__(self, classifier.dist)
        else:
            sopp.LevelsetStoppingCriterion.__init__(self, classifier)


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


def plot_poincare_data(fieldlines_phi_hits, phis, filename, mark_lost=False, aspect='equal', dpi=300):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    import matplotlib.pyplot as plt
    from math import ceil, sqrt
    nrowcol = ceil(sqrt(len(phis)))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(16, 10))
    color = None
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
        axs[row, col].set_xlabel("$r$")
        axs[row, col].set_ylabel("$z$")
        axs[row, col].set_aspect(aspect)
        axs[row, col].tick_params(direction="in")
        for j in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            if mark_lost:
                color = 'r' if lost else 'g'
            data_this_phi = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
            axs[row, col].scatter(r, data_this_phi[:, 4], marker='o', s=0.2, linewidths=0, c=color)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
