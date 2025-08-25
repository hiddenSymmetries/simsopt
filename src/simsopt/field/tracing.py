import logging
from math import sqrt

import numpy as np

import simsoptpp as sopp
from .._core import Optimizable, ObjectiveFailure
from .._core.util import parallel_loop_bounds
from ..field.magneticfield import MagneticField, InterpolatedField
from ..field.boozermagneticfield import BoozerMagneticField
from ..field.sampling import draw_uniform_on_curve, draw_uniform_on_surface
from ..geo.surface import SurfaceClassifier
from ..util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from .._core.types import RealArray
from numpy.typing import NDArray
from typing import Union, Iterable
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

__all__ = ['SurfaceClassifier', 'LevelsetStoppingCriterion',
           'MinToroidalFluxStoppingCriterion', 'MaxToroidalFluxStoppingCriterion',
           'MinRStoppingCriterion', 'MinZStoppingCriterion',
           'MaxRStoppingCriterion', 'MaxZStoppingCriterion',
           'IterationStoppingCriterion', 'ToroidalTransitStoppingCriterion',
           'compute_fieldlines', 'compute_resonances',
           'compute_poloidal_transits', 'compute_toroidal_transits',
           'trace_particles', 'trace_particles_boozer',
           'trace_particles_starting_on_curve',
           'trace_particles_starting_on_surface',
           'particles_to_vtk', 'plot_poincare_data', 'ScipyFieldlineIntegrator',
           'SimsoptFieldlineIntegrator']


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
                           stz_inits: RealArray,
                           parallel_speeds: RealArray,
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
                    xyz_inits: RealArray,
                    parallel_speeds: RealArray,
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
                        # index_mpol = np.argmax([mpoll, mpolm, mpolr])
                        ntor = np.amax([ntorl, ntorm, ntorr])
                        # index_ntor = np.argmax([ntorl, ntorm, ntorr])
                        # index = np.amax([index_mpol, index_ntor])
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
        comm: MPI communicator to parallelize over

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


class MinRStoppingCriterion(sopp.MinRStoppingCriterion):
    """
    Stop the iteration once a particle falls below a critical value of
    ``R``, the radial cylindrical coordinate. 

    Usage:

    .. code-block::

        stopping_criteria=[MinRStopingCriterion(crit_r)]

    where ``crit_r`` is the value of the critical coordinate.
    """
    pass


class MinZStoppingCriterion(sopp.MinZStoppingCriterion):
    """
    Stop the iteration once a particle falls below a critical value of
    ``Z``, the cylindrical vertical coordinate. 

    Usage:

    .. code-block::

        stopping_criteria=[MinZStopingCriterion(crit_z)]

    where ``crit_z`` is the value of the critical coordinate.
    """
    pass


class MaxRStoppingCriterion(sopp.MaxRStoppingCriterion):
    """
    Stop the iteration once a particle goes above a critical value of
    ``R``, the radial cylindrical coordinate. 

    Usage:

    .. code-block::

        stopping_criteria=[MaxRStopingCriterion(crit_r)]

    where ``crit_r`` is the value of the critical coordinate.
    """
    pass


class MaxZStoppingCriterion(sopp.MaxZStoppingCriterion):
    """
    Stop the iteration once a particle gove above a critical value of
    ``Z``, the cylindrical vertical coordinate. 

    Usage:

    .. code-block::

        stopping_criteria=[MaxZStopingCriterion(crit_z)]

    where ``crit_z`` is the value of the critical coordinate.
    """
    pass


def plot_poincare_data(fieldlines_phi_hits, phis, filename, mark_lost=False, aspect='equal', dpi=300, xlims=None,
                       ylims=None, surf=None, s=2, marker='o'):
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
    plt.figure()
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    color = None
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        if i != len(phis) - 1:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc='left', y=0.0)
        else:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc='right', y=0.0)
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$r$")
        if col == 0:
            axs[row, col].set_ylabel("$z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            if mark_lost:
                color = 'r' if lost else 'g'
            data_this_phi = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
            axs[row, col].scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)

        plt.rc('axes', axisbelow=True)
        axs[row, col].grid(True, linewidth=0.5)

        # if passed a surface, plot the plasma surface outline
        if surf is not None:
            cross_section = surf.cross_section(phi=phis[i]/(2.0*np.pi))
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c='k')

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


class Integrator(Optimizable):
    """
    Base class for Integrators. 
    Integrators provide a method `compute_poincare_hits` which returns 
    res_tys and res_phi_hits that interfaces with the PoincarePlotter. 
    It can also provide other integration services on a MagneticField. 
    and being an Optimizable, the results of integrations can be
    optimized. 
    """
    def __init__(self, field:MagneticField, comm=None, nfp=None, stellsym=False, R0=None, test_symmetries=True):
        """
        Args:
            field: the magnetic field to be used for the integration
            nfp: number of field periods
            stellsym: if True, the field is assumed to be stellarator symmetric
            R0: major radius of the device. Used for testing symmetries of the field
            test_symmetries: (bool): if True, test if the values of nfp and stellsym
                are consistent with the field. If False, no test is performed.  
        """
        self.field = field
        self.comm = comm
        if nfp is None:
            self.nfp = 1
        else:
            self.nfp = nfp
        if R0 is None:
            logger.warning("R0 is not set, using default value of 1.")
            self.R0 = 1
        else:
            self.R0 = R0
        self.stellsym = stellsym
        self._test_symmetries = test_symmetries
        if test_symmetries:
            self.test_symmetries()
        if isinstance(self.field, InterpolatedField):
            if self.field.nfp != self.nfp:
                raise ValueError(f"Field has {self.field.nfp} field periods, but integrator is set to {self.nfp}.")
            if self.field.stellsym != self.stellsym:
                raise ValueError(f"Field is {'not ' if not self.field.stellsym else ''}stellarator symmetric, but integrator is set to {'not ' if not self.stellsym else ''}be.")
            self.interpolated = True
        Optimizable.__init__(depends_on=self.field)

    def test_symmetries(self):
        """
        Test if the number of field periods and the 
        stellarator symmetry setting are consistent
        for the given field.
        """
        rphiz_point = np.array([self.R0, 0.1, .1])
        rphiz_stellsym = np.array([self.R0, -0.1, -0.1])
        rphiz_periodicity = np.array([self.R0, 0.1+(2*np.pi/self.nfp), 0.1])
        self.field.set_points_cyl(rphiz_point)
        test_field = np.copy(self.field.B_cyl())
        self.field.set_points_cyl(rphiz_stellsym)
        test_field_stellsym = np.copy(self.field.B_cyl())
        self.field.set_points_cyl(rphiz_periodicity)
        test_field_periodicity = np.copy(self.field.B_cyl())
        stellsym_test = np.allclose(test_field, test_field_stellsym*np.array([-1,1,1]))
        periodicity_test = np.allclose(test_field, test_field_periodicity)
        if self.stellsym and not stellsym_test:
            raise ValueError("The field is not stellarator symmetric, but the integrator is set to be.")
        if not periodicity_test:
            raise ValueError(f"The field does not adhere to the {self.nfp} field periods it is set to be.")
        
    def Interpolate_field(self, rrange, phirange, zrange, degree=2, skip=None, extrapolate=True): 
        """
        Replace the field of the integrator with an InterpolatedField, which
        takes a while to compute, but is much faster to evaluate. 
        For a single integration, this is often not worth it, but a Poincare 
        calculation requiring hundreds of field lines integrated thousands of times
        should be calculated on an InterpolatedField. 

        Args:
            rrange: range of the R coordinate
            phirange: range of the phi coordinate
            zrange: range of the Z coordinate
            degree: degree of the interpolation (default 2)
            skip: A function that takes in a point (r,phi,z) and returns True if the point should be skipped
                
            extrapolate: if True, extrapolate outside the range. 
                If False, raise an error if outside the range.
        """
        if self.interpolated:
            raise ValueError("Field is already interpolated.")
        interpolatedfield = InterpolatedField(self.field, rrange, phirange, zrange, degree=degree, skip=skip, extrapolate=extrapolate)
        self.field = interpolatedfield
        self.interpolated = True

    @staticmethod
    def _rphiz_to_xyz(array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert an array of R, phi, Z to x, y, z
        """
        if array.ndim == 1 & array.shape[0] == 3:
            array2 = array[None, :]
        elif array.ndim == 2 & array.shape[1] == 3:
            array2 = array[:, :]
        else: 
            raise ValueError("Input array must be of shape (3,) or (n,3)")

        return np.array([array2[:, 0]*np.cos(array2[:, 1]), array2[:, 0]*np.sin(array[:, 1]), array[:, 2]]).T

    @staticmethod
    def _xyz_to_rphiz(array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert an array of x, y, z to R, phi, Z
        """
        if array.ndim == 1 & array.shape[0] == 3:
            array2 = array[None, :]
        elif array.ndim == 2 & array.shape[1] == 3:
            array2 = array[:, :]
        else: 
            raise ValueError("Input array must be of shape (3,) or (n,3)")
        
        return np.array([np.sqrt(array2[:, 0]**2 + array2[:, 1]**2), np.arctan2(array2[:, 1], array2[:, 0]), array2[:, 2]]).T


class SimsoptFieldlineIntegrator(Integrator):
    """
    Integration of field lines using the `simsoptpp` routines. 
    Integration is performed in cylindrical coordinates. 
    MPI parallelization is supported. 
    
    """
    def __init__(self, field, comm=None, nfp=None, stellsym=False, R0=None, test_symmetries=True, stopping_criteria=[], tol=1e-9):
        self.tol = tol
        self.stopping_criteria = stopping_criteria
        super().__init__(field, comm=comm, nfp=nfp, stellsym=stellsym, R0=R0, test_symmetries=test_symmetries)
    
    def compute_poincare_hits(self, start_points_RZ, n_hits, phis=[],tmax=200):
        """
        calculate the res_tys and res_phi_hits. 
        Args:
            start_points_RZ: nx2 array of starting radial coordinates.
            n_hits: number of hits to compute for each field line
            phis: list of angles in [0, 2pi] for which intersection with the plane
                  corresponding to that phi should be computed. If empty, only phi=0 is used.
        Returns:
            res_tys: list of numpy arrays (one for each particle) describing the
                     solution over time. The numpy array is of shape (ntimesteps, 4).
                     Each row contains the time and the position, i.e.`[t, x, y, z]`.
            res_phi_hits: list of numpy arrays (one for each particle) containing
                     information on each time the particle hits one of the phi planes.
                     Each row of the array contains `[time, idx, x, y, z]`, where `idx`
                     tells us which of the `phis` was hit. If `idx>=0`, then
                     `phis[int(idx)]` was hit. If `idx<0`, then one of the stopping criteria
                     was hit, which is specified in the `stopping_criteria` argument.
                  """
        return compute_fieldlines(
            self.field, start_points_RZ[:,0], start_points_RZ[:,1], tmax=tmax, tol=self.tol,
            phis=phis, stopping_criteria=self.stopping_criteria, comm=self.comm)
    
    def integrate_in_phi_cart(self, start_xyz, delta_phi=2*np.pi, return_cartesian=False):
        """
        Integrate the field line in phi from phi_start to phi_end using simsoptpp routines.  

        Args:
            start_R: starting R coordinate
            start_Z: starting Z coordinate
            phi_end: ending phi coordinate
            phi_start: starting phi coordinate (default 0)
            return_cartesian: if True, return the cartesian coordinates of the end point, otherwise return only R,Z (phi_end assumed)
        """
        phi_start = self._xyz_to_rphiz(start_xyz)[0,1]
        phi_end = phi_start + delta_phi
        if isinstance(self.field, BoozerMagneticField): #can InterpolatedField also be a flux field, or is this only for regular and BiotSavart fields?
            flux=True
        else: 
            flux=False
        self.field.set_points(start_xyz)
        #test direction of the field and switch if necessary
        Bphi = self.field.B_cyl()[0,1]
        if Bphi < 0:
            field_for_tracing = (-1.0) * self.field
        else:
            field_for_tracing = self.field

        _, res_phi_hits = sopp.fieldline_tracing(
            field_for_tracing, 
            start_xyz, 
            tmax=200, 
            tol=self.tol, 
            phis=[phi_end], 
            stopping_criteria=[ToroidalTransitStoppingCriterion(1, flux=flux),]
            )
        
        if return_cartesian: #return [x,y,z] array
            return res_phi_hits[-1][2:]
        else: #return [R,Z] array
            return self._xyz_to_rphiz(res_phi_hits[-1][2:])

    def integrate_in_phi_cyl(self, start_RZ, start_phi, delta_phi=2*np.pi, return_cartesian=True):
        """
        Integrate the field line giving thes tarting location in cylindrical coordinates.
        Integrate the field line over an angle phi from the starting point given by 
        cartesian coordinates (x,y,z).
        """
        start_xyz = self._rphiz_to_xyz(np.array([start_RZ[0], start_phi, start_RZ[1]]))
        return self.integrate_in_phi_cart(
            start_xyz, delta_phi=delta_phi, return_cartesian=return_cartesian)

    def get_fieldlinepoints_cart(self, start_xyz, n_transits=1,  return_cartesian=True):
        """
        Calculate a string of points along a field line starting at 
        (start_R, start_Z, start_phi) and going to (start_R, start_Z, start_phi+delta_phi).
        The points are returned in the form of a numpy array of shape (npoints, 3). 
        If return_cartesian is True, the points are in cartesian coordinates (x,y,z), otherwise they are in (R, phi, Z). 

        The SimsoptFieldLineIntegrator does not give control over the number of points, but relies
        on the adaptive step size of the integrator. The number of points returned will depend on the field
        complexity and the tolerance set for the integrator.

        Args:
            start_R: starting R coordinate
            start_Z: starting Z coordinate
            start_phi: starting phi coordinate (default 0)
            delta_phi: ending phi coordinate (default 0)
            return_cartesian: if True, return the points in cartesian coordinates, otherwise return Cylindrical coordinates (default False). 
        """
        if isinstance(self.field, BoozerMagneticField): #can InterpolatedField also be a flux field, or is this only for regular and BiotSavart fields?
            flux=True
        else: 
            flux=False
        self.field.set_points(start_xyz)
        #test direction of the field and switch if necessary
        Bphi = self.field.B_cyl()[0,1]
        if Bphi < 0:
            field_for_tracing = (-1.0) * self.field
        else:
            field_for_tracing = self.field

        res_tys, res_phi_hits = sopp.fieldline_tracing(
            field_for_tracing, 
            start_xyz, 
            tmax=200, 
            tol=self.tol, 
            phis=[0,], 
            stopping_criteria=[ToroidalTransitStoppingCriterion(n_transits, flux=flux),])
        
        points_cart = np.array(res_tys[:,1:])
        if return_cartesian:
            return points_cart
        else: 
            return self._xyz_to_rphiz(points_cart)

    def get_fieldlinepoints_cyl(self, start_RZ, start_phi, n_transits=1, return_cartesian=True):
        """
        Integrate the field line over an angle phi from the starting point given by 
        cartesian coordinates (x,y,z).
        """
        start_xyz = self._rphiz_to_xyz(np.array([start_RZ[0], start_phi, start_RZ[1]]))
        return self.get_fieldlinepoints_cart(
            start_xyz, n_transits=n_transits, return_cartesian=return_cartesian)
    
class ScipyFieldlineIntegrator(Integrator):
    """
    Field line integration using scipy solve_ivp methods. 
    Slight slowdown compared to the simsopt integrator is expected, but
    allows for more flexibility in integration methods and tolerance
    specification. 
    """
    def __init__(self, field, comm=None, nfp=None, stellsym=False, R0=None, test_symmetries=True, integrator_type='dopri5', integrator_args=dict()):
        """
        Args:
            field: the magnetic field to be used for the integration
            nfp: number of field periods
            stellsym: if True, the field is assumed to be stellarator symmetric
            R0: major radius of the device. Used for testing symmetries of the field
            test_symmetries: (bool): if True, test if the values of nfp and stellsym
                are consistent with the field. If False, no test is performed.  
            integrator_type: type of integrator to use (default 'dopri5')
            integrator_args: additional arguments to pass to the integrator
        """
        super().__init__(field, comm, nfp, stellsym, R0, test_symmetries)
        self._integrator_type = integrator_type
        self._integrator_args = integrator_args
        # update integrator args with defaults
        if 'rtol' not in self._integrator_args:
            self._integrator_args['rtol'] = 1e-7
        if 'atol' not in self._integrator_args:
            self._integrator_args['atol'] = 1e-9

    
    def compute_poincare_hits(self, start_RZ, tmax=200, phis=[],):
        """
        calculate the res_tys and res_phi_hits
        """
        res_tys = []
        res_phi_hits = []
        first, last = parallel_loop_bounds(self.comm, len(start_RZ))
        for this_RZ in start_RZ[first:last, :]:
            res_ty, res_phi_hit = self.integrate_fieldlinepoints_RZ(
                this_RZ,
                tmax, phis=phis)
            res_tys.append(np.asarray(res_ty))
            res_phi_hits.append(np.asarray(res_phi_hit))
        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
            res_phi_hits = [i for o in self.comm.allgather(res_phi_hits) for i in o]
        return res_tys, res_phi_hits

    def integration_fn_cyl(self, t, rz):
        """
        Integrand for simple field line following, ugly written to speed operations
        dR/dphi = R*B_R/B_phi,
        dZ/dphi = R*B_Z/B_phi
        """
        if np.any(np.isnan(rz)):
            return np.ones_like(rz)*np.nan
        R, Z = rz
        rphiz = np.array([R, t, Z])
        B = self.field.set_points_cyl(rphiz[None, :])
        B = self.field.B_cyl().flatten()
        B = B.flatten()
        return np.array([R*B[0]/B[1], R*B[2]/B[1]])

    @property
    def event_function(self):
        """
        event function that stops integration when the field is gets close to a coil.
        This is determined by the B_phi component getting smaller than 1e-3 the full field stength.
        Not setting points cause event is only called
        after function eval, so super fast.

        Abs and subtraction because zero crossing is not
        passed by integrator.
        """
        def _event(t, rz):
            return np.abs(self.field.B_cyl().dot(np.array([0,1.,0.])/np.linalg.norm(self.field.B()))) - 1e-3  # B_phi = 0
        _event.terminal = True
        return _event

    def integrate_in_phi(self, RZ_start, phi_start, phi_end):
        """
        Integrate the field line using scipy's odeint method
        """
        sol = solve_ivp(self.integration_fn_cyl, [phi_start, phi_end], RZ_start, events=self.event_function, method='RK45', rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        if not sol.success:
            return np.array(np.nan, np.nan)
        return sol.y[:, -1]

    def integrate_cyl_planes(self, start_RZ, phis, return_cartesian=False):
        """
        Integrate the field line using scipy odeint method, performing integration 
        of the cylindrical field line ODE given by: 

        .. math::
            \frac{dR}{d\phi} = R \frac{B_R}{B_\phi}, \\
            \frac{dZ}{d\phi} = R \frac{B_Z}{B_\phi}.
        
        Args:
            start_RZ: starting point in cylindrical coordinates (R,Z)
            phis: list of angles in [0, 2pi] for which intersection with. 

        Default returns cylindrical coordinates, can also
        return cartesian coordinates.
        """
        
        sol = solve_ivp(self.integration_fn, [phis[0], phis[-1]], start_RZ, t_eval=phis, events=self.event_function, method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        if not sol.success:
            raise ObjectiveFailure("Failed to integrate field line")
        rphiz = np.array([sol.y[0, :], sol.t, sol.y[1, :]]).T
        if return_cartesian:
            return self._rphiz_to_xyz(rphiz)
        else:
            return rphiz
    
    def integrate_fieldlinepoints_RZ(self, start_RZ, phi_start, phi_end, n_points, endpoint=False, return_cartesian=True):
        """
        integrate a fieldline for a given toroidal distance, giving the start point in R,Z. 
        """
        if phi_end < phi_start:
            raise ValueError("phi_end must be greater than phi_start")
        phis = np.linspace(phi_start, phi_end, n_points, endpoint=endpoint)
        points = self.integrate_cyl_planes(start_RZ, phis, return_cartesian=return_cartesian)
        return points

    def integrate_fieldlinepoints_xyz(self, xyz_start, phi_total, n_points, endpoint=False, return_cartesian=True):
        """
        integrate a fieldline for a given toroidal distance, giving the start point in xyz. 
        """
        rphiz_start = self._xyz_to_rphiz(xyz_start)
        RZ_start = rphiz_start[::2]
        phi_start = rphiz_start[1]
        phi_end = phi_start + phi_total
        points = self.integrate_fieldlinepoints_RZ(RZ_start, phi_start, phi_end, n_points, endpoint=endpoint, return_cartesian=return_cartesian)
        return points

    def integration_fn_3d(self, t, xyz):
        self.field.set_points(xyz[None,:])
        B = self.field.B().flatten()
        return B/np.linalg.norm(B)

    def integrate_3d_fieldlinepoints_xyz(self, xyz_start, l_total, n_points):
        """
        integrate a fieldline in three dimensions for a given distance. 
        This solves the equations 
       
        .. math::
            \frac{\partial \mathbf{\gamma}(\tau)}{\partial \tau} = \mathbf{B}(\mathbf{\gamma}(\tau))/|\mathbf{B}(\mathbf{\gamma}(\tau))|
        where :math:`\tau` is the arc length along the field line, and :math:`\mathbf{\gamma}(\tau)` is the position vector of the field line at arc length :math:`\tau`.

        This method can integrate any field line, even if :math:`B_\phi < 0`, such as when field lines get caught by the coils. 

        Args:
            xyz_start: starting point in cartesian coordinates (x,y,z)
            l_total: total length of the field line to be integrated. A full transit is approximately 2*pi*R0, where R0 is the major radius of the device.
            n_points: number of points to return along the field line
        Returns:
            points: a numpy array of shape (n_points, 3) containing the points along the field line
            The points are in cartesian coordinates (x,y,z).
        """
        sol = solve_ivp(self.integration_function3d, [0, l_total], xyz_start, t_eval=np.linspace(0, l_total, n_points), method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        return sol.y.T
    
    #TODO: add jacobian of integrand


class PoincarePlotter(Optimizable):
    """
    Class to facilitate the calculation of field lines and 
    plotting the results in a Poincare plot. 
    Uses field periodicity to speed up calculation
    """
    def __init__(self, integrator: Integrator, start_points_R, start_points_Z,  phis=[], stopping_criteria=[]):
        """
        Initialize the PoincarePlotter.
        Args:
            start_points_R: list of radial components of initial points
            start_points_Z: list of vertical components of initial points
            integrator: the integrator to be used for the calculation
            phis: angles in [0, 2pi] for which we wish to compute Poincare.
                  *OR* int: number of planes to compute, equally spaced in [0, 2pi/nfp].
            stopping_criteria: list of stopping criteria to be used during the integration
        """
        self._start_points_R = start_points_R
        self._start_points_Z = start_points_Z
        self.integrator = integrator
        if isinstance(phis, int):
            self._phis = self.generate_phis(phis, nfp=self.integrator.nfp, stellsym=self.integrator.stellsym)
        elif len(phis) == 0:
            self._phis = np.array([0.0])
        else:
            self._phis = phis
        self.i_am_the_plotter = self.integrator.comm is None or self.integrator.comm.rank == 0  #  only rank 0 does plotting
        self.need_to_recompute = True
        Optimizable.__init__(depends_on=self.integrator)

    @staticmethod
    def generate_phis(nplanes, nfp=1, stellsym=True):
        if stellsym: 
            phis_halfperiod = np.linspace(0, np.pi/nfp, nplanes, endpoint=False)
            list_of_phis = [phis_halfperiod + per_idx*np.pi/nfp for per_idx in range(nfp*2)]
            return np.concatenate(list_of_phis)
        else:
            phis = np.linspace(0, 2*np.pi/nfp, nplanes, endpoint=False)
            list_of_phis = [phis + per_idx*2*np.pi/nfp for per_idx in range(nfp)]
            return np.concatenate(list_of_phis)

    @property
    def start_points_R(self):
        return self._start_points_R
    
    @start_points_R.setter
    def start_points_R(self, array):
        self._start_points_R = array
        self.recompute_bell()
    
    @property
    def start_points_Z(self):
        return self._start_points_Z
    
    @start_points_Z.setter
    def start_points_Z(self, array):
        self._start_points_Z = array
        self.recompute_bell()

    @property
    def phis(self):
        return self._phis
    
    @phis.setter
    def phis(self, value):
        self._phis = value
        self.recompute_bell()

    @classmethod
    def from_field(cls, field, start_points_R, start_points_Z, phis=[], stopping_criteria=[], comm=None, integrator_type='simsopt', tol=1e-9, tmax=200):
        """
        Create a PoincarePlotter object from a field, skipping the need to create a separate integrator.  
        """
        if integrator_type == 'simsopt':
            integrator = SimsoptFieldlineIntegrator(field, comm=comm, stopping_criteria=stopping_criteria)
        else:
            raise ValueError(f"Integrator type {integrator_type} not supported.")
        return cls(start_points_R, start_points_Z, integrator, phis=phis, stopping_criteria=stopping_criteria)


    def recompute_bell(self):
        self.res_tys = None
        self.res_phi_hits = None
        self.lost=None
        self.need_to_recompute = True
    
    @property
    def res_tys(self):
        """
        the solution of the integration with intermediate points, in 
        the form of a list (over trajectories) of lists (over timesteps)
        of  """
        if self.need_to_recompute:
            self.res_tys, self.res_phi_hits = self.integrator.compute_poincare_hits(
                self.start_points_R, self.start_points_Z)
            self.need_to_recompute = False
        return self.res_tys
    
    @property
    def res_phi_hits(self):
        if self.need_to_recompute:
            self.res_tys, self.res_phi_hits = self.integrator.compute_poincare_hits(
                self.start_points_R, self.start_points_Z)
            self.need_to_recompute = False
        return self.res_phi_hits
    
    def plane_hits_cyl(self, plane_idx):
        """
        Get the points where the field lines intersect the plane defined by the index.
        Returns a list of arrays containing the R,Z coordinates of the intersection 
        points.  
        """
        hits_cart = self.plane_hits_cart(plane_idx)
        hits = [self.integrator._xyz_to_rphiz(hc)[::2] for hc in hits_cart]
        return hits
    
    @property
    def lost(self): 
        """
        Get the points where the integration stopped due to a stopping criterion.
        """
        # list comprehension... look at final element, if element 1 negative, then lost.
        if self.lost is None:
            self.lost = [traj[traj_idx][-1, 1] < 0 for traj_idx, traj in enumerate(self.res_phi_hits)]
        return self.lost

    
    def plane_hits_cart(self, plane_idx):
        """
        Get the points where the field lines hit the plane defined by the index. 
        """
        if plane_idx >= len(self.phis):
            raise ValueError(f"Plane index {plane_idx} is larger than the number of planes {len(self.phis)}.")
        
        hits = []
        for traj in self.res_phi_hits:
            hits_xyz = traj[np.where(traj[:, 1] == plane_idx)[0], :] # res_phi_hits second column is the index of the plane or the stopping criterion
            hits.append(hits_xyz) # append the xyz points
        return hits
    
    @staticmethod
    def fix_axes(ax, xlabel='R', ylabel='Z', title=None):
        """
        Label the axes of the plot. 
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)

    def plot_poincare_plane_idx(self, plane_idx, **kwargs):
        """
        plot a single cross-section of the field at a given plane index. 
        """
        if plane_idx >= len(self.phis):
            raise ValueError(f"Plane index {plane_idx} is larger than the number of planes {len(self.phis)}.")
        return self.plot_poincare_single(self.phis[plane_idx], prevent_recompute=True, **kwargs)

    def plot_poincare_single(self, phi, prevent_recompute=False, ax=None, mark_lost=False, fix_axes=True, surf=None, flipped_plane=False, **kwargs):
        """
        plot a single cross-section of the field at a given phi value. 
        If this value is not in the list of phis, a recompute will be triggered. 
        *NOTE*: if running parallel, call this function on all ranks. 
        """
        if phi not in self.phis:
            if not prevent_recompute:
                self.phis = [phi,]
            else:
                raise ValueError(f"The requested plane at phi={phi} has not been computed.")
        else:
            phi_index = self.phis.index(phi)

        
        # can trigger recompute, so all ranks execute
        hits_thisplane = self.plane_hits_cyl(phi_index)

        # rank0 only
        if self.i_am_the_plotter: 
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            
            marker = kwargs.pop('marker', '.')
            s = kwargs.pop('s', 1)
            color = kwargs.pop('color', 'random')

            for idx, trajpoints in enumerate(hits_thisplane):
                if color == 'random':
                    color = np.random.rand(3,)

                if mark_lost:
                    lost = self.lost[idx]
                    if lost:
                        color = 'r'
                        thismarker = 'x'
                        this_s = s*3
                else:
                    thismarker = marker
                    this_s = s
                if flipped_plane:
                    trajpoints[:, 1] = -trajpoints[:, 1]
                ax.scatter(trajpoints[:, 0], trajpoints[:, 1], marker=thismarker, s=this_s, c=color, **kwargs)
            
            if surf is not None:
                cross_section = surf.cross_section(phi=phi)
                r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
                z_interp = cross_section[:, 2]
                ax.plot(r_interp, z_interp, linewidth=1, c='k')
            
            if fix_axes:
                self.fix_axes(ax)
            
            return fig, ax
        else: 
            return None, None  # other ranks do not plot anything


    def plot_poincare_all(self, mark_lost=False, fix_ax=True, **kwargs):
        """
        plot all the planes in a single figure. 
        *NOTE*: if running parallel, call this function on all ranks. 
        """

        if self.i_am_the_plotter:
            from math import ceil
            nrowcol = ceil(sqrt(len(self.phis)))
            plt.figure()
            fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))

            plot_period = np.pi/self.integrator.nfp if self.integrator.stellsym else 2*np.pi/self.integrator.nfp

            for section_idx, phi in enumerate(self.phis[np.where(self.phis<plot_period)]):  #ony the plane in the first field period
                ax=axs[section_idx]
                self.plot_poincare_single(phi, ax=ax, mark_lost=mark_lost, prevent_recompute=True, fix_axes=fix_ax, **kwargs)
                # find other planes in other field periods:
                for add_idx in np.where(np.isclose((self.phis - phi) % (2*np.pi/self.integrator.nfp), 0)):
                    if add_idx == section_idx: 
                        continue
                    self.plot_poincare_single(self.phis[add_idx], ax=ax, mark_lost=mark_lost, prevent_recompute=True, fix_axes=fix_ax, **kwargs)
                if self.integrator.stellsym:
                    for add_idx in np.where(np.isclose((self.phis + phi) % (np.pi/self.integrator.nfp), 0)):
                        if add_idx == section_idx: 
                            continue
                        self.plot_poincare_single(self.phis[add_idx], ax=ax, mark_lost=mark_lost, prevent_recompute=True, flipped_plane = True, fix_axes=fix_ax, **kwargs)
                textstr = f" φ = {phi/np.pi:.2f}π "
                props = dict(boxstyle='round', facecolor='white', edgecolor='black')
                ax.text(0.05, 0.02, textstr, transform=ax.transAxes, fontsize=6,
                verticalalignment='bottom', bbox=props)

            plt.tight_layout()

            return fig, axs
        else: 
            return None, None  # other ranks do not plot anything
