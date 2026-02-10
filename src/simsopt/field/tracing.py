import logging
from math import sqrt

import numpy as np
from pathlib import Path
import os

import simsoptpp as sopp
from .._core import Optimizable, ObjectiveFailure
from .._core.util import parallel_loop_bounds
from ..field.magneticfield import MagneticField
from ..field.boozermagneticfield import BoozerMagneticField
from ..field.sampling import draw_uniform_on_curve, draw_uniform_on_surface
from ..geo.surface import SurfaceClassifier
from ..util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from .._core.types import RealArray
from numpy.typing import NDArray
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
           'particles_to_vtk', 'plot_poincare_data', 'Integrator',
           'ScipyFieldlineIntegrator', 'SimsoptFieldlineIntegrator', 'PoincarePlotter']


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


def compute_fieldlines(field, R0, Z0, phi0 = 0, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None):
    r"""
    Compute magnetic field lines by solving

    .. math::

        [\dot x, \dot y, \dot z] = B(x, y, z)

    Integration is initialized on the :math:`\phi = 0` plane.

    Args:
        field: the magnetic field :math:`B`
        R0: list(float) of radial components of initial points
        Z0: list(float) of vertical components of initial points
        phi0: (float) toroidal angle of initial points
               or list(float) of toroidal angles of initial points. If a single float is given, it is used for all initial points.
        tmax: (float) longest allowable integration 'time'. will do roughly ``|B|*tmax/(2*pi*r0)`` revolutions of the device. 
               most integrations will stop due to other conditions, such as reaching a desired toroidal angle or number of transits. 
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
    assert isinstance(phi0, (float, int)) or len(phi0) == len(R0)
    nlines = len(R0)
    xyz_inits = np.zeros((nlines, 3))
    xyz_inits[:, 0] = np.cos(phi0) * np.asarray(R0)
    xyz_inits[:, 1] = np.sin(phi0) * np.asarray(R0)
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
    A Base class for integrators as Optimizable objects. Outputs of integrators can be optimized. 
    Instances of the Integrator class implement a compute_poincare_hits method which interface with the PoincarePlotter class.
    Do not use this base class, use the SimsoptFieldlineIntegrator or ScipyFieldlineIntegrator depending on need.
    """
    def __init__(self, field: MagneticField, comm=None, nfp=None, stellsym=False):
        """
        Args:
            field: the magnetic field to be used for the integration
            nfp: number of field periods
            stellsym: if True, the field is assumed to be stellarator symmetric
        """
        self.field = field
        self.comm = comm
        if nfp is None:
            self.nfp = 1
        else:
            self.nfp = nfp
        self.stellsym = stellsym
        Optimizable.__init__(self, depends_on=[field,])

    @staticmethod
    def _rphiz_to_xyz(array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert cylindrical coordinates to Cartesian coordinates.
        Args: 
            array: an nx3 array of cylindrical coordinates
        Returns: 
            nx3 array of Cartesian coordinates
        """
        array = np.atleast_2d(array)
        if np.shape(array)[1] != 3 or array.ndim > 2:
            raise ValueError("Input array must be of shape (3,) or (n,3)")
        return np.array([array[:, 0]*np.cos(array[:, 1]), array[:, 0]*np.sin(array[:, 1]), array[:, 2]]).T

    @staticmethod
    def _xyz_to_rphiz(array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert Cartesian coordinates to cylindrical. 
        Args: 
            array: an nx3 array of Cartesian coordinates
        Returns:
            nx3 array of cylindrical coordinates
        """
        array = np.atleast_2d(array)
        if np.shape(array)[1] != 3 or array.ndim > 2:
            raise ValueError("Input array must be of shape (3,) or (n,3)")
        
        return np.array([np.sqrt(array[:, 0]**2 + array[:, 1]**2), np.arctan2(array[:, 1], array[:, 0]), array[:, 2]]).T


class SimsoptFieldlineIntegrator(Integrator):
    """
    Integration of field lines using the `simsoptpp` routines. 
    Integration is performed in three dimensions, solving the 
    ODE:
    .. math::
        \frac{d\mathbf{x}(t)}{dt} = \mathbf{B}(\mathbf{x})
    where :math:`\mathbf{x}=(x,y,z)` are the cartesian coordinates.
    MPI parallelization is supported by passing an MPI communicator. 

    Args:
        field: the magnetic field to be used for the integration (BoozerMagneticfield is not supported yet)
        comm: MPI communicator to parallelize over
        nfp: number of field periods
        stellsym: if True, the field is assumed to be stellarator symmetric
        stopping_criteria: list of StoppingCriterion objects, only used when generating data for Poincare plots.
        tol: tolerance for the adaptive ODE solver
        tmax: maximum time to integrate each field line
    """

    def __init__(self, field: MagneticField, comm=None, nfp=None, stellsym=False, stopping_criteria=[], tol=1e-9, tmax=1e4):
        self.tol = tol
        self.stopping_criteria = stopping_criteria
        self.tmax = tmax
        super().__init__(field, comm=comm, nfp=nfp, stellsym=stellsym)

    def compute_poincare_hits(self, start_points_RZ, n_transits, phis=[], phi0=0):
        """
        calculate the fieldline trajecotories using the simsoptpp routines.
        Args:
            start_points_RZ: nx2 array of starting radial coordinates.
            n_transits: number of times to go around the device.
            phis: list of angles in [0, 2pi] for which intersection with the plane
                  corresponding to that phi should be computed. If empty, only phi=0 is used.
        Returns:
            res_phi_hits, res_tys: 2 element tuple containing
            - res_tys: list of numpy arrays (one for each field line) describing the
                     solution over time. The numpy array is of shape (ntimesteps, 4).
                     Each row contains the time and the position, i.e.`[t, x, y, z]`.
            - res_phi_hits: list of numpy arrays (one for each particle) containing
                     information on each time the particle hits one of the phi planes.
                     Each row of the array contains `[time, idx, x, y, z]`, where `idx`
                     tells us which of the `phis` was hit. If `idx>=0`, then
                     `phis[int(idx)]` was hit. If `idx<0`, then one of the stopping criteria
                     was hit, which is specified in the `stopping_criteria` argument.
                  """
        stopping_criteria = self.stopping_criteria + [ToroidalTransitStoppingCriterion(n_transits, False),]
        return compute_fieldlines(
            self.field, start_points_RZ[:, 0], start_points_RZ[:, 1], phi0=phi0, tmax=self.tmax, tol=self.tol,
            phis=phis, stopping_criteria=stopping_criteria, comm=self.comm)

    def integrate_toroidally(self, start_point, delta_phi=2*np.pi, phi0=None, input_coordinates='cartesian', output_coordinates='cartesian'):
        """
        Integrate the field line in phi from xyz location start_xyz over an angle delta_phi using simsoptpp routines.  

        Args:
            start_xyz: starting coordinates: if input_coordinates is 'cartesian', this should be Array[x,y,z], if 'cylindrical', Array[R,Z], and phi0 should also be provided.
            delta_phi: angle to integrate over
            return_cartesian: if True, return the cartesian coordinates of the end point, otherwise return only R,Z (phi_end assumed)
        Returns:
            end_point: Array[x,y,z] if return_cartesian is True, otherwise Array[R ,Z]
        """
        if input_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("input_coordinates must be either 'cartesian' or 'cylindrical'")
        if output_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("output_coordinates must be either 'cartesian' or 'cylindrical'")
        
        if input_coordinates == 'cylindrical':
            if phi0 is None:
                raise ValueError("If input_coordinates is 'cylindrical', phi0 must be provided")
            start_xyz = self._rphiz_to_xyz(np.array([start_point[0], phi0, start_point[1]]))[-1, :]  
            start_phi = phi0
        else:
            start_phi = self._xyz_to_rphiz(start_point)[0, 1]
            start_xyz = start_point

        phi_end = start_phi + delta_phi
        n_transits = int(np.abs(delta_phi)/(2.0*np.pi)) + 2
        self.field.set_points(start_xyz[None, :])
        # test direction of the field and switch if necessary, normalize to field strength at start
        Bstart = self.field.B_cyl()[0]
        if Bstart[1] < 0:  # direction of bphi
            field_for_tracing = (-1.0 / np.linalg.norm(Bstart)) * self.field
        else:
            field_for_tracing = (1.0 / np.linalg.norm(Bstart)) * self.field

        _, res_phi_hits = sopp.fieldline_tracing(
            field_for_tracing, 
            start_xyz, 
            tmax=self.tmax, 
            tol=self.tol, 
            phis=[phi_end,],
            stopping_criteria=[ToroidalTransitStoppingCriterion(n_transits, False),]
            )
        xyz = np.array(res_phi_hits[-2][2:])  # second to last hit is the phi_end hit
        if output_coordinates=='cartesian':  # return [x,y,z] array
            return xyz
        else:  # return [R,Z] array
            return self._xyz_to_rphiz(xyz)[0,::2]

    def integrate_fieldlinepoints(self, start_point, phi0=None, n_transits=1, input_coordinates='cartesian', output_coordinates='cartesian'):
        """
        Calculate a string of points along a field line starting at a location given by cartesian coordinates.
        The points are returned in the form of a numpy array of shape (npoints, 3). 
        If return_cartesian is True, the points are in cartesian coordinates (x,y,z), otherwise they are in (R, phi, Z). 

        The SimsoptFieldLineIntegrator does not give control over the number of points, but relies
        on the adaptive step size of the integrator. The number of points returned will depend on the field
        complexity and the tolerance set for the integrator.

        Args:
            start_xyz: starting cartesian coordinates Array[x,y,z]
            n_transits: number of times to go around the device.
            return_cartesian: if True, return the points in cartesian coordinates, otherwise return Cylindrical coordinates (default False). 
        Returns:
            points: numpy array of shape (npoints, 3) containing the field line points, either in cartesian or cylindrical coordinates depending on return_cartesian.
        """
        if input_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("input_coordinates must be either 'cartesian' or 'cylindrical'")
        if output_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("output_coordinates must be either 'cartesian' or 'cylindrical'")
        if input_coordinates == 'cylindrical':
            if phi0 is None:
                raise ValueError("If input_coordinates is 'cylindrical', phi0 must be provided")
            start_xyz = self._rphiz_to_xyz(np.array([start_point[0], phi0, start_point[1]]))[-1, :]
        else: 
            start_xyz = start_point

        self.field.set_points(np.atleast_2d(start_xyz))
        # test direction of the field and switch if necessary, normalize to field strength at start
        Bstart = self.field.B_cyl()[0]
        if Bstart[1] < 0:  # direction of bphi
            field_for_tracing = (-1.0 / np.linalg.norm(Bstart)) * self.field
        else:
            field_for_tracing = (1.0 / np.linalg.norm(Bstart)) * self.field

        res_tys, _ = sopp.fieldline_tracing(
            field_for_tracing, 
            start_xyz, 
            tmax=self.tmax, 
            tol=self.tol, 
            phis=[0,], 
            stopping_criteria=[ToroidalTransitStoppingCriterion(n_transits, False),])  # stop after n laps, this is a short fieldline segment, ignore other stopping criteria
        points_cart = np.array(res_tys)[:, 1:]
        if output_coordinates == 'cartesian':
            return points_cart
        else:
            return self._xyz_to_rphiz(points_cart)


class ScipyFieldlineIntegrator(Integrator):
    """
    Field line integration using scipy solve_ivp methods. 
    Slight slowdown compared to the simsopt integrator is expected, but
    allows for more flexibility in integration methods and tolerance
    specification. 

    For integration over phi, the ode solved is: 
    .. math::
        \frac{dR}{d\phi} = R \frac{B_R}{B_\phi}, \\
        \frac{dZ}{d\phi} = R \frac{B_Z}{B_\phi}
    where :math:`(R,\phi,Z)` are the cylindrical coordinates and
    :math:`(B_R, B_\phi, B_Z)` are the cylindrical components of the magnetic field.
    The integration can only be performed in one direction, and integration will fail for field lines that reach a point where B_phi=0.

    Three dimensional integration (such as in the SimsoptFieldlineIntegrator)
    can also be performed with the ``integrate_3d_fieldlinepoints`` method. 
    """
    def __init__(self, field, comm=None, nfp=None, stellsym=False, integrator_type='RK45', integrator_args=dict()):
        """
        Args:
            field: the magnetic field to be used for the integration
            comm: MPI communicator to parallelize over
            nfp: number of field periods
            stellsym: if True, the field is assumed to be stellarator symmetric
            integrator_type: type of integrator to use (default 'dopri5')
            integrator_args: additional arguments to pass to the integrator given as a dictionary {parameter: value} (for example, {'rtol': 1e-6, 'atol':1e-9})
        """
        super().__init__(field, comm, nfp, stellsym)
        self._integrator_type = integrator_type
        self._integrator_args = integrator_args
        # update integrator args with defaults
        if 'rtol' not in self._integrator_args:
            self._integrator_args['rtol'] = 1e-7
        if 'atol' not in self._integrator_args:
            self._integrator_args['atol'] = 1e-9

    def compute_poincare_hits(self, start_points_RZ, n_transits, phis=[], phi0=0):
        """
        calculate the poincare section hits. The start points for integration are given 
        by start_points_RZ, and started from the plane phi0. 
        Args:
            start_points_RZ: nx2 array of starting radial coordinates
            n_transits: number of times to go around the device.
            phis: list of angles in [0, 2pi] for which intersection with the plane
                  corresponding to that phi should be computed. 
            phi0: initial angle in the phi plane (default 0). if this plane is not in phis, it will be added. 
        Returns:
            res_phi_hits: list of numpy arrays (one for each start point) containing
                        information on each time the particle hits one of the phi planes. Each row of the array contains
                        `[phi, idx, x, y, z]`, where `idx` tells us which of the `phis` was hit. If `idx>=0`, then `phis[int(idx)]` was hit.
        """
        res_phi_hits = []
        first, last = parallel_loop_bounds(self.comm, len(start_points_RZ))
        for this_start_RZ in start_points_RZ[first:last, :]:
            logger.info(f'Integrating poincare section for field line starting at R={this_start_RZ[0]}, Z={this_start_RZ[1]}')
            all_phis = np.array([phis + 2*np.pi*nt for nt in range(int(n_transits))]).flatten()
            if all_phis[0] != phi0:
                all_phis = np.insert(all_phis, 0, phi0)
            status, rphiz = self.integrate_cyl_planes(this_start_RZ, all_phis)
            xyz_values = self._rphiz_to_xyz(rphiz)
            plane_idx = np.array(range(len(all_phis))) % len(phis)
            num_values = rphiz.shape[0]
            res_phi_hits_line = np.column_stack((rphiz[:, 1],
                                         plane_idx[:num_values], 
                                         xyz_values[:, :]))
            if status == -1 or status == 1:  # integration did not finish
                res_phi_hits_line[-1, 1] = -1
            res_phi_hits.append(res_phi_hits_line)
        if self.comm is not None:
            res_phi_hits = [hit for gathered_hits in self.comm.allgather(res_phi_hits) for hit in gathered_hits]
        return res_phi_hits
    
    def compute_poincare_trajectories(self, start_points_RZ, n_transits, phi0=0):
        """
        calculate the trajectories of the field lines starting at start_points_RZ in the phi0 plane. 
        Args:
            start_points_RZ: nx2 array of starting points in (R,Z) cylindrical coordinates.
            n_transits: number of times to go around the device.
            phi0: initial angle in the phi plane (default 0).
        Returns:
            res_tys: list of numpy arrays (one for each particle) containing
                     information on the trajectory of the field line. Each row of the array contains
                     `[phi, x, y, z]`.
        """
        res_tys = []
        first, last = parallel_loop_bounds(self.comm, len(start_points_RZ))
        for this_start_RZ in start_points_RZ[first:last, :]:
            logger.info(f'Integrating trajectory of field line starting at R={this_start_RZ[0]}, Z={this_start_RZ[1]}')
            start_phi = phi0
            phi_end = start_phi + n_transits*2*np.pi
            phi_eval = np.linspace(start_phi, phi_end, int(1000*int(n_transits)))
            status, rphiz = self.integrate_cyl_planes(this_start_RZ, phi_eval)
            xyz_values = self._rphiz_to_xyz(rphiz)
            res_tys_line = np.column_stack((rphiz[:, 1], 
                                            xyz_values))
            res_tys.append(res_tys_line)
        if self.comm is not None:
            res_tys = [ty for gathered_tys in self.comm.allgather(res_tys) for ty in gathered_tys]
        return res_tys

    def integration_fn_cyl(self, t, rz):
        """
        Integrand for field line tracing in cylindrical coordinates. 
        Returns the right hand side of the ODEs:
        .. math::
            dR/d\phi = R*B_R/B_\phi,
            dZ/d\phi = R*B_Z/B_\phi
        
        Args:
            t: toroidal angle phi
            rz: array of R,Z coordinates
        Returns:
            R*B_R/B_phi, R*B_Z/B_phi
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
    def _event_function(self):
        """
        event function used to terminate integration in solve_ivp. 
        the method tests whether B_phi component is smaller than 1e-3 the full field stength.
        this occurs (hopefully) just before B_phi changes sign, which can be around coils. 
        This function does not need an extra BiotSavart calculation, it reads it from cache. 
        """
        def _event(t, rz):
            return np.abs(self.field.B_cyl().dot(np.array([0, 1., 0.]) / np.linalg.norm(self.field.B()))) - 1e-3  # B_phi = 0
        #_event.terminal = True
        return _event

    def integrate_toroidally(self, start_point, phi0=None, delta_phi=2*np.pi, input_coordinates='cartesian', output_coordinates='cartesian'):
        """
        Integrate along the field in the phi direction, returning the end location. 
        Integration starts from cylindrical coordinates (R,Z) at angle start_phi. 
        The cylindrical field line ODE is given by: 
        .. math::
            \frac{dR}{d\phi} = R \frac{B_R}{B_\phi}, \\
            \frac{dZ}{d\phi} = R \frac{B_Z}{B_\phi}.

        Args:
            start_RZ: starting point in cylindrical coordinates (R,Z)
            start_phi: starting angle in phi
            delta_phi: angle to integrate over
            return_cartesian: if True, return the cartesian coordinates of the end point, otherwise return only R,Z (phi_end assumed)
        Returns:
            end_xyz or end_RZ depending on return_cartesian
        """
        if input_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("input_coordinates must be either 'cartesian' or 'cylindrical'")
        if output_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("output_coordinates must be either 'cartesian' or 'cylindrical'")

        if input_coordinates == 'cartesian':
            rphiz_start = self._xyz_to_rphiz(start_point)[-1]
            start_RZ = rphiz_start[[0, 2]]
            phi0 = rphiz_start[1]
        else: 
            if phi0 is None:
                raise ValueError("If input_coordinates is 'cylindrical', phi0 must be provided")
            start_RZ = start_point
        sol = solve_ivp(self.integration_fn_cyl, [phi0, phi0 + delta_phi], start_RZ, events=self._event_function, method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        if not sol.success:
            return np.array([np.nan, np.nan])
        if output_coordinates == 'cartesian':
            rphiz = np.array([sol.y[0, -1], phi0 + delta_phi, sol.y[1, -1]])
            return self._rphiz_to_xyz(rphiz)[-1, :]
        else:
            return sol.y[:, -1] # final R,Z
    
    # def d_integration_by_dcoeff(self, start_RZ, start_phi, phi_end):
    # def d_
    
    def integrate_cyl_planes(self, start_RZ, phis, output_coordinates="cylindrical"):
        """
        Integrate the field line using scipy odeint method, performing integration 
        of the cylindrical field line ODE given by: 

        .. math::
            \frac{dR}{d\phi} = R \frac{B_R}{B_\phi}, \\
            \frac{dZ}{d\phi} = R \frac{B_Z}{B_\phi}.
        
        Args:
            start_RZ: starting point in cylindrical coordinates (R,Z)
            phis: list of angles on which the solution should be evaluated. This should be in increasing order, and can span multiple toroidal turns.
            output_coordinates: if 'cartesian', return the solution in cartesian coordinates, if 'cylindrical', return the solution in cylindrical coordinates (R,Z) with phi given by the phis input.

        returns: 
            status, rphiz: tuple cointaining
            - status: integration status (0 if successful, -1 if failed, or 1 if stopped by event)
            - rphiz: the (R,Z,phi) coordinates of the field line 
            OR:
            - xyz: the (x,y,z) coordinates of the field line if return_cartesian is True
        """
        if output_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("output_coordinates must be either 'cartesian' or 'cylindrical'")
        sol = solve_ivp(self.integration_fn_cyl, [phis[0], phis[-1]], start_RZ, t_eval=phis, events=self._event_function, method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        status = sol.status
        rphiz = np.array([sol.y[0, :], sol.t, sol.y[1, :]]).T
        
        if output_coordinates == 'cartesian':
            return status, self._rphiz_to_xyz(rphiz)
        else:
            return status, rphiz

    def integrate_fieldlinepoints(self, start_point, delta_phi, n_points, phi0=None, endpoint=False, input_coordinates='cartesian', output_coordinates='cartesian'):
        """
        Calculate n_points along a field line starting at an R,Z location and starting angle in phi, for a 
        distance delta_phi. 
        Args:
            start_RZ: starting point in cylindrical coordinates (R,Z)
            start_phi: starting angle in phi
            delta_phi: angle to integrate over
            n_points: number of points to return along the field line
            endpoint: if True, include the end point in the returned points
            input_coordinates: if 'cartesian', the start point is given in cartesian coordinates, if 'cylindrical', the start point is given in cylindrical coordinates (R,Z), and phi0 should also be provided.
            output_coordinates: if 'cartesian', return the points in cartesian coordinates, otherwise return Cylindrical coordinates (default False).
        Returns:
            points: a numpy array of shape (n_points, 3) containing the points along the field line
        """
        if input_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("input_coordinates must be either 'cartesian' or 'cylindrical'")
        if output_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("output_coordinates must be either 'cartesian' or 'cylindrical'")
       
        if input_coordinates == 'cylindrical':
            if phi0 is None:
                raise ValueError("If input_coordinates is 'cylindrical', phi0 must be provided")
            if len(start_point) != 2:
                raise ValueError("If input_coordinates is 'cylindrical', start_point should be of the form [R,Z]")
            start_RZ = start_point
        
        if input_coordinates == 'cartesian':
            rphiz_start = self._xyz_to_rphiz(start_point)[-1]
            start_RZ = rphiz_start[[0, 2]]
            phi0 = rphiz_start[1]
        
        phis = np.linspace(phi0, phi0+delta_phi, n_points, endpoint=endpoint)
        status, points = self.integrate_cyl_planes(start_RZ, phis, output_coordinates=output_coordinates)
        if status != 0:
            raise ObjectiveFailure("Integration failed")
        return points

    def _integration_fn_3d(self, t, xyz):
        """
        Integrand for 3D field line following. 
        This function has the correct signature to pass to `scipy.solve_ivp`.
        Args:
            t: independent variable (not used)
            xyz: array of cartesian coordinates 
        """
        self.field.set_points(xyz[None, :])
        B = self.field.B().flatten()
        return B/np.linalg.norm(B)

    def integrate_3d_fieldlinepoints(self, start_xyz, l_total, n_points, phi0=None, input_coordinates='cartesian', output_coordinates='cartesian'):
        """
        integrate a fieldline in three dimensions for a given distance. 
        This solves the equations 
       
        .. math::
            \frac{\partial \mathbf{\gamma}(\tau)}{\partial \tau} = \mathbf{B}(\mathbf{\gamma}(\tau))/|\mathbf{B}(\mathbf{\gamma}(\tau))|
        where :math:`\tau` is the arc length along the field line, and :math:`\mathbf{\gamma}(\tau)` is the position vector of the field line at arc length :math:`\tau`.

        This method can integrate any field line, even if :math:`B_\phi < 0`, such as when field lines get caught by the coils. 

        Args:
            start_xyz: starting point in cartesian coordinates (x,y,z)
            l_total: total length of the field line to be integrated. A full transit is approximately 2*pi*R0, where R0 is the major radius of the device.
            n_points: number of points to return along the field line
        Returns:
            points: a numpy array of shape (n_points, 3) containing the points along the field line
            The points are in cartesian coordinates (x,y,z).
        """
        if input_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("input_coordinates must be either 'cartesian' or 'cylindrical'")
        if output_coordinates not in ['cartesian', 'cylindrical']:
            raise ValueError("output_coordinates must be either 'cartesian' or 'cylindrical'")
        if input_coordinates == 'cylindrical':
            if phi0 is None:
                raise ValueError("If input_coordinates is 'cylindrical', phi0 must be provided")
            start_xyz = self._rphiz_to_xyz(np.array([start_xyz[0], phi0, start_xyz[1]]))[-1, :]
            
        sol = solve_ivp(self._integration_fn_3d, [0, l_total], start_xyz, t_eval=np.linspace(0, l_total, n_points), method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        if output_coordinates == 'cartesian':
            return sol.y.T
        else:
            return self._xyz_to_rphiz(sol.y.T)
    
    #TODO: add jacobian of integrand

class PoincarePlotter(Optimizable):
    """
    Class to facilitate the calculation of field lines and 
    plotting the results in a Poincare plot. 
    Uses field periodicity to speed up calculation
    """
    def __init__(self, integrator: Integrator, start_points_RZ, phis=None, n_transits=100, add_symmetry_planes=True, store_results=False, phi0=None):
        """
        Initialize the PoincarePlotter. 
        This class uses an Integrator to compute field lines, and takes care of plotting them. 
        If the field is stellarator-symmetric, and symmetry planes are included, then 
        information from identical planes are all plotted together (resulting in better plots with shorter integration). 

        Args:
            integrator: the integrator to be used for the calculation
            start_points_RZ: nx2 array of starting points in cylindrical coordinates (R,Z)
        Kwargs:
            phis (None): angles in [0, 2pi] for which we wish to compute Poincare.
                  *OR* int: number of planes to compute, equally spaced in [0, 2pi/nfp].
            n_transits (100): number of toroidal transits to compute
            add_symmetry_planes (True): if true, we add planes that are identical through field periodicity, increasing the efficiency of the calculation. 
            store_results (False): if true, use a cache on disk to store/retrieve results. 
                The results are stored in a file named 'poincare_data.npz' in the current directory. 
                The results are given a hash based on the MagneticField degrees of freedom and poincare attributes, triggering recomputation only if relevant parameters change.
            phi0 (None): initial angle in the phi plane (default None). if None, the first phi in phis is used.
        """
        self._start_points_RZ = start_points_RZ
        self.integrator = integrator
        self.n_transits = n_transits
        if isinstance(phis, int):
            self._phis = self.generate_phis(phis, nfp=self.integrator.nfp)
        elif phis is None:
            self._phis = np.array([0.0,])
        else:
            self._phis = np.atleast_1d(phis)
        
        if add_symmetry_planes:
            self._phis = self.generate_symmetry_planes(self._phis, nfp=self.integrator.nfp)

        self.phi0 = self._phis[0] if phi0 is None else phi0
        self.i_am_the_plotter = self.integrator.comm is None or self.integrator.comm.rank == 0  # only rank 0 does plotting
        self.need_to_recompute = True
        self._randomcolors = None
        Optimizable.__init__(self, depends_on=[integrator,])
        self.store_results = store_results
        if store_results:
            # load file form disk if it exists
            self.retrieve_poincare_data()


    @property
    def randomcolors(self):
        """
        Generate a list of random colors for plotting.
        """
        #check if already generated:
        if self._randomcolors is not None:
            return self._randomcolors
        else:
            np.random.seed(0)  # for reproducibility
            self._randomcolors = np.random.rand(len(self.start_points_RZ), 3)
            return self._randomcolors

    @staticmethod
    def generate_phis(nplanes, nfp=1):
        """
        Generate nplanes equally spaced phis in [0, 2pi/nfp].
        Args:
            nplanes: number of planes to generate
            nfp: number of field periods (default: 1)
        Returns:
            phis: list of phis in [0, 2pi/nfp]
        """
        return np.linspace(0, 2*np.pi/nfp, nplanes, endpoint=False)
    
    @staticmethod
    def generate_symmetry_planes(phis, nfp=1):
        """
        Given a list of phis in [0, 2pi/nfp], generate the full list of phis
        in [0, 2pi] by adding the symmetry planes. 
        Args: 
            phis: list of phis in [0, 2pi/nfp]
            nfp: number of field periods (default: 1)
        Returns:
            list_of_phis: list of phis in [0, 2pi] including the symmetry planes
        """
        list_of_phis = [phis + per_idx*2*np.pi/nfp for per_idx in range(nfp)]
        # remove duplicates and sort
        list_of_phis = np.unique(np.concatenate(list_of_phis))
        return list_of_phis
    
    @property
    def phis_for_plotting(self):
        """
        the phis in the first period, useful for plotting
        """
        plot_period = 2*np.pi/self.integrator.nfp
        phis_for_plotting = self.phis[np.where(self.phis < plot_period)]
        return phis_for_plotting

    @property
    def start_points_RZ(self):
        """
        start ponts in R,Z for the field line integration
        """
        return self._start_points_RZ

    @start_points_RZ.setter
    def start_points_RZ(self, array):
        if array.shape != self._start_points_RZ.shape:
            self._randomcolors = None  # reset colors 
        self._start_points_RZ = array
        self.recompute_bell()

    @property
    def phis(self):
        """
        all the phi planes used for the calculation
        """
        return self._phis
    
    @phis.setter
    def phis(self, value):
        self._phis = value
        self.recompute_bell()


    @classmethod
    def from_field(cls, field, start_points_RZ, phis=None, n_transits=1, add_symmetry_planes=True,
                   stopping_criteria=None, comm=None, integrator_type='simsopt', **kwargs):
        """
        Helper to create a PoincarePlotter directly from a MagneticField bypassing 
        the manual creation of an Integrator. 

        Parameters mirror PoincarePlotter.__init__, while constructing the appropriate integrator.
        Args:
            field: the magnetic field to be used for the integration
            start_points_RZ: nx2 array of starting points in cylindrical coordinates (R,Z)
            phis: angles in [0, 2pi] for which we wish to compute Poincare.
                  *OR* int: number of planes to compute, equally spaced in [0, 2pi/nfp].
            n_transits: number of toroidal transits to compute
            add_symmetry_planes: if true, we add planes that are identical through field periodicity, increasing the efficiency of the calculation.
            stopping_criteria: list of StoppingCriterion objects that halt integration. Only used if integrator_type is 'simsopt'
            comm: MPI communicator for parallelization
            integrator_type: type of integrator to use ('simsopt' or 'scipy')
            **kwargs: additional arguments to pass to the integrator constructor
        """
        if stopping_criteria is None:
            stopping_criteria = []
        if phis is None:
            phis = 4  # default to 4 planes if not specified
        if integrator_type == 'simsopt':
            integrator = SimsoptFieldlineIntegrator(field, comm=comm, stopping_criteria=stopping_criteria, **kwargs)
        elif integrator_type == 'scipy':
            integrator = ScipyFieldlineIntegrator(field, comm=comm, **kwargs)
        else:
            raise ValueError(f"Integrator type {integrator_type} not supported.")
        return cls(integrator, start_points_RZ, phis=phis, n_transits=n_transits, add_symmetry_planes=add_symmetry_planes)


    def recompute_bell(self, parent=None):
        """
        clear the caches when any object on which this depends changes. 
        """
        self._res_phi_hits = None
        self._res_tys = None
        self._lost = None
        self.need_to_recompute = True
    
    @property 
    def res_tys(self):
        """
        Compute or retrieve the field line trajectories for the Poincare plot.
        If calculation is performed and store_results is True, the results are saved to disk.
        Returns:
            res_tys: list of numpy arrays (one for each particle) containing
                     the trajectory of each field line. Each row of the array contains
                     `[time, x, y, z]`.
        """
        if self.store_results and self._res_tys is None:
            # read from disk if it already exists
            self.retrieve_poincare_data()
        if self._res_tys is None or self.need_to_recompute:
            if isinstance(self.integrator, SimsoptFieldlineIntegrator):
                self._res_tys, self._res_phi_hits = self.integrator.compute_poincare_hits(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits, phis=self.phis)
            else:  # ScipyFieldlineIntegrator 
                self._res_tys = self.integrator.compute_poincare_trajectories(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits)
            if self.store_results:
                self.save_poincare_data()
            self.need_to_recompute = False
        return self._res_tys
    
    @property
    def res_phi_hits(self):
        """
        Compute or retrieve the Poincare section hits for the Poincare plot.
        If calculation is performed and store_results is True, the results are saved to disk.
        Returns:
            res_phi_hits: list of numpy arrays (one for each particle) containing
                          the Poincare section hits of each field line. Each row of the array contains
                          `[phi, plane_index, x, y, z]`.
        """
        if self.store_results and self._res_phi_hits is None:
            # read from disk if it already exists
            self.retrieve_poincare_data()
        if self._res_phi_hits is None or self.need_to_recompute:
            if isinstance(self.integrator, SimsoptFieldlineIntegrator):
                self._res_tys, self._res_phi_hits = self.integrator.compute_poincare_hits(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits, phis=self.phis)
            else:  # ScipyFieldlineIntegrator
                self._res_phi_hits = self.integrator.compute_poincare_hits(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits, phis=self.phis)
            if self.store_results:
                self.save_poincare_data()
            self.need_to_recompute = False
        return self._res_phi_hits
    
    @property
    def poincare_hash(self):
        """
        Generate a hash from the MagneticFields dofs, self.phis, self.start_points_RZ, and self.n_transits. 
        Returns: 
            poincare_hash: hash value
        """
        hash_list = self.integrator.field.full_x.tolist() + self.phis.tolist() + self.start_points_RZ.flatten().tolist() + [self.n_transits]
        poincare_hash = hash(tuple(hash_list))
        return poincare_hash
    
    def save_poincare_data(self, filename=None, name=None):
        """
        Save the computed Poincare data (res_phi_hits and res_tys) to disk for later retrieval.
        The data is by default saved in a file named ``poincare_data.npz``, under a key derived
        from a has from the MagneticField DoFs, and the PoincarePlotter settings. This ensures that data calculated in different sessions or even on different
        machines can be loaded. 
        Args:
            filename: optional filename to override the default ``poincare_data.npz``
            name: optional name to override the default hash-derived key prefix.
        """
        if not self.i_am_the_plotter:
            return

        if filename is None:
            filename = "poincare_data.npz"
        filename = Path(filename)
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")

        if name is None:
            name = self.poincare_hash
        name = str(name)

        data_to_save = {}
        if filename.exists():
            with np.load(filename, allow_pickle=True) as existing:
                data_to_save = {key: existing[key] for key in existing.files}

        updated = False
        if self._res_phi_hits is not None:
            data_to_save[f"res_phi_{name}"] = np.array(self._res_phi_hits, dtype=object)
            updated = True
        if self._res_tys is not None:
            data_to_save[f"res_tys_{name}"] = np.array(self._res_tys, dtype=object)
            updated = True

        if updated:
            np.savez_compressed(filename, **data_to_save)

    def retrieve_poincare_data(self, name=None, filename=None):
        """
        Check if cached Poincare data is available on disk, and load it
        into the current object. By default, a file named ``poincare_data.npz`` 
        is used, but this can be overridden by passing ``filename``.
        The data is stored in keys derived from a hash of the magnetic field DoFs, 
        and the PoincarePlotter settings. 
        This ensures that data calculated in different sessions or even on different
        machines can be loaded. 
        Args:
            name: optional name to override the hash-derived key prefix.
            filename: optional filename to override the default ``poincare_data.npz``.
        """
        if filename is None:
            filename = "poincare_data.npz"
        filename = Path(filename)
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")
        if name is None:
            name = self.poincare_hash
        name = str(name)

        if not filename.exists():
            logger.debug(f"File {filename} not found. Not loading cached poincare data.")
            return

        res_phi_key = f"res_phi_{name}"
        res_tys_key = f"res_tys_{name}"
        with np.load(filename, allow_pickle=True) as data:
            if res_phi_key in data.files:
                loaded_hits = data[res_phi_key]
                self._res_phi_hits = [np.asarray(arr, dtype=float).copy() for arr in loaded_hits]
            if res_tys_key in data.files:
                loaded_tys = data[res_tys_key]
                self._res_tys = [np.asarray(arr, dtype=float).copy() for arr in loaded_tys]

        if self._res_phi_hits is not None or self._res_tys is not None:
            self.need_to_recompute = False
        return
    
    def particles_to_vtk(self, filename):
        """
        Stores the trajectories in a vtk file
        for visualization in paraview.
        Export particle tracing or field lines to a vtk file.
        """
        from pyevtk.hl import polyLinesToVTK
        x = np.concatenate([xyz[:, 1] for xyz in self.res_tys])
        y = np.concatenate([xyz[:, 2] for xyz in self.res_tys])
        z = np.concatenate([xyz[:, 3] for xyz in self.res_tys])
        ppl = np.asarray([xyz.shape[0] for xyz in self.res_tys])
        data = np.concatenate([i*np.ones((self.res_tys[i].shape[0], )) for i in range(len(self.res_tys))])
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})

    
    def remove_poincare_data(self, filename=None):
        """
        Clear the saved poincare data file.
        The hash does not take into account the integrator type or tolerances, so if you need higher precision, use this method
        to clear the file and recompute.
        Args:
            filename: filename to remove (default: poincare_data.npz)
        """
        if not self.i_am_the_plotter:
            return
        if filename is None:
            filename = Path("poincare_data.npz")
        if filename.exists():
            os.remove(filename)
            logger.info(f"Removed poincare data file {filename}.")  
        return

        
    @property
    def lost(self): 
        """
        Get the points where the integration stopped due to a stopping criterion.
        This means the last entry of the 'idx' column in the res_phi_hits array is negative.
        Returns:
            lost: list of booleans indicating whether each field line was lost due to a stopping criterion.
        """
        # list comprehension... look at final element, if element 1 negative, then stopping 
        # criterion was encounterd. first stopping criterion is transit number, so ignore.
        if self._lost is None:
            self._lost = [traj[-1, 1] < -1 for traj in self.res_phi_hits]
        return self._lost

    def plane_hits_cyl(self, plane_idx):
        """
        Get the points where the field lines intersect the plane defined by the index.
        Returns a list of arrays containing the R,Z coordinates of the intersection 
        points.  
        Args: 
            plane_idx: index of the plane to get hits for
        Returns:
            hits: list of arrays of shape (n_hits, 2) containing the R,Z points of the hits. 
        """
        hits_cart = self.plane_hits_cart(plane_idx)
        hits = [self.integrator._xyz_to_rphiz(hc)[:, ::2] for hc in hits_cart]
        return hits  # return only R,Z
    
    
    def plane_hits_cart(self, plane_idx):
        """
        Get the points where the field lines hit the plane defined by the index. 
        Returns a list of arrays containing the x,y,z coordinates of the intersection 
        points.
        Args:
            plane_idx: index of the plane to get hits for
        Returns:
            hits: list of arrays of shape (n_hits, 3) containing the x,y,z points of the hits.
        """
        if plane_idx >= len(self.phis):
            raise ValueError(f"Plane index {plane_idx} is larger than the number of planes {len(self.phis)}.")
        
        hits = []
        for traj in self.res_phi_hits:
            hits_xyz = traj[np.where(traj[:, 1] == plane_idx)[0], 2:]  # res_phi_hits col1 = plane idx or stopping criterion
            hits.append(hits_xyz)  # append the xyz points
        return hits  # list of arrays of shape (n_hits, 3)
    
    @staticmethod
    def fix_axes(ax, xlabel='R', ylabel='Z', title=None):
        """
        Label the axes of the plot. 
        Args: 
            ax: matplotlib axis to label
            xlabel: label for the x-axis
            ylabel: label for the y-axis
            title: title for the plot (if None, no title is set)
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)

    def plot_poincare_plane_idx(self, plane_idx, mark_lost=False, ax=None, **kwargs):
        """
        plot a single cross-section of the field by referencing the index in the PoincarePlotter's phis. 
        *NOTE*: if running parallel, call this function on all ranks.
        Args:
            plane_idx: index of the plane to plot
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            ax: matplotlib axis to plot on (if None, create a new figure and axis)
            **kwargs: additional keyword arguments to pass to the scatter plotter
        Returns:
            fig, ax: the figure and axis objects (only on rank 0, otherwise None, None)
        """
        if plane_idx >= len(self.phis):
            raise ValueError(f"Plane index {plane_idx} is larger than the number of planes {len(self.phis)}.")
        
        # can trigger recompute, so all ranks execute
        hits_thisplane = self.plane_hits_cyl(plane_idx)

        # rank0 only
        if self.i_am_the_plotter: 
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            
            marker = kwargs.pop('marker', '.')
            s = kwargs.pop('s', 2.5)
            color = kwargs.pop('color', 'random')

            for idx, trajpoints in enumerate(hits_thisplane):
                if color == 'random':
                    this_color = self.randomcolors[idx]
                else: 
                    this_color = color

                this_marker = marker
                this_s = s
                if mark_lost:
                    lost = self.lost[idx]
                    if lost:
                        this_color = 'r'
                        this_marker = 'x'
                        this_s = s*3
                ax.scatter(trajpoints[:, 0], trajpoints[:, 1], marker=this_marker, s=this_s, color=this_color, linewidths=0, **kwargs)
            return fig, ax
        else: 
            return None, None  # other ranks do not plot anything

    def plot_poincare_single(self, phi, prevent_recompute=False, ax=None, mark_lost=False, fix_axes=True, surf=None, include_symmetry_planes=True, **kwargs):
        """
        plot a single cross-section of the field at a given phi value. 
        If this value is not in the list of phis, this phi will be added to the PoincarePlotters planes
        and the sections will be re-computed (unless prevent_recompute is True, in which case an error is raised). 
        *NOTE*: if running parallel, call this function on all ranks. 
        Args: 
            phi: angle in [0, 2pi] at which to plot the Poincare section
            prevent_recompute: if True, do not trigger a recompute if the requested plane is not available
            ax: matplotlib axis to plot on (if None, create a new figure and axis)
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            fix_axes: if True, fix the axes to be equal and labeled (otherwise, deal with the returned axes object)
            surf: if given, a simsopt surface to plot the cross-section of (in black)
            include_symmetry_planes: if True, include all planes that are identical through field periodicity in this plot
            **kwargs: additional keyword arguments to pass to the single plane plotter
        Returns:
            fig, ax: the figure and axis objects (only on rank 0, otherwise None, None)
        """
        if phi not in self.phis:
            if not prevent_recompute:
                self.phis = np.append(self.phis, phi)
                phi_indices = [len(self.phis) - 1]  # index of the newly added plane
            else:
                raise ValueError(f"The requested plane at phi={phi} has not been computed.")
        else:
            if include_symmetry_planes:
                phi_indices = np.where(np.isclose((self.phis - phi) % (2*np.pi/self.integrator.nfp), 0))[0]
            else: 
                phi_indices = np.where(np.isclose(self.phis, phi))[0]
        
        # trigger recompute on all ranks if necessary:
        _ = self.res_phi_hits

        if self.i_am_the_plotter:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            for phi_index in phi_indices:
                self.plot_poincare_plane_idx(phi_index, ax=ax, mark_lost=mark_lost, **kwargs)

            if surf is not None:
                # divide by 2pi cause simsopt surf phi is in [0,1]
                cross_section = surf.cross_section(phi=phi/(2*np.pi))
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
        Plot all the computed poincare planes in a grid. 
        A square grid is generated, so best results are achieved if the plotter uses 4 or 9 planes. 
        *NOTE*: if running parallel, call this function on all ranks. 
        Args:
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            fix_ax: if True, fix the axes to be equal and labeled (otherwise, deal with the returned axes object)
            **kwargs: additional keyword arguments to pass to the single plane plotter
        Returns:
            fig, axs: the figure and axes objects (only on rank 0, otherwise None)
        """
        _ = self.res_phi_hits  #trigger recompute on all ranks if necessary

        if self.i_am_the_plotter:
            from math import ceil
            nrowcol = ceil(sqrt(len(self.phis_for_plotting)))
            fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
            
            axs = np.atleast_1d(axs).ravel()  # make array and flatten

            for section_idx, phi in enumerate(self.phis_for_plotting):  #ony the plane in the first field period
                ax = axs[section_idx]
                self.plot_poincare_single(phi, ax=ax, mark_lost=mark_lost, prevent_recompute=True, fix_axes=fix_ax, **kwargs)
                textstr = f" φ = {phi/np.pi:.2f}π "
                props = dict(boxstyle='round', facecolor='white', edgecolor='black')
                ax.text(0.05, 0.02, textstr, transform=ax.transAxes, fontsize=6,
                        verticalalignment='bottom', bbox=props)

            plt.tight_layout()

            return fig, axs
        else: 
            return None, None  # other ranks do not plot anything

    def plot_fieldline_trajectories_3d(self, engine='mayavi', mark_lost=False, show=True,  **kwargs): 
        """
        Plot the full 3D trajectories of the field lines. 
        Can be very busy if lines are followed for long. 

        Hints: 
            - for mayavi, use tube_radius kwarg to adjust line thickness, opacity for making them transparent. 
        Args:
            engine: 'mayavi' or 'plotly' or 'matplotlib'
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            show: if True, show the plot immediately
            **kwargs: additional keyword arguments to pass to the plotting function
        Returns:
            None

        """
        trajectories = self.res_tys  # trigger recompute if necessary

        if self.i_am_the_plotter:
            # unify color kw handling across engines
            if 'color' in kwargs:
                base_color = kwargs.pop('color')
            else:
                base_color = 'random'
            if engine == 'mayavi':
                from mayavi import mlab
                tube_radius = kwargs.pop('tube_radius', 0.005)
                color = base_color
                for idx, traj in enumerate(trajectories): 
                    if color == 'random':
                        this_color = tuple(self.randomcolors[idx])
                    else: 
                        this_color = tuple(color)
                    lost = self.lost[idx] if mark_lost else False
                    this_tube_radius = tube_radius*3 if lost else tube_radius
                    this_color = (1, 0, 0) if lost else this_color
                    mlab.plot3d(traj[:, 1], traj[:, 2], traj[:, 3], tube_radius=this_tube_radius, color=this_color, **kwargs)
                if show:
                    mlab.show()
            if engine == 'plotly':
                import plotly.graph_objects as go
                fig = go.Figure()
                color = base_color
                for idx, traj in enumerate(trajectories): 
                    if color == 'random':
                        this_color = 'rgb({},{},{})'.format(*(self.randomcolors[idx]*255).astype(int))
                    else: 
                        this_color = color
                    lost = self.lost[idx] if mark_lost else False
                    this_width = 6 if lost else 2
                    this_color = 'rgb(255,0,0)' if lost else this_color
                    fig.add_trace(go.Scatter3d(x=traj[:, 1], y=traj[:, 2], z=traj[:, 3], mode='lines', line=dict(color=this_color, width=this_width), **kwargs))
                fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                    width=800,
                    margin=dict(r=20, b=10, l=10, t=10))
                if show:
                    fig.show()
            elif engine == 'matplotlib':
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                color = base_color
                for idx, traj in enumerate(trajectories):
                    if color == 'random':
                        this_color = self.randomcolors[idx]
                    else:
                        this_color = color
                    lost = self.lost[idx] if mark_lost else False
                    this_width = 6 if lost else 2
                    this_color = 'rgb(255,0,0)' if lost else this_color
                    ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color=this_color, linewidth=this_width, **kwargs)
                if show:
                    plt.show()

    def plot_poincare_in_3d(self, engine='mayavi', mark_lost=False, show=True, **kwargs):
        """
        Plot the Poincare points in 3D. Useful to visualize the poincare planes together with the coils and field lines. 
        Args: 
            engine: 'mayavi' or 'plotly' or 'matplotlib'
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            show: if True, show the plot immediately
            **kwargs: additional keyword arguments to pass to the plotting function
        """
        _ = self.res_phi_hits  # trigger recompute if necessary

        if self.i_am_the_plotter:
            # unify color kw handling across engines
            if 'color' in kwargs:
                base_color = kwargs.pop('color')
            else:
                base_color = 'random'
            if engine == 'mayavi':
                from mayavi import mlab
                marker = kwargs.pop('marker', 'sphere')
                scale_factor = kwargs.pop('scale_factor', 0.005)
                color = base_color
                for idx in range(len(self.phis)):
                    plane_hits = self.plane_hits_cart(idx)
                    for traj_idx, hit_group in enumerate(plane_hits):
                        if color == 'random':
                            this_color = tuple(self.randomcolors[traj_idx])
                        else: 
                            this_color = tuple(color)
                        
                        this_scale_factor = scale_factor
                        if mark_lost:
                            lost = self.lost[traj_idx]
                            this_scale_factor = scale_factor*3 if lost else scale_factor
                            this_color = (1, 0, 0) if lost else this_color
                        mlab.points3d(hit_group[:, 0], hit_group[:, 1], hit_group[:, 2], scale_factor=this_scale_factor, color=this_color, mode=marker, **kwargs)
                if show:
                    mlab.show()
            if engine == 'plotly':
                import plotly.graph_objects as go
                fig = go.Figure()
                color = base_color
                for idx in range(len(self.phis)):
                    plane_hits = self.plane_hits_cart(idx)
                    for traj_idx, hit_group in enumerate(plane_hits):
                        if color == 'random':
                            this_color = 'rgb({},{},{})'.format(*(self.randomcolors[traj_idx]*255).astype(int))
                        else: 
                            this_color = color
                        lost = self.lost[traj_idx] if mark_lost else False
                        this_size = 8 if lost else 4
                        this_color = 'rgb(255,0,0)' if lost else this_color
                        fig.add_trace(go.Scatter3d(x=hit_group[:, 0], y=hit_group[:, 1], z=hit_group[:, 2], mode='markers', marker=dict(size=this_size, color=this_color), **kwargs))
                fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                    width=800,
                    margin=dict(r=20, b=10, l=10, t=10))
                if show:
                    fig.show()
            elif engine == 'matplotlib':
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                color = base_color
                for idx in range(len(self.phis)):
                    plane_hits = self.plane_hits_cart(idx)
                    for traj_idx, hit_group in enumerate(plane_hits):
                        if color == 'random':
                            this_color = self.randomcolors[traj_idx]
                        else:
                            this_color = color
                        lost = self.lost[traj_idx] if mark_lost else False
                        this_size = 80 if lost else 40
                        this_color = 'rgb(255,0,0)' if lost else this_color
                        ax.scatter(hit_group[:, 0], hit_group[:, 1], hit_group[:, 2], color=this_color, s=this_size, **kwargs)
                if show:
                    plt.show()  
    # TODO: use res_tys to plot rotational transform
