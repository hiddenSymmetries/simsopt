import logging
from math import sqrt

import numpy as np
from pathlib import Path
import os

import simsoptpp as sopp
from .._core import Optimizable, ObjectiveFailure
from .._core.util import parallel_loop_bounds
from ..field.magneticfield import MagneticField
from ..field.magneticfieldclasses import InterpolatedField
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

    Args:
        field: the magnetic field :math:`B`
        R0: list of radial components of initial points
        Z0: list of vertical components of initial points
        phi0: toroidal angle of initial points
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
    Base class for Integrators. 
    Integrators provide a method `compute_poincare_hits` which returns 
    res_phi_hits that interfaces with the PoincarePlotter. 
    It can also provide other integration services on a MagneticField. 
    and being an Optimizable, the results of integrations can be
    optimized. 
    """
    def __init__(self, field: MagneticField, comm=None, nfp=None, stellsym=False, R0=None, test_symmetries=True):
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
            self.interpolated = True
        else:
            self.interpolated = False
        Optimizable.__init__(self, depends_on=[field,])

    def test_symmetries(self):
        """
        Test if the number of field periods and the 
        stellarator symmetry setting are consistent
        for the given field.
        """
        rphiz_point = np.array([self.R0, 0.1, .1])[None, :]
        rphiz_stellsym = np.array([self.R0, -0.1, -0.1])[None, :]
        rphiz_periodicity = np.array([self.R0, 0.1+(2*np.pi/self.nfp), 0.1])[None, :]
        self.field.set_points_cyl(rphiz_point)
        test_field = np.copy(self.field.B_cyl())
        self.field.set_points_cyl(rphiz_stellsym)
        test_field_stellsym = np.copy(self.field.B_cyl())
        self.field.set_points_cyl(rphiz_periodicity)
        test_field_periodicity = np.copy(self.field.B_cyl())
        stellsym_test = np.allclose(test_field, test_field_stellsym * np.array([-1, 1, 1]))
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
        # Normalize ranges to (min, max, n) tuples expected by InterpolatedField
        def _norm_range(rng):
            rng_t = tuple(rng)
            if len(rng_t) == 2:
                return (rng_t[0], rng_t[1], 4)
            elif len(rng_t) == 3:
                return rng_t
            else:
                raise ValueError("Range must have 2 or 3 elements: (min,max[,n])")

        rrange_n = _norm_range(rrange)
        phirange_n = _norm_range(phirange)
        zrange_n = _norm_range(zrange)

        interpolatedfield = InterpolatedField(
            self.field,
            degree,
            rrange_n,
            phirange_n,
            zrange_n,
            extrapolate=extrapolate,
            nfp=self.nfp,
            stellsym=self.stellsym,
            skip=skip,
        )
        self._true_field = self.field
        self._rrange = rrange_n
        self._phirange = phirange_n
        self._zrange = zrange_n
        self._degree = degree
        self._skip = skip
        self._extrapolate = extrapolate
        self.field = interpolatedfield
        self.interpolated = True

    def recompute_bell(self, parent=None):
        """
        If the field is interpolated, recompute the interpolation. 
        This should be called if the underlying field has changed.
        Currently every change will trigger recomputaton of the interpolation. 
        Do not use this in optimization! 
        """
        if self.interpolated:
            logger.warning("Integrator recompute bell was rung, indicating need to recmpute interpolation. " \
                           "Currently this is not implemented")
            # cannot call this every time a dof is changed, need to work with flag that triggers on first
            # call to the underlying field.
#            self.field = InterpolatedField(
#                self._true_field,
#                self._degree,
#                self._rrange if len(self._rrange) == 3 else (self._rrange[0], self._rrange[1], 4),
#                self._phirange if len(self._phirange) == 3 else (self._phirange[0], self._phirange[1], 4),
#                self._zrange if len(self._zrange) == 3 else (self._zrange[0], self._zrange[1], 4),
#                extrapolate=self._extrapolate,
#                nfp=self.nfp,
#                stellsym=self.stellsym,
#                skip=self._skip,
#            )
        else:
            pass


    @staticmethod
    def _rphiz_to_xyz(array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert an array of R, phi, Z to x, y, z
        """
        if array.ndim == 1 and array.shape[0] == 3:
            array2 = array[None, :]
        elif array.ndim == 2 and array.shape[1] == 3:
            array2 = array[:, :]
        else: 
            raise ValueError("Input array must be of shape (3,) or (n,3)")

        return np.array([array2[:, 0]*np.cos(array2[:, 1]), array2[:, 0]*np.sin(array2[:, 1]), array2[:, 2]]).T

    @staticmethod
    def _xyz_to_rphiz(array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert an array of x, y, z to R, phi, Z
        """
        if array.ndim == 1 and array.shape[0] == 3:
            array2 = array[None, :]
        elif array.ndim == 2 and array.shape[1] == 3:
            array2 = array[:, :]
        else: 
            raise ValueError("Input array must be of shape (3,) or (n,3)")
        
        return np.array([np.sqrt(array2[:, 0]**2 + array2[:, 1]**2), np.arctan2(array2[:, 1], array2[:, 0]), array2[:, 2]]).T


class SimsoptFieldlineIntegrator(Integrator):
    """
    Integration of field lines using the `simsoptpp` routines. 
    Integration is performed in three dimensions, solving the 
    ODE:
    .. math::
        \frac{d\mathbf{x}(t)}{dt} = \mathbf{B}(\mathbf{x})
    where :math:`\mathbf{x}=(x,y,z)` are the cartesian coordinates.
    MPI parallelization is supported by passing an MPI communicator. 
    """
    
    def __init__(self, field, comm=None, nfp=None, stellsym=False, R0=None, test_symmetries=True, stopping_criteria=[], tol=1e-9, tmax=1e4):
        self.tol = tol
        self.stopping_criteria = stopping_criteria
        self.tmax = tmax
        super().__init__(field, comm=comm, nfp=nfp, stellsym=stellsym, R0=R0, test_symmetries=test_symmetries)

    def compute_poincare_hits(self, start_points_RZ, n_transits, phis=[], phi0=0):
        """
        calculate the res_phi_hits. 
        Args:
            start_points_RZ: nx2 array of starting radial coordinates.
            n_transits: number of times to go around the device.
            phis: list of angles in [0, 2pi] for which intersection with the plane
                  corresponding to that phi should be computed. If empty, only phi=0 is used.
        Returns:
            res_phi_hits: list of numpy arrays (one for each particle) containing
                     information on each time the particle hits one of the phi planes.
                     Each row of the array contains `[time, idx, x, y, z]`, where `idx`
                     tells us which of the `phis` was hit. If `idx>=0`, then
                     `phis[int(idx)]` was hit. If `idx<0`, then one of the stopping criteria
                     was hit, which is specified in the `stopping_criteria` argument.
                  """
        # TODO: Add self.stopping_criteria
        stopping_criteria = [ToroidalTransitStoppingCriterion(n_transits, isinstance(self.field, BoozerMagneticField)), ]
        return compute_fieldlines(
            self.field, start_points_RZ[:, 0], start_points_RZ[:, 1], phi0=phi0, tmax=self.tmax, tol=self.tol,
            phis=phis, stopping_criteria=stopping_criteria, comm=self.comm)

    def integrate_in_phi_cart(self, start_xyz, delta_phi=2*np.pi, return_cartesian=True):
        """
        Integrate the field line in phi from xyz location start_xyz over an angle delta_phi using simsoptpp routines.  

        Args:
            start_xyz: starting cartesian coordinates Array[x,y,z]
            delta_phi: angle to integrate over
            return_cartesian: if True, return the cartesian coordinates of the end point, otherwise return only R,Z (phi_end assumed)
        """
        start_phi = self._xyz_to_rphiz(start_xyz)[0, 1]
        phi_end = start_phi + delta_phi
        if isinstance(self.field, BoozerMagneticField):  # can InterpolatedField also be a flux field?
            flux = True
        else:
            flux = False
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
            stopping_criteria=[ToroidalTransitStoppingCriterion(1.0, flux),]
            )
        xyz = np.array(res_phi_hits[-2][2:])  # second to last hit is the phi_end hit
        if return_cartesian:  # return [x,y,z] array
            return xyz
        else:  # return [R,Z] array
            return self._xyz_to_rphiz(xyz)[[0,2]]

    def integrate_in_phi_cyl(self, start_RZ, start_phi=0, delta_phi=2*np.pi, return_cartesian=True):
        """
        Integrate the field line giving thes tarting location in cylindrical coordinates.
        Integrate the field line over an angle phi from the starting point given by 
        cartesian coordinates (x,y,z).
        """
        start_xyz = self._rphiz_to_xyz(np.array([start_RZ[0], start_phi, start_RZ[1]]))[-1, :]  
        return self.integrate_in_phi_cart(
            start_xyz, delta_phi=delta_phi, return_cartesian=return_cartesian)

    def integrate_fieldlinepoints_cart(self, start_xyz, n_transits=1, return_cartesian=True):
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
        if isinstance(self.field, BoozerMagneticField):  # can InterpolatedField also be a flux field?
            flux = True
        else:
            flux = False
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
            stopping_criteria=[ToroidalTransitStoppingCriterion(n_transits, flux),])
        points_cart = np.array(res_tys)[:, 1:]
        if return_cartesian:
            return points_cart
        else:
            return self._xyz_to_rphiz(points_cart)

    def integrate_fieldlinepoints_cyl(self, start_RZ, start_phi, n_transits=1, return_cartesian=True):
        """
        Integrate the field line over an angle phi from the starting point given by 
        cartesian coordinates (x,y,z).
        """
        start_xyz = self._rphiz_to_xyz(np.array([start_RZ[0], start_phi, start_RZ[1]]))[-1, :]
        return self.integrate_fieldlinepoints_cart(
            start_xyz, n_transits=n_transits, return_cartesian=return_cartesian)


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
    def __init__(self, field, comm=None, nfp=None, stellsym=False, R0=None, test_symmetries=True, integrator_type='RK45', integrator_args=dict()):
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
        # check if field is in positive B_phi direction at (R0,0,0)
        if R0 is None:
            logger.warning("R0 is not set, using default value of 1.")
            R0 = 1
        field.set_points_cyl(np.array([[R0, 0.0, 0.0]]))
        B = field.B_cyl().flatten()
        if B[1] < 0:
            self.flip_B = True
        else:
            self.flip_B = False
        super().__init__(field, comm, nfp, stellsym, R0, test_symmetries)
        self._integrator_type = integrator_type
        self._integrator_args = integrator_args
        # update integrator args with defaults
        if 'rtol' not in self._integrator_args:
            self._integrator_args['rtol'] = 1e-7
        if 'atol' not in self._integrator_args:
            self._integrator_args['atol'] = 1e-9

    def compute_poincare_hits(self, start_points_RZ, n_transits, phis=[], phi0=0):
        """
        calculate the res_phi_hits array using integration in phi. 
        """
        res_phi_hits = []
        first, last = parallel_loop_bounds(self.comm, len(start_points_RZ))
        for this_start_RZ in start_points_RZ[first:last, :]:
            logger.info(f'Integrating poincare section for field line starting at R={this_start_RZ[0]}, Z={this_start_RZ[1]}')
            all_phis = np.array([phis + 2*np.pi*nt for nt in range(int(n_transits))]).flatten()
            integration_solution = solve_ivp(self.integration_fn_cyl, 
                                             [all_phis[0], all_phis[-1]], 
                                             this_start_RZ, 
                                             t_eval=all_phis, 
                                             events=self.event_function, 
                                             method=self._integrator_type, 
                                             rtol=self._integrator_args['rtol'], 
                                             atol=self._integrator_args['atol']
                                             )
            phis_values = integration_solution.t % (2*np.pi)
            rphiz_values = np.array([integration_solution.y[0, :], phis_values, integration_solution.y[1, :]]).T
            xyz_values = self._rphiz_to_xyz(rphiz_values)
            plane_idx = np.array(range(len(all_phis))) % len(phis)
            res_phi_hits_line = np.column_stack((phis_values, 
                                         plane_idx, 
                                         xyz_values))
            if integration_solution.status == -1 or integration_solution.status == 1:  # integration failed
                res_phi_hits_line[-1, 1] = -1
            res_phi_hits.append(res_phi_hits_line)
        if self.comm is not None:
            res_phi_hits = [hit for gathered_hits in self.comm.allgather(res_phi_hits) for hit in gathered_hits]
        return res_phi_hits
    
    def compute_poincare_trajectories(self, start_points_RZ, n_transits, phi0=0):
        """
        calculate the res_tys array using integration in phi. 
        """
        res_tys = []
        first, last = parallel_loop_bounds(self.comm, len(start_points_RZ))
        for this_start_RZ in start_points_RZ[first:last, :]:
            logger.info(f'Integrating trajectory of field line starting at R={this_start_RZ[0]}, Z={this_start_RZ[1]}')
            start_phi = phi0
            phi_end = start_phi + n_transits*2*np.pi
            phi_eval = np.linspace(start_phi, phi_end, int(1000*int(n_transits)))
            integration_solution = solve_ivp(self.integration_fn_cyl, 
                                             [start_phi, phi_end], 
                                             this_start_RZ, 
                                             t_eval=phi_eval, 
                                             events=self.event_function, 
                                             method=self._integrator_type, 
                                             rtol=self._integrator_args['rtol'], 
                                             atol=self._integrator_args['atol']
                                             )
            phis_values = integration_solution.t % (2*np.pi)
            rphiz_values = np.array([integration_solution.y[0, :], phis_values, integration_solution.y[1, :]]).T
            xyz_values = self._rphiz_to_xyz(rphiz_values)
            res_tys_line = np.column_stack((integration_solution.t, 
                                         xyz_values))
            res_tys.append(res_tys_line)
        if self.comm is not None:
            res_tys = [ty for gathered_tys in self.comm.allgather(res_tys) for ty in gathered_tys]
        return res_tys

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
            return np.abs(self.field.B_cyl().dot(np.array([0, 1., 0.]) / np.linalg.norm(self.field.B()))) - 1e-3  # B_phi = 0
        #_event.terminal = True
        return _event

    def integrate_in_phi_cyl(self, start_RZ, start_phi=0, delta_phi=2*np.pi, return_cartesian=False):
        """
        Integrate the field line using scipy's odeint method
        """
        sol = solve_ivp(self.integration_fn_cyl, [start_phi, start_phi + delta_phi], start_RZ, events=self.event_function, method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        if not sol.success:
            return np.array(np.nan, np.nan)
        if return_cartesian:
            rphiz = np.array([sol.y[0, -1], start_phi + delta_phi, sol.y[1, -1]])
            return self._rphiz_to_xyz(rphiz)[-1, :]
        else:
            return sol.y[:, -1]
    
    def integrate_in_phi_cart(self, start_xyz, delta_phi=2*np.pi, return_cartesian=True):
        """
        Integrate the field line using scipy's odeint method, giving the start point in cartesian coordinates.
        """
        rphiz_start = self._xyz_to_rphiz(start_xyz)[-1]
        start_RZ = rphiz_start[[0, 2]]
        start_phi = rphiz_start[1]
        return self.integrate_in_phi_cyl(start_RZ, start_phi, delta_phi, return_cartesian=return_cartesian)
    
    # def d_integration_by_dcoeff(self, start_RZ, start_phi, phi_end):
    # def d_

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
        sol = solve_ivp(self.integration_fn_cyl, [phis[0], phis[-1]], start_RZ, t_eval=phis, events=self.event_function, method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        success = sol.success
        if success:
            rphiz = np.array([sol.y[0, :], sol.t, sol.y[1, :]]).T
        else:
            rphiz = np.full((len(phis), 3), np.nan)
        if return_cartesian:
            return success, self._rphiz_to_xyz(rphiz)
        else:
            return success, rphiz
    
    def integrate_fieldlinepoints_cyl(self, start_RZ, start_phi, delta_phi, n_points, endpoint=False, return_cartesian=True):
        """
        integrate a fieldline for a given toroidal distance, giving the start point in R,Z. 
        """
        phis = np.linspace(start_phi, start_phi+delta_phi, n_points, endpoint=endpoint)
        sucess, points = self.integrate_cyl_planes(start_RZ, phis, return_cartesian=return_cartesian)
        if not sucess:
            raise ObjectiveFailure("Integration failed")
        return points

    def integrate_fieldlinepoints_cart(self, start_xyz, delta_phi, n_points, endpoint=False, return_cartesian=True):
        """
        integrate a fieldline for a given toroidal distance, giving the start point in xyz. 
        """
        rphiz_start = self._xyz_to_rphiz(start_xyz)[-1]
        start_RZ = rphiz_start[[0, 2]]
        start_phi = rphiz_start[1]
        points = self.integrate_fieldlinepoints_cyl(start_RZ, start_phi, delta_phi, n_points, endpoint=endpoint, return_cartesian=return_cartesian)
        return points

    def integration_fn_3d(self, t, xyz):
        self.field.set_points(xyz[None, :])
        B = self.field.B().flatten()
        return B/np.linalg.norm(B)

    def integrate_3d_fieldlinepoints_cart(self, start_xyz, l_total, n_points):
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
        sol = solve_ivp(self.integration_fn_3d, [0, l_total], start_xyz, t_eval=np.linspace(0, l_total, n_points), method=self._integrator_type, rtol=self._integrator_args['rtol'], atol=self._integrator_args['atol'])
        return sol.y.T
    
    #TODO: add jacobian of integrand


class PoincarePlotter(Optimizable):
    """
    Class to facilitate the calculation of field lines and 
    plotting the results in a Poincare plot. 
    Uses field periodicity to speed up calculation
    """
    def __init__(self, integrator: Integrator, start_points_RZ, phis=None, n_transits=100, add_symmetry_planes=True, store_results=False):
        """
        Initialize the PoincarePlotter. 
        This class uses an Integrator to compute field lines, and takes care of plotting them. 
        If the field is stellarator-symmetric, and symmetry planes are included, then 
        these are overplotted in the plot (resulting in better plots with shorter integration). 

        Args:
            integrator: the integrator to be used for the calculation
            start_points_RZ: nx2 array of starting points in cylindrical coordinates (R,Z)
            phis: angles in [0, 2pi] for which we wish to compute Poincare.
                  *OR* int: number of planes to compute, equally spaced in [0, 2pi/nfp].
            n_transits: number of toroidal transits to compute
            add_symmetry_planes: if true, we add planes that are identical through field periodicity, increasing the efficiency of the calculation. 
            store_results: if true, save the results to disk in files with a unique filename based on a hash. 
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
        
        self.phi0 = self._phis[0]
        self.i_am_the_plotter = self.integrator.comm is None or self.integrator.comm.rank == 0  # only rank 0 does plotting
        self.need_to_recompute = True
        self._randomcolors = None
        Optimizable.__init__(self, depends_on=[integrator,])
        self.store_results = store_results
        if store_results:
            # load file form disk if it exists
            self.load_from_disk()

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
        return np.linspace(0, 2*np.pi/nfp, nplanes, endpoint=False)
    
    @staticmethod
    def generate_symmetry_planes(phis, nfp=1):
        """
        Given a list of phis in [0, 2pi/nfp], generate the full list of phis
        in [0, 2pi] by adding the symmetry planes. 
        """
        list_of_phis = [phis + per_idx*2*np.pi/nfp for per_idx in range(nfp)]
        # remove duplicates and sort
        list_of_phis = np.unique(np.concatenate(list_of_phis))
        return list_of_phis
    
    @property
    def plot_phis(self):
        """
        the phis in the first period, useful for plotting
        """
        plot_period = 2*np.pi/self.integrator.nfp
        plot_phis = self.phis[np.where(self.phis < plot_period)]
        return plot_phis

    @property
    def start_points_RZ(self):
        return self._start_points_RZ

    @start_points_RZ.setter
    def start_points_RZ(self, array):
        self._start_points_RZ = array
        self.recompute_bell()

    @property
    def phis(self):
        return self._phis
    
    @phis.setter
    def phis(self, value):
        self._phis = value
        self.recompute_bell()


    @classmethod
    def from_field(cls, field, start_points_RZ, phis=None, n_transits=1, add_symmetry_planes=True,
                   stopping_criteria=None, comm=None, integrator_type='simsopt', tol=1e-9, tmax=200):
        """Factory to create a PoincarePlotter directly from a MagneticField.

        Parameters mirror PoincarePlotter.__init__, while constructing the appropriate integrator.
        phis: int | sequence | None. If int, that many planes in [0, 2pi/nfp). If None, defaults to a single plane at 0.
        n_transits: number of toroidal transits for poincare computation.
        add_symmetry_planes: if True, replicate phis over field periods.
        tol, tmax: integrator tolerance and max integration time (passed to SimsoptFieldlineIntegrator).
        """
        if stopping_criteria is None:
            stopping_criteria = []
        if phis is None:
            phis = 4  # default to 4 planes if not specified
        if integrator_type == 'simsopt':
            integrator = SimsoptFieldlineIntegrator(field, comm=comm, stopping_criteria=stopping_criteria, tol=tol, tmax=tmax)
        else:
            raise ValueError(f"Integrator type {integrator_type} not supported.")
        return cls(integrator, start_points_RZ, phis=phis, n_transits=n_transits, add_symmetry_planes=add_symmetry_planes)


    def recompute_bell(self, parent=None):
        self._res_phi_hits = None
        self._res_tys = None
        self._lost = None
        self.need_to_recompute = True
    
    @property  # TODO: make different if scipy integrator
    def res_tys(self):
        if self.store_results and self._res_tys is None:
            # read from disk if it already exists
            self.load_from_disk()
        if self._res_tys is None or self.need_to_recompute:
            if isinstance(self.integrator, SimsoptFieldlineIntegrator):
                self._res_tys, self._res_phi_hits = self.integrator.compute_poincare_hits(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits, phis=self.phis)
            else:  # ScipyFieldlineIntegrator 
                self._res_tys = self.integrator.compute_poincare_trajectories(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits)
            if self.store_results:
                self.save_to_disk()
            self.need_to_recompute = False
        return self._res_tys
    
    @property
    def res_phi_hits(self):
        if self.store_results and self._res_phi_hits is None:
            # read from disk if it already exists
            self.load_from_disk()
        if self._res_phi_hits is None or self.need_to_recompute:
            if isinstance(self.integrator, SimsoptFieldlineIntegrator):
                self._res_tys, self._res_phi_hits = self.integrator.compute_poincare_hits(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits, phis=self.phis)
            else:  # ScipyFieldlineIntegrator
                self._res_phi_hits = self.integrator.compute_poincare_hits(
                    self.start_points_RZ, phi0=self.phi0, n_transits=self.n_transits, phis=self.phis)
            if self.store_results:
                self.save_to_disk()
            self.need_to_recompute = False
        return self._res_phi_hits
    
    @property
    def poincare_hash(self):
        """
        Generate a hash from dofs, self.phis, self.start_points_RZ, and self.n_transits
        """
        hash_list = self.integrator.field.full_x.tolist() + self.phis.tolist() + self.start_points_RZ.flatten().tolist() + [self.n_transits]
        poincare_hash = hash(tuple(hash_list))
        return poincare_hash
    
    def save_to_disk(self, filename=None, name=None):
        """
        Persist the current state of the PoincarePlotter. By default the results
        are stored inside ``poincare_data.npz`` under keys derived from the
        poincare hash. Passing ``filename`` allows saving to an alternate archive
        location, while ``name`` can override the hash-derived key prefix.
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

    def load_from_disk(self, name=None, filename=None):
        """
        Load previously stored results for this plotter from ``poincare_data.npz``
        (or a user-specified archive). Results are keyed by the poincare hash,
        which can be overridden via ``name``.
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
        legacy method from sopp, stores the trajectories in a vtk file
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
        """
        hits_cart = self.plane_hits_cart(plane_idx)
        hits = [self.integrator._xyz_to_rphiz(hc)[:, ::2] for hc in hits_cart]
        return hits  # return only R,Z
    
    
    def plane_hits_cart(self, plane_idx):
        """
        Get the points where the field lines hit the plane defined by the index. 
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
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)

    def plot_poincare_plane_idx(self, plane_idx, mark_lost=False, ax=None, **kwargs):
        """
        plot a single cross-section of the field at a given plane index. 
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
        If this value is not in the list of phis, a recompute will be triggered. 
        *NOTE*: if running parallel, call this function on all ranks. 
        """
        if phi not in self.phis:
            if not prevent_recompute:
                self.phis.append(phi)
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
        _ = self.res_phi_hits  #trigger recompute on all ranks if necessary

        if self.i_am_the_plotter:
            from math import ceil
            nrowcol = ceil(sqrt(len(self.plot_phis)))
            fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
            
            axs = np.atleast_1d(axs).ravel()  # make array and flatten

            for section_idx, phi in enumerate(self.plot_phis):  #ony the plane in the first field period
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
        Uses mayavi or plotly for plotting. 
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
        Plot the Poincare points in 3D. 
        Uses mayavi or plotly for plotting. 
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