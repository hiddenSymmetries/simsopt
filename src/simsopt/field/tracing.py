from math import sqrt
import numpy as np
import simsoptpp as sopp
import logging
from simsopt.field.magneticfield import MagneticField
from nptyping import NDArray, Float


logger = logging.getLogger(__name__)


def compute_gc_radius(m, vperp, q, absb):

    """
    Computes the gyro radius of a particle in a field with strenght ``absb```,
    that is ``r=m*vperp/(abs(q)*absb)``.
    """

    return m*vperp/(abs(q)*absb)


def gc_to_fullorbit_initial_guesses(field, xyz_inits, vtangs, vtotal, m, q):

    """
    Takes in guiding center positions ``xyz_inits`` as well as a tangential
    velocities ``vtangs`` and total velocities ``vtotal`` to compute orbit
    positions for a full orbit calculation that matches the given guiding
    center.
    """

    nparticles = xyz_inits.shape[0]
    xyz_inits_full = np.zeros_like(xyz_inits)
    v_inits = np.zeros((nparticles, 3))
    rgs = np.zeros((nparticles, ))
    field.set_points(xyz_inits)
    Bs = field.B()
    AbsBs = field.AbsB()
    eB = Bs/AbsBs
    ez = np.asarray(nparticles*[[0., 0., -1.]])
    ez -= eB * np.sum(eB*ez, axis=1)[:, None]
    ez *= 1./np.linalg.norm(ez, axis=1)[:, None]
    Bperp = np.cross(eB, ez, axis=1)
    Bperp *= 1./np.linalg.norm(Bperp, axis=1)[:, None]
    vperp2s = vtotal**2 - vtangs**2
    for i in range(nparticles):
        rgs[i] = compute_gc_radius(m, sqrt(vperp2s[i]), q, AbsBs[i, 0])
        xyz_inits_full[i, :] = xyz_inits[i, :] + rgs[i] * ez[i, :]
        v_inits[i, :] = -sqrt(vperp2s[i]) * Bperp[i, :] + vtangs[i] * eB[i, :]
    return xyz_inits_full, v_inits, rgs


def parallel_loop_bounds(comm, n):

    """
    Split up an array [0, 1, ..., n-1] across an mpi communicator.  Example: n
    = 8, comm with size=2 will return (0, 4) on core 0, (4, 8) on core 1,
    meaning that the array is split up as [0, 1, 2, 3] + [4, 5, 6, 7].
    """

    if comm is None:
        return 0, n
    else:
        size = comm.size
        idxs = [i*n//size for i in range(size+1)]
        assert idxs[0] == 0
        assert idxs[-1] == n
        return idxs[comm.rank], idxs[comm.rank+1]


def trace_particles(field: MagneticField, xyz_inits: NDArray[Float],
                    tangential_velocities: NDArray[Float], tmax=1e-4,
                    mass=1.6726219e-27, charge=1, Ekin=9000, tol=1e-9,
                    comm=None, phis=[], stopping_criteria=[], mode='gc_vac'):

    r"""
    Follow particles in a magnetic field.

    Args:
        field: The magnetic field.
        xyz_inits: A (nparticles, 3) array with the initial positions of the particles.
        tangential_velocities: A (nparticles, ) array containing the velocities in direction of the B field.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of a proton
        charge: charge in elementary charges, defaults to 1 (charge of a proton)
        Ekin: kinetic energy in eV, defaults to 9000eV
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

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, M)
            with M depending on the ``mode``.  Each row contains the time and
            the state.  So for `mode='gc'` and `mode='gc_vac'` the state
            consists of the xyz position and the tangential velocity, hence
            each row contains `[t, x, y, z, v_tang]`.  For `mode='full'`, the
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
    assert xyz_inits.shape[0] == len(tangential_velocities)
    vtangs = tangential_velocities
    mode = mode.lower()
    assert mode in ['gc', 'gc_vac', 'full']
    e = 1.6e-19
    Ekine = Ekin*e
    m = mass
    q = charge*e
    vtotal = sqrt(2*Ekine/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)

    if mode == 'full':
        xyz_inits, v_inits, _ = gc_to_fullorbit_initial_guesses(field, xyz_inits, vtangs, vtotal, m, q)
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        if 'gc' in mode:
            res_ty, res_phi_hit = sopp.particle_guiding_center_tracing(
                field, xyz_inits[i, :],
                m, q, vtotal, vtangs[i], tmax, tol,
                vacuum=(mode == 'gc_vac'), phis=phis, stopping_criteria=stopping_criteria)
        else:
            res_ty, res_phi_hit = sopp.particle_fullorbit_tracing(
                field, xyz_inits[i, :], v_inits[i, :],
                m, q, tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        res_tys.append(np.asarray(res_ty))
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


def trace_particles_starting_on_axis(axis, field, nparticles, tmax=1e-4,
                                     mass=1.67e-27, charge=1, Ekin=9000, tol=1e-9,
                                     comm=None, seed=1, umin=-1, umax=+1,
                                     phis=[], stopping_criteria=[], mode='gc_vac'):
    r"""
    Follows particles spawned at random locations on the magnetic axis with random pitch angle.

    Args:
        axis: The magnetic axis.
        field: The magnetic field.
        nparticles: number of particles to follow.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of a proton
        charge: charge in elementary charges, defaults to 1 (charge of a proton)
        Ekin: kinetic energy in eV, defaults to 9000eV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        seed: random seed
        umin: the tangential velocity is defined as  ``v_tang = u * vtotal``
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

    Returns: see :mod:`simsopt.field.tracing.trace_particles`
    """
    e = 1.6e-19
    Ekine = Ekin*e
    m = mass
    vtotal = sqrt(2*Ekine/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    np.random.seed(seed)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles, ))
    vtangs = us*vtotal
    xyz_inits = axis[np.random.randint(0, axis.shape[0], size=(nparticles, )), :]
    return trace_particles(
        field, xyz_inits, vtangs, tmax=tmax, mass=mass, charge=charge,
        Ekin=Ekin, tol=tol, comm=comm, phis=phis,
        stopping_criteria=stopping_criteria, mode=mode)


def compute_fieldlines(field, R0, Z0, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None):
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


def signed_distance_from_surface(xyz, surface):
    """
    Compute the signed distances from points ``xyz`` to a surface.  The sign is
    positive for points inside the volume surrounded by the surface.
    """
    gammas = surface.gamma().reshape((-1, 3))
    from scipy.spatial.distance import cdist
    dists = cdist(xyz, gammas)
    mins = np.argmin(dists, axis=1)
    n = surface.normal().reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]

    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = np.sum((xyz-gammamins) * nmins, axis=1)

    a_point_in_the_surface = np.mean(surface.gamma()[0, :, :], axis=0)
    sign_of_interiorpoint = np.sign(np.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))

    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists


class SurfaceClassifier():

    def __init__(self, surface, p=1, h=0.05):
        gammas = surface.gamma()
        r = np.linalg.norm(gammas[:, :, :2], axis=2)
        z = gammas[:, :, 2]
        rmin = max(np.min(r) - 0.1, 0.)
        rmax = np.max(r) + 0.1
        zmin = np.min(z) - 0.1
        zmax = np.max(z) + 0.1

        self.zrange = (zmin, zmax)
        self.rrange = (rmin, rmax)

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*np.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)

        def fbatch(rs, phis, zs):
            xyz = np.zeros((len(rs), 3))
            xyz[:, 0] = rs * np.cos(phis)
            xyz[:, 1] = rs * np.sin(phis)
            xyz[:, 2] = zs
            return list(signed_distance_from_surface(xyz, surface))

        rule = sopp.UniformInterpolationRule(p)
        self.dist = sopp.RegularGridInterpolant3D(
            rule, [rmin, rmax, nr], [0., 2*np.pi, nphi], [zmin, zmax, nz], 1, True)
        self.dist.interpolate_batch(fbatch)

    def evaluate(self, xyz):
        rphiz = np.zeros_like(xyz)
        rphiz[:, 0] = np.linalg.norm(xyz[:, :2], axis=1)
        rphiz[:, 1] = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2*np.pi)
        rphiz[:, 2] = xyz[:, 2]
        d = np.zeros((xyz.shape[0], 1))
        self.dist.evaluate_batch(rphiz, d)
        return d

    def to_vtk(self, filename, h=0.01):
        from pyevtk.hl import gridToVTK

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*np.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)
        rs = np.linspace(self.rrange[0], self.rrange[1], nr)
        phis = np.linspace(0, 2*np.pi, nphi)
        zs = np.linspace(self.zrange[0], self.zrange[1], nz)

        R, Phi, Z = np.meshgrid(rs, phis, zs)
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        Z = Z

        RPhiZ = np.zeros((R.size, 3))
        RPhiZ[:, 0] = R.flatten()
        RPhiZ[:, 1] = Phi.flatten()
        RPhiZ[:, 2] = Z.flatten()
        vals = np.zeros((R.size, 1))
        self.dist.evaluate_batch(RPhiZ, vals)
        vals = vals.reshape(R.shape)
        gridToVTK(filename, X, Y, Z, pointData={"levelset": vals})


class LevelsetStoppingCriterion(sopp.LevelsetStoppingCriterion):
    pass
