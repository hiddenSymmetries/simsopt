import numpy as np

from .._core.util import parallel_loop_bounds

__all__ = [
    "initialize_position_uniform_surf",
    "initialize_position_profile",
    "initialize_position_uniform_vol",
    "initialize_velocity_uniform",
]


def initialize_position_uniform_surf(
    field, nparticles, s, ntheta_max=100, nzeta_max=100, comm=None, seed=None
):
    r"""
    Initialize particle positions on a given magnetic surface, s, uniformly with respect to the volume element
    in Boozer coordinates

    Args:
        field : The :class:`BoozerMagneticField` instance
        nparticles : Number of particles to initialize
        s: Normalized toroidal flux for surface to initialize particles.
        ns_max : Number of points in s direction for Jacobian computation
        ntheta_max : Number of points in theta direction for Jacobian computation
        nzeta_max : Number of points in zeta direction for Jacobian computation
        comm : MPI communicator (default: None)
        seed: Random seed for reproducibility (default: None, uses random seed from numpy)

    Returns:
        points: A numpy array of shape (nparticles, 3) containing the initialized particle positions in Boozer
                coordinates (s, theta, zeta).
    """
    nfp = field.nfp
    np.random.seed(seed)
    # Compute max value of Jacobian on a grid for normalization
    theta_grid = np.linspace(0, 2 * np.pi, ntheta_max, endpoint=False)
    zeta_grid = np.linspace(0, 2 * np.pi / nfp, nzeta_max, endpoint=False)
    [zeta_grid, theta_grid] = np.meshgrid(zeta_grid, theta_grid)
    points = np.zeros((len(theta_grid.flatten()), 3))
    points[:, 0] = s
    points[:, 1] = theta_grid.flatten()
    points[:, 2] = zeta_grid.flatten()

    field.set_points(points)
    G = field.G()
    iota = field.iota()
    I = field.I()
    modB = field.modB()
    J = (G + iota * I) / (modB**2)

    J_max = np.max(J)

    theta_init = []
    zeta_init = []
    s_init = []
    points = np.zeros((1, 3))

    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        while True:
            rand1 = np.random.uniform(0, 1, None)
            theta = np.random.uniform(0, 2 * np.pi, None)
            zeta = np.random.uniform(0, 2 * np.pi / nfp, None)
            points[:, 0] = s
            points[:, 1] = theta
            points[:, 2] = zeta
            field.set_points(points)
            J = (field.G()[0, 0] + field.iota()[0, 0] * field.I()[0, 0]) / (
                field.modB()[0, 0] ** 2
            )

            # Normalize the Jacobian
            prob_norm = J / J_max

            if rand1 <= prob_norm:
                s_init.append(s)
                theta_init.append(theta)
                zeta_init.append(zeta)
                break

    # Gather all particle positions across all processes
    if comm is not None:
        s_init = [i for o in comm.allgather(s_init) for i in o]
        theta_init = [i for o in comm.allgather(theta_init) for i in o]
        zeta_init = [i for o in comm.allgather(zeta_init) for i in o]

    points = np.zeros((nparticles, 3))
    points[:, 0] = np.asarray(s_init)
    points[:, 1] = np.asarray(theta_init)
    points[:, 2] = np.asarray(zeta_init)

    return points


def initialize_position_profile(
    field,
    nparticles,
    profile,
    ns_max=100,
    ntheta_max=100,
    nzeta_max=100,
    comm=None,
    seed=None,
):
    r"""
    Initialize particle positions from probability density function :math:`p(s)` define by a profile function of the normalized
    flux, :math:`s`. Typically, this is used for sampling from the fusion birth distribution,

    .. math::
        p(s) = n_D(s) n_T(s) \langle \sigma v \rangle(s)

    where :math:`n_D(s)` and :math:`n_T(s)` are the deuterium and tritium density profiles, respectively, and
    :math:`\langle \sigma v \rangle(s)` is the cross section profile.

    See Bader, A., et al. "Modeling of energetic particle transport in optimized stellarators." Nuclear Fusion 61.11 (2021): 116060.

    Note that :math:`p(s)` need not be normalized: normalization is performed internally by computing the maximum value of the probabiliy on
    the grid provided by `ns_max`.

    Args:
        field : The :class:`BoozerMagneticField` instance
        nparticles : Number of particles to initialize
        profile: A function that takes a single argument (the normalized flux `s`) and returns the probability profile value at that point.
        ns_max : Number of points in s direction for Jacobian computation
        ntheta_max : Number of points in theta direction for Jacobian computation
        nzeta_max : Number of points in zeta direction for Jacobian computation
        comm : MPI communicator (default: None)
        seed: Random seed for reproducibility (default: None, uses random seed from numpy)

    Returns:
        points: A numpy array of shape (nparticles, 3) containing the initialized particle positions in Boozer
                coordinates (s, theta, zeta).
    """
    if not callable(profile):
        raise ValueError(
            "The profile must be a callable function that takes a single argument (s) and returns a value."
        )

    np.random.seed(seed)
    # Compute max value of Jacobian and p on a grid for normalization
    nfp = field.nfp
    s_grid = np.linspace(0, 1, ns_max)
    theta_grid = np.linspace(0, 2 * np.pi, ntheta_max, endpoint=False)
    zeta_grid = np.linspace(0, 2 * np.pi / nfp, nzeta_max, endpoint=False)
    [zeta_grid, theta_grid, s_grid] = np.meshgrid(zeta_grid, theta_grid, s_grid)
    points = np.zeros((len(theta_grid.flatten()), 3))
    points[:, 0] = s_grid.flatten()
    points[:, 1] = theta_grid.flatten()
    points[:, 2] = zeta_grid.flatten()

    field.set_points(points)
    G = field.G()
    iota = field.iota()
    I = field.I()
    modB = field.modB()
    J = (G + iota * I) / (modB**2)

    # Compute normalized profile values on the grid
    profile_values = np.array([profile(s) for s in s_grid.flatten()])
    prob_max = np.max(
        J[:, 0] * profile_values
    )  # Normalize by the maximum value of J * profile

    theta_init = []
    zeta_init = []
    s_init = []
    points = np.zeros((1, 3))

    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        while True:
            rand1 = np.random.uniform(0, 1, None)
            s = np.random.uniform(0, 1.0, None)
            theta = np.random.uniform(0, 2 * np.pi, None)
            zeta = np.random.uniform(0, 2 * np.pi / nfp, None)
            points[:, 0] = s
            points[:, 1] = theta
            points[:, 2] = zeta
            field.set_points(points)
            J = (field.G()[0, 0] + field.iota()[0, 0] * field.I()[0, 0]) / (
                field.modB()[0, 0] ** 2
            )

            # Normalize the probability
            prob_norm = J * profile(s) / prob_max

            if rand1 <= prob_norm:
                s_init.append(s)
                theta_init.append(theta)
                zeta_init.append(zeta)
                break

    # Gather all particle positions across all processes
    if comm is not None:
        s_init = [i for o in comm.allgather(s_init) for i in o]
        theta_init = [i for o in comm.allgather(theta_init) for i in o]
        zeta_init = [i for o in comm.allgather(zeta_init) for i in o]

    points = np.zeros((nparticles, 3))
    points[:, 0] = np.asarray(s_init)
    points[:, 1] = np.asarray(theta_init)
    points[:, 2] = np.asarray(zeta_init)

    return points


def initialize_position_uniform_vol(
    field,
    nparticles,
    ns_max=100,
    ntheta_max=100,
    nzeta_max=100,
    comm=None,
    seed=None,
):
    r"""
    Initialize particle positions uniformly in the plasma volume with respect to the volume element in
    Boozer coordinates.

    Args:
        field : The :class:`BoozerMagneticField` instance
        nparticles : Number of particles to initialize
        ns_max : Number of points in s direction for Jacobian computation
        ntheta_max : Number of points in theta direction for Jacobian computation
        nzeta_max : Number of points in zeta direction for Jacobian computation
        comm : MPI communicator (default: None)
        seed: Random seed for reproducibility (default: None, uses random seed from numpy)

    Returns:
        points: A numpy array of shape (nparticles, 3) containing the initialized particle positions in Boozer
                coordinates (s, theta, zeta).
    """
    return initialize_position_profile(
        field,
        nparticles,
        lambda s: 1.0,
        ns_max=ns_max,
        ntheta_max=ntheta_max,
        nzeta_max=nzeta_max,
        comm=comm,
        seed=seed,
    )


def initialize_velocity_uniform(vpar0, nParticles, comm=None, seed=None):
    r"""
    Initialize parallel velocities uniformly distributed in the range [-vpar0, vpar0].
    Args:
        vpar0: Maximum parallel velocity magnitude.
        nParticles: Number of particles to initialize.
        comm: MPI communicator (default: None).
        seed: Random seed for reproducibility (default: None, uses random seed from numpy).

    Returns:
        vpar_init: A numpy array of shape (nParticles,) containing the initialized parallel velocities.
    """

    if comm is not None:
        verbose = comm.rank == 0
    else:
        verbose = True
    if seed is not None:
        np.random.seed(seed)

    # Initialize uniformly distributed parallel velocities
    if verbose:
        vpar_init = np.random.uniform(-vpar0, vpar0, (nParticles,))
    else:
        vpar_init = None
    if comm is not None:
        vpar_init = comm.bcast(vpar_init, root=0)

    return vpar_init
