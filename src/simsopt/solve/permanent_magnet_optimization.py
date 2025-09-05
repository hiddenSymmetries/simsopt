import warnings

import numpy as np
from .macromag import MacroMag, Tiles

import simsoptpp as sopp
from .._core.types import RealArray


__all__ = ['relax_and_split', 'GPMO']
M_rem_value = 1.465 / (4.0 * np.pi * 1e-7)  # aprox. 1.1659e6 A/m


def prox_l0(m: RealArray,
            mmax: RealArray,
            reg_l0: float,
            nu: float):
    r"""
    Proximal operator for L0 regularization.

    Note that the m values are normalized before hard
    thresholding to avoid only truncating the magnets on the
    inner side of the configuration (which are often important!).
    Note also the in principle the hard threshold here should be:
    :math:`hard\_threshold = \sqrt{2 * \text{reg_l0} * \nu}`. But we can always imagine
    rescaling :math:`\nu` (or reg_l0) such that :math:`hard\_threshold = 2 * \text{reg_l0} * \nu`
    instead (for instance :math:`\nu -> 2 * \nu^2` and :math:`\text{reg_l0} -> \text{reg_l0}^ 2)`.

    Args:
        m: The permanent magnet dipole vectors with shape (ndipoles, 3).
        mmax: The maximal dipole strengths of each of the permanent magnets of shape (ndipoles,)
        reg_l0: The amount of L0 regularization used in the problem.
        nu: The strength of the "relaxing" term in the relax-and-split algorithm
            used for permanent magnet optimization.
    """
    ndipoles = len(m) // 3
    mmax_vec = np.array([mmax, mmax, mmax]).T
    m_normalized = (np.abs(m).reshape(ndipoles, 3) / mmax_vec).reshape(ndipoles * 3)
    # in principle, hard threshold should be sqrt(2 * reg_l0 * nu) but can always renormalize everything
    return m * (m_normalized > 2 * reg_l0 * nu)


def prox_l1(m, mmax, reg_l1, nu):
    """
    Proximal operator for L1 regularization.

    Note that the m values are normalized before soft
    thresholding to avoid only truncating the magnets on the
    inner side of the configuration (which are often important!).

    Args:
        m: 2D numpy array, shape (ndipoles, 3)
            The permanent magnet dipole vectors.
        mmax: 1D numpy array, shape (ndipoles)
            The maximal dipole strengths of each of the permanent magnets.
        reg_l1: double
            The amount of L1 regularization used in the problem.
        nu: double
            The strength of the "relaxing" term in the relax-and-split algorithm
            used for permanent magnet optimization.
    """
    ndipoles = len(m) // 3
    mmax_vec = np.array([mmax, mmax, mmax]).T
    m_normalized = (np.abs(m).reshape(ndipoles, 3) / mmax_vec).reshape(ndipoles * 3)
    return np.sign(m) * np.maximum(np.abs(m_normalized) - reg_l1 * nu, 0) * np.ravel(mmax_vec)


def projection_L2_balls(x, mmax):
    """
    Project the vector x onto a series of L2 balls in R3.
    Only used here for checking if the initial guess for the
    permanent magnets is inside the feasible region
    before optimization begins.

    Args:
        x: 1D numpy array, shape (ndipoles * 3)
            The current solution vector for the dipole vectors of the magnets.
        mmax: 1D numpy array, shape (ndipoles)
            The maximal dipole strength of each of the magnets.
    """
    N = len(x) // 3
    x_shaped = x.reshape(N, 3)
    denom_fac = np.sqrt(np.sum(x_shaped ** 2, axis=-1)) / mmax
    denom = np.maximum(np.ones(len(denom_fac)), denom_fac)
    return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)


def setup_initial_condition(pm_opt, m0=None):
    """
    If an initial guess for the dipole moments is specified,
    checks the initial condition lies in the allowed hypersurface.
    If an initial guess is not specified, defaults to initializing
    the permanent magnet dipole moments to all zeros.

    Args:
        pm_opt: PermanentMagnetGrid class object
            Permanent magnet grid for optimization.
        m0: 1D numpy array, shape (ndipoles * 3)
            Initial guess for the dipole vectors.
    """
    # Initialize initial guess for the dipole strengths
    if m0 is not None:
        if len(m0) != pm_opt.ndipoles * 3:
            raise ValueError(
                'Initial dipole guess is incorrect shape --'
                ' guess must be 1D with shape (ndipoles * 3).'
            )
        m0_temp = projection_L2_balls(m0, pm_opt.m_maxima)
        # check if m0 lies inside the hypersurface spanned by
        # L2 balls, which we require it does
        if not np.allclose(m0, m0_temp):
            raise ValueError(
                'Initial dipole guess must contain values '
                'that are satisfy the maximum bound constraints.'
            )
        pm_opt.m0 = m0


def relax_and_split(pm_opt, m0=None, **kwargs):
    """
    Uses a relax-and-split algorithm for solving the permanent
    magnet optimization problem, which solves a convex and nonconvex
    part separately.

    Defaults to the MwPGP convex step and no
    nonconvex step.  If a nonconvexity is specified, the associated
    prox function must be defined in this file.  Relax-and-split
    allows for speedy algorithms for both steps and the imposition of
    convex equality and inequality constraints (including the required
    constraint on the strengths of the dipole moments).

    Args:
        pm_opt: The grid of permanent magnets to optimize.
        m0: Initial guess for the permanent magnet dipole moments. Defaults
            to a starting guess of all zeros. This vector must lie in the
            hypersurface spanned by the L2 ball constraints. Note that if
            algorithm is being used properly, the end result should be
            independent of the choice of initial condition.
        kwargs: Keyword arguments to pass to the algorithm. The following
            arguments can be passed to the MwPGP algorithm:

            epsilon:
                Error tolerance for the convex part of the algorithm (MwPGP).
            nu:
                Hyperparameter used for the relax-and-split
                least-squares. Set nu >> 1 to reduce the
                importance of nonconvexity in the problem.
            reg_l0:
                Regularization value for the L0 nonconvex term in the
                optimization. This value is automatically scaled based on
                the max dipole moment values, so that reg_l0 = 1 corresponds
                to reg_l0 = np.max(m_maxima). It follows that users should
                choose reg_l0 in [0, 1].
            reg_l1:
                Regularization value for the L1 nonsmooth term in the
                optimization,
            reg_l2:
                Regularization value for any convex regularizers in the
                optimization problem, such as the often-used L2 norm.
            max_iter_MwPGP:
                Maximum iterations to perform during a run of the convex
                part of the relax-and-split algorithm (MwPGP).
            max_iter_RS:
                Maximum iterations to perform of the overall relax-and-split
                algorithm. Therefore, also the number of times that MwPGP is
                called, and the number of times a prox is computed.
            verbose:
                Prints out all the loss term errors separately.

    Returns:
        A tuple of optimization loss, solution at each step, and sparse solution.

        The tuple contains

        errors:
            Total optimization loss after each convex sub-problem is solved.
        m_history:
            Solution for the permanent magnets after each convex
            sub-problem is solved.
        m_proxy_history:
            Sparse solution for the permanent magnets after each convex
            sub-problem is solved.

    """
    # change to row-major order for the C++ code
    # A_obj = np.ascontiguousarray(pm_opt.A_obj)
    ATb = np.ascontiguousarray(np.reshape(pm_opt.ATb, (pm_opt.ndipoles, 3)))

    # print initial errors and values before optimization
    pm_opt._print_initial_opt()

    # Begin the various algorithms
    errors = []
    m_history = []
    m_proxy_history = []

    # get optimal alpha value for the MwPGP algorithm
    alpha_max = 2.0 / pm_opt.ATA_scale
    alpha_max = alpha_max * (1 - 1e-5)
    convex_step = sopp.MwPGP_algorithm

    # set the nonconvex step in the algorithm
    reg_rs = 0.0
    nu = kwargs.get("nu", 1e100)
    reg_l0 = kwargs.get("reg_l0", 0.0)
    reg_l1 = kwargs.get("reg_l1", 0.0)

    max_iter_RS = kwargs.pop('max_iter_RS', 1)
    epsilon_RS = kwargs.pop('epsilon_RS', 1e-3)

    if (not np.isclose(reg_l0, 0.0, atol=1e-16)) and (not np.isclose(reg_l1, 0.0, atol=1e-16)):
        raise ValueError(' L0 and L1 loss terms cannot be used concurrently.')
    elif not np.isclose(reg_l0, 0.0, atol=1e-16):
        prox = prox_l0
        reg_rs = reg_l0
    elif not np.isclose(reg_l1, 0.0, atol=1e-16):
        prox = prox_l1
        reg_rs = reg_l1

    # Auxiliary variable in relax-and-split can be initialized
    # to prox(m0), where m0 is the initial guess for m.
    if m0 is not None:
        setup_initial_condition(pm_opt, m0)
    m0 = pm_opt.m0
    m_proxy = pm_opt.m0
    mmax = pm_opt.m_maxima
    if reg_rs > 0.0:
        m_proxy = prox(m_proxy, mmax, reg_rs, nu)
    kwargs['alpha'] = alpha_max

    # Begin optimization
    if reg_rs > 0.0:
        # Relax-and-split algorithm
        m = pm_opt.m0
        for i in range(max_iter_RS):
            # update m with the CONVEX part of the algorithm
            algorithm_history, _, _, m = convex_step(
                A_obj=pm_opt.A_obj,
                b_obj=pm_opt.b_obj,
                ATb=ATb,
                m_proxy=np.ascontiguousarray(m_proxy.reshape(pm_opt.ndipoles, 3)),
                m0=np.ascontiguousarray(m.reshape(pm_opt.ndipoles, 3)),  # note updated m is new guess
                m_maxima=mmax,
                **kwargs
            )
            m_history.append(m)
            m = np.ravel(m)
            algorithm_history = algorithm_history[algorithm_history != 0]
            errors.append(algorithm_history[-1])

            # Solve the nonconvex optimization -- i.e. take a prox
            m_proxy = prox(m, mmax, reg_rs, nu)
            m_proxy_history.append(m_proxy)
            if np.linalg.norm(m - m_proxy) < epsilon_RS:
                print('Relax-and-split finished early, at iteration ', i)
                break
    else:
        m0 = np.ascontiguousarray(m0.reshape(pm_opt.ndipoles, 3))
        # no nonconvex terms being used, so just need one round of the
        # convex algorithm called MwPGP
        algorithm_history, _, m_history, m = convex_step(
            A_obj=pm_opt.A_obj,
            b_obj=pm_opt.b_obj,
            ATb=ATb,
            m_proxy=m0,
            m0=m0,
            m_maxima=mmax,
            **kwargs
        )
        m = np.ravel(m)
        m_proxy = m

    # note m = m_proxy if not using relax-and-split (i.e. problem is convex)
    pm_opt.m = m
    pm_opt.m_proxy = m_proxy
    return errors, m_history, m_proxy_history


def GPMO(pm_opt, algorithm='baseline', **kwargs):
    r"""
    GPMO is a greedy algorithm for the permanent magnet optimization problem.

    GPMO is an alternative to to the relax-and-split algorithm.
    Full-strength magnets are placed one-by-one according to minimize the
    MSE (fB). Allows for a number of keyword arguments that facilitate some
    basic backtracking (error correction) and placing magnets together so no
    isolated magnets occur.

    Args:
        pm_opt: The grid of permanent magnets to optimize.
            PermanentMagnetGrid instance
        algorithm:
            The type of greedy algorithm to use. Options are

            baseline:
                the simple implementation of GPMO,
            multi:
                GPMO, but placing multiple magnets per iteration,
            backtracking:
                backtrack every few hundred iterations to improve the
                solution,
            ArbVec:
                the simple implementation of GPMO, but with arbitrary
                oriented polarization vectors,
            ArbVec_backtracking:
                same as above but w/ backtracking.

            Easiest algorithm to use is 'baseline' but most effective 
            algorithm is 'ArbVec_backtracking'.
        kwargs:
            Keyword arguments for the GPMO algorithm and its variants.
            The following variables can be passed:

            K: integer
                Maximum number of GPMO iterations to run.
            nhistory: integer
                Every 'nhistory' iterations, the loss terms are recorded.
            Nadjacent: integer
                Number of neighbor cells to consider 'adjacent' to one
                another, for the purposes of placing multiple magnets or
                doing backtracking. Not to be used with 'baseline' and 'ArbVec'.
            dipole_grid_xyz: 2D numpy array, shape (ndipoles, 3).
                XYZ coordinates of the permanent magnet locations. Needed for
                figuring out which permanent magnets are adjacent to one another.
                Not a keyword argument for 'baseline' and 'ArbVec'.
            max_nMagnets: integer.
                Maximum number of magnets to place before algorithm quits. Only
                a keyword argument for 'backtracking' and 'ArbVec_backtracking'
                since without any backtracking, this is the same parameter as 'K'.
            backtracking: integer.
                Every 'backtracking' iterations, a backtracking is performed to
                remove suboptimal magnets. Only a keyword argument for
                'backtracking' and 'ArbVec_backtracking' algorithms.
            thresh_angle: float.
                If the angle between adjacent dipole moments > thresh_angle,
                these dipoles are considered suboptimal and liable to removed
                during a backtracking step. Only a keyword argument for
                'ArbVec_backtracking' algorithm.
            single_direction: int, must be = 0, 1, or 2.
                Specify to only place magnets with orientations in a single
                direction, e.g. only magnets pointed in the +- x direction.
                Keyword argument only for 'baseline', 'multi', and 'backtracking'
                since the 'ArbVec...' algorithms have local coordinate systems
                and therefore can specify the same constraint and much more
                via the 'pol_vectors' argument.
            reg_l2: float.
                L2 regularization value, applied through the mmax argument in
                the GPMO algorithm. See the paper for how this works.
            verbose: bool.
                If True, print out the algorithm progress every 'nhistory'
                iterations. Also needed to record the algorithm history.

    Returns:
        Tuple of (errors, Bn_errors, m_history)

        errors:
            Total optimization loss values, recorded every 'nhistory'
            iterations.
        Bn_errors:
            :math:`|Bn|` errors, recorded every 'nhistory' iterations.
        m_history:
            Solution for the permanent magnets, recorded after 'nhistory'
            iterations.

    """
    if not hasattr(pm_opt, "A_obj"):
        raise ValueError("The PermanentMagnetClass needs to use geo_setup() or "
                         "geo_setup_from_famus() before calling optimization routines.")

    # Begin the various algorithms
    errors = []
    m_history = []

    # Need to normalize the m, so have
    # || A * m - b||^2 = || ( A * mmax) * m / mmax - b||^2
    mmax = pm_opt.m_maxima
    contig = np.ascontiguousarray
    mmax_vec = contig(np.array([mmax, mmax, mmax]).T.reshape(pm_opt.ndipoles * 3))
    A_obj = pm_opt.A_obj * mmax_vec

    if (algorithm != 'baseline' and algorithm != 'mutual_coherence' and algorithm != 'ArbVec') and 'dipole_grid_xyz' not in kwargs:
        raise ValueError('GPMO variants require dipole_grid_xyz to be defined.')

    # Set the L2 regularization if it is included in the kwargs
    reg_l2 = kwargs.pop("reg_l2", 0.0)

    # check that algorithm can generate K binary dipoles if no backtracking done
    if "K" in kwargs:
        if (algorithm not in ['backtracking', 'ArbVec_backtracking']) and kwargs["K"] > pm_opt.ndipoles:
            warnings.warn(
                'Parameter K to GPMO algorithm is greater than the total number of dipole locations '
                ' so the algorithm will set K = the total number and proceed.')
            kwargs["K"] = pm_opt.ndipoles
        print('Number of binary dipoles to use in GPMO algorithm = ', kwargs["K"])

    if "nhistory" in kwargs and "K" in kwargs:
        if kwargs['nhistory'] > kwargs['K']:
            raise ValueError('nhistory must be less than K for the GPMO algorithm.')

    Nnorms = contig(np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1))))

    # Note, only baseline method has the f_m loss term implemented!
    if algorithm == 'baseline':  # GPMO
        algorithm_history, Bn_history, m_history, m = sopp.GPMO_baseline(
            A_obj=contig(A_obj.T),
            b_obj=contig(pm_opt.b_obj),
            mmax=np.sqrt(reg_l2)*mmax_vec,
            normal_norms=Nnorms,
            **kwargs
        )
    elif algorithm == 'ArbVec':  # GPMO with arbitrary polarization vectors
        algorithm_history, Bn_history, m_history, m = sopp.GPMO_ArbVec(
            A_obj=contig(A_obj.T),
            b_obj=contig(pm_opt.b_obj),
            mmax=np.sqrt(reg_l2)*mmax_vec,
            normal_norms=Nnorms,
            pol_vectors=contig(pm_opt.pol_vectors),
            **kwargs
        )
    elif algorithm == 'backtracking':  # GPMOb
        algorithm_history, Bn_history, m_history, num_nonzeros, m = sopp.GPMO_backtracking(
            A_obj=contig(A_obj.T),
            b_obj=contig(pm_opt.b_obj),
            mmax=np.sqrt(reg_l2)*mmax_vec,
            normal_norms=Nnorms,
            **kwargs
        )
        pm_opt.num_nonzeros = num_nonzeros[num_nonzeros != 0]
    elif algorithm == 'ArbVec_backtracking':  # GPMOb with arbitrary vectors
        if pm_opt.coordinate_flag != 'cartesian':
            raise ValueError('ArbVec_backtracking algorithm currently '
                             'only supports dipole grids with \n'
                             'moment vectors in the Cartesian basis.')
        nGridPoints = int(A_obj.shape[1]/3)
        if "m_init" in kwargs.keys():
            if kwargs["m_init"].shape[0] != nGridPoints:
                raise ValueError('Initialization vector `m_init` must have '
                                 'as many rows as there are dipoles in the '
                                 'grid')
            elif kwargs["m_init"].shape[1] != 3:
                raise ValueError('Initialization vector `m_init` must have '
                                 'three columns')
            kwargs["x_init"] = contig(kwargs["m_init"]
                                      / (mmax_vec.reshape(pm_opt.ndipoles, 3)))
            kwargs.pop("m_init")
        else:
            kwargs["x_init"] = contig(np.zeros((nGridPoints, 3)))
        algorithm_history, Bn_history, m_history, num_nonzeros, m = sopp.GPMO_ArbVec_backtracking(
            A_obj=contig(A_obj.T),
            b_obj=contig(pm_opt.b_obj),
            mmax=np.sqrt(reg_l2)*mmax_vec,
            normal_norms=Nnorms,
            pol_vectors=contig(pm_opt.pol_vectors),
            **kwargs
        )
        

    elif algorithm == 'ArbVec_backtracking_py':  # Pure-Python mirror of GPMOb (ArbVec)
        if pm_opt.coordinate_flag != 'cartesian':
            raise ValueError('ArbVec_backtracking algorithm currently '
                             'only supports dipole grids with \n'
                             'moment vectors in the Cartesian basis.')
        nGridPoints = int(A_obj.shape[1]/3)
        if "m_init" in kwargs.keys():
            if kwargs["m_init"].shape[0] != nGridPoints:
                raise ValueError('Initialization vector `m_init` must have '
                                 'as many rows as there are dipoles in the '
                                 'grid')
            elif kwargs["m_init"].shape[1] != 3:
                raise ValueError('Initialization vector `m_init` must have '
                                 'three columns')
            kwargs["x_init"] = contig(kwargs["m_init"]
                                      / (mmax_vec.reshape(pm_opt.ndipoles, 3)))
            kwargs.pop("m_init")
        else:
            kwargs["x_init"] = contig(np.zeros((nGridPoints, 3)))
        algorithm_history, Bn_history, m_history, num_nonzeros, m = GPMO_ArbVec_backtracking_py(
            A_obj=contig(A_obj.T),                 # Python port expects (3N, ngrid)
            b_obj=contig(pm_opt.b_obj),
            mmax=np.sqrt(reg_l2) * mmax_vec,
            normal_norms=Nnorms,
            pol_vectors=contig(pm_opt.pol_vectors),
            **kwargs
        )

    elif algorithm == 'ArbVec_backtracking_macromag_py':  # MacroMag-driven ArbVec backtracking
        if pm_opt.coordinate_flag != 'cartesian':
            raise ValueError('ArbVec_backtracking algorithm currently '
                             'only supports dipole grids with \n'
                             'moment vectors in the Cartesian basis.')
        nGridPoints = int(A_obj.shape[1]/3)
        if "m_init" in kwargs.keys():
            if kwargs["m_init"].shape[0] != nGridPoints:
                raise ValueError('Initialization vector `m_init` must have '
                                 'as many rows as there are dipoles in the '
                                 'grid')
            elif kwargs["m_init"].shape[1] != 3:
                raise ValueError('Initialization vector `m_init` must have '
                                 'three columns')
            kwargs["x_init"] = contig(kwargs["m_init"]
                                      / (mmax_vec.reshape(pm_opt.ndipoles, 3)))
            kwargs.pop("m_init")
        else:
            kwargs["x_init"] = contig(np.zeros((nGridPoints, 3)))
            
        cube_dim = kwargs.pop('cube_dim', 0.004)
        mu_ea = kwargs.pop('mu_ea', 1.0)
        mu_oa = kwargs.pop('mu_oa', 1.0)
        
        use_coils = kwargs.pop('use_coils', False)
        coil_path = kwargs.pop('coil_path', None)
            
        algorithm_history, Bn_history, m_history, num_nonzeros, m = GPMO_ArbVec_backtracking_macromag_py(
            A_obj=contig(A_obj.T),                 # MacroMag path also expects (3N, ngrid)
            b_obj=contig(pm_opt.b_obj),
            mmax=mmax_vec,
            normal_norms=Nnorms,
            pol_vectors=contig(pm_opt.pol_vectors),
            cube_dim=cube_dim, 
            mu_ea=mu_ea, 
            mu_oa=mu_oa,
            use_coils=use_coils, 
            coil_path=coil_path, 
            **kwargs
        )
        
    elif algorithm == 'multi':  # GPMOm
        algorithm_history, Bn_history, m_history, m = sopp.GPMO_multi(
            A_obj=contig(A_obj.T),
            b_obj=contig(pm_opt.b_obj),
            mmax=np.sqrt(reg_l2)*mmax_vec,
            normal_norms=Nnorms,
            **kwargs
        )
    else:
        raise NotImplementedError('Requested algorithm variant is incorrect or not yet implemented')

    # rescale m and m_history
    m = m * (mmax_vec.reshape(pm_opt.ndipoles, 3))
    print('Number of binary dipoles returned by GPMO algorithm = ',
          np.count_nonzero(np.sum(m, axis=-1)))

    # rescale the m that have been saved every Nhistory iterations
    for i in range(m_history.shape[-1]):
        m_history[:, :, i] = m_history[:, :, i] * (mmax_vec.reshape(pm_opt.ndipoles, 3))
    errors = algorithm_history[algorithm_history != 0]
    Bn_errors = Bn_history[Bn_history != 0]

    # note m = m_proxy for GPMO because this is not using relax-and-split
    pm_opt.m = np.ravel(m)
    pm_opt.m_proxy = pm_opt.m
    return errors, Bn_errors, m_history


def _connectivity_matrix_py(dipole_grid_xyz: np.ndarray, Nadjacent: int) -> np.ndarray:
    """
    Python port of C++ connectivity_matrix:
    For each dipole j, return indices of the closest 'Nadjacent' dipoles
    (including itself). C++ filled 2000; here we fill exactly what's needed.
    """
    xyz = np.asarray(dipole_grid_xyz, dtype=np.float64, order="C")
    # pairwise squared distances (broadcasted)
    # dist^2 = ||x_i - x_j||^2 = |x|^2 + |y|^2 - 2 x_i·x_j
    norms = (xyz**2).sum(axis=1)
    d2 = norms[:, None] + norms[None, :] - 2.0 * (xyz @ xyz.T)
    # ensure numerical stability & self is exactly 0
    np.fill_diagonal(d2, 0.0)
    # argsort along rows -> nearest first
    idx_sorted = np.argsort(d2, axis=1)
    # take the first Nadjacent indices (includes self)
    Nadj = min(Nadjacent, idx_sorted.shape[1])
    return idx_sorted[:, :Nadj].astype(np.int32, copy=False)


def _print_GPMO_py(k, ngrid, print_iter_ref, x, Aij_mj_vec, objective_history, Bn_history, m_history, mmax_sum, normal_norms):
    """
    Python port of print_GPMO (records histories + prints status).
    """
    # R2 = 0.5 * sum (Aij_mj)^2
    R2 = 0.5 * np.dot(Aij_mj_vec, Aij_mj_vec)
    # sqrtR2 = sum |Aij_mj| * sqrt(normal_norms)
    sqrtR2 = np.sum(np.abs(Aij_mj_vec) * np.sqrt(normal_norms))
    # Record
    objective_history[print_iter_ref[0]] = R2
    Bn_history[print_iter_ref[0]] = sqrtR2 / np.sqrt(float(ngrid))
    m_history[:, :, print_iter_ref[0]] = x
    # Print (follow C++ format)
    print(f"{k} ... {R2:.2e} ... {mmax_sum:.2e} ")
    print_iter_ref[0] += 1


def _initialize_GPMO_ArbVec_py(x_init, pol_vectors, x, x_vec, x_sign,
                               A_obj, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref):
    """
    Python port of initialize_GPMO_ArbVec (maps x_init to nearest allowable vectors,
    updates residual, masks, and R2s).
    """
    # Shapes and views
    N = x.shape[0]
    nPolVecs = pol_vectors.shape[1]
    NNp = N * nPolVecs
    ngrid = A_obj.shape[2]  # using A_obj reshaped as (N, 3, ngrid)

    n_initialized = 0
    n_OutOfTol = 0
    tol = 4 * np.finfo(np.float32).eps

    for j in range(N):
        xin = x_init[j]
        # Skip if zero init
        if xin[0] == 0 and xin[1] == 0 and xin[2] == 0:
            continue

        n_initialized += 1
        # Find closest allowed direction (±pol_vec) OR 0
        min_sq = np.inf
        m_min = 0
        sign_min = 0
        # Compare to all pol vectors
        pv = pol_vectors[j]  # (nPolVecs, 3)
        # squared diffs to +pv and -pv
        diff_pos = ((xin[None, :] - pv)**2).sum(axis=1)
        diff_neg = ((xin[None, :] + pv)**2).sum(axis=1)
        # best among positives
        idx_pos = int(np.argmin(diff_pos))
        if diff_pos[idx_pos] < min_sq:
            min_sq = float(diff_pos[idx_pos]); m_min = idx_pos; sign_min = +1
        # best among negatives
        idx_neg = int(np.argmin(diff_neg))
        if diff_neg[idx_neg] < min_sq:
            min_sq = float(diff_neg[idx_neg]); m_min = idx_neg; sign_min = -1
        # compare to null
        diff_null = float((xin**2).sum())
        if diff_null < min_sq:
            min_sq = diff_null; m_min = 0; sign_min = 0

        if sign_min != 0:
            # place it
            x[j, :] = sign_min * pol_vectors[j, m_min, :]
            x_vec[j] = m_min
            x_sign[j] = sign_min
            Gamma_complement[j] = False
            num_nonzero_ref[0] += 1

            # Update running residual: Aij_mj_sum += sign * sum_l pol_vec[l] * A_obj[j,l,:]
            # A_obj reshaped as (N,3,ngrid)
            Aij_mj_sum += sign_min * (pol_vectors[j, m_min, :, None] * A_obj[j]).sum(axis=0)
            # Mask out R2s of this site (unavailable)
            R2s[j*nPolVecs:(j+1)*nPolVecs] = 1e50
            R2s[NNp + j*nPolVecs:NNp + (j+1)*nPolVecs] = 1e50

        # outof tolerance check (vs assigned x[j])
        if np.any(np.abs(xin - x[j]) > tol):
            n_OutOfTol += 1

    if n_OutOfTol != 0:
        print(
            "    WARNING: {} of {} dipoles in the initialization vector disagree \n"
            "    with the allowable dipole moments to single floating precision. These \n"
            "    dipoles will be initialized to the nearest allowable moments or zero."
            .format(n_OutOfTol, n_initialized)
        )

def GPMO_ArbVec_backtracking_py(
    A_obj: np.ndarray,
    b_obj: np.ndarray,
    mmax: np.ndarray,
    normal_norms: np.ndarray,
    pol_vectors: np.ndarray,
    K: int,
    verbose: bool,
    nhistory: int,
    backtracking: int,
    dipole_grid_xyz: np.ndarray,
    Nadjacent: int,
    thresh_angle: float,
    max_nMagnets: int,
    x_init: np.ndarray):
    """
    Python implementation of the C++ version of the following
    std::tuple<Array, Array, Array, Array, Array> GPMO_ArbVec_backtracking(...)
    Returns:
      (objective_history, Bn_history, m_history, num_nonzeros, x)
    """
    # Shapes
    # A_obj was passed as (3N, ngrid) in Python caller; reshape to (N,3,ngrid)
    A_obj = np.asarray(A_obj, dtype=np.float64, order="C")
    b_obj = np.asarray(b_obj, dtype=np.float64, order="C")
    mmax  = np.asarray(mmax,  dtype=np.float64, order="C")
    normal_norms = np.asarray(normal_norms, dtype=np.float64, order="C")
    pol_vectors  = np.asarray(pol_vectors, dtype=np.float64, order="C")
    x_init = np.asarray(x_init, dtype=np.float64, order="C")
    dipole_grid_xyz = np.asarray(dipole_grid_xyz, dtype=np.float64, order="C")

    ngrid = A_obj.shape[1]
    N = int(A_obj.shape[0] // 3)
    assert 3 * N == A_obj.shape[0], "A_obj first dim must be 3*N"
    A_obj = A_obj.reshape(N, 3, ngrid)

    nPolVecs = pol_vectors.shape[1]
    NNp = N * nPolVecs

    # Buffers (match C++)
    print_iter_ref = [0]  # mutable int
    x = np.zeros((N, 3), dtype=np.float64)
    x_vec  = np.zeros(N, dtype=np.int32)     # chosen pol index per site
    x_sign = np.zeros(N, dtype=np.int8)      # chosen sign per site

    m_history = np.zeros((N, 3, nhistory + 2), dtype=np.float64)
    objective_history = np.zeros((nhistory + 2,), dtype=np.float64)
    Bn_history = np.zeros((nhistory + 2,), dtype=np.float64)

    if verbose:
        print("Running the GPMO backtracking algorithm with arbitrary polarization vectors")
        print("Iteration ... |Am - b|^2 ... lam*|m|^2")

    Gamma_complement = np.ones((N,), dtype=bool)
    R2s = np.full((2 * NNp,), 1e50, dtype=np.float64)

    # Running residual Aij_mj_sum = -b_obj (shape ngrid)
    Aij_mj_sum = -b_obj.astype(np.float64, copy=True)

    # Precompute adjacency
    Connect = _connectivity_matrix_py(dipole_grid_xyz, Nadjacent)

    num_nonzero_ref = [0]
    num_nonzeros = np.zeros((nhistory + 2,), dtype=np.int32)

    # Initialize from x_init
    _initialize_GPMO_ArbVec_py(
        x_init, pol_vectors, x, x_vec, x_sign,
        A_obj, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref
    )
    num_nonzeros[0] = num_nonzero_ref[0]

    # Record the initialization snapshot
    _print_GPMO_py(0, ngrid, print_iter_ref, x, Aij_mj_sum, objective_history, Bn_history, m_history, 0.0, normal_norms)

    # backtracking threshold angle
    cos_thresh_angle = np.cos(thresh_angle)

    # Main iterations
    for k in range(K):

        # For each available site j, evaluate all allowable pol vectors (±)
        # Fill R2s[mj] and R2s[mj + NNp] like C++.
        # Here we compute R2 and R2minus vectorized over grid for each (j,m).
        for j in range(N):
            if Gamma_complement[j]:
                Aj = A_obj[j]  # (3, ngrid)
                # bnorm(j,m,:) = sum_l pol_vectors[j,m,l] * A[j,l,:]
                # shape: (nPolVecs, ngrid)
                bnorm = (pol_vectors[j, :, :, None] * Aj[None, :, :]).sum(axis=1)
                # R2 terms
                # (Aij_mj_sum +/- bnorm)
                tmp_plus  = Aij_mj_sum[None, :] + bnorm
                tmp_minus = Aij_mj_sum[None, :] - bnorm
                R2 = (tmp_plus**2).sum(axis=1)      # (nPolVecs,)
                R2minus = (tmp_minus**2).sum(axis=1)
                # Add mmax contribution as in cpp version of this
                # NOTE: C++ uses mmax_ptr[j] even though Python passes len=3N
                R2 += mmax[j] * mmax[j]
                R2minus += mmax[j] * mmax[j]
                base = j * nPolVecs
                R2s[base:base + nPolVecs] = R2
                R2s[NNp + base:NNp + base + nPolVecs] = R2minus

        # Choose best (j, m, sign)
        skj = int(np.argmin(R2s))
        if skj >= NNp:
            skj -= NNp
            sign_fac = -1.0
        else:
            sign_fac = +1.0
        skjj = int(skj % nPolVecs)
        skj  = int(skj // nPolVecs)

        # Commit selection
        pv = pol_vectors[skj, skjj, :]            # (3,)
        x[skj, :] = sign_fac * pv
        x_vec[skj] = skjj
        x_sign[skj] = int(sign_fac)
        Gamma_complement[skj] = False
        # Update residual: Aij_mj_sum += sign * sum_l pv[l] * A_obj[skj,l,:]
        Aij_mj_sum += sign_fac * (pv[:, None] * A_obj[skj]).sum(axis=0)
        # Mask out site skj in R2s
        R2s[skj*nPolVecs:(skj+1)*nPolVecs] = 1e50
        R2s[NNp + skj*nPolVecs:NNp + (skj+1)*nPolVecs] = 1e50

        num_nonzero_ref[0] += 1

        # Backtracking step
        if backtracking != 0: 
            if (k % backtracking) == 0:
                wyrm_sum = 0
                # Loop over all dipoles
                for j in range(N):
                    if Gamma_complement[j]:
                        continue  # skip if not placed
                    m = x_vec[j]
                    # Scan adjacent dipoles; find minimum cosine (largest angle)
                    min_cos_angle = 2.0   # > max possible 1
                    cj_min = -1
                    for jj in range(Connect.shape[1]):
                        cj = int(Connect[j, jj])
                        if Gamma_complement[cj]:
                            continue  # not placed
                        cos_angle = float(np.dot(x[j, :], x[cj, :]))  # dot of two (presumed unit) vectors
                        if cos_angle < min_cos_angle:
                            min_cos_angle = cos_angle
                            cj_min = cj
                    # Remove pair if angle threshold exceeded
                    if cj_min != -1 and (min_cos_angle <= cos_thresh_angle):
                        cm_min = x_vec[cj_min]
                        # Subtract pair's contribution to residual
                        # Aij_mj_sum -= x_sign[j]*pv_j·A_j + x_sign[cj_min]*pv_c·A_c
                        pv_j = pol_vectors[j, m, :]
                        pv_c = pol_vectors[cj_min, cm_min, :]
                        Aj = A_obj[j]
                        Ac = A_obj[cj_min]
                        Aij_mj_sum -= (
                            x_sign[j]     * (pv_j[:, None] * Aj).sum(axis=0) +
                            x_sign[cj_min]* (pv_c[:, None] * Ac).sum(axis=0)
                        )
                        # Reset solution / metadata
                        x[j, :] = 0.0; x[cj_min, :] = 0.0
                        x_vec[j] = 0;   x_vec[cj_min] = 0
                        x_sign[j] = 0;  x_sign[cj_min] = 0
                        Gamma_complement[j] = True
                        Gamma_complement[cj_min] = True
                        num_nonzero_ref[0] -= 2
                        wyrm_sum += 1
                print(f"Backtracking: {wyrm_sum} wyrms removed")

        # Verbose progress & stagnation check
        if verbose and (((k % max(1, K // nhistory)) == 0) or k == 0 or k == K - 1):
            _print_GPMO_py(k, ngrid, print_iter_ref, x, Aij_mj_sum, objective_history,
                           Bn_history, m_history, 0.0, normal_norms)
            print(f"Iteration = {k}, Number of nonzero dipoles = {num_nonzero_ref[0]}")
            # mirror the same (slightly quirky) C++ stagnation logic
            if print_iter_ref[0] - 1 >= 0:
                num_nonzeros[print_iter_ref[0]-1] = num_nonzero_ref[0]
            if print_iter_ref[0] > 10:
                a = num_nonzeros[print_iter_ref[0]-1]
                b = num_nonzeros[print_iter_ref[0]-2]
                c = num_nonzeros[print_iter_ref[0]-3]
                # The C++ checks equality of three consecutive counts
                if a == b == c:
                    print("Stopping iterations: number of nonzero dipoles unchanged over three backtracking cycles")
                    break

        # Stop if magnet limit reached
        if (num_nonzero_ref[0] >= N) or (num_nonzero_ref[0] >= max_nMagnets):
            _print_GPMO_py(k, ngrid, print_iter_ref, x, Aij_mj_sum, objective_history,
                           Bn_history, m_history, 0.0, normal_norms)
            print(f"Iteration = {k}, Number of nonzero dipoles = {num_nonzero_ref[0]}")
            if num_nonzero_ref[0] >= N:
                print("Stopping iterations: all dipoles in grid are populated")
            else:
                print("Stopping iterations: maximum number of nonzero magnets reached ")
            break

    return objective_history, Bn_history, m_history, num_nonzeros, x


def GPMO_ArbVec_backtracking_macromag_py(
    A_obj: np.ndarray,  # shape (3N, ngrid) == (A * mmax_vec)
    b_obj: np.ndarray,  # (ngrid,)
    mmax: np.ndarray,   # (3N,) scaling (used for normalization in objective eval)
    normal_norms: np.ndarray,  # (ngrid,)
    pol_vectors: np.ndarray,   # (N, nPolVecs, 3)
    K: int,
    verbose: bool,
    nhistory: int,
    backtracking: int,
    dipole_grid_xyz: np.ndarray,  # (N, 3)
    Nadjacent: int,
    thresh_angle: float,
    max_nMagnets: int,
    x_init: np.ndarray,
    cube_dim: float = 0.004,
    mu_ea: float = 1.00,
    mu_oa: float = 1.00,
    use_coils: bool = False,
    coil_path: str = None
):
    """
    Hybrid ArbVec GPMO + MacroMag:
    Returns (objective_history, Bn_history, m_history, num_nonzeros, x)

    Args:
        A_obj: (3N, ngrid)
        b_obj: (ngrid,)
        mmax: (3N,)
        normal_norms: (ngrid,)
        pol_vectors: (N, nPolVecs, 3)
        K: int
        verbose: bool
        nhistory: int
        backtracking: int
        dipole_grid_xyz: (N, 3)
        Nadjacent: int
        thresh_angle: float
        max_nMagnets: int
        x_init: (3N,)
        cube_dim: float
        mu_ea: float
        mu_oa: float
        use_coils: bool
        coil_path: str, Optional, default None
    """
    import time
    if use_coils and coil_path == None: 
        raise ValueError("Use coils set to True but missing coil path; Please provide the correct focus coil path")
    
    # Convert to numpy arrays
    A_obj = np.asarray(A_obj, dtype=np.float64, order="C")
    b_obj = np.asarray(b_obj, dtype=np.float64, order="C")
    normal_norms = np.asarray(normal_norms, dtype=np.float64, order="C")
    pol_vectors = np.asarray(pol_vectors, dtype=np.float64, order="C")
    x_init = np.asarray(x_init, dtype=np.float64, order="C")
    dipole_grid_xyz = np.asarray(dipole_grid_xyz, dtype=np.float64, order="C")
    mmax = np.asarray(mmax, dtype=np.float64, order="C")

    # Define constants
    ngrid = A_obj.shape[1]
    N = int(A_obj.shape[0] // 3)
    assert 3 * N == A_obj.shape[0], "A_obj first dim must be 3*N"
    A_obj_3x = A_obj.reshape(N, 3, ngrid)
    nPolVecs = int(pol_vectors.shape[1])
    cos_thresh_angle = float(np.cos(thresh_angle))
    print_iter_ref = [0]

    # Define dipole moment solution vectors
    x = np.zeros((N, 3), dtype=np.float64)
    x_vec = np.zeros(N, dtype=np.int32)
    x_sign = np.zeros(N, dtype=np.int8)

    # Define set of admissible dipole positions
    Gamma_complement = np.ones((N,), dtype=bool)
    gamma_inds = []

    # Define history of dipole moments, objective values, and number of nonzero dipoles
    m_history = np.zeros((N, 3, nhistory + 2), dtype=np.float64)
    objective_history = np.zeros((nhistory + 2,), dtype=np.float64)
    Bn_history = np.zeros((nhistory + 2,), dtype=np.float64)
    num_nonzeros = np.zeros((nhistory + 2,), dtype=np.int32)
    last_x_macro_flat = np.zeros(3 * N, dtype=np.float64)
    num_nonzero_ref = [0]

    if verbose:
        print("Hybrid GPMO (ArbVec score) with MacroMag-on-winner evaluation")
        print("Iteration ... |Am - b|^2 ... lam*|m|^2")

    # Define connectivity matrix
    t1 = time.time()
    Connect = None
    if backtracking != 0:
        Connect = _connectivity_matrix_py(dipole_grid_xyz, Nadjacent)
    t2 = time.time()
    print(f"Time taken for connectivity matrix: {t2 - t1} seconds")

    # Define MacroMag tiles
    MM_CUBOID_FULL_DIMS = (cube_dim, cube_dim, cube_dim)
    vol = float(np.prod(MM_CUBOID_FULL_DIMS))
    tiles_all = Tiles(N)
    dims = np.array(MM_CUBOID_FULL_DIMS, dtype=np.float64)
    for j in range(N):
        tiles_all.offset = (dipole_grid_xyz[j], j)
        tiles_all.size = (dims, j)
        tiles_all.rot = ((0.0, 0.0, 0.0), j)
        tiles_all.M_rem = (0.0, j)
        tiles_all.mu_r_ea = (mu_ea, j)
        tiles_all.mu_r_oa = (mu_oa, j)
        tiles_all.u_ea = ((0.0, 0.0, 1.0), j)
    mac_all = MacroMag(tiles_all)

    # Load coils if requested
    if use_coils and coil_path is not None:
        mac_all.load_coils(coil_path)  # sets mac_all._bs_coil

    # Precompute demagnetization tensor
    N_full = mac_all.fast_get_demag_tensor(cache=True)
    
    # Precompute coil field if requested
    if use_coils:
        Hcoil_all = mac_all.coil_field_at(tiles_all.offset).astype(np.float64)
    
    def solve_subset_and_score(active_idx: np.ndarray, ea_list: np.ndarray, x0_prev = None, 
                              prev_active_idx: np.ndarray = None, A_prev: np.ndarray = None, prev_n: int = 0):
        """
        Solve the subset of tiles and score the residual.

        Args:
            active_idx: np.ndarray, shape (n_active,), indices of the active tiles
            ea_list: np.ndarray, shape (n_active, 3), easy axes of the active tiles
            x0_prev: np.ndarray, shape (3 * n_active,), previous solution
            prev_active_idx: np.ndarray, shape (n_prev,), previous active indices
            A_prev: np.ndarray, shape (n_prev, n_prev, 3, 3), previous A matrix

        Returns:
            R2: float, score
            x_macro_flat: np.ndarray, shape (3 * n_active,), macro magnetization
            mac.tiles.M: np.ndarray, shape (n_active, 3), macro magnetization
            res: np.ndarray, shape (3 * n_active,), residual
            N_sub: np.ndarray, shape (n_active, n_active, 3, 3), current submatrix

        """
        # If no active tiles, return 0 score and empty solution
        if active_idx.size == 0:
            res0 = -b_obj
            return 0.5 * float(np.dot(res0, res0)), np.zeros(3 * N), np.zeros((0, 3)), res0, None

        # Create a subset of tiles using vectorized operations
        sub = Tiles(active_idx.size)
        
        # Vectorized property setting
        n_active = active_idx.size
        
        # Set offsets for all active tiles at once
        sub.offset = tiles_all.offset[active_idx]
        
        # Set size for all active tiles (broadcast dims to all tiles)
        sub.size = np.tile(dims, (n_active, 1))
        
        # Set rotation for all active tiles (all zeros)
        sub.rot = np.zeros((n_active, 3))
        
        # Set permeability for all active tiles (broadcast scalar values)
        sub.mu_r_ea = np.full(n_active, mu_ea)
        sub.mu_r_oa = np.full(n_active, mu_oa)
        
        # Set remanent magnetization for all active tiles
        sub.M_rem = np.full(n_active, M_rem_value)
        
        # Set easy axes for all active tiles (vectorized normalization)
        # Avoid triggering the u_ea setter which calls M setter for each tile
        ea_norms = np.linalg.norm(ea_list, axis=1, keepdims=True)
        ea_normalized = ea_list / (ea_norms + 1e-30)
        sub._u_ea = ea_normalized  # Direct assignment to avoid setter overhead

        mac = MacroMag(sub, bs_interp=mac_all.bs_interp) #Taking already built interpolanet field to avod reloading...
        
        n_prev = prev_active_idx.size if prev_active_idx is not None else 0
        n_active = active_idx.size
        n_new = n_active - n_prev
        
        # Add new rows and columns from newly placed magnets
        if n_prev > 0:
            new_indices = active_idx[n_prev:]
            old_indices = active_idx[:n_prev]
            N_new_rows = N_full[np.ix_(new_indices, old_indices)]  # (n_new, n, 3, 3)
            N_new_cols = N_full[np.ix_(old_indices, new_indices)]  # (n, n_new, 3, 3)
            N_new_diag = N_full[np.ix_(new_indices, new_indices)]  # (n_new, n_new, 3, 3)
        else:
            N_new_rows = N_full[np.ix_(active_idx, active_idx)]
            N_new_cols = None
            N_new_diag = None
            
        # Perform the magtense coupling direct solve
        mac, A_sub = mac.direct_solve(
            use_coils=use_coils, 
            N_new_rows=N_new_rows, 
            N_new_cols=N_new_cols, 
            N_new_diag=N_new_diag, 
            x0=x0_prev, 
            H_a_override=Hcoil_all[active_idx], print_progress=verbose, 
            A_prev=A_prev, 
            prev_n=prev_n
        )
        
        # Get the macro magnetization for the subset of tiles
        m_sub = mac.tiles.M * vol
        m_full = np.zeros((N, 3))
        m_full[active_idx, :] = m_sub
        m_full_vec = m_full.reshape(3 * N)
        x_macro_flat = m_full_vec / mmax
        res = A_obj.T @ x_macro_flat - b_obj
        R2 = 0.5 * float(np.dot(res, res))
        return R2, x_macro_flat, np.ravel(mac.tiles.M), res, A_sub

    active0 = np.where(~Gamma_complement)[0]
    ea0 = np.zeros((active0.size, 3))
    R2_0, x_macro_flat_0, _, res0, A_sub_0 = solve_subset_and_score(active0, ea0)
    last_x_macro_flat[:] = x_macro_flat_0

    # record init
    idx0 = print_iter_ref[0]
    m_history[:, :, idx0] = x_macro_flat_0.reshape(N, 3)
    objective_history[idx0] = R2_0
    Bn_history[idx0] = float(np.sum(np.abs(res0) * np.sqrt(normal_norms))) / np.sqrt(float(ngrid))
    num_nonzeros[idx0] = num_nonzero_ref[0]
    if verbose:
        print(f"0 ... {R2_0:.2e} ... 0.00e+00 ")
    print_iter_ref[0] += 1

    # Running residual for ArbVec scoring (classical)
    Aij_mj_sum = (A_obj.T @ x_macro_flat_0) - b_obj
    R2s = np.full((2 * N * nPolVecs,), 1e50, dtype=np.float64)

    # Initialize from x_init (may place some dipoles)
    _initialize_GPMO_ArbVec_py(
        x_init, pol_vectors, x, x_vec, x_sign,
        A_obj_3x, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref
    )

    # Main greedy loop
    t_gpmo = time.time()
    x0_prev = None
    x0_prev_partial = None
    prev_active_indices = None
    N_sub_prev = None
    A_sub_prev = None
    prev_n = 0
    
    for k in range(1, K + 1):
        # t_k = time.time()
        
        # x0_prev = None # ToDo: Implement and Profile if warm start helps speed-up iteartion

        # Stop if filled or hit magnet cap     
        # t1_magtens = time.time()   
        if (num_nonzero_ref[0] >= N) or (num_nonzero_ref[0] >= max_nMagnets):
            # record final snapshot (ALWAYS on early stop, like reference)
            active = np.where(~Gamma_complement)[0]
            # active = Gamma
            ea_list = x[active] if active.size else np.zeros((0, 3))
            R2_snap, x_macro_flat, _, res, _ = solve_subset_and_score(active, ea_list, x0_prev)
            last_x_macro_flat[:] = x_macro_flat
            idx = print_iter_ref[0]
            m_history[:, :, idx] = x_macro_flat.reshape(N, 3)
            objective_history[idx] = R2_snap
            Bn_history[idx] = float(np.sum(np.abs(res) * np.sqrt(normal_norms))) / np.sqrt(float(ngrid))
            num_nonzeros[idx] = num_nonzero_ref[0]
            if verbose:
                print(f"{k} ... {R2_snap:.2e} ... 0.00e+00 ")
                print(f"Iteration = {k}, Number of nonzero dipoles = {num_nonzero_ref[0]}")
            print_iter_ref[0] += 1
            print("Stopping iterations: maximum number of nonzero magnets reached " if (num_nonzero_ref[0] >= max_nMagnets) else "Stopping iterations: all dipoles in grid are populated")
            break
        # t2_magtens = time.time()
        # print(f"Time taken for magtense coupling direct solve: {t2_magtens - t1_magtens} seconds")

        # t1_selection = time.time()
        # Selection with classical ArbVec score (no MacroMag here to prevent N^4 complexity) 
        NNp = N * nPolVecs
        
        # Vectorized version of the loop over j
        # Initialize R2s with large values for occupied positions
        R2s[:] = 1e50
        
        # Find active (unoccupied) positions
        active_mask = Gamma_complement
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) > 0:
            # Extract active A_obj_3x and pol_vectors
            A_active = A_obj_3x[active_indices]  # shape: (n_active, 3, ngrid)
            pol_active = pol_vectors[active_indices]  # shape: (n_active, nPolVecs, 3)
            mmax_active = mmax[active_indices]  # shape: (n_active,)
            
            # Vectorized computation of bnorm for all active positions
            # bnorm[j, p, :] = sum over k of pol_vectors[j, p, k] * A_obj_3x[j, k, :]
            # Shape: (n_active, nPolVecs, ngrid)
            bnorm = np.einsum('jpk,jkg->jpg', pol_active, A_active)
            
            # Compute tmp_plus and tmp_minus for all active positions
            # Shape: (n_active, nPolVecs, ngrid)
            tmp_plus = Aij_mj_sum[None, None, :] + bnorm
            tmp_minus = Aij_mj_sum[None, None, :] - bnorm
            
            # Compute squared norms for all active positions
            # Shape: (n_active, nPolVecs)
            R2 = np.sum(tmp_plus ** 2, axis=2)
            R2minus = np.sum(tmp_minus ** 2, axis=2)
            
            # Add mmax^2 terms
            R2 += (mmax_active ** 2)[:, None]
            R2minus += (mmax_active ** 2)[:, None]
            
            # Store results in R2s array using vectorized indexing
            # Create base indices for all active positions
            base_indices = active_indices * nPolVecs  # shape: (n_active,)
            
            # Create all indices for R2s storage
            # For positive signs: base to base + nPolVecs
            pos_indices = base_indices[:, None] + np.arange(nPolVecs)[None, :]  # shape: (n_active, nPolVecs)
            pos_indices = pos_indices.ravel()  # flatten to 1D
            
            # For negative signs: NNp + base to NNp + base + nPolVecs  
            neg_indices = NNp + pos_indices
            
            # Store results using advanced indexing
            R2s[pos_indices] = R2.ravel()
            R2s[neg_indices] = R2minus.ravel()

        # Best (site, pol, sign)
        skj = int(np.argmin(R2s))
        sign_fac = -1.0 if skj >= NNp else +1.0
        if skj >= NNp: skj -= NNp
        skjj = int(skj % nPolVecs)
        j_best = int(skj // nPolVecs)

        # Commit in classical state
        pv = pol_vectors[j_best, skjj, :]
        x[j_best, :] = sign_fac * pv / (np.linalg.norm(pv) + 1e-30)
        x_vec[j_best] = skjj
        x_sign[j_best] = int(np.sign(sign_fac))
        Gamma_complement[j_best] = False
        gamma_inds.append(j_best)
        num_nonzero_ref[0] += 1
        Aij_mj_sum += sign_fac * (pv[:, None] * A_obj_3x[j_best]).sum(axis=0)
        base = j_best * nPolVecs
        R2s[base:base + nPolVecs] = 1e50
        R2s[NNp + base:NNp + base + nPolVecs] = 1e50
        # t2_selection = time.time()
        # print(f"Time taken for selection: {t2_selection - t1_selection} seconds")

        # Backtracking
        if backtracking != 0:  # Omitting backtracking in case 0 is passed
            if (k % backtracking) == 0:
                removed = 0
                for j in range(N):
                    if Gamma_complement[j]:
                        continue
                    min_cos = 2.0
                    cj_min = -1
                    for jj in range(Connect.shape[1]):
                        cj = int(Connect[j, jj])
                        if Gamma_complement[cj]:
                            continue
                        cos_angle = float(np.dot(x[j, :], x[cj, :]))
                        if cos_angle < min_cos:
                            min_cos = cos_angle
                            cj_min = cj
                    if cj_min != -1 and (min_cos <= cos_thresh_angle):
                        pv_j = pol_vectors[j, x_vec[j], :]
                        pv_c = pol_vectors[cj_min, x_vec[cj_min], :]
                        Aj = A_obj_3x[j]
                        Ac = A_obj_3x[cj_min]
                        Aij_mj_sum -= (
                            float(x_sign[j]) * (pv_j[:, None] * Aj).sum(axis=0) +
                            float(x_sign[cj_min]) * (pv_c[:, None] * Ac).sum(axis=0)
                        )
                        x[j, :] = 0.0; x[cj_min, :] = 0.0
                        x_vec[j] = 0;  x_vec[cj_min] = 0
                        x_sign[j] = 0; x_sign[cj_min] = 0
                        Gamma_complement[j] = True
                        Gamma_complement[cj_min] = True
                        gamma_inds.remove(j)
                        gamma_inds.remove(cj_min)
                        num_nonzero_ref[0] -= 2
                        removed += 1
                if verbose:
                    print(f"Backtracking: {removed} pairs removed")

        # Probably only need to run MacroMag for the committed set ONLY every couple dozen iterations
        if (((k % 20) == 0)):
            # active = np.where(~Gamma_complement)[0]
            # active = gamma_inds
            # ea_list = x[gamma_inds]  # if gamma_inds.size else np.zeros((0, 3))
            # if x0_prev_partial is not None:
            #     j_best_position = np.where(active == j_best)[0][0]
            #     # print(j_best, j_best_position, active[j_best_position])
            #     x0_prev = np.insert(x0_prev_partial, j_best_position, np.zeros(3))
            t1_macromag = time.time()
            R2_snap, x_macro_flat, _, Aij_mj_sum, A_sub_prev = solve_subset_and_score(
                np.array(gamma_inds), x[gamma_inds], prev_active_idx=prev_active_indices, 
                A_prev=A_sub_prev, prev_n=prev_n
                ) #, x0_prev=x0_prev)
            t2_macromag = time.time()
            print(f"Iteration {k}: Time in macromag call: {t2_macromag - t1_macromag} seconds")
            prev_active_indices = np.array(gamma_inds.copy())
            prev_n = len(gamma_inds)
            # last_x_macro_flat[:] = x_macro_flat
            # Aij_mj_sum = res
    

        # print(f"x0_prev_partial: {np.shape(x0_prev_partial)}")
        # print(f"x0_prev: {np.shape(x0_prev)}")
        # print(f"x0_prev_norm: {np.linalg.norm(x0_prev)}")
        # print(f"x0_prev_partial_norm: {np.linalg.norm(x0_prev_partial)}")
            
        # Store current active indices for next iteration
        # prev_active_indices = active.copy()

        if verbose and (((k % max(1, K // nhistory)) == 0) or (k == K)):
            idx = print_iter_ref[0]
            m_history[:, :, idx] = x_macro_flat.reshape(N, 3)
            objective_history[idx] = R2_snap
            Bn_history[idx] = float(np.sum(np.abs(Aij_mj_sum) * np.sqrt(normal_norms))) / np.sqrt(float(ngrid))
            num_nonzeros[idx] = num_nonzero_ref[0]
            print(f"{k} ... {R2_snap:.2e} ... 0.00e+00 ")
            print(f"Iteration = {k}, Number of nonzero dipoles = {num_nonzero_ref[0]}")
            print_iter_ref[0] += 1

        # Stagnation stop (match reference: no forced recording here)
        if print_iter_ref[0] > 3:
            a = num_nonzeros[print_iter_ref[0] - 1]
            b = num_nonzeros[print_iter_ref[0] - 2]
            c = num_nonzeros[print_iter_ref[0] - 3]
            if a == b == c:
                print("Stopping: number of nonzero dipoles unchanged over three reporting cycles")
                break

        # t_k = time.time() - t_k
        # print(f"Time taken for iteration {k}: {t_k} seconds")

    t_gpmo = time.time() - t_gpmo
    print(f"Time taken for GPMO: {t_gpmo} seconds")

    return objective_history, Bn_history, m_history, num_nonzeros, x_macro_flat.reshape(N, 3)
