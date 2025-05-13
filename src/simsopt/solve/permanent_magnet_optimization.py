import warnings


import numpy as np

import simsoptpp as sopp
from .._core.types import RealArray


__all__ = ['relax_and_split', 'GPMO']


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
