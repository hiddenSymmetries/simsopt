import numpy as np
import simsoptpp as sopp


def prox_l0(m, mmax, reg_l0, nu):
    """
    Proximal operator for L0 regularization.
    Note that the m values are normalized before hard
    thresholding to avoid only truncating the magnets on the
    inner side of the configuration (which are often important!).
    Note also the in principle the hard threshold here should be:
    hard_threshold = sqrt(2 * reg_l0 * nu). But we can always imagine
    rescaling nu (or reg_l0) such that hard_threshold = 2 * reg_l0 * nu
    instead (for instance nu -> 2 * nu ** 2 and reg_l0 -> reg_l0 ** 2).
    """
    ndipoles = len(m) // 3
    mmax_vec = np.array([mmax, mmax, mmax]).T
    m_normalized = (np.abs(m).reshape(ndipoles, 3) / mmax_vec).reshape(ndipoles * 3)
    # in principle, hard threshold should be sqrt(2 * reg_l0 * nu)
    return m * (m_normalized > 2 * reg_l0 * nu)


def prox_l1(m, mmax, reg_l1, nu):
    """
    Proximal operator for L1 regularization.
    Note that the m values are normalized before soft
    thresholding to avoid only truncating the magnets on the
    inner side of the configuration (which are often important!).
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
    """
    N = len(x) // 3
    x_shaped = x.reshape(N, 3)
    denom_fac = np.sqrt(np.sum(x_shaped ** 2, axis=-1)) / mmax
    denom = np.maximum(np.ones(len(denom_fac)), denom_fac)
    return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)


def rescale_for_opt(pm_opt, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu):
    """
        Scale regularizers to the largest scale of ATA (~1e-6)
        to avoid regularization >> ||Am - b|| ** 2 term in the optimization.
        The prox operator uses reg_l0 * nu for the threshold so normalization
        below allows reg_l0 and reg_l1 values to be exactly the thresholds
        used in calculation of the prox. Then add contributions to ATA and
        ATb coming from extra loss terms such as L2 regularization and
        relax-and-split.
    """

    print('L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2))
    print('Shifted L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2_shifted))

    if reg_l0 < 0 or reg_l0 > 1:
        raise ValueError(
            'L0 regularization must be between 0 and 1. This '
            'value is automatically scaled to the largest of the '
            'dipole maximum values, so reg_l0 = 1 should basically '
            'truncate all the dipoles to zero. '
        )
    # Compute singular values of A, use this to determine optimal step size
    # for the MwPGP algorithm, with alpha ~ 2 / ATA_scale
    S = np.linalg.svd(pm_opt.A_obj, full_matrices=False, compute_uv=False)
    pm_opt.ATA_scale = S[0] ** 2

    # Rescale L0 and L1 so that the values used for thresholding
    # are only parametrized by the values of reg_l0 and reg_l1
    reg_l0 = reg_l0 / (2 * nu)
    reg_l1 = reg_l1 / nu

    # may want to rescale nu, otherwise just scan this value
    # nu = nu / pm_opt.ATA_scale

    # Do not rescale L2 terms for now. reg_l2_shifted is not well tested.
    reg_l2 = reg_l2
    reg_l2_shifted = reg_l2_shifted

    # Update algorithm step size if we have extra smooth, convex loss terms
    pm_opt.ATA_scale = S[0] ** 2 + 2 * (reg_l2 + reg_l2_shifted) + 1.0 / nu

    # Add shifted L2 contribution to ATb. Note that relax-and-split
    # contribution must be added within the optimization, because it
    # depends on the 'w' optimization variable.
    ATb = pm_opt.ATb + reg_l2_shifted * np.ravel(np.outer(pm_opt.m_maxima, np.ones(3)))
    return ATb, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu


def setup_initial_condition(pm_opt, m0=None):
    """
        If an initial guess for the dipole moments is specified,
        checks the initial condition lies in the allowed hypersurface.
        If an initial guess is not specified, defaults to initializing
        the permanent magnet dipole moments to all zeros.
    """
    # Initialize initial guess for the dipole strengths
    if m0 is not None:
        if len(m0) != pm_opt.ndipoles * 3:
            raise ValueError(
                'Initial dipole guess is incorrect shape --'
                ' guess must be 1D with shape (ndipoles * 3).'
            )
        m0_temp = projection_L2_balls(
            m0,
            pm_opt.m_maxima
        )
        # check if m0 lies inside the hypersurface spanned by
        # L2 balls, which we require it does
        if not np.allclose(m0, m0_temp):
            raise ValueError(
                'Initial dipole guess must contain values '
                'that are satisfy the maximum bound constraints.'
            )
        pm_opt.m0 = m0
    elif not hasattr(pm_opt, 'm0'):
        # initialization to proj(pinv(A) * b) is usually a bad guess,
        # so defaults to zero here, but can be overwritten
        # when optimization is performed
        pm_opt.m0 = np.zeros(pm_opt.ndipoles * 3)


def relax_and_split(pm_opt, m0=None, algorithm='MwPGP', **kwargs):
    """
        Use a relax-and-split algorithm for solving the permanent
        magnet optimization problem, which solves a convex and nonconvex part
        separately. Defaults to the MwPGP convex step and no nonconvex step.
        If a nonconvexity is specified, the associated prox function must
        be defined in this file.
        Relax-and-split allows for speedy algorithms for both steps and
        the imposition of convex equality and inequality
        constraints (including the required constraint on
        the strengths of the dipole moments).
    Args:
        m0: Initial guess for the permanent magnet
            dipole moments. Defaults to a
            starting guess of all zeros.
            This vector must lie in the hypersurface
            spanned by the L2 ball constraints. Note
            that if algorithm is being used properly,
            the end result should be independent of
            the choice of initial condition.
        epsilon: Error tolerance for the convex
            part of the algorithm (MwPGP).
        nu: Hyperparameter used for the relax-and-split
            least-squares. Set nu >> 1 to reduce the
            importance of nonconvexity in the problem.
        reg_l0: Regularization value for the L0
            nonconvex term in the optimization. This value
            is automatically scaled based on the max dipole
            moment values, so that reg_l0 = 1 corresponds
            to reg_l0 = np.max(m_maxima). It follows that
            users should choose reg_l0 in [0, 1].
        reg_l1: Regularization value for the L1
            nonsmooth term in the optimization,
        reg_l2: Regularization value for any convex
            regularizers in the optimization problem,
            such as the often-used L2 norm.
        reg_l2_shifted: Regularization value for the L2
            smooth and convex term in the optimization,
            shifted by the vector of maximum dipole magnitudes.
        max_iter_MwPGP: Maximum iterations to perform during
            a run of the convex part of the relax-and-split
            algorithm (MwPGP).
        max_iter_RS: Maximum iterations to perform of the
            overall relax-and-split algorithm. Therefore,
            also the number of times that MwPGP is called,
            and the number of times a prox is computed.
        verbose: Prints out all the loss term errors separately.
    """

    # Really not sure if the kwargs approach will work here,
    # especially with the C++ function bindings
    if max_iter_RS not in kwargs.keys():
        kwargs['max_iter_RS'] = 1
    if reg_l0 not in kwargs.keys():
        kwargs['reg_l0'] = 0.0
    if reg_l1 not in kwargs.keys():
        kwargs['reg_l1'] = 0.0
    if reg_l2 not in kwargs.keys():
        kwargs['reg_l2'] = 0.0
    if reg_l2_shifted not in kwargs.keys():
        kwargs['reg_l2_shifted'] = 0.0
    if nu not in kwargs.keys():
        kwargs['nu'] = 1e100

    # Rescale the hyperparameters and then add contributions to ATA and ATb
    ATb, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu = rescale_for_opt(
        pm_opt, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu
    )

    # change to row-major order for the C++ code
    A_obj = np.ascontiguousarray(pm_opt.A_obj)
    ATb = np.ascontiguousarray(np.reshape(ATb, (pm_opt.ndipoles, 3)))

    # print initial errors and values before optimization
    pm_opt._print_initial_opt()

    # Begin the various algorithms
    errors = []
    m_history = []
    m_proxy_history = []
    if algorithm == 'MwPGP':
        # get optimal alpha value for the MwPGP algorithm
        alpha_max = 2.0 / pm_opt.ATA_scale
        alpha_max = alpha_max * (1 - 1e-5)
        convex_step = sopp.MwPGP_algorithm
    elif algorithm == 'SPG':
        convex_step = sopp.SPG_algorithm
    else:
        raise NotImplementedError(
            "Only spectral projected gradient (SPG) and MwPGP have been "
            "implemented for the convex step in the relax-and-split "
            "algorithm."
        )

    # set the nonconvex step in the algorithm
    if (not np.isclose(reg_l0, 0.0)) and (not np.isclose(reg_l1, 0.0)):
        raise ValueError(' L0 and L1 loss terms cannot be used concurrently.')
    elif not np.isclose(reg_l0, 0.0):
        prox = prox_l0
    elif not np.isclose(reg_l1, 0.0):
        prox = prox_l1
    # Auxiliary variable in relax-and-split is initialized
    # to prox(m0), where m0 is the initial guess for m.
    if m0 is not None:
        setup_initial_condition(pm_opt, m0)
    m0 = pm_opt.m0
    m_proxy = pm_opt.m0
    mmax = pm_opt.m_maxima
    m_proxy = prox(m_proxy, mmax, reg_l0, nu)

    # Begin optimization
    if reg_l0 > 0.0 or reg_l1 > 0.0:
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
                    #alpha=alpha_max,
                    #verbose=True,
                    #reg_l0=reg_l0,
                    #reg_l1=reg_l1,
                    #reg_l2=reg_l2,
                    #reg_l2_shifted=reg_l2_shifted,
                    #nu=nu,
                    # min_fb=min_fb,
                    # epsilon=epsilon,
                    # max_iter=max_iter_MwPGP,
                )
            m_history.append(m)
            m = np.ravel(m)
            algorithm_history = algorithm_history[algorithm_history != 0]
            errors.append(algorithm_history[-1])

            # Solve the nonconvex optimization -- i.e. take a prox
            m_proxy = prox(m, reg_l0, nu)
            m_proxy_history.append(m_proxy)
            if np.linalg.norm(m - m_proxy) < epsilon:
                print('Relax-and-split finished early, at iteration ', i)
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


def BMP_wrapper(pm_opt, **kwargs):
    """
        Binary matching pursuit is a greedy algorithm, derived
        from the orthorgonal matching pursuit algorithm, and
        is an alternative to
        the relax-and-split algorithm for solving the permanent
        magnet optimization problem. Full-strength magnets are placed
        one-by-one according to minimize a submodular objective function
        such as the determinant of ATA.
    Args:
        K:  The number of magnets to place one-by-one.
        reg_l2: Regularization value for any convex
            regularizers in the optimization problem,
            such as the often-used L2 norm.
        reg_l2_shifted: Regularization value for the L2
            smooth and convex term in the optimization,
            shifted by the vector of maximum dipole magnitudes.
        verbose: Prints out all the loss term errors separately.
    """
    # Begin the various algorithms
    errors = []
    m_history = []
    m_proxy_history = []

    # Need to normalize the m here, so have
    # || A * m - b||^2 = || ( A * mmax) * m / mmax - b||^2
    mmax = pm_opt.m_maxima
    mmax_vec = np.array([mmax, mmax, mmax]).T.reshape(pm_opt.ndipoles * 3)
    A_obj = pm_opt.A_obj / mmax_vec

    # If have L2 regularization, this just rescales reg_l2,
    # since reg_l2 * ||m||^2 = reg_l2 * ||mmax||^2 * ||m / mmax||^2
    mmax_norm2 = np.linal.norm(mmax_vec, ord=2) ** 2
    reg_l2 = reg_l2 * mmax_norm2
    reg_l2_shifted = reg_l2_shifted * mmax_norm2

    algorithm_history, _, m_history, m = sopp.BMP_algorithm(
        A_obj=A_obj,
        b_obj=pm_opt.b_obj,
        ATb=ATb,
        **kwargs
    )

    # rescale m and m_history
    m = m * (mmax_vec.reshape(pm_opt.ndipoles, 3))

    # check that algorithm worked correctly to generate K binary dipoles
    print('Number of binary dipoles to use in BMP algorithm = ', K)
    print(np.count_nonzero(m))
    print(
        'Number of binary dipoles returned by BMP algorithm = ',
        np.count_nonzero(np.sum(m.reshape(pm_opt.ndipoles, 3), axis=-1))
    )

    for i in range(len(m_history)):
        m_history[i] = m_history[i] * (mmax_vec.reshape(pm_opt.ndipoles, 3))
    errors = algorithm_history[algorithm_history != 0]

    # note m = m_proxy for BMP because this is not using relax-and-split
    pm_opt.m = np.ravel(m)
    pm_opt.m_proxy = pm_opt.m
    return errors, m_history, m_proxy_history
