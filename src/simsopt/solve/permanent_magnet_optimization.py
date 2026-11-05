"""Permanent magnet optimization algorithms: relax-and-split, GPMO, GPMOmr."""
from __future__ import annotations

import dataclasses
import time
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from .macromagnetics import MacroMag, SolverParams, Tiles, assemble_blocks_subset

import simsoptpp as sopp
from .._core.types import RealArray
from ..util.permanent_magnet_helper_functions import (
    GPMOBacktrackParams,
    GPMOMacroMagParams,
    GPMOParams,
    RSParams,
)

if TYPE_CHECKING:
    from simsopt.geo import PermanentMagnetGrid

__all__ = ['relax_and_split', 'GPMO']

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_UNAVAILABLE_R2: float = 1e50
"""Sentinel used to mask unavailable (site, polarization) entries in R2s."""

_NORMALIZATION_EPS: float = 1e-30
"""Small epsilon to prevent division-by-zero when normalizing vectors."""

_INIT_TOL_FACTOR: float = 4.0 * np.finfo(np.float32).eps
"""Tolerance factor for checking initialized dipoles against allowable moments."""


def prox_l0(
    m: RealArray,
    mmax: RealArray,
    reg_l0: float,
    nu: float,
) -> np.ndarray:
    r"""
    Proximal operator for :math:`\ell_0` regularisation (hard thresholding).

    Dipole magnitudes are normalised by *mmax* before thresholding so
    that the sparsification does not preferentially remove magnets on the
    inboard side of the configuration where magnets are typically weaker.

    The standard hard threshold is
    :math:`\sqrt{2\,\lambda_0\,\nu}`, but the code uses
    :math:`2\,\lambda_0\,\nu` (equivalent after re-scaling
    :math:`\nu \to 2\nu^2`).

    Args:
        m: Dipole vectors, shape ``(ndipoles * 3,)``.
        mmax: Per-magnet maximum dipole strengths, shape ``(ndipoles,)``.
        reg_l0: :math:`\ell_0` regularisation weight :math:`\lambda_0`.
        nu: Relaxation parameter :math:`\nu` in the relax-and-split scheme.

    Returns:
        Hard-thresholded dipole vector, same shape as *m*.
    """
    ndipoles = len(m) // 3
    mmax_vec = np.array([mmax, mmax, mmax]).T
    m_normalized = (np.abs(m).reshape(ndipoles, 3) / mmax_vec).reshape(ndipoles * 3)
    # in principle, hard threshold should be sqrt(2 * reg_l0 * nu) but can always renormalize everything
    return m * (m_normalized > 2 * reg_l0 * nu)


def prox_l1(
    m: RealArray,
    mmax: RealArray,
    reg_l1: float,
    nu: float,
) -> np.ndarray:
    r"""
    Proximal operator for :math:`\ell_1` regularisation (soft thresholding).

    Applies the soft-thresholding operator

    .. math::

        \operatorname{prox}_{\lambda_1 \nu \|\cdot\|_1}(m)_k
        = \operatorname{sign}(m_k)\,
          \max\!\bigl(|\bar m_k| - \lambda_1 \nu,\; 0\bigr)\,m_{\max,k}

    where :math:`\bar m_k = m_k / m_{\max,k}` is the normalised component.

    Args:
        m: Dipole vectors, shape ``(ndipoles * 3,)``.
        mmax: Per-magnet maximum dipole strengths, shape ``(ndipoles,)``.
        reg_l1: :math:`\ell_1` regularisation weight :math:`\lambda_1`.
        nu: Relaxation parameter :math:`\nu` in the relax-and-split scheme.

    Returns:
        Soft-thresholded dipole vector, same shape as *m*.
    """
    ndipoles = len(m) // 3
    mmax_vec = np.array([mmax, mmax, mmax]).T
    m_normalized = (np.abs(m).reshape(ndipoles, 3) / mmax_vec).reshape(ndipoles * 3)
    return np.sign(m) * np.maximum(np.abs(m_normalized) - reg_l1 * nu, 0) * np.ravel(mmax_vec)


def projection_L2_balls(x: RealArray, mmax: RealArray) -> np.ndarray:
    r"""
    Project onto the product of :math:`\ell_2`-balls in :math:`\mathbb{R}^3`.

    Each magnet has the feasible set
    :math:`\|\mathbf{m}_i\| \le m_{\max,i}`.  This projects *x*
    (interpreted as stacked :math:`(\mathbf{m}_1, \mathbf{m}_2, \dots)`)
    onto that product by scaling down any magnet that exceeds its ball.

    Args:
        x: Dipole vector, shape ``(ndipoles * 3,)``.
        mmax: Per-magnet maximum norms, shape ``(ndipoles,)``.

    Returns:
        Projected vector, same shape as *x*.
    """
    N = len(x) // 3
    x_shaped = x.reshape(N, 3)
    denom_fac = np.sqrt(np.sum(x_shaped ** 2, axis=-1)) / mmax
    denom = np.maximum(np.ones(len(denom_fac)), denom_fac)
    return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)


def setup_initial_condition(
    pm_opt: "PermanentMagnetGrid",
    m0: Optional[RealArray] = None,
) -> None:
    """
    Validate and set the initial condition for permanent magnet optimization.

    If m0 is provided, verifies it lies within the L2-ball product
    (|m_i| ≤ m_maxima_i) and stores it. If m0 is None, no explicit
    initialization is performed (algorithms typically start from zero).

    Args:
        pm_opt: PermanentMagnetGrid instance to optimize.
        m0: Optional initial guess, shape (ndipoles * 3). Must satisfy
            constraints; otherwise raises ValueError.

    Raises:
        ValueError: If m0 has wrong shape or violates constraints.
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


def relax_and_split(
    pm_opt: PermanentMagnetGrid,
    m0: RealArray | None = None,
    params: RSParams | None = None,
    **kwargs,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    r"""
    Relax-and-split algorithm for permanent magnet optimization.

    Solves the permanent magnet optimization problem by alternating
    between a convex sub-problem and a nonconvex proximal step.

    The optimisation minimises

    .. math::

        \min_{\mathbf{m}} \;\tfrac{1}{2}\|A\,\mathbf{m} - \mathbf{b}\|_2^2
        + \lambda_2\|\mathbf{m}\|_2^2
        + \lambda_0 R_0(\mathbf{m})
        + \lambda_1 R_1(\mathbf{m})

    subject to per-dipole L2-ball constraints
    :math:`\|\mathbf{m}_i\| \le m_{\max,i}`.

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
        params: Typed parameter dataclass.  When provided, *params*
            takes precedence over any remaining *kwargs*.
        kwargs: **Deprecated.**  Legacy keyword arguments; prefer
            *params*.

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
    if params is not None:
        kwargs = dataclasses.asdict(params)
    elif kwargs:
        warnings.warn(
            "Passing bare **kwargs to relax_and_split() is deprecated. "
            "Use an RSParams dataclass via the 'params' argument instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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


def GPMO(
    pm_opt: PermanentMagnetGrid,
    algorithm: str = 'baseline',
    params: GPMOParams | None = None,
    **kwargs,
) -> tuple[list[float], list[float], list[np.ndarray]]:
    r"""
    Greedy permanent magnet optimisation (GPMO).

    GPMO iteratively places full-strength magnets to greedily minimise
    the surface-flux objective

    .. math::

        f_B = \tfrac{1}{2}\,\|A\,\mathbf{m} - \mathbf{b}\|_2^2.

    Optional keyword arguments enable backtracking (error correction)
    and adjacency constraints that prevent isolated magnets.

    Args:
        pm_opt: The grid of permanent magnets to optimize.
            PermanentMagnetGrid instance.
        algorithm:
            The type of greedy algorithm to use. Options are

            baseline:
                the simple implementation of GPMO,
            multi:
                GPMO, but placing multiple magnets per iteration,
            GPMO_Backtracking:
                basis-vector backtracking variant (no ArbVec),
            GPMO_ArbVec:
                ArbVec GPMO (no backtracking),
            GPMO:
                ArbVec GPMO; backtracking is controlled by the
                ``backtracking`` field (set to 0 to disable),
            GPMO_py:
                pure-Python implementation of GPMO (used by GPMOmr),
            GPMOmr:
                ArbVec GPMO with macromagnetic refinement (MacroMag);
                backtracking controlled by ``backtracking`` and
                macromag refinement cadence by ``mm_refine_every``.

        params:
            Typed parameter dataclass.  Pass :class:`GPMOParams` for
            ``baseline`` / ``multi`` / ``GPMO_ArbVec``,
            :class:`GPMOBacktrackParams` for ``GPMO`` / ``GPMO_py`` /
            ``GPMO_Backtracking``, or :class:`GPMOMacroMagParams` for
            ``GPMOmr``.  When provided, *params* takes precedence over
            any remaining *kwargs*.
        kwargs:
            **Deprecated.**  Legacy keyword arguments; prefer *params*.

    Returns:
        Tuple of (errors, Bn_errors, m_history)

        errors:
            Total optimization loss values, recorded every *nhistory*
            iterations.
        Bn_errors:
            :math:`|Bn|` errors, recorded every *nhistory* iterations.
        m_history:
            Solution for the permanent magnets, recorded after
            *nhistory* iterations.
    """
    if not hasattr(pm_opt, "A_obj"):
        raise ValueError("The PermanentMagnetClass needs to use geo_setup() or "
                         "geo_setup_from_famus() before calling optimization routines.")

    # --- resolve params vs legacy **kwargs into a working dict ---------------
    if params is not None:
        _d: dict = {}
        _d["K"] = params.K
        _d["nhistory"] = params.nhistory
        _d["verbose"] = params.verbose
        _d["single_direction"] = params.single_direction
        if isinstance(params, GPMOBacktrackParams):
            _d["backtracking"] = params.backtracking
            _d["Nadjacent"] = params.Nadjacent
            _d["max_nMagnets"] = params.max_nMagnets
            _d["thresh_angle"] = params.thresh_angle
            if params.dipole_grid_xyz is not None:
                _d["dipole_grid_xyz"] = params.dipole_grid_xyz
            if params.m_init is not None:
                _d["m_init"] = params.m_init
        if isinstance(params, GPMOMacroMagParams):
            _d["cube_dim"] = params.cube_dim
            _d["mu_ea"] = params.mu_ea
            _d["mu_oa"] = params.mu_oa
            _d["use_coils"] = params.use_coils
            _d["use_demag"] = params.use_demag
            _d["coil_path"] = params.coil_path
            _d["mm_refine_every"] = params.mm_refine_every
            _d["current_scale"] = params.current_scale
        kwargs = _d
    elif kwargs:
        warnings.warn(
            "Passing bare **kwargs to GPMO() is deprecated. "
            "Use a GPMOParams / GPMOBacktrackParams / GPMOMacroMagParams "
            "dataclass via the 'params' argument instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Begin the various algorithms
    errors = []
    m_history = []

    # Need to normalize the m, so have
    # || A * m - b||^2 = || ( A * mmax) * m / mmax - b||^2
    mmax = pm_opt.m_maxima
    contig = np.ascontiguousarray
    mmax_vec = contig(np.array([mmax, mmax, mmax]).T.reshape(pm_opt.ndipoles * 3))
    A_obj = pm_opt.A_obj * mmax_vec

    if (algorithm != 'baseline' and algorithm != 'mutual_coherence' and algorithm != 'GPMO_ArbVec') and 'dipole_grid_xyz' not in kwargs:
        raise ValueError('GPMO variants require dipole_grid_xyz to be defined.')

    # Set the L2 regularization if it is included in the kwargs
    reg_l2 = kwargs.pop("reg_l2", 0.0)

    # check that algorithm can generate K binary dipoles if no backtracking done
    if "K" in kwargs:
        if (algorithm not in ['GPMO_Backtracking', 'GPMO', 'GPMO_py', 'GPMOmr']) and kwargs["K"] > pm_opt.ndipoles:
            warnings.warn(
                'Parameter K to GPMO algorithm is greater than the total number of dipole locations '
                ' so the algorithm will set K = the total number and proceed.')
            kwargs["K"] = pm_opt.ndipoles
        print('Number of binary dipoles to use in GPMO algorithm = ', kwargs["K"])

    if "nhistory" in kwargs and "K" in kwargs:
        if kwargs['nhistory'] > kwargs['K']:
            raise ValueError('nhistory must be less than K for the GPMO algorithm.')

    Nnorms = contig(np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1))))
    A_obj_T = contig(A_obj.T)
    b_obj_c = contig(pm_opt.b_obj)
    mmax_reg = np.sqrt(reg_l2) * mmax_vec
    pol_vec_c = contig(pm_opt.pol_vectors)
    nGridPoints = int(A_obj.shape[1] / 3)

    # --- algorithm dispatch helpers (closures over the shared locals) ------

    def _pick(keys: set[str]) -> dict:
        """Return a filtered copy of *kwargs* containing only *keys*."""
        return {k: kwargs[k] for k in keys if k in kwargs}

    _BASELINE_KEYS = {"K", "verbose", "nhistory", "single_direction"}
    _ARBVEC_KEYS = {"K", "verbose", "nhistory"}
    _BACKTRACK_KEYS = {"K", "verbose", "nhistory", "backtracking", "dipole_grid_xyz",
                       "single_direction", "Nadjacent", "max_nMagnets"}
    _ARBVEC_BT_KEYS = {"K", "verbose", "nhistory", "backtracking", "dipole_grid_xyz",
                       "Nadjacent", "thresh_angle", "max_nMagnets", "x_init"}
    _MULTI_KEYS = {"K", "verbose", "nhistory", "dipole_grid_xyz",
                   "single_direction", "Nadjacent"}

    def _run_baseline():
        ah, bh, mh, m_ = sopp.GPMO_baseline(
            A_obj=A_obj_T, b_obj=b_obj_c, mmax=mmax_reg,
            normal_norms=Nnorms, **_pick(_BASELINE_KEYS),
        )
        return ah, bh, mh, m_, None, None

    def _run_arbvec():
        ah, bh, mh, m_ = sopp.GPMO_ArbVec(
            A_obj=A_obj_T, b_obj=b_obj_c, mmax=mmax_reg,
            normal_norms=Nnorms, pol_vectors=pol_vec_c,
            **_pick(_ARBVEC_KEYS),
        )
        return ah, bh, mh, m_, None, None

    def _run_backtracking():
        ah, bh, mh, nn, m_ = sopp.GPMO_backtracking(
            A_obj=A_obj_T, b_obj=b_obj_c, mmax=mmax_reg,
            normal_norms=Nnorms, **_pick(_BACKTRACK_KEYS),
        )
        return ah, bh, mh, m_, nn, None

    def _run_gpmo():
        _require_cartesian(pm_opt, 'GPMO')
        _prepare_x_init(kwargs, nGridPoints, mmax_vec, pm_opt.ndipoles)
        ah, bh, mh, nn, m_ = sopp.GPMO_ArbVec_backtracking(
            A_obj=A_obj_T, b_obj=b_obj_c, mmax=mmax_reg,
            normal_norms=Nnorms, pol_vectors=pol_vec_c,
            **_pick(_ARBVEC_BT_KEYS),
        )
        return ah, bh, mh, m_, nn, None

    def _run_gpmo_py():
        _require_cartesian(pm_opt, 'GPMO_py')
        _prepare_x_init(kwargs, nGridPoints, mmax_vec, pm_opt.ndipoles)
        py_params = GPMOBacktrackParams(
            K=kwargs["K"],
            nhistory=kwargs["nhistory"],
            verbose=kwargs.get("verbose", True),
            backtracking=kwargs.get("backtracking", 0),
            dipole_grid_xyz=kwargs["dipole_grid_xyz"],
            Nadjacent=kwargs.get("Nadjacent", 7),
            thresh_angle=kwargs.get("thresh_angle", np.pi),
            max_nMagnets=kwargs.get("max_nMagnets", 5000),
            m_init=kwargs.get("x_init", np.zeros((pm_opt.ndipoles, 3))),
        )
        ah, bh, mh, nn, m_ = GPMO_py(
            A_obj=A_obj_T, b_obj=b_obj_c, mmax=mmax_reg,
            normal_norms=Nnorms, pol_vectors=pol_vec_c, params=py_params,
        )
        return ah, bh, mh, m_, nn, None

    def _run_gpmomr():
        _require_cartesian(pm_opt, 'GPMOmr')
        _prepare_x_init(kwargs, nGridPoints, mmax_vec, pm_opt.ndipoles)
        mm_params = GPMOMacroMagParams(
            K=kwargs["K"],
            nhistory=kwargs["nhistory"],
            verbose=kwargs.get("verbose", True),
            backtracking=kwargs.get("backtracking", 0),
            dipole_grid_xyz=kwargs["dipole_grid_xyz"],
            Nadjacent=kwargs.get("Nadjacent", 7),
            thresh_angle=kwargs.get("thresh_angle", np.pi),
            max_nMagnets=kwargs.get("max_nMagnets", 5000),
            m_init=kwargs.get("x_init", np.zeros((pm_opt.ndipoles, 3))),
            cube_dim=kwargs.pop("cube_dim", 0.004),
            mu_ea=kwargs.pop("mu_ea", 1.0),
            mu_oa=kwargs.pop("mu_oa", 1.0),
            use_coils=kwargs.pop("use_coils", False),
            use_demag=kwargs.pop("use_demag", True),
            coil_path=kwargs.pop("coil_path", None),
            mm_refine_every=kwargs.pop("mm_refine_every", 20),
            current_scale=kwargs.pop("current_scale", 1),
        )
        ah, bh, mh, nn, m_, kh = GPMOmr(
            A_obj=A_obj_T, b_obj=b_obj_c, mmax=mmax_vec,
            normal_norms=Nnorms, pol_vectors=pol_vec_c, params=mm_params,
        )
        return ah, bh, mh, m_, nn, kh

    def _run_multi():
        ah, bh, mh, m_ = sopp.GPMO_multi(
            A_obj=A_obj_T, b_obj=b_obj_c, mmax=mmax_reg,
            normal_norms=Nnorms, **_pick(_MULTI_KEYS),
        )
        return ah, bh, mh, m_, None, None

    _GPMO_REGISTRY: dict[str, callable] = {
        'baseline': _run_baseline,
        'GPMO_ArbVec': _run_arbvec,
        'GPMO_Backtracking': _run_backtracking,
        'GPMO': _run_gpmo,
        'GPMO_py': _run_gpmo_py,
        'GPMOmr': _run_gpmomr,
        'multi': _run_multi,
    }

    if algorithm not in _GPMO_REGISTRY:
        raise NotImplementedError(
            f'Algorithm {algorithm!r} is not recognised. '
            f'Choose from: {", ".join(_GPMO_REGISTRY)}'
        )

    algorithm_history, Bn_history, m_history, m, num_nonzeros, k_history = (
        _GPMO_REGISTRY[algorithm]()
    )

    # rescale m and m_history
    m = m * (mmax_vec.reshape(pm_opt.ndipoles, 3))
    print('Number of binary dipoles returned by GPMO algorithm = ',
          np.count_nonzero(np.sum(m, axis=-1)))

    # rescale the m that have been saved every Nhistory iterations
    for i in range(m_history.shape[-1]):
        m_history[:, :, i] = m_history[:, :, i] * (mmax_vec.reshape(pm_opt.ndipoles, 3))

    # History arrays are allocated with trailing zeros. Use the objective history
    # as the authoritative mask so the returned arrays (and num_nonzeros, when
    # available) stay aligned even if some entries could be exactly 0.
    hist_mask = algorithm_history != 0
    errors = algorithm_history[hist_mask]
    Bn_errors = Bn_history[hist_mask]
    if num_nonzeros is not None:
        pm_opt.num_nonzeros = num_nonzeros[hist_mask]

    if k_history is not None:
        pm_opt.k_history = np.asarray(k_history[hist_mask], dtype=int)

    if k_history is None and "K" in kwargs and "nhistory" in kwargs and errors.size > 0:
        K = int(kwargs["K"])
        nhistory = int(kwargs["nhistory"])
        record_every = max(1, K // nhistory)
        pm_opt.record_every = record_every

        is_macromag_loop_1_based = algorithm == "GPMOmr"
        has_init_snapshot = algorithm in ("GPMO", "GPMO_py", "GPMOmr")

        k_hist = []
        if has_init_snapshot:
            k_hist.append(0)

        if is_macromag_loop_1_based:
            # Records at k in [1..K] plus an init snapshot at 0.
            for k in range(1, K + 1):
                if (k % record_every) == 0 or k == K:
                    k_hist.append(k)
        else:
            # Loop index k_loop in [0..K-1] -> k = k_loop + 1.
            for k_loop in range(K):
                if (k_loop % record_every) == 0 or k_loop == 0 or k_loop == K - 1:
                    k_hist.append(k_loop + 1)

        # Truncate to the returned number of recorded entries.
        pm_opt.k_history = np.asarray(k_hist[: errors.size], dtype=int)

    # note m = m_proxy for GPMO because this is not using relax-and-split
    pm_opt.m = np.ravel(m)
    pm_opt.m_proxy = pm_opt.m
    return errors, Bn_errors, m_history


def _connectivity_matrix_py(
    dipole_grid_xyz: np.ndarray,
    Nadjacent: int,
) -> np.ndarray:
    """
    Build adjacency matrix for backtracking: nearest neighbors per dipole.

    For each dipole j, returns the indices of the Nadjacent closest dipoles
    (including j itself), ordered by distance. Used by GPMO backtracking to
    identify adjacent magnet pairs for angle-based removal.

    Args:
        dipole_grid_xyz: Dipole positions, shape (N, 3).
        Nadjacent: Maximum number of neighbors per dipole (including self).

    Returns:
        Index array, shape (N, min(Nadjacent, N)).
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


def _print_GPMO_py(
    k: int,
    ngrid: int,
    print_iter_ref: list,
    x: np.ndarray,
    Aij_mj_vec: np.ndarray,
    objective_history: np.ndarray,
    Bn_history: np.ndarray,
    m_history: np.ndarray,
    mmax_sum: float,
    normal_norms: np.ndarray,
) -> None:
    """
    Record GPMO iteration state and optionally print progress.

    Computes R² = 0.5|Am - b|² and mean |Bn|, stores them in the history
    arrays, and advances the print index.
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


def _initialize_GPMO_ArbVec_py(
    x_init: np.ndarray,
    pol_vectors: np.ndarray,
    x: np.ndarray,
    x_vec: np.ndarray,
    x_sign: np.ndarray,
    A_obj: np.ndarray,
    Aij_mj_sum: np.ndarray,
    R2s: np.ndarray,
    Gamma_complement: np.ndarray,
    num_nonzero_ref: list,
) -> None:
    """
    Initialize GPMO state from an initial dipole configuration.

    Maps each nonzero dipole in x_init to the nearest allowable polarization
    vector (±pol_vectors or zero). Updates the running residual Aij_mj_sum,
    marks placed sites in Gamma_complement, and masks unavailable choices in R2s.

    Args:
        x_init: Initial dipole moments, shape (N, 3).
        pol_vectors: Allowable polarizations per site, shape (N, nPolVecs, 3).
        x: Output dipole state, shape (N, 3); modified in-place.
        x_vec: Output pol-vector index per site, shape (N,); modified in-place.
        x_sign: Output sign per site (±1 or 0), shape (N,); modified in-place.
        A_obj: Objective matrix, shape (N, 3, ngrid).
        Aij_mj_sum: Running residual (Am - b), shape (ngrid,); modified in-place.
        R2s: R² values for each (site, pol, sign); unavailable entries set to ``_UNAVAILABLE_R2``.
        Gamma_complement: True where site is still available; modified in-place.
        num_nonzero_ref: Single-element list holding the nonzero magnet count.
    """
    # Shapes and views
    N = x.shape[0]
    nPolVecs = pol_vectors.shape[1]
    NNp = N * nPolVecs

    n_initialized = 0
    n_OutOfTol = 0
    tol = _INIT_TOL_FACTOR

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
            R2s[j*nPolVecs:(j+1)*nPolVecs] = _UNAVAILABLE_R2
            R2s[NNp + j*nPolVecs:NNp + (j+1)*nPolVecs] = _UNAVAILABLE_R2

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

def GPMO_py(
    A_obj: np.ndarray,
    b_obj: np.ndarray,
    mmax: np.ndarray,
    normal_norms: np.ndarray,
    pol_vectors: np.ndarray,
    params: GPMOBacktrackParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Pure-Python reference implementation of the ArbVec+backtracking GPMO algorithm.

    This routine mirrors the C++/simsoptpp backend
    (``sopp.GPMO_ArbVec_backtracking``) and exists for validation and testing.
    It is not intended for production-scale runs.

    The greedy objective is the squared surface-flux residual

    .. math::

        f_B = \tfrac{1}{2}\,\|A\,\mathbf{m} - \mathbf{b}\|_2^2,

    where :math:`A` is the sensitivity matrix mapping dipole moments to
    normal-field values and :math:`\mathbf{b}` is the target field.

    At each iteration the greedy step selects the (site, polarization, sign)
    triple that most reduces :math:`f_B`.  Every *backtracking*
    iterations, adjacent dipole pairs whose mutual angle exceeds
    *thresh_angle* are removed and may be re-placed later.

    Args:
        A_obj: Sensitivity matrix, shape ``(3*N, ngrid)``, where *N* is the
            number of dipole sites and *ngrid* the number of surface
            evaluation points.  Pre-multiplied by ``mmax_vec``.
        b_obj: Target normal-field vector, shape ``(ngrid,)``.
        mmax: Per-component maximum dipole strengths, shape ``(3*N,)``.
            Used as an L2 penalty in the greedy selection criterion.
        normal_norms: Surface-normal magnitudes at each grid point,
            shape ``(ngrid,)``.  Used to convert squared residuals into
            :math:`|B_n|` error.
        pol_vectors: Allowable polarization directions per site,
            shape ``(N, nPolVecs, 3)``.  Each row is a unit vector.
        params: Backtracking parameters including ``K``, ``nhistory``,
            ``verbose``, ``backtracking``, ``dipole_grid_xyz``,
            ``Nadjacent``, ``thresh_angle``, ``max_nMagnets``, and
            ``m_init`` (used as ``x_init``).

    Returns:
        A 5-tuple ``(objective_history, Bn_history, m_history,
        num_nonzeros, x)`` where

        - **objective_history** — ``0.5 * ||Am - b||^2`` at each snapshot,
          shape ``(nhistory + 2,)``.
        - **Bn_history** — ``|B_n|`` error at each snapshot,
          shape ``(nhistory + 2,)``.
        - **m_history** — dipole solution snapshots,
          shape ``(N, 3, nhistory + 2)``.
        - **num_nonzeros** — number of placed magnets at each snapshot,
          shape ``(nhistory + 2,)``.
        - **x** — final dipole solution, shape ``(N, 3)``.
    """
    K = params.K
    verbose = params.verbose
    nhistory = params.nhistory
    backtracking = params.backtracking
    Nadjacent = params.Nadjacent
    thresh_angle = params.thresh_angle
    max_nMagnets = params.max_nMagnets
    x_init = params.m_init
    dipole_grid_xyz = params.dipole_grid_xyz

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
    R2s = np.full((2 * NNp,), _UNAVAILABLE_R2, dtype=np.float64)

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
        R2s[skj*nPolVecs:(skj+1)*nPolVecs] = _UNAVAILABLE_R2
        R2s[NNp + skj*nPolVecs:NNp + (skj+1)*nPolVecs] = _UNAVAILABLE_R2

        num_nonzero_ref[0] += 1

        # Backtracking step
        if backtracking != 0 and (k % backtracking) == 0:
            _backtracking_step(
                N, Connect, cos_thresh_angle,
                x, x_vec, x_sign, Gamma_complement,
                pol_vectors, A_obj, Aij_mj_sum, num_nonzero_ref,
                verbose=True,
            )

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


def _expand3(idx_1d: np.ndarray) -> np.ndarray:
    """Expand tile indices to flattened ``(3*N)`` indices: ``i -> [3i, 3i+1, 3i+2]``."""
    base = (3 * idx_1d).reshape(-1, 1)
    return (base + np.array([0, 1, 2])).reshape(-1)


# ---------------------------------------------------------------------------
# Shared helpers for GPMO variants
# ---------------------------------------------------------------------------

def _require_cartesian(pm_opt: PermanentMagnetGrid, algorithm: str) -> None:
    """Raise if the dipole grid is not in the Cartesian basis."""
    if pm_opt.coordinate_flag != 'cartesian':
        raise ValueError(
            f'{algorithm} algorithm currently only supports dipole grids '
            'with moment vectors in the Cartesian basis.'
        )


def _prepare_x_init(
    kwargs: dict,
    nGridPoints: int,
    mmax_vec: np.ndarray,
    ndipoles: int,
) -> None:
    """Validate ``m_init`` kwarg and convert it to a normalised ``x_init``.

    Modifies *kwargs* in-place: pops ``m_init`` and inserts ``x_init``.
    """
    contig = np.ascontiguousarray
    if "m_init" in kwargs:
        m_init = kwargs["m_init"]
        if m_init.shape[0] != nGridPoints:
            raise ValueError(
                'Initialization vector `m_init` must have '
                'as many rows as there are dipoles in the grid'
            )
        if m_init.shape[1] != 3:
            raise ValueError(
                'Initialization vector `m_init` must have three columns'
            )
        kwargs["x_init"] = contig(m_init / mmax_vec.reshape(ndipoles, 3))
        kwargs.pop("m_init")
    else:
        kwargs["x_init"] = contig(np.zeros((nGridPoints, 3)))


def _backtracking_step(
    N: int,
    Connect: np.ndarray,
    cos_thresh_angle: float,
    x: np.ndarray,
    x_vec: np.ndarray,
    x_sign: np.ndarray,
    Gamma_complement: np.ndarray,
    pol_vectors: np.ndarray,
    A_obj_3x: np.ndarray,
    Aij_mj_sum: np.ndarray,
    num_nonzero_ref: list,
    gamma_inds: list | None = None,
    verbose: bool = False,
) -> int:
    """Remove adjacent dipole pairs whose mutual angle exceeds the threshold.

    Returns the number of pairs removed.
    """
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
        if cj_min != -1 and min_cos <= cos_thresh_angle:
            pv_j = pol_vectors[j, x_vec[j], :]
            pv_c = pol_vectors[cj_min, x_vec[cj_min], :]
            Aj = A_obj_3x[j]
            Ac = A_obj_3x[cj_min]
            Aij_mj_sum -= (
                float(x_sign[j]) * (pv_j[:, None] * Aj).sum(axis=0)
                + float(x_sign[cj_min]) * (pv_c[:, None] * Ac).sum(axis=0)
            )
            x[j, :] = 0.0;       x[cj_min, :] = 0.0
            x_vec[j] = 0;         x_vec[cj_min] = 0
            x_sign[j] = 0;        x_sign[cj_min] = 0
            Gamma_complement[j] = True
            Gamma_complement[cj_min] = True
            if gamma_inds is not None:
                gamma_inds.remove(j)
                gamma_inds.remove(cj_min)
            num_nonzero_ref[0] -= 2
            removed += 1
    if verbose:
        print(f"Backtracking: {removed} pairs removed")
    return removed


def GPMOmr(
    A_obj: np.ndarray,
    b_obj: np.ndarray,
    mmax: np.ndarray,
    normal_norms: np.ndarray,
    pol_vectors: np.ndarray,
    params: GPMOMacroMagParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    GPMO with macromagnetic refinement (GPMOmr).

    Extends :func:`GPMO_py` by periodically solving the full
    macromagnetic equilibrium problem via :class:`MacroMag`, as
    described in `Ulrich, Haberle & Kaptanoglu (2025)
    <https://arxiv.org/abs/2512.14997>`_.  After each greedy
    placement/backtracking cycle, the solver updates dipole moments
    to account for finite-permeability coupling through the global
    linear system

    .. math::

        \bigl[\delta_{ij}\,\mathbf{I}_3
        + \boldsymbol{\chi}_i\,\underline{\underline{N}}_{ij}\bigr]
        \,\mathbf{M}_j
        = M_{\text{rem}}\,\hat{\mathbf{u}}_i
        + \boldsymbol{\chi}_i\,\mathbf{H}_a,

    where :math:`\underline{\underline{N}}_{ij}` is the demagnetization
    tensor and :math:`\boldsymbol{\chi}_i` the anisotropic susceptibility
    tensor of tile *i*.  The corrected magnetisations are then
    substituted back into the greedy surface-flux objective

    .. math::

        f_B = \tfrac{1}{2}\,\|A\,\mathbf{m} - \mathbf{b}\|_2^2.

    Args:
        A_obj: Sensitivity matrix pre-multiplied by ``mmax_vec``,
            shape ``(3*N, ngrid)``.
        b_obj: Target normal-field vector, shape ``(ngrid,)``.
        mmax: Per-component maximum dipole strengths, shape ``(3*N,)``.
        normal_norms: Surface-normal magnitudes at each grid point,
            shape ``(ngrid,)``.
        pol_vectors: Allowable polarization directions per site,
            shape ``(N, nPolVecs, 3)``.
        params: :class:`GPMOMacroMagParams` containing all algorithm,
            backtracking, and macromagnetic-refinement parameters.

    Returns:
        A 6-tuple ``(objective_history, Bn_history, m_history,
        num_nonzeros, x, k_history)`` with the same semantics as
        :func:`GPMO_py` plus *k_history*.

    Raises:
        ValueError: If *use_coils* is True but *coil_path* is None.
    """
    K = params.K
    verbose = params.verbose
    nhistory = params.nhistory
    backtracking = params.backtracking
    Nadjacent = params.Nadjacent
    thresh_angle = params.thresh_angle
    max_nMagnets = params.max_nMagnets
    x_init = params.m_init
    dipole_grid_xyz = params.dipole_grid_xyz
    cube_dim = params.cube_dim
    mu_ea = params.mu_ea
    mu_oa = params.mu_oa
    use_coils = params.use_coils
    use_demag = params.use_demag
    coil_path = params.coil_path
    mm_refine_every = params.mm_refine_every
    current_scale = params.current_scale

    if use_coils and coil_path is None:
        raise ValueError("Use coils set to True but missing coil path; Please provide the correct focus coil path")

    print("use_coils mode =",use_coils)
    print("use_demag mode =",use_demag)

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
    C_gram = np.einsum('jkg,jlg->jkl', A_obj_3x, A_obj_3x, optimize=True)
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
    k_history = np.zeros((nhistory + 2,), dtype=np.int32)
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
        mac_all.load_coils(coil_path, current_scale=current_scale)  # sets mac_all._bs_coil

    # Precompute demagnetization tensor
    # Sign convention: assemble_blocks_subset returns -N (since H_d = -N M).
    # Flipping sign is applied at each incremental assembly step.
    # Note: rotated magnets would require revisiting the local-vs-global
    # tensor convention here.

    # Precompute coil field if requested
    Hcoil_all = np.zeros((N, 3))
    if use_coils:
        Hcoil_all = mac_all.coil_field_at(tiles_all.offset).astype(np.float64)

    def solve_subset_and_score(
        active_idx: np.ndarray,
        ea_list: np.ndarray,
        *,
        prev_active_idx: np.ndarray | None = None,
        A_prev: np.ndarray | None = None,
        prev_n: int = 0,
    ):
        """
        Solve the subset of active tiles via MacroMag and score the residual.

        Builds a MacroMag instance for the active tiles, assembles the demagnetization
        blocks (N_new_rows, N_new_cols, N_new_diag) for incremental assembly when
        prev_n > 0, calls direct_solve to compute the coupled magnetization, then
        evaluates the squared flux residual 0.5 * ||A @ x_macro - b||^2.

        Args:
            active_idx: ndarray, shape (n_active,). Indices of the active tiles in the
                full grid. Must be ordered so that the first prev_n entries match
                prev_active_idx (for incremental assembly).
            ea_list: ndarray, shape (n_active, 3). Easy-axis vectors for each active
                tile (used for magnetization direction).
            prev_active_idx: ndarray, shape (n_prev,), optional. Indices of the previous
                active set (survivors). Required when prev_n > 0 for incremental N-block
                assembly.
            A_prev: ndarray, shape (3*n_prev, 3*n_prev), optional. Dense system matrix
                for the previous survivor set in flattened (3*N) indexing. Used by
                direct_solve for incremental assembly.
            prev_n: int, optional. Number of previous survivors.

        Returns:
            R2: float. Squared flux residual 0.5 * ||A @ x_macro - b||^2.
            x_macro_flat: ndarray, shape (3*N,). Macro magnetization vector for the
                full grid, normalized by mmax.
            m_sub_flat: ndarray, shape (3*n_active,). Macro magnetization for the
                active tiles only (raveled).
            res: ndarray, shape (ngrid,). Residual vector A.T @ x_macro_flat - b_obj.
            A_sub: ndarray, shape (3*n_active, 3*n_active), or None. Dense system
                matrix for the active set (for caching in next iteration).
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
        M_rem_inferred = mmax[active_idx] / vol
        sub.M_rem = M_rem_inferred

        # Set easy axes for all active tiles (vectorized normalization)
        # Avoid triggering the u_ea setter which calls M setter for each tile
        ea_norms = np.linalg.norm(ea_list, axis=1, keepdims=True)
        ea_normalized = ea_list / (ea_norms + _NORMALIZATION_EPS)
        sub._u_ea = ea_normalized  # Direct assignment to avoid setter overhead

        mac = MacroMag(sub, bs_interp=mac_all.bs_interp) #Taking already built interpolanet field to avod reloading...

        n_prev = prev_active_idx.size if prev_active_idx is not None else 0
        n_active = active_idx.size

        # Add new rows and columns from newly placed magnets
        t0_n_construction = time.time()
        if n_prev > 0:
            new_indices = active_idx[n_prev:]
            old_indices = active_idx[:n_prev]
            # assemble_blocks_subset returns -N (because H_d = -N M); flip sign to get +N
            N_new_rows = -assemble_blocks_subset(
                mac_all.centres, mac_all.half, mac_all.Rg2l,
                new_indices.astype(np.int64), old_indices.astype(np.int64)
            )
            N_new_cols = -assemble_blocks_subset(
                mac_all.centres, mac_all.half, mac_all.Rg2l,
                old_indices.astype(np.int64), new_indices.astype(np.int64)
            )
            N_new_diag = -assemble_blocks_subset(
                mac_all.centres, mac_all.half, mac_all.Rg2l,
                new_indices.astype(np.int64), new_indices.astype(np.int64)
            )
        else:
            # initial pass: only rows are needed by direct_solve(prev_n == 0)
            N_new_rows = -assemble_blocks_subset(
                mac_all.centres, mac_all.half, mac_all.Rg2l,
                active_idx.astype(np.int64), active_idx.astype(np.int64)
            )
            N_new_cols = None
            N_new_diag = None

        t1_n_construction = time.time()
        print(f"Iteration {k}: Time in N block assembly: {t1_n_construction - t0_n_construction} seconds")

        # Perform the magtense coupling direct solve
        params = SolverParams(use_coils=use_coils, H_a_override=Hcoil_all[active_idx])
        mac, A_sub = mac.direct_solve(
            N_new_rows=N_new_rows,
            N_new_cols=N_new_cols,
            N_new_diag=N_new_diag,
            A_prev=A_prev,
            prev_n=prev_n,
            params=params,
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

    x_macro_flat = x_macro_flat_0
    R2_snap = R2_0
    A_sub_prev = A_sub_0
    prev_active_indices = active0
    prev_n = active0.size

    # record init
    idx0 = print_iter_ref[0]
    m_history[:, :, idx0] = x_macro_flat_0.reshape(N, 3)
    objective_history[idx0] = R2_0
    Bn_history[idx0] = float(np.sum(np.abs(res0) * np.sqrt(normal_norms))) / np.sqrt(float(ngrid))
    k_history[idx0] = 0
    num_nonzeros[idx0] = num_nonzero_ref[0]
    if verbose:
        print(f"0 ... {R2_0:.2e} ... 0.00e+00 ")
    print_iter_ref[0] += 1

    # Running residual for ArbVec scoring (classical)
    Aij_mj_sum = (A_obj.T @ x_macro_flat_0) - b_obj
    R2s = np.full((2 * N * nPolVecs,), _UNAVAILABLE_R2, dtype=np.float64)

    # Initialize from x_init (may place some dipoles)
    _initialize_GPMO_ArbVec_py(
        x_init, pol_vectors, x, x_vec, x_sign,
        A_obj_3x, Aij_mj_sum, R2s, Gamma_complement, num_nonzero_ref
    )

    # Main greedy loop
    t_gpmo = time.time()

    def _prepare_cached_macromag_call(active_idx: np.ndarray):
        """
        Prepare arguments for incremental MacroMag assembly.

        When the active set changes (new magnets added, or backtracking removes some),
        the incremental path in solve_subset_and_score requires:
        (1) active_ordered: [survivors in old order] + [new magnets]
        (2) A_prev_common: dense block for survivors only, aligned to that order
        (3) prev_active_for_call, prev_n_for_call: survivor indices and count

        This function computes the intersection of the current active set with the
        previous one, orders survivors by their position in the previous list, and
        extracts the corresponding block from A_sub_prev.

        Args:
            active_idx: ndarray, shape (n_active,). Current active tile indices
                (from gamma_inds).

        Returns:
            active_ordered: ndarray, shape (n_active,). Active indices ordered as
                [survivors in prev order] + [new].
            ea_list: ndarray, shape (n_active, 3). Easy-axis vectors for active_ordered.
            A_prev_common: ndarray, shape (3*n_prev, 3*n_prev), or None. Dense block
                for survivors; None if no previous solve.
            prev_active_for_call: ndarray, shape (n_prev,), or None. Survivor indices
                in old order.
            prev_n_for_call: int. Number of survivors.
        """
        active_idx = np.asarray(active_idx, dtype=np.int64).ravel()
        if active_idx.size == 0:
            return (
                active_idx,
                np.zeros((0, 3), dtype=np.float64),
                None,
                None,
                0,
            )

        if prev_active_indices is not None and A_sub_prev is not None and prev_n > 0:
            # intersection in "old" order to keep A_prev alignment
            common = np.intersect1d(prev_active_indices, active_idx, assume_unique=True)

            # indices of 'common' in the old order (prev_active_indices)
            old_pos = {idx: i for i, idx in enumerate(prev_active_indices)}
            common_in_prev_order = np.array(sorted(common, key=lambda v: old_pos[v]), dtype=np.int64)

            # anything new (not in 'common')
            is_new = ~np.isin(active_idx, common_in_prev_order)
            new_only = active_idx[is_new]

            # assemble the order: [survivors in old order] + [new ones]
            active_ordered = (
                np.concatenate([common_in_prev_order, new_only]) if new_only.size else common_in_prev_order
            )

            # positions of survivors inside the previous active list
            iprev = np.array([old_pos[v] for v in common_in_prev_order], dtype=np.int64)
            iprev3 = _expand3(iprev)

            # A_prev_common is the cached block for the survivors (aligned to their old order)
            A_prev_common = A_sub_prev[np.ix_(iprev3, iprev3)]
            prev_active_for_call = common_in_prev_order
            prev_n_for_call = len(common_in_prev_order)

            # easy axis list in the same order we pass to MacroMag
            ea_list = x[active_ordered]
            return active_ordered, ea_list, A_prev_common, prev_active_for_call, prev_n_for_call

        active_ordered = active_idx
        ea_list = x[active_ordered]
        return active_ordered, ea_list, None, None, 0

    for k in range(1, K + 1):

        # Stop if filled or hit magnet cap
        # t1_magtens = time.time()
        if (num_nonzero_ref[0] >= N) or (num_nonzero_ref[0] >= max_nMagnets):
            # record final snapshot (ALWAYS on early stop, like reference)
            # IMPORTANT: keep the active ordering aligned with the cached A_sub_prev block,
            # otherwise the incremental assembly uses the wrong "old" ordering and produces
            # a spurious final objective (often seen as a jump in f_B near termination).
            active = np.array(gamma_inds, dtype=np.int64)
            active_ordered, ea_list, A_prev_common, prev_active_for_call, prev_n_for_call = _prepare_cached_macromag_call(active)

            # One final solve before commiting set
            R2_snap, x_macro_flat, _, res, _ = solve_subset_and_score(
                active_ordered,
                ea_list,
                prev_active_idx=prev_active_for_call,
                A_prev=A_prev_common,
                prev_n=prev_n_for_call,
            )

            last_x_macro_flat[:] = x_macro_flat
            idx = print_iter_ref[0]
            m_history[:, :, idx] = x_macro_flat.reshape(N, 3)
            objective_history[idx] = R2_snap
            Bn_history[idx] = float(np.sum(np.abs(res) * np.sqrt(normal_norms))) / np.sqrt(float(ngrid))
            k_history[idx] = int(k)
            num_nonzeros[idx] = num_nonzero_ref[0]
            if verbose:
                print(f"{k} ... {R2_snap:.2e} ... 0.00e+00 ")
                print(f"Iteration = {k}, Number of nonzero dipoles = {num_nonzero_ref[0]}")
            print_iter_ref[0] += 1
            print("Stopping iterations: maximum number of nonzero magnets reached " if (num_nonzero_ref[0] >= max_nMagnets) else "Stopping iterations: all dipoles in grid are populated")
            break

        NNp = N * nPolVecs
        # Vectorized version of the loop over j
        # Initialize R2s with large values for occupied positions
        R2s[:] = _UNAVAILABLE_R2

        # Find active (unoccupied) positions
        active_mask = Gamma_complement
        active_indices = np.where(active_mask)[0]

        if len(active_indices) > 0:
            A_active   = A_obj_3x[active_indices]              # (n_active, 3, ngrid)
            pol_active = pol_vectors[active_indices]           # (n_active, nPolVecs, 3)
            mmax_active = mmax[active_indices]                 # (n_active,)
            C_active   = C_gram[active_indices]                # (n_active, 3, 3)

            r = Aij_mj_sum                                     # (ngrid,)
            r2 = float(r @ r)                                  # ||r||^2 scalar

            # d_jk = <A_jk, r> for k in {x,y,z}, shape (n_active,3)
            d = np.einsum('jkg,g->jk', A_active, r, optimize=True)

            # s = r·(A p) = p·d, shape (n_active, nPolVecs)
            s = np.einsum('jpk,jk->jp', pol_active, d, optimize=True)

            # q = ||A p||^2 = p^T C p, shape (n_active, nPolVecs)
            q = np.einsum('jpk,jkl,jpl->jp', pol_active, C_active, pol_active, optimize=True)

            # R2_plus  = ||r + A p||^2 = r2 + 2*s + q
            # R2_minus = ||r - A p||^2 = r2 - 2*s + q
            R2_plus  = r2 + 2.0*s + q + (mmax_active**2)[:, None]
            R2_minus = r2 - 2.0*s + q + (mmax_active**2)[:, None]

            # Scatter into flat R2s
            base = (active_indices * nPolVecs)[:, None] + np.arange(nPolVecs)[None, :]
            pos_indices = base.ravel()
            neg_indices = (NNp + base).ravel()

            R2s[pos_indices] = R2_plus.ravel()
            R2s[neg_indices] = R2_minus.ravel()


        # Best (site, pol, sign)
        skj = int(np.argmin(R2s))
        sign_fac = -1.0 if skj >= NNp else +1.0
        if skj >= NNp: skj -= NNp
        skjj = int(skj % nPolVecs)
        j_best = int(skj // nPolVecs)

        # Commit in classical state
        pv = pol_vectors[j_best, skjj, :]
        x[j_best, :] = sign_fac * pv / (np.linalg.norm(pv) + _NORMALIZATION_EPS)
        x_vec[j_best] = skjj
        x_sign[j_best] = int(np.sign(sign_fac))
        Gamma_complement[j_best] = False
        gamma_inds.append(j_best)
        num_nonzero_ref[0] += 1
        Aij_mj_sum += sign_fac * (pv[:, None] * A_obj_3x[j_best]).sum(axis=0)
        base = j_best * nPolVecs
        R2s[base:base + nPolVecs] = _UNAVAILABLE_R2
        R2s[NNp + base:NNp + base + nPolVecs] = _UNAVAILABLE_R2

        # Backtracking
        if backtracking != 0 and (k % backtracking) == 0:
            _backtracking_step(
                N, Connect, cos_thresh_angle,
                x, x_vec, x_sign, Gamma_complement,
                pol_vectors, A_obj_3x, Aij_mj_sum, num_nonzero_ref,
                gamma_inds=gamma_inds, verbose=verbose,
            )

        # Probably only need to run MacroMag for the committed set ONLY every couple dozen iterations
        if (k % mm_refine_every) == 0:
            t1_macromag = time.time()
            active = np.array(gamma_inds, dtype=np.int64)
            active_ordered, ea_list, A_prev_common, prev_active_for_call, prev_n_for_call = _prepare_cached_macromag_call(active)

            R2_snap, x_macro_flat, _, Aij_mj_sum, A_sub = solve_subset_and_score(
                active_ordered, ea_list,
                prev_active_idx=prev_active_for_call,
                A_prev=A_prev_common,
                prev_n=prev_n_for_call
            )


            # update caches for next iteration
            A_sub_prev = A_sub
            prev_active_indices = active_ordered
            prev_n = len(active_ordered)

            t2_macromag = time.time()
            print(f"Iteration {k}: Time in macromag call: {t2_macromag - t1_macromag} seconds")

        # Store current active indices for next iteration
        # prev_active_indices = active.copy()

        if verbose and (((k % max(1, K // nhistory)) == 0) or (k == K)):
            idx = print_iter_ref[0]
            m_history[:, :, idx] = x_macro_flat.reshape(N, 3)
            objective_history[idx] = R2_snap
            Bn_history[idx] = float(np.sum(np.abs(Aij_mj_sum) * np.sqrt(normal_norms))) / np.sqrt(float(ngrid))
            k_history[idx] = int(k)
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

    t_gpmo = time.time() - t_gpmo
    print(f"Time taken for GPMO: {t_gpmo} seconds")

    return objective_history, Bn_history, m_history, num_nonzeros, x_macro_flat.reshape(N, 3), k_history
