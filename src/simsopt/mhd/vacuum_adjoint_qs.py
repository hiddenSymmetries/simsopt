# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
Adjoint-based quasisymmetry (QS) optimization on a single flux surface.

Implements the method of Nies et al. 2022 (J. Plasma Phys., arXiv:2108.11433):
given a forward SPEC vacuum-field solve, an analytic adjoint chain gives O(1)-cost
gradients of QS and iota-target objectives with respect to boundary Fourier coefficients.

Three PDEs are solved sequentially:
  1. Straight-field-line (SFL) equation  → iota, λ
  2. q_α adjoint equation                → q_α
  3. Adjoint SPEC solve (Lconstraint=-2) → q_ω
Then the shape gradient G(θ,φ) is assembled and projected onto boundary DOFs.
"""

import logging
import os
import shutil

import numpy as np

from .._core.optimizable import Optimizable
from ..geo.surfaceobjectives import parameter_derivatives

logger = logging.getLogger(__name__)

try:
    import py_spec
    from py_spec import SPECout, SPECNamelist
except ImportError:
    py_spec = None
    SPECout = None
    SPECNamelist = None
    logger.debug("py_spec not available; VacuumAdjointQS requires it")

__all__ = ['VacuumAdjointQS']


# ---------------------------------------------------------------------------
# Fourier basis helpers (ported from adjoint_QS/source/helper_Fourier.py)
# ---------------------------------------------------------------------------

def _sine_modes(mpol, ntor):
    """Return (nmodes, xm, xn) for sine modes with m·θ − n·φ convention."""
    nmodes = ntor + mpol * (2 * ntor + 1)
    xm = np.zeros(nmodes, dtype=np.int64)
    xn = np.zeros(nmodes, dtype=np.int64)
    idx = 0
    for n in range(1, ntor + 1):
        xn[idx] = n; idx += 1
    for m in range(1, mpol + 1):
        for n in range(-ntor, ntor + 1):
            xm[idx] = m; xn[idx] = n; idx += 1
    return nmodes, xm, xn


def _cosine_modes(mpol, ntor):
    """Return (nmodes, xm, xn) for cosine modes with m·θ − n·φ convention."""
    nmodes = (ntor + 1) + mpol * (2 * ntor + 1)
    xm = np.zeros(nmodes, dtype=np.int64)
    xn = np.zeros(nmodes, dtype=np.int64)
    idx = 0
    for n in range(0, ntor + 1):
        xn[idx] = n; idx += 1
    for m in range(1, mpol + 1):
        for n in range(-ntor, ntor + 1):
            xm[idx] = m; xn[idx] = n; idx += 1
    return nmodes, xm, xn


def _basis_and_derivs(nmodes, xm, xn, symm, thetas_2d, phis_2d, nfp,
                      second_deriv=False):
    """
    Evaluate Fourier basis functions and derivatives on a 2D grid.

    Args:
        symm: ``'sin'`` or ``'cos'``
        second_deriv: if True, also return second-order derivatives

    Returns (basis, d/dθ, d/dφ[, d²/dθ², d²/dθdφ, d²/dφ²])
    Each has shape (nmodes, ntheta, nphi).
    """
    nt, np_ = thetas_2d.shape
    basis = np.zeros((nmodes, nt, np_))
    db_dt = np.zeros_like(basis)
    db_dp = np.zeros_like(basis)
    if second_deriv:
        d2b_dt2 = np.zeros_like(basis)
        d2b_dtp = np.zeros_like(basis)
        d2b_dp2 = np.zeros_like(basis)

    for i in range(nmodes):
        angle = xm[i] * thetas_2d - nfp * xn[i] * phis_2d
        m, n = xm[i], xn[i]
        if symm == 'sin':
            basis[i] = np.sin(angle)
            db_dt[i] = m * np.cos(angle)
            db_dp[i] = -n * nfp * np.cos(angle)
            if second_deriv:
                d2b_dt2[i] = -m**2 * np.sin(angle)
                d2b_dtp[i] = m * n * nfp * np.sin(angle)
                d2b_dp2[i] = -(n * nfp)**2 * np.sin(angle)
        else:  # cos
            basis[i] = np.cos(angle)
            db_dt[i] = -m * np.sin(angle)
            db_dp[i] = n * nfp * np.sin(angle)
            if second_deriv:
                d2b_dt2[i] = -m**2 * np.cos(angle)
                d2b_dtp[i] = m * n * nfp * np.cos(angle)
                d2b_dp2[i] = -(n * nfp)**2 * np.cos(angle)

    if second_deriv:
        return basis, db_dt, db_dp, d2b_dt2, d2b_dtp, d2b_dp2
    return basis, db_dt, db_dp


def _fourier_coeffs(func, basis):
    """Project ``func`` onto each basis function (normalised L² projection)."""
    n = len(basis)
    coeffs = np.zeros(n)
    for i in range(n):
        coeffs[i] = np.sum(func * basis[i]) / np.sum(basis[i] ** 2)
    return coeffs


# ---------------------------------------------------------------------------
# Straight-field-line (SFL) equation solver
# (ported from adjoint_QS/source/solver_STFL_eq.py)
# ---------------------------------------------------------------------------

def _solve_stfl(Bsuptheta, Bsupphi, mpol, ntor, thetas_2d, phis_2d, nfp):
    """
    Solve B·∇_Γ α = 0, α = θ − ι·φ + λ(θ,φ), for ι and λ.

    Uses a spectral Galerkin method with FFT-based matrix assembly.

    Returns:
        iota, lambdaSF, dlambdadtheta, dlambdadphi,
        d2lambdadtheta2, d2lambdadthetadphi, d2lambdadphi2
    """
    nmodes_test, xm_test, xn_test = _cosine_modes(mpol, ntor)
    nmodes_basis, xm_basis, xn_basis = _sine_modes(mpol, ntor)

    basis, db_dt, db_dp, d2b_dt2, d2b_dtp, d2b_dp2 = _basis_and_derivs(
        nmodes_basis, xm_basis, xn_basis, 'sin', thetas_2d, phis_2d, nfp,
        second_deriv=True)

    nmat = nmodes_test
    matrix = np.zeros((nmat, nmat))
    rhs = np.zeros(nmat)

    Bphi_mn = np.fft.fft2(Bsupphi)
    Bthe_mn = np.fft.fft2(Bsuptheta)
    Bthe_mn[0, 0] *= 2
    Bphi_mn[0, 0] *= 2

    for i in range(nmodes_test):
        m, n = xm_test[i], xn_test[i]
        matrix[i, 0] = 0.5 * np.real(Bphi_mn[m, -n] + Bphi_mn[-m, n])
        rhs[i] = -0.5 * np.real(Bthe_mn[m, -n] + Bthe_mn[-m, n])

    for j in range(nmodes_basis):
        col_mn = np.fft.fft2(Bsuptheta * db_dt[j] + Bsupphi * db_dp[j])
        col_mn[0, 0] *= 2
        for i in range(nmodes_test):
            m, n = xm_test[i], xn_test[i]
            matrix[i, j + 1] = 0.5 * np.real(col_mn[m, -n] + col_mn[-m, n])

    sol = np.linalg.solve(matrix, rhs)
    iota = -sol[0]

    lambdaSF = np.einsum('i,iab->ab', sol[1:], basis)
    dldt = np.einsum('i,iab->ab', sol[1:], db_dt)
    dldp = np.einsum('i,iab->ab', sol[1:], db_dp)
    d2ldt2 = np.einsum('i,iab->ab', sol[1:], d2b_dt2)
    d2ldtp = np.einsum('i,iab->ab', sol[1:], d2b_dtp)
    d2ldp2 = np.einsum('i,iab->ab', sol[1:], d2b_dp2)

    return iota, lambdaSF, dldt, dldp, d2ldt2, d2ldtp, d2ldp2


# ---------------------------------------------------------------------------
# q_α adjoint equation solver
# (ported from adjoint_QS/source/solver_qalpha_adjoint_eq.py)
# ---------------------------------------------------------------------------

def _solve_qalpha(fom, iota_weight, divgamma_B, iota_target,
                  Bsuptheta, Bsupphi, iota, B0,
                  nabla_alpha_supt, nabla_alpha_supp,
                  vQS, BxnormdotgradBmag, abs_nabla_psi,
                  helicity_QS, abs_nabla_alpha,
                  mpol, ntor, thetas_2d, phis_2d, nfp,
                  jac_surf, dtheta, dphi, norm_QS):
    """
    Solve the q_α adjoint magnetic differential equation.

    Returns:
        qalpha, dqalpha_dt, dqalpha_dp, error_diffeq, integral_cond
    """
    nmodes_test, xm_test, xn_test = _sine_modes(mpol, ntor)
    nmodes_basis, xm_basis, xn_basis = _cosine_modes(mpol, ntor)

    test_basis, _, _ = _basis_and_derivs(
        nmodes_test, xm_test, xn_test, 'sin', thetas_2d, phis_2d, nfp)
    basis, db_dt, db_dp = _basis_and_derivs(
        nmodes_basis, xm_basis, xn_basis, 'cos', thetas_2d, phis_2d, nfp)

    nmat = nmodes_test + 1
    matrix = np.zeros((nmat, nmat))

    # Integral-constraint row (row 0)
    ic_mn = np.fft.fft2(Bsupphi / B0 * jac_surf * dtheta * dphi * nfp)
    for j in range(nmodes_basis):
        m, n = xm_basis[j], xn_basis[j]
        matrix[0, j] = 0.5 * np.real(ic_mn[m, -n] + ic_mn[-m, n])

    # Magnetic-differential-equation rows (rows 1..nmodes_test)
    for j in range(nmodes_basis):
        integrand = (Bsuptheta * db_dt[j] + Bsupphi * db_dp[j]
                     + divgamma_B * basis[j]) * dtheta * dphi * nfp / B0
        col_mn = np.fft.fft2(integrand)
        for i in range(nmodes_test):
            m, n = xm_test[i], xn_test[i]
            matrix[1 + i, j] = 0.5 * np.real((col_mn[m, -n] - col_mn[-m, n]) * 1j)

    # Build RHS
    rhs = np.zeros(nmat)
    if fom == 'iota_target':
        rhs[0] = iota - iota_target

    elif fom == 'quasisymm':
        if helicity_QS is np.infty:
            scriptJ = (vQS * BxnormdotgradBmag / B0**2 * (abs_nabla_psi / B0)
                       * nabla_alpha_supp / abs_nabla_alpha**2 / norm_QS)
            scriptI = (vQS * BxnormdotgradBmag / B0**2 * (abs_nabla_psi / B0)
                       / abs_nabla_alpha**2 / norm_QS)
        elif helicity_QS is np.nan:
            scriptI = scriptJ = 0
        else:
            scriptJ = (vQS * BxnormdotgradBmag / B0**2 * (abs_nabla_psi / B0)
                       * (nabla_alpha_supp * (iota - helicity_QS) / abs_nabla_alpha**2 + 1)
                       / norm_QS)
            scriptI = (vQS * BxnormdotgradBmag / B0**2 * (abs_nabla_psi / B0)
                       * (iota - helicity_QS) / abs_nabla_alpha**2 / norm_QS)

        rhs[0] = -np.sum(jac_surf * scriptJ) * dtheta * dphi * nfp
        if iota_target is not None:
            rhs[0] += iota_weight * (iota - iota_target)

        # Source term for the magnetic-differential-equation rows
        arg_t = scriptI * jac_surf * nabla_alpha_supt
        arg_p = scriptI * jac_surf * nabla_alpha_supp
        nmodes_cos, xm_cos, xn_cos = _cosine_modes(mpol, ntor)
        cos_b, dcos_dt, dcos_dp = _basis_and_derivs(
            nmodes_cos, xm_cos, xn_cos, 'cos', thetas_2d, phis_2d, nfp)
        coeffs_t = _fourier_coeffs(arg_t, cos_b)
        coeffs_p = _fourier_coeffs(arg_p, cos_b)
        dt_field = np.einsum('i,iab->ab', coeffs_t, dcos_dt)
        dp_field = np.einsum('i,iab->ab', coeffs_p, dcos_dp)
        source = -(dt_field + dp_field) / jac_surf

        for i in range(nmodes_test):
            rhs[1 + i] = np.sum(source * test_basis[i]) * dtheta * dphi * nfp

    sol = np.linalg.solve(matrix, rhs)

    qalpha = np.einsum('i,iab->ab', sol, basis)
    dqa_dt = np.einsum('i,iab->ab', sol, db_dt)
    dqa_dp = np.einsum('i,iab->ab', sol, db_dp)

    divgamma_qaB = dqa_dt * Bsuptheta + dqa_dp * Bsupphi + qalpha * divgamma_B
    if fom == 'iota_target':
        integral_cond = (np.sum(qalpha * Bsupphi / B0 * dtheta * dphi * nfp * jac_surf)
                         - (iota - iota_target))
        error_diffeq = divgamma_qaB
    else:
        integral_cond = np.sum(
            (qalpha * Bsupphi / B0 + scriptJ) * dtheta * dphi * nfp * jac_surf)
        error_diffeq = divgamma_qaB / B0 - source

    return qalpha, dqa_dt, dqa_dp, error_diffeq, integral_cond


# ---------------------------------------------------------------------------
# Helper: covariant ↔ contravariant conversion
# ---------------------------------------------------------------------------

def _covariant_from_contravariant(B_contra, gsub):
    """
    Convert contravariant vector to covariant using metric tensor gsub.

    Args:
        B_contra: list of 3 arrays [Bs, Bt, Bp], each shape (ntheta, nphi)
        gsub: shape (3, 3, ntheta, nphi)

    Returns:
        list [Bsubs, Bsubtheta, Bsubphi]
    """
    return [sum(gsub[i, j] * B_contra[j] for j in range(3)) for i in range(3)]


# ---------------------------------------------------------------------------
# VacuumAdjointQS: main class
# ---------------------------------------------------------------------------

class VacuumAdjointQS(Optimizable):
    r"""
    Adjoint-based quasisymmetry objective for vacuum SPEC equilibria.

    Computes the quasisymmetry figure of merit :math:`f_\mathrm{QS}` and
    (optionally) the iota-target figure of merit :math:`f_\iota`, together
    with their analytical gradients with respect to the plasma-boundary
    Fourier coefficients, using the adjoint method of Nies *et al.* 2022.

    The combined objective is

    .. math::
        J = w_\mathrm{QS}\,f_\mathrm{QS} + w_\iota\,f_\iota

    where

    .. math::
        f_\mathrm{QS} &= \sqrt{\int (\nu_\mathrm{QS})^2\,\mathrm{d}A}, \\
        f_\iota       &= \tfrac{1}{2}(\iota - \iota_*)^2.

    The gradient :math:`\mathrm{d}J/\mathrm{d}\Omega` (with :math:`\Omega`
    any boundary DOF) is computed via an adjoint chain requiring only two
    SPEC solves (one forward, one adjoint) per gradient evaluation, compared
    with :math:`\mathcal{O}(N_\mathrm{DOF})` finite-difference evaluations.

    Args:
        spec: A :class:`~simsopt.mhd.spec.Spec` instance.  Its boundary
          surface and its output HDF5 file are used as inputs.  ``spec``
          becomes the sole parent of this :class:`Optimizable`.
        helicity_m: Poloidal helicity integer M for the target QS.
          E.g. ``M=1`` for quasi-axisymmetry or quasi-helical symmetry.
        helicity_n: Toroidal helicity integer N/nfp for the target QS.
          E.g. ``N=0`` (QA), ``N=1`` (QH in 1 field period).
        s: Normalised radial coordinate (0 < s ≤ 1) of the surface on
          which QS is evaluated.  Default 1.0 (outermost surface).
        lvol: SPEC volume index (0-based) containing the surface.
        iota_target: Target rotational transform.  If ``None`` the iota
          term is omitted.
        iota_weight: Weight on the iota-target objective.
        qs_weight: Weight on the QS objective.
        spec_adjoint_executable: Path to the SPEC binary built with
          ``Lconstraint=-2`` support (required only for :meth:`dJ`).
        mpol_adj: Poloidal resolution for the adjoint PDE solvers.
          Defaults to ``spec.mpol``.
        ntor_adj: Toroidal resolution for the adjoint PDE solvers.
          Defaults to ``spec.ntor``.
        ndiscrete: Grid-refinement factor for the surface quadrature.
          The actual grid is ``(4·mpol·ndiscrete) × (4·ntor·ndiscrete)``.
        Lrad: SPEC radial (Chebyshev) resolution for the adjoint run.
        normalise_qs: If ``True``, normalise :math:`\nu_\mathrm{QS}` by
          :math:`(B/B_0)^2`.
    """

    def __init__(self, spec, helicity_m, helicity_n,
                 s=1.0, lvol=0,
                 iota_target=None, iota_weight=1.0,
                 qs_weight=1.0,
                 spec_adjoint_executable=None,
                 mpol_adj=None, ntor_adj=None,
                 ndiscrete=30, Lrad=4,
                 normalise_qs=True):

        if py_spec is None:
            raise RuntimeError(
                "VacuumAdjointQS requires py_spec to be installed.")

        self.spec = spec
        self.helicity_m = helicity_m
        self.helicity_n = helicity_n
        self.s = np.atleast_1d(np.float64(s))
        self.lvol = lvol
        self.iota_target = iota_target
        self.iota_weight = iota_weight
        self.qs_weight = qs_weight
        self.spec_adjoint_executable = spec_adjoint_executable
        self.ndiscrete = ndiscrete
        self.Lrad = Lrad
        self.normalise_qs = normalise_qs

        self.mpol_adj = mpol_adj if mpol_adj is not None else spec.mpol
        self.ntor_adj = ntor_adj if ntor_adj is not None else spec.ntor

        # Set up 2D quadrature grid
        nfp = spec.nfp
        ntheta = max(4 * self.mpol_adj * ndiscrete, 16)
        nphi = max(4 * self.ntor_adj * ndiscrete, 16)
        self.ntheta = ntheta
        self.nphi = nphi
        self.nfp = nfp

        thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phis = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        self.dtheta = thetas[1] - thetas[0]
        self.dphi = phis[1] - phis[0]
        self.phis_2d, self.thetas_2d = np.meshgrid(phis, thetas)

        # Cache
        self._J = None
        self._dJ = None
        self._f_QS = None
        self._f_iota = None
        self._vQS = None
        self._shape_gradient = None
        self._iota = None

        super().__init__(depends_on=[spec])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def J(self):
        """Compute and return the combined objective J = qs_weight·f_QS + iota_weight·f_iota."""
        self._compute_fom()
        return self._J

    def dJ(self):
        """
        Compute and return the gradient dJ/d(boundary DOFs).

        Returns a 1-D numpy array aligned with ``self.spec.boundary.x``.
        Requires the adjoint SPEC executable (``spec_adjoint_executable``).
        """
        if self.spec_adjoint_executable is None:
            raise ValueError(
                "spec_adjoint_executable must be set to compute gradients.")
        self._compute_fom()
        self._compute_gradient()
        return self._dJ

    @property
    def vQS(self):
        """2D array of QS violation integrand on the surface grid (after J())."""
        return self._vQS

    @property
    def shape_gradient(self):
        """2D shape gradient G(θ,φ) on the surface grid (after dJ())."""
        return self._shape_gradient

    @property
    def iota(self):
        """Rotational transform on the boundary surface (after J())."""
        return self._iota

    def recompute_bell(self, parent=None):
        """Invalidate cache when a parent's DOFs change."""
        self._J = None
        self._dJ = None
        self._f_QS = None
        self._f_iota = None
        self._vQS = None
        self._shape_gradient = None
        self._iota = None
        self._geom_loaded = False

    # ------------------------------------------------------------------
    # Forward computation: FOM
    # ------------------------------------------------------------------

    def _compute_fom(self):
        """Run SPEC (if needed), load geometry, compute f_QS and f_iota."""
        if self._J is not None:
            return  # cache valid

        # Ensure SPEC has run with the current boundary
        self.spec.run()

        h5file = self.spec.extension + ".sp.h5"
        self._load_geometry(h5file)

        Bsups, Bsuptheta, Bsupphi = self._read_B(h5file)
        B_contra = [Bsups, Bsuptheta, Bsupphi]
        B_cov = _covariant_from_contravariant(B_contra, self._gsub)
        Bsubtheta, Bsubphi = B_cov[1], B_cov[2]

        B0 = np.mean(Bsubphi)
        self._B0 = B0
        self._B_contra = B_contra
        self._B_cov = B_cov

        # Straight-field-line equation
        (iota, lambdaSF, dldt, dldp, d2ldt2, d2ldtp, d2ldp2) = _solve_stfl(
            Bsuptheta, Bsupphi,
            self.mpol_adj, self.ntor_adj,
            self.thetas_2d, self.phis_2d, self.nfp)
        self._iota = iota
        self._lambdaSF = lambdaSF
        self._dldt = dldt; self._dldp = dldp
        self._d2ldt2 = d2ldt2; self._d2ldtp = d2ldtp; self._d2ldp2 = d2ldp2
        logger.info(f"VacuumAdjointQS: iota = {iota:.10f}")

        # iota FOM
        f_iota = 0.0
        if self.iota_target is not None:
            f_iota = 0.5 * (iota - self.iota_target) ** 2
        self._f_iota = f_iota

        # QS violation integrand
        helicity_QS = self.helicity_n / self.helicity_m if self.helicity_m != 0 else np.infty
        self._helicity_QS = helicity_QS

        # α tangential gradient
        (nabla_alpha_supt, nabla_alpha_supp,
         abs_nabla_alpha, dabs_nabla_alpha_dt, dabs_nabla_alpha_dp) = \
            self._nabla_alpha(iota, dldt, dldp, d2ldt2, d2ldtp, d2ldp2)
        self._nabla_alpha_supt = nabla_alpha_supt
        self._nabla_alpha_supp = nabla_alpha_supp
        self._abs_nabla_alpha = abs_nabla_alpha
        self._dabs_nabla_alpha_dt = dabs_nabla_alpha_dt
        self._dabs_nabla_alpha_dp = dabs_nabla_alpha_dp

        Bmag, dBmag_dt, dBmag_dp = self._Bmag_and_tangential_derivs(
            B_contra, Bsuptheta, Bsupphi)
        self._Bmag = Bmag
        self._dBmag_dt = dBmag_dt; self._dBmag_dp = dBmag_dp

        abs_nabla_psi = Bmag / abs_nabla_alpha
        self._abs_nabla_psi = abs_nabla_psi

        BgradBmag = Bsuptheta * dBmag_dt + Bsupphi * dBmag_dp
        BxnormdotgradBmag = -(Bsubtheta * dBmag_dp - Bsubphi * dBmag_dt) / self._jac_surf

        norm = (Bmag / B0) ** 2 if self.normalise_qs else 1.0
        vQS, dvQS_dt, dvQS_dp = self._vQS_and_derivs(
            BgradBmag, BxnormdotgradBmag, abs_nabla_psi, iota, B0, norm)
        self._vQS = vQS
        self._dvQS_dt = dvQS_dt; self._dvQS_dp = dvQS_dp
        self._BxnormdotgradBmag = BxnormdotgradBmag

        if self.normalise_qs:
            f_QS = np.sqrt(np.sum((vQS * (Bmag / B0) ** 2) ** 2
                                  * self._jac_surf) * self.dtheta * self.dphi * self.nfp)
            self._norm_QS = f_QS
        else:
            f_QS = 0.5 * np.sum(vQS ** 2 * self._jac_surf) * self.dtheta * self.dphi * self.nfp
            self._norm_QS = 1.0
        self._f_QS = f_QS
        logger.info(f"VacuumAdjointQS: f_QS = {f_QS:.6e}, f_iota = {f_iota:.6e}")

        self._J = self.qs_weight * f_QS + self.iota_weight * f_iota

    # ------------------------------------------------------------------
    # Adjoint computation: gradient
    # ------------------------------------------------------------------

    def _compute_gradient(self):
        """Solve adjoint PDEs, assemble shape gradient, project to DOFs."""
        if self._dJ is not None:
            return

        h5file = self.spec.extension + ".sp.h5"
        B_contra = self._B_contra
        B_cov = self._B_cov
        Bsups, Bsuptheta, Bsupphi = B_contra
        Bsubtheta, Bsubphi = B_cov[1], B_cov[2]
        B0 = self._B0
        iota = self._iota
        helicity_QS = self._helicity_QS

        # Scalar potential ω and its derivatives (needed for divgamma_B)
        (omega, domega_dt, domega_dp,
         d2omega_dt2, d2omega_dtp, d2omega_dp2) = self._scalar_potential(
            Bsubtheta, Bsubphi, B0)
        divgamma_B = self._divgamma_B(B0, domega_dt, domega_dp,
                                      d2omega_dt2, d2omega_dtp, d2omega_dp2)
        self._divgamma_B_cache = divgamma_B

        # q_α adjoint
        qalpha, dqa_dt, dqa_dp, err_de, err_ic = _solve_qalpha(
            fom='quasisymm',
            iota_weight=self.iota_weight,
            divgamma_B=divgamma_B,
            iota_target=self.iota_target,
            Bsuptheta=Bsuptheta, Bsupphi=Bsupphi,
            iota=iota, B0=B0,
            nabla_alpha_supt=self._nabla_alpha_supt,
            nabla_alpha_supp=self._nabla_alpha_supp,
            vQS=self._vQS,
            BxnormdotgradBmag=self._BxnormdotgradBmag,
            abs_nabla_psi=self._abs_nabla_psi,
            helicity_QS=helicity_QS,
            abs_nabla_alpha=self._abs_nabla_alpha,
            mpol=self.mpol_adj, ntor=self.ntor_adj,
            thetas_2d=self.thetas_2d, phis_2d=self.phis_2d, nfp=self.nfp,
            jac_surf=self._jac_surf, dtheta=self.dtheta, dphi=self.dphi,
            norm_QS=self._norm_QS)
        logger.info(f"q_α solve: diff-eq error = {np.max(np.abs(err_de)):.2e}, "
                    f"integral cond = {err_ic:.2e}")
        self._qalpha = qalpha

        # q_ω boundary condition
        qomega_BC = self._qomega_BC(
            B0=B0, qalpha=qalpha,
            iota=iota, helicity_QS=helicity_QS,
            nabla_alpha_supt=self._nabla_alpha_supt,
            nabla_alpha_supp=self._nabla_alpha_supp,
            vQS=self._vQS, dvQS_dt=self._dvQS_dt, dvQS_dp=self._dvQS_dp,
            B_contra=B_contra, B_cov=B_cov,
            divgamma_B=divgamma_B,
            abs_nabla_alpha=self._abs_nabla_alpha,
            abs_nabla_psi=self._abs_nabla_psi,
            Bmag=self._Bmag,
            dBmag_dt=self._dBmag_dt, dBmag_dp=self._dBmag_dp,
            BxnormdotgradBmag=self._BxnormdotgradBmag)

        # Adjoint SPEC solve
        dqomega_dt, dqomega_dp = self._run_adjoint_spec(qomega_BC, h5file)

        # B and its s-derivative needed for shape gradient
        dBsups_ds, dBsuptheta_ds, dBsupphi_ds = self._read_dB_ds(h5file)
        dBcontrajac_ds = [dBsups_ds, dBsuptheta_ds, dBsupphi_ds]
        dBmag_ds = self._Bmag_s_deriv(self._Bmag, B_contra, dBcontrajac_ds)

        # Assemble shape gradient
        G = self._assemble_shape_gradient(
            B0=B0, Bsuptheta=Bsuptheta, Bsupphi=Bsupphi,
            Bsubtheta=Bsubtheta, Bsubphi=Bsubphi, Bsubs=B_cov[0],
            dqomega_dt=dqomega_dt, dqomega_dp=dqomega_dp,
            qalpha=qalpha, iota=iota,
            dldt=self._dldt, dldp=self._dldp,
            dBcontrajac_ds=dBcontrajac_ds,
            vQS=self._vQS, dvQS_dt=self._dvQS_dt, dvQS_dp=self._dvQS_dp,
            divgamma_B=divgamma_B,
            helicity_QS=helicity_QS,
            abs_nabla_psi=self._abs_nabla_psi,
            Bmag=self._Bmag, dBmag_dt=self._dBmag_dt,
            dBmag_dp=self._dBmag_dp, dBmag_ds=dBmag_ds,
            abs_nabla_alpha=self._abs_nabla_alpha,
            dabs_nabla_alpha_dt=self._dabs_nabla_alpha_dt,
            dabs_nabla_alpha_dp=self._dabs_nabla_alpha_dp,
            norm_QS=self._norm_QS)
        self._shape_gradient = G

        # Project shape gradient onto boundary Fourier parameters → DOF gradient
        self._dJ = self._param_derivs(G)

    # ------------------------------------------------------------------
    # Geometry: load SPEC geometry from HDF5
    # (ported from vacuum_adjoint.load_SPEC_geometry)
    # ------------------------------------------------------------------

    def _load_geometry(self, h5file):
        """Read SPEC HDF5 and populate all geometric arrays."""
        myspec = SPECout(h5file)

        Rac, Rbc = myspec.output.Rbc[self.lvol: self.lvol + 2]
        Zas, Zbs = myspec.output.Zbs[self.lvol: self.lvol + 2]
        xm = myspec.output.im
        xn = myspec.output.in_

        # Save the boundary actually used (SPEC may have modified it)
        self._boundary_in_h5 = np.concatenate((Rbc, Zbs[1:]))

        # Zernike regularisation factors
        sbar = (self.s[-1] + 1) / 2
        mn = Rac.size
        fac1 = np.zeros(mn); fac2 = np.zeros(mn); fac3 = np.zeros(mn)
        for j in range(mn):
            if self.lvol == 0 and xm[j] == 0:
                fac1[j] = sbar**2; fac2[j] = sbar; fac3[j] = 0.5
            elif self.lvol == 0 and xm[j] > 0:
                fac1[j] = sbar**xm[j]
                fac2[j] = (xm[j] / 2.0) * sbar**(xm[j] - 1.0)
                fac3[j] = xm[j] / 4.0 * (xm[j] - 1.0) * sbar**(xm[j] - 2.0)
            else:
                fac1[j] = sbar; fac2[j] = 0.5; fac3[j] = 0.0

        dRmnds = fac2 * (Rbc - Rac)
        dZmnds = fac2 * (Zbs - Zas)
        d2Rmnds2 = fac3 * (Rbc - Rac)
        d2Zmnds2 = fac3 * (Zbs - Zas)

        nt, np_ = self.ntheta, self.nphi
        thetas = self.thetas_2d[:, 0]  # shape (ntheta,)
        phis_1d = self.phis_2d[0, :]   # shape (nphi,)

        # Vectorised IFT over modes
        xm_m = xm[:, None, None]
        xn_m = xn[:, None, None]
        th_m = thetas[None, :, None]
        ph_m = phis_1d[None, None, :]
        angle = xm_m * th_m - xn_m * ph_m
        cos_a = np.cos(angle); sin_a = np.sin(angle)

        Rbc_m = Rbc[:, None, None]; Zbs_m = Zbs[:, None, None]
        dRds_m = dRmnds[:, None, None]; dZds_m = dZmnds[:, None, None]
        d2Rds2_m = d2Rmnds2[:, None, None]; d2Zds2_m = d2Zmnds2[:, None, None]

        R = np.sum(Rbc_m * cos_a, 0)           # (ntheta, nphi)
        Z = np.sum(Zbs_m * sin_a, 0)
        dRdt = -np.sum(Rbc_m * xm_m * sin_a, 0)
        dRdp = np.sum(Rbc_m * xn_m * sin_a, 0)
        dZdt = np.sum(Zbs_m * xm_m * cos_a, 0)
        dZdp = -np.sum(Zbs_m * xn_m * cos_a, 0)
        dRds = np.sum(dRds_m * cos_a, 0)
        dZds = np.sum(dZds_m * sin_a, 0)
        d2Rdt2 = -np.sum(Rbc_m * xm_m**2 * cos_a, 0)
        d2Rdp2 = -np.sum(Rbc_m * xn_m**2 * cos_a, 0)
        d2Rdtp = np.sum(Rbc_m * xn_m * xm_m * cos_a, 0)
        d2Zdt2 = -np.sum(Zbs_m * xm_m**2 * sin_a, 0)
        d2Zdp2 = -np.sum(Zbs_m * xn_m**2 * sin_a, 0)
        d2Zdtp = np.sum(Zbs_m * xm_m * xn_m * sin_a, 0)
        d2Rdsdt = -np.sum(dRds_m * xm_m * sin_a, 0)
        d2Rdsdp = np.sum(dRds_m * xn_m * sin_a, 0)
        d2Zdsdt = np.sum(dZds_m * xm_m * cos_a, 0)
        d2Zdsdp = -np.sum(dZds_m * xn_m * cos_a, 0)
        d2Rds2 = np.sum(d2Rds2_m * cos_a, 0)
        d2Zds2 = np.sum(d2Zds2_m * sin_a, 0)

        ph = self.phis_2d   # (ntheta, nphi)
        cos_ph = np.cos(ph); sin_ph = np.sin(ph)

        X = R * cos_ph; Y = R * sin_ph
        dXdt = dRdt * cos_ph; dYdt = dRdt * sin_ph
        dXdp = dRdp * cos_ph - R * sin_ph
        dYdp = dRdp * sin_ph + R * cos_ph
        dXds = dRds * cos_ph; dYds = dRds * sin_ph

        d2Xdt2 = d2Rdt2 * cos_ph
        d2Ydt2 = d2Rdt2 * sin_ph
        d2Xdtp = d2Rdtp * cos_ph - dRdt * sin_ph
        d2Ydtp = d2Rdtp * sin_ph + dRdt * cos_ph
        d2Xdp2 = d2Rdp2 * cos_ph - 2 * dRdp * sin_ph - R * cos_ph
        d2Ydp2 = d2Rdp2 * sin_ph + 2 * dRdp * cos_ph - R * sin_ph
        d2Xdsdt = d2Rdsdt * cos_ph; d2Ydsdt = d2Rdsdt * sin_ph
        d2Xdsdp = d2Rdsdp * cos_ph - dRds * sin_ph
        d2Ydsdp = d2Rdsdp * sin_ph + dRds * cos_ph
        d2Xds2 = d2Rds2 * cos_ph; d2Yds2 = d2Rds2 * sin_ph

        # Store Cartesian partial derivatives
        self._drdtheta = np.stack([dXdt, dYdt, dZdt])     # (3, nt, np)
        self._drdphi = np.stack([dXdp, dYdp, dZdp])
        self._drds = np.stack([dXds, dYds, dZds])
        self._d2rdtheta2 = np.stack([d2Xdt2, d2Ydt2, d2Zdt2])
        self._d2rdthetadphi = np.stack([d2Xdtp, d2Ydtp, d2Zdtp])
        self._d2rdphi2 = np.stack([d2Xdp2, d2Ydp2, d2Zdp2])
        self._d2rdsdtheta = np.stack([d2Xdsdt, d2Ydsdt, d2Zdsdt])
        self._d2rdsdphi = np.stack([d2Xdsdp, d2Ydsdp, d2Zdsdp])
        self._d2rds2 = np.stack([d2Xds2, d2Yds2, d2Zds2])

        # Metric tensor g_sub (covariant)
        gss = np.einsum('iab,iab->ab', self._drds, self._drds)
        gst = np.einsum('iab,iab->ab', self._drds, self._drdtheta)
        gsp = np.einsum('iab,iab->ab', self._drds, self._drdphi)
        gtt = np.einsum('iab,iab->ab', self._drdtheta, self._drdtheta)
        gtp = np.einsum('iab,iab->ab', self._drdtheta, self._drdphi)
        gpp = np.einsum('iab,iab->ab', self._drdphi, self._drdphi)
        gsub = np.array([[gss, gst, gsp],
                         [gst, gtt, gtp],
                         [gsp, gtp, gpp]])  # (3,3,nt,np)
        self._gsub = gsub
        self._gsup = np.linalg.inv(gsub.T).T

        gsub_surf = gsub[1:, 1:]
        self._gsub_surf = gsub_surf
        self._gsup_surf = np.linalg.inv(gsub_surf.T).T

        # Metric derivatives for the shape gradient computation
        def _dg(drd1, drd2, d2rd1, d2rd2, d2rdcr1, d2rdcr2):
            """d(g_12)/d(coord) for symmetric metric component."""
            return (np.einsum('iab,iab->ab', d2rdcr1, drd2)
                    + np.einsum('iab,iab->ab', drd1, d2rdcr2))

        # d gsub / dtheta
        dg_dt = np.zeros_like(gsub)
        dg_dt[0, 0] = 2 * np.einsum('iab,iab->ab', self._d2rdsdtheta, self._drds)
        dg_dt[0, 1] = dg_dt[1, 0] = (
            np.einsum('iab,iab->ab', self._d2rdsdtheta, self._drdtheta)
            + np.einsum('iab,iab->ab', self._drds, self._d2rdtheta2))
        dg_dt[0, 2] = dg_dt[2, 0] = (
            np.einsum('iab,iab->ab', self._d2rdsdtheta, self._drdphi)
            + np.einsum('iab,iab->ab', self._drds, self._d2rdthetadphi))
        dg_dt[1, 1] = 2 * np.einsum('iab,iab->ab', self._d2rdtheta2, self._drdtheta)
        dg_dt[1, 2] = dg_dt[2, 1] = (
            np.einsum('iab,iab->ab', self._d2rdtheta2, self._drdphi)
            + np.einsum('iab,iab->ab', self._drdtheta, self._d2rdthetadphi))
        dg_dt[2, 2] = 2 * np.einsum('iab,iab->ab', self._d2rdthetadphi, self._drdphi)
        self._dgdtheta = dg_dt
        self._dgsupdtheta = -np.einsum('ijab,jkab->ikab',
                                       np.einsum('ijab,jkab->ikab', self._gsup, dg_dt),
                                       self._gsup)
        self._dgdtheta_surf = dg_dt[1:, 1:]
        self._dgsupdtheta_surf = -np.einsum(
            'ijab,jkab->ikab',
            np.einsum('ijab,jkab->ikab', self._gsup_surf, dg_dt[1:, 1:]),
            self._gsup_surf)

        # d gsub / dphi
        dg_dp = np.zeros_like(gsub)
        dg_dp[0, 0] = 2 * np.einsum('iab,iab->ab', self._d2rdsdphi, self._drds)
        dg_dp[0, 1] = dg_dp[1, 0] = (
            np.einsum('iab,iab->ab', self._d2rdsdphi, self._drdtheta)
            + np.einsum('iab,iab->ab', self._drds, self._d2rdthetadphi))
        dg_dp[0, 2] = dg_dp[2, 0] = (
            np.einsum('iab,iab->ab', self._d2rdsdphi, self._drdphi)
            + np.einsum('iab,iab->ab', self._drds, self._d2rdphi2))
        dg_dp[1, 1] = 2 * np.einsum('iab,iab->ab', self._d2rdthetadphi, self._drdtheta)
        dg_dp[1, 2] = dg_dp[2, 1] = (
            np.einsum('iab,iab->ab', self._d2rdthetadphi, self._drdphi)
            + np.einsum('iab,iab->ab', self._drdtheta, self._d2rdphi2))
        dg_dp[2, 2] = 2 * np.einsum('iab,iab->ab', self._d2rdphi2, self._drdphi)
        self._dgdphi = dg_dp
        self._dgsupdphi = -np.einsum('ijab,jkab->ikab',
                                     np.einsum('ijab,jkab->ikab', self._gsup, dg_dp),
                                     self._gsup)
        self._dgdphi_surf = dg_dp[1:, 1:]
        self._dgsupdphi_surf = -np.einsum(
            'ijab,jkab->ikab',
            np.einsum('ijab,jkab->ikab', self._gsup_surf, dg_dp[1:, 1:]),
            self._gsup_surf)

        # d gsub / ds
        dg_ds = np.zeros_like(gsub)
        dg_ds[0, 0] = 2 * np.einsum('iab,iab->ab', self._d2rds2, self._drds)
        dg_ds[0, 1] = dg_ds[1, 0] = (
            np.einsum('iab,iab->ab', self._d2rds2, self._drdtheta)
            + np.einsum('iab,iab->ab', self._drds, self._d2rdsdtheta))
        dg_ds[0, 2] = dg_ds[2, 0] = (
            np.einsum('iab,iab->ab', self._d2rds2, self._drdphi)
            + np.einsum('iab,iab->ab', self._drds, self._d2rdsdphi))
        dg_ds[1, 1] = 2 * np.einsum('iab,iab->ab', self._d2rdsdtheta, self._drdtheta)
        dg_ds[1, 2] = dg_ds[2, 1] = (
            np.einsum('iab,iab->ab', self._d2rdsdtheta, self._drdphi)
            + np.einsum('iab,iab->ab', self._drdtheta, self._d2rdsdphi))
        dg_ds[2, 2] = 2 * np.einsum('iab,iab->ab', self._d2rdsdphi, self._drdphi)
        self._dgds = dg_ds

        # Jacobians
        jac_surf = np.sqrt(gsub[1, 1] * gsub[2, 2] - gsub[1, 2] ** 2)
        self._jac_surf = jac_surf
        self._djac_surf_dt = ((gsub[1, 1] * dg_dt[2, 2] + dg_dt[1, 1] * gsub[2, 2]
                               - 2 * gsub[1, 2] * dg_dt[1, 2]) / (2 * jac_surf))
        self._djac_surf_dp = ((gsub[1, 1] * dg_dp[2, 2] + dg_dp[1, 1] * gsub[2, 2]
                               - 2 * gsub[1, 2] * dg_dp[1, 2]) / (2 * jac_surf))

        jac = (dXds * (dYdt * dZdp - dYdp * dZdt)
               + dYds * (dZdt * dXdp - dZdp * dXdt)
               + dZds * (dXdt * dYdp - dXdp * dYdt))
        self._jac = jac
        self._djac_ds = (
            d2Xds2 * (dYdt * dZdp - dYdp * dZdt)
            + dXds * (d2Ydsdt * dZdp - d2Ydsdp * dZdt)
            + dXds * (dYdt * d2Zdsdp - dYdp * d2Zdsdt)
            + d2Yds2 * (dZdt * dXdp - dZdp * dXdt)
            + dYds * (d2Zdsdt * dXdp - d2Zdsdp * dXdt)
            + dYds * (dZdt * d2Xdsdp - dZdp * d2Xdsdt)
            + d2Zds2 * (dXdt * dYdp - dXdp * dYdt)
            + dZds * (d2Xdsdt * dYdp - d2Xdsdp * dYdt)
            + dZds * (dXdt * d2Ydsdt - dXdp * d2Ydsdp)
        )
        self._djac_dt = (
            d2Xdsdt * (dYdt * dZdp - dYdp * dZdt)
            + dXds * (d2Ydt2 * dZdp - d2Ydtp * dZdt + dYdt * d2Zdtp - dYdp * d2Zdt2)
            + d2Ydsdt * (dZdt * dXdp - dZdp * dXdt)
            + dYds * (d2Zdt2 * dXdp - d2Zdtp * dXdt + dZdt * d2Xdtp - dZdp * d2Xdt2)
            + d2Zdsdt * (dXdt * dYdp - dXdp * dYdt)
            + dZds * (d2Xdt2 * dYdp - d2Xdtp * dYdt + dXdt * d2Ydtp - dXdp * d2Ydt2)
        )
        self._djac_dp = (
            d2Xdsdp * (dYdt * dZdp - dYdp * dZdt)
            + dXds * (d2Ydtp * dZdp - d2Ydp2 * dZdt + dYdt * d2Zdp2 - dYdp * d2Zdtp)
            + d2Ydsdp * (dZdt * dXdp - dZdp * dXdt)
            + dYds * (d2Zdtp * dXdp - d2Zdp2 * dXdt + dZdt * d2Xdp2 - dZdp * d2Xdtp)
            + d2Zdsdp * (dXdt * dYdp - dXdp * dYdt)
            + dZds * (d2Xdtp * dYdp - d2Xdp2 * dYdt + dXdt * d2Ydp2 - dXdp * d2Ydtp)
        )

        # Surface normal
        Nx = -dYdp * dZdt + dYdt * dZdp
        Ny = -dZdp * dXdt + dZdt * dXdp
        Nz = -dXdp * dYdt + dXdt * dYdp
        norm_N = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        self._norm_normal = norm_N
        nx = Nx / norm_N; ny = Ny / norm_N; nz = Nz / norm_N
        self._normal = np.stack([nx, ny, nz])   # (3, nt, np)

        # Normal derivatives (needed for shape gradient)
        dNxdt = -d2Ydtp * dZdt - dYdp * d2Zdt2 + d2Ydt2 * dZdp + dYdt * d2Zdtp
        dNydt = -d2Zdtp * dXdt - dZdp * d2Xdt2 + d2Zdt2 * dXdp + dZdt * d2Xdtp
        dNzdt = -d2Xdtp * dYdt - dXdp * d2Ydt2 + d2Xdt2 * dYdp + dXdt * d2Ydtp
        NdNdt = Nx * dNxdt + Ny * dNydt + Nz * dNzdt

        dNxdp = -d2Ydp2 * dZdt - dYdp * d2Zdtp + d2Ydtp * dZdp + dYdt * d2Zdp2
        dNydp = -d2Zdp2 * dXdt - dZdp * d2Xdtp + d2Zdtp * dXdp + dZdt * d2Xdp2
        dNzdp = -d2Xdp2 * dYdt - dXdp * d2Ydtp + d2Xdtp * dYdp + dXdt * d2Ydp2
        NdNdp = Nx * dNxdp + Ny * dNydp + Nz * dNzdp

        dNxds = -d2Ydsdp * dZdt - dYdp * d2Zdsdt + d2Ydsdt * dZdp + dYdt * d2Zdsdp
        dNyds = -d2Zdsdp * dXdt - dZdp * d2Xdsdt + d2Zdsdt * dXdp + dZdt * d2Xdsdp
        dNzds = -d2Xdsdp * dYdt - dXdp * d2Ydsdt + d2Xdsdt * dYdp + dXdt * d2Ydsdp
        NdNds = Nx * dNxds + Ny * dNyds + Nz * dNzds

        def _dn(dNx, dNy, dNz, NdN):
            dnx = (dNx - nx * NdN / norm_N**2) / norm_N
            dny = (dNy - ny * NdN / norm_N**2) / norm_N
            dnz = (dNz - nz * NdN / norm_N**2) / norm_N
            return np.stack([dnx, dny, dnz])

        self._dnormaldt = _dn(dNxdt, dNydt, dNzdt, NdNdt)
        self._dnormaldp = _dn(dNxdp, dNydp, dNzdp, NdNdp)
        self._dnormalds = _dn(dNxds, dNyds, dNzds, NdNds)

        # Normal contravariant components (for BC and shape gradient)
        nablas_X = (dYdt * dZdp - dZdt * dYdp) / jac
        nablas_Y = (dZdt * dXdp - dXdt * dZdp) / jac
        nablas_Z = (dXdt * dYdp - dYdt * dXdp) / jac
        nablat_X = (dYdp * dZds - dZdp * dYds) / jac
        nablat_Y = (dZdp * dXds - dXdp * dZds) / jac
        nablat_Z = (dXdp * dYds - dYdp * dXds) / jac
        nablap_X = (dYds * dZdt - dZds * dYdt) / jac
        nablap_Y = (dZds * dXdt - dXds * dZdt) / jac
        nablap_Z = (dXds * dYdt - dYds * dXdt) / jac

        self._nsups = nx * nablas_X + ny * nablas_Y + nz * nablas_Z
        self._nsuptheta = nx * nablat_X + ny * nablat_Y + nz * nablat_Z
        self._nsupphi = nx * nablap_X + ny * nablap_Y + nz * nablap_Z

        ncontra = [self._nsups, self._nsuptheta, self._nsupphi]
        ncov = _covariant_from_contravariant(ncontra, gsub)
        self._nsubs = ncov[0]

        # Mean curvature (for shape gradient QS terms)
        gE = np.einsum('iab,iab->ab', self._drdphi, self._drdphi)
        gF = np.einsum('iab,iab->ab', self._drdtheta, self._drdphi)
        gG = np.einsum('iab,iab->ab', self._drdtheta, self._drdtheta)
        e2 = np.einsum('iab,iab->ab', self._normal, self._d2rdphi2)
        f2 = np.einsum('iab,iab->ab', self._normal, self._d2rdthetadphi)
        g2 = np.einsum('iab,iab->ab', self._normal, self._d2rdtheta2)
        denom = gE * gG - gF**2
        H = (e2 * gG - 2 * f2 * gF + g2 * gE) / (2 * denom)
        self._summed_curvature = -H * 2   # sign convention from adjoint_QS

    # ------------------------------------------------------------------
    # Read SPEC magnetic field from HDF5
    # ------------------------------------------------------------------

    def _read_B(self, h5file):
        """Return (Bsups, Bsuptheta, Bsupphi) on the 2D surface grid."""
        myspec = SPECout(h5file)
        B = myspec.get_B_FFT(lvol=self.lvol, jacobian=self._jac,
                             sarr=self.s, Nt=self.ntheta, Nz=self.nphi)[:, 0, :, :]
        return B[0], B[1], B[2]

    def _read_dB_ds(self, h5file):
        """Return s-derivatives of (jac * B^i) on the surface grid."""
        myspec = SPECout(h5file)
        dB = myspec.get_s_der_B_FFT(lvol=self.lvol, jacobian=self._jac,
                                     sarr=self.s, Nt=self.ntheta, Nz=self.nphi)[:, 0, :, :]
        return dB[0] * self._jac, dB[1] * self._jac, dB[2] * self._jac

    def _read_dB_adjoint(self, h5file_adj):
        """Return (Bsups, Bsuptheta, Bsupphi) from adjoint SPEC HDF5."""
        myspec = SPECout(h5file_adj)
        B = myspec.get_B_FFT(lvol=self.lvol, jacobian=self._jac,
                             sarr=self.s, Nt=self.ntheta, Nz=self.nphi)[:, 0, :, :]
        return B[0], B[1], B[2]

    # ------------------------------------------------------------------
    # Field-derived quantities
    # ------------------------------------------------------------------

    def _Bmag_and_tangential_derivs(self, B_contra, Bsuptheta, Bsupphi):
        """|B| and its θ,φ derivatives via Fourier differentiation."""
        Bmag = np.sqrt(np.einsum('iab,jiab,jab->ab', B_contra, self._gsub, B_contra))
        nmodes, xm, xn = _cosine_modes(self.mpol_adj, self.ntor_adj)
        basis, db_dt, db_dp = _basis_and_derivs(
            nmodes, xm, xn, 'cos', self.thetas_2d, self.phis_2d, self.nfp)
        c = _fourier_coeffs(Bmag, basis)
        return Bmag, np.einsum('i,iab->ab', c, db_dt), np.einsum('i,iab->ab', c, db_dp)

    def _Bmag_s_deriv(self, Bmag, B_contra, dBcontrajac_ds):
        """s-derivative of |B|."""
        dBcontrajac_ds_arr = np.array(dBcontrajac_ds)
        return (-Bmag * self._djac_ds / self._jac
                + (np.einsum('iab,jiab,jab->ab', dBcontrajac_ds_arr, self._gsub, B_contra)
                   / self._jac
                   + np.einsum('iab,jiab,jab->ab', B_contra, self._dgds, B_contra) / 2)
                / Bmag)

    def _nabla_alpha(self, iota, dldt, dldp, d2ldt2, d2ldtp, d2ldp2):
        """Contravariant components of ∇_Γ α and their θ,φ derivatives."""
        dalpha_dt = 1 + dldt
        dalpha_dp = -iota + dldp
        d2alpha_dt2 = d2ldt2
        d2alpha_dtp = d2ldtp
        d2alpha_dp2 = d2ldp2

        gsupt = self._gsup_surf[0]  # (2, nt, np)
        gsupp = self._gsup_surf[1]
        na_supt = gsupt[0] * dalpha_dt + gsupt[1] * dalpha_dp
        na_supp = gsupp[0] * dalpha_dt + gsupp[1] * dalpha_dp
        abs_na = np.sqrt(na_supt * dalpha_dt + na_supp * dalpha_dp)

        # θ-derivative of |∇_Γ α|
        dna_supt_dt = (self._dgsupdtheta_surf[0, 0] * dalpha_dt + gsupt[0] * d2alpha_dt2
                       + self._dgsupdtheta_surf[0, 1] * dalpha_dp + gsupt[1] * d2alpha_dtp)
        dna_supp_dt = (self._dgsupdtheta_surf[1, 0] * dalpha_dt + gsupp[0] * d2alpha_dt2
                       + self._dgsupdtheta_surf[1, 1] * dalpha_dp + gsupp[1] * d2alpha_dtp)
        dabs_na_dt = (d2alpha_dt2 * na_supt + dalpha_dt * dna_supt_dt
                      + d2alpha_dtp * na_supp + dalpha_dp * dna_supp_dt) / (2 * abs_na)

        # φ-derivative of |∇_Γ α|
        dna_supt_dp = (self._dgsupdphi_surf[0, 0] * dalpha_dt + gsupt[0] * d2alpha_dtp
                       + self._dgsupdphi_surf[0, 1] * dalpha_dp + gsupt[1] * d2alpha_dp2)
        dna_supp_dp = (self._dgsupdphi_surf[1, 0] * dalpha_dt + gsupp[0] * d2alpha_dtp
                       + self._dgsupdphi_surf[1, 1] * dalpha_dp + gsupp[1] * d2alpha_dp2)
        dabs_na_dp = (d2alpha_dtp * na_supt + dalpha_dt * dna_supt_dp
                      + d2alpha_dp2 * na_supp + dalpha_dp * dna_supp_dp) / (2 * abs_na)

        return na_supt, na_supp, abs_na, dabs_na_dt, dabs_na_dp

    def _vQS_and_derivs(self, BgradBmag, BxnormdotgradBmag, abs_nabla_psi,
                        iota, B0, norm):
        """QS violation integrand v_QS and its θ,φ derivatives."""
        h = self.helicity_n / self.helicity_m if self.helicity_m != 0 else np.infty
        if h is np.infty:
            vQS = -BxnormdotgradBmag * abs_nabla_psi / B0 / B0**2 / norm
        else:
            vQS = (BgradBmag - BxnormdotgradBmag * (iota - h) * abs_nabla_psi / B0) / B0**2 / norm

        # Fourier-differentiate vQS
        nmodes, xm, xn = _sine_modes(self.mpol_adj, self.ntor_adj)
        basis, db_dt, db_dp = _basis_and_derivs(
            nmodes, xm, xn, 'sin', self.thetas_2d, self.phis_2d, self.nfp)
        c = _fourier_coeffs(vQS, basis)
        return vQS, np.einsum('i,iab->ab', c, db_dt), np.einsum('i,iab->ab', c, db_dp)

    def _scalar_potential(self, Bsubtheta, Bsubphi, B0):
        """Compute scalar potential ω as in B = B0 ∇(φ + ω), and derivatives."""
        nmodes, xm, xn = _sine_modes(self.mpol_adj, self.ntor_adj)
        omega = np.zeros_like(self.thetas_2d)
        do_dt = np.zeros_like(omega); do_dp = np.zeros_like(omega)
        d2o_dt2 = np.zeros_like(omega); d2o_dtp = np.zeros_like(omega)
        d2o_dp2 = np.zeros_like(omega)

        for i in range(nmodes):
            angle = xm[i] * self.thetas_2d - self.nfp * xn[i] * self.phis_2d
            norm = np.sum(np.cos(angle)**2)
            if xn[i] != 0:
                omn = -np.sum(Bsubphi * np.cos(angle)) / (self.nfp * xn[i] * norm)
            else:
                omn = np.sum(Bsubtheta * np.cos(angle)) / (xm[i] * norm)
            omega += omn * np.sin(angle)
            do_dt += omn * np.cos(angle) * xm[i]
            do_dp -= omn * np.cos(angle) * xn[i] * self.nfp
            d2o_dt2 -= omn * np.sin(angle) * xm[i]**2
            d2o_dtp += omn * np.sin(angle) * xm[i] * xn[i] * self.nfp
            d2o_dp2 -= omn * np.sin(angle) * (xn[i] * self.nfp)**2

        return omega, do_dt, do_dp, d2o_dt2, d2o_dtp, d2o_dp2

    def _divgamma_B(self, B0, do_dt, do_dp, d2o_dt2, d2o_dtp, d2o_dp2):
        """Tangential divergence of B from scalar potential representation."""
        gs = self._gsup_surf
        dgs_dt = self._dgsupdtheta_surf
        dgs_dp = self._dgsupdphi_surf
        jac_s = self._jac_surf
        djs_dt = self._djac_surf_dt
        djs_dp = self._djac_surf_dp

        d_A00_dt = djs_dt * gs[0, 0] + jac_s * dgs_dt[0, 0]
        d_A01_dp = djs_dp * gs[0, 1] + jac_s * dgs_dp[0, 1]
        d_A01_dt = djs_dt * gs[0, 1] + jac_s * dgs_dt[0, 1]
        d_A11_dp = djs_dp * gs[1, 1] + jac_s * dgs_dp[1, 1]

        return B0 * (gs[0, 0] * d2o_dt2 + 2 * gs[0, 1] * d2o_dtp + gs[1, 1] * d2o_dp2
                     + (do_dt * (d_A00_dt + d_A01_dp)
                        + (1 + do_dp) * (d_A01_dt + d_A11_dp)) / jac_s)

    # ------------------------------------------------------------------
    # q_ω boundary condition (Neumann BC for the adjoint Laplace solve)
    # ------------------------------------------------------------------

    def _qomega_BC(self, B0, qalpha, iota, helicity_QS,
                   nabla_alpha_supt, nabla_alpha_supp,
                   vQS, dvQS_dt, dvQS_dp,
                   B_contra, B_cov, divgamma_B,
                   abs_nabla_alpha, abs_nabla_psi,
                   Bmag, dBmag_dt, dBmag_dp, BxnormdotgradBmag):
        """Assemble the Neumann BC on q_ω for the adjoint SPEC run."""
        Bsuptheta = B_contra[1]; Bsupphi = B_contra[2]
        gs = self._gsup_surf

        vec_supt = -qalpha * nabla_alpha_supt
        vec_supp = -qalpha * nabla_alpha_supp

        if self.normalise_qs:
            norm = self._norm_QS
            vec_supt += 2 * vQS**2 * (Bmag / B0)**2 * Bsuptheta / B0 / norm
            vec_supp += 2 * vQS**2 * (Bmag / B0)**2 * Bsupphi / B0 / norm

        # Tangential gradient of |B|
        nablaG_B_supt = gs[0, 0] * dBmag_dt + gs[0, 1] * dBmag_dp
        nablaG_B_supp = gs[1, 0] * dBmag_dt + gs[1, 1] * dBmag_dp
        norm = self._norm_QS
        vec_supt -= vQS * nablaG_B_supt / B0 / norm
        vec_supp -= vQS * nablaG_B_supp / B0 / norm

        # div_Γ(vQS · B)
        BdotvQS = Bsuptheta * dvQS_dt + Bsupphi * dvQS_dp
        divgamma_BvQS = BdotvQS + vQS * divgamma_B
        vec_supt += Bsuptheta / Bmag * divgamma_BvQS / B0 / norm
        vec_supp += Bsupphi / Bmag * divgamma_BvQS / B0 / norm

        # Cross-product terms (tune_out_2=False branch)
        h = helicity_QS
        factor = (iota - h) if h is not np.infty else 1.0
        vec_supt += factor * vQS * abs_nabla_psi / B0 * (-dBmag_dp / self._jac_surf) / B0 / norm
        vec_supp += factor * vQS * abs_nabla_psi / B0 * (dBmag_dt / self._jac_surf) / B0 / norm

        # Tangential divergence of (vQS |∇ψ|/B) n×B
        Bsubtheta = B_cov[1]; Bsubphi = B_cov[2]
        arg_t = vQS * abs_nabla_psi / B0 * Bsubphi / Bmag
        arg_p = -vQS * abs_nabla_psi / B0 * Bsubtheta / Bmag
        nmodes_sin, xm_sin, xn_sin = _sine_modes(self.mpol_adj, self.ntor_adj)
        sin_b, dsin_dt, dsin_dp = _basis_and_derivs(
            nmodes_sin, xm_sin, xn_sin, 'sin', self.thetas_2d, self.phis_2d, self.nfp)
        ct = _fourier_coeffs(arg_t, sin_b)
        cp = _fourier_coeffs(arg_p, sin_b)
        divgamma_arg = (np.einsum('i,iab->ab', ct, dsin_dt)
                        + np.einsum('i,iab->ab', cp, dsin_dp)) / self._jac_surf
        vec_supt -= factor * Bsuptheta * divgamma_arg / B0 / norm
        vec_supp -= factor * Bsupphi * divgamma_arg / B0 / norm

        # Fourier-transform the tangential field and compute tangential divergence
        nmodes_cos, xm_cos, xn_cos = _cosine_modes(self.mpol_adj, self.ntor_adj)
        cos_b, dcos_dt, dcos_dp = _basis_and_derivs(
            nmodes_cos, xm_cos, xn_cos, 'cos', self.thetas_2d, self.phis_2d, self.nfp)
        jac_s = self._jac_surf
        ft = vec_supt * jac_s; fp = vec_supp * jac_s
        ct2 = _fourier_coeffs(ft, cos_b)
        cp2 = _fourier_coeffs(fp, cos_b)
        dft_dt = np.einsum('i,iab->ab', ct2, dcos_dt)
        dfp_dp = np.einsum('i,iab->ab', cp2, dcos_dp)
        qomega_BC = dft_dt + dfp_dp
        return qomega_BC

    # ------------------------------------------------------------------
    # Adjoint SPEC run
    # ------------------------------------------------------------------

    def _run_adjoint_spec(self, qomega_BC, h5file_fwd):
        """
        Set the Neumann BC on q_ω, run adjoint SPEC with Lconstraint=-2,
        and return (dqomega/dtheta, dqomega/dphi) on the surface grid.
        """
        # Fourier-decompose the boundary condition
        mpol = self.mpol_adj; ntor = self.ntor_adj
        nmodes_sin, xm_sin, xn_sin = _sine_modes(mpol, ntor)
        sin_b, _, _ = _basis_and_derivs(
            nmodes_sin, xm_sin, xn_sin, 'sin', self.thetas_2d, self.phis_2d, self.nfp)
        vns_mn = _fourier_coeffs(qomega_BC, sin_b)
        vnc_mn = np.zeros(vns_mn.size)   # stellarator symmetry

        # Build adjoint input file from forward .sp file
        sp_fwd = self.spec.extension + ".sp"
        sp_adj = self.spec.extension + "_adjoint.sp"
        shutil.copy(sp_fwd, sp_adj)

        nml = SPECNamelist(sp_adj)
        nml['physicslist']['Lconstraint'] = -2
        nml['physicslist']['Lbdybnzero'] = False
        nml['physicslist']['curtor'] = 0
        nml['physicslist']['curpol'] = 0
        nml['numericlist']['Lrad'] = self.Lrad

        # Write boundary (use the boundary as read from the forward HDF5)
        _update_spec_boundary(nml, 'Rbc', self._boundary_in_h5[:len(self._boundary_in_h5)//2 + 1],
                              mpol, ntor, 'cos')
        _update_spec_boundary(nml, 'Zbs', self._boundary_in_h5[len(self._boundary_in_h5)//2 + 1:],
                              mpol, ntor, 'sin')

        # Write Neumann BC (Vns)
        _update_spec_boundary(nml, 'Vns', -vns_mn, mpol, ntor, 'sin')
        _update_spec_boundary(nml, 'Vnc', -vnc_mn, mpol, ntor, 'cos')

        nml.write(sp_adj, force=True)
        nml.run(spec_command=self.spec_adjoint_executable,
                filename=sp_adj, force=True, quiet=True)

        # Read adjoint field: B_adj covariant components give dqomega/dtheta, dqomega/dphi
        h5_adj = sp_adj + ".h5"
        Badj_s, Badj_t, Badj_p = self._read_dB_adjoint(h5_adj)
        Badj_contra = [Badj_s, Badj_t, Badj_p]
        Badj_cov = _covariant_from_contravariant(Badj_contra, self._gsub)
        dqomega_dt = Badj_cov[1]
        dqomega_dp = Badj_cov[2]
        return dqomega_dt, dqomega_dp

    # ------------------------------------------------------------------
    # Shape gradient assembly
    # (ported from vacuum_adjoint.compute_shape_gradient for QS + iota)
    # ------------------------------------------------------------------

    def _assemble_shape_gradient(self, B0, Bsuptheta, Bsupphi,
                                 Bsubtheta, Bsubphi, Bsubs,
                                 dqomega_dt, dqomega_dp,
                                 qalpha, iota, dldt, dldp,
                                 dBcontrajac_ds,
                                 vQS, dvQS_dt, dvQS_dp, divgamma_B,
                                 helicity_QS, abs_nabla_psi,
                                 Bmag, dBmag_dt, dBmag_dp, dBmag_ds,
                                 abs_nabla_alpha, dabs_nabla_alpha_dt,
                                 dabs_nabla_alpha_dp, norm_QS):
        """Assemble the full shape gradient G(θ,φ)."""
        nax = np.newaxis
        G = np.zeros_like(self.thetas_2d)

        # Term 1: B · ∇q_ω / B0
        G += (Bsuptheta * dqomega_dt + Bsupphi * dqomega_dp) / B0

        # Re-derive ∇_Γ α contravariant components (for this method scope)
        dalpha_dt = 1 + dldt; dalpha_dp = -iota + dldp
        gs = self._gsup_surf
        na_supt = gs[0, 0] * dalpha_dt + gs[0, 1] * dalpha_dp
        na_supp = gs[1, 0] * dalpha_dt + gs[1, 1] * dalpha_dp

        # Terms 2+3: (n·∇B − B·∇n) · ∇_Γα * q_α / B0
        # B·∇n (Cartesian)
        Bdotnablan = (Bsuptheta[nax, :, :] * self._dnormaldt
                      + Bsupphi[nax, :, :] * self._dnormaldp)
        Bdotn_subt = np.einsum('iab,iab->ab', Bdotnablan, self._drdtheta)
        Bdotn_subp = np.einsum('iab,iab->ab', Bdotnablan, self._drdphi)

        # n·∇B (via Bcontrajac ds, dt, dp)
        dBjac_ds_arr = np.array(dBcontrajac_ds)
        Bjac_t = Bsuptheta * self._jac
        Bjac_p = Bsupphi * self._jac

        # Fourier derivatives of Bjac components
        nmodes, xm, xn = _cosine_modes(self.mpol_adj, self.ntor_adj)
        cos_b, dcos_dt, dcos_dp = _basis_and_derivs(
            nmodes, xm, xn, 'cos', self.thetas_2d, self.phis_2d, self.nfp)
        ct = _fourier_coeffs(Bjac_t, cos_b)
        cp_c = _fourier_coeffs(Bjac_p, cos_b)
        dBjact_dt = np.einsum('i,iab->ab', ct, dcos_dt)
        dBjact_dp = np.einsum('i,iab->ab', ct, dcos_dp)
        dBjacp_dt = np.einsum('i,iab->ab', cp_c, dcos_dt)
        dBjacp_dp = np.einsum('i,iab->ab', cp_c, dcos_dp)

        dBjac_dt_cart = (dBjact_dt[nax] * self._drdtheta
                         + Bjac_t[nax] * self._d2rdtheta2
                         + dBjacp_dt[nax] * self._drdphi
                         + Bjac_p[nax] * self._d2rdthetadphi)
        dBjac_dp_cart = (dBjact_dp[nax] * self._drdtheta
                         + Bjac_t[nax] * self._d2rdthetadphi
                         + dBjacp_dp[nax] * self._drdphi
                         + Bjac_p[nax] * self._d2rdphi2)
        dBjac_ds_cart = (dBjac_ds_arr[1][nax] * self._drdtheta
                         + Bjac_t[nax] * self._d2rdsdtheta
                         + dBjac_ds_arr[2][nax] * self._drdphi
                         + Bjac_p[nax] * self._d2rdsdphi
                         + dBjac_ds_arr[0][nax] * self._drds)

        nsubs = self._nsubs[nax]
        ndotgradBjac = nsubs * (self._gsup[0, 0][nax] * dBjac_ds_cart
                                + self._gsup[0, 1][nax] * dBjac_dt_cart
                                + self._gsup[0, 2][nax] * dBjac_dp_cart)
        ndotgradjac = (self._nsups * self._djac_ds
                       + self._nsuptheta * self._djac_dt
                       + self._nsupphi * self._djac_dp)
        ndotnablaB_subt = (np.einsum('iab,iab->ab', ndotgradBjac, self._drdtheta)
                           - Bsubtheta * ndotgradjac) / self._jac
        ndotnablaB_subp = (np.einsum('iab,iab->ab', ndotgradBjac, self._drdphi)
                           - Bsubphi * ndotgradjac) / self._jac

        vec_subt = ndotnablaB_subt - Bdotn_subt
        vec_subp = ndotnablaB_subp - Bdotn_subp

        G += qalpha * (vec_subt * na_supt + vec_subp * na_supp) / B0

        # QS-specific terms
        h = helicity_QS
        factor = (iota - h) if h is not np.infty else 1.0

        # Curvature term
        if self.normalise_qs:
            G += self._summed_curvature * 0.5 * vQS**2 * (Bmag / B0)**4 / norm_QS
            ndotgradB = (self._jac / self._jac_surf
                         * (self._gsup[0, 0] * dBmag_ds
                            + self._gsup[0, 1] * dBmag_dt
                            + self._gsup[0, 2] * dBmag_dp))
            G -= 2 * vQS**2 * (Bmag / B0)**3 * ndotgradB / B0 / norm_QS
        else:
            G += self._summed_curvature * 0.5 * vQS**2 / norm_QS

        # ndotgradB for remaining terms
        ndotgradB = (self._jac / self._jac_surf
                     * (self._gsup[0, 0] * dBmag_ds
                        + self._gsup[0, 1] * dBmag_dt
                        + self._gsup[0, 2] * dBmag_dp))

        # B · ∇(vQS) and div_Γ(B) terms
        divgamma_BvQS = Bsuptheta * dvQS_dt + Bsupphi * dvQS_dp + vQS * divgamma_B
        G -= divgamma_BvQS * ndotgradB / B0**2 / norm_QS

        # Tangential gradient of |B|
        nablaG_Bmag_supt = gs[0, 0] * dBmag_dt + gs[0, 1] * dBmag_dp
        nablaG_Bmag_supp = gs[1, 0] * dBmag_dt + gs[1, 1] * dBmag_dp
        G += vQS * (vec_subt * nablaG_Bmag_supt + vec_subp * nablaG_Bmag_supp) / B0**2 / norm_QS

        # Cross-product terms with ∇_Γ α
        nablaG_vQS_supt = gs[0, 0] * dvQS_dt + gs[0, 1] * dvQS_dp
        nablaG_vQS_supp = gs[1, 0] * dvQS_dt + gs[1, 1] * dvQS_dp
        nablaG_absna_supt = gs[0, 0] * dabs_nabla_alpha_dt + gs[0, 1] * dabs_nabla_alpha_dp
        nablaG_absna_supp = gs[1, 0] * dabs_nabla_alpha_dt + gs[1, 1] * dabs_nabla_alpha_dp
        nablaG_f_supt = nablaG_vQS_supt - vQS / abs_nabla_alpha * nablaG_absna_supt
        nablaG_f_supp = nablaG_vQS_supp - vQS / abs_nabla_alpha * nablaG_absna_supp
        BxgradBdotgradf = ((-Bsuptheta * nablaG_f_supp + Bsupphi * nablaG_f_supt)
                           * self._jac_surf * ndotgradB)
        G += factor * abs_nabla_psi / B0 * BxgradBdotgradf / B0**2 / norm_QS

        # Second-order normal-curvature correction
        nablaG_alpha_nablan = (na_supt[nax] * self._dnormaldt
                               + na_supp[nax] * self._dnormaldp)
        nablaG_alpha_nablan_subt = np.einsum('iab,iab->ab', nablaG_alpha_nablan, self._drdtheta)
        nablaG_alpha_nablan_subp = np.einsum('iab,iab->ab', nablaG_alpha_nablan, self._drdphi)
        nablaG_alpha_nablan_dot_nablaG_alpha = (
            nablaG_alpha_nablan_subt * na_supt + nablaG_alpha_nablan_subp * na_supp)
        BxnormdotgradBmag = -(Bsubtheta * dBmag_dp - Bsubphi * dBmag_dt) / self._jac_surf
        G -= factor * vQS * abs_nabla_psi / B0 * (
            nablaG_alpha_nablan_dot_nablaG_alpha / abs_nabla_alpha**2
            - self._summed_curvature) * BxnormdotgradBmag / B0**2 / norm_QS

        return G

    # ------------------------------------------------------------------
    # Parameter derivatives: shape gradient → DOF gradient
    # ------------------------------------------------------------------

    def _param_derivs(self, shape_gradient):
        """
        Convert shape gradient G(θ,φ) to derivative wrt each boundary DOF.

        Uses the same formulation as adjoint_QS `convert_dfdomega_shape_grad.py`
        (Eq. from E.J. Paul's ALPOpt code), on the high-resolution adjoint grid.
        Returns a 1-D array ordered to match ``spec.boundary.x``.
        """
        boundary = self.spec.boundary
        mpol = boundary.mpol; ntor = boundary.ntor; nfp = boundary.nfp

        # Normal vector components (un-normalized, = jac_surf × n̂)
        Nx_jac = self._normal[0] * self._jac_surf
        Ny_jac = self._normal[1] * self._jac_surf
        Nz_jac = self._normal[2] * self._jac_surf

        # We accumulate derivatives in a dict keyed by (m, n)
        dfdrmnc = {}   # d J / d rc(m,n)
        dfdzmns = {}   # d J / d zs(m,n)

        for m in range(mpol + 1):
            for n in range(-ntor, ntor + 1):
                angle = m * self.thetas_2d - nfp * n * self.phis_2d
                cos_a = np.cos(angle); sin_a = np.sin(angle)
                dfdrmnc[(m, n)] = (
                    np.sum(cos_a * (Nx_jac * np.cos(self.phis_2d)
                                   + Ny_jac * np.sin(self.phis_2d))
                           * shape_gradient) * self.dtheta * self.dphi * nfp)
                if not (m == 0 and n == 0):
                    dfdzmns[(m, n)] = (
                        np.sum(sin_a * Nz_jac * shape_gradient)
                        * self.dtheta * self.dphi * nfp)

        # Map to simsopt DOF array (matches boundary.x ordering)
        names = boundary.local_dof_names
        grad = np.zeros(len(names))
        for idx, name in enumerate(names):
            # parse "rc(m,n)" or "zs(m,n)"
            kind = name[:2]   # 'rc' or 'zs'
            inner = name[3:-1]  # "m,n"
            parts = inner.split(',')
            m, n = int(parts[0]), int(parts[1])
            if kind == 'rc' and (m, n) in dfdrmnc:
                grad[idx] = dfdrmnc[(m, n)]
            elif kind == 'zs' and (m, n) in dfdzmns:
                grad[idx] = dfdzmns[(m, n)]
        return grad


# ---------------------------------------------------------------------------
# Helper: update SPEC namelist boundary arrays
# ---------------------------------------------------------------------------

def _update_spec_boundary(nml, var, values, mpol, ntor, parity):
    """Write a 1-D array of Fourier coefficients into a SPEC namelist."""
    nmodes_cos = (ntor + 1) + mpol * (2 * ntor + 1)
    nmodes_sin = ntor + mpol * (2 * ntor + 1)

    if parity == 'cos':
        _, xm, xn = _cosine_modes(mpol, ntor)
    else:
        _, xm, xn = _sine_modes(mpol, ntor)

    if var not in nml['physicslist']:
        nml['physicslist'][var] = {}
    for i, (m, n) in enumerate(zip(xm, xn)):
        if i < len(values):
            nml['physicslist'][var][(int(m), int(n))] = float(values[i])
