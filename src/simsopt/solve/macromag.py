from __future__ import annotations
import functools
import math
import os
import sys
import numpy as np
from scipy.constants import mu_0
from scipy.sparse.linalg import gmres
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union
from scipy.sparse import coo_matrix

if TYPE_CHECKING:  # pragma: no cover
    from simsopt.field import BiotSavart, InterpolatedField

__all__ = ["MacroMag", "Tiles", "assemble_blocks_subset", "muse2tiles", "build_prism", "rotation_angle", "get_rotmat"]

_INV4PI  = 1.0 / (4.0 * math.pi)
_MACHEPS = np.finfo(np.float64).eps

# Perturbation factors for singularity handling in demag kernels
_EPS_HIGH: float = 1.0001
_EPS_LOW: float = 0.9999

# Tolerance for rotation angle gimbal-lock detection
_ROTATION_TOL: float = 1e-8

# Default GMRES settings
_DEFAULT_KRYLOV_TOL: float = 1e-6
_DEFAULT_KRYLOV_IT: int = 20

# Small epsilon to prevent division-by-zero when normalizing
_NORMALIZATION_EPS: float = 1e-30


def rotation_angle(R: np.ndarray, xyz: bool = False) -> tuple[float, float, float]:
    """
    Compute Euler angles from a 3×3 rotation matrix.

    This function is a small, self-contained replacement for
    ``coilpy.misc.rotation_angle``, included here to avoid an external runtime
    dependency in the MacroMag pipeline.

    Args:
        R: Rotation matrix, shape (3, 3).
        xyz: If True, interpret as the RxRyRz convention and return (alpha, beta, gamma)
            about x/y/z. If False, use the RzRyRx convention (coilpy default).

    Returns:
        (alpha, beta, gamma) in radians.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected R with shape (3,3), got {R.shape}")
    if xyz:
        return _rotation_angle_xyz(R)
    return _rotation_angle_zyx(R)


def _rotation_matrix(alpha: float, beta: float, gamma: float, *, xyz: bool = False) -> np.ndarray:
    r"""
    Construct a :math:`3\times 3` rotation matrix from Euler angles.

    Two conventions are supported:

    * **xyz** (``xyz=True``): :math:`R = R_x(\alpha)\,R_y(\beta)\,R_z(\gamma)`.
    * **zyx** (``xyz=False``, default): :math:`R = R_z(\alpha)\,R_y(\beta)\,R_x(\gamma)`.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles in radians.
    xyz : bool, optional
        If True, use the X-Y-Z convention; otherwise Z-Y-X.

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.
    """
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cb = math.cos(beta)
    sb = math.sin(beta)
    cc = math.cos(gamma)
    sc = math.sin(gamma)
    if xyz:
        # Rx(alpha) Ry(beta) Rz(gamma)
        return np.array(
            [
                [cb * cc, -cb * sc, sb],
                [sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb],
                [-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb],
            ],
            dtype=float,
        )
    # Rz(alpha) Ry(beta) Rx(gamma)
    return np.array(
        [
            [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
            [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
            [-sb, cb * sc, cb * cc],
        ],
        dtype=float,
    )


def _rotation_angle_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """
    Inverse of :func:`_rotation_matrix` for the Z-Y-X (RzRyRx) convention.

    Recovers Euler angles (alpha, beta, gamma) such that
    ``_rotation_matrix(alpha, beta, gamma, xyz=False)`` reproduces *R*.
    Handles the gimbal-lock case (cos(beta) ~ 0) by fixing alpha = 0.

    Args:
        R: Rotation matrix, shape ``(3, 3)``.

    Returns:
        Tuple ``(alpha, beta, gamma)`` in radians.
    """
    tol = _ROTATION_TOL
    # cos(beta) = 0 => beta=pi/2 or 3/2*pi; only alpha+/-gamma is determined
    if abs(R[0, 0]) < tol and abs(R[1, 0]) < tol:
        beta = math.asin(-R[2, 0])
        alpha_pm_gamma = math.atan2(R[1, 2], R[0, 2])
        alpha = 0.0
        gamma = R[2, 0] * alpha_pm_gamma
        return alpha, beta, gamma

    beta = math.asin(-R[2, 0])
    cb = math.cos(beta)
    alpha = math.atan2(cb * R[1, 0], cb * R[0, 0])
    sp = math.sin(alpha)
    cp = math.cos(alpha)
    gamma = math.atan2(sp * R[0, 2] - cp * R[1, 2], cp * R[1, 1] - sp * R[0, 1])

    if not np.allclose(_rotation_matrix(alpha, beta, gamma, xyz=False), R):
        beta = math.pi - math.asin(-R[2, 0])
        cb = math.cos(beta)
        alpha = math.atan2(cb * R[1, 0], cb * R[0, 0])
        sp = math.sin(alpha)
        cp = math.cos(alpha)
        gamma = math.atan2(sp * R[0, 2] - cp * R[1, 2], cp * R[1, 1] - sp * R[0, 1])
    return alpha, beta, gamma


def _rotation_angle_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """
    Inverse of :func:`_rotation_matrix` for the X-Y-Z (RxRyRz) convention.

    Recovers Euler angles (alpha, beta, gamma) such that
    ``_rotation_matrix(alpha, beta, gamma, xyz=True)`` reproduces *R*.
    Handles the gimbal-lock case (cos(beta) ~ 0) by fixing alpha = 0.

    Args:
        R: Rotation matrix, shape ``(3, 3)``.

    Returns:
        Tuple ``(alpha, beta, gamma)`` in radians.
    """
    tol = _ROTATION_TOL
    # cos(beta) = 0 => beta=pi/2 or 3/2*pi; only alpha-gamma is determined
    if abs(R[0, 0]) < tol and abs(R[0, 1]) < tol:
        beta = math.asin(R[0, 2])
        alpha_pm_gamma = math.atan2(R[1, 0], -R[2, 0])
        alpha = 0.0
        gamma = R[0, 2] * alpha_pm_gamma
        return alpha, beta, gamma

    beta = math.asin(R[0, 2])
    cb = math.cos(beta)
    alpha = math.atan2(-cb * R[1, 2], cb * R[2, 2])
    gamma = math.atan2(-cb * R[0, 1], cb * R[0, 0])

    if not np.allclose(_rotation_matrix(alpha, beta, gamma, xyz=True), R):
        beta = math.pi - math.asin(R[0, 2])
        cb = math.cos(beta)
        alpha = math.atan2(-cb * R[1, 2], cb * R[2, 2])
        gamma = math.atan2(-cb * R[0, 1], cb * R[0, 0])
    return alpha, beta, gamma


def _f_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    r"""
    Auxiliary function for the diagonal :math:`N_{xx}` demagnetization tensor component.

    For a uniformly magnetized rectangular prism with half-dimensions
    :math:`(a, b, c)` and evaluation point :math:`(x, y, z)`, this function
    evaluates the kernel that appears inside the :math:`\arctan` sum for the
    diagonal :math:`N_{xx}` component (Newell/Aharoni formulas):

    .. math::

        f(a,b,c,x,y,z) = \frac{(b - y)(c - z)}{(a - x)\,r},
        \qquad r = \sqrt{(a-x)^2 + (b-y)^2 + (c-z)^2}.

    The diagonal component is then assembled by a signed sum over the eight
    corner combinations (see :func:`_prism_N_local_nb`):

    .. math::

        N_{xx} = \frac{1}{4\pi}
        \sum_{s \in \{{\pm 1}\}^3}
        \arctan\!\bigl(f(a,b,c,\;s_x x_0,\;s_y y_0,\;s_z z_0)\bigr).

    Notes
    -----
    If the denominator :math:`(a - x)` is exactly zero in floating point, a
    small symmetric perturbation is used to approximate the limiting value.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Scalar value used inside :math:`\arctan` for the :math:`N_{xx}` entry.
    """
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = _EPS_HIGH, _EPS_LOW

    dx = half_a - x
    if dx == 0.0:
        dxh, dxl = half_a - x*eps_h, half_a - x*eps_l
        r_h = math.sqrt(dxh*dxh + (half_b - y)**2 + (half_c - z)**2)
        r_l = math.sqrt(dxl*dxl + (half_b - y)**2 + (half_c - z)**2)
        val_h = (half_b - y)*(half_c - z)/(dxh * r_h)
        val_l = (half_b - y)*(half_c - z)/(dxl * r_l)
        return 0.5*(val_h + val_l)
    else:
        r = math.sqrt(dx*dx + (half_b - y)**2 + (half_c - z)**2)
        return (half_b - y)*(half_c - z)/(dx * r)

def _g_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    r"""
    Auxiliary function for the diagonal :math:`N_{yy}` demagnetization tensor component.

    Like :func:`_f_3D`, this kernel appears inside the :math:`\arctan` sum for
    :math:`N_{yy}`:

    .. math::

        g(a,b,c,x,y,z) = \frac{(a - x)(c - z)}{(b - y)\,r},
        \qquad r = \sqrt{(a-x)^2 + (b-y)^2 + (c-z)^2}.

    Notes
    -----
    If the denominator :math:`(b - y)` is exactly zero in floating point, a
    small symmetric perturbation is used to approximate the limiting value.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Scalar value used inside :math:`\arctan` for the :math:`N_{yy}` entry.
    """
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = _EPS_HIGH, _EPS_LOW

    dy = half_b - y
    if dy == 0.0:
        dyh, dyl = half_b - y*eps_h, half_b - y*eps_l
        r_h = math.sqrt((half_a - x)**2 + dyh*dyh + (half_c - z)**2)
        r_l = math.sqrt((half_a - x)**2 + dyl*dyl + (half_c - z)**2)
        val_h = (half_a - x)*(half_c - z)/(dyh * r_h)
        val_l = (half_a - x)*(half_c - z)/(dyl * r_l)
        return 0.5*(val_h + val_l)
    else:
        r = math.sqrt((half_a - x)**2 + dy*dy + (half_c - z)**2)
        return (half_a - x)*(half_c - z)/(dy * r)


def _h_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    r"""
    Auxiliary function for the diagonal :math:`N_{zz}` demagnetization tensor component.

    Like :func:`_f_3D` and :func:`_g_3D`, this kernel appears inside the
    :math:`\arctan` sum for :math:`N_{zz}`:

    .. math::

        h(a,b,c,x,y,z) = \frac{(a - x)(b - y)}{(c - z)\,r},
        \qquad r = \sqrt{(a-x)^2 + (b-y)^2 + (c-z)^2}.

    Notes
    -----
    If the denominator :math:`(c - z)` is exactly zero in floating point, a
    small symmetric perturbation is used to approximate the limiting value.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Scalar value used inside :math:`\arctan` for the :math:`N_{zz}` entry.
    """
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = _EPS_HIGH, _EPS_LOW

    dz = half_c - z
    if dz == 0.0:
        dzh, dzl = half_c - z*eps_h, half_c - z*eps_l
        r_h = math.sqrt((half_a - x)**2 + (half_b - y)**2 + dzh*dzh)
        r_l = math.sqrt((half_a - x)**2 + (half_b - y)**2 + dzl*dzl)
        val_h = (half_a - x)*(half_b - y)/(dzh * r_h)
        val_l = (half_a - x)*(half_b - y)/(dzl * r_l)
        return 0.5*(val_h + val_l)
    else:
        r = math.sqrt((half_a - x)**2 + (half_b - y)**2 + dz*dz)
        return (half_a - x)*(half_b - y)/(dz * r)

def _FF_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    r"""
    Log-kernel helper for the off-diagonal :math:`N_{xy}` demagnetization component.

    The off-diagonal demag entries use :math:`\ln`-ratio terms assembled from
    auxiliary functions evaluated at the eight signed corner combinations.  For
    :math:`N_{xy}` the auxiliary function is

    .. math::

        F(a,b,c,x,y,z) = (c - z) + r,
        \qquad r = \sqrt{(a-x)^2 + (b-y)^2 + (c-z)^2},

    and the off-diagonal entry is

    .. math::

        N_{xy} = -\frac{1}{4\pi}
        \ln\!\Bigl(\frac{\text{numerator}}{\text{denominator}}\Bigr),

    where the numerator and denominator are specific signed products of
    :math:`F` (see :func:`_prism_N_local_nb`).

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Positive scalar used in ratios inside :math:`\ln` terms for
        :math:`N_{xy}`.
    """
    ha, hb, hc = a, b, c
    return (hc - z) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)

def _GG_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    r"""
    Log-kernel helper for the off-diagonal :math:`N_{yz}` demagnetization component.

    For :math:`N_{yz}` the auxiliary function is

    .. math::

        G(a,b,c,x,y,z) = (a - x) + r,
        \qquad r = \sqrt{(a-x)^2 + (b-y)^2 + (c-z)^2}.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Positive scalar used in ratios inside :math:`\ln` terms for
        :math:`N_{yz}`.
    """
    ha, hb, hc = a, b, c
    return (ha - x) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)

def _HH_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    r"""
    Log-kernel helper for the off-diagonal :math:`N_{xz}` demagnetization component.

    For :math:`N_{xz}` the auxiliary function is

    .. math::

        H(a,b,c,x,y,z) = (b - y) + r,
        \qquad r = \sqrt{(a-x)^2 + (b-y)^2 + (c-z)^2}.

    See :func:`getF_limit` for the stabilized limit evaluation used when the
    raw products underflow.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Positive scalar used in ratios inside :math:`\ln` terms for
        :math:`N_{xz}`.
    """
    ha, hb, hc = a, b, c
    return (hb - y) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)


def getF_limit(
    a: float, b: float, c: float,
    x: float, y: float, z: float,
    func: callable,
) -> float:
    """
    Numerically stable limit for the off-diagonal log-kernel ratio.

    The off-diagonal demag components use terms of the form
    ``log(numerator/denominator)``, where numerator and denominator are products
    of the auxiliary functions evaluated at signed corner combinations. At some
    symmetric points, numerator and/or denominator can underflow or become
    exactly 0 in floating point. This helper computes the ratio using a small
    symmetric perturbation and returns a stabilized approximation to the limit.

    Concretely, for a chosen off-diagonal kernel ``F`` (one of :func:`_FF_3D`,
    :func:`_GG_3D`, :func:`_HH_3D`), we form the ratio::

        ratio = Π F(+a,+b,+c, x,y,z) * F(-a,-b,+c, x,y,z) * F(+a,-b,-c, x,y,z) * F(-a,+b,-c, x,y,z)
                -------------------------------------------------------------------------------------
                Π F(+a,-b,+c, x,y,z) * F(-a,+b,+c, x,y,z) * F(+a,+b,-c, x,y,z) * F(-a,-b,-c, x,y,z)

    and then use ``log(ratio)`` in the off-diagonal entry. When the raw products
    become numerically 0, we compute numerator/denominator at ``(x,y,z)*(1±ε)``
    and return ``(nom_l + nom_h) / (denom_l + denom_h)`` as a stabilized
    approximation to the limit.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.
        func: One of :func:`_FF_3D`, :func:`_GG_3D`, :func:`_HH_3D`.

    Returns:
        Scalar approximation to the ratio (numerator/denominator) in the limit.
    """
    lim_h, lim_l = _EPS_HIGH, _EPS_LOW
    xl, yl, zl = x * lim_l, y * lim_l, z * lim_l
    xh, yh, zh = x * lim_h, y * lim_h, z * lim_h

    # flip only the half dim signs, keep coords fixed
    nom_l = ( func( +a, +b, +c, xl, yl, zl )
            * func( -a, -b, +c, xl, yl, zl )
            * func( +a, -b, -c, xl, yl, zl )
            * func( -a, +b, -c, xl, yl, zl ) )

    denom_l = ( func( +a, -b, +c, xl, yl, zl )
              * func( -a, +b, +c, xl, yl, zl )
              * func( +a, +b, -c, xl, yl, zl )
              * func( -a, -b, -c, xl, yl, zl ) )

    nom_h = ( func( +a, +b, +c, xh, yh, zh )
            * func( -a, -b, +c, xh, yh, zh )
            * func( +a, -b, -c, xh, yh, zh )
            * func( -a, +b, -c, xh, yh, zh ) )

    denom_h = ( func( +a, -b, +c, xh, yh, zh )
              * func( -a, +b, +c, xh, yh, zh )
              * func( +a, +b, -c, xh, yh, zh )
              * func( -a, -b, -c, xh, yh, zh ) )

    return (nom_l + nom_h) / (denom_l + denom_h)

def _prism_N_local_nb(
    a: float, b: float, c: float,
    x0: float, y0: float, z0: float,
) -> np.ndarray:
    r"""
    Demagnetization tensor for a uniformly magnetized rectangular prism.

    Evaluates the :math:`3 \times 3` demagnetization tensor block
    :math:`\underline{\underline{N}}_{ij}` for a rectangular prism with
    half-dimensions :math:`(a, b, c)`, at evaluation point
    :math:`(x_0, y_0, z_0)` in the prism's local coordinate system.

    The expressions follow the closed-form Newell/Aharoni-style formulas.
    Diagonal entries are signed sums of :math:`\arctan` terms and
    off-diagonals are :math:`\ln`-ratio terms:

    .. math::

        N_{xx} &= \frac{1}{4\pi}
            \sum_{s \in \{\pm 1\}^3} \arctan\bigl(f(\dots)\bigr), \\
        N_{xy} &= -\frac{1}{4\pi}
            \ln\!\Bigl(\frac{\text{num}(F)}{\text{den}(F)}\Bigr),

    with analogous expressions for the remaining components (see
    :func:`_f_3D`, :func:`_g_3D`, :func:`_h_3D`, :func:`_FF_3D`,
    :func:`_GG_3D`, :func:`_HH_3D`).  Near-singular ratios are handled
    via :func:`getF_limit`.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x0, y0, z0: Relative vector from source centre to target centre
            in the same local frame.

    Returns:
        Demag *operator* block, shape ``(3, 3)``, in local coordinates.

        This module uses the convention
        :math:`\mathbf{H}_d = \underline{\underline{N}}_{\text{op}} \,
        \mathbf{M}`, i.e. the returned tensor equals
        :math:`-\underline{\underline{N}}_{\text{std}}` (Newell/Aharoni).
        Consequently :math:`\operatorname{tr}(N_{\text{op}}) = -1` for
        self-terms.
    """
    N = np.empty((3,3), dtype=np.float64)
    sum_f = sum_g = sum_h = 0.0
    for si in (1.0, -1.0):
        for sj in (1.0, -1.0):
            for sk in (1.0, -1.0):
                xi, yi, zi = si*x0, sj*y0, sk*z0
                sum_f += math.atan(_f_3D(a, b, c, xi, yi, zi))
                sum_g += math.atan(_g_3D(a, b, c, xi, yi, zi))
                sum_h += math.atan(_h_3D(a, b, c, xi, yi, zi))
    N[0,0] = _INV4PI * sum_f
    N[1,1] = _INV4PI * sum_g
    N[2,2] = _INV4PI * sum_h

    def _off_diag(func, idx1, idx2):
        """Assemble a symmetric off-diagonal entry via log product ratios."""
        nom = ( func( +a, +b, +c, x0, y0, z0 )
            * func( -a, -b, +c, x0, y0, z0 )
            * func( +a, -b, -c, x0, y0, z0 )
            * func( -a, +b, -c, x0, y0, z0 ) )
        denom = ( func( +a, -b, +c, x0, y0, z0 )
                * func( -a, +b, +c, x0, y0, z0 )
                * func( +a, +b, -c, x0, y0, z0 )
                * func( -a, -b, -c, x0, y0, z0 ) )

        if nom == 0 or denom == 0:
            val = -_INV4PI * math.log( getF_limit(a,b,c,x0,y0,z0, func) )
        else:
            val = -_INV4PI * math.log(nom/denom)
        if denom and abs((nom-denom)/denom) < 10*_MACHEPS:
            val = 0.0
        N[idx1, idx2] = N[idx2, idx1] = val

    _off_diag(_FF_3D, 0, 1)
    _off_diag(_GG_3D, 1, 2)
    _off_diag(_HH_3D, 0, 2)

    return -N

_ASSEMBLE_BLOCKS_BATCH_I: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)
_ASSEMBLE_BLOCKS_BATCH_J: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)


def _pick_batch_size(n: int, candidates: tuple[int, ...]) -> int:
    """
    Return the smallest candidate >= *n* (or the maximum candidate).

    Used to select compile-time-fixed batch sizes for the JAX demag
    assembly kernel, avoiding recompilation for every unique tile count.

    Args:
        n: Actual number of tiles in this batch.
        candidates: Sorted tuple of supported batch sizes.

    Returns:
        Batch size to use for the kernel.
    """
    for c in candidates:
        if n <= c:
            return c
    return candidates[-1]


def _assemble_blocks_subset_numpy(
    centres: np.ndarray,
    half: np.ndarray,
    Rg2l: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
) -> np.ndarray:
    """
    Pure NumPy implementation of demag block assembly.

    Matches the JAX kernel logic exactly. Used when
    ``SIMSOPT_MACROMAG_USE_NUMPY=1`` to avoid JAX heap corruption on
    macOS (CPU fusion compiler).

    Args:
        centres: Tile centre positions, shape ``(n, 3)``.
        half: Half-dimensions per tile, shape ``(n, 3)``.
        Rg2l: Global-to-local rotation matrices, shape ``(n, 3, 3)``.
        I: Row indices of the block sub-matrix, shape ``(nI,)``.
        J: Column indices of the block sub-matrix, shape ``(nJ,)``.

    Returns:
        Demag tensor blocks, shape ``(nI, nJ, 3, 3)``.
    """
    centres = np.asarray(centres, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    Rg2l = np.asarray(Rg2l, dtype=np.float64)
    I = np.asarray(I).ravel()
    J = np.asarray(J).ravel()
    nI, nJ = I.size, J.size

    centres_I = centres[I]
    centres_J = centres[J]
    d = centres_I[:, None, :] - centres_J[None, :, :]
    R = Rg2l[I]
    d_loc = np.einsum("iab,ijb->ija", R, d)

    out = np.empty((nI, nJ, 3, 3), dtype=np.float64)
    for ii in range(nI):
        a, b, c = half[I[ii], 0], half[I[ii], 1], half[I[ii], 2]
        for jj in range(nJ):
            x0, y0, z0 = d_loc[ii, jj, 0], d_loc[ii, jj, 1], d_loc[ii, jj, 2]
            out[ii, jj] = _prism_N_local_nb(a, b, c, x0, y0, z0)
    return out


def assemble_blocks_subset(
    centres: np.ndarray,
    half: np.ndarray,
    Rg2l: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
) -> np.ndarray:
    """
    Assemble a subset of demag operator blocks.

    This helper returns the blocks ``N_op[I[ii], J[jj]]`` without forming the
    full dense ``(n, n, 3, 3)`` tensor. Uses NumPy when
    ``SIMSOPT_MACROMAG_USE_NUMPY=1`` (avoids JAX heap corruption on macOS);
    otherwise uses JAX-jitted kernel.

    Parameters
    ----------
    centres : ndarray, shape (n, 3)
        Tile centre positions in global coordinates.
    half : ndarray, shape (n, 3)
        Tile half-dimensions in local coordinates.
    Rg2l : ndarray, shape (n, 3, 3)
        Global-to-local rotation matrices for each tile.
    I : ndarray, shape (nI,)
        Target tile indices.
    J : ndarray, shape (nJ,)
        Source tile indices.

    Returns
    -------
    ndarray, shape (nI, nJ, 3, 3)
        Demag operator blocks for the requested pairs.

        The returned blocks use the module-wide convention that the demagnetizing
        field is obtained via ``H_demag = N_op @ M``. Relative to the standard
        Newell/Aharoni demagnetization tensor ``N_std`` (for which
        ``H_demag = -N_std @ M``), we return ``N_op = -N_std``.

    Notes
    -----
    Set ``SIMSOPT_MACROMAG_USE_NUMPY=1`` to use the pure NumPy backend (slower
    but avoids JAX CPU fusion compiler heap corruption on macOS).
    """
    I = np.asarray(I).ravel()
    J = np.asarray(J).ravel()
    if I.size == 0 or J.size == 0:
        return np.empty((I.size, J.size, 3, 3), dtype=np.float64)

    use_numpy = os.environ.get("SIMSOPT_MACROMAG_USE_NUMPY", "").lower() in ("1", "true", "yes")
    use_jax = os.environ.get("SIMSOPT_MACROMAG_USE_JAX", "").lower() in ("1", "true", "yes")
    if not use_numpy and not use_jax and sys.platform == "darwin":
        use_numpy = True  # Avoid JAX CPU fusion compiler heap corruption on macOS
    if use_numpy:
        return _assemble_blocks_subset_numpy(centres, half, Rg2l, I, J)

    import jax

    # Force CPU to avoid Metal-related heap corruption on macOS (see e.g. JAX issue #20750).
    jax.config.update("jax_platform_name", "cpu")
    # simsopt.geo sets this globally, but MacroMag can be used without importing geo.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    centres_np = np.asarray(centres, dtype=np.float64, order="C")
    half_np = np.asarray(half, dtype=np.float64, order="C")
    Rg2l_np = np.asarray(Rg2l, dtype=np.float64, order="C")
    I_np = np.asarray(I, dtype=np.int32, order="C")
    J_np = np.asarray(J, dtype=np.int32, order="C")

    centres_j = jax.device_put(centres_np)
    half_j = jax.device_put(half_np)
    Rg2l_j = jax.device_put(Rg2l_np)

    nI = int(I_np.size)
    nJ = int(J_np.size)

    i_batch = _pick_batch_size(nI, _ASSEMBLE_BLOCKS_BATCH_I)
    j_batch = _pick_batch_size(nJ, _ASSEMBLE_BLOCKS_BATCH_J)

    out = np.empty((nI, nJ, 3, 3), dtype=np.float64)
    kernel = _get_assemble_blocks_subset_kernel()

    for i0 in range(0, nI, i_batch):
        I_chunk = I_np[i0 : i0 + i_batch]
        nIi = int(I_chunk.size)
        if nIi < i_batch:
            I_pad = np.empty((i_batch,), dtype=np.int32)
            I_pad[:nIi] = I_chunk
            I_pad[nIi:] = 0
        else:
            I_pad = I_chunk

        I_pad_j = jnp.asarray(I_pad)

        for j0 in range(0, nJ, j_batch):
            J_chunk = J_np[j0 : j0 + j_batch]
            nJj = int(J_chunk.size)
            if nJj < j_batch:
                J_pad = np.empty((j_batch,), dtype=np.int32)
                J_pad[:nJj] = J_chunk
                J_pad[nJj:] = 0
            else:
                J_pad = J_chunk

            J_pad_j = jnp.asarray(J_pad)

            blocks = kernel(centres_j, half_j, Rg2l_j, I_pad_j, J_pad_j)
            blocks.block_until_ready()
            blocks_np = np.asarray(blocks)

            out[i0 : i0 + nIi, j0 : j0 + nJj] = blocks_np[:nIi, :nJj]

    return out


@functools.lru_cache(maxsize=1)
def _get_assemble_blocks_subset_kernel():
    """
    Return a cached JAX-jitted kernel for demag block assembly.

    The returned function has signature ``(centres, half, Rg2l, I, J)`` and
    returns blocks with shape ``(len(I), len(J), 3, 3)``.

    Notes
    -----
    JAX compilation is specialized to input shapes and dtypes. The returned
    callable is jitted once at creation; compiled executables are cached by JAX
    internally for each distinct input shape.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    inv4pi = 1.0 / (4.0 * math.pi)
    macheps = np.finfo(np.float64).eps

    # -- JAX-traceable replicas of the scalar demag helpers ----------------
    # These mirror the module-level _f_3D, _g_3D, … but use jnp operations
    # so they can be traced/compiled by JAX.  They are vectorised over
    # batched (nI, nJ) tile-pair arrays and use jnp.where instead of Python
    # if/else to remain compatible with jit.

    def _f_3D_jax(a, b, c, x, y, z):
        """JAX-traceable replica of :func:`_f_3D` (N_xx kernel)."""
        eps_h, eps_l = _EPS_HIGH, _EPS_LOW
        dx = a - x
        r = jnp.sqrt(dx * dx + (b - y) ** 2 + (c - z) ** 2)
        val = (b - y) * (c - z) / (dx * r)

        dxh = a - x * eps_h
        dxl = a - x * eps_l
        r_h = jnp.sqrt(dxh * dxh + (b - y) ** 2 + (c - z) ** 2)
        r_l = jnp.sqrt(dxl * dxl + (b - y) ** 2 + (c - z) ** 2)
        val_h = (b - y) * (c - z) / (dxh * r_h)
        val_l = (b - y) * (c - z) / (dxl * r_l)
        val_lim = 0.5 * (val_h + val_l)
        return jnp.where(dx == 0.0, val_lim, val)

    def _g_3D_jax(a, b, c, x, y, z):
        """JAX-traceable replica of :func:`_g_3D` (N_yy kernel)."""
        eps_h, eps_l = _EPS_HIGH, _EPS_LOW
        dy = b - y
        r = jnp.sqrt((a - x) ** 2 + dy * dy + (c - z) ** 2)
        val = (a - x) * (c - z) / (dy * r)

        dyh = b - y * eps_h
        dyl = b - y * eps_l
        r_h = jnp.sqrt((a - x) ** 2 + dyh * dyh + (c - z) ** 2)
        r_l = jnp.sqrt((a - x) ** 2 + dyl * dyl + (c - z) ** 2)
        val_h = (a - x) * (c - z) / (dyh * r_h)
        val_l = (a - x) * (c - z) / (dyl * r_l)
        val_lim = 0.5 * (val_h + val_l)
        return jnp.where(dy == 0.0, val_lim, val)

    def _h_3D_jax(a, b, c, x, y, z):
        """JAX-traceable replica of :func:`_h_3D` (N_zz kernel)."""
        eps_h, eps_l = _EPS_HIGH, _EPS_LOW
        dz = c - z
        r = jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + dz * dz)
        val = (a - x) * (b - y) / (dz * r)

        dzh = c - z * eps_h
        dzl = c - z * eps_l
        r_h = jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + dzh * dzh)
        r_l = jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + dzl * dzl)
        val_h = (a - x) * (b - y) / (dzh * r_h)
        val_l = (a - x) * (b - y) / (dzl * r_l)
        val_lim = 0.5 * (val_h + val_l)
        return jnp.where(dz == 0.0, val_lim, val)

    def _FF_3D_jax(a, b, c, x, y, z):
        """JAX-traceable replica of :func:`_FF_3D` (N_xy log-kernel)."""
        return (c - z) + jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2)

    def _GG_3D_jax(a, b, c, x, y, z):
        """JAX-traceable replica of :func:`_GG_3D` (N_yz log-kernel)."""
        return (a - x) + jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2)

    def _HH_3D_jax(a, b, c, x, y, z):
        """JAX-traceable replica of :func:`_HH_3D` (N_xz log-kernel)."""
        return (b - y) + jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2)

    def _getF_limit_jax(a, b, c, x, y, z, func):
        """JAX-traceable replica of :func:`getF_limit`."""
        lim_h, lim_l = _EPS_HIGH, _EPS_LOW
        xl, yl, zl = x * lim_l, y * lim_l, z * lim_l
        xh, yh, zh = x * lim_h, y * lim_h, z * lim_h

        nom_l = (
            func(+a, +b, +c, xl, yl, zl)
            * func(-a, -b, +c, xl, yl, zl)
            * func(+a, -b, -c, xl, yl, zl)
            * func(-a, +b, -c, xl, yl, zl)
        )
        denom_l = (
            func(+a, -b, +c, xl, yl, zl)
            * func(-a, +b, +c, xl, yl, zl)
            * func(+a, +b, -c, xl, yl, zl)
            * func(-a, -b, -c, xl, yl, zl)
        )

        nom_h = (
            func(+a, +b, +c, xh, yh, zh)
            * func(-a, -b, +c, xh, yh, zh)
            * func(+a, -b, -c, xh, yh, zh)
            * func(-a, +b, -c, xh, yh, zh)
        )
        denom_h = (
            func(+a, -b, +c, xh, yh, zh)
            * func(-a, +b, +c, xh, yh, zh)
            * func(+a, +b, -c, xh, yh, zh)
            * func(-a, -b, -c, xh, yh, zh)
        )
        return (nom_l + nom_h) / (denom_l + denom_h)

    def prism_N_op_jax(a, b, c, x0, y0, z0):
        """JAX-traceable, vectorised replica of :func:`_prism_N_local_nb`."""
        # diagonal via signed corner combos
        signs = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=jnp.float64,
        )
        xi = signs[:, 0, None, None] * x0[None, :, :]
        yi = signs[:, 1, None, None] * y0[None, :, :]
        zi = signs[:, 2, None, None] * z0[None, :, :]

        a_ = a[None, :, None]
        b_ = b[None, :, None]
        c_ = c[None, :, None]

        sum_f = jnp.sum(jnp.arctan(_f_3D_jax(a_, b_, c_, xi, yi, zi)), axis=0)
        sum_g = jnp.sum(jnp.arctan(_g_3D_jax(a_, b_, c_, xi, yi, zi)), axis=0)
        sum_h = jnp.sum(jnp.arctan(_h_3D_jax(a_, b_, c_, xi, yi, zi)), axis=0)

        Nxx = inv4pi * sum_f
        Nyy = inv4pi * sum_g
        Nzz = inv4pi * sum_h

        def off_diag(func):
            a2 = a[:, None]
            b2 = b[:, None]
            c2 = c[:, None]
            nom = (
                func(+a2, +b2, +c2, x0, y0, z0)
                * func(-a2, -b2, +c2, x0, y0, z0)
                * func(+a2, -b2, -c2, x0, y0, z0)
                * func(-a2, +b2, -c2, x0, y0, z0)
            )
            denom = (
                func(+a2, -b2, +c2, x0, y0, z0)
                * func(-a2, +b2, +c2, x0, y0, z0)
                * func(+a2, +b2, -c2, x0, y0, z0)
                * func(-a2, -b2, -c2, x0, y0, z0)
            )
            ratio_lim = _getF_limit_jax(a2, b2, c2, x0, y0, z0, func)
            use_lim = (nom == 0.0) | (denom == 0.0)
            ratio = jnp.where(use_lim, ratio_lim, nom / denom)
            val = -inv4pi * jnp.log(ratio)
            close = (denom != 0.0) & (jnp.abs((nom - denom) / denom) < 10.0 * macheps)
            return jnp.where(close, 0.0, val)

        Nxy = off_diag(_FF_3D_jax)
        Nyz = off_diag(_GG_3D_jax)
        Nxz = off_diag(_HH_3D_jax)

        N = jnp.stack(
            [
                jnp.stack([Nxx, Nxy, Nxz], axis=-1),
                jnp.stack([Nxy, Nyy, Nyz], axis=-1),
                jnp.stack([Nxz, Nyz, Nzz], axis=-1),
            ],
            axis=-2,
        )
        return -N  # operator convention

    def _kernel_impl(centres, half, Rg2l, I, J):
        """Compute demag blocks for tile pairs (I, J); returns ``(nI, nJ, 3, 3)``."""
        centres_I = centres[I]
        centres_J = centres[J]
        d = centres_I[:, None, :] - centres_J[None, :, :]  # (nI, nJ, 3)
        R = Rg2l[I]  # (nI, 3, 3)
        d_loc = jnp.einsum("iab,ijb->ija", R, d)  # (nI, nJ, 3)
        x0 = d_loc[:, :, 0]
        y0 = d_loc[:, :, 1]
        z0 = d_loc[:, :, 2]
        a = half[I][:, 0]
        b = half[I][:, 1]
        c = half[I][:, 2]
        return prism_N_op_jax(a, b, c, x0, y0, z0)

    return jax.jit(_kernel_impl)


class MacroMag:
    r"""
    Device-scale macromagnetics solver for coupled permanent-magnet arrays.

    This class implements the block-level macromagnetic model described in
    `Ulrich, Haberle & Kaptanoglu (2025) <https://arxiv.org/abs/2512.14997>`_.
    It computes the demagnetization tensor for arrays of uniformly magnetized
    rectangular prisms and solves for the coupled equilibrium magnetization
    under finite-permeability and anisotropy effects.

    **Physical model.**
    Each magnet block is modelled as a uniformly magnetized rectangular prism
    whose crystallographic easy axis :math:`\hat{\mathbf{u}}` is fixed after
    fabrication.  The magnetization is decomposed into remanent and induced
    parts via an anisotropic susceptibility tensor

    .. math::

        \boldsymbol{\chi}
        = \chi_{\parallel}\,\hat{\mathbf{u}}\hat{\mathbf{u}}^{\!\top}
        + \chi_{\perp}\!\bigl(\mathbf{I}
          - \hat{\mathbf{u}}\hat{\mathbf{u}}^{\!\top}\bigr),

    where :math:`\chi_{\parallel} = \mu_{\text{ea}} - 1` and
    :math:`\chi_{\perp} = \mu_{\text{oa}} - 1` are the susceptibilities
    along and transverse to the easy axis.

    **Demagnetization tensor.**
    The demagnetizing field of the full array is encoded by the
    tensor :math:`\underline{\underline{N}}_{ij}`, which connects the
    demagnetizing field at block *i* to the magnetization at block *j*:

    .. math::

        H_{d,\alpha}^{(i)}
        = -\sum_{j} N_{ij,\alpha\beta}\, M_\beta^{(j)}.

    For rectangular prisms the :math:`3\times 3` blocks
    :math:`\underline{\underline{N}}_{ij}` are available in closed form
    (Newell/Aharoni-style formulas); see :func:`_prism_N_local_nb`.

    **Equilibrium condition.**
    Stationarity of the macromagnetic free energy yields the global
    linear system (Eq. 21 of the reference):

    .. math::

        \bigl[\delta_{ij}\,\mathbf{I}_3
        + \boldsymbol{\chi}_i\,\underline{\underline{N}}_{ij}\bigr]
        \,\mathbf{M}_j
        = M_{\text{rem}}\,\hat{\mathbf{u}}_i
        + \boldsymbol{\chi}_i\,\mathbf{H}_a,

    which is solved via GMRES (see :meth:`direct_solve`).

    **Usage.**
    Build a :class:`Tiles` container describing the magnet geometry and
    material properties, then pass it to ``MacroMag(tiles)``.  Call
    :meth:`fast_get_demag_tensor` to assemble
    :math:`\underline{\underline{N}}` and :meth:`direct_solve` to
    compute the equilibrium magnetization.
    """

    _INV4PI = _INV4PI
    _MACHEPS = _MACHEPS

    def __init__(
        self,
        tiles,
        bs_interp: Optional[Union["BiotSavart", "InterpolatedField"]] = None,
    ):
        """
        Create a macromagnetics helper for a given tile set.

        Parameters
        ----------
        tiles : Tiles
            Tile container holding geometry and material parameters.
        bs_interp : BiotSavart | InterpolatedField | None
            Optional coil-field evaluator. If provided, it must support
            ``set_points`` and ``B``.
        """
        self.tiles = tiles
        self.n = tiles.n
        self.centres = tiles.offset.copy()
        self.half    = (tiles.size * 0.5).astype(np.float64)
        self.Rg2l    = np.zeros((self.n, 3, 3), dtype=np.float64)
        self.bs_interp = bs_interp
        self._interp_cfg = dict(
            degree=3,   # interpolant degree
            nr=12, nphi=2, nz=12,
            pad_frac=0.05,
            nfp=1, stellsym=False,
            extra_pts=None,
        )

        for k in range(self.n):
            self.Rg2l[k], _ = MacroMag.get_rotation_matrices(*tiles.rot[k])

    @staticmethod
    def _f_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
        """Static wrapper; see module-level :func:`_f_3D`."""
        return _f_3D(a, b, c, x, y, z)

    @staticmethod
    def _g_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
        """Static wrapper; see module-level :func:`_g_3D`."""
        return _g_3D(a, b, c, x, y, z)

    @staticmethod
    def _h_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
        """Static wrapper; see module-level :func:`_h_3D`."""
        return _h_3D(a, b, c, x, y, z)

    @staticmethod
    def _FF_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
        """Static wrapper; see module-level :func:`_FF_3D`."""
        return _FF_3D(a, b, c, x, y, z)

    @staticmethod
    def _GG_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
        """Static wrapper; see module-level :func:`_GG_3D`."""
        return _GG_3D(a, b, c, x, y, z)

    @staticmethod
    def _HH_3D(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
        """Static wrapper; see module-level :func:`_HH_3D`."""
        return _HH_3D(a, b, c, x, y, z)

    @staticmethod
    def _prism_N_local_nb(a: float, b: float, c: float, x0: float, y0: float, z0: float) -> np.ndarray:
        """Static wrapper; see module-level :func:`_prism_N_local_nb`."""
        return _prism_N_local_nb(a, b, c, x0, y0, z0)

    @staticmethod
    def get_rotation_matrices(phi_x: float, phi_y: float, phi_z: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Build global→local and local→global rotation matrices from Euler angles.

        Parameters
        ----------
        phi_x, phi_y, phi_z : float
            Rotation angles (radians) about x, y, z.

        Returns
        -------
        (Rg2l, Rl2g) : tuple[ndarray, ndarray]
            Rotation matrices mapping global→local and local→global.
        """
        ax, ay, az = -phi_x, -phi_y, -phi_z
        RotX = np.array([[1,0,0],
                       [0,math.cos(ax),-math.sin(ax)],
                       [0,math.sin(ax), math.cos(ax)]])
        RotY = np.array([[math.cos(ay),0,math.sin(ay)],
                       [0,1,0],
                       [-math.sin(ay),0,math.cos(ay)]])
        RotZ = np.array([[math.cos(az),-math.sin(az),0],
                       [math.sin(az), math.cos(az),0],
                       [0,0,1]])
        Rg2l = RotZ @ RotY @ RotX    # global -> local
        Rl2g = RotX.T @ RotY.T @ RotZ.T  # local -> global

        return Rg2l, Rl2g

    def fast_get_demag_tensor(self, cache: bool = True):
        """
        Assemble (or reuse) the dense demag operator tensor for the current tiles.

        Parameters
        ----------
        cache : bool, optional
            If True, reuse a cached full tensor when it matches the current tile count.

        Returns
        -------
        ndarray, shape (n, n, 3, 3)
            Dense demag operator blocks ``N_op`` such that ``H_demag = N_op @ M``.
        """
        # Reuse a cached full tensor if it matches the current size
        if cache and hasattr(self, "_N_full") and self._N_full is not None and self._N_full.shape[:2] == (self.n, self.n):
            return self._N_full
        idx = np.arange(self.n, dtype=np.int32)
        N = assemble_blocks_subset(self.centres, self.half, self.Rg2l, idx, idx)
        self._N_full = N
        return N

    def load_coils(self, focus_file: Path, current_scale: float = 1.0) -> None:
        """
        Load coils from a FOCUS file and store a field evaluator.

        Parameters
        ----------
        focus_file : Path
            Path to the FOCUS coil file.
        current_scale : float, optional
            Multiplicative factor applied to all coil currents.

        Notes
        -----
        This method currently stores a :class:`~simsopt.field.BiotSavart` instance
        in ``self.bs_interp`` (despite the attribute name). An
        :class:`~simsopt.field.InterpolatedField` is not constructed unless you
        explicitly wrap the field elsewhere.
        """
        from simsopt.field import BiotSavart, Coil
        from simsopt.util.coil_optimization_helper_functions import read_focus_coils

        curves, currents, ncoils = read_focus_coils(str(focus_file))
        coils = [Coil(curves[i], currents[i]*current_scale) for i in range(ncoils)]

        if(current_scale != 1):
                print(f"[INFO Macromag] Scaled Coil currents scaled by {current_scale}x")

        bs = BiotSavart(coils)

        self.bs_interp = bs

    def coil_field_at(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate the applied field H from coils at a set of points.

        Parameters
        ----------
        pts : ndarray, shape (n_pts, 3)
            Evaluation points in meters.

        Returns
        -------
        ndarray, shape (n_pts, 3)
            Applied field in A/m (computed from B via ``H = B / μ0``).
        """
        if self.bs_interp is None:
            raise ValueError("Coils not loaded correctly")
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        self.bs_interp.set_points(pts)
        return self.bs_interp.B() / mu_0

    def _demag_field_at(self, pts: np.ndarray, demag_tensor: np.ndarray) -> np.ndarray:
        """
        Compute the demagnetizing field at a set of points for rectangular prisms.

        Parameters
        ----------
        pts : ndarray of shape (n_pts, 3)
            Coordinates of the evaluation points in global frame.
        demag_tensor : ndarray of shape (n_tiles, n_pts, 3, 3)
            Precomputed local demagnetization tensor for each tile and point.

            Notes
            -----
            This helper assumes the *standard* demag tensor convention ``N_std``
            (Newell/Aharoni), for which the demagnetizing field is
            ``H_demag = -N_std @ M``. Many routines in this module instead work
            with the operator ``N_op = -N_std`` (so that ``H_demag = N_op @ M``);
            if you pass such an operator here, you must account for the sign.

        Returns
        -------
        H : ndarray of shape (n_pts, 3)
            Demagnetizing field (A/m) at each evaluation point in global frame.
        """

        n, n_pts = self.tiles.n, pts.shape[0]
        H = np.zeros((n_pts,3), dtype=np.float64)

        # Precompute rotations and local M once
        Rg2l = np.zeros((n,3,3), dtype=np.float64)
        Rl2g = np.zeros((n,3,3), dtype=np.float64)
        M_loc = np.zeros((n,3), dtype=np.float64)
        for i in range(n):
            Rg2l[i], Rl2g[i] = self.get_rotation_matrices(*self.tiles.rot[i])
            M_loc[i] = Rg2l[i] @ self.tiles.M[i]

        # Now accumulate H
        for i in range(n):
            # demag_local[i] is shape (n_pts,3,3)
            # multiply each (3×3) by the same M_loc[i] -> yields (n_pts,3)
            # This uses the standard convention H = -N_std @ M.
            H_loc = -np.einsum('pqr,r->pq', demag_tensor[i], M_loc[i])
            # rotate back to global
            H += (Rl2g[i] @ H_loc.T).T

        return H

    # ------------------------------------------------------------------
    # Shared helpers for direct_solve* methods
    # ------------------------------------------------------------------

    def _validate_coils_and_Ha(
        self,
        use_coils: bool,
        H_a_override: np.ndarray | None,
    ) -> np.ndarray | None:
        """Validate coil/Ha inputs and return a cleaned ``H_a_override``."""
        if use_coils and (self.bs_interp is None) and (H_a_override is None):
            raise ValueError(
                "use_coils=True, but no coil field evaluator is loaded "
                "and no H_a_override was provided."
            )
        if H_a_override is not None:
            H_a_override = np.asarray(H_a_override, dtype=np.float64, order="C")
            if H_a_override.shape != (self.centres.shape[0], 3):
                raise ValueError("H_a_override must have shape (n_tiles, 3) in A/m.")
        return H_a_override

    def _get_applied_field(
        self,
        use_coils: bool,
        H_a_override: np.ndarray | None,
    ) -> np.ndarray:
        """Return the applied field at tile centres, shape ``(n, 3)``."""
        if H_a_override is not None:
            return H_a_override.astype(np.float64)
        if use_coils:
            return self.coil_field_at(self.centres).astype(np.float64)
        return np.zeros((self.tiles.n, 3), dtype=np.float64)

    @staticmethod
    def _compute_chi_and_rhs(
        tiles: "Tiles",
        H_a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute susceptibility tensors and RHS vector.

        Returns
        -------
        chi_tensor : ndarray, shape ``(n, 3, 3)``
        chi_parallel : ndarray, shape ``(n,)``
        chi_perp : ndarray, shape ``(n,)``
        u_outer : ndarray, shape ``(n, 3, 3)``
        b : ndarray, shape ``(3*n,)``
        """
        u = tiles.u_ea
        m_rem = tiles.M_rem
        chi_parallel = tiles.mu_r_ea - 1.0
        chi_perp = tiles.mu_r_oa - 1.0
        chi_diff = chi_parallel - chi_perp

        u_dot_Ha = np.sum(u * H_a, axis=1)
        chi_Ha = (
            chi_perp[:, None] * H_a
            + chi_diff[:, None] * u_dot_Ha[:, None] * u
        )
        b = (m_rem[:, None] * u + chi_Ha).ravel()

        u_outer = np.einsum('ik,il->ikl', u, u, optimize=True)
        chi_tensor = (
            chi_perp[:, None, None] * np.eye(3)[None, :, :]
            + chi_diff[:, None, None] * u_outer
        )
        return chi_tensor, chi_parallel, chi_perp, u_outer, b

    @staticmethod
    def _make_gmres_callback(prefix: str = "") -> callable:
        """Build a stateful GMRES callback that prints iteration and residual norm."""
        iteration = [0]
        tag = f"[{prefix}] " if prefix else ""

        def _callback(residual_norm: float) -> None:
            iteration[0] += 1
            print(f"{tag}GMRES iteration {iteration[0]}: residual norm = {residual_norm:.6e}")
        return _callback

    def _check_trivial_chi(
        self,
        chi_parallel: np.ndarray,
        chi_perp: np.ndarray,
        use_coils: bool,
        H_a_override: np.ndarray | None,
    ) -> bool:
        """Return True if χ=0 and no external field => trivial solution."""
        if np.allclose(chi_parallel, 0.0) and np.allclose(chi_perp, 0.0):
            if (not use_coils) and (H_a_override is None):
                self.tiles.M = self.tiles.M_rem[:, None] * self.tiles.u_ea
                return True
        return False

    def direct_solve(self,
        use_coils: bool = False,
        krylov_tol: float = 1e-6,
        krylov_it: int = 20,
        N_new_rows: np.ndarray = None,
        N_new_cols: np.ndarray | None = None,
        N_new_diag: np.ndarray | None = None,
        print_progress: bool = False,
        x0: np.ndarray | None = None,
        H_a_override: Optional[np.ndarray] = None,
        A_prev: np.ndarray | None = None,
        prev_n: int = 0) -> tuple["MacroMag", Optional[np.ndarray]]:

        r"""
        Solve for the equilibrium tile magnetization via GMRES.

        Assembles and solves the macromagnetic equilibrium system
        (Eq. 21 of the reference):

        .. math::

            \bigl[\delta_{ij}\,\mathbf{I}_3
            + \boldsymbol{\chi}_i\,
            \underline{\underline{N}}_{ij}\bigr]\,\mathbf{M}_j
            = M_{\text{rem}}\,\hat{\mathbf{u}}_i
            + \boldsymbol{\chi}_i\,\mathbf{H}_a,

        where :math:`\boldsymbol{\chi}_i` is the per-tile anisotropic
        susceptibility tensor in global coordinates and
        :math:`\underline{\underline{N}}_{ij}` is the demagnetization
        operator produced by :func:`_prism_N_local_nb`.

        Parameters
        ----------
        use_coils : bool
            If True, include the coil field in the applied-field term (unless
            ``H_a_override`` is provided).
        krylov_tol : float
            Relative tolerance passed to :func:`scipy.sparse.linalg.gmres` as ``rtol``.
        krylov_it : int
            Maximum number of GMRES iterations (``maxiter``).
        N_new_rows, N_new_cols, N_new_diag :
            Demag operator blocks used for assembly. Two modes are supported:

            - Initial assembly (``prev_n == 0``): ``N_new_rows`` must be the full
              dense tensor of shape ``(n, n, 3, 3)`` and ``A_prev`` is ignored.
            - Incremental assembly (``prev_n > 0``): supply the new block rows/cols:
              ``N_new_rows`` has shape ``(n-prev_n, prev_n, 3, 3)``,
              ``N_new_cols`` has shape ``(prev_n, n-prev_n, 3, 3)``, and
              ``N_new_diag`` has shape ``(n-prev_n, n-prev_n, 3, 3)``. ``A_prev``
              must have shape ``(3*prev_n, 3*prev_n)``.
        print_progress : bool
            If True, print per-iteration residual norms from GMRES.
        x0 : ndarray or None
            Optional initial guess for GMRES (shape ``(3*n,)``).
        H_a_override : ndarray or None
            Optional applied field per tile in A/m (shape ``(n, 3)``). If provided,
            it overrides coil evaluation even when ``use_coils=True``.
        A_prev : ndarray or None
            Previous system matrix for incremental assembly (shape ``(3*prev_n, 3*prev_n)``).
        prev_n : int
            Number of tiles represented by ``A_prev``.

        Returns
        -------
        self : MacroMag
            Updated object (``tiles.M`` is overwritten).
        A : ndarray or None
            Dense system matrix used in the solve, or None in trivial ``χ=0`` cases.
        """
        if prev_n == 0:
            if N_new_rows is None:
                raise ValueError(
                    "direct_solve requires N_new_rows (full demag tensor) for the initial pass "
                    "when prev_n == 0."
                )
        else:
            if N_new_rows is None or N_new_cols is None or N_new_diag is None:
                raise ValueError(
                    "direct_solve requires N_new_rows, N_new_cols, and N_new_diag for incremental "
                    "updates when prev_n > 0."
                )

        H_a_override = self._validate_coils_and_Ha(use_coils, H_a_override)
        tiles = self.tiles
        n = tiles.n

        chi_tensor, chi_parallel, chi_perp, _, b = self._compute_chi_and_rhs(
            tiles, self._get_applied_field(use_coils, H_a_override),
        )
        if self._check_trivial_chi(chi_parallel, chi_perp, use_coils, H_a_override):
            return self, None

        A = np.zeros((3 * n, 3 * n))
        A[:3 * prev_n, :3 * prev_n] = A_prev
        for i in range(prev_n, n):
            A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.eye(3)

        if prev_n > 0:
            chi_new = chi_tensor[prev_n:]
            chi_prev = chi_tensor[:prev_n]
            A[3 * prev_n:, :3 * prev_n] += np.einsum('iab,ijbc->iajc', chi_new, N_new_rows).reshape(3 * (n - prev_n), 3 * prev_n)
            A[:3 * prev_n, 3 * prev_n:] += np.einsum('iab,ijbc->iajc', chi_prev, N_new_cols).reshape(3 * prev_n, 3 * (n - prev_n))
            A[3 * prev_n:, 3 * prev_n:] += np.einsum('iab,ijbc->iajc', chi_new, N_new_diag).reshape(3 * (n - prev_n), 3 * (n - prev_n))
        else:
            A += np.einsum('iab,ijbc->iajc', chi_tensor, N_new_rows).reshape(3 * n, 3 * n)

        gmres_kw: dict = dict(rtol=krylov_tol, maxiter=krylov_it, x0=x0)
        if print_progress:
            gmres_kw.update(callback=self._make_gmres_callback("direct_solve"), callback_type='pr_norm')
        x, info = gmres(A, b, **gmres_kw)

        tiles.M = x.reshape(n, 3)
        return self, A

    def direct_solve_neighbor_sparse(
        self,
        neighbors: np.ndarray,
        use_coils: bool = False,
        krylov_tol: float = 1e-6,
        krylov_it: int = 200,
        print_progress: bool = False,
        x0: np.ndarray | None = None,
        H_a_override: Optional[np.ndarray] = None,
        drop_tol: float = 0.0,):
        """
        Sparse near-field demag solve.

        This builds A as a sparse 3n×3n matrix using only the demag couplings
        specified in `neighbors`, instead of the dense all-to-all demag tensor.

        Mathematically, we are solving

            (I + χ N_near) M = M_rem u + χ H_a

        where N_near is the demag operator truncated to the pairs (i,j)
        listed in `neighbors`. The far-field tail is discarded, with a
        controllable Frobenius-norm threshold `drop_tol` on the 3×3 blocks
        χ_i @ N_ij.

        Parameters
        ----------
        neighbors : ndarray, shape (n, k)
            neighbors[i, :] are local tile indices j (0 ≤ j < n) that
            couple to tile i via demag. Entries < 0 are ignored.
            It is recommended that i itself is included in neighbors[i, :].
        use_coils : bool
            If True, use the coil field (or H_a_override) as in direct_solve.
        krylov_tol, krylov_it :
            GMRES settings.
        print_progress : bool
            If True, print GMRES status.
        x0 : ndarray or None
            Optional initial guess for GMRES (length 3n).
        H_a_override : ndarray or None, shape (n, 3)
            External field at each tile. If not None, overrides coil field.
        drop_tol : float
            If > 0, any 3×3 block χ_i @ N_ij with Frobenius norm < drop_tol
            is dropped (far-field truncation).

        Returns
        -------
        self, A_sparse
            self with updated tiles.M, and the sparse A used in the solve.
        """
        neighbors = np.asarray(neighbors, dtype=np.int64)
        n = self.tiles.n

        if neighbors.shape[0] != n:
            raise ValueError(
                f"neighbors must have shape (n, k) with n={n}, "
                f"got {neighbors.shape}"
            )

        H_a_override = self._validate_coils_and_Ha(use_coils, H_a_override)
        tiles = self.tiles

        chi_tensor, chi_parallel, chi_perp, _, b = self._compute_chi_and_rhs(
            tiles, self._get_applied_field(use_coils, H_a_override),
        )
        if self._check_trivial_chi(chi_parallel, chi_perp, use_coils, H_a_override):
            return self, None

        # Build sparse A in COO format: A = I + χ N_near
        rows = []
        cols = []
        data = []

        # Identity blocks
        for i in range(n):
            base = 3 * i
            for a in range(3):
                rows.append(base + a)
                cols.append(base + a)
                data.append(1.0)

        # Near-field demag blocks
        for i in range(n):
            nbrs = neighbors[i]
            nbrs = nbrs[nbrs >= 0]
            if nbrs.size == 0:
                continue

            I = np.array([i], dtype=np.int64)
            J = np.asarray(nbrs, dtype=np.int64)
            # assemble_blocks_subset returns -N (because H_d = -N M), match direct_solve convention
            N_row = assemble_blocks_subset(self.centres, self.half, self.Rg2l, I, J)[0]
            # N_row has shape (len(J), 3, 3)

            chi_i = chi_tensor[i]  # (3,3)
            base_i = 3 * i

            for idx_j, j in enumerate(J):
                Nij = N_row[idx_j]  # (3,3)
                block = chi_i @ Nij  # (3,3)

                if drop_tol > 0.0:
                    # Frobenius norm threshold
                    if np.linalg.norm(block) < drop_tol:
                        continue

                base_j = 3 * j
                for a in range(3):
                    for c in range(3):
                        rows.append(base_i + a)
                        cols.append(base_j + c)
                        data.append(block[a, c])

        A_sparse = coo_matrix(
            (np.asarray(data, dtype=np.float64),
             (np.asarray(rows, dtype=np.int64),
              np.asarray(cols, dtype=np.int64))),
            shape=(3 * n, 3 * n),
        ).tocsr()

        gmres_kw: dict = dict(rtol=krylov_tol, maxiter=krylov_it, x0=x0)
        if print_progress:
            gmres_kw.update(callback=self._make_gmres_callback("neighbor_sparse"), callback_type='pr_norm')
        x, info = gmres(A_sparse, b, **gmres_kw)

        if info != 0:
            print(f"[MacroMag] direct_solve_neighbor_sparse GMRES did not fully converge, info = {info}")

        tiles.M = x.reshape(n, 3)
        return self, A_sparse



    def direct_solve_toggle(self,
        use_coils: bool = False,
        use_demag: bool = True,
        krylov_tol: float = 1e-6,
        krylov_it: int = 20,
        N_new_rows: np.ndarray = None,
        N_new_cols: np.ndarray | None = None,
        N_new_diag: np.ndarray | None = None,
        print_progress: bool = False,
        x0: np.ndarray | None = None,
        H_a_override: Optional[np.ndarray] = None,
        A_prev: np.ndarray | None = None,
        prev_n: int = 0) -> tuple["MacroMag", Optional[np.ndarray]]:

        """
        Variant of :meth:`direct_solve` with an explicit demag toggle.

        When ``use_demag=False``, the solve reduces to the local constitutive
        response (no tile–tile coupling): ``A = I`` and ``M = b``.

        Parameters
        ----------
        use_coils, krylov_tol, krylov_it, N_new_rows, N_new_cols, N_new_diag, print_progress,
        x0, H_a_override, A_prev, prev_n :
            Same meaning as in :meth:`direct_solve`.
        use_demag : bool
            If False, skip the demagnetization operator contribution.

        Returns
        -------
        self : MacroMag
            Updated object (``tiles.M`` is overwritten).
        A : ndarray or None
            Dense system matrix used in the solve, or None in trivial ``χ=0`` cases.
        """
        if use_demag and prev_n == 0:
            if N_new_rows is None:
                raise ValueError(
                    "direct_solve requires N_new_rows (full demag tensor) for the initial pass "
                    "when prev_n == 0."
                )
        elif use_demag and prev_n > 0:
            if N_new_rows is None or N_new_cols is None or N_new_diag is None:
                raise ValueError(
                    "direct_solve requires N_new_rows, N_new_cols, and N_new_diag for incremental "
                    "updates when prev_n > 0."
                )

        H_a_override = self._validate_coils_and_Ha(use_coils, H_a_override)
        tiles = self.tiles
        n = tiles.n

        chi_tensor, chi_parallel, chi_perp, _, b = self._compute_chi_and_rhs(
            tiles, self._get_applied_field(use_coils, H_a_override),
        )
        if self._check_trivial_chi(chi_parallel, chi_perp, use_coils, H_a_override):
            return self, None

        A = np.zeros((3 * n, 3 * n))
        A[:3 * prev_n, :3 * prev_n] = A_prev
        for i in range(prev_n, n):
            A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.eye(3)

        if use_demag:
            if prev_n > 0:
                chi_new = chi_tensor[prev_n:]
                chi_prev = chi_tensor[:prev_n]
                A[3 * prev_n:, :3 * prev_n] += np.einsum('iab,ijbc->iajc', chi_new, N_new_rows).reshape(3 * (n - prev_n), 3 * prev_n)
                A[:3 * prev_n, 3 * prev_n:] += np.einsum('iab,ijbc->iajc', chi_prev, N_new_cols).reshape(3 * prev_n, 3 * (n - prev_n))
                A[3 * prev_n:, 3 * prev_n:] += np.einsum('iab,ijbc->iajc', chi_new, N_new_diag).reshape(3 * (n - prev_n), 3 * (n - prev_n))
            else:
                A += np.einsum('iab,ijbc->iajc', chi_tensor, N_new_rows).reshape(3 * n, 3 * n)

        gmres_kw: dict = dict(rtol=krylov_tol, maxiter=krylov_it, x0=x0)
        if print_progress:
            gmres_kw.update(callback=self._make_gmres_callback("direct_solve_toggle"), callback_type='pr_norm')
        x, info = gmres(A, b, **gmres_kw)

        tiles.M = x.reshape(n, 3)
        return self, A




"""
MagTense “Tile” and “Tiles” classes
-----------------------------------

This code is adapted from the MagTense framework to eliminate its
GPU dependency and allow pure-CPU use in SimsOpt:

    - GitHub:  https://github.com/cmt-dtu-energy/MagTense
    - Paper:   R. Bjørk, E. B. Poulsen, K. K. Nielsen & A. R. Insinga,
               “MagTense: A micromagnetic framework using the analytical
               demagnetization tensor,” Journal of Magnetism and Magnetic Materials,
               vol. 535, 168057 (2021).
               https://doi.org/10.1016/j.jmmm.2021.168057

MagTense is authored and maintained by the DTU CMT group (Rasmus Bjørk et al.).
It provides fully analytic demagnetization‐tensor calculations for uniformly
magnetized tiles (rectangular prisms, cylinders, tetrahedra, etc.) in
micromagnetic simulations.

Here we re­implement the core Tile/Tiles functionality in pure Python/NumPy
to avoid the original MagTense GPU requirement.
"""


class Tile:
    """
    This is a proxy class to enable Tiles to be accessed as if they were a list of Tile
    objects. So, for example,

    t = Tiles(5)
    print(t[0].center_pos)  # [0, 0, 0]
    t[0].center_pos = [1, 1, 1]
    print(t[0].center_pos)  # [1, 1, 1]
    """

    def __init__(self, parent: Tiles, index: int) -> None:
        """Create a view into ``parent`` exposing tile ``index`` attributes."""
        self._parent = parent
        self._index = index

    def __getattr__(self, name) -> np.float64 | np.ndarray:
        """Read-through accessor: index into parent arrays when appropriate."""
        # Get the attribute from the parent
        attr = getattr(self._parent, name)
        # If the attribute is a np.ndarray, return the element at our index
        if isinstance(attr, np.ndarray):
            return attr[self._index]
        # Otherwise, just return the attribute itself
        return attr

    def __setattr__(self, name, value) -> None:
        """Write-through accessor: assign into parent arrays when appropriate."""
        if name in ["_parent", "_index"]:
            # For these attributes, set them normally
            super().__setattr__(name, value)
        else:
            # For all other attributes, set the value in the parent's attribute array
            attr = getattr(self._parent, name)
            if isinstance(attr, np.ndarray):
                attr[self._index] = value


class Tiles:
    r"""
    Container holding MagTense-style tile arrays for macromagnetic solves.

    Each tile represents a uniformly magnetized rectangular prism with
    material parameters (remanence :math:`M_{\text{rem}}`, easy-axis
    direction :math:`\hat{\mathbf{u}}`, and permeabilities
    :math:`\mu_{\parallel}`, :math:`\mu_{\perp}`) needed by
    :class:`MacroMag`.

    This class is modelled after the MagTense ``MagTile`` Fortran derived
    type but implemented in pure Python/NumPy.

    Attributes
    ----------
    n : int
        Number of tiles.
    offset : ndarray, shape ``(n, 3)``
        Centre positions of each tile in global coordinates.
    size : ndarray, shape ``(n, 3)``
        Full side lengths ``(a, b, c)`` of each tile.
    rot : ndarray, shape ``(n, 3)``
        Euler angles ``(alpha, beta, gamma)`` describing the
        local-to-global rotation of each tile.
    M : ndarray, shape ``(n, 3)``
        Magnetisation vector :math:`\mathbf{M}_i` of each tile.
    u_ea : ndarray, shape ``(n, 3)``
        Easy-axis unit vectors :math:`\hat{\mathbf{u}}_i`.
    mu_r_ea, mu_r_oa : ndarray, shape ``(n,)``
        Relative permeabilities along the easy axis
        (:math:`\mu_{\parallel}`) and the two hard axes
        (:math:`\mu_{\perp}`).
    M_rem : ndarray, shape ``(n,)``
        Remanent magnetisation magnitude :math:`M_{\text{rem},i}`.
    tile_type : int
        Geometry flag (2 = prism).
    """

    def __init__(
        self,
        n: int,
        center_pos: list[float] | None = None,
        dev_center: list[float] | None = None,
        size: list[float] | None = None,
        vertices: list[float] | None = None,
        M_rem: float | list[float] | None = None,
        easy_axis: list[float] | None = None,
        mag_angle: list[float] | None = None,
        mu_r_ea: float | list[float] | None = None,
        mu_r_oa: float | list[float] | None = None,
        tile_type: int | list[int] | None = None,
        offset: list[float] | None = None,
        rot: list[float] | None = None,
        color: list[float] | None = None,
        magnet_type: list[int] | None = None,
    ) -> None:
        """
        Construct a tile container with ``n`` elements.

        Parameters
        ----------
        n : int
            Number of tiles.
        center_pos : list[float] or None
            Tile center positions. May be a single 3-vector (broadcast) or an
            array-like of shape ``(n, 3)``.
        dev_center : list[float] or None
            Tile center deviations. Same shape conventions as ``center_pos``.
        size : list[float] or None
            Tile sizes. Same shape conventions as ``center_pos``.
        vertices : list[float] or None
            Tile vertices. Same conventions as in :meth:`vertices` setter.
        M_rem : float | list[float] | None
            Remanent magnetization magnitude(s), in A/m.
        easy_axis : list[float] | None
            Easy-axis unit vector(s). If not provided, computed from ``mag_angle``.
        mag_angle : list[float] | None
            Euler angles used to define the easy axis when ``easy_axis`` is not given.
        mu_r_ea, mu_r_oa : float | list[float] | None
            Relative permeabilities along the easy axis and orthogonal directions.
        tile_type : int | list[int] | None
            Tile geometry type code.
        offset, rot, color, magnet_type : list[...] | None
            Additional MagTense-style metadata arrays.
        """
        self.n = n
        self.center_pos = center_pos
        self.dev_center = dev_center
        self.size = size
        self.vertices = vertices
        self.M_rem = M_rem

        if easy_axis is not None:
            self.u_ea = easy_axis
        else:
            self.set_easy_axis(mag_angle)

        self.mu_r_ea = mu_r_ea
        self.mu_r_oa = mu_r_oa
        self.tile_type = tile_type
        self.offset = offset
        self.rot = rot

        self.color = color
        self.magnet_type = magnet_type

        # Initialize other attributes
        self.stfcn_index = None
        self.incl_it = None
        self.M_rel = None
        self._use_sym = np.zeros(shape=(self.n), dtype=np.int32, order="F")
        self._sym_op = np.ones(shape=(self.n, 3), dtype=np.float64, order="F")

    def __str__(self) -> str:
        """Return a human-readable summary of the tile offsets."""
        res = ""
        for i in range(self.n):
            res += f"Tile_{i} with coordinates {self.offset[i]}.\n"
        return res

    def __getitem__(self, index: int) -> Tile:
        """Return a lightweight :class:`Tile` view for a single index."""
        if index < 0 or index >= self.n:
            idx_err = f"Index {index} out of bounds"
            raise IndexError(idx_err)
        return Tile(self, index)

    @property
    def n(self) -> int:
        """Number of tiles in the container."""
        return self._n

    @n.setter
    def n(self, val: int) -> None:
        """Set the number of tiles (initializes internal arrays on first use)."""
        self._n = val

    def _set_nx3_array(
        self,
        attr_name: str,
        val,
        dtype: type = np.float64,
    ) -> None:
        """Generic setter for ``(n, 3)``-shaped tile arrays.

        Handles the four standard input forms: ``None`` (no-op),
        ``(vec, i)`` tuple (single tile), scalar-like vector (broadcast),
        or full ``(n, 3)`` array.
        """
        if not hasattr(self, attr_name):
            setattr(self, attr_name, np.zeros((self.n, 3), dtype=dtype, order="F"))

        if val is None:
            return
        if isinstance(val, tuple):
            getattr(self, attr_name)[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            getattr(self, attr_name)[:] = np.asarray(val)
        elif len(val) == self.n:
            setattr(self, attr_name, np.asarray(val, dtype=dtype))

    @property
    def center_pos(self) -> np.ndarray:
        """Tile center positions, array of shape ``(n, 3)``."""
        return self._center_pos

    @center_pos.setter
    def center_pos(self, val) -> None:
        """Set tile center positions (supports None / tuple / broadcast / full array)."""
        self._set_nx3_array("_center_pos", val)

    @property
    def dev_center(self) -> np.ndarray:
        """Tile center deviations, array of shape ``(n, 3)``."""
        return self._dev_center

    @dev_center.setter
    def dev_center(self, val) -> None:
        """Set tile center deviations (same input conventions as :meth:`center_pos`)."""
        self._set_nx3_array("_dev_center", val)

    @property
    def size(self) -> np.ndarray:
        """Tile sizes (edge lengths), array of shape ``(n, 3)``."""
        return self._size

    @size.setter
    def size(self, val) -> None:
        """Set tile sizes (same input conventions as :meth:`center_pos`)."""
        self._set_nx3_array("_size", val)

    @property
    def vertices(self) -> np.ndarray:
        """Tile vertices, array of shape ``(n, 3, 4)`` (four 3D vertices per tile)."""
        return self._vertices

    @vertices.setter
    def vertices(self, val) -> None:
        """
        Set tile vertices.

        Accepted input forms:
        - ``None``: no change.
        - ``(verts, i)``: set a single tile ``i``.
        - ``verts`` as (3,4) or (4,3): broadcast to all tiles.
        - array-like of shape ``(n, 3, 4)`` or ``(n, 4, 3)``.
        """
        if not hasattr(self, "_vertices"):
            self._vertices = np.zeros(shape=(self.n, 3, 4), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            vert = np.asarray(val[0])
            if vert.shape == (3, 4):
                self._vertices[val[1]] = vert
            elif vert.shape == (4, 3):
                self._vertices[val[1]] = vert.T
            else:
                value_err = "Four 3-D vertices have to be defined!"
                raise ValueError(value_err)
        else:
            vert = np.asarray(val)
            if vert.shape == (3, 4):
                self._vertices[:] = vert
            elif vert.shape == (4, 3):
                self._vertices[:] = vert.T
            else:
                assert vert[0].shape == (3, 4)
                self._vertices = vert

    @property
    def tile_type(self) -> np.ndarray:
        """Tile geometry type codes (MagTense convention), shape ``(n,)``."""
        return self._tile_type

    @tile_type.setter
    def tile_type(self, val) -> None:
        """Set tile geometry type codes (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_tile_type"):
            self._tile_type = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._tile_type[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._tile_type = np.asarray([val for _ in range(self.n)], dtype=np.int32)
        elif len(val) == self.n:
            self._tile_type = np.asarray(val, dtype=np.int32)

    @property
    def offset(self) -> np.ndarray:
        """Tile offsets (centres in global coordinates), array of shape ``(n, 3)``."""
        return self._offset

    @offset.setter
    def offset(self, val) -> None:
        """Set tile offsets (same input conventions as :meth:`center_pos`)."""
        self._set_nx3_array("_offset", val)

    @property
    def rot(self) -> np.ndarray:
        """Tile rotation angles (radians), array of shape ``(n, 3)``."""
        return self._rot

    @rot.setter
    def rot(self, val) -> None:
        """Set tile rotations (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_rot"):
            self._rot = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._rot[val[1]] = (
                _euler_to_rot_axis(val[0])
                if self.tile_type[val[1]] == 7
                else np.asarray(val[0])
            )
        elif isinstance(val[0], (list, np.ndarray)):
            for i in range(self.n):
                self._rot[i] = (
                    _euler_to_rot_axis(val[i])
                    if self.tile_type[i] == 7
                    else np.asarray(val[i])
                )
        else:
            self._rot[0] = (
                _euler_to_rot_axis(val) if self.tile_type[0] == 7 else np.asarray(val)
            )

    @property
    def M(self) -> np.ndarray:
        """Tile magnetization vectors, array of shape ``(n, 3)`` (A/m)."""
        return self._M

    @M.setter
    def M(self, val) -> None:
        """Set tile magnetization vectors (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_M"):
            self._M = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._M[val[1]] = val[0]
        elif isinstance(val[0], (int, float)):
            assert len(val) == 3
            self._M = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            assert len(val[0]) == 3
            self._M = np.asarray(val)

    @property
    def u_ea(self) -> np.ndarray:
        """Easy-axis unit vectors, array of shape ``(n, 3)``."""
        return self._u_ea

    @u_ea.setter
    def u_ea(self, val) -> None:
        """Set easy-axis vectors (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_u_ea"):
            self._u_ea = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._u_ea[val[1]] = np.around(val[0] / np.linalg.norm(val[0]), decimals=9)
            self.M = (self.M_rem[val[1]] * self.u_ea[val[1]], val[1])
        else:
            if isinstance(val[0], (int, float)):
                val = [val for _ in range(self.n)]

            # Convert to numpy array for vectorized operations
            val = np.asarray(val)

            # Vectorized normalization
            ea_norms = np.linalg.norm(val, axis=1, keepdims=True)
            self._u_ea[:] = np.around(val / (ea_norms + _NORMALIZATION_EPS), decimals=9)

            # Vectorized M computation
            self._M[:] = self.M_rem[:, None] * self._u_ea

            # Vectorized orthogonal axis computation
            # Choose reference vector: [1,0,0] if y or z component is non-zero, else [0,1,0]
            w_choice = np.array([[1, 0, 0], [0, 1, 0]])
            w_mask = (val[:, 1] != 0) | (val[:, 2] != 0)
            w = w_choice[w_mask.astype(int)]  # (n, 3)

            # Vectorized cross products
            self._u_oa1[:] = np.around(np.cross(self._u_ea, w), decimals=9)
            self._u_oa2[:] = np.around(np.cross(self._u_ea, self._u_oa1), decimals=9)

    @property
    def u_oa1(self) -> np.ndarray:
        """First orthogonal axis unit vectors, array of shape ``(n, 3)``."""
        return self._u_oa1

    @u_oa1.setter
    def u_oa1(self, val) -> None:
        """Set first orthogonal axis vectors (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_u_oa1"):
            self._u_oa1 = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._u_oa1[val[1]] = val[0]
        elif isinstance(val[0], (int, float)):
            assert len(val) == 3
            self._u_oa1 = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            assert len(val[0]) == 3
            self._u_oa1 = np.asarray(val)

    @property
    def u_oa2(self) -> np.ndarray:
        """Second orthogonal axis unit vectors, array of shape ``(n, 3)``."""
        return self._u_oa2

    @u_oa2.setter
    def u_oa2(self, val) -> None:
        """Set second orthogonal axis vectors (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_u_oa2"):
            self._u_oa2 = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._u_oa2[val[1]] = val[0]
        elif isinstance(val[0], (int, float)):
            assert len(val) == 3
            self._u_oa2 = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            assert len(val[0]) == 3
            self._u_oa2 = np.asarray(val)

    @property
    def mu_r_ea(self) -> np.ndarray:
        """Relative permeability along the easy axis, array of shape ``(n,)``."""
        return self._mu_r_ea

    @mu_r_ea.setter
    def mu_r_ea(self, val) -> None:
        """Set easy-axis relative permeability (scalar or per-tile array)."""
        if not hasattr(self, "_mu_r_ea"):
            self._mu_r_ea = np.ones(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._mu_r_ea[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._mu_r_ea = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._mu_r_ea = np.asarray(val)

    @property
    def mu_r_oa(self) -> np.ndarray:
        """Relative permeability orthogonal to the easy axis, array of shape ``(n,)``."""
        return self._mu_r_oa

    @mu_r_oa.setter
    def mu_r_oa(self, val) -> None:
        """Set orthogonal relative permeability (scalar or per-tile array)."""
        if not hasattr(self, "_mu_r_oa"):
            self._mu_r_oa = np.ones(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._mu_r_oa[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._mu_r_oa = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._mu_r_oa = np.asarray(val)

    @property
    def M_rem(self) -> np.ndarray:
        """Remanent magnetization magnitude(s), array of shape ``(n,)`` (A/m)."""
        return self._M_rem

    @M_rem.setter
    def M_rem(self, val) -> None:
        """Set remanent magnetization magnitude(s) (scalar or per-tile array)."""
        if not hasattr(self, "_M_rem"):
            self._M_rem = np.zeros(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._M_rem[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._M_rem = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._M_rem = np.asarray(val)

    @property
    def color(self) -> np.ndarray:
        """Tile RGB colors for visualization, array of shape ``(n, 3)``."""
        return self._color

    @color.setter
    def color(self, val) -> None:
        """Set tile colors (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_color"):
            self._color = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._color[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._color[:] = np.asarray(val)
        elif len(val) == self.n:
            self._color = np.asarray(val)

    @property
    def magnet_type(self) -> np.ndarray:
        """Magnet material type codes, array of shape ``(n,)`` (MagTense convention)."""
        return self._magnet_type

    @magnet_type.setter
    def magnet_type(self, val) -> None:
        """Set magnet type codes (scalar or per-tile array)."""
        if not hasattr(self, "_magnet_type"):
            self._magnet_type = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._magnet_type[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._magnet_type = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._magnet_type = np.asarray(val)

    @property
    def stfcn_index(self) -> np.ndarray:
        """State-function index array, shape ``(n,)`` (MagTense convention)."""
        return self._stfcn_index

    @stfcn_index.setter
    def stfcn_index(self, val) -> None:
        """Set state-function index codes (scalar or per-tile array)."""
        if not hasattr(self, "_stfcn_index"):
            self._stfcn_index = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._stfcn_index[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._stfcn_index = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._stfcn_index = np.asarray(val)

    @property
    def incl_it(self) -> np.ndarray:
        """Inclusion flags for iteration/solve, shape ``(n,)`` (1 include, 0 exclude)."""
        return self._incl_it

    @incl_it.setter
    def incl_it(self, val) -> None:
        """Set inclusion flags (scalar or per-tile array)."""
        if not hasattr(self, "_incl_it"):
            self._incl_it = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._incl_it[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._incl_it = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._incl_it = np.asarray(val)

    @property
    def M_rel(self) -> np.ndarray:
        """Relative change in magnetization from last iteration, shape ``(n,)``."""
        return self._M_rel

    @M_rel.setter
    def M_rel(self, val) -> None:
        """Set relative magnetization-change values (scalar or per-tile array)."""
        if not hasattr(self, "_M_rel"):
            self._M_rel = np.zeros(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._M_rel[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._M_rel = [val for _ in range(self.n)]
        elif len(val) == self.n:
            self._M_rel = val

    @property
    def use_sym(self) -> np.ndarray:
        """Symmetry-use flags, shape ``(n,)``."""
        return self._use_sym

    @property
    def sym_op(self) -> np.ndarray:
        """Symmetry operator signs per tile and axis, shape ``(n, 3)``."""
        return self._sym_op

    def set_easy_axis(
        self, val: list | None = None, idx: int | None = None, seed: int = 42
    ) -> None:
        """
        Set the easy axis from spherical angles.

        If ``val`` is provided, it is interpreted as ``[polar_angle, azimuth]``
        with polar angle in ``[0, π]`` and azimuth in ``[0, 2π]`` (radians). If
        ``val`` is None, random angles are generated using ``seed``.

        If ``idx`` is None, the operation is applied to all tiles. Otherwise,
        only tile ``idx`` is updated.
        """
        if val is None:
            rng = np.random.default_rng(seed)
            rand = rng.random(size=(self.n, 2))

        if idx is None:
            for i in range(self.n):
                if val is None:
                    self._set_ea_i([np.pi, 2 * np.pi] * rand[i], i)
                elif isinstance(val[0], (int, float)):
                    self._set_ea_i(val, i)
                else:
                    self._set_ea_i(val[i], i)
        else:
            i_val = [np.pi, 2 * np.pi] * rand[idx] if val is None else val
            self._set_ea_i(i_val, idx)

    def _set_ea_i(self, val: list[float] | np.ndarray, i: int) -> None:
        """
        Set easy-axis and orthogonal axes for tile *i* from spherical angles.

        Args:
            val: ``[polar_angle, azimuth]`` in radians.
            i: Tile index.

        Raises:
            TypeError: If *val* is a scalar instead of a pair.
        """
        if isinstance(val, (int, float)):
            value_err = "Both spherical angles have to be set!"
            raise TypeError(value_err)
        else:
            polar_angle, azimuth = val
            self.u_ea = (
                np.array(
                    [
                        np.sin(polar_angle) * np.cos(azimuth),
                        np.sin(polar_angle) * np.sin(azimuth),
                        np.cos(polar_angle),
                    ]
                ),
                i,
            )
            self.u_oa1 = (
                np.array(
                    [
                        np.sin(polar_angle) * np.sin(azimuth),
                        np.sin(polar_angle) * (-np.cos(azimuth)),
                        0,
                    ]
                ),
                i,
            )
            self.u_oa2 = (
                np.array(
                    [
                        0.5 * np.sin(2 * polar_angle) * np.cos(azimuth),
                        0.5 * np.sin(2 * polar_angle) * np.sin(azimuth),
                        -1 * np.sin(polar_angle) ** 2,
                    ]
                ),
                i,
            )

    def _add_tiles(self, n: int) -> None:
        """Append space for ``n`` additional tiles to all internal arrays."""
        self._center_pos = np.append(
            self._center_pos,
            np.zeros(shape=(n, 3), dtype=np.float64, order="F"),
            axis=0,
        )
        self._dev_center = np.append(
            self._dev_center,
            np.zeros(shape=(n, 3), dtype=np.float64, order="F"),
            axis=0,
        )
        self._size = np.append(
            self._size, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._vertices = np.append(
            self._vertices,
            np.zeros(shape=(n, 3, 4), dtype=np.float64, order="F"),
            axis=0,
        )
        self._M = np.append(
            self._M, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._u_ea = np.append(
            self._u_ea, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._u_oa1 = np.append(
            self._u_oa1, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._u_oa2 = np.append(
            self._u_oa2, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._mu_r_ea = np.append(
            self._mu_r_ea, np.ones(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._mu_r_oa = np.append(
            self._mu_r_oa, np.ones(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._M_rem = np.append(
            self._M_rem, np.zeros(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._tile_type = np.append(
            self._tile_type, np.ones(n, dtype=np.int32, order="F"), axis=0
        )
        self._offset = np.append(
            self._offset, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._rot = np.append(
            self._rot, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._color = np.append(
            self._color, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._magnet_type = np.append(
            self._magnet_type, np.ones(n, dtype=np.int32, order="F"), axis=0
        )
        self._stfcn_index = np.append(
            self._stfcn_index, np.ones(shape=(n), dtype=np.int32, order="F"), axis=0
        )
        self._incl_it = np.append(
            self._incl_it, np.ones(shape=(n), dtype=np.int32, order="F"), axis=0
        )
        self._use_sym = np.append(
            self._use_sym, np.zeros(shape=(n), dtype=np.int32, order="F"), axis=0
        )
        self._sym_op = np.append(
            self._sym_op, np.ones(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._M_rel = np.append(
            self._M_rel, np.zeros(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._n += n

    def refine_prism(self, idx: int | float | list, mat: list) -> None:
        """
        Subdivide one or more prism tiles into a regular rectangular grid.

        Parameters
        ----------
        idx : int | float | list
            Tile index (or list of indices) to refine.
        mat : list
            Subdivision counts ``[nx, ny, nz]``.
        """
        if isinstance(idx, (int, float)):
            self._refine_prism_i(idx, mat)
        else:
            for i in idx:
                self._refine_prism_i(i, mat)

    def _refine_prism_i(self, i: int | float, mat: list) -> None:
        """Refine a single prism tile ``i`` into ``prod(mat)`` sub-prisms."""
        assert self.tile_type[i] == 2
        old_n = self.n
        self._add_tiles(np.prod(mat) - 1)
        R_mat = get_rotmat(self.rot[i])

        x_off, y_off, z_off = self.size[i] / mat
        c_0 = -self.size[i] / 2 + (self.size[i] / mat) / 2
        c_pts = [
            c_0 + np.array([x_off * j, y_off * k, z_off * m])
            for j in range(mat[0])
            for k in range(mat[1])
            for m in range(mat[2])
        ]
        ver_cube = (np.dot(R_mat, np.array(c_pts).T)).T + self.offset[i]

        for j in range(mat[0]):
            for k in range(mat[1]):
                for m in range(mat[2]):
                    cube_idx = j * mat[1] * mat[2] + k * mat[2] + m
                    idx = cube_idx + old_n
                    if idx == self.n:
                        self._offset[i] = ver_cube[cube_idx]
                        self._size[i] = self.size[i] / mat
                    else:
                        self._size[idx] = self.size[i] / mat
                        self._M[idx] = self.M[i]
                        self._M_rel[idx] = self.M_rel[i]
                        self._color[idx] = self.color[i]
                        self._magnet_type[idx] = self.magnet_type[i]
                        self._mu_r_ea[idx] = self.mu_r_ea[i]
                        self._mu_r_oa[idx] = self.mu_r_oa[i]
                        self._rot[idx] = self.rot[i]
                        self._tile_type[idx] = self.tile_type[i]
                        self._u_ea[idx] = self.u_ea[i]
                        self._u_oa1[idx] = self.u_oa1[i]
                        self._u_oa2[idx] = self.u_oa2[i]
                        self._offset[idx] = ver_cube[cube_idx]
                        self._incl_it[idx] = self.incl_it[i]



def get_rotmat(rot: np.ndarray) -> np.ndarray:
    """
    Compute a 3×3 rotation matrix from x/y/z Euler angles.

    This helper matches the convention used in CoilPy's
    ``coilpy.magtense_interface.get_rotmat``: the returned matrix is formed as::

        R = R_x(rot[0]) @ R_y(rot[1]) @ R_z(rot[2])

    where each ``R_*`` is the right-handed active rotation about the indicated
    axis. In the MagTense/CoilPy adapter code, this is used as a global-to-local
    rotation in the tile's local coordinate system.

    Parameters
    ----------
    rot : ndarray, shape (3,)
        Rotation angles (radians) about x, y, z.

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix ``R`` assembled in x→y→z order.
    """
    rot_x = np.asarray(
        (
            [1, 0, 0],
            [0, np.cos(rot[0]), -np.sin(rot[0])],
            [0, np.sin(rot[0]), np.cos(rot[0])],
        )
    )
    rot_y = np.asarray(
        (
            [np.cos(rot[1]), 0, np.sin(rot[1])],
            [0, 1, 0],
            [-np.sin(rot[1]), 0, np.cos(rot[1])],
        )
    )
    rot_z = np.asarray(
        (
            [np.cos(rot[2]), -np.sin(rot[2]), 0],
            [np.sin(rot[2]), np.cos(rot[2]), 0],
            [0, 0, 1],
        )
    )
    return rot_x @ rot_y @ rot_z


def _euler_to_rot_axis(euler: np.ndarray) -> np.ndarray:
    """
    Convert a direction vector to MagTense-style Euler angles for spheroids.

    MagTense represents spheroid orientation via a *rotation axis* rather than
    a full set of Euler angles. This helper maps a 3-vector ``euler`` (treated
    as a direction/axis) to Euler angles ``(rot_x, rot_y, 0)`` consistent with
    the local-coordinate rotation convention used in the MagTense interface.

    The implementation follows the steps encoded below:

    1) Rotate the axis by ``π/2`` about the y-axis:

       ``ax = R_y(π/2) @ euler``

    2) Compute yaw/pitch-like angles that align the local x-axis with ``ax``:

       ``rot_x = -atan2(ax_y, ax_x)``
       ``rot_y = arccos(ax_z / ||ax||) - π/2``

    Returns ``[rot_x, rot_y, 0]``.

    Parameters
    ----------
    euler : ndarray, shape (3,)
        Input axis vector.

    Returns
    -------
    ndarray, shape (3,)
        Euler angles ``(rot_x, rot_y, rot_z)`` in radians, with ``rot_z = 0``.
    """
    # symm-axis of geometry points in the direction of z-axis of L
    # Rotates given rotation axis with pi/2 around y_L''
    # Moves x-axis of L'' to z-axis
    # ax[0] = ax[2], ax[1] = ax[1], ax[2] = -ax[0]
    R_y = np.array(
        [
            [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
            [0, 1, 0],
            [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)],
        ]
    )
    ax = np.dot(R_y, np.asarray(euler).T)

    # Calculate the spherical coordinates: yaw and pitch
    # x_L'' has to match ax
    # Perform negative yaw around x_L and pitch around y_L'
    # The azimuthal angle is offset by pi/2 (zero position of x_L'')
    rot_x = -np.arctan2(ax[1], ax[0])
    rot_y = np.arccos(ax[2] / np.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)) - np.pi / 2

    return np.array([rot_x, rot_y, 0])

"""
CoilPy‐derived muse2tiles → local Tiles adapter
------------------------------------------------

This module re‐implements CoilPy’s `muse2magntense` + `build_prism` workflow
to populate our local `Tiles` container, removing the original MagTense‐GPU
dependency.

Adapted from:
  - CoilPy GitHub: https://github.com/zhucaoxiang/CoilPy
    (see `coilpy.magtense_interface.muse2magntense`)
  - CoilPy authors: Zhu, Cao et al.
"""

# Inspired by: https://zhucaoxiang.github.io/CoilPy/_modules/coilpy/magtense_interface.html#muse2magntense
def muse2tiles(
    muse_file: Union[str, Path],
    mu: tuple[float, float] = (1.05, 1.05),
    magnetization: float = 1.16e6,
    **kwargs,
) -> Tiles:
    """Convert a MUSE-exported CSV file into magnet tiles for MacroMag.

    Parses the CSV (one row per magnet: position, size, basis vectors,
    magnetization) and builds a Tiles container via build_prism.

    Args:
        muse_file: Path to the MUSE CSV file.
        mu: (mu_ea, mu_oa) relative permeabilities. Defaults to (1.05, 1.05).
        magnetization: Remanent magnetization [A/m]. Defaults to 1.16e6.
        **kwargs: Passed to build_prism.

    Returns:
        Tiles instance with prism geometries and material parameters.
    """

    data = np.loadtxt(muse_file, skiprows=1, delimiter=",")
    nmag = len(data)
    print(f"{nmag} magnets are identified in {muse_file!r}.")

    center = np.zeros((nmag, 3))
    rot    = np.zeros((nmag, 3))
    lwh    = np.zeros((nmag, 3))
    ang    = np.zeros((nmag, 2))
    mu_arr = np.repeat([mu], nmag, axis=0)
    Br     = np.zeros(nmag)

    for i in range(nmag):
        center[i] = data[i, 0:3]
        lwh[i]    = data[i, 3:6]
        n1, n2, n3 = data[i, 6:9], data[i, 9:12], data[i, 12:15]

        # build normalized basis → Euler angles
        rot_mat = np.vstack([
            n1 / np.linalg.norm(n1),
            n2 / np.linalg.norm(n2),
            n3 / np.linalg.norm(n3),
        ])
        rot[i] = rotation_angle(rot_mat.T, xyz=True)

        # magnetization angles
        mxyz = data[i, 15:18]
        mp   = math.atan2(mxyz[1], mxyz[0])
        mt   = math.acos(mxyz[2] / np.linalg.norm(mxyz))
        ang[i, 0:2] = [mt, mp]
        Br[i] = magnetization

    return build_prism(lwh, center, rot, ang, mu_arr, Br)


def build_prism(
    lwh: np.ndarray,
    center: np.ndarray,
    rot: np.ndarray,
    mag_angle: np.ndarray,
    mu: np.ndarray,
    remanence: np.ndarray,
) -> Tiles:
    """
    Construct rectangular prism tiles in a Tiles container.

    Each prism has length/width/height (lwh), center position, Euler rotation,
    easy-axis spherical angles, permeabilities, and remanent magnetization.

    Args:
        lwh: Prism dimensions, shape (n, 3) – length, width, height [m].
        center: Center positions, shape (n, 3) [m].
        rot: Euler angles (radians), shape (n, 3).
        mag_angle: Easy-axis (theta, phi) in radians, shape (n, 2).
        mu: Relative permeabilities (mu_ea, mu_oa), shape (n, 2).
        remanence: Remanent magnetization magnitude [A/m], shape (n,).

    Returns:
        Tiles instance with n prism tiles.
    """
    lwh       = np.atleast_2d(lwh)
    nmag      = lwh.shape[0]
    center    = np.atleast_2d(center)
    rot       = np.atleast_2d(rot)
    mag_angle = np.atleast_2d(mag_angle)
    mu        = np.atleast_2d(mu)
    rem       = np.atleast_1d(remanence)

    prism = Tiles(nmag)

    # set broadcast properties
    prism.tile_type = 2         # all prisms

    for i in range(nmag):
        # tuple‐setter form writes into the i-th row of each array
        prism.size     = (lwh[i],i)
        prism.offset   = (center[i],i)
        prism.rot      = (rot[i],i)
        prism.M_rem    = (rem[i],i)
        prism.mu_r_ea  = (mu[i,0],i)
        prism.mu_r_oa  = (mu[i,1],i)
        prism.set_easy_axis(val=mag_angle[i], idx=i)
        prism.color    = ([0, 0, (i+1)/nmag], i)

    return prism
