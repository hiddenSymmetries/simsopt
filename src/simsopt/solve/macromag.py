from __future__ import annotations # Temporary fix for python compatibility
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

__all__ = ["MacroMag"]

_INV4PI  = 1.0 / (4.0 * math.pi)
_MACHEPS = np.finfo(np.float64).eps


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
    """
    Construct a 3×3 rotation matrix from Euler angles.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles in radians.
    xyz : bool, optional
        If True, use the Rx(alpha) Ry(beta) Rz(gamma) convention. If False, use
        the Rz(alpha) Ry(beta) Rx(gamma) convention.

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
    """Inverse of :func:`_rotation_matrix` for the Z–Y–X (RzRyRx) convention."""
    tol = 1e-8
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
    """Inverse of :func:`_rotation_matrix` for the X–Y–Z (RxRyRz) convention."""
    tol = 1e-8
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


def _f_3D(a, b, c, x, y, z):
    """
    Auxiliary function for the analytical demagnetization tensor of a rectangular prism.

    This term appears in the closed-form expressions for the diagonal demag
    components via sums of ``atan(f)`` evaluated at signed corner combinations.

    In the notation of the standard prism formulas (Newell/Aharoni), for half-dimensions
    ``(a,b,c)`` and a local evaluation point ``(x,y,z)``, one contribution takes the form::

        f(a,b,c,x,y,z) = (b - y)(c - z) / ((a - x) * r)
        r = sqrt((a - x)^2 + (b - y)^2 + (c - z)^2)

    The diagonal demag term is then assembled by a signed sum over the 8 corner
    combinations (implemented in :func:`_prism_N_local_nb`)::

        N_xx = (1 / 4π) * Σ_{s∈{±1}^3} atan( f(a,b,c, s_x x0, s_y y0, s_z z0) )

    Notes
    -----
    If the denominator ``(a - x)`` is exactly zero in floating point, a small
    symmetric perturbation is used to approximate the limiting value.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Scalar value used inside ``atan`` for the N_xx expression.
    """
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = 1.0001, 0.9999

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

def _g_3D(a, b, c, x, y, z):
    """
    Auxiliary function for the analytical demagnetization tensor of a rectangular prism.

    Like :func:`_f_3D`, this term is used inside a signed sum of ``atan(g)``
    contributions when evaluating the diagonal demag component N_yy.

    Formula (local frame)::

        g(a,b,c,x,y,z) = (a - x)(c - z) / ((b - y) * r)
        r = sqrt((a - x)^2 + (b - y)^2 + (c - z)^2)

    Notes
    -----
    If the denominator ``(b - y)`` is exactly zero in floating point, a small
    symmetric perturbation is used to approximate the limiting value.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Scalar value used inside ``atan`` for the N_yy expression.
    """
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = 1.0001, 0.9999

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


def _h_3D(a, b, c, x, y, z):
    """
    Auxiliary function for the analytical demagnetization tensor of a rectangular prism.

    Like :func:`_f_3D` and :func:`_g_3D`, this term is used inside a signed sum
    of ``atan(h)`` contributions for the diagonal demag component N_zz.

    Formula (local frame)::

        h(a,b,c,x,y,z) = (a - x)(b - y) / ((c - z) * r)
        r = sqrt((a - x)^2 + (b - y)^2 + (c - z)^2)

    Notes
    -----
    If the denominator ``(c - z)`` is exactly zero in floating point, a small
    symmetric perturbation is used to approximate the limiting value.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Scalar value used inside ``atan`` for the N_zz expression.
    """
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = 1.0001, 0.9999

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
    
def _FF_3D(a, b, c, x, y, z):
    """
    Log-kernel helper used for off-diagonal demagnetization tensor components.

    In the closed-form prism demagnetization formulas (Newell/Aharoni style),
    the off-diagonal entries are assembled from logarithms of products of
    auxiliary functions evaluated at the 8 signed corner combinations. For the
    ``N_xy`` entry, one such auxiliary function can be written (local frame)::

        F(a,b,c,x,y,z) = (c - z) + r
        r = sqrt((a - x)^2 + (b - y)^2 + (c - z)^2)

    :func:`_prism_N_local_nb` forms the corresponding numerator/denominator
    products and uses ``-1/(4π) * log(numerator/denominator)`` to obtain the
    symmetric ``(x,y)`` off-diagonal block.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Positive scalar used in ratios inside ``log`` terms for ``N_xy``.
    """
    ha, hb, hc = a, b, c
    return (hc - z) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)

def _GG_3D(a, b, c, x, y, z):
    """
    Log-kernel helper used for off-diagonal demagnetization tensor components.

    For the ``N_yz`` off-diagonal entry, a standard auxiliary function is
    (local frame)::

        G(a,b,c,x,y,z) = (a - x) + r
        r = sqrt((a - x)^2 + (b - y)^2 + (c - z)^2)

    :func:`_prism_N_local_nb` assembles ``N_yz`` via a signed product ratio and
    ``-1/(4π) * log(·)``.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Positive scalar used in ratios inside ``log`` terms for ``N_yz``.
    """
    ha, hb, hc = a, b, c
    return (ha - x) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)

def _HH_3D(a, b, c, x, y, z):
    """
    Log-kernel helper used for off-diagonal demagnetization tensor components.

    For the ``N_xz`` off-diagonal entry, a standard auxiliary function is
    (local frame)::

        H(a,b,c,x,y,z) = (b - y) + r
        r = sqrt((a - x)^2 + (b - y)^2 + (c - z)^2)

    :func:`_prism_N_local_nb` assembles ``N_xz`` via a signed product ratio and
    ``-1/(4π) * log(·)``. See :func:`getF_limit` for the corresponding
    numerator/denominator structure and the stabilized limit evaluation used
    when the raw products underflow.

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x, y, z: Evaluation point coordinates in the same local frame.

    Returns:
        Positive scalar used in ratios inside ``log`` terms for ``N_xz``.
    """
    ha, hb, hc = a, b, c
    return (hb - y) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)


def getF_limit(a, b, c, x, y, z, func):
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
    lim_h, lim_l = 1.0001, 0.9999
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

def _prism_N_local_nb(a, b, c, x0, y0, z0):
    """
    Demagnetization tensor for a uniformly magnetized rectangular prism (local frame).

    This function evaluates the 3×3 demagnetization tensor block N_ij for a
    rectangular prism with half-dimensions (a,b,c), for a target point located at
    (x0,y0,z0) in the *prism's local coordinate system*.

    The expressions are the standard closed-form Newell/Aharoni-style formulas:
    diagonal entries are sums of ``atan`` terms; off-diagonals are ``log`` terms,
    with special handling for near-singular ratios via :func:`getF_limit`.

    In particular, with ``(x0,y0,z0)`` the local vector from source to target,
    the implementation matches the common structure::

        N_xx = (1 / 4π) * Σ atan( f(...) )
        N_yy = (1 / 4π) * Σ atan( g(...) )
        N_zz = (1 / 4π) * Σ atan( h(...) )

        N_xy = -(1 / 4π) * log( ratio(F=FF_3D) )
        N_yz = -(1 / 4π) * log( ratio(F=GG_3D) )
        N_xz = -(1 / 4π) * log( ratio(F=HH_3D) )

    where the sums/products are taken over signed corner combinations (see code).

    Args:
        a, b, c: Half-dimensions of the prism in its local frame.
        x0, y0, z0: Relative vector from source centre to target centre in the same local frame.

    Returns:
        N_op: ndarray, shape (3, 3). Demag *operator* block in local coordinates.

        This module uses the convention that demagnetizing field is obtained by
        a linear operator acting on magnetization, i.e. ``H_demag = N_op @ M``.
        For the standard demag tensor ``N_std`` (Newell/Aharoni; ``tr(N_std)=1``
        for a self-term), the operator is ``N_op = -N_std``. Consequently,
        ``tr(N_op) = -1`` for self-terms.
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
    """Return the smallest candidate >= ``n`` (or the maximum candidate)."""
    for c in candidates:
        if n <= c:
            return c
    return candidates[-1]


def _assemble_blocks_subset_numpy(centres, half, Rg2l, I, J):
    """
    Pure NumPy implementation of demag block assembly.

    Matches the JAX kernel logic exactly. Used when SIMSOPT_MACROMAG_USE_NUMPY=1
    to avoid JAX heap corruption on macOS (CPU fusion compiler).
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


def assemble_blocks_subset(centres, half, Rg2l, I, J):
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

    def _f_3D_jax(a, b, c, x, y, z):
        eps_h, eps_l = 1.0001, 0.9999
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
        eps_h, eps_l = 1.0001, 0.9999
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
        eps_h, eps_l = 1.0001, 0.9999
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
        return (c - z) + jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2)

    def _GG_3D_jax(a, b, c, x, y, z):
        return (a - x) + jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2)

    def _HH_3D_jax(a, b, c, x, y, z):
        return (b - y) + jnp.sqrt((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2)

    def _getF_limit_jax(a, b, c, x, y, z, func):
        lim_h, lim_l = 1.0001, 0.9999
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
    """Compute demag tensor for rectangular prisms."""

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
    def _f_3D(a, b, c, x, y, z):
        """Public wrapper for :func:`_f_3D`."""
        return _f_3D(a, b, c, x, y, z)

    @staticmethod
    def _g_3D(a, b, c, x, y, z):
        """Public wrapper for :func:`_g_3D`."""
        return _g_3D(a, b, c, x, y, z)

    @staticmethod
    def _h_3D(a, b, c, x, y, z):
        """Public wrapper for :func:`_h_3D`."""
        return _h_3D(a, b, c, x, y, z)

    @staticmethod
    def _FF_3D(a, b, c, x, y, z):
        """Public wrapper for :func:`_FF_3D`."""
        return _FF_3D(a, b, c, x, y, z)

    @staticmethod
    def _GG_3D(a, b, c, x, y, z):
        """Public wrapper for :func:`_GG_3D`."""
        return _GG_3D(a, b, c, x, y, z)

    @staticmethod
    def _HH_3D(a, b, c, x, y, z):
        """Public wrapper for :func:`_HH_3D`."""
        return _HH_3D(a, b, c, x, y, z)

    @staticmethod
    def _prism_N_local_nb(a, b, c, x0, y0, z0):
        """Public wrapper for :func:`_prism_N_local_nb`."""
        return _prism_N_local_nb(a, b, c, x0, y0, z0)

    @staticmethod
    def get_rotation_matrices(phi_x, phi_y, phi_z):
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
        
    def coil_field_at(self, pts):
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
        
        """
        Solve for the tile magnetization using GMRES.

        This method solves a dense linear system of the form ``A M = b``, where
        ``M`` is the stacked magnetization vector (length ``3*n_tiles``). The
        system is assembled as::

            A = I + χ N_op
            b = M_rem u + χ H_a

        where ``χ`` is the per-tile uniaxial susceptibility tensor in global
        coordinates and ``N_op`` is the demagnetization *operator* produced by
        :func:`_prism_N_local_nb` (i.e. ``H_demag = N_op @ M``).

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
            
        if use_coils and (self.bs_interp is None) and (H_a_override is None):
            raise ValueError("use_coils=True, but no coil field evaluator is loaded and no H_a_override was provided.")

        if H_a_override is not None:
            H_a_override = np.asarray(H_a_override, dtype=np.float64, order="C") 
            if H_a_override.shape != (self.centres.shape[0], 3):
                raise ValueError("H_a_override must have shape (n_tiles, 3) in A/m.")

        # Define helper function to compute the coil field or override it
        def Hcoil_func(pts):
            """Return applied field in A/m (override takes precedence over coils)."""
            if H_a_override is not None:
                return H_a_override 
            else: 
                return self.coil_field_at(pts)

        tiles   = self.tiles
        centres = self.centres
        n       = tiles.n
        u       = tiles.u_ea
        m_rem   = tiles.M_rem
        chi_parallel  = tiles.mu_r_ea - 1
        chi_perp  = tiles.mu_r_oa - 1
        
        # necessary short circuit in chi=0 case 
        if np.allclose(chi_parallel, 0.0) and np.allclose(chi_perp, 0.0):
            if (not use_coils) and (H_a_override is None):
                # trivial hard-magnet solution: M = M_rem u
                tiles.M = m_rem[:, None] * u
                return self, None

        # assemble b and B directly without forming chi tensor
        H_a = np.zeros((n, 3)) if (not use_coils) else Hcoil_func(centres).astype(np.float64)
        
        # Vectorized computation of u[i]^T * H_a[i] for all i
        u_dot_Ha = np.sum(u * H_a, axis=1)  # shape: (n,)
        
        # Vectorized computation of chi[i] @ H_a[i] for all i
        # chi[i] @ H_a[i] = chi_pe[i] * H_a[i] + (chi_pa[i] - chi_pe[i]) * (u[i]^T * H_a[i]) * u[i]
        chi_Ha = (chi_perp[:, None] * H_a + 
                 (chi_parallel - chi_perp)[:, None] * u_dot_Ha[:, None] * u)
        
        # Vectorized computation of b
        b = (m_rem[:, None] * u + chi_Ha).ravel()

        # t1 = time.time()
        
        # Precompute frequently used terms
        chi_diff = chi_parallel - chi_perp  # (n,)
        u_outer = np.einsum('ik,il->ikl', u, u, optimize=True)  # (n, 3, 3) - u_i * u_i^T
        
        # Uniaxial susceptibility tensor in global coordinates:
        #   χ_i = χ⊥ I + (χ∥ - χ⊥) (u_i u_iᵀ)
        chi_tensor = (chi_perp[:, None, None] * np.eye(3)[None, :, :] + 
                    chi_diff[:, None, None] * u_outer)  # (n, 3, 3)

        
        # Start with previous A matrix and extend it
        A = np.zeros((3*n, 3*n))
        A[:3*prev_n, :3*prev_n] = A_prev
        
        # Add identity blocks for new tiles
        for i in range(prev_n, n):
            A[3*i:3*i+3, 3*i:3*i+3] = np.eye(3)
        
        # Add new rows and columns 
        if prev_n > 0:
            chi_new = chi_tensor[prev_n:, :, :] 
            chi_prev = chi_tensor[:prev_n, :, :]  # (prev_n, 3, 3)

            # Assemble the off-diagonal demag blocks using a blockwise contraction:
            # each 3×3 block is (χ_i @ N_ij). Use einsum (not elementwise multiplication).
            # new rows vs old cols
            A[3*prev_n:, :3*prev_n] += np.einsum('iab,ijbc->iajc', chi_new, N_new_rows).reshape(3*(n-prev_n), 3*prev_n)
            # old rows vs new cols
            A[:3*prev_n, 3*prev_n:] += np.einsum('iab,ijbc->iajc', chi_prev, N_new_cols).reshape(3*prev_n, 3*(n-prev_n))
            # new rows vs new cols (diagonal block of the “new” part)
            A[3*prev_n:, 3*prev_n:] += np.einsum('iab,ijbc->iajc', chi_new, N_new_diag).reshape(3*(n-prev_n), 3*(n-prev_n))

        else:
            # Full dense block assembly: A = I + χ N, with χ_i applied on each row block.
            A += np.einsum('iab,ijbc->iajc', chi_tensor, N_new_rows).reshape(3*n, 3*n)
            
        # t2 = time.time()
        # print(f"Time taken for direct A assembly: {t2 - t1} seconds")
        
        if print_progress:
            
            # Use a closure to track iteration number
            def make_gmres_callback():
                """Build a stateful GMRES callback that prints iteration and residual norm."""
                iteration = [0]  # Use list to make it mutable in closure
                def gmres_callback(residual_norm):
                    """GMRES callback printing the current preconditioned residual norm."""
                    iteration[0] += 1
                    print(f"GMRES iteration {iteration[0]}: residual norm = {residual_norm:.6e}")
                return gmres_callback
            
            x, info = gmres(A, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0, callback=make_gmres_callback(), callback_type='pr_norm')
        else:
            x, info = gmres(A, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0)
        
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

        tiles   = self.tiles
        centres = self.centres
        u       = tiles.u_ea
        m_rem   = tiles.M_rem
        mu_r_ea = tiles.mu_r_ea
        mu_r_oa = tiles.mu_r_oa

        chi_parallel = mu_r_ea - 1.0
        chi_perp     = mu_r_oa - 1.0

        # trivial case: no susceptibility, no coils
        if np.allclose(chi_parallel, 0.0) and np.allclose(chi_perp, 0.0):
            if (not use_coils) and (H_a_override is None):
                tiles.M = m_rem[:, None] * u
                return self, None

        # external field setup
        if use_coils and (self.bs_interp is None) and (H_a_override is None):
            raise ValueError(
                "use_coils=True, but no coil field is loaded and no H_a_override was provided."
            )

        if H_a_override is not None:
            H_a_override = np.asarray(H_a_override, dtype=np.float64, order="C")
            if H_a_override.shape != (centres.shape[0], 3):
                raise ValueError("H_a_override must have shape (n_tiles, 3).")

        def Hcoil_func(pts):
            """Return applied field in A/m (override takes precedence over coils)."""
            if H_a_override is not None:
                return H_a_override
            else:
                return self.coil_field_at(pts)

        n = centres.shape[0]
        # H_a: either zero, coil field, or override
        H_a = np.zeros((n, 3), dtype=np.float64)
        if use_coils or (H_a_override is not None):
            H_a = Hcoil_func(centres).astype(np.float64)

        # Build RHS: b = M_rem u + χ H_a, tilewise
        u_dot_Ha = np.sum(u * H_a, axis=1)  # shape (n,)
        chi_Ha = (chi_perp[:, None] * H_a +
                  (chi_parallel - chi_perp)[:, None] * u_dot_Ha[:, None] * u)
        b = (m_rem[:, None] * u + chi_Ha).ravel()  # length 3n

        # Build χ tensor per tile (3×3)
        chi_diff = chi_parallel - chi_perp
        u_outer = np.einsum("ik,il->ikl", u, u, optimize=True)  # (n,3,3)
        chi_tensor = (chi_perp[:, None, None] * np.eye(3)[None, :, :] +
                      chi_diff[:, None, None] * u_outer)

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

        if print_progress:
            # Hook for detailed GMRES logging if you want it later
            def make_gmres_callback():
                """Build a stateful GMRES callback for neighbor-sparse runs."""
                it = [0]
                def cb(res_norm):
                    """GMRES callback printing the current preconditioned residual norm."""
                    it[0] += 1
                    print(f"[neighbor_sparse] GMRES it {it[0]}: ||r|| = {res_norm:.3e}")
                return cb
            x, info = gmres(A_sparse, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0 , callback=make_gmres_callback(), callback_type='pr_norm')
        else:
            x, info = gmres(A_sparse, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0)

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
            
        if use_coils and (self.bs_interp is None) and (H_a_override is None):
            raise ValueError("use_coils=True, but no coil field evaluator is loaded and no H_a_override was provided.")

        if H_a_override is not None:
            H_a_override = np.asarray(H_a_override, dtype=np.float64, order="C") 
            if H_a_override.shape != (self.centres.shape[0], 3):
                raise ValueError("H_a_override must have shape (n_tiles, 3) in A/m.")

        def Hcoil_func(pts):
            """Return applied field in A/m (override takes precedence over coils)."""
            if H_a_override is not None:
                return H_a_override 
            else: 
                return self.coil_field_at(pts)

        tiles   = self.tiles
        centres = self.centres
        n       = tiles.n
        u       = tiles.u_ea
        m_rem   = tiles.M_rem
        chi_parallel  = tiles.mu_r_ea - 1
        chi_perp  = tiles.mu_r_oa - 1
        
        if np.allclose(chi_parallel, 0.0) and np.allclose(chi_perp, 0.0):
            if (not use_coils) and (H_a_override is None):
                tiles.M = m_rem[:, None] * u
                return self, None

        H_a = np.zeros((n, 3)) if (not use_coils) else Hcoil_func(centres).astype(np.float64)
        u_dot_Ha = np.sum(u * H_a, axis=1)
        chi_Ha = (chi_perp[:, None] * H_a + 
                 (chi_parallel - chi_perp)[:, None] * u_dot_Ha[:, None] * u)
        b = (m_rem[:, None] * u + chi_Ha).ravel()
        chi_diff = chi_parallel - chi_perp
        u_outer = np.einsum('ik,il->ikl', u, u, optimize=True)
        chi_tensor = (chi_perp[:, None, None] * np.eye(3)[None, :, :] + 
                    chi_diff[:, None, None] * u_outer)
        A = np.zeros((3*n, 3*n))
        A[:3*prev_n, :3*prev_n] = A_prev
        for i in range(prev_n, n):
            A[3*i:3*i+3, 3*i:3*i+3] = np.eye(3)
        if use_demag:
            if prev_n > 0:
                chi_new = chi_tensor[prev_n:, :, :]
                chi_prev = chi_tensor[:prev_n, :, :]
                A[3*prev_n:, :3*prev_n] += np.einsum('iab,ijbc->iajc', chi_new, N_new_rows).reshape(3*(n-prev_n), 3*prev_n)
                A[:3*prev_n, 3*prev_n:] += np.einsum('iab,ijbc->iajc', chi_prev, N_new_cols).reshape(3*prev_n, 3*(n-prev_n))
                A[3*prev_n:, 3*prev_n:] += np.einsum('iab,ijbc->iajc', chi_new, N_new_diag).reshape(3*(n-prev_n), 3*(n-prev_n))
            else:
                A += np.einsum('iab,ijbc->iajc', chi_tensor, N_new_rows).reshape(3*n, 3*n)
        if print_progress and False:
            def make_gmres_callback():
                """Build a stateful GMRES callback that prints iteration and residual norm."""
                iteration = [0]
                def gmres_callback(residual_norm):
                    """GMRES callback printing the current preconditioned residual norm."""
                    iteration[0] += 1
                    print(f"GMRES iteration {iteration[0]}: residual norm = {residual_norm:.6e}")
                return gmres_callback
            x, info = gmres(A, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0)
        else:
            x, info = gmres(A, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0)
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

    def __init__(self, parent, index) -> None:
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
    """
    Container holding MagTense-style tile arrays.

    This class is modeled after the MagTense ``MagTile`` Fortran derived type,
    but implemented in pure Python/NumPy. The attributes below describe the
    conceptual data structure; not all fields are exposed as constructor inputs.

    Fields (MagTense-style)
    -----------------------
        center_pos: r0, theta0, z0
        dev_center: dr, dtheta, dz
        size: a, b, c
        vertices: v1, v2, v3, v4 as column vectors.
        M: Mx, My, Mz
        u_ea: Easy axis.
        u_oa: Other axis.
        mu_r_ea: Relative permeability in easy axis.
        mu_r_oa: Relative permeability in other axis.
        M_rem: Remanent magnetization.
        tile_type: 1 = cylinder, 2 = prism, 3 = circ_piece, 4 = circ_piece_inv,
                   5 = tetrahedron, 6 = sphere, 7 = spheroid, 10 = ellipsoid
        offset: Offset of global coordinates.
        rot: Rotation in local coordinate system.
        color: Color in visualization.
        magnet_type: 1 = hard magnet, 2 = soft magnet, 3 = soft + constant mu_r
        stfcn_index: default index into the state function.
        incl_it: If equal to zero the tile is not included in the iteration.
        use_sym: Whether to exploit symmetry.
        sym_op: 1 for symmetry and -1 for anti-symmetry respectively to the planes.
        M_rel: Change in magnetization during last iteration of iterate_magnetization().
        n: Number of tiles in the simulation.
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

    def __getitem__(self, index) -> Tile:
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

    @property
    def center_pos(self) -> np.ndarray:
        """Tile center positions, array of shape ``(n, 3)``."""
        return self._center_pos

    @center_pos.setter
    def center_pos(self, val) -> None:
        """
        Set tile center positions.

        Accepted input forms:
        - ``None``: no change.
        - ``(vec, i)``: set a single tile ``i`` to the 3-vector ``vec``.
        - a single length-3 vector: broadcast to all tiles.
        - an array-like of shape ``(n, 3)``.
        """
        if not hasattr(self, "_center_pos"):
            self._center_pos = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._center_pos[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._center_pos[:] = np.asarray(val)
        elif len(val) == self.n:
            self._center_pos = np.asarray(val)

    @property
    def dev_center(self) -> np.ndarray:
        """Tile center deviations, array of shape ``(n, 3)``."""
        return self._dev_center

    @dev_center.setter
    def dev_center(self, val) -> None:
        """Set tile center deviations (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_dev_center"):
            self._dev_center = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._dev_center[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._dev_center[:] = np.asarray(val)
        elif len(val) == self.n:
            self._dev_center = np.asarray(val)

    @property
    def size(self) -> np.ndarray:
        """Tile sizes (edge lengths), array of shape ``(n, 3)``."""
        return self._size

    @size.setter
    def size(self, val) -> None:
        """Set tile sizes (same input conventions as :meth:`center_pos`)."""
        if not hasattr(self, "_size"):
            self._size = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._size[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._size[:] = np.asarray(val)
        elif len(val) == self.n:
            self._size = np.asarray(val)

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
        if not hasattr(self, "_offset"):
            self._offset = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._offset[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._offset[:] = np.asarray(val)
        elif len(val) == self.n:
            self._offset = np.asarray(val)

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
            self._u_ea[:] = np.around(val / (ea_norms + 1e-30), decimals=9)
            
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

    def _set_ea_i(self, val, i) -> None:
        """Internal helper to set easy-axis and orthogonal axes for tile ``i``."""
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
def muse2tiles(muse_file, mu=(1.05, 1.05), magnetization=1.16e6, **kwargs):
    """Convert a MUSE-exported CSV file into a set of magnet "tiles" suitable for MacroMag

    Parameters
    ----------
    muse_file : str or Path
        Path to a CSV file exported from MUSE. The file is expected to contain one
        row per magnet with columns for position, size, local basis vectors, and
        magnetization vector.
    mu : tuple of float, optional
        Relative permeabilities assigned to each tile. The **first entry is mu_ea**
        (permeability along the easy axis), and the **second entry is mu_oa**
        (permeability orthogonal to the easy axis). Defaults to (1.05, 1.05).
    magnetization : float, optional
        Remanent magnetization (A/m) assigned to each magnet. Defaults to 1.16e6.
    **kwargs
        Extra keyword arguments passed downstream to `build_prism`.

    Returns
    -------
    Tiles
        A `Tiles` object constructed by `build_prism`, containing the prism
        geometries, orientations, anisotropy parameters, and magnetizations
        parsed from the MUSE file.

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


def build_prism(lwh, center, rot, mag_angle, mu, remanence):
    """
    Construct a prism in our local Tiles container (no bind helper needed).

    Args:
        lwh        (n x 3): The dimensions of the prism in length, width, height.
        center     (n x 3): centers
        rot        (n x 3): Euler angles
        mag_angle  (n x 2): (theta,phi)
        mu         (n x 2): (mu_r_ea, mu_r_oa)
        remanence  (n,) : M_rem
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
