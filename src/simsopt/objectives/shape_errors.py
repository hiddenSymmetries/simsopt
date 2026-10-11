from dataclasses import dataclass

import numpy as np

__all__ = [
    'AngleMatchedReference', 'build_angle_matched_reference', 'angle_matched_shape_error',
    'ExactShapeReference', 'build_exact_shape_reference', 'exact_shape_error',
]

def _eval_rz_fourier(surf, nu, v_1d):
    '''
    Evaluate R, Z of a SurfaceRZFourier on a (u, v) grid by direct Fourier summation.

    Parameters
    ----------
    surf : SurfaceRZFourier
        Surface to evaluate.
    nu : int
        Number of equispaced poloidal angles u in [0, 2*pi).
    v_1d : ndarray
        Toroidal angles (radians).

    Returns
    -------
    R, Z : ndarray, shape (nu, len(v_1d))
    '''
    rbc = surf.rc.T
    zbs = surf.zs.T
    M = surf.mpol
    N = surf.ntor
    nfp = surf.nfp

    u_1d = np.linspace(0, 2*np.pi, nu, endpoint=False)
    v_grid, u_grid = np.meshgrid(v_1d, u_1d)

    cosnmuv = np.array(
        [[np.cos(m*u_grid - n*(nfp*v_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )
    sinnmuv = np.array(
        [[np.sin(m*u_grid - n*(nfp*v_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )

    R = np.einsum('nm,nmuv->uv', rbc, cosnmuv)
    Z = np.einsum('nm,nmuv->uv', zbs, sinnmuv)
    return R, Z

@dataclass
class AngleMatchedReference:
    '''
    Reference data for `angle_matched_shape_error`; build it with `build_angle_matched_reference`.
    '''
    v_1d: np.ndarray
    R_c: np.ndarray
    Z_c: np.ndarray
    phi_geom: np.ndarray
    R: np.ndarray
    Z: np.ndarray

def build_angle_matched_reference(surf, nu=64, nv=64):
    '''
    Tag points on `surf` by geometric poloidal angle about each cross section's centroid.

    Build once from the fixed surface, then reuse in `angle_matched_shape_error`.
    Assumes each cross section is star-shaped about its centroid.

    Parameters
    ----------
    surf : SurfaceRZFourier
        Reference surface.
    nu : int
        Number of poloidal points per cross section.
    nv : int
        Number of toroidal cross sections over [0, 2*pi).

    Returns
    -------
    AngleMatchedReference
    '''
    v_1d = np.linspace(0, 2*np.pi, nv, endpoint=False)
    R, Z = _eval_rz_fourier(surf, nu, v_1d)

    R_c = np.mean(R, axis=0)
    Z_c = np.mean(Z, axis=0)
    phi_geom = np.arctan2(Z - Z_c[None, :], R - R_c[None, :])

    return AngleMatchedReference(v_1d=v_1d, R_c=R_c, Z_c=Z_c, phi_geom=phi_geom, R=R, Z=Z)

def angle_matched_shape_error(surf, reference, nu=None):
    '''
    Fast, parametrization-invariant distance to the reference at equal geometric poloidal angle.

    Each point on `surf` is compared with the reference point at the same angle about the
    reference centroid, linearly interpolated between reference points.

    Parameters
    ----------
    surf : SurfaceRZFourier
        Surface to compare.
    reference : AngleMatchedReference
        From `build_angle_matched_reference`.
    nu : int, optional
        Number of poloidal points on `surf`; defaults to the reference resolution.

    Returns
    -------
    ndarray, shape (nu, len(reference.v_1d))
        (R, Z) distances.
    '''
    if nu is None:
        nu = reference.R.shape[0]

    R_new, Z_new = _eval_rz_fourier(surf, nu, reference.v_1d)
    phi_geom_new = np.arctan2(Z_new - reference.Z_c[None, :], R_new - reference.R_c[None, :])

    R_matched = np.empty_like(R_new)
    Z_matched = np.empty_like(Z_new)
    for j in range(len(reference.v_1d)):
        order = np.argsort(reference.phi_geom[:, j])
        phi_ref = reference.phi_geom[order, j]
        R_ref = reference.R[order, j]
        Z_ref = reference.Z[order, j]
        # pad with one wrapped point on each side so np.interp handles the
        # +-pi branch cut correctly (np.interp doesn't wrap on its own)
        phi_pad = np.concatenate([phi_ref[-1:] - 2*np.pi, phi_ref, phi_ref[:1] + 2*np.pi])
        R_pad = np.concatenate([R_ref[-1:], R_ref, R_ref[:1]])
        Z_pad = np.concatenate([Z_ref[-1:], Z_ref, Z_ref[:1]])
        R_matched[:, j] = np.interp(phi_geom_new[:, j], phi_pad, R_pad)
        Z_matched[:, j] = np.interp(phi_geom_new[:, j], phi_pad, Z_pad)

    return np.hypot(R_new - R_matched, Z_new - Z_matched)

def _fourier_eval_with_theta_derivs(surf, theta, phi):
    '''
    R, Z and their first two theta-derivatives of a stellarator-symmetric SurfaceRZFourier.

    Parameters
    ----------
    surf : SurfaceRZFourier
        Stellarator-symmetric surface (only rc and zs are used).
    theta : ndarray
        Poloidal angles (radians).
    phi : float or ndarray
        Toroidal angles (radians), broadcastable with `theta`.

    Returns
    -------
    R, Z, dR, dZ, d2R, d2Z : ndarray
        Each shaped like `theta`.
    '''
    theta = np.asarray(theta, dtype=float)
    phi = np.broadcast_to(np.asarray(phi, dtype=float), theta.shape)

    rc = surf.rc.T  # shape (2*ntor+1, mpol+1)
    zs = surf.zs.T
    mpol, ntor, nfp = surf.mpol, surf.ntor, surf.nfp

    m = np.arange(0, mpol + 1)
    n = np.arange(-ntor, ntor + 1)

    angle = (
        m[None, None, :] * theta[..., None, None]
        - (n[None, :, None] * nfp) * phi[..., None, None]
    )
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    m_b = m[None, None, :]

    R = np.einsum('...nm,nm->...', cos_a, rc)
    Z = np.einsum('...nm,nm->...', sin_a, zs)
    dR = np.einsum('...nm,nm->...', -m_b * sin_a, rc)
    dZ = np.einsum('...nm,nm->...', m_b * cos_a, zs)
    d2R = np.einsum('...nm,nm->...', -(m_b**2) * cos_a, rc)
    d2Z = np.einsum('...nm,nm->...', -(m_b**2) * sin_a, zs)
    return R, Z, dR, dZ, d2R, d2Z

def _closest_theta_newton(
        surf,
        phi,
        R0,
        Z0,
        ntheta_search=200,
        newton_iters=20,
        tol=1e-13,
    ):
    '''
    Closest point on the cross section of `surf` at `phi` to each target (R0, Z0).

    A grid search over theta picks the starting guess, then Newton's method solves
    d/dtheta[(R - R0)^2 + (Z - Z0)^2] = 0 with analytic theta-derivatives.

    Parameters
    ----------
    surf : SurfaceRZFourier
        Stellarator-symmetric surface.
    phi : float
        Toroidal angle (radians).
    R0, Z0 : ndarray
        Target points, same shape.
    ntheta_search : int
        Number of theta points in the starting grid search.
    newton_iters : int
        Maximum number of Newton iterations.
    tol : float
        Stop once every Newton step is smaller than this (radians).

    Returns
    -------
    distance, theta : ndarray
        Each shaped like `R0`.
    '''
    R0 = np.asarray(R0, dtype=float)
    Z0 = np.asarray(Z0, dtype=float)

    theta_grid = np.linspace(0, 2 * np.pi, ntheta_search, endpoint=False)
    R_grid, Z_grid, *_ = _fourier_eval_with_theta_derivs(
        surf, theta_grid, phi
    )
    dist2 = (
        (R_grid[None, :] - R0[:, None]) ** 2
        + (Z_grid[None, :] - Z0[:, None]) ** 2
    )
    theta = theta_grid[np.argmin(dist2, axis=1)].copy()

    for _ in range(newton_iters):
        R, Z, dR, dZ, d2R, d2Z = _fourier_eval_with_theta_derivs(
            surf, theta, phi
        )
        res_R = R - R0
        res_Z = Z - Z0
        grad = res_R * dR + res_Z * dZ
        hess = dR ** 2 + res_R * d2R + dZ ** 2 + res_Z * d2Z
        step = np.where(np.abs(hess) > 1e-300, grad / hess, 0.0)
        theta = theta - step
        if np.max(np.abs(step)) < tol:
            break

    R, Z, *_ = _fourier_eval_with_theta_derivs(surf, theta, phi)
    return np.hypot(R - R0, Z - Z0), theta

@dataclass
class ExactShapeReference:
    '''
    Reference data for `exact_shape_error`; build it with `build_exact_shape_reference`.
    '''
    phi_1d: np.ndarray
    R_ref: np.ndarray
    Z_ref: np.ndarray

def build_exact_shape_reference(reference_surf, phi_1d, ntheta=64):
    '''
    Sample ground-truth points on cross sections of `reference_surf`.

    Build once (this is the expensive step) and reuse for every surface compared in
    `exact_shape_error`.

    Parameters
    ----------
    reference_surf : Surface
        Ground-truth surface with a `cross_section` method, e.g. a SurfaceBSpline.
    phi_1d : ndarray
        Cylindrical toroidal angles (radians) of the cross sections.
    ntheta : int
        Number of points per cross section.

    Returns
    -------
    ExactShapeReference
    '''
    phi_1d = np.asarray(phi_1d, dtype=float)
    R_ref = np.empty((len(phi_1d), ntheta))
    Z_ref = np.empty((len(phi_1d), ntheta))
    for i, phi in enumerate(phi_1d):
        xyz = reference_surf.cross_section(phi / (2 * np.pi), thetas=ntheta)
        R_ref[i] = np.hypot(xyz[:, 0], xyz[:, 1])
        Z_ref[i] = xyz[:, 2]
    return ExactShapeReference(phi_1d=phi_1d, R_ref=R_ref, Z_ref=Z_ref)

def exact_shape_error(
        surf,
        reference,
        ntheta_search=200,
        newton_iters=20,
        tol=1e-13,
    ):
    '''
    Nearest-point distance from each reference point to the cross section of `surf` at the same phi.

    Parameters
    ----------
    surf : SurfaceRZFourier
        Stellarator-symmetric surface to evaluate.
    reference : ExactShapeReference
        From `build_exact_shape_reference`.
    ntheta_search : int
        Number of theta points in the starting grid search; must resolve the finest
        poloidal feature of `surf`.
    newton_iters : int
        Maximum number of Newton iterations per point.
    tol : float
        Newton convergence tolerance (radians).

    Returns
    -------
    ndarray, shape reference.R_ref.shape
        (R, Z) distances, (n_phi, ntheta).
    '''
    errors = np.empty_like(reference.R_ref)
    for i, phi in enumerate(reference.phi_1d):
        d, _ = _closest_theta_newton(
            surf, phi, reference.R_ref[i], reference.Z_ref[i],
            ntheta_search=ntheta_search, newton_iters=newton_iters, tol=tol,
        )
        errors[i] = d
    return errors
