from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.optimize import minimize_scalar, newton
import matplotlib.pyplot as plt
import shapely

__all__ = [
    'any_to_uz_grid', 'jaccard_index', 'AngleMatchedReference',
    'build_angle_matched_reference', 'angle_matched_shape_error',
    'ExactShapeReference', 'build_exact_shape_reference', 'exact_shape_error',
]

def any_to_uz_grid(
        surf,
        nu=64,
        nv=64,
        plot=False,
    ):
    '''
    Given a SurfaceRZFourier, get R, Z values on a (u, v) grid of size `(nu, nv)`,
    where `v` is the polar angle (i.e. the VMEC toroidal angle) and `u` is an arbitrary
    poloidal angle in a plane of constant v. 

    :param surf: SurfaceRZFourier
    :param nu: number of poloidal gridpoints
    :param nv: number of toroidal gridpoints
    :param plot: whether or not to plot grid
    '''
    rbc = surf.rc.T
    zbs = surf.zs.T
    M = surf.mpol
    N = surf.ntor
    nfp = surf.nfp
    # nfp=1

    u_1d = np.linspace(0, 2*np.pi, nu, endpoint=True)
    zeta_1d = np.linspace(0, 2*np.pi, nv, endpoint=True)        
    zeta_grid, u_grid = np.meshgrid(zeta_1d, u_1d)
    u_zeta_points = np.vstack((u_grid.flatten(), zeta_grid.flatten()))

    zeta_eval, theta_eval = np.meshgrid(np.linspace(np.pi/nfp, 2*np.pi/nfp, nv, endpoint=True), np.linspace(0, 2*np.pi, nu, endpoint=True))  
    eval_grid = np.vstack((theta_eval.flatten(), zeta_eval.flatten()))

    cosnmuz = np.array(
        [[np.cos(m*u_grid - n*(nfp*zeta_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )
    sinnmuz = np.array(
        [[np.sin(m*u_grid - n*(nfp*zeta_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )

    R_surf = np.einsum('nm,nmuz->uz', rbc, cosnmuz)
    z_surf = np.einsum('nm,nmuz->uz', zbs, sinnmuz)

    R_uz_callable = CloughTocher2DInterpolator(
        points=u_zeta_points.T,
        values=R_surf.flatten(),
    )
    z_uz_callable = CloughTocher2DInterpolator(
        points=u_zeta_points.T,
        values=z_surf.flatten(),
    )

    R_on_uz_grid = R_uz_callable(eval_grid.T).reshape(nu, nv)
    z_on_uz_grid = z_uz_callable(eval_grid.T).reshape(nu, nv)

    if plot:
        x_on_uz_grid = R_on_uz_grid * np.cos(zeta_eval)
        y_on_uz_grid = R_on_uz_grid * np.sin(zeta_eval)
        z_on_uz_grid = z_on_uz_grid

        fig3d, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.scatter(x_on_uz_grid, y_on_uz_grid, z_on_uz_grid)

        ax.set_box_aspect((1, 1, 1))
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.show()
        
    return R_on_uz_grid, z_on_uz_grid, eval_grid.T

def jaccard_index(
        surf1,
        surf2,
        nu=64,
        nv=64,
        plot=False,
        log=False
    ):
    '''
    Compute the Jaccard index between two SurfaceRZFourier surfaces,
    computed on `nv` toroidal cross-sections. The Jaccard index is the
    ratio between the intersection and union of two different shapes.
    The intersection and union are computed here using the `shapely`
    package, and treats each toroidal cross-sections as an `nu`-gon. 
    
    :param surf1: SurfaceRZFourier for first surface
    :param surf2: SurfaceRZFourier for second surface
    :param nu: number of poloidal points at which to evaluate surfaces
    :param nv: number of toroidal points at which to evaluate surfaces
    :param plot: whether to plot surfaces as evaluated on grids
    '''

    R_uz_1, z_uz_1, eval_grid = any_to_uz_grid(
        surf1,
        nu=nu,
        nv=nv,
        plot=plot
    )

    R_uz_2, z_uz_2, _ = any_to_uz_grid(
        surf2,
        nu=nu,
        nv=nv,
        plot=plot
    )

    j = []

    for i, zeta in enumerate(eval_grid.T[1,:].reshape(nu, nv)):
        coords1 = list(zip(R_uz_1[:-1,i], z_uz_1[:-1,i]))
        coords2 = list(zip(R_uz_2[:-1,i], z_uz_2[:-1,i]))
        poly1 = shapely.Polygon(coords1)
        poly2 = shapely.Polygon(coords2)
        j.append(shapely.intersection(poly1, poly2).area/shapely.union(poly1, poly2).area)
    if log:
        return np.log10(j)
    else:
        return np.array(j)-1

def pointwise_minimum_poly_distance(
        surf1,
        target_surf,
        nu=16,
        nv=16,
        plot=False,
    ):
    '''
    Compute the Jaccard index between two SurfaceRZFourier surfaces,
    computed on `nv` toroidal cross-sections. The Jaccard index is the
    ratio between the intersection and union of two different shapes.
    The intersection and union are computed here using the `shapely`
    package, and treats each toroidal cross-sections as an `nu`-gon. 
    
    :param surf1: SurfaceRZFourier for first surface
    :param surf2: SurfaceRZFourier for second surface
    :param nu: number of poloidal points at which to evaluate surfaces
    :param nv: number of toroidal points at which to evaluate surfaces
    :param plot: whether to plot surfaces as evaluated on grids
    '''

    R_uz_1, z_uz_1, eval_grid = any_to_uz_grid(
        surf1,
        nu=nu,
        nv=nv,
        plot=plot
    )

    R_uz_2, z_uz_2, _ = any_to_uz_grid(
        target_surf,
        nu=nu,
        nv=nv,
        plot=plot
    )

    #plt.show()

    j = []

    # print(eval_grid.T[1,:].reshape(nu, nv)[0])
    for i, zeta in enumerate(eval_grid.T[1,:].reshape(nu, nv)[0]):
        # fig, ax = plt.subplots()
        coords1 = list(zip(R_uz_1[:-1,i], z_uz_1[:-1,i]))
        coords2 = list(zip(R_uz_2[:-1,i], z_uz_2[:-1,i]))
        arg_polygon = shapely.Polygon(coords1)
        # ax.plot(R_uz_1[:-1,i], z_uz_1[:-1,i])
        # ax.plot(R_uz_2[:-1,i], z_uz_2[:-1,i])
        #target_poly = shapely.Polygon(coords2)
        for coords in coords2:
            point = shapely.Point(coords[0], coords[1])
            # ax.scatter(coords[0], coords[1])
            #print(point)
            dist = shapely.distance(arg_polygon.boundary, point)
            # print(f'dist: {dist}')
            # ax.text(coords[0], coords[1], f'{dist}')
            j.append(dist)
        # plt.show()
    #print(f'j: {j}')
    j=np.array(j)
    return j

def _eval_rz_fourier(surf, nu, v_1d):
    '''
    Direct Fourier-series evaluation of R, Z on a (u, v) grid, where u
    is `surf`'s own poloidal Fourier angle and v is its own toroidal
    Fourier angle. Unlike `any_to_uz_grid`, this does no interpolation --
    it's exact (up to floating point) and much cheaper, since a
    SurfaceRZFourier already *is* a Fourier series.

    :param surf: SurfaceRZFourier
    :param nu: number of poloidal gridpoints
    :param v_1d: 1D array of toroidal angles (VMEC zeta) to evaluate at
    :return: R, Z, each of shape (nu, len(v_1d))
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
    Data returned by `build_angle_matched_reference`, consumed by
    `angle_matched_shape_error`. Not meant to be constructed directly.
    '''
    v_1d: np.ndarray
    R_c: np.ndarray
    Z_c: np.ndarray
    phi_geom: np.ndarray
    R: np.ndarray
    Z: np.ndarray

def build_angle_matched_reference(surf, nu=64, nv=64):
    '''
    Precompute a fast, reparametrization-invariant reference for
    `angle_matched_shape_error`, from a *fixed* original SurfaceRZFourier
    `surf`. This is the only expensive step (a dense grid evaluation) --
    call it once, before a loop that repeatedly perturbs a copy of
    `surf`'s coefficients (e.g. inside `variational_spec_cond`), not once
    per iteration.

    Each point on the reference surface is tagged by its geometric
    poloidal angle -- atan2(Z - Z_c, R - R_c) about that cross-section's
    own centroid (R_c, Z_c) -- rather than by its Fourier angle u. This
    is what makes the comparison reparametrization-invariant: two points
    at the same geometric angle are "the same point" on the shape,
    regardless of how u happens to be distributed along the curve.
    `angle_matched_shape_error` reuses this fixed (R_c, Z_c) rather than
    recomputing it from the perturbed surface, so both surfaces' angles
    are measured from the same physical reference frame.

    Assumes each toroidal cross-section is star-shaped about its own
    centroid (true for any reasonable, non-self-intersecting stellarator
    boundary), so geometric poloidal angle is a valid, single-valued
    coordinate along the curve -- the same assumption other collocation
    routines in this codebase make about surfaces being star-shaped
    about a computed axis.

    :param surf: SurfaceRZFourier, the reference (original) surface
    :param nu: number of poloidal grid points used to build the reference
    :param nv: number of toroidal grid points used to build the reference
    :return: an `AngleMatchedReference` for `angle_matched_shape_error`
    '''
    v_1d = np.linspace(0, 2*np.pi, nv, endpoint=False)
    R, Z = _eval_rz_fourier(surf, nu, v_1d)

    R_c = np.mean(R, axis=0)
    Z_c = np.mean(Z, axis=0)
    phi_geom = np.arctan2(Z - Z_c[None, :], R - R_c[None, :])

    return AngleMatchedReference(v_1d=v_1d, R_c=R_c, Z_c=Z_c, phi_geom=phi_geom, R=R, Z=Z)

def angle_matched_shape_error(surf, reference, nu=None):
    '''
    Fast, reparametrization-invariant shape error between `surf` and the
    surface used to build `reference` (see `build_angle_matched_reference`).
    Meant for use inside an optimization loop that repeatedly perturbs
    `surf`'s Fourier coefficients: only a direct Fourier evaluation and a
    1D linear interpolation are done here -- no root-finding, no shapely
    calls, unlike `pointwise_minimum_poly_distance`.

    :param surf: SurfaceRZFourier to compare against `reference`
    :param reference: an `AngleMatchedReference` from
        `build_angle_matched_reference`, built from the original surface
    :param nu: number of poloidal points at which to evaluate `surf`
        (defaults to the resolution `reference` was built with)
    :return: array of shape (nu, len(reference.v_1d)) of Euclidean (R, Z)
        distances between each evaluated point on `surf` and the point on
        the reference curve at the same geometric poloidal angle
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
    Exact (Fourier-summation, not interpolated) evaluation of a
    stellarator-symmetric SurfaceRZFourier's R, Z and their first two
    derivatives with respect to theta, at arbitrary poloidal angles --
    not restricted to a grid. Used by `_closest_theta_newton` to do a
    Newton solve in theta at fixed phi.

    :param surf: SurfaceRZFourier (stellarator symmetric: uses only
        surf.rc, surf.zs)
    :param theta: 1D array of poloidal angles (radians)
    :param phi: 1D array broadcastable with `theta`, or scalar -- toroidal
        angle(s) (radians), matching surf's own Fourier v-angle convention
    :return: R, Z, dR/dtheta, dZ/dtheta, d2R/dtheta2, d2Z/dtheta2, each an
        array with the same shape as `theta`
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
    For a fixed toroidal angle `phi`, find -- for each target point
    (R0[i], Z0[i]) -- the poloidal angle theta on `surf`'s cross section
    at that phi minimizing the Euclidean (R, Z) distance to the target.

    A coarse grid search over `ntheta_search` equispaced theta values
    supplies the starting guess (surf's cross section is a closed curve,
    so the objective can have multiple local minima; the grid search
    finds the right basin), then `scipy.optimize.newton` (vectorized
    across every target point at once, since `theta0` is an array) solves
    the stationarity condition d/dtheta[(R-R0)^2 + (Z-Z0)^2] = 0 --
    supplying `surf`'s own analytic first and second theta-derivatives
    (from `_fourier_eval_with_theta_derivs`) as `func`/`fprime`, rather
    than letting scipy fall back to a derivative-free secant method.

    :param surf: SurfaceRZFourier
    :param phi: scalar toroidal angle (radians)
    :param R0, Z0: 1D arrays of target point coordinates, same shape
    :param ntheta_search: number of points in the coarse grid search
    :param newton_iters: max Newton iterations (scipy.optimize.newton's
        `maxiter`)
    :param tol: scipy.optimize.newton's `tol`, in radians
    :return: (distance, theta), each a 1D array shaped like R0
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
    theta0 = theta_grid[np.argmin(dist2, axis=1)]

    def grad(theta, R0, Z0):
        R, Z, dR, dZ, _, _ = _fourier_eval_with_theta_derivs(surf, theta, phi)
        return (R - R0) * dR + (Z - Z0) * dZ

    def hess(theta, R0, Z0):
        R, Z, dR, dZ, d2R, d2Z = _fourier_eval_with_theta_derivs(
            surf, theta, phi
        )
        return dR ** 2 + (R - R0) * d2R + dZ ** 2 + (Z - Z0) * d2Z

    theta = newton(
        grad, theta0, fprime=hess, args=(R0, Z0),
        tol=tol, maxiter=newton_iters,
    )

    R, Z, *_ = _fourier_eval_with_theta_derivs(surf, theta, phi)
    return np.hypot(R - R0, Z - Z0), theta

def _closest_theta_newton_manual(
        surf,
        phi,
        R0,
        Z0,
        ntheta_search=200,
        newton_iters=20,
        tol=1e-13,
    ):
    '''
    Same problem and math as `_closest_theta_newton`, but with the Newton
    iteration hand-rolled (a plain Python for-loop computing the step
    grad/hess directly) instead of delegated to `scipy.optimize.newton`.
    This is `_closest_theta_newton`'s original implementation, kept only
    as a profiling baseline -- see
    examples/2_Intermediate/compare_closest_theta_methods.py -- since
    `scipy.optimize.newton` calls `func` and `fprime` as two separate
    evaluations per iteration where this loop shares one evaluation of
    R, Z, dR, dZ, d2R, d2Z between them, so the two aren't quite doing
    identical work despite converging to the same answer.

    :param surf: SurfaceRZFourier
    :param phi: scalar toroidal angle (radians)
    :param R0, Z0: 1D arrays of target point coordinates, same shape
    :param ntheta_search: number of points in the coarse grid search
    :param newton_iters: max Newton iterations
    :param tol: stop once every Newton step is smaller than this (radians)
    :return: (distance, theta), each a 1D array shaped like R0
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

def _closest_theta_scipy(
        surf,
        phi,
        R0,
        Z0,
        ntheta_search=200,
    ):
    '''
    Same closest-point-in-theta problem as `_closest_theta_newton` -- for
    each target point (R0[i], Z0[i]), find the poloidal angle theta on
    `surf`'s cross section at fixed `phi` minimizing squared (R, Z)
    distance -- but solved with `scipy.optimize.minimize_scalar` (bounded
    Brent search, confined to +-one coarse-grid cell around the same
    grid-search starting guess `_closest_theta_newton` uses) instead of a
    hand-rolled Newton iteration on the analytic derivatives.

    Provided only for comparison against `_closest_theta_newton` (see
    examples/2_Intermediate/compare_closest_theta_methods.py) -- scipy's
    minimize_scalar has no array/vectorized mode, so this loops over
    every target point in pure Python and is much slower.

    :param surf: SurfaceRZFourier
    :param phi: scalar toroidal angle (radians)
    :param R0, Z0: 1D arrays of target point coordinates, same shape
    :param ntheta_search: number of points in the coarse grid search --
        also sets the width of the bracket (+- one grid cell) that each
        point's minimize_scalar search is confined to
    :return: (distance, theta), each a 1D array shaped like R0
    '''
    R0 = np.asarray(R0, dtype=float)
    Z0 = np.asarray(Z0, dtype=float)

    theta_grid = np.linspace(0, 2 * np.pi, ntheta_search, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]
    R_grid, Z_grid, *_ = _fourier_eval_with_theta_derivs(
        surf, theta_grid, phi
    )
    dist2 = (
        (R_grid[None, :] - R0[:, None]) ** 2
        + (Z_grid[None, :] - Z0[:, None]) ** 2
    )
    theta0 = theta_grid[np.argmin(dist2, axis=1)]

    theta = np.empty_like(theta0)
    for i in range(R0.size):
        R0i, Z0i = R0[i], Z0[i]

        def objective(t, R0i=R0i, Z0i=Z0i):
            R, Z, *_ = _fourier_eval_with_theta_derivs(
                surf, np.array([t]), phi
            )
            return (R[0] - R0i) ** 2 + (Z[0] - Z0i) ** 2

        result = minimize_scalar(
            objective,
            bounds=(theta0[i] - dtheta, theta0[i] + dtheta),
            method="bounded",
        )
        theta[i] = result.x

    R, Z, *_ = _fourier_eval_with_theta_derivs(surf, theta, phi)
    return np.hypot(R - R0, Z - Z0), theta

@dataclass
class ExactShapeReference:
    '''
    Data returned by `build_exact_shape_reference`, consumed by
    `exact_shape_error`. Not meant to be constructed directly.
    '''
    phi_1d: np.ndarray
    R_ref: np.ndarray
    Z_ref: np.ndarray

def build_exact_shape_reference(reference_surf, phi_1d, ntheta=64):
    '''
    Precompute a grid of ground-truth points on `reference_surf` (any
    Surface with a `cross_section` method -- e.g. a SurfaceBSpline), a
    fixed number of toroidal cross sections given by `phi_1d`, with
    `ntheta` points poloidally around each. This is the only expensive
    (root-finding via `cross_section`) step -- build it once from the
    ground-truth surface and reuse it for every candidate surface in a
    Fourier-mode convergence study; `exact_shape_error` then only has to
    redo the (cheap, analytic-Newton) closest-point solve per candidate.

    :param reference_surf: the ground-truth surface (e.g. a SurfaceBSpline)
    :param phi_1d: 1D array of toroidal angles (radians) at which to take
        cross sections -- should match the angle convention of the
        SurfaceRZFourier `surf` passed to `exact_shape_error` (i.e. VMEC's
        cylindrical toroidal angle)
    :param ntheta: number of poloidal points per cross section
    :return: an `ExactShapeReference` for `exact_shape_error`
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
        method="newton",
    ):
    '''
    Exact point-to-curve shape error between a SurfaceRZFourier `surf`
    and `reference` (an `ExactShapeReference` built from the ground-truth
    surface via `build_exact_shape_reference`).

    Unlike `angle_matched_shape_error` (which tags points by geometric
    angle and linearly interpolates) or `pointwise_minimum_poly_distance`
    (which approximates each cross section as a polygon), this computes
    the true nearest-point distance: for each of `reference`'s fixed
    toroidal cross sections, and each ground-truth point on that cross
    section, it finds the poloidal angle theta on `surf`'s cross section
    at the same phi minimizing the Euclidean (R, Z) distance to that
    point -- exact up to the solver's own tolerance, not an interpolation
    or polygon approximation.

    :param surf: SurfaceRZFourier to evaluate (e.g. a Fourier transform of
        the reference surface, truncated at some mode number, in a
        convergence study)
    :param reference: `ExactShapeReference` from
        `build_exact_shape_reference`, built from the ground-truth surface
    :param ntheta_search: number of theta points used for the coarse
        grid-search starting guess (see `_closest_theta_newton` /
        `_closest_theta_scipy`) -- must be dense enough that `surf` has no
        poloidal features finer than 2*pi/ntheta_search
    :param newton_iters: max Newton iterations per point (method="newton"
        only)
    :param tol: Newton convergence tolerance in radians (method="newton"
        only)
    :param method: "newton" (default) uses `_closest_theta_newton` --
        `scipy.optimize.newton` supplied with `surf`'s own analytic theta
        derivatives as func/fprime, vectorized across every reference
        point at once. "newton_manual" uses `_closest_theta_newton_manual`
        -- the same math, hand-rolled as a plain Python for-loop instead
        of delegated to scipy (a profiling baseline, see
        examples/2_Intermediate/compare_closest_theta_methods.py).
        "scipy" uses `_closest_theta_scipy` -- `scipy.optimize.minimize_scalar`,
        called once per point in a Python loop (derivative-free, much
        slower; also for comparison).
    :return: array of shape `reference.R_ref.shape` (n_phi, ntheta) of
        nearest-point (R, Z) distances
    '''
    if method == "newton":
        closest_theta = lambda *a, **kw: _closest_theta_newton(
            *a, newton_iters=newton_iters, tol=tol, **kw
        )
    elif method == "newton_manual":
        closest_theta = lambda *a, **kw: _closest_theta_newton_manual(
            *a, newton_iters=newton_iters, tol=tol, **kw
        )
    elif method == "scipy":
        closest_theta = _closest_theta_scipy
    else:
        raise ValueError(
            "method must be 'newton', 'newton_manual', or 'scipy', "
            f"got {method!r}"
        )

    errors = np.empty_like(reference.R_ref)
    for i, phi in enumerate(reference.phi_1d):
        d, _ = closest_theta(
            surf, phi, reference.R_ref[i], reference.Z_ref[i],
            ntheta_search=ntheta_search,
        )
        errors[i] = d
    return errors

def frechet_distance(
        surf1,
        surf2,
        nu=64,
        nv=64,
        plot=False,
    ):
    '''
    Compute the Frechet distance between two SurfaceRZFourier surfaces,
    computed on `nv` toroidal cross-sections. The Jaccard index is the
    ratio between the intersection and union of two different shapes.
    The intersection and union are computed here using the `shapely`
    package, and treats each toroidal cross-sections as an `nu`-gon. 
    
    :param surf1: SurfaceRZFourier for first surface
    :param surf2: SurfaceRZFourier for second surface
    :param nu: number of poloidal points at which to evaluate surfaces
    :param nv: number of toroidal points at which to evaluate surfaces
    :param plot: whether to plot surfaces as evaluated on grids
    '''

    R_uz_1, z_uz_1, eval_grid = any_to_uz_grid(
        surf1,
        nu=nu,
        nv=nv,
        plot=plot
    )

    R_uz_2, z_uz_2, _ = any_to_uz_grid(
        surf2,
        nu=nu,
        nv=nv,
        plot=plot
    )

    j = []

    for i, zeta in enumerate(eval_grid.T[1,:].reshape(nu, nv)):
        coords1 = list(zip(R_uz_1[:-1,i], z_uz_1[:-1,i]))
        coords2 = list(zip(R_uz_2[:-1,i], z_uz_2[:-1,i]))
        poly1 = shapely.Polygon(coords1)
        poly2 = shapely.Polygon(coords2)
        j.append(shapely.frechet_distance(poly1.boundary, poly2.boundary))
    return j

def hausdorff_distance(
        surf1,
        surf2,
        nu=64,
        nv=64,
        plot=False,
    ):
    '''
    Compute the Jaccard index between two SurfaceRZFourier surfaces,
    computed on `nv` toroidal cross-sections. The Jaccard index is the
    ratio between the intersection and union of two different shapes.
    The intersection and union are computed here using the `shapely`
    package, and treats each toroidal cross-sections as an `nu`-gon. 
    
    :param surf1: SurfaceRZFourier for first surface
    :param surf2: SurfaceRZFourier for second surface
    :param nu: number of poloidal points at which to evaluate surfaces
    :param nv: number of toroidal points at which to evaluate surfaces
    :param plot: whether to plot surfaces as evaluated on grids
    '''

    R_uz_1, z_uz_1, eval_grid = any_to_uz_grid(
        surf1,
        nu=nu,
        nv=nv,
        plot=plot
    )

    R_uz_2, z_uz_2, _ = any_to_uz_grid(
        surf2,
        nu=nu,
        nv=nv,
        plot=plot
    )

    j = []

    for i, zeta in enumerate(eval_grid.T[1,:].reshape(nu, nv)):
        coords1 = list(zip(R_uz_1[:-1,i], z_uz_1[:-1,i]))
        coords2 = list(zip(R_uz_2[:-1,i], z_uz_2[:-1,i]))
        poly1 = shapely.Polygon(coords1)
        poly2 = shapely.Polygon(coords2)
        j.append(shapely.hausdorff_distance(poly1.boundary, poly2.boundary))
    return j

def print_dofs_nicely(surf, lb=None, ub=None):
    if lb is None and ub is None:
        dofs_lb_ub = list(zip(surf.x, surf.lower_bounds, surf.upper_bounds))
    else:
        dofs_lb_ub = list(zip(surf.x, lb, ub))
    dofs_dict = dict(zip(surf.dof_names, dofs_lb_ub))
    print("{:<30} {:<20} {:<20} {:<20}".format('dof','value','lower bound', 'upper bound'))
    for k, v in dofs_dict.items():
        val, lb, ub = v
        print("{:<30} {:<20} {:<20} {:<20}".format(k, val, lb, ub))