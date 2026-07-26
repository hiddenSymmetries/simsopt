from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
import shapely

__all__ = [
    'any_to_uz_grid', 'jaccard_index', 'AngleMatchedReference',
    'build_angle_matched_reference', 'angle_matched_shape_error',
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