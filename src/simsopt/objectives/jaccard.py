import numpy as np 
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
import shapely

__all__ = ['any_to_uz_grid', 'jaccard_index']

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
        coords1 = zip(R_uz_1[:-1,i], z_uz_1[:-1,i])
        coords2 = zip(R_uz_2[:-1,i], z_uz_2[:-1,i])
        poly1 = shapely.Polygon(coords1)
        poly2 = shapely.Polygon(coords2)
        j.append(shapely.intersection(poly1, poly2).area/shapely.union(poly1, poly2).area)
    return np.log10(j)

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