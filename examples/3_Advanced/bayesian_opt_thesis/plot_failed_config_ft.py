import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from simsopt.geo import SurfaceBSpline
import numpy as np 

spline_kwargs = {
    'axis_points':3,
    'points_per_cs':4,
    'n_cs':4,
    'nfp':2,
    'M':9,
    'N':4,
    'p_u':3,
    'p_v':3,
    'cs_equispaced':True,
    'rays_equispaced':False,
    'cs_global_angle_free':False,
    'axis_angles_fixed':True,
    'cs_basis':'polar',
    'nurbs': False,
}

ft_kwargs = {
    'collocation': 'arclength',
    'plot': False,
    'spec_cond': False,
    'spec_cond_options': {
        'plot': False,
        'ftol': 1e-4,
        'Mtol': 1.1,
        'shapetol': None,
        'niters': 5000,
        'verbose': False,
        'cutoff': 1e-6,
    }
}

if __name__ == "__main__":
    fig = plt.figure(figsize = (8,4))
    gs = GridSpec(1,2)

    ax0 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    ax0.set_box_aspect((1,1,1), zoom=1.5)
    ax0.view_init(elev=30, azim=45, roll=0,)
    ax1.view_init(elev=30, azim=45, roll=0)

    ax0.set_axis_off()
    ax1.set_axis_off()

    dofs = np.array([2.5918e-01, 2.6240e-01, 5.4234e-01, 2.7901e-03, 1.0832e-01, 9.4839e-02,
        3.4719e-01, 1.1332e-01, 1.9049e+00, 2.5640e+00, 5.4405e+00, 5.5462e-01,
        3.8733e-01, 2.3294e-01, 5.5684e-01, 1.1307e+00, 3.2890e+00, 3.9961e+00,
        3.7917e-01, 7.3292e-02, 3.4285e-01, 2.9872e-01, 8.6435e-01, 1.2797e+00,
        1.5568e-01])

    spline_surf = SurfaceBSpline(
        **spline_kwargs
    )
    spline_surf.axis.fix('r_axis_0')
    spline_surf.set_dofs_from_vec(dofs)

    spline_surf.plot(
            _surf = True,
            _surf_points = False,
            _ctrl_points = True,
            _ctrl_points_full=False,
            _pseudo_axis = True,
            _centroid_axis = True,
            _rtz_vectors = True,
            _RZ_vectors = False,
            _surf_kwargs = {'color':'#FFFF66', 'alpha':0.3, 'rcount':200, 'ccount':200},
            _ctrl_points_kwargs = {'color': '#0066FF','marker': '.','ls': '--', 'markersize':1, 'linewidth':1},
            _pseudo_axis_kwargs = { 'color': "#6A00FF",'ls': '-', 'lw':1},
            _centroid_axis_kwargs = { 'color': '#CC99FF','ls': '--', 'lw':1},
            _pseudo_axis_ctrl_pts_kwargs = { 'color': '#000000','marker': '.', 'lw':0.5, 'markersize':2},
            _rtz_vectors_kwargs = {'color':'#0066FF', 'lw':1},
            ax=ax0
    )
    
    rzsurf = spline_surf.to_RZFourier(
        **ft_kwargs
    )

    rzsurf.plot(ax=ax1, close=True, rcount=100)
    
    def zoom_3d(ax, factor):
        """
        factor < 1  -> zoom in
        factor > 1  -> zoom out
        """
        x0, x1 = ax.get_xlim3d()
        y0, y1 = ax.get_ylim3d()
        z0, z1 = ax.get_zlim3d()

        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        zc = 0.5 * (z0 + z1)

        xr = 0.5 * (x1 - x0) * factor
        yr = 0.5 * (y1 - y0) * factor
        zr = 0.5 * (z1 - z0) * factor

        ax.set_xlim3d(xc - xr, xc + xr)
        ax.set_ylim3d(yc - yr, yc + yr)
        ax.set_zlim3d(zc - zr, zc + zr)

    plt.show()


