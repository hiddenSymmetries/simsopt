from simsopt.geo import SurfaceBSpline
import matplotlib.pyplot as plt
import numpy as np
import fnmatch
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition
import matplotlib.colors as mpl_colors
from simsopt import save 
np.random.seed(0)

mpi = MpiPartition()
mpi.write()

def write_doflist_maxlist_minlist(
        spline_kwargs
    ):

    template_surf = SurfaceBSpline(
        **spline_kwargs
    )
    template_surf.axis.fix('r_axis_0')

    doflist = template_surf.dof_names
    
    lb = np.copy(template_surf.lower_bounds)
    ub = np.copy(template_surf.upper_bounds)

    cs_r_indices = [fnmatch.fnmatch(dof, 'CrossSectionFixedZeta*r*') for dof in template_surf.dof_names]
    r_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*r_axis*') for dof in template_surf.dof_names]
    z_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*z_axis*') for dof in template_surf.dof_names]
    
    lb[cs_r_indices] = 0.1
    ub[cs_r_indices] = 0.6

    lb[r_axis_indices] = 0.7
    ub[r_axis_indices] = 1.2

    lb[z_axis_indices] = -0.5
    ub[z_axis_indices] = 0.5

    lb_dict = dict(zip(doflist, lb))
    ub_dict = dict(zip(doflist, ub))

    # print(f'lb_dict: {lb_dict}')
    # print(f'ub_dict: {ub_dict}')

    return doflist, ub, lb

if __name__ == "__main__":

    spline_kwargs = {
        'axis_points':3,
        'points_per_cs':12,
        'n_cs':5,
        'nfp':2,
        'M':12,
        'N':12,
        'p_u':3,
        'p_v':3,
        'cs_equispaced':True,
        'rays_equispaced':True,
        'cs_global_angle_free':False,
        'axis_angles_fixed':True,
        'cs_basis':'polar',
        'nurbs': False,
    }

    doflist, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)
    x = np.random.uniform(lb, ub)

    spline_surf = SurfaceBSpline(**spline_kwargs)
    spline_surf.axis.fix('r_axis_0')
    spline_surf.set_dofs_from_vec(x)

    spline_surf.plot()

    rz_surf = spline_surf.to_RZFourier(
        nu=64,
        nv=64,
        nv_interp=128,
        nu_interp=128,
        collocation='arclength',
        plot=True,
    #     spec_cond='direct',
    #     spec_cond_options={
    #         'verbose':True,
    #         'method':'trf',
    #         'Fourier_continuation':True
    # } 
        spec_cond='variational',
        spec_cond_options={
        'plot':False,
        'ftol':1e-4,
        'Mtol':1.1,
        'shapetol':None,
        'niters':4000,
        'verbose':True,
        'cutoff':1e-5
    }
    )

    fig, axes = plt.subplots(1, 2)

    for j_rz in range(2):
        if j_rz == 0:
            data_to_plot = rz_surf.rc
            data_name = 'Rmnc'
        else:
            data_to_plot = rz_surf.zs
            data_name = 'Zmns'

        ax = axes[j_rz]
        extent = (-rz_surf.ntor - 0.5, rz_surf.ntor + 0.5, rz_surf.mpol + 0.5, -0.5)
        colorbar_min = 1e-6
        # Replace white -> blue using maximum
        final_data = np.maximum(colorbar_min, np.abs(data_to_plot / rz_surf.minor_radius()))
        im = ax.imshow(final_data, extent=extent, norm=mpl_colors.LogNorm(vmin=colorbar_min, vmax=10))
        fig.colorbar(im, ax=ax)
        ax.set_title(data_name, fontsize=9)
        ax.set_xlabel('n / nfp')
        ax.set_ylabel('m')

    output_data = [ub, lb, spline_surf, 'asdfasdf', 1]
    save(output_data, filename='outputs.json')

    # vmec = Vmec.vmec_from_surf(
    #     nfp=rz_surf.nfp,
    #     surf=rz_surf,
    #     mpi=mpi,
    #     ntheta=64,
    #     nzeta=64,
    #     ns=13,
    #     M=16, 
    #     N=16,
    #     ftol=1e-7
    # )

    # vmec.run()






    
    