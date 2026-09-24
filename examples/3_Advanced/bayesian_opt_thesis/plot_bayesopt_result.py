#!/usr/bin/env python3

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import booz_xform as bx
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition

mpi=MpiPartition()

spline_kwargs = {
    'axis_points':3,
    'points_per_cs':6,
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

def vmec_contour_plot(vmec, axlist):    

    ns = vmec.wout.ns

    sarr = [ns-2] #[1, ns/4, ns/2, ns-2]

    assert len(sarr) == len(axlist)

    b = bx.Booz_xform()
    b.read_wout('wout_default_000_000000.nc')

    mpol = 20
    ntor = 20

    b.mboz = mpol
    b.nboz = ntor
    b.compute_surfs = sarr
    b.run()

    sarr = (np.array(sarr) + 2) / ns

    nphi = 100
    ntheta = 100

    nconts = 18

    axs = []

    nfp = b.nfp
    lasym = b.asym

    if nfp == 1:
        phimax_lab = '2$\pi$'
    elif nfp % 2 == 1:
        phimax_lab = '2$\pi$/' + str(int(nfp))
    else:
        if nfp == 2:
            phimax_lab = '$\pi$'
        else:
            phimax_lab = '$\pi$/' + str(int(nfp/2))

    B = np.zeros((4, ntheta, nphi))

    for js in range(len(sarr)):
        s = sarr[js]

        if b.bmnc[1,1] < 0:
            phimin = np.pi/nfp
        else:
            phimin = 0
        phimax = phimin + 2*np.pi/nfp

        phis = np.linspace(0,2*np.pi/nfp,nphi) + phimin
        thetas = np.linspace(0,2*np.pi,ntheta)
        phis2D,thetas2D=np.meshgrid(phis,thetas)

        # Extract relevant quantities from Boozer object
        # Mode numbers
        xm_nyq = b.xm_b
        xn_nyq = b.xn_b

        # Fourier coefficients
        bmnc = np.array(b.bmnc_b).T
        bmns = np.array(b.bmns_b).T

        # Loop through fourier modes, construct |B| array
        for jmn in range(len(xm_nyq)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * thetas2D - n * phis2D
            cosangle = np.cos(angle)
            B[js,:,:] += bmnc[js,jmn] * cosangle[:,:]
            if lasym == True:
                sinangle = np.sin(angle)
                B[js,:,:] += bmns[jmn] * sinangle[:,:]

    from matplotlib import cm
    for js in range(len(sarr)):
        s = sarr[js]
        ax = axlist[js]
        print(js)
        ax.contour(phis2D-phimin,thetas2D,B[js,:,:], nconts, linewidths=1.0,cmap=cm.plasma)

        ax.set_yticks([0, 2 * np.pi], ['0', '$2\pi$'])
        ax.set_ylabel(r'$\theta$', labelpad=-12)
        ax.set_xticks([0, 2*np.pi/nfp], ['0', phimax_lab])
        ax.set_xlabel(r'$\varphi$', labelpad=-7)

        # plt.colorbar()
        # axs.append(plt.gca())
        # plt.gca().tick_params(direction='in', length=0)
        color = (0.9, 0.9, 0.9)
        ax.text(0.05, 6.13, f"$s={s:.4f}$", ha='left', va='top',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.1",
                        ec=color,
                        fc=color,
                        ))
        # plt.yticks([0, 2 * np.pi], ['0', '$2\pi$'])
        # plt.ylabel(r'$\theta$', labelpad=-12)
        # plt.xticks([0, 2*np.pi/nfp], ['0', phimax_lab])
        # plt.xlabel(r'$\varphi$', labelpad=-7)

    plt.subplots_adjust(left=0.054, bottom=0.07, right=0.94, top=0.982, wspace=0.39, hspace=0.244)

def parse_bo_log(filename):
    with open(filename, "r") as f:
        text = f.read()

    best_vals = []
    mean_candidate_vals = []
    lengthscales_per_iter = []
    iter_ids = []
    dofs_per_iter = []

    # Capture iteration blocks like:
    # 0/500:
    #   ...
    # Current best: ...
    # ...
    # Length scales: tensor([[...]], dtype=torch.float64)
    #
    # and stop right before the next iteration block or end of file
    block_pattern = re.compile(
        r'(\d+)/(\d+):\s*(.*?)(?=\n\d+/\d+:\s*|\Z)',
        re.DOTALL
    )

    blocks = block_pattern.findall(text)

    for iter_str, total_str, block in blocks:
        iteration = int(iter_str)

        # All queried candidate values in this iteration
        cand_vals = [
            float(x)
            for x in re.findall(
                r'f\(candidate\d+\):\s*tensor\(\[\s*([-\deE+.]+)\s*\]\)',
                block
            )
        ]

        # Current best after this iteration
        best_match = re.search(
            r'Current best:\s*([-\deE+.]+)',
            block
        )

        # Length scales tensor
        ls_match = re.search(
            r'Length scales:\s*tensor\(\[\[(.*?)\]\],\s*dtype=torch\.float64\)',
            block,
            re.DOTALL
        )

        dofs_match = re.search(
            r'at tensor\(\[(.*?)\],\s*dtype=torch\.float64\)',
            block,
            re.DOTALL
        )

        if best_match is None or ls_match is None or dofs_match is None:
            continue

        best_val = float(best_match.group(1))

        ls_str = ls_match.group(1).replace("\n", " ")
        ls_vals = np.fromstring(ls_str.replace(",", " "), sep=" ")
        dofs_str = dofs_match.group(1).replace("\n", " ")
        dofs = np.fromstring(dofs_str.replace(",", " "), sep=" ")

        if cand_vals:
            mean_candidate = float(np.mean(cand_vals))
        else:
            mean_candidate = np.nan

        iter_ids.append(iteration)
        best_vals.append(best_val)
        mean_candidate_vals.append(mean_candidate)
        lengthscales_per_iter.append(ls_vals)
        dofs_per_iter.append(dofs)

    if not lengthscales_per_iter:
        raise ValueError("No iteration blocks with length scales were parsed.")

    # Ensure all lengthscale vectors have same length
    n_ls = len(lengthscales_per_iter[0])
    for i, ls in enumerate(lengthscales_per_iter):
        if len(ls) != n_ls:
            raise ValueError(
                f"Inconsistent number of length scales at parsed iteration index {i}: "
                f"expected {n_ls}, got {len(ls)}"
            )

    return (
        dofs_per_iter[-1],
        np.array(iter_ids),
        np.array(best_vals),
        np.array(mean_candidate_vals),
        np.vstack(lengthscales_per_iter),
    )

def plot_one_case(fpath, subfig):
    dofs, iter_ids, best_vals, mean_candidate_vals, lengthscales = parse_bo_log(fpath)
    
    #gs = GridSpec(2, 3, width_ratios=[1, 1,3], hspace=0.2, figure=subfig)
    gs = subfig.add_gridspec(1, 4, width_ratios=[1, 1, 2.5, 2.5], hspace=0.3, wspace=0.3)

    ax_top = subfig.add_subplot(gs[0, 2])
    ax_bot = subfig.add_subplot(gs[0, 3])
    ax_splineplot = subfig.add_subplot(gs[0, 0], projection='3d')
    ax_splineplot.set_aspect('equal')
    ax_splineplot.view_init(elev=30, azim=45, roll=0)
    ax_vmecplot = subfig.add_subplot(gs[0, 1])
    ax_vmecplot.set_box_aspect(1)

    surf = SurfaceBSpline(
            **spline_kwargs
    )
    surf.axis.fix('r_axis_0')
    surf.set_dofs_from_vec(dofs)

    dof_namelist = surf.dof_names

    colors_map = {
        "r": "tab:blue",
        "theta": "tab:orange",
        "r_axis": "tab:green",
        "z_axis": "tab:red",
    }

    def classify(name):
        if "r_axis" in name:
            return "r_axis"
        elif "z_axis" in name:
            return "z_axis"
        elif re.search(r":r_\d+", name):
            return "r"
        elif re.search(r":theta_\d+", name):
            return "theta"
        else:
            raise ValueError(f"Unrecognized pattern: {name}")

    # Build the color list
    color_list = [colors_map[classify(n)] for n in dof_namelist]
    line_names = [classify(n) for n in dof_namelist]

    surf.plot(
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
            ax=ax_splineplot
    )

    rz_surf = surf.to_RZFourier(
        **ft_kwargs
    )

    vmec = Vmec.vmec_from_surf(
        surf.nfp,
        mpi,
        rz_surf,
        ns=13, 
        ftol=1e-7,
    )
    vmec.run()

    vmec_contour_plot(vmec,[ax_vmecplot])

    # Plotting BO details

    # Top panel: best-so-far target + mean queried target
    ax_top.semilogy(iter_ids, -best_vals, marker='o', markersize=3, label='Incumbent best target value')
    ax_top.semilogy(iter_ids, -mean_candidate_vals, marker='o', markersize=3, label='Mean queried target value')
    ax_top.set_xlabel("Iteration")
    #ax_top.set_ylabel("Target function value")
    ax_top.set_title("Target function value by iteration")
    ax_top.grid(True, which='both', alpha=0.3)
    ax_top.legend()

    # Bottom panel: all length scales
    n_ls = lengthscales.shape[1]

    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0

    for j in range(n_ls):
        llabel = None
        if line_names[j] == 'r' and counter1 ==0:
            llabel = '$r_{cs}$'
            counter1 += 1
        if line_names[j] == 'theta' and counter2 ==0:
            llabel = '$\\theta_{cs}$'
            counter2 += 1
        if line_names[j] == 'r_axis' and counter3 ==0:
            llabel = '$r_{ax}$'
            counter3 += 1
        if line_names[j] == 'z_axis' and counter4 ==0:
            llabel = '$z_{ax}$'
            counter4 += 1
        ax_bot.semilogy(iter_ids, lengthscales[:, j], linewidth=1, c=color_list[j], label=llabel)

    ax_bot.set_xlabel("Iteration")
    ax_bot.set_ylabel("Length scale")
    ax_bot.set_title("GP length scales by iteration")
    ax_bot.grid(True, which='both', alpha=0.3)
    ax_bot.legend()

if __name__ == "__main__":
    fpath_list=['1_percent.out',
                '3_percent.out',
                '5_percent.out',
                '10_percent.out',
                '20_percent.out',
    ]

    namelist = [
        '$p=0.01$',
        '$p=0.03$',
        '$p=0.05$',
        '$p=0.10$',
        '$p=0.20$',
    ]
    
    fig = plt.figure(figsize=(13, 13))
    subfigs = fig.subfigures(5, 1, wspace=0.07, hspace = 0.3)

    for i, subfig in enumerate(subfigs):
        plot_one_case(fpath_list[i], subfig)
        subfig.suptitle(namelist[i],
                        x=0.02,
                        ha='left'
    )

    plt.savefig('all_bo_centered.pdf', bbox_inches='tight')
