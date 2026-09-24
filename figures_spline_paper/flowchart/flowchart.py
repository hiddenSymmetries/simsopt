import booz_xform as bx
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.pyplot import rcParams
from simsopt.geo import Surface, SurfaceBSpline, SurfaceRZFourier


def vmec_contour_plot(vmec, axlist):

    ns = vmec.wout.ns

    sarr = [1, ns / 4, ns / 2, ns - 2]

    assert len(sarr) == len(axlist)

    b = bx.Booz_xform()
    b.read_wout("wout_default_000_000000.nc")

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
        phimax_lab = "2$\pi$"
    elif nfp % 2 == 1:
        phimax_lab = "2$\pi$/" + str(int(nfp))
    else:
        if nfp == 2:
            phimax_lab = "$\pi$"
        else:
            phimax_lab = "$\pi$/" + str(int(nfp / 2))

    B = np.zeros((4, ntheta, nphi))

    for js in range(len(sarr)):
        s = sarr[js]

        if b.bmnc[1, 1] < 0:
            phimin = np.pi / nfp
        else:
            phimin = 0
        phimax = phimin + 2 * np.pi / nfp

        phis = np.linspace(0, 2 * np.pi / nfp, nphi) + phimin
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        phis2D, thetas2D = np.meshgrid(phis, thetas)

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
            B[js, :, :] += bmnc[js, jmn] * cosangle[:, :]
            if lasym == True:
                sinangle = np.sin(angle)
                B[js, :, :] += bmns[jmn] * sinangle[:, :]

    for js in range(len(sarr)):
        s = sarr[js]
        ax = axlist[js]
        print(js)
        ax.contour(
            phis2D - phimin,
            thetas2D,
            B[js, :, :],
            nconts,
            linewidths=1.0,
            cmap=cm.plasma,
        )

        ax.set_yticks([0, 2 * np.pi], ["0", "$2\pi$"])
        ax.set_ylabel(r"$\theta$", labelpad=-12)
        ax.set_xticks([0, 2 * np.pi / nfp], ["0", phimax_lab])
        ax.set_xlabel(r"$\varphi$", labelpad=-7)

        # plt.colorbar()
        # axs.append(plt.gca())
        # plt.gca().tick_params(direction='in', length=0)
        color = (0.9, 0.9, 0.9)
        ax.text(
            0.05,
            6.13,
            f"$s={s:.4f}$",
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.1",
                ec=color,
                fc=color,
            ),
        )
        # plt.yticks([0, 2 * np.pi], ['0', '$2\pi$'])
        # plt.ylabel(r'$\theta$', labelpad=-12)
        # plt.xticks([0, 2*np.pi/nfp], ['0', phimax_lab])
        # plt.xlabel(r'$\varphi$', labelpad=-7)

    plt.subplots_adjust(
        left=0.054,
        bottom=0.07,
        right=0.94,
        top=0.982,
        wspace=0.39,
        hspace=0.244,
    )


if __name__ == "__main__":
    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 6,
        "n_cs": 5,
        "nfp": 5,
        "M": 12,
        "N": 12,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": True,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
    }

    surf = SurfaceBSpline(**spline_kwargs)

    surf.x = np.array(
        [
            4.05247181e-62,
            2.18674898e-01,
            1.61673161e-01,
            9.27558884e-02,
            1.18847652e00,
            1.57079633e00,
            5.55176298e-41,
            2.15958282e-01,
            1.73734624e-01,
            8.15742795e-02,
            2.17619686e-01,
            1.37203386e-01,
            6.41839361e-01,
            1.57079633e00,
            3.66519143e00,
            4.71238898e00,
            4.85176018e00,
            1.40532392e-01,
            2.16056728e-01,
            1.24474530e-01,
            8.96945769e-02,
            2.11725895e-01,
            3.97578557e-03,
            6.59295508e-01,
            1.57079633e00,
            3.66519143e00,
            4.57136006e00,
            4.71238898e00,
            1.64293469e-01,
            1.64267848e-01,
            7.96049660e-02,
            1.16422038e-01,
            1.79508228e-01,
            9.93539362e-02,
            1.17513485e00,
            2.41146910e00,
            3.66519143e00,
            3.66519143e00,
            5.68966332e00,
            1.05535800e-01,
            1.59739705e-01,
            1.47522042e-01,
            1.48479889e-01,
            9.84174928e-01,
            3.14159265e00,
            1.09388638e00,
            1.08348931e00,
            8.99041841e-01,
            6.09278924e-02,
        ]
    )

    ########################
    # plotting

    # style
    rcParams.update(
        {
            "font.size": 8,
            "text.usetex": True,
            "font.weight": "book",
            "font.family": "Futura PT",
            "lines.linewidth": 1,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    paper_width = (174) / 25.4  # width of paper in inches
    fig_height = 2.5  # figure height in inches
    cmap = mpl.colormaps["plasma"]
    color = cmap(np.linspace(0.8, 0.2, 3))
    style_dict_cond = {"label": "Condensed", "color": color[2], "ls": "-"}

    fig = plt.figure(figsize=(paper_width, fig_height))
    gs = gridspec.GridSpec(1, 5, wspace=0.1, width_ratios=[5, 3, 5, 3, 5])
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    ax5 = fig.add_subplot(gs[0, 4], projection="3d")
    ax1.view_init(elev=30, azim=45, roll=0)
    ax3.view_init(elev=30, azim=45, roll=0)
    ax5.view_init(elev=30, azim=45, roll=0)

    ax3.set_axis_off()
    ax5.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[0, 3])
    ax2.set_axis_off()
    ax4.set_axis_off()

    box = 0.8

    ax1.set_ylim(-box, box)
    ax1.set_xlim(-box, box)
    ax1.set_zlim(-box, box)
    ax1.set_title(
        r"\begin{align*} \textbf{S}(u,v) &= [X, Y, Z]^T(u,v)\\&= \sum_{i,j}B_{i,p}(u)B_{j,q}(v) \textbf{p}_{ij}\end{align*}",
        fontsize=10,
    )

    ax3.set_ylim(-box, box)
    ax3.set_xlim(-box, box)
    ax3.set_zlim(-box, box)
    ax3.set_title("${R(u,\\varphi), Z(u,\\varphi)}$", fontsize=10)

    ax5.set_ylim(-box, box)
    ax5.set_xlim(-box, box)
    ax5.set_zlim(-box, box)
    ax5.set_title(
        r"\begin{align*}R_{mn} = \frac{1}{N_\theta N_\varphi}\sum_{i,j} R(\theta_i, \varphi_i) \cos(m\theta_i-nN_{fp}\varphi_i),\\Z_{mn} = \frac{1}{N_\theta N_\varphi}\sum_{i,j} Z(\theta_i, \varphi_j) \sin(m\theta_i-nN_{fp}\varphi_j)\end{align*}",
        fontsize=8,
    )

    ax2.text(
        0.5,
        0.6,
        "Evaluate on\n$(u, \\varphi)$ grid",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax2.transAxes,
        fontsize=10,
    )
    ax4.text(
        0.5,
        0.6,
        "F.T.",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax4.transAxes,
        fontsize=12,
    )
    ax2.text(
        0.5,
        0.45,
        "$\\rightarrow$",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax2.transAxes,
        fontsize=20,
    )
    ax4.text(
        0.5,
        0.45,
        "$\\rightarrow$",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax4.transAxes,
        fontsize=20,
    )

    surf.plot(
        _surf=True,
        _surf_points=False,
        _ctrl_points=True,
        _ctrl_points_full=False,
        _pseudo_axis=True,
        _centroid_axis=True,
        _rtz_vectors=True,
        _RZ_vectors=False,
        _surf_kwargs={
            "color": "#FFFF66",
            "alpha": 0.3,
            "rcount": 200,
            "ccount": 200,
        },
        _ctrl_points_kwargs={
            "color": "#0066FF",
            "marker": ".",
            "ls": "--",
            "markersize": 1,
            "linewidth": 1,
        },
        _pseudo_axis_kwargs={"color": "#6A00FF", "ls": "-", "lw": 1},
        _centroid_axis_kwargs={"color": "#99B3FF", "ls": "--", "lw": 1},
        _pseudo_axis_ctrl_pts_kwargs={
            "color": "#000000",
            "marker": ".",
            "lw": 0.5,
            "markersize": 2,
        },
        _rtz_vectors_kwargs={"color": "#0066FF", "lw": 1},
        ax=ax1,
    )

    # Collocation points used by the Fourier transform, sampled at the
    # Nyquist-critical rate for M=N=12 (2*M+1 points resolve M harmonics).
    M, N = surf.M, surf.N
    nu_colloc = 2 * M + 1
    nv_colloc = 2 * N * surf.nfp + 2

    R_colloc, z_colloc, zeta_colloc, theta_colloc = surf.exact_tz_interp(
        nu=nu_colloc, nv=nv_colloc
    )
    x_colloc = R_colloc.T * np.cos(zeta_colloc)
    y_colloc = R_colloc.T * np.sin(zeta_colloc)
    z_colloc_pts = z_colloc.T

    # ax3.plot_wireframe(
    #     x_colloc,
    #     y_colloc,
    #     z_colloc_pts,
    #     color="#0066FF",
    #     linewidth=0.4,
    #     alpha=0.3,
    # )
    ax3.scatter(
        x_colloc,
        y_colloc,
        z_colloc_pts,
        alpha=1,
        s=0.005,
        color="#0066FF",
        depthshade=True,
    )

    rz_surf = surf.to_RZFourier(spec_cond=None)
    rz_surf_plot = SurfaceRZFourier(
        nfp=rz_surf.nfp,
        mpol=rz_surf.mpol,
        ntor=rz_surf.ntor,
        quadpoints_theta=Surface.get_theta_quadpoints(ntheta=200),
        quadpoints_phi=Surface.get_phi_quadpoints(nphi=200, nfp=rz_surf.nfp),
    )
    rz_surf_plot.x = rz_surf.x
    rz_surf_plot.plot(
        ax=ax5,
        show=False,
        close=True,
        axis_equal=False,
        color="#0066FF",
        rcount=200,
        ccount=200,
        alpha=1,
    )

    def set_zoom(ax, zoom, orig):
        def get_proj():
            M = orig()
            S = np.diag([zoom, zoom, zoom, 1.0])  # scale x and y (zoom)
            return M @ S

        ax.get_proj = get_proj

    set_zoom(ax1, 1.3, ax1.get_proj)
    # set_zoom(ax3, 1.3, ax3.get_proj)
    # set_zoom(ax5, 1.3, ax5.get_proj)

    # plt.show()

    plt.savefig(
        fname="transforms.pdf",
        bbox_inches="tight",
        # pad_inches=0.05, dpi=600
    )
