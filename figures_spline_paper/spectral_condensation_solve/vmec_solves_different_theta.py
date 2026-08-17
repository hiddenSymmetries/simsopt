import matplotlib as mpl
import matplotlib.gridspec as gridspec

# plotting
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import rcParams
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.mhd.vmec import Vmec


def flux_surface_cross_sections(wout_file, phi=0.0, nrho=15, ntheta=400):
    """
    (X, Y, Z) cross sections at cylindrical angle `phi` (normalized by
    2*pi) for `nrho` flux surfaces spanning the interior out to the
    boundary (s=1), each built directly from the wout file via
    SurfaceRZFourier.from_wout(s=...) and Surface.cross_section(phi) --
    replaces the previous hand-rolled Fourier-mode-summation loops.

    Returns a list of (ntheta, 3) arrays, ordered from the core (s
    smallest) to the boundary (s=1, last entry).
    """
    s_values = np.linspace(1.0 / nrho, 1.0, nrho)
    cross_sections = []
    for s in s_values:
        surf = SurfaceRZFourier.from_wout(wout_file, s=s)
        xyz = surf.cross_section(phi, thetas=ntheta)
        cross_sections.append(xyz)
    return cross_sections


def spectral_power(vmec_input):
    """
    Sum_n Rmn^2 + Zmn^2 vs m, from a Vmec object loaded from an
    input.* file (i.e. the boundary as specified, before VMEC's
    equilibrium solve). indata.rbc/zbs are padded fortran arrays sized
    to a fixed maximum mode count regardless of the actual mpol/ntor
    used -- sliced down to [:, :mpol+1] for a clean plot (the padding
    columns are exactly zero, so summing over the full n-range is
    harmless, but the m-range needs slicing to avoid a long flat zero
    tail).
    """
    mpol = int(vmec_input.indata.mpol)
    rbc = vmec_input.indata.rbc[:, : mpol + 1]
    zbs = vmec_input.indata.zbs[:, : mpol + 1]
    power = np.einsum("nm->m", rbc**2 + zbs**2)
    m = np.arange(mpol + 1)
    return m, power


def ftol_history(vmec_wout, vmec_input):
    """
    Total force residual (fsqt) vs iteration, from a Vmec object loaded
    from a wout_*.nc file. fsqt is padded with trailing zeros past
    convergence -- trimmed here since a hard zero can't be shown on the
    log-scale plot anyway.

    wout.itfsq is NOT the recording interval -- it's a running *count*
    of how many entries got stored (VMEC2000's eqsolve.f increments it
    once per stored point, capped at the compiled constant
    nstore_seq=100). The actual interval between stored iterations is
    niter_requested // nstore_seq + 1, and niter_requested (the run's
    configured max iteration count) isn't retained in the wout file at
    all -- wout.niter holds the *actual* final iteration reached
    instead -- so it has to come from the corresponding input.* file's
    niter_array[0].
    """
    nstore_seq = 100
    fsqt = np.asarray(vmec_wout.wout.fsqt)
    n_valid = np.count_nonzero(fsqt)
    fsqt = fsqt[:n_valid]
    niter_requested = int(vmec_input.indata.niter_array[0])
    interval = niter_requested // nstore_seq + 1
    iterations = interval * np.arange(1, n_valid + 1)
    return iterations, fsqt


def build_spline_surf():
    """
    The same SurfaceBSpline (spline_kwargs + dofs) used by
    run_vmec_different_theta.py to produce the wout/input files this
    script reads -- the ground-truth boundary that the "exact" and
    "cond" VMEC boundary representations are each approximating.
    """
    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 4,
        "n_cs": 6,
        "nfp": 3,
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
        "knot_parametrization": "uniform",
    }
    spline_surf = SurfaceBSpline(default_r=0.3, **spline_kwargs)
    new_x = np.array(
        [
            5.3896343573543060e-02,
            5.0883230048103856e-01,
            2.2489121027379422e-01,
            1.5929768581422210e00,
            1.0304048534365813e-01,
            4.3412658709604501e-01,
            1.9194536849729077e-01,
            5.3647438961678784e-01,
            1.7794171743465446e00,
            3.6038044630128279e00,
            4.9259924741397159e00,
            2.5036189767220018e-01,
            3.5802891724900809e-01,
            9.6212580154516142e-02,
            5.2064914206501489e-01,
            2.0474558259649123e00,
            3.8964954930212143e00,
            5.3228745694944024e00,
            3.2185070391451231e-01,
            2.5146308466749123e-01,
            2.1954123698502295e-01,
            4.5420692619007164e-01,
            2.2214282741552713e00,
            3.0009540172725946e00,
            5.4975172012762998e00,
            3.6991589429711308e-01,
            1.6285912057838300e-01,
            3.6446964917388275e-01,
            3.2242251631177343e-01,
            1.7920172479908414e00,
            3.0210349920179866e00,
            5.4977786325813431e00,
            3.8958534284832108e-01,
            1.9630111359305188e-01,
            4.0847103311261590e-01,
            1.0489177365984679e00,
            2.0043666242346596e00,
            1.3580534688161376e00,
            5.0279307449500388e-01,
            -7.0064725504961656e-01,
        ]
    )
    spline_surf.x = new_x
    return spline_surf


if __name__ == "__main__":
    # Colors used consistently for these two cases everywhere they're
    # plotted (overlay, spectral power, convergence), drawn from the same
    # "plasma" colormap used for the per-case flux-surface gradient below.
    cmap = mpl.colormaps["plasma"]
    cases = [
        {
            "label": r"$\theta=u$",
            "wout": "wout_exact_000_000000.nc",
            "input": "input.exact_000_000000",
            "color": cmap(0.45),
        },
        {
            "label": "Condensed",
            "wout": "wout_cond_000_000000.nc",
            "input": "input.cond_000_000000",
            "color": cmap(0.15),
        },
    ]

    for case in cases:
        case["vmec_wout"] = Vmec(case["wout"])
        vmec_input = Vmec(case["input"])
        # .boundary is populated by an explicit element-by-element copy in
        # Vmec.__init__ (not a live view like .indata), so it stays valid
        # after later Vmec(input_file) constructions -- safe to keep.
        case["boundary"] = vmec_input.boundary
        case["m"], case["power"] = spectral_power(vmec_input)
        case["niter_requested"] = int(vmec_input.indata.niter_array[0])

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
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    paper_width = 174 / 25.4  # width of paper in inches
    row_height = 3.5 / 4  # figure height per row, in inches

    n_bottom_rows = len(cases)
    top_ratio = 2 * n_bottom_rows  # top row much taller than the bottom rows

    fig = plt.figure(
        figsize=(paper_width, row_height * (top_ratio + n_bottom_rows)),
        dpi=600,
    )
    gs = gridspec.GridSpec(
        1 + n_bottom_rows,
        3,
        height_ratios=[top_ratio] + [1] * n_bottom_rows,
        wspace=0.6,
        hspace=0.4,
    )

    # top row, spanning the full width: the ground-truth spline boundary
    # and each VMEC boundary representation ("exact" vs "cond"), overlaid
    # at the same cylindrical angle to compare shapes directly
    ax_overlay = fig.add_subplot(gs[0, :])
    phi_array = np.linspace(0, 1/6, 6)
    ntheta_overlay = 400

    spline_surf = build_spline_surf()
    for phi in phi_array:
        if phi == 0:
            label = "Spline (ground truth)"
        else:
            label = None
        xyz_spline = spline_surf.cross_section(phi, thetas=ntheta_overlay)
        R_spline = np.hypot(xyz_spline[:, 0], xyz_spline[:, 1])
        Z_spline = xyz_spline[:, 2]
        ax_overlay.plot(
            np.append(R_spline, R_spline[0]),
            np.append(Z_spline, Z_spline[0]),
            "k--",
            linewidth=0.2,
            label=label,
        )

    ax_spectral_power = fig.add_subplot(gs[1:, 1])
    ax_convergence = fig.add_subplot(gs[1:, 2])

    izeta = 0

    for row, case in enumerate(cases):
        vmec_wout = Vmec(case["wout"])
        vmec_input = Vmec(case["input"])

        # top row: this case's VMEC boundary representation, overlaid on
        # the ground-truth spline boundary
        for phi in phi_array:
            if phi == 0:
                label = case["label"]
            else:
                label = None
            xyz_boundary = vmec_input.boundary.cross_section(
                phi, thetas=ntheta_overlay
            )
            R_boundary = np.hypot(xyz_boundary[:, 0], xyz_boundary[:, 1])
            Z_boundary = xyz_boundary[:, 2]
            ax_overlay.plot(
                np.append(R_boundary, R_boundary[0]),
                np.append(Z_boundary, Z_boundary[0]),
                color=case["color"],
                label=label,
                linewidth=0.2,
            )

        # column 1: boundary + interior flux-surface contours
        ax = fig.add_subplot(gs[row + 1, 0])
        cross_sections = flux_surface_cross_sections(case["wout"], phi=0.12)
        for i, xyz in enumerate(cross_sections):
            R = np.hypot(xyz[:, 0], xyz[:, 1])
            Z = xyz[:, 2]
            ax.plot(
                np.append(R, R[0]),
                np.append(Z, Z[0]),
                color=case["color"],
                linewidth=0.5,
            )
        ax.set_aspect(True)
        ax.set_title(case["label"])
        if row == 0:
            ax.set_title("(b)", loc="left", fontweight="bold")
        ax.set_axis_off()

        # column 2: spectral power

        m, power = spectral_power(vmec_input)
        ax_spectral_power.semilogy(
            m, power, color=case["color"], label=case["label"]
        )
        ax_spectral_power.set_xlabel("$m$")
        ax_spectral_power.set_ylabel(r"$\sum_n R_{mn}^2 + Z_{mn}^2$")

        # column 3: ftol convergence
        iterations, fsqt = ftol_history(vmec_wout, vmec_input)
        ax_convergence.semilogy(
            iterations, fsqt, color=case["color"], label=case["label"]
        )
        ax_convergence.set_xlabel("Iteration")
        ax_convergence.set_ylabel(r"$\overline{F}^2$")

    ax_overlay.set_aspect(True)
    ax_overlay.set_title("(a)", loc="left", fontweight="bold")
    ax_overlay.set_xlabel("$R$")
    ax_overlay.set_ylabel("$Z$")
    ax_overlay.legend(fontsize=7, loc="best", frameon=False)

    # Zoomed inset: the three representations are nearly indistinguishable
    # at this resolution, so re-plot every already-drawn line into a small
    # inset axes zoomed 1500x into the region where the spline and condensed
    # boundaries diverge the most -- found via a nearest-point (not
    # same-index -- each curve traces the same shape at a different
    # "speed", so same-index points aren't at the same physical location)
    # geometric distance scan between the two curves (true max divergence
    # there is only ~0.001, against a ~1.3 plot scale -- 20x isn't nearly
    # enough magnification to make that visible; 1500x makes the gap span
    # most of the inset's width).
    # The inset box itself is placed in data coordinates (transData),
    # entirely to the right of the rightmost plotted boundary point (max
    # R ~1.8), so it can't cover any curve regardless of its Z placement;
    # mark_inset then draws the connector lines back to the zoomed-in
    # region.
    axins = ax_overlay.inset_axes(
        [2.15, -0.2, 0.4, 0.4], transform=ax_overlay.transData
    )
    for line in ax_overlay.get_lines():
        axins.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            linestyle=line.get_linestyle(),
        )
    # Tip (max-Z point) of the phi=0 slice's spline curve -- the other two
    # representations' own tips sit within ~3e-4 of this in R, Z, so a
    # single shared center is fine.
    zoom_R, zoom_Z = 1.56759, 0.50450
    main_data_span = 1.318  # max of the plotted R and Z extents
    zoom_half_width = main_data_span / (2 * 1700)  # 1500x zoom
    axins.set_xlim(zoom_R - zoom_half_width, zoom_R + zoom_half_width)
    axins.set_ylim(zoom_Z - zoom_half_width, zoom_Z + zoom_half_width)
    axins.set_aspect(True)
    # Ticks (rather than none) so the actual R, Z scale here -- and thus
    # how tiny this divergence really is -- is visible to the reader.
    axins.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    axins.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    axins.tick_params(axis="both", which="major", labelsize=4, length=2, pad=1)
    axins.xaxis.get_offset_text().set_fontsize(4)
    axins.yaxis.get_offset_text().set_fontsize(4)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.5")
    mark_inset(ax_overlay, axins, loc1=1, loc2=3, fc="none", ec="0.5", lw=0.5)

    ax_spectral_power.set_title("(c)", loc="left", fontweight="bold")
    ax_convergence.set_title("(d)", loc="left", fontweight="bold")

    ax_convergence.legend(frameon=False)
    ax_spectral_power.legend(frameon=False)

    # plt.show()
    plt.savefig(
        dpi=600,
        fname="spec_cond_solve.pdf",
        bbox_inches="tight",
    )

    # plt.show()
