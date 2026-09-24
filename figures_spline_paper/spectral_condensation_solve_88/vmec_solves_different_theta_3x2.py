import matplotlib as mpl
import matplotlib.gridspec as gridspec

# plotting
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import rcParams
from matplotlib.ticker import FuncFormatter, LogLocator
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.mhd.vmec import Vmec


def sparse_decade_label(val, pos):
    """
    Label formatter for a log axis ticked at every decade (see
    LogLocator(base=10.0, numticks=15) below) -- keeps a tick/gridline at
    every decade, but only labels every other one, to reduce clutter over
    a wide (~9-12 decade) range. `pos` is the tick's index in the current
    tick list (supplied by FuncFormatter), so this stays correct
    regardless of which decade the range happens to start at.
    """
    if pos % 2 == 1 or val <= 0:
        return ""
    exponent = int(round(np.log10(val)))
    return rf"$10^{{{exponent}}}$"


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


def ftol_history(vmec_wout, niter_requested):
    """
    Total force residual (fsqt) vs iteration, from a Vmec object loaded
    from a wout_*.nc file. fsqt is padded with trailing zeros past
    convergence -- trimmed here since a hard zero can't be shown on the
    log-scale plot anyway.

    wout.itfsq is NOT the recording interval -- it's a running *count*
    of how many entries got stored (VMEC2000's eqsolve.f increments it
    once per stored point, capped at the compiled constant
    nstore_seq=100). The actual interval between stored iterations is
    niter_requested // nstore_seq + 1. niter_requested (the run's
    configured max iteration count) isn't retained in the wout file at
    all -- wout.niter holds the *actual* final iteration reached
    instead -- so it has to be passed in, read from the corresponding
    input.* file's niter_array[0] (by the caller, immediately after that
    Vmec(input_file) object was constructed -- see the caution in
    __main__ about that object's .indata not staying valid afterward).
    """
    nstore_seq = 100
    fsqt = np.asarray(vmec_wout.wout.fsqt)
    n_valid = np.count_nonzero(fsqt)
    fsqt = fsqt[:n_valid]
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
    # plotted (overlay, interior surfaces, spectral power, convergence),
    # drawn from the same "plasma" colormap.
    cmap = mpl.colormaps["plasma"]
    case_theta_u = {
        "label": r"$\theta=u$",
        "wout": "wout_exact_000_000000.nc",
        "input": "input.exact_000_000000",
        "color": cmap(0.45),
    }
    case_condensed = {
        "label": "Condensed",
        "wout": "wout_cond_000_000000.nc",
        "input": "input.cond_000_000000",
        "color": cmap(0.15),
    }
    cases = [case_theta_u, case_condensed]

    # Load each case's Vmec objects once, up front, so the per-axes blocks
    # below are each self-contained (don't re-load files case-by-case).
    # Vmec(input_file) is readin-only, but it shares *process-global*
    # Fortran state (vmec.vmec_input) across every instance -- the next
    # case's Vmec(input_file) call silently overwrites this one's .indata
    # too. So everything needed from .indata has to be pulled out right
    # now, before the next case's construction clobbers it; storing the
    # Vmec object itself for later use here would be wrong (both cases'
    # .indata would end up reflecting whichever was constructed last).
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
        }
    )
    paper_width = 174 / 25.4  # width of paper in inches
    row_height = 3.5 / 4  # figure height per row, in inches

    fig = plt.figure(
        figsize=(paper_width, 5 * row_height),
        dpi=600,
    )
    gs = gridspec.GridSpec(
        3, 2, wspace=0.4, hspace=0.2, height_ratios=[1, 1, 1.2]
    )

    # (0:2, 0): cross section overlays -- tall, spans the top two rows
    ax_overlay = fig.add_subplot(gs[0:2, 0])
    # (0, 1): theta=u interior surface plot
    ax_theta_u_interior = fig.add_subplot(gs[0, 1])
    # (1, 1): condensed interior surface plot
    ax_condensed_interior = fig.add_subplot(gs[1, 1])
    # (2, 0): spectral power
    ax_spectral_power = fig.add_subplot(gs[2, 0])
    # (2, 1): convergence
    ax_convergence = fig.add_subplot(gs[2, 1])

    ########################
    # ax_overlay: the ground-truth spline boundary and each VMEC boundary
    # representation ("exact" vs "cond"), overlaid at the same cylindrical
    # angle to compare shapes directly
    phi_array = np.linspace(0, 0.125, 5)
    ntheta_overlay = 400

    spline_surf = build_spline_surf()
    for phi in phi_array:
        label = "Spline (ground truth)" if phi == 0 else None
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

    for case in cases:
        for phi in phi_array:
            label = case["label"] if phi == 0 else None
            xyz_boundary = case["boundary"].cross_section(
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

    ax_overlay.set_aspect(True)
    ax_overlay.set_title("(a)", loc="left", fontweight="bold")
    ax_overlay.set_axis_off()
    ax_overlay.legend(fontsize=7, loc="best", frameon=False)

    ########################
    # ax_theta_u_interior: interior flux-surface contours, theta=u case
    cross_sections = flux_surface_cross_sections(case_theta_u["wout"], phi=0.12)
    for xyz in cross_sections:
        R = np.hypot(xyz[:, 0], xyz[:, 1])
        Z = xyz[:, 2]
        ax_theta_u_interior.plot(
            np.append(R, R[0]),
            np.append(Z, Z[0]),
            color=case_theta_u["color"],
            linewidth=0.5,
        )
    ax_theta_u_interior.set_aspect(True)
    ax_theta_u_interior.set_title(case_theta_u["label"])
    ax_theta_u_interior.set_title("(b)", loc="left", fontweight="bold")
    ax_theta_u_interior.set_axis_off()

    ########################
    # ax_condensed_interior: interior flux-surface contours, condensed case
    cross_sections = flux_surface_cross_sections(
        case_condensed["wout"], phi=0.12
    )
    for xyz in cross_sections:
        R = np.hypot(xyz[:, 0], xyz[:, 1])
        Z = xyz[:, 2]
        ax_condensed_interior.plot(
            np.append(R, R[0]),
            np.append(Z, Z[0]),
            color=case_condensed["color"],
            linewidth=0.5,
        )
    ax_condensed_interior.set_aspect(True)
    ax_condensed_interior.set_title(case_condensed["label"])
    ax_condensed_interior.set_axis_off()

    ########################
    # ax_spectral_power
    for case in cases:
        ax_spectral_power.semilogy(
            case["m"], case["power"], color=case["color"], label=case["label"]
        )
    ax_spectral_power.set_xlabel("$m$")
    ax_spectral_power.set_ylabel(r"$\sum_n R_{mn}^2 + Z_{mn}^2$")
    ax_spectral_power.set_title("(c)", loc="left", fontweight="bold")
    # This axis spans ~9-12 decades, and matplotlib's LogLocator flatly
    # refuses to place 2-9 sub-decade minor ticks over that wide a span
    # (regardless of minorticks_on() or an explicit minor locator -- it's
    # a hard cutoff in LogLocator, not a styling issue) -- so "every
    # decade as a tick" (instead of the every-3rd-decade default) is the
    # closest a log axis can get to a finer tick spacing here.
    ax_spectral_power.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax_spectral_power.yaxis.set_major_formatter(FuncFormatter(sparse_decade_label))
    ax_spectral_power.grid(True, which="major", alpha=0.3)
    ax_spectral_power.legend(frameon=False)

    ########################
    # ax_convergence
    for case in cases:
        iterations, fsqt = ftol_history(
            case["vmec_wout"], case["niter_requested"]
        )
        ax_convergence.semilogy(
            iterations, fsqt, color=case["color"], label=case["label"]
        )
    ax_convergence.set_xlabel("Iteration")
    ax_convergence.set_ylabel(r"$\overline{F}^2$")
    ax_convergence.set_title("(d)", loc="left", fontweight="bold")
    ax_convergence.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax_convergence.yaxis.set_major_formatter(FuncFormatter(sparse_decade_label))
    ax_convergence.grid(True, which="major", alpha=0.3)
    ax_convergence.legend(frameon=False)

    # plt.show()
    plt.savefig(
        dpi=600,
        fname="spec_cond_solve_3x2.pdf",
        bbox_inches="tight",
    )
