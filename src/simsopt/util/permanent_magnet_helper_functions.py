"""
This module contains the a number of useful functions for using 
the permanent magnets functionality in the SIMSOPT code.
"""
__all__ = [
           'initialize_coils_for_pm_optimization',
           'make_optimization_plots', 'run_Poincare_plots_with_permanent_magnets',
           'initialize_default_kwargs'
           ]

import numpy as np
from pathlib import Path

def initialize_coils_for_pm_optimization(config_flag, TEST_DIR, s, out_dir=''):
    """
    Initializes coils for each of the target configurations that are
    used for permanent magnet optimization. The total current is chosen
    to achieve a desired average field strength along the major radius.

    Args:
        config_flag: String denoting the stellarator configuration 
          being initialized.
        TEST_DIR: String denoting where to find the input files.
        out_dir: Path or string for the output directory for saved files.
        s: plasma boundary surface.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk
    from simsopt.util.coil_optimization_helper_functions import read_focus_coils

    out_dir = Path(out_dir)
    if 'muse' in config_flag:
        # Load in pre-optimized coils
        coils_filename = TEST_DIR / 'muse_tf_coils.focus'

        # Slight difference from older MUSE initialization. 
        # Here we fix the total current summed over all coils and 
        # older MUSE opt. fixed the first coil current. 
        base_curves, base_currents0, ncoils = read_focus_coils(coils_filename)
        total_current = np.sum([curr.get_value() for curr in base_currents0])
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = []
        for i in range(ncoils):
            coils.append(Coil(base_curves[i], base_currents[i]))

    elif config_flag == 'qh':
        # generate planar TF coils
        ncoils = 4
        R0 = s.get_rc(0, 0)
        R1 = s.get_rc(1, 0) * 4
        order = 2

        # QH reactor scale needs to be 5.7 T average magnetic field strength
        total_current = 48712698  # Amperes
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True,
                                                   R0=R0, R1=R1, order=order, numquadpoints=64)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    elif config_flag == 'qa':
        # generate planar TF coils
        ncoils = 8
        R0 = 1.0
        R1 = 0.65
        order = 5

        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        total_current = 187500
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    
    # fix all the coil shapes so only the currents are optimized
    for i in range(ncoils):
        base_curves[i].fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir / "curves_init")
    return base_curves, curves, coils


def make_optimization_plots(RS_history, m_history, m_proxy_history, pm_opt, out_dir=''):
    """
    Make line plots of the algorithm convergence and make histograms
    of m, m_proxy, and if available, the FAMUS solution.

    Args:
        RS_history: List of the relax-and-split optimization progress.
        m_history: List of the permanent magnet solutions generated
          as the relax-and-split optimization progresses.
        m_proxy_history: Same but for the 'proxy' variable in the
          relax-and-split optimization.
        pm_opt: PermanentMagnetGrid class object that was optimized.
        out_dir: Path or string for the output directory for saved files.
    """
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    
    out_dir = Path(out_dir)

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.plot(RS_history)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(out_dir / 'RS_objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    m0_abs = np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima
    mproxy_abs = np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima

    # get FAMUS rho values for making comparison histograms
    if hasattr(pm_opt, 'famus_filename'):
        m0, p = np.loadtxt(
            pm_opt.famus_filename, skiprows=3,
            usecols=[7, 8],
            delimiter=',', unpack=True
        )
        # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
        momentq = np.loadtxt(pm_opt.famus_filename, skiprows=1, max_rows=1, usecols=[1])
        rho = p ** momentq
        rho = rho[pm_opt.Ic_inds]
        x_multi = [m0_abs, mproxy_abs, abs(rho)]
    else:
        x_multi = [m0_abs, mproxy_abs]
        rho = None
    plt.hist(x_multi, bins=np.linspace(0, 1, 40), log=True, histtype='bar')
    plt.grid(True)
    plt.legend(['m', 'w', 'FAMUS'])
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(out_dir / 'm_histograms.png')

    # If relax-and-split was used with l0 norm,
    # make a nice histogram of the algorithm progress
    if len(RS_history) != 0:

        m_history = np.array(m_history)
        m_history = m_history.reshape(m_history.shape[0] * m_history.shape[1], pm_opt.ndipoles, 3)
        m_proxy_history = np.array(m_proxy_history).reshape(m_history.shape[0], pm_opt.ndipoles, 3)

        for i, datum in enumerate([m_history, m_proxy_history]):
            # Code from https://matplotlib.org/stable/gallery/animation/animated_histogram.html
            def prepare_animation(bar_container):
                def animate(frame_number):
                    plt.title(frame_number)
                    data = np.sqrt(np.sum(datum[frame_number, :] ** 2, axis=-1)) / pm_opt.m_maxima
                    n, _ = np.histogram(data, np.linspace(0, 1, 40))
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    return bar_container.patches
                return animate

            # make histogram animation of the dipoles at each relax-and-split save
            fig, ax = plt.subplots()
            data = np.sqrt(np.sum(datum[0, :] ** 2, axis=-1)) / pm_opt.m_maxima
            if rho is not None:
                ax.hist(abs(rho), bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='g')
            _, _, bar_container = ax.hist(data, bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='r')
            ax.set_ylim(top=1e5)  # set safe limit to ensure that all data is visible.
            plt.grid(True)
            if rho is not None:
                plt.legend(['FAMUS', np.array([r'm$^*$', r'w$^*$'])[i]])
            else:
                plt.legend([np.array([r'm$^*$', r'w$^*$'])[i]])

            plt.xlabel('Normalized magnitudes')
            plt.ylabel('Number of dipoles')
            animation.FuncAnimation(
                fig, prepare_animation(bar_container),
                range(0, m_history.shape[0], 2),
                repeat=False, blit=True
            )


def run_Poincare_plots_with_permanent_magnets(s_plot, bs, b_dipole, comm, filename_poincare, out_dir=''):
    """
    Wrapper function for making Poincare plots.

    Args:
        s_plot: plasma boundary surface with range = 'full torus'.
        bs: MagneticField class object.
        b_dipole: DipoleField class object.
        comm: MPI COMM_WORLD object for using MPI for tracing.
        filename_poincare: Filename for the output poincare picture.
        out_dir: Path or string for the output directory for saved files.
    """
    from simsopt.field.magneticfieldclasses import InterpolatedField
    from simsopt.util.coil_optimization_helper_functions import trace_fieldlines
    # from simsopt.objectives import SquaredFlux

    out_dir = Path(out_dir)

    n = 32
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    r_margin = 0.05
    rrange = (np.min(rs) - r_margin, np.max(rs) + r_margin, n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 2  # 2 is sufficient sometimes
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))

    bsh = InterpolatedField(
        bs + b_dipole, degree, rrange, phirange, zrange, 
        True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    trace_fieldlines(bsh, 'bsh_PMs_' + filename_poincare, s_plot, comm, out_dir)

def initialize_default_kwargs(algorithm='RS'):
    """
    Keywords to the permanent magnet optimizers are now passed
    by the kwargs dictionary. Default dictionaries are initialized
    here.

    Args:
        algorithm: String denoting which algorithm is being used
          for permanent magnet optimization. Options are 'RS'
          (relax and split), 'GPMO' (greedy placement), 
          'backtracking' (GPMO with backtracking), 
          'multi' (GPMO, placing multiple magnets each iteration),
          'ArbVec' (GPMO with arbitrary dipole orientiations),
          'ArbVec_backtracking' (GPMO with backtracking and 
          arbitrary dipole orientations).

    Returns:
        kwargs: Dictionary of keywords to pass to a permanent magnet 
          optimizer.
    """

    kwargs = {}
    kwargs['verbose'] = True   # print out errors every few iterations
    if algorithm == 'RS':
        kwargs['nu'] = 1e100  # Strength of the "relaxation" part of relax-and-split
        kwargs['max_iter'] = 100  # Number of iterations to take in a convex step
        kwargs['reg_l0'] = 0.0
        kwargs['reg_l1'] = 0.0
        kwargs['alpha'] = 0.0
        kwargs['min_fb'] = 0.0
        kwargs['epsilon'] = 1e-3
        kwargs['epsilon_RS'] = 1e-3
        kwargs['max_iter_RS'] = 2  # Number of total iterations of the relax-and-split algorithm
        kwargs['reg_l2'] = 0.0
    elif 'GPMO' in algorithm or 'ArbVec' in algorithm:
        kwargs['K'] = 1000
        kwargs["reg_l2"] = 0.0
        kwargs['nhistory'] = 500  # K > nhistory and nhistory must be divisor of K
    return kwargs
