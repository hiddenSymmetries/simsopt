"""
This module contains the a number of useful functions for using 
the permanent magnets functionality in the SIMSOPT code.
"""
__all__ = ['read_focus_coils', 'coil_optimization', 
           'trace_fieldlines', 'make_qfm', 
           'initialize_coils', 'calculate_on_axis_B',
           'get_FAMUS_dipoles', 
           'make_optimization_plots', 'run_Poincare_plots',
           'make_Bnormal_plots', 'write_pm_optimizer_to_famus',
           'rescale_for_opt', 'initialize_default_kwargs'
           ]

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize
import time


def read_focus_coils(filename):
    """
        Reads in the coils from a FOCUS file. For instance, this is
        used for loading in the MUSE phased TF coils.
    """
    from simsopt.geo import CurveXYZFourier
    from simsopt.field import Current

    ncoils = np.loadtxt(filename, skiprows=1, max_rows=1, dtype=int)
    order = int(np.loadtxt(filename, skiprows=8, max_rows=1, dtype=int))
    coilcurrents = np.zeros(ncoils)
    xc = np.zeros((ncoils, order + 1))
    xs = np.zeros((ncoils, order + 1))
    yc = np.zeros((ncoils, order + 1))
    ys = np.zeros((ncoils, order + 1))
    zc = np.zeros((ncoils, order + 1))
    zs = np.zeros((ncoils, order + 1))
    # load in coil currents and fourier representations of (x, y, z)
    for i in range(ncoils):
        coilcurrents[i] = np.loadtxt(filename, skiprows=6 + 14 * i, max_rows=1, usecols=1)
        xc[i, :] = np.loadtxt(filename, skiprows=10 + 14 * i, max_rows=1, usecols=range(order + 1))
        xs[i, :] = np.loadtxt(filename, skiprows=11 + 14 * i, max_rows=1, usecols=range(order + 1))
        yc[i, :] = np.loadtxt(filename, skiprows=12 + 14 * i, max_rows=1, usecols=range(order + 1))
        ys[i, :] = np.loadtxt(filename, skiprows=13 + 14 * i, max_rows=1, usecols=range(order + 1))
        zc[i, :] = np.loadtxt(filename, skiprows=14 + 14 * i, max_rows=1, usecols=range(order + 1))
        zs[i, :] = np.loadtxt(filename, skiprows=15 + 14 * i, max_rows=1, usecols=range(order + 1))

    # CurveXYZFourier wants data in order sin_x, cos_x, sin_y, cos_y, ...
    coil_data = np.zeros((order + 1, ncoils * 6))
    for i in range(ncoils):
        coil_data[:, i * 6 + 0] = xs[i, :]
        coil_data[:, i * 6 + 1] = xc[i, :]
        coil_data[:, i * 6 + 2] = ys[i, :]
        coil_data[:, i * 6 + 3] = yc[i, :]
        coil_data[:, i * 6 + 4] = zs[i, :]
        coil_data[:, i * 6 + 5] = zc[i, :]

    # Set the degrees of freedom in the coil objects
    base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
    ppp = 20
    coils = [CurveXYZFourier(order*ppp, order) for i in range(ncoils)]
    for ic in range(ncoils):
        dofs = coils[ic].dofs_matrix
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, min(order, coil_data.shape[0]-1)):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].local_x = np.concatenate(dofs)
    return coils, base_currents, ncoils


def coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, config_flag):
    """
        Optimize the coils for the QA, QH, or other configurations.
    """

    from simsopt.geo import CurveLength, CurveCurveDistance, \
        MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
    from simsopt.objectives import QuadraticPenalty
    from simsopt.geo import curves_to_vtk
    from simsopt.objectives import SquaredFlux

    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    ncoils = len(base_curves)

    if 'QH' in config_flag:
        # Weight on the curve lengths in the objective function:
        LENGTH_WEIGHT = 1e-1

        # Threshold and weight for the coil-to-coil distance penalty in the objective function:
        CC_THRESHOLD = 0.5
        CC_WEIGHT = 1e3

        # Threshold and weight for the coil-to-surface distance penalty in the objective function:
        CS_THRESHOLD = 0.3
        CS_WEIGHT = 10

        # Threshold and weight for the curvature penalty in the objective function:
        CURVATURE_THRESHOLD = 1.
        CURVATURE_WEIGHT = 1e-6

        # Threshold and weight for the mean squared curvature penalty in the objective function:
        MSC_THRESHOLD = 1
        MSC_WEIGHT = 1e-6
    else:
        # Weight on the curve lengths in the objective function:
        LENGTH_WEIGHT = 1e-4

        # Threshold and weight for the coil-to-coil distance penalty in the objective function:
        CC_THRESHOLD = 0.1
        CC_WEIGHT = 1e-1

        # Threshold and weight for the coil-to-surface distance penalty in the objective function:
        CS_THRESHOLD = 0.1
        CS_WEIGHT = 1e-2

        # Threshold and weight for the curvature penalty in the objective function:
        CURVATURE_THRESHOLD = 0.1
        CURVATURE_WEIGHT = 1e-9

        # Threshold and weight for the mean squared curvature penalty in the objective function:
        MSC_THRESHOLD = 0.1
        MSC_WEIGHT = 1e-9

    MAXITER = 500  # number of iterations for minimize

    # Define the objective function:
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    # Form the total objective function.
    JF = Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        + CC_WEIGHT * Jccdist \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)

    def fun(dofs):
        """ Function for coil optimization grabbed from stage_two_optimization.py """
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad

    print("""
    ################################################################################
    ### Perform a Taylor test ######################################################
    ################################################################################
    """)
    f = fun
    dofs = JF.x
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)

    J0, dJ0 = f(dofs)
    dJh = sum(dJ0 * h)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        J1, _ = f(dofs + eps*h)
        J2, _ = f(dofs - eps*h)
        print("err", (J1-J2)/(2*eps) - dJh)

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    curves_to_vtk(curves, OUT_DIR + f"curves_opt")
    bs.set_points(s.gamma().reshape((-1, 3)))
    return s, bs


def trace_fieldlines(bfield, label, config, s, comm, OUT_DIR):
    """
        Make Poincare plots on a surface as in the trace_fieldlines
        example in the examples/1_Simple/ directory.
    """
    from simsopt.field.tracing import particles_to_vtk, compute_fieldlines, \
        LevelsetStoppingCriterion, plot_poincare_data, \
        IterationStoppingCriterion, SurfaceClassifier

    t1 = time.time()

    # set fieldline tracer parameters
    nfieldlines = 20
    tmax_fl = 60000

    # Different configurations have different cross-sections
    if 'muse' in config:
        R0 = np.linspace(0.32, 0.34, nfieldlines)
    elif 'qa' in config:
        R0 = np.linspace(0.5, 1.0, nfieldlines)
    elif 'qh' in config:
        R0 = np.linspace(0.5, 1.3, nfieldlines)
    elif config == 'ncsx':
        R0 = np.linspace(1.0, 1.75, nfieldlines)
    elif config == 'QH':
        R0 = np.linspace(12.0, 18.0, nfieldlines)
    else:
        raise NotImplementedError(
            'The configuration flag indicates a plasma equilibrium '
            'for which the fieldline tracer does not have default '
            'parameters for.'
        )
    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]

    # compute the fieldlines from the initial locations specified above
    sc_fieldline = SurfaceClassifier(s, h=0.05, p=2)
    sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm,
        phis=phis,  # stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        stopping_criteria=[IterationStoppingCriterion(2000000)])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)

    # make the poincare plots
    if comm is None or comm.rank == 0:
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=400, xlims=(0.225, 0.375), ylims=(-0.075, 0.075), surf=s)


def make_qfm(s, Bfield):
    """
        Given some Bfield generated by dipoles AND a set of TF coils, 
        compute a quadratic flux-minimizing surface (QFMS)
        for the total field configuration on the surface s.
    """
    from simsopt.geo.qfmsurface import QfmSurface
    from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume

    # weight for the optimization
    constraint_weight = 1e0

    # First optimize at fixed volume
    qfm = QfmResidual(s, Bfield)
    qfm.J()

    s.change_resolution(16, 16)
    vol = Volume(s)
    vol_target = vol.J()
    qfm_surface = QfmSurface(Bfield, s, vol, vol_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=500,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # repeat the optimization for further convergence
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=2000,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    return qfm_surface


def initialize_coils(config_flag, TEST_DIR, OUT_DIR, s):
    """
        Initializes coils for each of the target configurations that are
        used for permanent magnet optimization.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk

    if 'muse' in config_flag:
        # Load in pre-optimized coils
        coils_filename = TEST_DIR / 'muse_tf_coils.focus'
        base_curves, base_currents, ncoils = read_focus_coils(coils_filename)
        coils = []
        for i in range(ncoils):
            coils.append(Coil(base_curves[i], base_currents[i]))
        base_currents[0].fix_all()

        # fix all the coil shapes
        for i in range(ncoils):
            base_curves[i].fix_all()
    elif config_flag == 'qh':
        # generate planar TF coils
        ncoils = 4
        R0 = 1.0
        R1 = 0.75
        order = 5

        # qh needs to be scaled to 0.1 T on-axis magnetic field strength
        from simsopt.mhd.vmec import Vmec
        vmec_file = 'wout_LandremanPaul_QH_variant.nc'
        total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 8.75
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

        # fix all the coil shapes so only the currents are optimized
        for i in range(ncoils):
            base_curves[i].fix_all()
    elif config_flag == 'qa':
        # generate planar TF coils
        ncoils = 8
        R0 = 1.0
        R1 = 0.65
        order = 5

        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        from simsopt.mhd.vmec import Vmec
        vmec_file = 'wout_LandremanPaul2021_QA_lowres.nc'
        total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 7.2
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
    curves_to_vtk(curves, OUT_DIR + "curves_init")
    return base_curves, curves, coils


def calculate_on_axis_B(bs, s):
    """
        Check the average, approximate, on-axis
        magnetic field strength to make sure the
        configuration is scaled correctly.
    """
    nphi = len(s.quadpoints_phi)
    bspoints = np.zeros((nphi, 3))

    # rescale phi from [0, 1) to [0, 2 * pi)
    phi = s.quadpoints_phi * 2 * np.pi

    R0 = s.get_rc(0, 0)
    for i in range(nphi):
        bspoints[i] = np.array([R0 * np.cos(phi[i]),
                                R0 * np.sin(phi[i]),
                                0.0]
                               )
    bs.set_points(bspoints)
    B0 = np.linalg.norm(bs.B(), axis=-1)
    B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
    surface_area = s.area()
    bnormalization = B0avg * surface_area
    print("Bmag at R = ", R0, ", Z = 0: ", B0)
    print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg)
    return B0avg


def get_FAMUS_dipoles(famus_filename):
    """
        Reads in and makes vtk plots for a FAMUS grid and
        solution. Used for the MUSE and NCSX examples.
    """
    # FAMUS files are for the half-period surface
    ox, oy, oz, Ic, m0, p, mp, mt = np.loadtxt(
        famus_filename, skiprows=3,
        usecols=[3, 4, 5, 6, 7, 8, 10, 11],
        delimiter=',', unpack=True
    )

    print('Number of FAMUS dipoles = ', len(ox))

    # Ic = 0 indices are used to denote grid locations
    # that should be removed because the ports go there
    nonzero_inds = (Ic == 1.0)
    ox = ox[nonzero_inds]
    oy = oy[nonzero_inds]
    oz = oz[nonzero_inds]
    m0 = m0[nonzero_inds]
    p = p[nonzero_inds]
    mp = mp[nonzero_inds]
    mt = mt[nonzero_inds]
    print('Number of FAMUS dipoles (with ports) = ', len(ox))

    phi = np.arctan2(oy, ox)

    # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
    momentq = np.loadtxt(famus_filename, skiprows=1, max_rows=1, usecols=[1])
    rho = p ** momentq

    mm = rho * m0

    # Convert from spherical to cartesian vectors
    mx = mm * np.sin(mt) * np.cos(mp)
    my = mm * np.sin(mt) * np.sin(mp)
    mz = mm * np.cos(mt)
    m_FAMUS = np.ravel((np.array([mx, my, mz]).T))
    return m_FAMUS, m0


def make_optimization_plots(RS_history, m_history, m_proxy_history, pm_opt, OUT_DIR):
    """
        Make line plots of the algorithm convergence and make histograms
        of m, m_proxy, and if available, the FAMUS solution.
    """
    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.plot(RS_history)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(OUT_DIR + 'RS_objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    m0_abs = np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima
    mproxy_abs = np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima

    # get FAMUS rho values for making comparison histograms
    if pm_opt.famus_filename is not None:
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
    plt.savefig(OUT_DIR + 'm_histograms.png')

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
            ani = animation.FuncAnimation(
                fig, prepare_animation(bar_container),
                range(0, m_history.shape[0], 2),
                repeat=False, blit=True
            )
            # ani.save(OUT_DIR + 'm_history' + str(i) + '.mp4')


def run_Poincare_plots(s_plot, bs, b_dipole, config_flag, comm, filename_poincare, OUT_DIR):
    """
        Wrapper function for making Poincare plots.
    """
    from simsopt.field.magneticfieldclasses import InterpolatedField
    from simsopt.objectives import SquaredFlux

    n = 64
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    r_margin = 0.05
    rrange = (np.min(rs) - r_margin, np.max(rs) + r_margin, n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 2
    t1 = time.time()
    nphi = len(s_plot.quadpoints_phi)
    ntheta = len(s_plot.quadpoints_theta)
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    Bnormal_dipole = np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    f_B = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
    print('Bnormal = ', Bnormal)
    make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_pre_poincare_check")
    print('Bnormal dipoles = ', Bnormal_dipole)
    make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "dipole_pre_poincare_check")
    print('Bnormal total = ', Bnormal + Bnormal_dipole)
    make_Bnormal_plots(bs + b_dipole, s_plot, OUT_DIR, "total_pre_poincare_check")
    print('f_B = ', f_B)

    bsh = InterpolatedField(
        bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    trace_fieldlines(bsh, 'bsh_PMs_' + filename_poincare, config_flag, s_plot, comm, OUT_DIR)


def make_Bnormal_plots(bs, s_plot, OUT_DIR, bs_filename):
    """
        Plot Bnormal on plasma surface from a MagneticField object.
        Do this quite a bit in the permanent magnet optimization
        and initialization so this is a wrapper function to reduce
        the amount of code.
    """
    nphi = len(s_plot.quadpoints_phi)
    ntheta = len(s_plot.quadpoints_theta)
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + bs_filename, extra_data=pointData)


def write_pm_optimizer_to_famus(OUT_DIR, pm_opt):
    """
        Takes a PermanentMagnetGrid object and saves the geometry
        and optimization solution into a FAMUS input file.
    """
    ndipoles = pm_opt.ndipoles
    m = pm_opt.m.reshape(ndipoles, 3)
    ox = pm_opt.dipole_grid_xyz[:, 0]
    oy = pm_opt.dipole_grid_xyz[:, 1]
    oz = pm_opt.dipole_grid_xyz[:, 2]

    # Transform the solution vector to the Cartesian basis if necessary
    if pm_opt.coordinate_flag == 'cartesian':
        mx = m[:, 0]
        my = m[:, 1]
        mz = m[:, 2]
    elif pm_opt.coordinate_flag == 'cylindrical':
        cos_ophi = np.cos(pm_opt.pm_phi)
        sin_ophi = np.sin(pm_opt.pm_phi)
        mx = m[:, 0] * cos_ophi - m[:, 1] * sin_ophi
        my = m[:, 0] * sin_ophi + m[:, 1] * cos_ophi
        mz = m[:, 2]
    elif pm_opt.coordinate_flag == 'toroidal':
        ormajor = np.sqrt(ox**2 + oy**2)
        otheta = np.arctan2(oz, ormajor - pm_opt.R0)
        cos_ophi = np.cos(pm_opt.pm_phi)
        sin_ophi = np.sin(pm_opt.pm_phi)
        cos_otheta = np.cos(otheta)
        sin_otheta = np.sin(otheta)
        mx = m[:, 0] * cos_ophi * cos_otheta \
            - m[:, 1] * sin_ophi              \
            - m[:, 2] * cos_ophi * sin_otheta
        my = m[:, 0] * sin_ophi * cos_otheta \
            + m[:, 1] * cos_ophi              \
            - m[:, 2] * sin_ophi * sin_otheta
        mz = m[:, 0] * sin_otheta \
            + m[:, 2] * cos_otheta

    m0 = pm_opt.m_maxima
    pho = np.sqrt(np.sum(m ** 2, axis=-1)) / m0
    Lc = 0

    mp = np.arctan2(my, mx)
    mt = np.arctan2(np.sqrt(mx ** 2 + my ** 2), mz)
    coilname = ["pm_{:010d}".format(i) for i in range(1, ndipoles + 1)]
    Ic = 1
    # symmetry = 2 for stellarator symmetry
    symmetry = int(pm_opt.plasma_boundary.stellsym) + 1
    filename = OUT_DIR + 'SIMSOPT_dipole_solution.focus'

    with open(filename, "w") as wfile:
        wfile.write(" # Total number of dipoles,  momentq \n")
        wfile.write(
            "{:6d}  {:4d}\n".format(
                ndipoles, 1
            )
        )
        wfile.write(
            "#coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt \n"
        )
        for i in range(ndipoles):
            wfile.write(
                " 2, {:1d}, {:}, {:15.8E}, {:15.8E}, {:15.8E}, {:2d}, {:15.8E},"
                "{:15.8E}, {:2d}, {:15.8E}, {:15.8E} \n".format(
                    symmetry,
                    coilname[i],
                    ox[i],
                    oy[i],
                    oz[i],
                    Ic,
                    m0[i],
                    pho[i],
                    Lc,
                    mp[i],
                    mt[i],
                )
            )
    return


def rescale_for_opt(pm_opt, reg_l0, reg_l1, reg_l2, nu):
    """
        Scale regularizers to the largest scale of ATA (~1e-6)
        to avoid regularization >> ||Am - b|| ** 2 term in the optimization.
        The prox operator uses reg_l0 * nu for the threshold so normalization
        below allows reg_l0 and reg_l1 values to be exactly the thresholds
        used in calculation of the prox. Then add contributions to ATA and
        ATb coming from extra loss terms such as L2 regularization and
        relax-and-split.
    """

    print('L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2))

    if reg_l0 < 0 or reg_l0 > 1:
        raise ValueError(
            'L0 regularization must be between 0 and 1. This '
            'value is automatically scaled to the largest of the '
            'dipole maximum values, so reg_l0 = 1 should basically '
            'truncate all the dipoles to zero. '
        )

    # Rescale L0 and L1 so that the values used for thresholding
    # are only parametrized by the values of reg_l0 and reg_l1
    reg_l0 = reg_l0 / (2 * nu)
    reg_l1 = reg_l1 / nu

    # may want to rescale nu, otherwise just scan this value
    # nu = nu / pm_opt.ATA_scale

    # Do not rescale L2 term for now.
    reg_l2 = reg_l2

    # Update algorithm step size if we have extra smooth, convex loss terms
    pm_opt.ATA_scale += 2 * reg_l2 + 1.0 / nu

    return reg_l0, reg_l1, reg_l2, nu


def initialize_default_kwargs(algorithm='RS'):
    """
        Keywords to the permanent magnet optimizers are now passed
        by the kwargs dictionary. Default dictionaries are initialized
        here.
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
