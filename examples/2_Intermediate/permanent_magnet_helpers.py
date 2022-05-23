import numpy as np
from pyevtk.hl import pointsToVTK
from scipy.optimize import minimize
from scipy.io import netcdf
from simsopt.field.coil import Current
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk

# Reads in the coils for the MUSE phased TF coils


def read_focus_coils(filename):
    ncoils = np.loadtxt(filename, skiprows=1, max_rows=1, dtype=int)
    order = np.loadtxt(filename, skiprows=8, max_rows=1, dtype=int)
    coilcurrents = np.zeros(ncoils)
    xc = np.zeros((ncoils, order + 1))
    xs = np.zeros((ncoils, order + 1))
    yc = np.zeros((ncoils, order + 1))
    ys = np.zeros((ncoils, order + 1))
    zc = np.zeros((ncoils, order + 1))
    zs = np.zeros((ncoils, order + 1))
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
    # coilcurrents = coilcurrents * 1e3  # rescale from kA to A
    base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
    ppp = 20
    coils = [CurveXYZFourier(order*ppp, order) for i in range(ncoils)]
    for ic in range(ncoils):
        dofs = coils[ic].dofs
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

# Optimize the coils for the QA or QH configurations


def coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, config_flag):

    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    ncoils = len(base_curves)

    if 'QH' in config_flag:
        # Weight on the curve lengths in the objective function:
        LENGTH_WEIGHT = 1e-2

        # Threshold and weight for the coil-to-coil distance penalty in the objective function:
        CC_THRESHOLD = 0.1
        CC_WEIGHT = 10

        # Threshold and weight for the coil-to-surface distance penalty in the objective function:
        CS_THRESHOLD = 0.3
        CS_WEIGHT = 10

        # Threshold and weight for the curvature penalty in the objective function:
        CURVATURE_THRESHOLD = 1.
        CURVATURE_WEIGHT = 1e-6

        # Threshold and weight for the mean squared curvature penalty in the objective function:
        MSC_THRESHOLD = 1
        MSC_WEIGHT = 1e-6

    if 'qa' in config_flag or 'qh' in config_flag:
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

    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    JF = Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        + CC_WEIGHT * Jccdist \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)

    def fun(dofs):
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
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((2 * nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
    bs.set_points(s.gamma().reshape((-1, 3)))
    return s, bs

# Load NCSX Bn data on the plasma surface


def load_ncsx_coil_data(s, coil_name):
    n, m, bmnc, bmns = np.loadtxt('../../tests/test_files/' + coil_name, skiprows=2, unpack=True)
    phi = s.quadpoints_phi * 2 * np.pi
    theta = s.quadpoints_theta * 2 * np.pi
    nfp = s.nfp
    Bn = np.zeros((len(phi), len(theta)))
    for i in range(len(phi)):
        for j in range(len(theta)):
            for ii in range(len(n)):
                Bn[i, j] += bmnc[ii] * np.cos(- n[ii] * nfp * phi[i] + m[ii] * theta[j]) + bmns[ii] * np.sin(- n[ii] * nfp * phi[i] + m[ii] * theta[j])
    return Bn


def read_regcoil_pm(filename, surface_filename, OUT_DIR):
    f = netcdf.netcdf_file(filename, 'r', mmap=False)
    nfp = f.variables['nfp'][()]
    ntheta_plasma = f.variables['ntheta_plasma'][()]
    ntheta_coil = f.variables['ntheta_coil'][()]
    nzeta_plasma = f.variables['nzeta_plasma'][()]
    nzeta_coil = f.variables['nzeta_coil'][()]
    nzetal_plasma = f.variables['nzetal_plasma'][()]
    nzetal_coil = f.variables['nzetal_coil'][()]
    print(ntheta_plasma, nzeta_plasma, nzetal_plasma)
    print(ntheta_coil, nzeta_coil, nzetal_coil)
    theta_plasma = f.variables['theta_plasma'][()]
    theta_coil = f.variables['theta_coil'][()]
    zeta_plasma = f.variables['zeta_plasma'][()]
    zeta_coil = f.variables['zeta_coil'][()]
    zetal_plasma = f.variables['zetal_plasma'][()]
    zetal_coil = f.variables['zetal_coil'][()]
    r_plasma = f.variables['r_plasma'][()]
    r_coil = f.variables['r_coil'][()]
    chi2_B = f.variables['chi2_B'][()]
    print('chi2B = ', chi2_B)
    chi2_M = f.variables['chi2_M'][()]
    max_M = f.variables['max_M'][()]
    min_M = f.variables['min_M'][()]
    abs_M = f.variables['abs_M'][()]
    print(f.variables['volume_magnetization'][()])
    rmnc_outer = f.variables['rmnc_outer'][()]
    rmns_outer = f.variables['rmns_outer'][()]
    zmnc_outer = f.variables['zmnc_outer'][()]
    zmns_outer = f.variables['zmns_outer'][()]
    mnmax_coil = f.variables['mnmax_coil'][()]
    xm_coil = f.variables['xm_coil'][()]
    xn_coil = f.variables['xn_coil'][()]
    ports_weight = f.variables['ports_weight'][()]
    d = f.variables['d'][()]
    ns_magnetization = f.variables['ns_magnetization'][()]
    Bnormal_from_TF_and_plasma_current = f.variables['Bnormal_from_TF_and_plasma_current'][()]
    max_Bnormal = f.variables['max_Bnormal'][()]
    print('REGCOIL_PM max Bnormal = ', max_Bnormal)
    Bnormal_total = f.variables['Bnormal_total'][()]
    Mvec = f.variables['magnetization_vector'][()]
    r_coil = f.variables['r_coil'][()]
    print(Bnormal_from_TF_and_plasma_current.shape)
    quadpoints_phi = np.linspace(0, 1, nzetal_plasma, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta_plasma, endpoint=True)
    s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    print(np.sum(np.sum(Bnormal_total ** 2, axis=-1), axis=-1))
    Bnormal_TF = np.zeros((nzetal_plasma, ntheta_plasma))
    Bnormal = np.zeros((nzetal_plasma, ntheta_plasma))
    Bnormal_total = Bnormal_total[-1, :, :]
    for i in range(nfp):
        Bnormal_TF[i * nzeta_plasma:(i + 1) * nzeta_plasma, :] = Bnormal_from_TF_and_plasma_current
        Bnormal[i * nzeta_plasma:(i + 1) * nzeta_plasma, :] = Bnormal_total
    pointData = {"B_N": Bnormal_TF[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "regcoil_pm_ncsx_TF", extra_data=pointData)
    pointData = {"B_N": Bnormal[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "regcoil_pm_ncsx_total", extra_data=pointData)
    # Plot the dipole grid
    Mvec = Mvec[-1, :, :, :, :].reshape(3, nzeta_coil, ntheta_coil)
    Mvec_total = np.zeros((nzetal_coil, ntheta_coil, 3))
    # convert Mvec to cartesian
    for i in range(ntheta_coil):
        Mvec_x = Mvec[0, :, i] * np.cos(zeta_coil) - Mvec[1, :, i] * np.sin(zeta_coil)
        Mvec_y = Mvec[0, :, i] * np.sin(zeta_coil) + Mvec[1, :, i] * np.cos(zeta_coil)
        for fp in range(nfp):
            phi0 = (2 * np.pi / nfp) * fp
            Mvec_total[fp * nzeta_coil:(fp + 1) * nzeta_coil, i, 0] = Mvec_x * np.cos(phi0) - Mvec_y * np.sin(phi0)
            Mvec_total[fp * nzeta_coil:(fp + 1) * nzeta_coil, i, 1] = Mvec_x * np.sin(phi0) + Mvec_y * np.cos(phi0)

    Mvec_total = Mvec_total.reshape(nzetal_coil * ntheta_coil, 3)
    m_maxima = np.ravel(abs_M[-1, :, :, :])
    m_maxima = np.ravel(np.array([m_maxima, m_maxima, m_maxima]))
    ox = np.ascontiguousarray(r_coil[:, :, 0])  # * np.cos(r_coil[:, :, 1]))
    oy = np.ascontiguousarray(r_coil[:, :, 1])  # * np.sin(r_coil[:, :, 1]))
    oz = np.ascontiguousarray(r_coil[:, :, 2])

    # For FP symmetry Mvec rotates

    mx = np.ascontiguousarray(Mvec_total[:, 0])
    my = np.ascontiguousarray(Mvec_total[:, 1])
    mz = np.ascontiguousarray(Mvec_total[:, 2])
    mmag = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    mx_normalized = np.ascontiguousarray(mx / m_maxima)
    my_normalized = np.ascontiguousarray(my / m_maxima)
    mz_normalized = np.ascontiguousarray(mz / m_maxima)
    print("write VTK as points")
    data = {"m": (mx, my, mz), "m_normalized": (mx_normalized, my_normalized, mz_normalized)}
    pointsToVTK(
        OUT_DIR + "Dipole_Fields_REGCOIL_PM", ox, oy, oz, data=data
    )


def trace_fieldlines(bfield, label, config): 
    t1 = time.time()
    if config == 'muse':
        R0 = np.linspace(0.2, 0.4, nfieldlines)
    elif 'qa' in config: 
        R0 = np.linspace(0.5, 1.0, nfieldlines)
    elif 'qh' in config:
        R0 = np.linspace(0.5, 1.3, nfieldlines)
    elif config == 'ncsx': 
        R0 = np.linspace(1.0, 1.75, nfieldlines)
    elif config == 'QH':
        R0 = np.linspace(12.0, 18.0, nfieldlines)

    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-15, comm=comm,
        phis=phis, stopping_criteria=[IterationStoppingCriterion(200000)])
    t2 = time.time()
    # print(fieldlines_phi_hits, np.shape(fieldlines_phi_hits))
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm is None or comm.rank == 0:
        # particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)


def make_qfm(s, Bfield, Bfield_tf):
    constraint_weight = 1e0

    # First optimize at fixed volume

    qfm = QfmResidual(s, Bfield)
    qfm.J()

    vol = Volume(s)
    vol_target = vol.J()

    qfm_surface = QfmSurface(Bfield, s, vol, vol_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Now optimize at fixed toroidal flux
    tf = ToroidalFlux(s, Bfield_tf)
    tf_target = tf.J()

    qfm_surface = QfmSurface(Bfield, s, tf, tf_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                             constraint_weight=constraint_weight)
    print(f"||tf constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Check that volume is not changed
    print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

    # Now optimize at fixed area

    ar = Area(s)
    ar_target = ar.J()

    qfm_surface = QfmSurface(Bfield, s, ar, ar_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000, constraint_weight=constraint_weight)
    print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Check that volume is not changed
    print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")
    # s.plot()
    return qfm_surface.surface 


