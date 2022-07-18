import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pyevtk.hl import pointsToVTK
from scipy.optimize import minimize
from scipy.io import netcdf
import time


def sanitised_input(prompt, type_=None, min_=None, max_=None, range_=None, default_=None):
    """ 
        Copied function from https://stackoverflow.com/questions/23294658/asking-the-user-for-input-until-they-give-a-valid-response for asking user for 
        repeated input values. 
    """
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min_ must be less than or equal to max_.")
    while True:
        ui = input(prompt + '(Press enter for default value = ' + str(default_) + ') ')
        if ui == '':
            return default_
        elif type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        if max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))
            else:
                template = "Input must be {0}."
                if len(range_) == 1:
                    print(template.format(*range_))
                else:
                    expected = " or ".join((
                        ", ".join(str(x) for x in range_[:-1]),
                        str(range_[-1])
                    ))
                    print(template.format(expected))
        else:
            return ui


def read_input():
    """
        Wrapper function for santitised_input function that reads in 
        all the parameters needed for the permanent magnet optimization.
        Provides interactive user inputs if running the code from an
        interactive compute node, otherwise needs command line arguments
        to work correctly from a slurm script.
    """
    if len(sys.argv) < 2:
        raise ValueError(
            'You must specify if the run is interactive or not '
            'with a command line flag of True or False.'
        )
    else:
        interactive = str(sys.argv[1])
        if interactive not in ['True', 'False']:
            raise ValueError(
                'Flag for interactivity must be True or False'
            )
        interactive = (interactive == 'True')

    if interactive:
        config_flag = sanitised_input("Enter the plasma configuration name: ", str, range_=('qa', 'qa_nonplanar', 'QH', 'qh', 'qh_nonplanar', 'muse', 'muse_famus', 'ncsx'), default_='muse_famus')
        res_flag = sanitised_input("Enter the resolution: ", str, range_=('low', 'medium', 'high'), default_='low')
        run_type = sanitised_input("Initialization, optimization, or post-processing a permanent magnet grid class? ", str, range_=('initialization', 'optimization', 'post-processing'), default_='optimization')
        coordinate_flag = sanitised_input("Coordinate system? ", str, range_=('cartesian', 'cylindrical', 'toroidal'), default_='cartesian')
        if run_type == 'optimization':
            reg_l2 = sanitised_input("L2 Regularization = ", float, 0.0, 1.0, default_=0.0)
            epsilon = sanitised_input("Error tolerance for the convex subproblem = ", float, 0.0, 1.0, default_=1e-2)
            max_iter_MwPGP = sanitised_input("Maximum iterations for the convex subproblem = ", int, 0, 1e5, default_=100)
            min_fb = sanitised_input("Value of f_B metric to end the algorithm on = ", float, 0.0, 1.0, default_=1e-20)
            reg_l0 = sanitised_input("L0 Regularization = ", float, 0.0, 1.0, default_=0.0)
            if np.isclose(reg_l0, 0.0, atol=1e-16):
                nu = 1.0e100
                max_iter_RS = 1
            else:
                nu = sanitised_input("nu (relax-and-split hyperparameter) = ", float, 0.0, 1e100, default_=1e6)
                max_iter_RS = sanitised_input("Maximum number of relax-and-split iterations: ", int, 0, 1e5, default_=10) 
        elif run_type == 'post-processing':
            reg_l2 = sanitised_input("L2 Regularization = ", float, 0.0, 1.0, default_=0.0)
            reg_l0 = sanitised_input("L0 Regularization = ", float, 0.0, 1.0, default_=0.0)
            if np.isclose(reg_l0, 0.0, atol=1e-16):
                nu = 1.0e100
                max_iter_RS = 1
            else:
                nu = sanitised_input("nu (relax-and-split hyperparameter) = ", float, 0.0, 1e100, default_=1e6)
    else:
        if len(sys.argv) < 5:
            print(
                "Error! If the run is interactive, then "
                "you must specify at least 4 arguments: "
                "the flag to indicate if the run is interactive, "
                "the configuration flag, resolution flag, and the run type flag. "
            )
            exit(1)
        config_flag = str(sys.argv[2])
        if config_flag not in ['qa', 'qa_nonplanar', 'QH', 'qh', 'qh_nonplanar', 'muse', 'muse_famus', 'ncsx']:
            raise ValueError(
                "Error! The configuration flag must specify one of "
                "the pre-set plasma equilibria: qa, qa_nonplanar, "
                "QH, qh, qh_nonplanar, muse, muse_famus, or ncsx. "
            )
        res_flag = str(sys.argv[3])
        if res_flag not in ['low', 'medium', 'high']:
            raise ValueError(
                "Error! The resolution flag must specify one of "
                "low or high."
            )
        run_type = str(sys.argv[4])
        if run_type not in ['initialization', 'optimization', 'post-processing']:
            raise ValueError(
                "Error! The initialization flag must specify one of "
                "initialization, optimization, or post-processing."
            )
        if run_type == 'optimization':

            # L2 regularization
            if len(sys.argv) >= 6:
                reg_l2 = float(sys.argv[5])
            else:
                reg_l2 = 1e-12

            # Error tolerance for declaring convex problem finished
            if len(sys.argv) >= 7:
                epsilon = float(sys.argv[6])
            else:
                epsilon = 1e-2

            # Maximum iterations for solving the convex problem
            if len(sys.argv) >= 8:
                max_iter_MwPGP = int(sys.argv[7])
            else:
                max_iter_MwPGP = 100

            # Error tolerance for declaring nonconvex problem finished
            if len(sys.argv) >= 9:
                min_fb = float(sys.argv[8])
            else:
                min_fb = 1e-20

            # L0 regularization
            if len(sys.argv) >= 10:
                reg_l0 = float(sys.argv[9])
            else:
                reg_l0 = 0.0  # default is no L0 norm

            # L0 regularization
            if len(sys.argv) >= 11:
                reg_l1 = float(sys.argv[10])
            else:
                reg_l1 = 0.0  # default is no L1 norm
            
            # nu (relax-and-split hyperparameter)
            if len(sys.argv) >= 12:
                nu = float(sys.argv[11])

            # Set to huge value if reg_l0 is zero so it is ignored
            if np.isclose(reg_l0, 0.0, atol=1e-16) and np.isclose(reg_l1, 0.0, atol=1e-16):
                nu = 1e100

            # Maximum iterations for solving the nonconvex problem
            if len(sys.argv) >= 13:
                max_iter_RS = int(sys.argv[12])
            else:
                max_iter_RS = 100

            if len(sys.argv) >= 14:
                coordinate_flag = str(sys.argv[13])
                if coordinate_flag not in ['cartesian', 'cylindrical', 'toroidal']:
                    raise ValueError(
                        "Error! The coordinate flag must specify one of "
                        "cartesian, cylindrical, or toroidal."
                    )
            else:
                coordinate_flag = 'cartesian' 
        elif run_type == 'initialization':
            if len(sys.argv) >= 6:
                coordinate_flag = str(sys.argv[5])
                if coordinate_flag not in ['cartesian', 'cylindrical', 'toroidal']:
                    raise ValueError(
                        "Error! The coordinate flag must specify one of "
                        "cartesian, cylindrical, or toroidal."
                    )
            else:
                coordinate_flag = 'cartesian' 
        elif run_type == 'post-processing':
            # L2 regularization
            if len(sys.argv) >= 6:
                reg_l2 = float(sys.argv[5])
            else:
                reg_l2 = 1e-12

            # L0 regularization
            if len(sys.argv) >= 7:
                reg_l0 = float(sys.argv[6])
            else:
                reg_l0 = 0.0  # default is no L0 norm

            # L1 regularization
            if len(sys.argv) >= 8:
                reg_l1 = float(sys.argv[7])
            else:
                reg_l1 = 0.0  # default is no L1 norm

            # nu (relax-and-split hyperparameter)
            if len(sys.argv) >= 9:
                nu = float(sys.argv[8])

            # Set to huge value if reg_l0 is zero so it is ignored
            if np.isclose(reg_l0, 0.0, atol=1e-16) and np.isclose(reg_l1, 0.0, atol=1e-16):
                nu = 1e100

            if len(sys.argv) >= 10:
                coordinate_flag = str(sys.argv[9])
                if coordinate_flag not in ['cartesian', 'cylindrical', 'toroidal']:
                    raise ValueError(
                        "Error! The coordinate flag must specify one of "
                        "cartesian, cylindrical, or toroidal."
                    )
            else:
                coordinate_flag = 'cartesian' 

    # Set the remaining parameters
    surface_flag = 'vmec'
    pms_name = None
    is_premade_famus_grid = False
    if res_flag == 'high':
        nphi = 64
        ntheta = 64
    elif res_flag == 'medium':
        nphi = 16
        ntheta = 16
    else:
        nphi = 8
        ntheta = 8
    if config_flag == 'muse':
        dr = 0.01
        coff = 0.1
        poff = 0.05
        surface_flag = 'focus'
        input_name = 'input.' + config_flag 
    if config_flag == 'muse_famus':
        dr = 0.01
        coff = 0.1
        poff = 0.02
        surface_flag = 'focus'
        input_name = 'input.muse'
        pms_name = 'zot80.focus'
        is_premade_famus_grid = True
    elif 'QH' in config_flag:
        dr = 0.4
        coff = 2.4
        poff = 1.6
        input_name = 'wout_LandremanPaul2021_' + config_flag[:2].upper() + '_reactorScale_lowres_reference.nc'
        surface_flag = 'wout'
    elif 'qa' in config_flag or 'qh' in config_flag:
        dr = 0.01
        coff = 0.1
        poff = 0.04
        if 'qa' in config_flag:
            input_name = 'input.LandremanPaul2021_' + config_flag[:2].upper()
        else:
            input_name = 'wout_LandremanPaul_' + config_flag[:2].upper() + '_variant.nc'
            surface_flag = 'wout'
    elif config_flag == 'ncsx':
        dr = 0.02
        coff = 0.02
        poff = 0.1  # 0.2
        surface_flag = 'wout'
        input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
        pms_name = 'init_orient_pm_nonorm_5E4_q4_dp.focus' 
        is_premade_famus_grid = True

    print('Config flag = ', config_flag)
    print('Resolution flag = ', res_flag)
    print('Type of run = ', run_type)
    if run_type == 'optimization':
        print('L2 regularization = ', reg_l2)
        print('Error tolerance for the convex subproblem = ', epsilon)
        print('Maximum iterations for the convex subproblem = ', max_iter_MwPGP)
        print('min f_B value (if achieved, algorithm quits) = ', min_fb)
        print('L0 regularization = ', reg_l0)
        print('nu = ', nu)
        print('Maximum iterations for relax-and-split = ', max_iter_RS)
    print('Coordinate system = ', coordinate_flag)
    print('Input file name = ', input_name)
    print('nphi = ', nphi)
    print('ntheta = ', ntheta)
    print('Pre-made grid of dipoles (if the grid is from a FAMUS run) = ', pms_name)
    if run_type == 'initialization':
        return config_flag, res_flag, run_type, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dr, coff, poff, surface_flag, input_name, nphi, ntheta, pms_name, is_premade_famus_grid, coordinate_flag
    elif run_type == 'optimization':
        return config_flag, res_flag, run_type, reg_l2, epsilon, max_iter_MwPGP, min_fb, reg_l0, reg_l1, nu, max_iter_RS, dr, coff, poff, surface_flag, input_name, nphi, ntheta, pms_name, is_premade_famus_grid, coordinate_flag
    elif run_type == 'post-processing':
        return config_flag, res_flag, run_type, reg_l2, 0.0, 0, 0.0, reg_l0, reg_l1, nu, 0, dr, coff, poff, surface_flag, input_name, nphi, ntheta, pms_name, is_premade_famus_grid, coordinate_flag 


def read_focus_coils(filename):
    """
        Reads in the coils for the MUSE phased TF coils.
    """
    from simsopt.geo import CurveXYZFourier
    from simsopt.field import Current

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

    # Set the degrees of freedom in the coil objects
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


def coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, config_flag):
    """
        Optimize the coils for the QA or QH configurations.
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


def read_regcoil_pm(filename, surface_filename, OUT_DIR):
    """
        Load in REGCOIL_PM solution for NCSX half-Tesla example. This
        is likely to have errors and need to double check this.
    """
    from simsopt.geo import SurfaceRZFourier

    f = netcdf.netcdf_file(filename, 'r', mmap=False)
    nfp = f.variables['nfp'][()]
    ntheta_plasma = f.variables['ntheta_plasma'][()]
    ntheta_coil = f.variables['ntheta_coil'][()]
    nzeta_plasma = f.variables['nzeta_plasma'][()]
    nzeta_coil = f.variables['nzeta_coil'][()]
    nzetal_plasma = f.variables['nzetal_plasma'][()]
    nzetal_coil = f.variables['nzetal_coil'][()]
    zeta_coil = f.variables['zeta_coil'][()]
    zetal_coil = f.variables['zetal_coil'][()]
    r_coil = f.variables['r_coil'][()]
    chi2_B = f.variables['chi2_B'][()]
    print('chi2B = ', chi2_B)

    chi2_M = f.variables['chi2_M'][()]
    abs_M = f.variables['abs_M'][()]

    print('volume = ', f.variables['volume_magnetization'][()])
    d = f.variables['d'][()]
    Bnormal_from_TF_and_plasma_current = f.variables['Bnormal_from_TF_and_plasma_current'][()]
    max_Bnormal = f.variables['max_Bnormal'][()]
    print('REGCOIL_PM max Bnormal = ', max_Bnormal)
    Bnormal_total = f.variables['Bnormal_total'][()]
    Mvec = f.variables['magnetization_vector'][()]

    # Make a surface for plotting the solution
    quadpoints_phi = np.linspace(0, 1, nzetal_plasma, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta_plasma, endpoint=True)
    s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    #s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

    # Get Bnormal around the full torus and plot results
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

    # Get the magnetization vectors and max magnetization 
    Mvec = Mvec[-1, :, :, :, :].reshape(3, nzeta_coil, ntheta_coil)
    m_maxima = np.ravel(abs_M[-1, :, :, :])
    rho = np.ravel(np.sqrt(np.sum(Mvec ** 2, axis=0))) / np.ravel(m_maxima)
    print(rho)
    print('Effective volume = ', np.sum(rho))
    
    bf_inds = np.logical_and((rho < 0.99), (rho > 0.01))
    print('Binary fraction = ', np.count_nonzero(rho) / len(rho))
    
    Mvec_total = np.zeros((nzetal_coil, ntheta_coil, 3))

    # convert Mvec to cartesian
    for i in range(ntheta_coil):
        Mvec_x = Mvec[0, :, i] * np.cos(zeta_coil) - Mvec[1, :, i] * np.sin(zeta_coil)
        Mvec_y = Mvec[0, :, i] * np.sin(zeta_coil) + Mvec[1, :, i] * np.cos(zeta_coil)
        # somehow need to put in stellarator symmetry below...
        for fp in range(nfp):
            phi0 = (2 * np.pi / nfp) * fp
            Mvec_total[fp * nzeta_coil:(fp + 1) * nzeta_coil, i, 0] = Mvec_x * np.cos(phi0) - Mvec_y * np.sin(phi0)
            Mvec_total[fp * nzeta_coil:(fp + 1) * nzeta_coil, i, 1] = Mvec_x * np.sin(phi0) + Mvec_y * np.cos(phi0)
            Mvec_total[fp * nzeta_coil:(fp + 1) * nzeta_coil, i, 2] = Mvec[2, :, i]

    Mvec_total = Mvec_total.reshape(nzetal_coil * ntheta_coil, 3)
    Mvec = np.transpose(Mvec, [1, 2, 0]).reshape(nzeta_coil * ntheta_coil, 3)
    m_maxima = np.ravel(np.array([m_maxima, m_maxima, m_maxima]))

    # Make 3D vtk file of dipole vectors and locations
    # Ideally would like to have the individual volumes to convert
    # M -> m for comparison with other codes
    ox = np.ascontiguousarray(r_coil[:, :, 0])
    oy = np.ascontiguousarray(r_coil[:, :, 1])
    oz = np.ascontiguousarray(r_coil[:, :, 2])
    mx = np.ascontiguousarray(Mvec_total[:, 0])
    my = np.ascontiguousarray(Mvec_total[:, 1])
    mz = np.ascontiguousarray(Mvec_total[:, 2])
    mx_normalized = np.ascontiguousarray(mx / m_maxima)
    my_normalized = np.ascontiguousarray(my / m_maxima)
    mz_normalized = np.ascontiguousarray(mz / m_maxima)
    print("write VTK as points")
    data = {"m": (mx, my, mz), "m_normalized": (mx_normalized, my_normalized, mz_normalized)}
    pointsToVTK(
        OUT_DIR + "Dipole_Fields_REGCOIL_PM", ox, oy, oz, data=data
    )


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
    nfieldlines = 120
    tmax_fl = 20000

    # Different configurations have different cross-sections
    #if 'muse' in config:
    #R0 = np.linspace(1.2125346, 1.295, nfieldlines)
    R0 = np.linspace(0.25, 0.35, nfieldlines)
    if 'qa' in config: 
        R0 = np.linspace(0.5, 1.0, nfieldlines)
    elif 'qh' in config:
        R0 = np.linspace(0.5, 1.3, nfieldlines)
    elif config == 'ncsx': 
        R0 = np.linspace(1.0, 1.75, nfieldlines)
    elif config == 'QH':
        R0 = np.linspace(12.0, 18.0, nfieldlines)
    Z0 = np.ones(nfieldlines) * 0.0
    #Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]

    # compute the fieldlines from the initial locations specified above
    sc_fieldline = SurfaceClassifier(s, h=0.05, p=2)
    #sc_fieldline = SurfaceClassifier(s, h=0.05, p=2)
    sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm,
        phis=phis,  # stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        stopping_criteria=[IterationStoppingCriterion(500000)])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)

    # make the poincare plots
    if comm is None or comm.rank == 0:
        # particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=400, xlims=(0.225, 0.375), ylims=(-0.075, 0.075), surf=s)


def make_qfm(s, Bfield, Bfield_tf):
    """
        Given some Bfield generated by dipoles, and some Bfield_tf generated
        by a set of TF coils, compute a quadratic flux-minimizing surface (QFMS)
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

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=1000,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    #vol = Volume(s)
    #vol_target = vol.J()
    #qfm = QfmResidual(s, Bfield_tf)
    #qfm_surface = QfmSurface(Bfield_tf, s, vol, vol_target)
    #print("initial qfm value for higher-res surface", qfm.J())

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=3000,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Now optimize at fixed toroidal flux
#    tf = ToroidalFlux(s, Bfield_tf)
#    tf_target = tf.J()

 #   qfm_surface = QfmSurface(Bfield, s, tf, tf_target)

  #  res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
    #                   constraint_weight=constraint_weight)
    #print(f"||tf constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    #res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    #print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Check that volume is not changed
    print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

    # Now optimize at fixed area
    #ar = Area(s)
    #ar_target = ar.J()
    #qfm_surface = QfmSurface(Bfield, s, ar, ar_target)

    #res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000, constraint_weight=constraint_weight)
    #print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    #res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    #print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # Check that volume is not changed
    #print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")
    return qfm_surface  # return QFMS


def initialize_coils(config_flag, TEST_DIR, OUT_DIR, s):
    """
        Initializes coils for each of the target configurations that are
        used for permanent magnet optimization.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, ScaledCurrent, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk

    if 'muse' in config_flag:
        # Load in pre-optimized coils
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
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
        R1 = 0.6
        order = 5

        # qh needs to be scaled to 0.1 T on-axis magnetic field strength
        from simsopt.mhd.vmec import Vmec
        vmec_file = 'wout_LandremanPaul2021_QH.nc'
        total_current = Vmec(vmec_file).external_current() / (2 * s.nfp) / 8.75
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [ScaledCurrent(Current(total_current / ncoils * 1e-5), 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

        # fix all the coil shapes so only the currents are optimized 
        for i in range(ncoils):
            base_curves[i].fix_all()
    elif config_flag == 'qa':
        # generate planar TF coils
        ncoils = 4
        R0 = 1.0
        R1 = 0.6
        order = 5

        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        from simsopt.mhd.vmec import Vmec
        vmec_file = 'wout_LandremanPaul2021_QA.nc'
        total_current = Vmec(vmec_file).external_current() / (2 * s.nfp) / 8
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [ScaledCurrent(Current(total_current / ncoils * 1e-5), 1e5) for _ in range(ncoils-1)]
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

    R0 = s.get_rc(0, 0)
    for i in range(nphi):
        bspoints[i] = np.array([R0 * np.cos(s.quadpoints_phi[i] * 2 * np.pi), 
                                R0 * np.sin(s.quadpoints_phi[i] * 2 * np.pi), 
                                0.0]
                               ) 
    bs.set_points(bspoints)
    B0 = np.linalg.norm(bs.B(), axis=-1)
    B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
    surface_area = s.area()
    bnormalization = B0avg * surface_area
    print("Bmag at R = ", R0, ", Z = 0: ", B0) 
    print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg) 


def get_FAMUS_dipoles(pms_name, famus_path='../../tests/test_files/'):
    """
        Reads in and makes vtk plots for a FAMUS grid and
        solution. Used for the MUSE and NCSX examples. 
    """
    famus_file = famus_path + pms_name

    # FAMUS files are for the half-period surface 
    ox, oy, oz, Ic, m0, p, mp, mt = np.loadtxt(
        famus_file, skiprows=3, 
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
    momentq = np.loadtxt(famus_file, skiprows=1, max_rows=1, usecols=[1]) 
    rho = p ** momentq

    mm = rho * m0

    # Convert from spherical to cartesian vectors
    mx = mm * np.sin(mt) * np.cos(mp) 
    my = mm * np.sin(mt) * np.sin(mp) 
    mz = mm * np.cos(mt)
    m_FAMUS = np.ravel((np.array([mx, my, mz]).T))
    return m_FAMUS, m0


def read_FAMUS_grid(pms_name, pm_opt, s, s_plot, Bnormal, Bnormal_plot, OUT_DIR, famus_path='../../tests/test_files/'):
    """
        Reads in and makes vtk plots for a FAMUS grid and
        solution. Used for the MUSE and NCSX examples. 
    """
    from simsopt.objectives import SquaredFlux
    from simsopt.field.magneticfieldclasses import DipoleField

    famus_file = famus_path + pms_name

    # FAMUS files are for the half-period surface 
    ox, oy, oz, Ic, m0, p, mp, mt = np.loadtxt(
        famus_file, skiprows=3, 
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
    momentq = np.loadtxt(famus_file, skiprows=1, max_rows=1, usecols=[1]) 
    rho = p ** momentq

    print('Percent of nonzero FAMUS magnets = ', np.count_nonzero(rho) / len(rho))
    bf_inds = np.logical_or((rho > 0.99), (rho < 0.01))
    print('Binary fraction = ', np.count_nonzero(bf_inds) / len(rho))

    # Make histogram of the normalized dipole magnitudes
    plt.figure()
    plt.hist(abs(rho), bins=np.linspace(0, 1, 30), log=True)
    plt.grid(True)
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(OUT_DIR + 'm_FAMUS_histogram.png')

    # Calculate the effective magnet volume
    mm = rho * m0
    mu0 = 4 * np.pi * 1e-7
    Bmax = 1.4 
    print('FAMUS effective volume = ', np.sum(abs(mm)) * mu0 * 2 * s.nfp / Bmax)

    # Convert from spherical to cartesian vectors
    mx = mm * np.sin(mt) * np.cos(mp) 
    my = mm * np.sin(mt) * np.sin(mp) 
    mz = mm * np.cos(mt)
    m_FAMUS = np.ravel((np.array([mx, my, mz]).T))

    coordinate_flag_temp = pm_opt.coordinate_flag
    pm_opt.coordinate_flag = 'cartesian'

    # Set pm_opt m values to the FAMUS solution so we can use the
    # plotting routines from the DipoleField class object. 
    pm_opt.m = m_FAMUS 
    pm_opt.m_proxy = m_FAMUS

    # critical change to make sure SIMSOPT and FAMUS use 
    # the same-size magnets
    pm_opt.m_maxima = m0 

    b_dipole_FAMUS = DipoleField(pm_opt)
    b_dipole_FAMUS.set_points(s.gamma().reshape((-1, 3)))
    b_dipole_FAMUS._toVTK(OUT_DIR + "Dipole_Fields_FAMUS")

    nphi = len(s.quadpoints_phi)
    qphi = len(s_plot.quadpoints_phi)
    ntheta = len(s.quadpoints_phi)
    Bnormal_FAMUS = np.sum(b_dipole_FAMUS.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=-1)
    f_B_famus = SquaredFlux(s, b_dipole_FAMUS).J()
    f_B_sf = SquaredFlux(s, b_dipole_FAMUS, -Bnormal).J()
    print('f_B (only the FAMUS dipoles) = ', f_B_famus)
    print('f_B (FAMUS total) = ', f_B_sf)

    # Plot Bnormal from optimized Bnormal dipoles
    b_dipole_FAMUS.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_FAMUS = np.sum(b_dipole_FAMUS.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    pointData = {"B_N": Bnormal_FAMUS[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "Bnormal_opt_FAMUS", extra_data=pointData)

    # Plot total Bnormal from optimized Bnormal dipoles + coils
    pointData = {"B_N": (Bnormal_FAMUS + Bnormal_plot)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "Bnormal_total_FAMUS", extra_data=pointData)
    pm_opt.coordinate_flag = coordinate_flag_temp


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
    if pm_opt.is_premade_famus_grid:
        famus_file = '../../tests/test_files/' + pm_opt.pms_name
        m0, p = np.loadtxt(
            famus_file, skiprows=3, 
            usecols=[7, 8], 
            delimiter=',', unpack=True
        )
        # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
        momentq = np.loadtxt(famus_file, skiprows=1, max_rows=1, usecols=[1]) 
        rho = p ** momentq
        rho = rho[pm_opt.Ic_inds]
        x_multi = [m0_abs, mproxy_abs, abs(rho)]
    else:
        x_multi = [m0_abs, mproxy_abs]
    plt.hist(x_multi, bins=np.linspace(0, 1, 40), log=True, histtype='bar')
    plt.grid(True)
    plt.legend(['m', 'w', 'FAMUS'])
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(OUT_DIR + 'm_histograms.png')

    if len(RS_history) != 0:

        m_history = np.array(m_history)
        m_history = m_history.reshape(m_history.shape[0] * m_history.shape[1], pm_opt.ndipoles, 3)
        print(m_history.shape)
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
            ani.save(OUT_DIR + 'm_history' + str(i) + '.mp4')


def run_Poincare_plots(s_plot, bs, b_dipole, config_flag, comm, filename_poincare, OUT_DIR):
    """
        Wrapper function for making Poincare plots.
    """
    from simsopt.field.magneticfieldclasses import InterpolatedField
    from simsopt.objectives import SquaredFlux
    #from simsopt.geo import SurfaceRZFourier
    #import simsopt

    #TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
    #filename = TEST_DIR / 'input.LandremanPaul2021_QA'
    # Note that the range must be "full torus"!
    #s_plot = SurfaceRZFourier.from_vmec_input(filename, nphi=200, ntheta=30, range="full torus")
    # Load in the optimized coils from stage_two_optimization.py:
    #coils_filename = Path(__file__).parent / "../1_Simple/inputs" / "biot_savart_opt.json"
    #bs = simsopt.load(coils_filename)

    n = 32
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
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
    make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_check")
    print('Bnormal dipoles = ', Bnormal_dipole)
    make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "dipole_check")
    print('Bnormal total = ', Bnormal + Bnormal_dipole)
    make_Bnormal_plots(bs + b_dipole, s_plot, OUT_DIR, "total_check")
    print('f_B = ', f_B)

    if config_flag != 'ncsx':
        bsh = InterpolatedField(
            #bs, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
            bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
        )
    else:
        bsh = InterpolatedField(
            b_dipole, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
        )
    # bsh.to_vtk('dipole_fields')
    #try:
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    trace_fieldlines(bsh, 'bsh_PMs_' + filename_poincare, config_flag, s_plot, comm, OUT_DIR)
    #except SystemError:
    #    print('Poincare plot failed.')


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
    ndipoles = pm_opt.ndipoles
    m = pm_opt.m.reshape(ndipoles, 3)
    mx = m[:, 0]
    my = m[:, 1]
    mz = m[:, 2]
    m0 = pm_opt.m_maxima
    pho = np.sqrt(np.sum(m ** 2, axis=-1)) / m0
    Lc = 0
    ox = pm_opt.dipole_grid_xyz[:, 0]
    oy = pm_opt.dipole_grid_xyz[:, 1]
    oz = pm_opt.dipole_grid_xyz[:, 2]
    #phi = np.arctan2(oy, ox)
    #theta = np.arctan2( np.sqrt(ox ** 2 + oy ** 2), oz)
    #mt = -np.sin(phi) * mx + np.cos(phi) * my 
    #mp = np.cos(phi) * np.cos(theta) * mx + np.sin(phi) * np.cos(theta) * my - np.sin(theta) * mz
    mt = np.arctan2(my, mx)
    mp = np.arctan2(np.sqrt(mx ** 2 + my ** 2), mz)
    coilname = ["pm_{:010d}".format(i) for i in range(1, ndipoles + 1)]
    Ic = 1
    symmetry = 2 
    filename = OUT_DIR + 'SIMSOPT_dipole_solution.focus'

    with open(filename, "w") as wfile:
        wfile.write(" # Total number of dipoles,  momentq \n")
        wfile.write(
            "{:6d},  {:4d}\n".format(
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
