#!/usr/bin/env python
r"""
This example script optimizes a set of relatively simple toroidal field coils
and passive superconducting coils (PSCs)
for an ARIES-CS reactor-scale version of the precise-QH stellarator from 
Landreman and Paul. 

The script should be run as:
    mpirun -n 1 python QH_psc_example.py
on a cluster machine but 
    python QH_psc_example.py
is sufficient on other machines. Note that this code does not use MPI, but is 
parallelized via OpenMP, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.geo.psc_grid import PSCgrid
from simsopt.objectives import SquaredFlux
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *
import time
import shutil

# np.random.seed(1)  # set a seed so that the same PSCs are initialized each time

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
else:
    # Resolution needs to be reasonably high if you are doing permanent magnets
    # or small coils because the fields are quite local
    nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    # Make higher resolution surface for plotting Bnormal
    qphi = nphi * 2
    quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta * 2, endpoint=True)

poff = 1.0  # PSC grid will be offset 'poff' meters from the plasma surface
coff = 1.75  # PSC grid will be initialized between 1 m and 2 m from plasma

# Read in the plasma equilibrium file
input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
range_param = 'half period'
s = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi, ntheta=ntheta
)
# Print major and minor radius
print('s.R = ', s.get_rc(0, 0))
print('s.r = ', s.get_rc(1, 0))

# Make inner and outer toroidal surfaces very high resolution,
# which helps to initialize coils precisely between the surfaces. 
s_inner = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4
)
s_outer = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4
)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

# Make the output directory
out_str = "QH_psc_output/"
# Try to remove the tree; if it fails, throw an error using try...except.
try:
    shutil.rmtree(out_str)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))
out_dir = Path("QH_psc_output")
out_dir.mkdir(parents=True, exist_ok=True)

# Save the inner and outer surfaces for debugging purposes
s_inner.to_vtk(out_str + 'inner_surf')
s_outer.to_vtk(out_str + 'outer_surf')

def initialize_coils_QH(TEST_DIR, s):
    """
    Initializes coils for each of the target configurations that are
    used for permanent magnet optimization.

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

    # generate planar TF coils
    ncoils = 2
    R0 = s.get_rc(0, 0)
    R1 = s.get_rc(1, 0) * 4
    order = 3

    # qh needs to be scaled to 0.1 T on-axis magnetic field strength
    from simsopt.mhd.vmec import Vmec
    vmec_file = 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc'
    total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 1.311753 / 1.1
    print('Total current = ', total_current)
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, 
                                               R0=R0, R1=R1, order=order, numquadpoints=128)
    base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
    # base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils)]
    # base_currents[0].fix_all()

    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir / "curves_init")
    return base_curves, curves, coils

# initialize the coils
base_curves, curves, coils = initialize_coils_QH(TEST_DIR, s)
currents = np.array([coil.current.get_value() for coil in coils])

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, 
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)
s_plot.save(filename=out_dir / 'plasma_boundary.json')

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils and plot results
# fix all the coil shapes so only the currents are optimized
# for i in range(ncoils):
#     base_curves[i].fix_all()

def coil_optimization_QH(s, bs, base_curves, curves):
    from scipy.optimize import minimize
    from simsopt.geo import CurveLength, CurveCurveDistance, \
        MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
    from simsopt.objectives import QuadraticPenalty
    from simsopt.geo import curves_to_vtk
    from simsopt.objectives import SquaredFlux

    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    ncoils = len(base_curves)

    # Weight on the curve lengths in the objective function:
    LENGTH_WEIGHT = 1.5

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 1.6
    CC_WEIGHT = 1

    # Threshold and weight for the coil-to-surface distance penalty in the objective function:
    CS_THRESHOLD = 3.2
    CS_WEIGHT = 1

    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = 0.1
    CURVATURE_WEIGHT = 1e-6

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = 0.1
    MSC_WEIGHT = 1e-6

    MAXITER = 2000  # number of iterations for minimize

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
        outstr = f"J={J:.3e}, Jf={jf:.3e}, ⟨B·n⟩={BdotN:.3e}"
        cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.3f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.3f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.3f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.3f}, C-S-Sep={Jcsdist.shortest_distance():.3f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.3e}"
        print(outstr)
        return J, grad

    print("""
    ################################################################################
    ### Perform a Taylor test ######################################################
    ################################################################################
    """)
    f = fun
    dofs = JF.x
    # np.random.seed(1)
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
    minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 1000}, tol=1e-15)
    curves_to_vtk(curves, out_dir / "curves_opt")
    bs.set_points(s.gamma().reshape((-1, 3)))
    return bs

bs = coil_optimization_QH(s, bs, base_curves, curves)
bs.save('B_TF.json')
currents = np.array([coil.current.get_value() for coil in coils])
curves_to_vtk(curves, out_dir / "TF_coils", close=True)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized")

# check after-optimization average on-axis magnetic field strength
B_axis = calculate_on_axis_B(bs, s)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized", B_axis)

# Finally, initialize the psc class
kwargs_geo = {"Nx": 4, "out_dir": out_str, 
              # "initialization": "plasma", 
              "poff": poff,}
psc_array = PSCgrid.geo_setup_between_toroidal_surfaces(
    s, coils, s_inner, s_outer,  **kwargs_geo
)
print('Number of PSC locations = ', len(psc_array.grid_xyz))

currents = []
for i in range(psc_array.num_psc):
    currents.append(Current(psc_array.I[i]))
all_coils = coils_via_symmetries(
    psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
)
B_PSC = BiotSavart(all_coils)
# Plot initial errors from only the PSCs, and then together with the TF coils
make_Bnormal_plots(B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
make_Bnormal_plots(bs + B_PSC, s_plot, out_dir, "PSC_and_TF_initial", B_axis)

# Check SquaredFlux values using different ways to calculate it
x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
fB = SquaredFlux(s, bs, np.zeros((nphi, ntheta))).J()
fB_TF = fB / (B_axis ** 2 * s.area())
print('fB only TF coils = ', fB_TF)
bs.set_points(s.gamma().reshape(-1, 3))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
print('fB only TF direct = ', np.sum(Bnormal.reshape(-1) ** 2 * psc_array.grid_normalization ** 2
                                    ) / (2 * B_axis ** 2 * s.area()))
make_Bnormal_plots(B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
fB = SquaredFlux(s, B_PSC, np.zeros((nphi, ntheta))).J()
print(fB/ (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, B_PSC + bs, np.zeros((nphi, ntheta))).J()
print('fB with both, before opt = ', fB / (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, B_PSC, -Bnormal).J()
print('fB with both (minus sign), before opt = ', fB / (B_axis ** 2 * s.area()))
# exit()

from scipy.optimize import approx_fprime, check_grad

def callback(x):
    print('fB: ', psc_array.least_squares(x))
    print('approx: ', approx_fprime(x, psc_array.least_squares, 1E-3))
    print('exact: ', psc_array.least_squares_jacobian(x))
    print('-----')
    print(check_grad(psc_array.least_squares, psc_array.least_squares_jacobian, x) / np.linalg.norm(psc_array.least_squares_jacobian(x)))

# Actually do the minimization now
print('beginning optimization: ')
eps = 1e-6
options = {"disp": True, "maxiter": 50}
verbose = True

# Run STLSQ with BFGS in the loop
kwargs_manual = {
                 "out_dir": out_str, 
                 "plasma_boundary" : s,
                 "coils_TF" : coils
                 }
I_threshold = 2e4
I_threshold_scaling = 1.2
I_scaling_scaling = 0.997
STLSQ_max_iters = 2
BdotN2_list = []
num_pscs = []
for k in range(STLSQ_max_iters):
    
    # Threshold scale gets exponentially smaller each iteration
    I_threshold_scaling *= I_scaling_scaling
    I_threshold *= I_threshold_scaling
    x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
    print('Number of PSCs = ', len(x0) // 2, ' in iteration ', k)
    print('I_threshold = ', I_threshold)
    
    # Define the linear bound constraints for each set of angles
    opt_bounds1 = tuple([(-np.pi / 2.0 + eps, np.pi / 2.0 - eps) for i in range(psc_array.num_psc)])
    opt_bounds2 = tuple([(-np.pi + eps, np.pi - eps) for i in range(psc_array.num_psc)])
    opt_bounds = np.vstack((opt_bounds1, opt_bounds2))
    opt_bounds = tuple(map(tuple, opt_bounds))
    t1_min = time.time()
    x_opt = minimize(psc_array.least_squares, 
                     x0, 
                     args=(verbose,),
                     method='L-BFGS-B',
                     bounds=opt_bounds,
                     jac=psc_array.least_squares_jacobian, 
                     options=options,
                     tol=1e-20,  # Required to make progress when fB is small
                     callback=callback
                     )
    t2_min = time.time()
    print(t2_min - t1_min, ' seconds for optimization')
    
    t1_save = time.time()
    psc_array.setup_curves()
    psc_array.plot_curves('final_Ithresh_{0:.3e}'.format(I_threshold) + '_N{0:d}'.format(psc_array.num_psc) + '_')
    currents = []
    for i in range(psc_array.num_psc):
        currents.append(Current(psc_array.I[i]))
    all_coils = coils_via_symmetries(
        psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
    )
    B_PSC = BiotSavart(all_coils)
    B_tot = bs + B_PSC
    B_tot.save('B_total_Ithresh_{0:.3e}'.format(I_threshold) + '_N{0:d}'.format(psc_array.num_psc) + '.json')
    make_Bnormal_plots(B_tot, s_plot, out_dir, 'PSC_and_TF_final_Ithresh_{0:.3e}'.format(I_threshold) + '_N{0:d}'.format(psc_array.num_psc), B_axis)
    t2_save = time.time()
    print(t2_save - t1_save, ' seconds to save all the B field data')
    
    # Do the thresholding and reinitialize a grid without the chopped PSCs
    I = psc_array.I
    grid_xyz = psc_array.grid_xyz
    alphas = psc_array.alphas
    deltas = psc_array.deltas
    if len(BdotN2_list) > 0:
        BdotN2_list = np.hstack((BdotN2_list, np.array(psc_array.BdotN2_list)))
        num_pscs = np.hstack((num_pscs, (len(x0) // 2) * np.ones(len(np.array(psc_array.BdotN2_list)))))
    else:
        BdotN2_list = np.array(psc_array.BdotN2_list)
        num_pscs = np.array((len(x0) // 2) * np.ones(len(np.array(psc_array.BdotN2_list))))
    big_I_inds = np.ravel(np.where(np.abs(I) > I_threshold))
    if len(big_I_inds) != psc_array.num_psc:
        grid_xyz = grid_xyz[big_I_inds, :]
        alphas = alphas[big_I_inds]
        deltas = deltas[big_I_inds]
    else:
        print('STLSQ converged, breaking out of loop')
        break
    kwargs_manual["alphas"] = alphas
    kwargs_manual["deltas"] = deltas
    try:
        psc_array = PSCgrid.geo_setup_manual(
            grid_xyz, psc_array.R, **kwargs_manual
        )
    except TypeError:
        print('Grid initialization raised TypeError, quitting STLSQ loop')
        break

# Plot the data from optimization
num_pscs = np.ravel(num_pscs)
BdotN2_list = np.ravel(BdotN2_list)
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Iterations')
ax1.set_ylabel(r'$f_B$', color=color)
ax1.semilogy(BdotN2_list, color=color)
ax1.plot(fB_TF * np.ones(len(BdotN2_list)), 'k--', label='TF coils only')
ax1.grid()
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend()
ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('# of PSCs', color=color)  # we already handled the x-label with ax1
ax2.plot(num_pscs, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_dir / 'convergence.jpg')

# Last check that direct f_B calculation is still consistent with the optimization
psc_array.setup_curves()
psc_array.plot_curves('final_')
currents = []
for i in range(psc_array.num_psc):
    currents.append(Current(psc_array.I[i]))
all_coils = coils_via_symmetries(
    psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
)
B_PSC = BiotSavart(all_coils)
fB = SquaredFlux(s, B_PSC + bs, np.zeros((nphi, ntheta))).J()
print('fB with both, after opt = ', fB / (B_axis ** 2 * s.area()))
make_Bnormal_plots(B_PSC, s_plot, out_dir, "PSC_final", B_axis)
make_Bnormal_plots(bs + B_PSC, s_plot, out_dir, "PSC_and_TF_final", B_axis)
t2 = time.time()
print('Total time: ', t2 - t1)
plt.show()