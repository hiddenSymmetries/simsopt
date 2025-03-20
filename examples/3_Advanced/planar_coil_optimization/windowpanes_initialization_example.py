"""
File: windowpane_initialization_example.py
Author: Jake Halpern
Last Edit Date: 03/2025
Description: This script shows how to set up planar non-encircling coils tangent to an 
             axisymmetric winding surface and planar toroidal field coils that can be 
             shifted and tilted radially 
             
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from simsopt.geo import CurvePlanarFourier, create_equally_spaced_curves
from simsopt.field import BiotSavart, Current, coils_via_symmetries, regularization_rect
# from simsopt.field.force import LpCurveForce, \
#     SquaredMeanForce, \
#     SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_on_axis_B, remove_inboard_dipoles, \
    initialize_coils, save_coil_sets
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt._core import Optimizable
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
import scipy.integrate as spi
from scipy.special import ellipe
from scipy.integrate import quad
from scipy.optimize import root_scalar
import os
import time

outdir = './columbia_tokamak_stellarator_hybrid/'
os.makedirs(outdir, exist_ok=True)
nphi = 32
ntheta = 32
qphi = 2 * nphi
qtheta = 2 * ntheta

# These four functions are used to compute the rotation quaternion for the coil
def quaternion_from_axis_angle(axis, theta):
    """Compute a quaternion from a rotation axis and angle."""
    axis = axis / np.linalg.norm(axis)
    q0 = np.cos(theta / 2)
    q_vec = axis * np.sin(theta / 2)
    return np.array([q0, *q_vec])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vector(v, q):
    """Rotate a vector v using quaternion q."""
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])  # q* (conjugate)
    v_quat = np.array([0, *v])  # Convert v to quaternion form
    v_rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conjugate)
    return v_rotated[1:]  # Return the vector part

def compute_quaternion(normal, tangent):
    """
    Compute the quaternion to rotate the upward direction [0, 0, 1] to the given unit normal vector.
    Then, compute the change needed to align the x direction [1, 0, 0] to the desired tangent after the first rotation.
    
    Parameters:
        normal (array-like): A 3-element array representing the unit normal vector [n_x, n_y, n_z].
        tangent (array-like): A 3-element array representing the unit tangent vector [t_x, t_y, t_z].
    
    Returns:
        np.array: Quaternion as [q0, qi, qj, qk].
    """
    # Ensure input vectors are numpy arrays and normalized
    normal = np.asarray(normal)
    tangent = np.asarray(tangent)

    if not np.isclose(np.linalg.norm(normal), 1.0):
        raise ValueError("The input normal must be a unit vector.")
    if not np.isclose(np.linalg.norm(tangent), 1.0):
        raise ValueError("The input tangent must be a unit vector.")

    # Step 1: Rotate +z (upward) to normal
    upward = np.array([0.0, 0.0, 1.0])
    cos_theta1 = np.dot(upward, normal)
    axis1 = np.cross(upward, normal)

    if np.allclose(axis1, 0):
        q1 = np.array([1.0, 0.0, 0.0, 0.0]) if np.allclose(normal, upward) else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        axis1 /= np.linalg.norm(axis1)
        theta1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
        q1 = quaternion_from_axis_angle(axis1, theta1)

    # Step 2: Rotate initial x-direction using q1
    initial_x = np.array([1.0, 0.0, 0.0])
    rotated_x = rotate_vector(initial_x, q1)

    # Step 3: Rotate rotated_x to match desired tangent
    axis2 = np.cross(rotated_x, tangent)
    if np.linalg.norm(axis2) < 1e-8:
        q2 = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation needed
    else:
        axis2 /= np.linalg.norm(axis2)
        theta2 = np.arccos(np.clip(np.dot(rotated_x, tangent), -1.0, 1.0))
        q2 = quaternion_from_axis_angle(axis2, theta2)

    # Final quaternion (q2 * q1)
    q_final = quaternion_multiply(q2, q1)
    return q_final

# These functions compute the Fourier series coefficients for the superellipse
def rho(theta, a, b, n):
    return ( 1/(abs(np.cos(theta)/a)**(n) + abs(np.sin(theta)/b)**(n) ) ** (1/(n)))
# Define Fourier coefficient integrals
def a_m(m, a, b, n):
    integrand = lambda theta: rho(theta, a, b, n) * np.cos(m * theta)
    if m==0:
        return (1 / (2 * np.pi)) * spi.quad(integrand, 0, 2 * np.pi)[0]
    else:
        return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]

def b_m(m, a, b, n):
    integrand = lambda theta: rho(theta, a, b, n) * np.sin(m * theta)
    return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]

# Compute Fourier coefficients up to a given order
def compute_fourier_coeffs(max_order, a, b, n):
    coeffs = {'a_m': [], 'b_m': []}
    for m in range(max_order + 1):
        coeffs['a_m'].append(a_m(m, a, b, n))
        coeffs['b_m'].append(b_m(m, a, b, n))
    return coeffs

# Reconstruct the Fourier series approximation
def rho_fourier(theta, coeffs, max_order):
    rho_approx = coeffs['a_m'][0]
    for m in range(1, max_order + 1):
        rho_approx += coeffs['a_m'][m] * np.cos(m * theta) + coeffs['b_m'][m] * np.sin(m * theta)
    return rho_approx

# use this to evenly space coils on elliptical grid
# since evenly spaced in poloidal angle won't work
# must use quadrature for elliptic integral
def generate_even_arc_angles(a, b, ntheta):
    def arc_length_diff(theta):
        return np.sqrt((a * np.sin(theta))**2 + (b * np.cos(theta))**2)
    # Total arc length of the ellipse
    total_arc_length, _ = quad(arc_length_diff, 0, 2 * np.pi)
    arc_lengths = np.linspace(0, total_arc_length, ntheta, endpoint=False)
    def arc_length_to_theta(theta, s_target):
        s, _ = quad(arc_length_diff, 0, theta)
        return s - s_target
    # Solve for theta corresponding to each arc length
    thetas = np.zeros(ntheta)
    for i, s in enumerate(arc_lengths):
        if i != 0:
            result = root_scalar(arc_length_to_theta, args=(s,), bracket=[thetas[i-1], 2*np.pi])
            thetas[i] = result.root
    return thetas

def generate_windowpane_array(winding_surface, inboard_radius, wp_fil_spacing, half_per_spacing, wp_n, numquadpoints=32, order=12, verbose=False):
    """
    Initialize an array of nwps_poloidal x nwps_toroidal planar windowpane coils on a winding surface
    Coils are initialized with a current of 1, that can then be scaled using ScaledCurrent
    Parameters:
        winding_surface: surface upon which to place the coils, with coil plane locally tangent to the surface normal
                         assumed to be an elliptical cross section
        inboard_radius: radius of dipoles at inboard midplane - constant poloidally, will increase toroidally
        wp_fil_spacing: spacing wp filaments
        half_per_spacing: spacing between half period segments
        wp_n: value of n for superellipse, see https://en.wikipedia.org/wiki/Superellipse
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle 
               (see CurvePlanarFourier documentation), more for ellipse approximation
    Returns:
        base_wp_curves: list of initialized curves (half field period)
    """    
    # Identify locations of windowpanes
    VV_a = winding_surface.get_rc(1,0)
    VV_b = winding_surface.get_zs(1,0)
    VV_R0 = winding_surface.get_rc(0,0)
    arc_length = 4 * VV_a * ellipe(1-(VV_b/VV_a)**2)
    nwps_poloidal = int(arc_length / (2 * inboard_radius + wp_fil_spacing)) # figure out how many poloidal dipoles can fit for target radius
    Rpol = arc_length / 2 / nwps_poloidal - wp_fil_spacing / 2 # adjust the poloidal length based off npol to fix filament distance
    theta_locs = generate_even_arc_angles(VV_a, VV_b, nwps_poloidal)
    nwps_toroidal = int((np.pi/winding_surface.nfp*(VV_R0 - VV_a) - half_per_spacing + wp_fil_spacing) / (2 * inboard_radius + wp_fil_spacing))
    if verbose:
        print(f'     Number of Toroidal Dipoles: {nwps_toroidal}')
        print(f'     Number of Poroidal Dipoles: {nwps_poloidal}')
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    unitn_interpolators =        [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.unitnormal()[..., i], method='linear') for i in range(3)]
    gamma_interpolators =        [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.gamma()[..., i], method='linear') for i in range(3)]
    dgammadtheta_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.gammadash2()[..., i], method='linear') for i in range(3)]
    # Initialize curves
    base_wp_curves = []
    for ii in range(nwps_poloidal):
        for jj in range(nwps_toroidal):
            theta_coil = theta_locs[ii]
            r = VV_a*VV_b / np.sqrt((VV_b*np.cos(theta_coil))**2 + (VV_a*np.sin(theta_coil))**2)
            Rtor = (np.pi/winding_surface.nfp*(VV_R0 + r * np.cos(theta_coil)) - half_per_spacing - (nwps_toroidal-1) * wp_fil_spacing) / (2 * nwps_toroidal)
            # Calculate toroidal angle of center of coil
            dphi = (half_per_spacing/2 + Rtor) / (VV_R0 + r * np.cos(theta_coil)) # need to add buffer in phi for gaps in panels
            phi_coil = dphi + jj * (2 * Rtor + wp_fil_spacing) / (VV_R0 + r * np.cos(theta_coil))
            # Interpolate coil center and rotation vectors
            unitn_interp =        np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in unitn_interpolators], axis=-1)
            gamma_interp =        np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in gamma_interpolators], axis=-1)
            dgammadtheta_interp = np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in dgammadtheta_interpolators], axis=-1)
            curve = CurvePlanarFourier(numquadpoints, order, winding_surface.nfp, stellsym=True)
            # dofs stored as: [r0, higher order curve terms, q_0, q_i, q_j, q_k, x0, y0, z0]
            # Compute fourier coefficients for given super-ellipse
            coeffs = compute_fourier_coeffs(order, Rpol, Rtor, wp_n)
            for m in range(order+1):
                curve.set(f'x{m}', coeffs['a_m'][m])
                if m!=0:
                    curve.set(f'x{m+order+1}', coeffs['b_m'][m])
            # Align the coil normal with the surface normal and Rpol axis with dgamma/dtheta
            # Renormalize the vector because interpolation can slightly modify its norm
            quaternion = compute_quaternion(unitn_interp/np.linalg.norm(unitn_interp), dgammadtheta_interp/np.linalg.norm(dgammadtheta_interp))
            curve.set(f'x{curve.dof_size-7}', quaternion[0])
            curve.set(f'x{curve.dof_size-6}', quaternion[1])
            curve.set(f'x{curve.dof_size-5}', quaternion[2])
            curve.set(f'x{curve.dof_size-4}', quaternion[3])
            # Align the coil center with the winding surface gamma
            curve.set(f"x{curve.dof_size-3}", gamma_interp[0])
            curve.set(f"x{curve.dof_size-2}", gamma_interp[1])
            curve.set(f"x{curve.dof_size-1}", gamma_interp[2])
            base_wp_curves.append(curve)
    return base_wp_curves

def generate_tf_array(winding_surface, ntf, TF_R0, TF_a, TF_b, fixed_geo_tfs=False, numquadpoints=32):
    """
    Initialize an array of planar toroidal field coils over a half field period
    Parameters:
        winding_surface: surface upon which to obtain field periodicity/stellarator symmetry for the coils
        ntf: number of TF coils per half field period
        TF_R0: major radius of the TF coils
        TF_a: minor radius of the TF coils (in R direction)
        TF_b: minor radius of the TF coils (in Z direction)
        fixed_geo_tfs: whether to fix the geometric degrees of freedom of the TF coils
        numquadpoints: number of quadrature points representing each coil
    Returns:
        base_tf_curves: list of initialized curves (half field period)
    """  
    if not fixed_geo_tfs:
        try:
            from simsopt.geo import create_equally_spaced_cylindrical_curves
            base_tf_curves = create_equally_spaced_cylindrical_curves(ntf, winding_surface.nfp, stellsym=winding_surface.stellsym, R0=TF_R0, a=TF_a, b=TF_b, numquadpoints=numquadpoints)
        except ImportError:
            raise ImportError("Need to be on the windowpane branch with the correct TF curve class to unfix TF geometry")
    else:
        # order=1 is fine for elliptical
        base_tf_curves = create_equally_spaced_curves(ncurves=ntf, nfp=winding_surface.nfp, stellsym=winding_surface.stellsym, R0=TF_R0, R1=TF_a, order=1, numquadpoints=numquadpoints)
        # add this for elliptical TF coils - keep same ellipticity as VV
        for c in base_tf_curves:
            c.set("zs(1)", -TF_b) # see create_equally_spaced_curves doc for minus sign info

    return base_tf_curves

def generate_curves():
    from simsopt.geo import SurfaceRZFourier
    from simsopt.geo import curves_to_vtk
    # load in a sample hybrid torus equilibria
    TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
    surf_plas = SurfaceRZFourier.from_wout(TEST_DIR / 'wout_columbia_tokamak_stellarator_hybrid.nc', range='half period', nphi=nphi, ntheta=ntheta)
    quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
    surf_full = SurfaceRZFourier.from_wout(
        TEST_DIR / 'wout_columbia_tokamak_stellarator_hybrid.nc',
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta
    )
    # create a vacuum vessel to place coils on with sufficient plasma-coil distance
    VV = SurfaceRZFourier(nfp=surf_plas.nfp)
    VV.set_rc(0, 0, 1.0)
    VV.set_rc(1, 0, 0.25)
    VV.set_zs(1, 0, 0.27)
    # choose some reasonable parameters for array initialization
    # note for intialization, we use the VV as the surface
    base_wp_curves = generate_windowpane_array(winding_surface=VV,
                                               inboard_radius=0.05,
                                               wp_fil_spacing=0.05,
                                               half_per_spacing=0.05,
                                               wp_n=2, # elliptical coils
                                               numquadpoints=32,
                                               order=12, # want high order to approximate ellipse
                                               verbose=True,
                                            )
    # generate TFs of the class CurvePlanarEllipticalCylindrical (fixed_geo_TFs=False)
    base_tf_curves = generate_tf_array(winding_surface=VV, 
                                       ntf=5,
                                       TF_R0=1.0,
                                       TF_a=0.4,
                                       TF_b=0.46,
                                       fixed_geo_tfs=False,
                                       numquadpoints=32,
    )
    # unfix the relevant TF dofs
    for c in base_tf_curves:
        c.fix_all()
        c.unfix('R0')
        c.unfix('r_rotation')

    # export curves
    curves_to_vtk(base_wp_curves + base_tf_curves, outdir + 'test_curves')
    surf_plas.to_vtk(outdir + 'test_winding_surf')
    VV.to_vtk(outdir + 'test_vessel')
    return base_wp_curves, base_tf_curves, surf_plas, surf_full

if __name__=='__main__':
    base_wp_curves, base_tf_curves, s, s_plot = generate_curves()

    # wire cross section for the TF coils is a square 20 cm x 20 cm
    # Only need this if make self forces and TVE nonzero in the objective!
    a = 0.2
    b = 0.2
    nturns = 100
    nturns_TF = 200

    # wire cross section for the dipole coils should be more like 5 cm x 5 cm
    aa = 0.05
    bb = 0.05
    total_current = 650000
    print('Total current = ', total_current)

    ncoils_TF = len(base_tf_curves)
    base_currents_TF = [(Current(total_current / ncoils_TF * 1e-5) * 1e5) for _ in range(ncoils_TF - 1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents_TF += [total_current - sum(base_currents_TF)]
    coils_TF = coils_via_symmetries(base_tf_curves, base_currents_TF, s.nfp, True)
    base_coils_TF = coils_TF[:ncoils_TF]
    curves_TF = [c.curve for c in coils_TF]
    bs_TF = BiotSavart(coils_TF)
    # Finished initializing the TF coils

    ncoils = len(base_wp_curves)
    # Fix the window pane curve dofs
    [c.fix_all() for c in base_wp_curves]
    base_wp_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
    # Fix currents in each coil
    # for i in range(ncoils):
    #     base_currents[i].fix_all()

    coils = coils_via_symmetries(base_wp_curves, base_wp_currents, s.nfp, True)
    base_coils = coils[:ncoils]

    bs = BiotSavart(coils)
    btot = bs + bs_TF

    calculate_on_axis_B(btot, s)
    btot.set_points(s.gamma().reshape((-1, 3)))
    bs.set_points(s.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    currents = [c.current.get_value() for c in coils]
    a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
    b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))
    base_a_list = np.hstack((np.ones(len(base_coils)) * aa, np.ones(len(base_coils_TF)) * a))
    base_b_list = np.hstack((np.ones(len(base_coils)) * bb, np.ones(len(base_coils_TF)) * b))

    LENGTH_WEIGHT = Weight(0.01)
    CC_THRESHOLD = 0.2
    CC_WEIGHT = 1e2
    CS_THRESHOLD = 0.5
    CS_WEIGHT = 1e2
    LENGTH_TARGET = 50
    LINK_WEIGHT = 1e3

    # Weight for the Coil Coil forces term
    # FORCE_WEIGHT = Weight(0.0)  # 1e-34 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
    # FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
    # TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
    # TORQUE_WEIGHT2 = Weight(0.0)  # 1e-22 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons

    save_coil_sets(btot, outdir, "_initial", a, b, nturns_TF, aa, bb, nturns)
    # Force and Torque calculations spawn a bunch of spurious BiotSavart child objects -- erase them!
    for c in (coils + coils_TF):
        c._children = set()

    btot.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
                "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(outdir + "surf_initial", extra_data=pointData)
    btot.set_points(s.gamma().reshape((-1, 3)))

    # Define the individual terms objective function:
    Jf = SquaredFlux(s, btot)
    # Separate length penalties on the dipole coils and the TF coils
    # since they have very different sizes
    # Jls = [CurveLength(c) for c in base_curves]
    Jls_TF = [CurveLength(c) for c in base_tf_curves]
    Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")

    # coil-coil and coil-plasma distances should be between all coils
    Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(coils + coils_TF))
    Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
    Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

    # While the coil array is not moving around, they cannot
    # interlink.
    linkNum = LinkingNumber(curves + curves_TF, downsample=2)

    # Currently, all force terms involve all the coils
    # all_coils = coils + coils_TF
    # all_base_coils = base_coils + base_coils_TF
    # Jforce = sum([LpCurveForce(c, all_coils, regularization_rect(a_list[i], b_list[i]), p=4, threshold=4e5 * 100, downsample=1
    #                         ) for i, c in enumerate(all_base_coils)])
    # Jforce2 = sum([SquaredMeanForce(c, all_coils, downsample=1) for c in all_base_coils])

    # Errors creep in when downsample = 2
    # Jtorque = sum([LpCurveTorque(c, all_coils, regularization_rect(a_list[i], b_list[i]), p=2, threshold=4e5 * 100, downsample=1
    #                             ) for i, c in enumerate(all_base_coils)])
    # Jtorque2 = sum([SquaredMeanTorque(c, all_coils, downsample=1) for c in all_base_coils])

    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    CURVATURE_WEIGHT = 1e-4
    MSC_WEIGHT = 1e-5
    Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
    Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

    JF = Jf \
        + CC_WEIGHT * Jccdist \
        + CC_WEIGHT * Jccdist2 \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + LENGTH_WEIGHT * Jlength
        # + LINK_WEIGHT * linkNum \

    # if FORCE_WEIGHT.value > 0.0:
    #     JF += FORCE_WEIGHT.value * Jforce  # \

    # if FORCE_WEIGHT2.value > 0.0:
    #     JF += FORCE_WEIGHT2.value * Jforce2  # \

    # if TORQUE_WEIGHT.value > 0.0:
    #     JF += TORQUE_WEIGHT * Jtorque

    # if TORQUE_WEIGHT2.value > 0.0:
    #     JF += TORQUE_WEIGHT2 * Jtorque2

    # import cProfile, io
    # import re
    # import pstats
    # from pstats import SortKey
    def fun(dofs):
        # pr = cProfile.Profile()
        # pr.enable()
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        length_val = LENGTH_WEIGHT.value * Jlength.J()
        cc_val = CC_WEIGHT * (Jccdist.J() + Jccdist2.J())
        cs_val = CS_WEIGHT * Jcsdist.J()
        link_val = LINK_WEIGHT * linkNum.J()
        # forces_val = FORCE_WEIGHT.value * Jforce.J()
        # forces_val2 = FORCE_WEIGHT2.value * Jforce2.J()
        # torques_val = TORQUE_WEIGHT.value * Jtorque.J()
        # torques_val2 = TORQUE_WEIGHT2.value * Jtorque2.J()
        BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        BdotN_over_B = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                            ) / np.mean(btot.AbsB())
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
        valuestr = f"J={J:.2e}, Jf={jf:.2e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
        kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_tf_curves)
        msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
        outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
        valuestr += f", LenObj={length_val:.2e}"
        valuestr += f", ccObj={cc_val:.2e}"
        valuestr += f", csObj={cs_val:.2e}"
        valuestr += f", Lk1Obj={link_val:.2e}"
        # valuestr += f", forceObj={forces_val:.2e}"
        # valuestr += f", forceObj2={forces_val2:.2e}"
        # valuestr += f", torqueObj={torques_val:.2e}"
        # valuestr += f", torqueObj2={torques_val2:.2e}"
        # outstr += f", F={Jforce.J():.2e}"
        # outstr += f", Fnet={Jforce2.J():.2e}"
        # outstr += f", T={Jtorque.J():.2e}"
        # outstr += f", Tnet={Jtorque2.J():.2e}"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-C-Sep2={Jccdist2.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", Link Number = {linkNum.J()}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        print(valuestr)
        # pr.disable()
        # ss = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=ss).sort_stats(sortby)
        # ps.print_stats(20)
        # print(ss.getvalue())
        # exit()
        return J, grad

    print(JF.dof_names)
    # exit()

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
        t1 = time.time()
        J1, _ = f(dofs + eps*h)
        J2, _ = f(dofs - eps*h)
        t2 = time.time()
        print("err", (J1-J2)/(2*eps) - dJh)

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)

    MAXITER = 500
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                options={'maxiter': MAXITER, 'maxcor': 500}, tol=1e-10)
    save_coil_sets(btot, outdir, "_optimized", a, b, nturns_TF, aa, bb, nturns)

    btot.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
                "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(outdir + "surf_optimized", extra_data=pointData)

    btot.set_points(s.gamma().reshape((-1, 3)))
    calculate_on_axis_B(btot, s)

    t2 = time.time()
    print('Total time = ', t2 - t1)
    btot.save(outdir + "biot_savart_optimized" + ".json")
    print(outdir)

