"""
This module contains the a number of useful functions for  
optimizing dipole arrays in the SIMSOPT code.
"""
__all__ = ['remove_inboard_dipoles',
           'remove_interlinking_dipoles_and_TFs',
           'align_dipoles_with_plasma',
           'initialize_coils',
           'dipole_array_optimization_function',
           'save_coil_sets',
           'quaternion_from_axis_angle',
           'quaternion_multiply',
           'rotate_vector',
           'compute_quaternion',
           'compute_fourier_coeffs',
           'rho_fourier',
           'generate_even_arc_angles',
           'generate_windowpane_array',
           'generate_tf_array',
           'generate_curves',
           'rho',
           'a_m',
           'b_m'
           ]

import numpy as np


def remove_inboard_dipoles(plasma_surf, base_curves, eps=-0.4):
    """
    Remove all the dipole coils on the inboard side. Useful if the desired plasma
    configuration is fairly compact, so no room for dipoles on inboard side.

    Args:
        plasma_surf: Surface object
            The plasma boundary surface.
        base_curves:
            The curve objects for the dipoles in the array.
        eps: float
            Controls how inboard a dipole can be without being removed.

    Returns:
        base_curves:
            The same objects, minus any dipoles on the inboard side. 
    """
    import warnings
    keep_inds = []
    for ii in range(len(base_curves)):
        counter = 0
        for i in range(base_curves[0].gamma().shape[0]):
            dij = np.sqrt(np.sum((base_curves[ii].gamma()[i, :]) ** 2))
            conflict_bool = (dij < (1.0 + eps) * plasma_surf.get_rc(0, 0))
            if conflict_bool:
                print('bad index = ', i, dij, plasma_surf.get_rc(0, 0))
                warnings.warn(
                    'There is a PSC coil initialized such that it is within a radius'
                    'of a TF coil. Deleting these PSCs now.')
                counter += 1
                break
        if counter == 0:
            keep_inds.append(ii)
    return np.array(base_curves)[keep_inds]


def remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF, eps=0.05):
    """
    Similar process to remove any dipole coils that are initialized
    intersecting with the TF coils, to avoid interlinking at the very beginning of optimization.

    Args:
        base_curves:
            The curve objects for the dipoles in the array.
        base_curves_TF:
            The curve objects for the TF (modular) coils in the array.
        eps: float
            controls how close a TF-dipole coil distance can be before being removed.

    Returns:
        base_curves:
            The same objects, minus any dipoles that were interlinking the TF coils.
    """
    import warnings
    keep_inds = []
    for ii in range(len(base_curves)):
        counter = 0
        for i in range(base_curves[0].gamma().shape[0]):
            for j in range(len(base_curves_TF)):
                for k in range(base_curves_TF[j].gamma().shape[0]):
                    dij = np.sqrt(np.sum((base_curves[ii].gamma()[i, :] - base_curves_TF[j].gamma()[k, :]) ** 2))
                    conflict_bool = (dij < (1.0 + eps) * base_curves[0].x[0])
                    if conflict_bool:
                        # print('bad indices = ', i, j, dij, base_curves[0].x[0])
                        warnings.warn(
                            'There is a PSC coil initialized such that it is within a radius'
                            'of a TF coil. Deleting these PSCs now.')
                        counter += 1
                        break
        if counter == 0:
            keep_inds.append(ii)
    return np.array(base_curves)[keep_inds]


def align_dipoles_with_plasma(plasma_surf, base_curves):
    """
    Initialize a set of dipole coils in base_curves
    to point in the same direction as the nearest plasma surface normal.

    Args:
        plasma_surf: Surface object
            The plasma boundary surface.
        base_curves: list of Curve objects
            The dipole coils to align with the plasma normals.

    Returns:
        alphas: np.ndarray
            One of the unique angles of the planar dipole coils.
        deltas: np.ndarray
            The other unique Euler angle of the planar dipole coils.
    """
    ncoils = len(base_curves)
    coil_normals = np.zeros((ncoils, 3))
    plasma_points = plasma_surf.gamma().reshape(-1, 3)
    plasma_unitnormals = plasma_surf.unitnormal().reshape(-1, 3)
    for i in range(ncoils):
        point = (base_curves[i].get_dofs()[-3:])
        dists = np.sum((point - plasma_points) ** 2, axis=-1)
        min_ind = np.argmin(dists)
        coil_normals[i, :] = plasma_unitnormals[min_ind, :]
    coil_normals = coil_normals / np.linalg.norm(coil_normals, axis=-1)[:, None]
    alphas = np.arcsin(
        -coil_normals[:, 1],
    )
    deltas = np.arctan2(coil_normals[:, 0], coil_normals[:, 2])
    return alphas, deltas


def initialize_coils(s, TEST_DIR, configuration):
    """
    Initializes appropriate coils for the Schuett-Henneberg 2-field-period QA,
    Landreman-Paul QA, Landreman-Paul QH, etc., usually for purposes
    of finding a dipole coil array solution. 

    Args:
        TEST_DIR: String denoting where to find the input files.
        s: plasma boundary surface.
        configuration: String denoting the stellarator name.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
        base_currents: List of Current class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries

    # parameters for the TF coils, increase order for a better solution
    # Total current scaled to give B ~ 5.7 T on axis (actually averaged over the major radius)
    if configuration == 'LandremanPaulQA':
        ncoils = 3
        R0 = s.get_rc(0, 0) * 1
        R1 = s.get_rc(1, 0) * 3
        order = 8
        total_current = 66237606
    elif configuration == 'LandremanPaulQH':
        ncoils = 2
        R0 = s.get_rc(0, 0) * 1
        R1 = s.get_rc(1, 0) * 4
        order = 8
        total_current = 45642162
    elif configuration == 'SchuettHennebergQAnfp2':
        ncoils = 2
        R0 = s.get_rc(0, 0) * 1.4
        R1 = s.get_rc(1, 0) * 4
        order = 16
        total_current = 35000000
    else:
        raise ValueError("Stellarator configuration not recognized.")
    print('Total current = ', total_current)

    # Create the initial coils
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True,
        R0=R0, R1=R1, order=order, numquadpoints=256,
    )
    base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    curves = [c.curve for c in coils]
    return base_curves, curves, coils, base_currents


def dipole_array_optimization_function(dofs, obj_dict, weight_dict, psc_array=None):
    """
    Wrapper function for performing dipole array optimization.

    Args:
        dofs: np.ndarray
            The degrees of freedom for the optimization.
        obj_dict: dict
            Dictionary of objective functions.
        weight_dict: dict
            Dictionary of weights for the objective functions.

    Returns:
        J: float
            The objective function value.
        grad: np.ndarray
            The gradient of the objective function.
    """
    from simsopt.objectives import QuadraticPenalty
    # unpack all the dictionary objects
    btot = obj_dict["btot"]
    s = obj_dict["s"]
    base_curves_TF = obj_dict["base_curves_TF"]
    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    JF = obj_dict["JF"]
    Jf = obj_dict["Jf"]
    Jlength = obj_dict["Jlength"]
    Jlength2 = obj_dict["Jlength2"]
    Jcs = obj_dict["Jcs"]
    Jmscs = obj_dict["Jmscs"]
    Jls = obj_dict["Jls"]
    Jls_TF = obj_dict["Jls_TF"]
    Jccdist = obj_dict["Jccdist"]
    Jccdist2 = obj_dict["Jccdist2"]
    Jcsdist = obj_dict["Jcsdist"]
    linkNum = obj_dict["linkNum"]
    Jforce = obj_dict["Jforce"]
    Jforce2 = obj_dict["Jforce2"]
    Jtorque = obj_dict["Jtorque"]
    Jtorque2 = obj_dict["Jtorque2"]
    length_weight = weight_dict["length_weight"]
    curvature_weight = weight_dict["curvature_weight"]
    msc_weight = weight_dict["msc_weight"]
    msc_threshold = weight_dict["msc_threshold"]
    cc_weight = weight_dict["cc_weight"]
    cs_weight = weight_dict["cs_weight"]
    link_weight = weight_dict["link_weight"]
    force_weight = weight_dict["force_weight"]
    net_force_weight = weight_dict["net_force_weight"]
    torque_weight = weight_dict["torque_weight"]
    net_torque_weight = weight_dict["net_torque_weight"]

    # Set the overall objective degrees of freedom
    JF.x = dofs
    if psc_array is not None:
        # absolutely essential line that updates the PSC currents even though they are not directly optimized.
        psc_array.recompute_currents()
        # absolutely essential line if the PSCs do not have any dofs
        btot.Bfields[0].invalidate_cache()
    J = JF.J()
    grad = JF.dJ()

    # Get all the individual objective values and print them
    jf = Jf.J()
    length_val = length_weight * Jlength.J()
    length_val2 = length_weight * Jlength2.J()
    cc_val = cc_weight * Jccdist.J()
    cc_val2 = cc_weight * Jccdist.J()
    cs_val = cs_weight * Jcsdist.J()
    curvature_val = curvature_weight * sum(Jcs).J()
    msc_val = msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs).J()
    link_val = link_weight * linkNum.J()
    forces_val = Jforce.J()
    forces_val2 = Jforce2.J()
    torques_val = Jtorque.J()
    torques_val2 = Jtorque2.J()
    BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                           ) / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    cl_string2 = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    outstr += f", LenDipoles=sum([{cl_string2}])={sum(J.J() for J in Jls):.2f}"
    valuestr += f", LenDipoles={length_val2:.2e}"
    valuestr += f", LenObj={length_val:.2e}"
    valuestr += f", ccObj={cc_val:.2e}"
    valuestr += f", ccObj2={cc_val2:.2e}"
    valuestr += f", csObj={cs_val:.2e}"
    valuestr += f", curvatureObj={curvature_val:.2e}"
    valuestr += f", mscObj={msc_val:.2e}"
    valuestr += f", Lk1Obj={link_val:.2e}"
    valuestr += f", forceObj={force_weight * forces_val:.2e}"
    valuestr += f", forceObj2={net_force_weight * forces_val2:.2e}"
    valuestr += f", torqueObj={torque_weight * torques_val:.2e}"
    valuestr += f", torqueObj2={net_torque_weight * torques_val2:.2e}"
    outstr += f", F={forces_val:.2e}"
    outstr += f", Fnet={forces_val2:.2e}"
    outstr += f", T={torques_val:.2e}"
    outstr += f", Tnet={torques_val2:.2e}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-C-Sep2={Jccdist2.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", Link Number = {linkNum.J()}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    print(valuestr, '\n')
    return J, grad


def save_coil_sets(btot, OUT_DIR, file_suffix, a, b, nturns_TF, aa, bb, nturns, regularization=None):
    """
    Save separate TF and dipole array coil sets for a dipole array solution.

    Args:
        btot: BiotSavart object
            The BiotSavart object containing (both) the coil sets.
        OUT_DIR: str
            The output directory.
        file_suffix: str
            The suffix for the output files.
        a: float
            The width (radius) of a rectangular (circular) cross section TF coil
        b: float
            The length (radius) of a rectangular (circular) cross section TF coil
        nturns_TF: int
            The number of turns in the TF coils.
        a: float
            The width (radius) of a rectangular (circular) cross section dipole coil
        b: float
            The length (radius) of a rectangular (circular) cross section dipole coil
        nturns_TF: int
            The number of turns in the dipole coils.
    """
    from simsopt.field.force import pointData_forces_torques, coil_net_torques, coil_net_forces
    from simsopt.geo import curves_to_vtk
    from simsopt.field import regularization_rect
    if regularization is None:
        regularization = regularization_rect
    bs = btot.Bfields[0]
    bs_TF = btot.Bfields[1]
    coils = bs.coils
    coils_TF = bs_TF.coils
    allcoils = coils + coils_TF
    dipole_currents = [c.current.get_value() for c in coils]
    curves_to_vtk(
        [c.curve for c in bs.coils],
        OUT_DIR + "dipole_curves" + file_suffix,
        close=True,
        extra_point_data=pointData_forces_torques(coils, allcoils, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
        I=dipole_currents,
        NetForces=coil_net_forces(coils, allcoils, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
        NetTorques=coil_net_torques(coils, allcoils, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
    )
    curves_to_vtk(
        [c.curve for c in bs_TF.coils],
        OUT_DIR + "TF_curves" + file_suffix,
        close=True,
        extra_point_data=pointData_forces_torques(coils_TF, allcoils, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
        I=[c.current.get_value() for c in coils_TF],
        NetForces=coil_net_forces(coils_TF, allcoils, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
        NetTorques=coil_net_torques(coils_TF, allcoils, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    )
    print('Max I = ', np.max(np.abs(dipole_currents)))
    print('Min I = ', np.min(np.abs(dipole_currents)))

# These four functions are used to compute the rotation quaternion for the coil


def quaternion_from_axis_angle(axis, theta):
    """
    Compute a quaternion from a rotation axis and angle.
    Parameters:
        axis (array-like): A 3-element array representing the rotation axis.
        theta (float): The rotation angle in radians.
    Returns:
        np.array: The resulting quaternion as a 4-element array [q0, qi, qj, qk].
    """
    axis = axis / np.linalg.norm(axis)
    q0 = np.cos(theta / 2)
    q_vec = axis * np.sin(theta / 2)
    return np.array([q0, *q_vec])


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q = q1 * q2.
    Parameters:
        q1 (array-like): A 4-element array representing the first quaternion [q0, qi, qj, qk].
        q2 (array-like): A 4-element array representing the second quaternion [q0, qi, qj, qk].
    Returns:
        np.array: The resulting quaternion as a 4-element array [q0, qi, qj, qk].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rotate_vector(v, q):
    """
    Rotate a vector v using quaternion q.
    Parameters:
        v (array-like): A 3-element array representing the vector to rotate.
        q (array-like): A 4-element array representing the quaternion [q0, qi, qj, qk].
    Returns:
        np.array: The rotated vector.
    """
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
    """
    Compute the radius of a superellipse at angle theta.
    Parameters:
        theta: angle in radians
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        rho: radius at angle theta
    """
    return (1/(abs(np.cos(theta)/a)**(n) + abs(np.sin(theta)/b)**(n)) ** (1/(n)))

# Define Fourier coefficient integrals


def a_m(m, a, b, n):
    """
    Compute the Fourier coefficient a_m for a superellipse.
    Parameters:
        m: order of the Fourier coefficient
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        a_m: Fourier coefficient a_m
    """
    from scipy import integrate as spi
    def integrand(theta): return rho(theta, a, b, n) * np.cos(m * theta)
    if m == 0:
        return (1 / (2 * np.pi)) * spi.quad(integrand, 0, 2 * np.pi)[0]
    else:
        return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]


def b_m(m, a, b, n):
    """
    Compute the Fourier coefficient b_m for a superellipse.
    Parameters:
        m: order of the Fourier coefficient
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        b_m: Fourier coefficient b_m
    """
    from scipy import integrate as spi
    def integrand(theta): return rho(theta, a, b, n) * np.sin(m * theta)
    return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]

# Compute Fourier coefficients up to a given order


def compute_fourier_coeffs(max_order, a, b, n):
    """
    Compute Fourier coefficients for a superellipse.

    Parameters: 
        max_order: maximum order of the Fourier series
        a: semi-major axis of the superellipse
        b: semi-minor axis of the superellipse
        n: exponent for the superellipse equation
    Returns:
        coeffs: dictionary of Fourier coefficients
    """
    coeffs = {'a_m': [], 'b_m': []}
    for m in range(max_order + 1):
        coeffs['a_m'].append(a_m(m, a, b, n))
        coeffs['b_m'].append(b_m(m, a, b, n))
    return coeffs

# Reconstruct the Fourier series approximation


def rho_fourier(theta, coeffs, max_order):
    """
    Reconstruct the Fourier series approximation of the superellipse.

    Parameters:
        theta: angle in radians
        coeffs: dictionary of Fourier coefficients
        max_order: maximum order of the Fourier series
    Returns:
        rho_approx: approximate radius at angle theta
    """
    rho_approx = coeffs['a_m'][0]
    for m in range(1, max_order + 1):
        rho_approx += coeffs['a_m'][m] * np.cos(m * theta) + coeffs['b_m'][m] * np.sin(m * theta)
    return rho_approx

# use this to evenly space coils on elliptical grid
# since evenly spaced in poloidal angle won't work
# must use quadrature for elliptic integral


def generate_even_arc_angles(a, b, ntheta):
    """
    Generate ntheta evenly spaced angles along the arc length of an ellipse.
    Parameters:
        a: semi-major axis of the ellipse
        b: semi-minor axis of the ellipse
        ntheta: number of angles to generate
    Returns:
        thetas: array of angles corresponding to the arc length
    """
    from scipy.integrate import quad
    from scipy.optimize import root_scalar

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
    from simsopt.geo import CurvePlanarFourier
    from scipy.special import ellipe
    from scipy.interpolate import RegularGridInterpolator
    # Identify locations of windowpanes
    VV_a = winding_surface.get_rc(1, 0)
    VV_b = winding_surface.get_zs(1, 0)
    VV_R0 = winding_surface.get_rc(0, 0)
    arc_length = 4 * VV_a * ellipe(1-(VV_b/VV_a)**2)
    nwps_poloidal = int(arc_length / (2 * inboard_radius + wp_fil_spacing))  # figure out how many poloidal dipoles can fit for target radius
    Rpol = arc_length / 2 / nwps_poloidal - wp_fil_spacing / 2  # adjust the poloidal length based off npol to fix filament distance
    theta_locs = generate_even_arc_angles(VV_a, VV_b, nwps_poloidal)
    nwps_toroidal = int((np.pi/winding_surface.nfp*(VV_R0 - VV_a) - half_per_spacing + wp_fil_spacing) / (2 * inboard_radius + wp_fil_spacing))
    if verbose:
        print(f'     Number of Toroidal Dipoles: {nwps_toroidal}')
        print(f'     Number of Poroidal Dipoles: {nwps_poloidal}')
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    unitn_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.unitnormal()[..., i], method='linear') for i in range(3)]
    gamma_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.gamma()[..., i], method='linear') for i in range(3)]
    dgammadtheta_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.gammadash2()[..., i], method='linear') for i in range(3)]
    # Initialize curves
    base_wp_curves = []
    for ii in range(nwps_poloidal):
        for jj in range(nwps_toroidal):
            theta_coil = theta_locs[ii]
            r = VV_a*VV_b / np.sqrt((VV_b*np.cos(theta_coil))**2 + (VV_a*np.sin(theta_coil))**2)
            Rtor = (np.pi/winding_surface.nfp*(VV_R0 + r * np.cos(theta_coil)) - half_per_spacing - (nwps_toroidal-1) * wp_fil_spacing) / (2 * nwps_toroidal)
            # Calculate toroidal angle of center of coil
            dphi = (half_per_spacing/2 + Rtor) / (VV_R0 + r * np.cos(theta_coil))  # need to add buffer in phi for gaps in panels
            phi_coil = dphi + jj * (2 * Rtor + wp_fil_spacing) / (VV_R0 + r * np.cos(theta_coil))
            # Interpolate coil center and rotation vectors
            unitn_interp = np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in unitn_interpolators], axis=-1)
            gamma_interp = np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in gamma_interpolators], axis=-1)
            dgammadtheta_interp = np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in dgammadtheta_interpolators], axis=-1)
            curve = CurvePlanarFourier(numquadpoints, order, winding_surface.nfp, stellsym=True)
            # dofs stored as: [r0, higher order curve terms, q_0, q_i, q_j, q_k, x0, y0, z0]
            # Compute fourier coefficients for given super-ellipse
            coeffs = compute_fourier_coeffs(order, Rpol, Rtor, wp_n)
            for m in range(order+1):
                curve.set(f'x{m}', coeffs['a_m'][m])
                if m != 0:
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


def generate_tf_array(winding_surface, ntf, TF_R0, TF_a, TF_b, fixed_geo_tfs=False, planar_tfs=True, order=6, numquadpoints=32):
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
    from simsopt.geo import create_equally_spaced_curves
    if not fixed_geo_tfs:
        if planar_tfs:
            try:
                from simsopt.geo import create_equally_spaced_cylindrical_curves
                base_tf_curves = create_equally_spaced_cylindrical_curves(ntf, winding_surface.nfp, stellsym=winding_surface.stellsym, R0=TF_R0, a=TF_a, b=TF_b, numquadpoints=numquadpoints)
            except ImportError:
                raise ImportError("Need to be on the windowpane branch with the correct TF curve class to unfix TF geometry")
        else:
            base_tf_curves = create_equally_spaced_curves(ncurves=ntf, nfp=winding_surface.nfp, stellsym=winding_surface.stellsym, R0=TF_R0, R1=TF_a, order=order, numquadpoints=numquadpoints)
    else:
        # order=1 is fine for elliptical
        base_tf_curves = create_equally_spaced_curves(ncurves=ntf, nfp=winding_surface.nfp, stellsym=winding_surface.stellsym, R0=TF_R0, R1=TF_a, order=order, numquadpoints=numquadpoints)
        # add this for elliptical TF coils - keep same ellipticity as VV
        for c in base_tf_curves:
            c.set("zs(1)", -TF_b)  # see create_equally_spaced_curves doc for minus sign info

    return base_tf_curves


def generate_curves(surf, VV, planar_tfs=False, outdir=''):
    """
    Generate the curves for the winding surface and TF coils.

    Args:
        VV: SurfaceRZFourier object
            The winding surface to put the dipole coils on.
        surf: SurfaceRZFourier object   
            The plasma boundary surface.
        planar_tfs: bool
            Whether to use planar TF coils.
        outdir: str
            The output directory for the generated curves.
    """
    from simsopt.geo import curves_to_vtk

    # choose some reasonable parameters for array initialization
    base_wp_curves = generate_windowpane_array(winding_surface=VV,
                                               inboard_radius=0.8,
                                               wp_fil_spacing=0.75,
                                               half_per_spacing=0.75,
                                               wp_n=2,  # elliptical coils
                                               numquadpoints=32,
                                               order=12,  # want high order to approximate ellipse
                                               verbose=True,
                                               )
    # generate TFs of the class CurvePlanarEllipticalCylindrical (fixed_geo_TFs=False)
    base_tf_curves = generate_tf_array(winding_surface=VV,
                                       ntf=3,  # 3 TF coils
                                       TF_R0=surf.get_rc(0, 0),
                                       TF_a=surf.get_rc(1, 0) * 5,
                                       TF_b=surf.get_rc(1, 0) * 5,
                                       fixed_geo_tfs=False,
                                       planar_tfs=planar_tfs,
                                       order=3,
                                       numquadpoints=128,
                                       )
    if planar_tfs:
        # unfix the relevant TF dofs
        for c in base_tf_curves:
            c.fix_all()
            c.unfix('R0')
            c.unfix('r_rotation')

    # export curves
    curves_to_vtk(base_wp_curves + base_tf_curves, outdir + 'test_curves')
    surf.to_vtk(outdir + 'test_winding_surf')
    VV.to_vtk(outdir + 'test_vessel')
    return base_wp_curves, base_tf_curves
