from math import lcm
import numpy as np
from scipy.optimize import root, root_scalar, least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import simsoptpp as sopp
from .tracing import ToroidalTransitStoppingCriterion, IterationStoppingCriterion
from ..util.spectral_diff_matrix import spectral_diff_matrix
from .magneticfield import MagneticField

try:
    from pyevtk.hl import polyLinesToVTK
except ImportError:
    polyLinesToVTK = None

__all__ = ["find_periodic_field_line", "PeriodicFieldLine"]

def _integrate_field_line(field, R0, Z0, Delta_phi, tol=1e-10, phi0=0, nphi=1):
    """Integrate a single field line in the toroidal direction.

    Integration is done in cylindrical coordinates. Integration is always done
    in the +phi direction, regardless of the direction of B.

    For this function, phi0 and Delta_phi range over [0, 2pi], not [0, 1].

    if nphi == 1, then the function returns the final R and Z coordinates, i.e.
    a pair of floats. If nphi > 1, then the function returns a pair of arrays
    of the final R and Z coordinates, i.e. a pair of arrays, corresponding to
    equally spaced points in the phi direction including both phi0 and phi0 + Delta_phi.

    Warning!!! This function does not work correctly when Delta_phi is > 2pi and nphi > 1.

    Args:
        field: The magnetic field object.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        Delta_phi: Distance in radians to integrate in the toroidal direction.
        tol: Tolerance for the integration (default: 1e-10).
        phi0: Initial toroidal angle (default: 0).
        nphi: Number of points in the toroidal direction to record the field line location (default: 1).

    Returns:
        R: Final R coordinate(s).
        Z: Final Z coordinate(s).
    """

    xyz_inits = np.zeros((1, 3))
    xyz_inits[0, 0] = R0 * np.cos(phi0)
    xyz_inits[0, 1] = R0 * np.sin(phi0)
    xyz_inits[0, 2] = Z0

    # Determine if B points towards positive or negative phi. If it points
    # towards negative phi, we need to trace antiparallel to B.
    field.set_points(xyz_inits)
    B = field.B_cyl()
    Bphi = B[0, 1]
    # print("B at initial point:", B)
    if Bphi < 0:
        # print("Bphi < 0, tracing antiparallel to B")
        field_for_tracing = (-1.0) * field
    else:
        # print("Bphi > 0, tracing parallel to B")
        field_for_tracing = field

    # Set up phi values at which to record the field line
    if nphi == 1:
        phi_targets = [phi0 + Delta_phi]
    elif nphi > 1:
        phi_targets = np.linspace(phi0, phi0 + Delta_phi, nphi)
        phi_targets = phi_targets[1:]  # A point at phi=phi0 can cause problems, so we'll add this point manually later
    else:
        raise ValueError("nphi must be >= 1")
    
    n_phi_hits_expected = len(phi_targets)

    tmax = 10000.0
    # print("phi_targets:", phi_targets)
    res_ty, res_phi_hit = sopp.fieldline_tracing(
        field_for_tracing,
        xyz_inits[0, :],
        tmax,
        tol,
        phis=phi_targets,
        stopping_criteria=[
            ToroidalTransitStoppingCriterion(Delta_phi / (2 * np.pi), False),
            IterationStoppingCriterion(10000),
        ],
    )

    if False:
        res = np.array(res_ty)
        import matplotlib.pyplot as plt
        n_rows = 2
        n_cols = 2
        plt.figure(figsize=(14, 8))

        plt.subplot(n_rows, n_cols, 1)
        plt.plot(res[:, 0], res[:, 1])
        plt.xlabel("t")
        plt.ylabel("X")

        plt.subplot(n_rows, n_cols, 2)
        plt.plot(res[:, 0], res[:, 2])
        plt.xlabel("t")
        plt.ylabel("Y")

        plt.subplot(n_rows, n_cols, 3)
        plt.plot(res[:, 0], res[:, 3])
        plt.xlabel("t")
        plt.ylabel("Z")

        plt.subplot(n_rows, n_cols, 4)
        plt.plot(res[:, 1], res[:, 2])
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.tight_layout()
        plt.show()

    # print("Last point in tracing:", res_ty[-1])
    # print("res_phi_hits")
    # for r in res_phi_hit:
    #     print(r)

    xyz_hits = np.zeros((n_phi_hits_expected, 3))
    n_phi_hits = 0
    for j in range(len(res_phi_hit)):
        t, idx, x, y, z = res_phi_hit[j]
        if idx >= 0:
            # xyz_hits[n_phi_hits, :] = [x, y, z]
            xyz_hits[int(idx), :] = [x, y, z]
            n_phi_hits += 1
            # if n_phi_hits > n_phi_hits_expected:
            #     print("res_ty", res_ty)
            #     print("res_phi_hit", res_phi_hit)
            #     raise RuntimeError(f"Should not have more than {n_phi_hits_expected} hits!")
            
    if n_phi_hits < n_phi_hits_expected:
        print("res_ty", res_ty)
        print("res_phi_hit", res_phi_hit)
        raise RuntimeError(f"When tracing field line, {n_phi_hits} hits to the expected phi value were found; expected at least {n_phi_hits_expected}.")
    
    if nphi > 1:
        # Add the point at phi0:
        xyz_hits = np.concatenate((xyz_inits, xyz_hits), axis=0)
    
    R = np.sqrt(xyz_hits[:, 0] ** 2 + xyz_hits[:, 1] ** 2)
    Z = xyz_hits[:, 2]
    if nphi == 1:
        return R[0], Z[0]

    return R, Z


def _field_line_tracing_func(t, y, field):
    """Function for the derivatives that define a field line in cylindrical coordinates."""
    phi = t
    R, Z = y
    eval_points = np.array([[R, phi, Z]])
    MagneticField.set_points_cyl(field, eval_points)
    BR, Bphi, Bz = field.B_cyl()[0]

    d_R_d_phi = R * BR / Bphi
    d_Z_d_phi = R * Bz / Bphi
    return np.array([d_R_d_phi, d_Z_d_phi])


def _integrate_field_line_cyl(field, R0, Z0, Delta_phi, tol=1e-10, phi0=0, nphi=1):
    """Integrate a single field line in the toroidal direction.

    In contrast to _integrate_field_line, this function integrates in cylindrical coordinates
    rather than Cartesian coordinates, and uses scipy for the integration instead of boost.

    Integration is done in cylindrical coordinates. Integration is always done
    in the +phi direction, regardless of the direction of B.

    For this function, phi0 and Delta_phi range over [0, 2pi], not [0, 1].

    if nphi == 1, then the function returns the final R and Z coordinates, i.e.
    a pair of floats. If nphi > 1, then the function returns a pair of arrays
    of the final R and Z coordinates, i.e. a pair of arrays, corresponding to
    equally spaced points in the phi direction including both phi0 and phi0 + Delta_phi.

    In contrast to _integrate_field_line(), this function DOES work correctly when Delta_phi is > 2pi and nphi > 1.

    Args:
        field: The magnetic field object.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        Delta_phi: Distance in radians to integrate in the toroidal direction.
        tol: Tolerance for the integration (default: 1e-10).
        phi0: Initial toroidal angle (default: 0).
        nphi: Number of points in the toroidal direction to record the field line location (default: 1).

    Returns:
        R: Final R coordinate(s).
        Z: Final Z coordinate(s).
    """
    if nphi == 1:
        t_eval = [phi0 + Delta_phi]
    elif nphi > 1:
        t_eval = np.linspace(phi0, phi0 + Delta_phi, nphi)
    else:
        raise ValueError("nphi must be >= 1")

    result = solve_ivp(
        _field_line_tracing_func,
        (phi0, phi0 + Delta_phi),
        (R0, Z0),
        args=(field,),
        t_eval=t_eval,
        rtol=tol,
        atol=tol,
    )

    if nphi == 1:
        return result.y[0, -1], result.y[1, -1]
    elif nphi > 1:
        return result.y[0, :], result.y[1, :]


def _find_periodic_field_line_2D(
    field,
    nfp,
    m,
    R0,
    Z0,
    half_period=False,
    solve_tol=1e-6,
    follow_tol=1e-10,
    deflation_R=[],
    deflation_k=1.0,
):
    """Find a periodic field line using a 2D search in (R, Z).
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    Args:
        field: The magnetic field object.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        half_period: If True, look for periodic field lines in the half-period plane instead of the phi=0 plane (default: False).
        solve_tol: Tolerance for the root finding (default: 1e-6).
        follow_tol: Tolerance for the field line integration (default: 1e-10).
    """

    x0 = [R0, Z0]
    Delta_phi = m * 2 * np.pi / nfp
    if half_period:
        phi0 = np.pi / nfp
    else:
        phi0 = 0.0

    def func(x):
        print("  Evaluating x =", x)
        R, Z = _integrate_field_line(field, x[0], x[1], Delta_phi, follow_tol, phi0=phi0)
        return np.array([R - x[0], Z - x[1]])

    sol = root(func, x0, tol=solve_tol)
    print(sol)
    return sol.x[0], sol.x[1]


def _find_periodic_field_line_1D(
    field,
    nfp,
    m,
    R0,
    residual,
    half_period=False,
    solve_tol=1e-6,
    follow_tol=1e-10,
    verbose=1,
    deflation_R=[],
    deflation_k=1.0,
    R_axis=1.0,
):
    """Find a periodic field line using a 1D search along the line Z=0.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    Args:
        field: The magnetic field object.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        R0: Initial R coordinate.
        residual: Which residual to use (default: "R", "Z", or "theta").
        half_period: If True, look for periodic field lines in the half-period plane instead of the phi=0 plane (default: False).
        solve_tol: Tolerance for the root finding (default: 1e-6).
        follow_tol: Tolerance for the field line integration (default: 1e-10).
        R_axis: Major radius of the magnetic axis, used only if residual="theta"
    """

    Delta_phi = m * 2 * np.pi / nfp
    if half_period:
        phi0 = np.pi / nfp
    else:
        phi0 = 0.0

    def func_R(x):
        R, Z = _integrate_field_line(field, x, 0, Delta_phi, follow_tol, phi0=phi0)
        residual = R - x
        if verbose > 0:
            print(f"  R residual, evaluating x = {x:17}, residual = {residual:15}")
        return residual

    def func_Z(x):
        try:
            R, Z = _integrate_field_line(field, x, 0, Delta_phi, follow_tol, phi0=phi0)
            residual = Z
        except RuntimeError:
            print("Error in _integrate_field_line")
            R = np.nan
            Z = np.nan
            residual = 10

        for Rd in deflation_R:
            residual *= (deflation_k + 1 / abs(x - Rd))
        if verbose > 0:
            print(f"  Z residual, evaluating x = {x:17}, final R = {R}, Z = {Z}, residual = {residual:15}, Delta_phi = {Delta_phi}, phi0 = {phi0}, follow_tol = {follow_tol}")
        return residual

    def func_theta(x):
        R, Z = _integrate_field_line(field, x, 0, Delta_phi, follow_tol, phi0=phi0)
        residual = R - x
        if verbose > 0:
            print(f"  theta residual, evaluating x = {x:17}, residual = {residual:15}")
        return residual
    
    if residual == "R":
        func = func_R
    elif residual == "Z":
        func = func_Z
    elif residual == "theta":
        func = func_theta
    else:
        raise ValueError(f"Unknown residual: {residual}")

    sol = root_scalar(func, x0=R0, x1=R0 * 1.0001, xtol=solve_tol, rtol=solve_tol)
    print(sol)
    R0 = sol.root

    # Check difference in final vs starting location:
    R, Z = _integrate_field_line(field, R0, 0, Delta_phi, follow_tol, phi0=phi0)
    print(f"Final - initial R: {R - R0}, Z: {Z}")
    np.testing.assert_allclose(R0, R, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(0.0, Z, atol=1e-6, rtol=1e-6)

    return sol.root


def _find_periodic_field_line_1D_optimization(
    field,
    nfp,
    m,
    R0,
    half_period=False,
    solve_tol=1e-6,
    follow_tol=1e-10,
    verbose=1,
    deflation_R=[],
    deflation_k=1.0,
    R_axis=1.0,
):
    """Find a periodic field line using a 1D search along the line Z=0.

    This routine works by minimizing the objective f = (R - R0)^2 + (Z - 0)^2
    where R and Z are the final coordinates of the field line after integrating.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    Args:
        field: The magnetic field object.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        R0: Initial R coordinate.
        half_period: If True, look for periodic field lines in the half-period plane instead of the phi=0 plane (default: False).
        solve_tol: Tolerance for the root finding (default: 1e-6).
        follow_tol: Tolerance for the field line integration (default: 1e-10).
    """

    Delta_phi = m * 2 * np.pi / nfp
    if half_period:
        phi0 = np.pi / nfp
    else:
        phi0 = 0.0

    def compute_residual(xarray):
        x = xarray[0]
        try:
            R, Z = _integrate_field_line(field, x, 0, Delta_phi, follow_tol, phi0=phi0)
            residual = np.array([R - x, Z])
        except RuntimeError:
            print("Error in _integrate_field_line")
            R = np.nan
            Z = np.nan
            residual = 10

        for Rd in deflation_R:
            residual *= (deflation_k + 1 / abs(x - Rd))
        if verbose > 0:
            cost = 0.5 * np.dot(residual, residual)
            print(f"  optimizing, evaluating x = {x:17}, final R = {R}, Z = {Z}, cost = {cost:15}, Delta_phi = {Delta_phi}, phi0 = {phi0}, follow_tol = {follow_tol}")

        return residual

    sol = least_squares(compute_residual, x0=R0, xtol=solve_tol, ftol=solve_tol, verbose=2)
    print(sol)
    R0 = sol.x[0]

    # Check difference in final vs starting location:
    R, Z = _integrate_field_line(field, R0, 0, Delta_phi, follow_tol, phi0=phi0)
    print(f"Final - initial R: {R - R0}, Z: {Z}")
    np.testing.assert_allclose(R0, R, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(0.0, Z, atol=1e-6, rtol=1e-6)

    return R0

def _pseudospectral_residual(x, n, D, phi, field):
    """
    This is the vector-valued function that returns the residual for the
    pseudospectral method of finding a periodic field line.

    Args:
        x: The state vector (R, Z).
        n: Number of points in the toroidal direction.
        D: The spectral differentiation matrix.
        phi: The grid points of toroidal angle.
        field: The MagneticField object.
    """
    # print("_pseudospectral_residual x =", x)
    R = x[0:n]
    Z = x[n:2 * n]
    eval_points = np.stack([R, phi, Z], axis=-1)
    # In the next line, for some reason there is an error if we try field.set_points_cyl(eval_points)
    MagneticField.set_points_cyl(field, eval_points)
    B_cyl = field.B_cyl()
    BR = B_cyl[:, 0]
    Bphi = B_cyl[:, 1]
    BZ = B_cyl[:, 2]

    R_residual = R * BR / Bphi - (D @ R)
    Z_residual = R * BZ / Bphi - (D @ Z)
    return np.concatenate((R_residual, Z_residual))


def _pseudospectral_jacobian(x, n, D, phi, field):
    """
    This is the matrix-valued function that returns the Jacobian for the
    pseudospectral method of finding a periodic field line.

    Args:
        x: The state vector (R, Z).
        n: Number of points in the toroidal direction.
        D: The spectral differentiation matrix.
        phi: The grid points of toroidal angle.
        field: The MagneticField object.
    """
    R = x[0:n]
    Z = x[n:2 * n]
    print('jacobian eval ')
    eval_points = np.stack([R, phi, Z], axis=-1)
    # In the next line, for some reason there is an error if we try field.set_points_cyl(eval_points)
    MagneticField.set_points_cyl(field, eval_points)
    B_cyl = field.B_cyl()
    BR = B_cyl[:, 0]
    Bphi = B_cyl[:, 1]
    BZ = B_cyl[:, 2]

    # - ``m.dB_by_dX()`` returns an array of size ``(n, 3, 3)`` with the Cartesian coordinates of :math:`\nabla B`. Denoting the indices
    #   by :math:`(i,j,l)`, the result contains  :math:`\partial_j B_l(x_i)`.
    grad_B = field.dB_by_dX()
    d_Bx_d_x = grad_B[:, 0, 0]
    d_By_d_x = grad_B[:, 0, 1]
    d_Bz_d_x = grad_B[:, 0, 2]
    d_Bx_d_y = grad_B[:, 1, 0]
    d_By_d_y = grad_B[:, 1, 1]
    d_Bz_d_y = grad_B[:, 1, 2]
    d_Bx_d_z = grad_B[:, 2, 0]
    d_By_d_z = grad_B[:, 2, 1]
    d_Bz_d_z = grad_B[:, 2, 2]
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    # For the following formulas, see 20250409-01 Converting between Cartesian vs cylindrical for grad B tensor.lyx
    d_Bz_d_R = d_Bz_d_x * cosphi + d_Bz_d_y * sinphi
    d_BR_d_z = d_Bx_d_z * cosphi + d_By_d_z * sinphi
    d_Bphi_d_z = -d_Bx_d_z * sinphi + d_By_d_z * cosphi
    d_BR_d_R = d_Bx_d_x * cosphi**2 + (d_Bx_d_y + d_By_d_x) * sinphi * cosphi + d_By_d_y * sinphi**2
    d_Bphi_d_R = (-d_Bx_d_x + d_By_d_y) * sinphi * cosphi - d_Bx_d_y * sinphi**2 + d_By_d_x * cosphi**2

    jac = np.zeros((2 * n, 2 * n))

    # For reference:
    # R_residual = R * BR / Bphi - (D @ R)
    # Z_residual = R * BZ / Bphi - (D @ Z)

    # Top left quadrant: d (R residual) / d R
    jac[0 : n, 0 : n] = -D + np.diag(
        BR / Bphi 
        + R * d_BR_d_R / Bphi 
        - R * BR / (Bphi**2) * d_Bphi_d_R
    )
    # Top right quadrant: d (R residual) / d Z
    jac[0 : n, n : 2 * n] = np.diag(
        R * d_BR_d_z / Bphi
        - R * BR / (Bphi**2) * d_Bphi_d_z
    )
    # Bottom left quadrant: d (Z residual) / d R
    jac[n : 2 * n, 0 : n] = np.diag(
        BZ / Bphi 
        + R * d_Bz_d_R / Bphi 
        - R * BZ / (Bphi**2) * d_Bphi_d_R
    )
    # Bottom right quadrant: d (Z residual) / d Z
    jac[n:, n:] = -D + np.diag(
        R * d_Bz_d_z / Bphi 
        - R * BZ / (Bphi**2) * d_Bphi_d_z
    )

    return jac


def _find_periodic_field_line_pseudospectral(
    field,
    nfp,
    m,
    R0,
    Z0,
    half_period=False,
    solve_tol=1e-6,
    nphi=21,
    deflation_R=[],
    deflation_k=1.0,
):
    """Find a periodic field line.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    Args:
        field: The magnetic field object.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        half_period: If True, look for periodic field lines in the half-period plane instead of the phi=0 plane (default: False).
        solve_tol: Tolerance for the root finding (default: 1e-6).
        nphi: Number of points in the toroidal direction (default: 21).
    """

    # Make sure nphi is odd
    if nphi % 2 == 0:
        nphi += 1

    phimax = m * 2 * np.pi / nfp
    dphi = phimax / nphi
    phi = np.linspace(0, phimax, nphi, endpoint=False)
    assert np.abs(phi[1] - phi[0] - dphi) < 1.0e-13

    if half_period:
        phi0 = np.pi / nfp
        phi += phi0
    else:
        phi0 = 0.0


    D = spectral_diff_matrix(nphi, xmin=0, xmax=phimax)
        
    # # Initial condition is a circle:
    # state = np.concatenate(
    #     [
    #         np.full(nphi, R0), 
    #         np.full(nphi, Z0)
    #     ]
    # )

    # Establish initial condition:
    if isinstance(R0, (list, np.ndarray)) and isinstance(Z0, (list, np.ndarray)):
        if len(R0) != nphi or len(Z0) != nphi:
            raise ValueError("Length of R0 and Z0 must match nphi")
    else:
        R0, Z0 = _integrate_field_line(field, R0, Z0, phimax, 1e-10, phi0=phi0, nphi=nphi + 1)
        # Subtract off a linear function to make the initial condition periodic
        Delta_R = R0[-1] - R0[0]
        Delta_Z = Z0[-1] - Z0[0]
        R0 -= np.linspace(0, Delta_R, nphi + 1)
        Z0 -= np.linspace(0, Delta_Z, nphi + 1)
        R0 = R0[:-1]
        Z0 = Z0[:-1]
        print("Initial condition R0:", R0)
        print("Initial condition Z0:", Z0)
    
    state = np.concatenate((R0, Z0))

    sol = root(
        _pseudospectral_residual, 
        state,
        tol=solve_tol,
        args=(nphi, D, phi, field),
        jac=_pseudospectral_jacobian,
        method='lm',
        options={'maxiter':1000},
    )
    R = sol.x[0:nphi]
    Z = sol.x[nphi:2 * nphi]

    residual = sol.fun
    print('Residual: ', np.max(np.abs(residual)))

    print(sol)
    return R[0], Z[0]


def find_periodic_field_line(
    field,
    nfp,
    m,
    R0,
    Z0=0.0,
    half_period=False,
    method="2D",
    solve_tol=1e-6,
    follow_tol=1e-10,
    nphi=21,
    deflation_R=[],
    deflation_k=1.0,
):
    """Find a periodic field line.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    The argument ``Z0`` is ignored if method = "1D".

    Args:
        field: The magnetic field object.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        half_period: If True, look for periodic field lines in the half-period plane instead of the phi=0 plane (default: False).
        method: Method for finding the periodic field line (default: "2D").
        solve_tol: Tolerance for the root finding (default: 1e-6).
        follow_tol: Tolerance for the field line integration (default: 1e-10). Matters only for method="2D".
        nphi: Number of points in the toroidal direction (default: 21). Matters only for method="pseudospectral".
    """

    if method == "2D":
        return _find_periodic_field_line_2D(
            field, nfp, m, R0, Z0, half_period, solve_tol, follow_tol, deflation_R=deflation_R, deflation_k=deflation_k
        )
    elif method == "pseudospectral":
        return _find_periodic_field_line_pseudospectral(
            field, nfp, m, R0, Z0, half_period, solve_tol, nphi, deflation_R=deflation_R, deflation_k=deflation_k
        )
    elif method == "1D optimization":
        return _find_periodic_field_line_1D_optimization(
            field, nfp, m, R0, half_period, solve_tol, follow_tol, deflation_R=deflation_R, deflation_k=deflation_k
        ), 0.0
    elif method in ["1D R", "1D Z", "1D theta"]:
        return _find_periodic_field_line_1D(
            field, nfp, m, R0, method[3:], half_period, solve_tol, follow_tol, deflation_R=deflation_R, deflation_k=deflation_k
        ), 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

class PeriodicFieldLine():
    """Class representing a periodic field line.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    The argument ``Z0`` is ignored if method = "1D".

    Args:
        field: The magnetic field object.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        half_period: If True, look for periodic field lines in the half-period plane instead of the phi=0 plane (default: False).
        method: Method for finding the periodic field line (default: "2D").
        solve_tol: Tolerance for the root finding (default: 1e-6).
        follow_tol: Tolerance for the field line integration (default: 1e-10). Irrelevant for method="pseudospectral".
        nphi_solve: Number of points in the toroidal direction for the pseudospectral solve (default: 21). Matters only for method="pseudospectral".
        nphi: Number of points in the toroidal direction for plotting etc (default: 200).
    """
    def __init__(
        self,
        field,
        nfp,
        m,
        R0,
        Z0=0.0,
        half_period=False,
        method="1D Z",
        solve_tol=1e-6,
        follow_tol=1e-10,
        nphi_solve=21,
        nphi=400,
        deflation_R=[],
        deflation_k=1.0,
        asserts=True,
    ):
        self.field = field
        self.nfp = nfp
        self.m = m
        self.half_period = half_period
        self.nphi = nphi
        self.follow_tol = follow_tol

        R0, Z0 = find_periodic_field_line(
            field,
            nfp,
            m,
            R0,
            Z0,
            half_period=half_period,
            method=method,
            solve_tol=solve_tol,
            follow_tol=follow_tol,
            nphi=nphi_solve,
            deflation_R=deflation_R,
            deflation_k=deflation_k,
        )
        self.R0 = R0
        self.Z0 = Z0

        # Now get R and Z along equally spaced phi points:
        if half_period:
            phi0 = np.pi / nfp
        else:
            phi0 = 0

        Delta_phi_to_close = lcm(nfp, m) * 2 * np.pi / nfp
        self.phi0 = phi0
        self.Delta_phi_to_close = Delta_phi_to_close
        self.phi = np.linspace(0, Delta_phi_to_close, nphi) + phi0
        self.R, self.z = _integrate_field_line_cyl(field, R0, Z0, Delta_phi_to_close, tol=follow_tol, phi0=phi0, nphi=nphi)
        self.x = self.R * np.cos(self.phi)
        self.y = self.R * np.sin(self.phi)
        if asserts:
            np.testing.assert_allclose(self.R[0], self.R[-1], atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(self.z[0], self.z[-1], atol=1e-6, rtol=1e-7)
            np.testing.assert_allclose(self.x[0], self.x[-1], atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(self.y[0], self.y[-1], atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(self.R[0], R0, atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(self.z[0], Z0, atol=1e-14, rtol=1e-14)

    def to_vtk(self, filename):
        """Write the field line to a VTK file."""
        if polyLinesToVTK is None:
            raise ImportError("pyevtk is not installed. Cannot write VTK file.")

        pointsPerLine = np.array([self.nphi])
        polyLinesToVTK(filename, self.x, self.y, self.z, pointsPerLine=pointsPerLine, pointData={'phi': self.phi})

    def _integral_A_dl(self):
        """Compute the flux integral ∫A⋅dℓ associated with the periodic field line.
        
        This function also returns the intermediate quantity d_r_d_phi for testing.
        """
        dphi = self.phi[1] - self.phi[0]
        points = np.stack([self.x, self.y, self.z], axis=-1)
        self.field.set_points(points)
        A = self.field.A()
        B_cart = self.field.B()
        B_cyl = self.field.B_cyl()
        B_phi = B_cyl[:, 1]

        d_r_d_phi = (self.R / B_phi)[:, None] * B_cart

        product = A * d_r_d_phi
        # drop repeated point
        integral = dphi * np.sum(product[:-1, :])  # A dot dℓ
        return integral, d_r_d_phi
    
    def integral_A_dl(self):
        """Return the flux integral ∫A⋅dℓ associated with the periodic field line."""
        integral, _ = self._integral_A_dl()
        return integral
    
    def get_R_Z(self, nphi):
        """Return R and Z coordinates at nphi points along the periodic field line.
        
        The returned arrays will include the repeated point, i.e. the last point will be identical to the first point.

        Args:
            nphi: Number of points to return.

        Returns:
            R: Array of R coordinates.
            Z: Array of Z coordinates.
        """
        return _integrate_field_line_cyl(
            self.field,
            self.R0,
            self.Z0,
            self.Delta_phi_to_close, 
            tol=self.follow_tol,
            phi0=self.phi0, 
            nphi=nphi,
        )

    def get_R_Z_at_phi(self, phi):
        """Get R and Z at a specified phi value."""

        # scipy requires that the last point _exactly_ matches the first point.
        R_for_spline = np.concatenate((self.R[:-1], [self.R[0]]))
        z_for_spline = np.concatenate((self.z[:-1], [self.z[0]]))
        R_spline = CubicSpline(self.phi, R_for_spline, bc_type='periodic')
        R = R_spline(phi)
        z_spline = CubicSpline(self.phi, z_for_spline, bc_type='periodic')
        z = z_spline(phi)
        return R, z

