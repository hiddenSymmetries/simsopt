import numpy as np
from scipy.optimize import root
import simsoptpp as sopp
from .tracing import ToroidalTransitStoppingCriterion, IterationStoppingCriterion
from ..util.spectral_diff_matrix import spectral_diff_matrix
from .magneticfield import MagneticField

__all__ = ["find_periodic_field_line"]

def _integrate_field_line(field, R0, Z0, Delta_phi, tol=1e-10, phi0=0):
    """Integrate a single field line in the toroidal direction.

    Integration is done in cylindrical coordinates.

    For this function, phi0 and Delta_phi range over [0, 2pi], not [0, 1].

    Args:
        field: The magnetic field object.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        Delta_phi: Distance in radians to integrate in the toroidal direction.
        tol: Tolerance for the integration (default: 1e-10).
        phi0: Initial toroidal angle (default: 0).

    Returns:
        R: Final R coordinate.
        Z: Final Z coordinate.
    """

    xyz_inits = np.zeros(3)
    xyz_inits[0] = R0 * np.cos(phi0)
    xyz_inits[1] = R0 * np.sin(phi0)
    xyz_inits[2] = Z0
    tmax = 10000.0
    res_ty, res_phi_hit = sopp.fieldline_tracing(
        field,
        xyz_inits,
        tmax,
        tol,
        phis=[phi0 - Delta_phi, phi0 + Delta_phi],  # Try both positive and negative, since we don't know which way B points.
        stopping_criteria=[
            ToroidalTransitStoppingCriterion(Delta_phi / (2 * np.pi), False),
            IterationStoppingCriterion(1000),
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

    print("Last point in tracing:", res_ty[-1])

    if len(res_phi_hit) < 1:
        print("res_ty", res_ty)
        print("res_phi_hit", res_phi_hit)
        raise RuntimeError("When tracing field line, no hits to the expected phi value were found.")
    
    found_hit = False
    for j in range(len(res_phi_hit)):
        t, idx, x, y, z = res_phi_hit[j]
        if t > 0 and idx >= 0:
            chosen_res_phi_hit = res_phi_hit[j]
            found_hit = True

    if not found_hit:
        print("res_phi_hit", res_phi_hit)
        raise RuntimeError("When tracing field line, no hits to the expected phi value were found.")

    if len(res_phi_hit) > 2:
        print("Warning!!! More than 2 hits were found.")
        print("res_phi_hit", res_phi_hit)

    R = np.sqrt(chosen_res_phi_hit[2] ** 2 + chosen_res_phi_hit[3] ** 2)
    Z = chosen_res_phi_hit[4]

    return R, Z


def _find_periodic_field_line_2D(
    field,
    R0,
    Z0,
    nfp,
    m,
    solve_tol=1e-6,
    follow_tol=1e-10,
):
    """Find a periodic field line.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    Args:
        field: The magnetic field object.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        solve_tol: Tolerance for the root finding (default: 1e-6).
        follow_tol: Tolerance for the field line integration (default: 1e-10).
    """

    x0 = [R0, Z0]
    Delta_phi = m * 2 * np.pi / nfp

    def func(x):
        print("  Evaluating x =", x)
        R, Z = _integrate_field_line(field, x[0], x[1], Delta_phi, follow_tol)
        return np.array([R - x[0], Z - x[1]])

    sol = root(func, x0, tol=solve_tol)
    print(sol)
    return sol.x[0], sol.x[1]


def pseudospectral_residual(x, n, D, phi, field):
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
    # print("pseudospectral_residual x =", x)
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


def _find_periodic_field_line_pseudospectral(
    field,
    R0,
    Z0,
    nfp,
    m,
    solve_tol=1e-6,
    nphi=21,
):
    """Find a periodic field line.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    Args:
        field: The magnetic field object.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        solve_tol: Tolerance for the root finding (default: 1e-6).
        nphi: Number of points in the toroidal direction (default: 21).
    """

    # Make sure nphi is odd
    if nphi % 2 == 0:
        nphi += 1

    x0 = [R0, Z0]
    phimax = m * 2 * np.pi / nfp
    dphi = phimax / nphi
    phi = np.linspace(0, phimax, nphi, endpoint=False)
    assert np.abs(phi[1] - phi[0] - dphi) < 1.0e-13

    D = spectral_diff_matrix(nphi, xmin=0, xmax=phimax)
        
    # Initial condition is a circle:
    state = np.concatenate(
        [
            np.full(nphi, R0), 
            np.full(nphi, Z0)
        ]
    )
    sol = root(
        pseudospectral_residual, 
        state,
        tol=solve_tol,
        args=(nphi, D, phi, field),
        # jac=jacobian,
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
    R0,
    Z0,
    nfp,
    m,
    method="2D",
    solve_tol=1e-6,
    follow_tol=1e-10,
    nphi=21,
):
    """Find a periodic field line.
    
    The argument ``m`` is typically the number of times the field line appears
    in a cross-section.

    Args:
        field: The magnetic field object.
        R0: Initial R coordinate.
        Z0: Initial Z coordinate.
        nfp: Number of field periods.
        m: Number of field periods over which the field line is periodic.
        method: Method for finding the periodic field line (default: "2D").
        solve_tol: Tolerance for the root finding (default: 1e-6).
        follow_tol: Tolerance for the field line integration (default: 1e-10). Matters only for method="2D".
        nphi: Number of points in the toroidal direction (default: 21). Matters only for method="pseudospectral".
    """

    if method == "2D":
        return _find_periodic_field_line_2D(
            field, R0, Z0, nfp, m, solve_tol, follow_tol
        )
    elif method == "pseudospectral":
        return _find_periodic_field_line_pseudospectral(
            field, R0, Z0, nfp, m, solve_tol, nphi
        )
    else:
        raise ValueError(f"Unknown method: {method}")
