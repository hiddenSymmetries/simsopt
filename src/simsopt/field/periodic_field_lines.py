import numpy as np
from scipy.optimize import root
import simsoptpp as sopp
from .tracing import ToroidalTransitStoppingCriterion, IterationStoppingCriterion

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


def find_periodic_field_line(
    field,
    R0,
    Z0,
    nfp,
    m,
    method="2D",
    follow_tol=1e-10,
    solve_tol=1e-6,
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
        follow_tol: Tolerance for the field line integration (default: 1e-10).
        solve_tol: Tolerance for the root finding (default: 1e-6).
    """

    x0 = [R0, Z0]
    Delta_phi = m * 2 * np.pi / nfp

    def func(x):
        print("  Evaluating x =", x)
        R, Z = _integrate_field_line(field, x[0], x[1], Delta_phi, follow_tol)
        return np.array([R - x[0], Z - x[1]])

    sol = root(func, x0)
    print(sol)
    return sol.x[0], sol.x[1]