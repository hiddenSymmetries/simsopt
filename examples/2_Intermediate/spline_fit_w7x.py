#!/usr/bin/env python
"""
Fit a SurfaceBSpline to the W7-X boundary by least squares on the exact shape error.

The target is read from tests/test_files/input.W7-X_without_coil_ripple_beta0p05_d23p4_tm
and rescaled to R0 = 1; no VMEC solve is needed. The dofs' box bounds are the only
constraints. Run with e.g. `mpirun -n 8 python spline_fit_w7x.py` to spread the
finite-difference Jacobian across ranks.
"""

from pathlib import Path

import matplotlib

matplotlib.use("qtagg")
import matplotlib.pyplot as plt
import numpy as np
from simsopt._core import make_optimizable
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.objectives.shape_errors import (
    build_exact_shape_reference,
    exact_shape_error,
)
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
TARGET_FILE = str(TEST_DIR / "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm")

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/spline_fit_w7x.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 6,
    "n_cs": 5,
    "nfp": 5,
    "M": 6,
    "N": 6,
    "p_u": 3,
    "p_v": 3,
    "toroidal_equispaced": True,
    "poloidal_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": True,
    "nurbs": False,
    "use_bishop_frame": True,
}


def plot_cross_section_comparison(target_surf, spline_surf, title, n_cuts=4):
    """Plot spline and target cross sections at n_cuts angles across one field period."""
    phi_fracs = np.linspace(0, 1.0 / target_surf.nfp, n_cuts, endpoint=False)

    fig, axes = plt.subplots(1, n_cuts, figsize=(4 * n_cuts, 4.5))
    for ax, phi in zip(axes, phi_fracs):
        target_pts = target_surf.cross_section(phi, thetas=200)
        spline_pts = spline_surf.cross_section(phi, thetas=200)

        target_r = np.hypot(target_pts[:, 0], target_pts[:, 1])
        spline_r = np.hypot(spline_pts[:, 0], spline_pts[:, 1])

        ax.plot(
            np.append(target_r, target_r[0]),
            np.append(target_pts[:, 2], target_pts[0, 2]),
            "k-",
            lw=2,
            label="target (W7-X)",
        )
        ax.plot(
            np.append(spline_r, spline_r[0]),
            np.append(spline_pts[:, 2], spline_pts[0, 2]),
            "r--",
            lw=2,
            label="optimized spline",
        )
        ax.set_title(rf"$\zeta$ = {phi * 2 * np.pi:.3f} rad")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def spline_shape_residuals(spline_surf, reference):
    """Exact shape error of the spline's Fourier fit (no spectral condensation, for speed)."""
    rz_surf = spline_surf.to_RZFourier(spec_cond=None)
    return exact_shape_error(rz_surf, reference).flatten()


# Target boundary from the VMEC input file; no equilibrium solve needed.
target_vmec = Vmec(TARGET_FILE, verbose=False)
target_surf = target_vmec.boundary
R0_orig = abs(target_surf.get_rc(0, 0))
a0_orig = abs(target_surf.get_rc(1, 0))
proc0_print(
    f"Target original: R0={target_surf.get_rc(0, 0):.4f}, a0~{a0_orig:.4f}"
)

# Rescale the target uniformly to major radius 1.
scale = 1.0 / R0_orig
target_surf.rc[:, :] *= scale
target_surf.zs[:, :] *= scale
if not target_surf.stellsym:
    target_surf.rs[:, :] *= scale
    target_surf.zc[:, :] *= scale

R0 = target_surf.get_rc(0, 0)
a0 = abs(target_surf.get_rc(1, 0))
proc0_print(
    f"Target rescaled (x{scale:.4f}): R0={R0:.4f}, a0~{a0:.4f}, "
    f"nfp={target_surf.nfp}, mpol={target_surf.mpol}, ntor={target_surf.ntor}"
)

n_cross_sections = 2 * 12 * target_surf.nfp + 2
ntheta_ref = 2 * 12 + 1
phi_1d = np.linspace(0, 2 * np.pi, n_cross_sections, endpoint=False)
reference = build_exact_shape_reference(target_surf, phi_1d, ntheta=ntheta_ref)

spline_surf = SurfaceBSpline(default_r=0.3, **spline_kwargs)

proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")
proc0_print(f"ndofs: {len(spline_surf.x)}")

initial_residuals = spline_shape_residuals(spline_surf, reference)
proc0_print(f"Initial max shape error: {np.max(initial_residuals):.4e}")
proc0_print(f"Initial mean shape error: {np.mean(initial_residuals):.4e}")

# Only rank 0 plots.
if mpi.proc0_world:
    plot_cross_section_comparison(
        target_surf, spline_surf, "Initial spline vs. target cross sections"
    )

shape_obj = make_optimizable(spline_shape_residuals, spline_surf, reference)
prob = LeastSquaresProblem(goals=0, weights=1, funcs_in=[shape_obj.J])

proc0_print("Beginning optimization")
try:
    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        abs_step=1e-6,
    )
except Exception as e:
    proc0_print(f"Optimization raised: {e}")

final_residuals = spline_shape_residuals(spline_surf, reference)
proc0_print("")
proc0_print(f"Final max shape error: {np.max(final_residuals):.4e}")
proc0_print(f"Final mean shape error: {np.mean(final_residuals):.4e}")

proc0_print("")
proc0_print("Spline dofs:")
spline_dofs = repr(np.array(spline_surf.x))
proc0_print(spline_dofs)

# Confirm the solver respected the box bounds.
lb, ub = spline_surf.bounds
x = np.asarray(spline_surf.x)
below = x < lb
above = x > ub
proc0_print("")
if np.any(below) or np.any(above):
    proc0_print("Bounds violation detected:")
    for name, xi, lbi, ubi, is_below, is_above in zip(
        spline_surf.dof_names, x, lb, ub, below, above
    ):
        if is_below or is_above:
            proc0_print(f"  {name}: x={xi:.6e} not in [{lbi:.6e}, {ubi:.6e}]")
else:
    proc0_print("No bounds violations.")

if mpi.proc0_world:
    plot_cross_section_comparison(
        target_surf, spline_surf, "Optimized spline vs. target cross sections"
    )

proc0_print("")
proc0_print("End of 2_Intermediate/spline_fit_w7x.py")
proc0_print("=================================================")
