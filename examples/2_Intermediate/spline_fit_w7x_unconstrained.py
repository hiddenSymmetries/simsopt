#!/usr/bin/env python
"""
Fit a SurfaceBSpline (pseudo-axis + NURBS cross-section representation)
to the boundary of a target VMEC equilibrium (W7-X), using the exact
(Newton-solved nearest-point) shape-error metric
(simsopt.objectives.shape_errors.exact_shape_error) as the objective.

Unlike spline_fit_w7x.py, this drops SurfaceBSpline's own linear
inequality constraints (write_inequality_constraints, which enforce
ordering relationships like theta_k <= theta_{k+1} that keep a cross
section from folding over on itself) entirely -- there is no
ConstrainedProblem/constrained_mpi_solve here, just a plain
LeastSquaresProblem solved with least_squares_mpi_solve. Instead, this
relies on the dofs' own per-dof box bounds (the defaults set by
CrossSectionFixedZeta/PseudoAxis's own construction, left untouched here
rather than widened to +-inf as in spline_fit_w7x.py) to keep the
optimizer in a reasonable region. Box bounds can't express the ordering
relationships the linear constraints do, so this is a strictly weaker
guardrail against a folded-over cross section -- the final bounds check
at the end of this script reports whether the box bounds were actually
respected (scipy's bounded least_squares should never violate them, but
it's worth confirming rather than assuming).

No VMEC equilibrium solve is needed for either surface -- the target's
boundary Fourier coefficients are read directly from its input file, and
the spline's own Fourier coefficients come from SurfaceBSpline.to_RZFourier().
This is a purely geometric shape-fitting problem.

Runtime: with dozens of dofs and forward-difference gradients, each
finite-difference Jacobian costs ~ndofs+1 shape-error evaluations --
each of those does one Newton solve per reference point (see
exact_shape_error), so runtime scales with n_cross_sections * ntheta_ref
below; run this with mpirun -n <nprocs> to parallelize those evaluations
across ranks (least_squares_mpi_solve distributes Jacobian columns across
the MPI pool), e.g. `mpirun -n 8 python spline_fit_w7x_unconstrained.py`.
W7-X is a strongly-shaped stellarator, so don't expect a tight match at a
coarse spline resolution -- this demonstrates the fitting pipeline, not a
high-fidelity reconstruction.
"""

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

TARGET_FILE = (
    "/Users/issraali/codes/simsopt/tests/test_files/"
    "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"
)

mpi = MpiPartition()
mpi.write()

proc0_print("Running 2_Intermediate/spline_fit_w7x_unconstrained.py")
proc0_print("==================================================")

spline_kwargs = {
    "axis_points": 3,
    "points_per_cs": 6,
    "n_cs": 5,
    "nfp": 5,
    "M": 12,
    "N": 12,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": True,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": True,
    "cs_basis": "polar",
    "nurbs": False,
    "use_bishop_frame": True,
}


def plot_cross_section_comparison(target_surf, spline_surf, title, n_cuts=4):
    """
    Compare the spline's own cross sections (not a re-fit Fourier
    approximation of them -- the actual spline geometry, via Surface's
    generic cross_section, which uses SurfaceBSpline.gamma_lin directly)
    against the target's, at a few toroidal cuts across one field period.
    """
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
            "k-", lw=2, label="target (W7-X)",
        )
        ax.plot(
            np.append(spline_r, spline_r[0]),
            np.append(spline_pts[:, 2], spline_pts[0, 2]),
            "r--", lw=2, label="optimized spline",
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
    """
    Flattened exact shape error between spline_surf's current Fourier fit
    and the fixed target reference: for each of the reference's fixed
    toroidal cross sections, a Newton solve (using the Fourier surface's
    own analytic theta derivatives -- see exact_shape_error) finds the
    nearest point on that cross section to each reference point.
    spec_cond=None skips to_RZFourier's spectral condensation entirely
    (not needed here -- the objective only cares about matching the
    shape, not producing a spectrally-optimal fit for downstream VMEC
    use), which matters since this gets called ~ndofs times per
    finite-difference Jacobian. Used directly for reporting (max/mean
    shape error), and wrapped via make_optimizable below to become the
    LeastSquaresProblem's funcs_in.
    """
    rz_surf = spline_surf.to_RZFourier(spec_cond=None)
    return exact_shape_error(rz_surf, reference).flatten()


# Target boundary: read directly from the VMEC input file's Fourier
# coefficients -- no equilibrium solve needed, just the boundary shape.
target_vmec = Vmec(TARGET_FILE, verbose=False)
target_surf = target_vmec.boundary
R0_orig = abs(target_surf.get_rc(0, 0))
a0_orig = abs(target_surf.get_rc(0, 1))
proc0_print(
    f"Target original: R0={target_surf.get_rc(0, 0):.4f}, a0~{a0_orig:.4f}"
)

# Rescale the whole target uniformly (every Rmn, Zmn coefficient by the
# same factor) so its minor radius is exactly 1 m -- R,Z are linear in
# the Fourier coefficients, so this scales the entire shape rigidly,
# preserving aspect ratio and all QS properties.
scale = 1.0 / R0_orig
target_surf.rc[:, :] *= scale
target_surf.zs[:, :] *= scale
if not target_surf.stellsym:
    target_surf.rs[:, :] *= scale
    target_surf.zc[:, :] *= scale

# target_surf.plot()

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

# Build the spline surface, initialized to roughly the target's physical
# scale (major/minor radius) rather than the class's tiny unit-scale
# default, so the optimizer starts from a sane shape.
spline_surf = SurfaceBSpline(default_r=0.3, **spline_kwargs)
# for i in range(spline_kwargs["axis_points"]):
#     # PseudoAxis's default r bounds ([0.3, 2.5]) assume its own unit-scale
#     # default (r_axis=1) -- widen them before setting r_axis to W7-X's
#     # actual major radius, or the solver rejects the initial guess outright.
#     spline_surf.axis.set_lower_bound(f"r_axis_{i}", 0.5 * R0)
#     spline_surf.axis.set_upper_bound(f"r_axis_{i}", 1.5 * R0)
#     spline_surf.axis.set(f"r_axis_{i}", R0)

proc0_print(f"spline_surf.dof_names: {spline_surf.dof_names}")
proc0_print(f"ndofs: {len(spline_surf.x)}")

# Unlike spline_fit_w7x.py, the dofs' own per-dof box bounds (set by
# CrossSectionFixedZeta/PseudoAxis's own construction) are left in place
# here rather than widened to +-inf -- there are no linear inequality
# constraints in this script, so these box bounds are the only guardrail
# keeping the optimizer in a reasonable region.

initial_residuals = spline_shape_residuals(spline_surf, reference)
proc0_print(f"Initial max shape error: {np.max(initial_residuals):.4e}")
proc0_print(f"Initial mean shape error: {np.mean(initial_residuals):.4e}")

# Only rank 0 plots -- every rank would otherwise open its own figure
# under mpirun.
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

# Box bounds are supposed to be respected exactly by scipy's bounded
# least_squares, but that relies on prob.bounds having been threaded
# through correctly -- check explicitly rather than assuming.
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
proc0_print("End of 2_Intermediate/spline_fit_w7x_unconstrained.py")
proc0_print("=================================================")
