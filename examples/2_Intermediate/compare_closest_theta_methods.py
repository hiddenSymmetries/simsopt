#!/usr/bin/env python
"""
Compare the three closest-point-in-theta solvers behind
simsopt.objectives.shape_errors.exact_shape_error:

  - "newton": simsopt.objectives.shape_errors._closest_theta_newton --
    scipy.optimize.newton, supplied with the Fourier surface's own
    analytic theta derivatives as func/fprime, vectorized across every
    reference point at once. This is the method exact_shape_error uses
    by default.
  - "newton_manual": simsopt.objectives.shape_errors._closest_theta_newton_manual
    -- the same math (same stationarity condition
    d/dtheta[(R-R0)^2 + (Z-Z0)^2] = 0, same analytic derivatives), but
    with the Newton iteration hand-rolled as a plain Python for-loop
    instead of delegated to scipy.optimize.newton. Profiled here to see
    what (if anything) delegating to scipy costs or saves versus the
    original hand-rolled loop.
  - "scipy": simsopt.objectives.shape_errors._closest_theta_scipy --
    scipy.optimize.minimize_scalar (derivative-free bounded Brent
    search), called once per reference point in a plain Python loop.

All three solve the same problem -- for a fixed toroidal angle, find the
poloidal angle on a SurfaceRZFourier's cross section closest to a given
target point -- so this checks that they agree (scipy's derivative-free
minimize_scalar serves as an independent check on the two derivative-
based Newton methods) and compares their wall-clock cost.

The "candidate" surface here is a low-resolution truncation of a
higher-resolution "reference" surface (the same setup exact_shape_error
is built for -- see figures_spline_paper/fourier_convergence), so the
closest points are genuinely off-grid, not trivially coincident.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives.shape_errors import (
    build_exact_shape_reference,
    exact_shape_error,
)

print("Running 2_Intermediate/compare_closest_theta_methods.py")
print("==================================================")

NFP = 3
M_REF, N_REF = 12, 12
M_CAND, N_CAND = 4, 4
N_CROSS_SECTIONS = 8
NTHETA_REF = 200

METHODS = [
    ("newton", "scipy.optimize.newton\n(analytic, vectorized)", "C0"),
    ("newton_manual", "hand-rolled Newton\n(analytic, vectorized)", "C1"),
    ("scipy", "minimize_scalar\n(derivative-free, looped)", "C2"),
]

# Reference surface: a non-trivial (elongated, rotating, sheared) closed
# shape -- not simply a circle -- so the candidate's closest points to it
# are genuinely off that candidate's own quadrature grid.
reference_surf = SurfaceRZFourier(
    nfp=NFP, mpol=M_REF, ntor=N_REF, stellsym=True
)
reference_surf.make_rotating_ellipse(1.0, 0.3, 1.8, 0.2)

# Candidate surface: reference_surf truncated to a much lower mode
# number -- a coarse approximation of the same shape, exactly the
# situation exact_shape_error is used for in a Fourier-mode convergence
# study.
candidate_surf = SurfaceRZFourier(
    nfp=NFP, mpol=M_CAND, ntor=N_CAND, stellsym=True
)
for m in range(M_CAND + 1):
    for n in range(-N_CAND, N_CAND + 1):
        if m == 0 and n < 0:
            continue
        candidate_surf.set_rc(m, n, reference_surf.get_rc(m, n))
        candidate_surf.set_zs(m, n, reference_surf.get_zs(m, n))

phi_1d = np.linspace(0, 2 * np.pi, N_CROSS_SECTIONS, endpoint=False)
reference = build_exact_shape_reference(
    reference_surf, phi_1d, ntheta=NTHETA_REF
)
n_points = N_CROSS_SECTIONS * NTHETA_REF
print(f"Comparing over {n_points} reference points "
      f"({N_CROSS_SECTIONS} cross sections x {NTHETA_REF} points each)")

times = {}
errors = {}
for key, label, color in METHODS:
    t0 = time.perf_counter()
    errors[key] = exact_shape_error(candidate_surf, reference, method=key)
    times[key] = time.perf_counter() - t0
    print(f"{key:14s}: {times[key]:.4f} s total, "
          f"{1e6 * times[key] / n_points:.2f} us/point")

baseline_key = "newton"
print(f"speedup vs {baseline_key} "
      f"(other method time / {baseline_key} time):")
for key, _, _ in METHODS:
    if key == baseline_key:
        continue
    print(f"  {key}: {times[key] / times[baseline_key]:.1f}x")
    diff = (errors[key] - errors[baseline_key]).flatten()
    print(f"    max |{key} - {baseline_key}| distance difference: "
          f"{np.max(np.abs(diff)):.3e}")
    print(f"    rms |{key} - {baseline_key}| distance difference: "
          f"{np.sqrt(np.mean(diff**2)):.3e}")

########################
# plotting

fig, (ax_time, ax_diff) = plt.subplots(1, 2, figsize=(10, 4))

labels = [label for _, label, _ in METHODS]
colors = [color for _, _, color in METHODS]
bar_times = [times[key] for key, _, _ in METHODS]
ax_time.bar(labels, bar_times, color=colors)
ax_time.set_ylabel("Wall-clock time [s]")
ax_time.set_yscale("log")
ax_time.set_title(f"Total time over {n_points} points")
for i, t in enumerate(bar_times):
    ax_time.text(i, t, f"{t:.3f} s", ha="center", va="bottom")
ax_time.tick_params(axis="x", labelsize=8)

for key, label, color in METHODS:
    if key == baseline_key:
        continue
    diff = (errors[key] - errors[baseline_key]).flatten()
    # newton_manual matches newton bit-for-bit (same math, both
    # derivative-based and run to convergence) -- an all-zero-variance
    # histogram has no meaningful range to show (and breaks matplotlib's
    # auto-binning for the one dataset that does), so it's noted as a
    # title annotation instead of overlaid here.
    if np.all(diff == 0):
        continue
    ax_diff.hist(
        diff, bins=50, color=color, alpha=0.6,
        label=f"{key} $-$ {baseline_key}",
    )
ax_diff.set_xlabel("Distance difference vs. newton")
ax_diff.set_ylabel("Count")
ax_diff.set_title(
    "Agreement with the default method\n"
    "(newton_manual matches newton exactly, not shown)"
)
ax_diff.legend(fontsize=8)

fig.tight_layout()
plt.savefig("compare_closest_theta_methods.png", dpi=150)
plt.show()

print("")
print("End of 2_Intermediate/compare_closest_theta_methods.py")
print("=================================================")
