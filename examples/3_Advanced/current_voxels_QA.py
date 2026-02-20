#!/usr/bin/env python
r"""
This example uses the current voxels method
outlined in Kaptanoglu, Langlois & Landreman 2024 in order
to make finite-build coils with the voxels method:

Kaptanoglu, A. A., Langlois, G. P., & Landreman, M. (2024).
Topology optimization for inverse magnetostatics as sparse regression:
Application to electromagnetic coils for stellarators.
Computer Methods in Applied Mechanics and Engineering, 418, 116504.

The script performs the following steps:

1. **Voxel optimization**: Discretizes the current into a grid of voxels between
   inner and outer toroidal surfaces, then optimizes the voxel currents via
   relax-and-split with group sparsity (L0) to match the target B·n on the
   plasma boundary while enforcing divergence-free current.

2. **Filament extraction**: Extracts a single helical filament curve from the
   optimized voxel solution using ``make_filament_from_voxels``, which fits
   the nonzero voxel centers to a ``CurveXYZFourierSymmetries`` representation.

3. **Filamentary coil optimization**: Optimizes the extracted filament to
   minimize squared flux on the plasma surface, with penalties on length,
   curvature, mean-squared curvature, and coil-to-surface distance. The
   filament is treated as one helical coil wrapping the full torus (not
   expanded via symmetries).

The script should be run as:
    mpirun -n 1 python current_voxels_QA.py
or
    python current_voxels_QA.py
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from simsopt.geo import SurfaceRZFourier, CurrentVoxelsGrid
from simsopt.objectives import SquaredFlux
from simsopt.field import (
    Current,
    BiotSavart,
    CurrentVoxelsField,
    RegularizedCoil,
    coils_to_vtk,
)
from simsopt.solve import relax_and_split_minres
from simsopt.util import (
    make_filament_from_voxels,
    in_github_actions,
)
import time


# perform the filament optimization
def perform_filament_optimization(
    s: "SurfaceRZFourier",
    bs: "BiotSavart",
    coils: list,
    out_dir: str,
    s_full: "SurfaceRZFourier" = None,
    **kwargs,
) -> None:
    from scipy.optimize import minimize
    from simsopt.geo import (
        CurveLength,
        CurveCurveDistance,
        MeanSquaredCurvature,
        LpCurveCurvature,
        CurveSurfaceDistance,
    )
    from simsopt.objectives import Weight, QuadraticPenalty

    out_dir = kwargs.pop("out_dir", out_dir)
    curves = [c.curve for c in coils]

    LENGTH_WEIGHT = Weight(kwargs.pop("length_weight", 1e-5))
    CS_THRESHOLD = kwargs.pop("cs_threshold", 0.1)
    CS_WEIGHT = kwargs.pop("cs_weight", 1e-12)
    CC_THRESHOLD = kwargs.pop("cc_threshold", 0.1)
    CC_WEIGHT = kwargs.pop("cc_weight", 1e-12)
    CURVATURE_THRESHOLD = kwargs.pop("curvature_threshold", 0.1)
    CURVATURE_WEIGHT = kwargs.pop("curvature_weight", 1e-8)
    MSC_THRESHOLD = kwargs.pop("msc_threshold", 0.1)
    MSC_WEIGHT = kwargs.pop("msc_weight", 1e-8)

    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(curve) for curve in curves]
    if len(curves) > 1:
        Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(curve, 2, CURVATURE_THRESHOLD) for curve in curves]
    Jmscs = [MeanSquaredCurvature(curve) for curve in curves]

    JF = (
        Jf
        + LENGTH_WEIGHT * sum(Jls)
        + CS_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
    )

    if len(curves) > 1:
        JF += CC_WEIGHT * Jccdist

    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(
            np.abs(np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2))
        )
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        if len(curves) > 1:
            outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        valuestr = f"J={J:.2e}, Jf={jf:.2e}"
        valuestr += f", LenObj={(LENGTH_WEIGHT * sum(Jls)).J():.2e}"
        valuestr += f", C-S-SepObj={(CS_WEIGHT * Jcsdist).J():.2e}"
        valuestr += f", CurvatureObj={(CURVATURE_WEIGHT * sum(Jcs)).J():.2e}"
        valuestr += f", MeanSquaredCurvatureObj={(MSC_WEIGHT * sum(Jmscs)).J():.2e}"
        if len(curves) > 1:
            valuestr += f", C-C-SepObj={(CC_WEIGHT * Jccdist).J():.2e}"
        valuestr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(valuestr)
        return J, grad

    bs.set_points(s.gamma().reshape((-1, 3)))
    MAXITER = kwargs.pop("max_iter", 500)
    _ = minimize(
        fun,
        JF.x,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAXITER, "maxcor": 200},
        tol=1e-15,
    )

    bs.set_points(s.gamma().reshape((-1, 3)))
    coils_to_vtk(coils, out_dir + "curves_opt")
    nphi_half, ntheta_half = s.gamma().shape[0], s.gamma().shape[1]
    Bn_half = np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)
    pointData = {"B_N": Bn_half.reshape((1, nphi_half, ntheta_half))}
    s.to_vtk(out_dir + "surf_opt", extra_data=pointData)
    if s_full is not None:
        bs.set_points(s_full.gamma().reshape((-1, 3)))
        Bn_filament = np.sum(
            bs.B().reshape((s_full.gamma().shape[0], s_full.gamma().shape[1], 3))
            * s_full.unitnormal(),
            axis=2,
        )
        pointData = {"B_N": Bn_filament.reshape((1, Bn_filament.shape[0], Bn_filament.shape[1]))}
        s_full.to_vtk(out_dir + "surf_opt_full", extra_data=pointData)
    bs.save(out_dir + "biot_savart_opt.json")


t_start = time.time()

# Set surface parameters
nphi = 32
ntheta = nphi
input_name = "input.LandremanPaul2021_QA"

if in_github_actions:
    max_iter = 20
    rs_max_iter = 10
    nphi = 16
    ntheta = nphi
    Nx = 10
else:
    max_iter = 20
    rs_max_iter = 20
    Nx = 20

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)

qphi = s.nfp * nphi * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

out_dir = "current_voxels_QA/"
os.makedirs(out_dir, exist_ok=True)

poff = 0.5
coff = 0.3
s_inner = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)
s_outer = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="half period", nphi=nphi, ntheta=ntheta
)

s_inner.extend_via_projected_normal(poff)
s_outer.extend_via_projected_normal(poff + coff)

kwargs = {"Nx": Nx, "Ny": Nx, "Nz": Nx}
current_voxels_grid = CurrentVoxelsGrid(s, s_inner, s_outer, **kwargs)

reg_param = 0.01
coils_to_vtk(
    [
        RegularizedCoil(
            current_voxels_grid.Itarget_curve,
            Current(current_voxels_grid.Itarget),
            reg_param,
        )
    ],
    out_dir + "Itarget_curve",
)
current_voxels_grid.to_vtk_before_solve(out_dir + "grid_before_solve_Nx" + str(Nx))

# Optimize the voxels
kappa = 1e-3
kwargs = {"out_dir": out_dir, "kappa": kappa, "max_iter": 5000, "precondition": True}
minres_dict = relax_and_split_minres(current_voxels_grid, **kwargs)

nu = 1e2
l0_threshold = 1e4
l0_thresholds = np.linspace(l0_threshold, 100 * l0_threshold, 5, endpoint=True)
kwargs["alpha0"] = minres_dict["alpha_opt"]
kwargs["l0_thresholds"] = l0_thresholds
kwargs["nu"] = nu
kwargs["max_iter"] = max_iter
kwargs["rs_max_iter"] = rs_max_iter
minres_dict = relax_and_split_minres(current_voxels_grid, **kwargs)

current_voxels_grid.to_vtk_after_solve(
    out_dir + "grid_after_Tikhonov_solve_Nx" + str(Nx),
    quiver_savepath=out_dir + "quiver_J_sparse.png",
)

# Set up CurrentVoxels Bfield
bs_current_voxels = CurrentVoxelsField(
    current_voxels_grid.J,
    current_voxels_grid.XYZ_integration,
    current_voxels_grid.grid_scaling,
    nfp=s.nfp,
    stellsym=s.stellsym,
)
bs_current_voxels.set_points(s.gamma().reshape((-1, 3)))

# Save Bnormal plots from the voxel fields
bs_current_voxels.set_points(s_plot.gamma().reshape((-1, 3)))
Bn_voxels = np.sum(
    bs_current_voxels.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2
)
pointData = {"B_N": Bn_voxels.reshape((1, qphi, ntheta))}
s_plot.to_vtk(out_dir + "surf_voxels_full", extra_data=pointData)

# Optimization progress figure
fB = minres_dict["fB"]
fK = minres_dict["fK"]
fC = minres_dict["fC"]
fI = minres_dict["fI"]
fRS = minres_dict["fRS"]
fminres = minres_dict["fminres"]
fig1 = plt.figure(figsize=(10, 6))
plt.semilogy(fB, "r", label=r"$f_B$ (Bnormal matching)")
plt.semilogy(fK, "b", label=r"$\kappa \|\alpha\|^2$ (Tikhonov)")
plt.semilogy(fC, "c", label=r"$f_C$ (divergence-free)")
plt.semilogy(fI, "m", label=r"$\sigma f_I$ (Itarget)")
plt.semilogy(fminres, "k--", label="MINRES residual")
if l0_thresholds[-1] > 0:
    fRS = np.abs(fRS)
    fRS[fRS < 1e-20] = 1e-20
    plt.semilogy(fRS, "g", label=r"$\nu^{-1}\|\alpha{-}w\|^2$ (relax-and-split)")
plt.xlabel("Relax-and-split iteration")
plt.ylabel("Loss term value")
plt.title("Current voxels optimization progress")
plt.grid(True)
plt.legend(loc="best", fontsize=10, ncol=2)
plt.ylim(1e-10, 1e5)
plt.tight_layout()
plt.savefig(out_dir + "optimization_progress.png", dpi=150)

# Initialize filament from voxels
filament_curve = make_filament_from_voxels(
    current_voxels_grid, l0_thresholds[-1], num_fourier=30
)
current = Current(current_voxels_grid.Itarget / 2.0)
current.fix_all()
coils = [RegularizedCoil(filament_curve, current, reg_param)]
coils_to_vtk(coils, out_dir + "extracted_filament_curve")
bs = BiotSavart(coils)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bn_initial = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
pointData = {"B_N": Bn_initial.reshape((1, qphi, ntheta))}
s_plot.to_vtk(out_dir + "surf_filament_initial", extra_data=pointData)

perform_filament_optimization(s, bs, coils, out_dir=out_dir, s_full=s_plot)
plt.show()
