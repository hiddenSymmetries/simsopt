#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from simsopt.configs import get_data
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, FiniteBetaBoozerSurface, Volume
from simsopt.geo.surfaceobjectives import surface_field_nonquasisymmetric_ratio
from simsopt.mhd import Vmec, VirtualCasing
from simsopt.util import in_github_actions

r"""
This example is the finite-beta counterpart to boozerQA.py.

The main workflow is a single-surface prescribed-field QA optimization.  The
interior and exterior magnetic fields are specified on the surface, and
FiniteBetaBoozerSurface solves the finite-beta Boozer residual system on that
surface without any ideal-MHD solver in the optimization loop.

An auxiliary VMEC-backed mode is available to generate or validate surface field
data, but it is not the primary optimization path.
"""


OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA_finitebeta.py")
print("==============================================")

legacy_synthetic_mode = os.environ.get("SIMSOPT_FINITE_BETA_SYNTHETIC")
default_mode = "prescribed" if legacy_synthetic_mode in (None, "1") else "vmec"
mode = os.environ.get("SIMSOPT_FINITE_BETA_MODE", default_mode).strip().lower()

nphi_default = 12 if in_github_actions else 24
ntheta_default = 12 if in_github_actions else 24
nphi = int(os.environ.get("SIMSOPT_FINITE_BETA_NPHI", str(nphi_default)))
ntheta = int(os.environ.get("SIMSOPT_FINITE_BETA_NTHETA", str(ntheta_default)))
ls_max_nfev = int(os.environ.get("SIMSOPT_FINITE_BETA_MAX_NFEV", "40"))
qa_maxiter_default = 10 if in_github_actions else 30
qa_maxiter = int(os.environ.get("SIMSOPT_FINITE_BETA_QA_MAXITER", str(qa_maxiter_default)))


def print_blocks(title, result):
    print(title)
    for name, values in result["blocks"].items():
        print(f"{name:>13s} residual norm = {np.linalg.norm(values):.6e}")


def write_residual_vtk(surface, result):
    extra_data = {
        "B_out_normal": result["blocks"]["normal"][:, :, None],
        "pressure_balance": result["blocks"]["pressure"][:, :, None],
        "jump_x": result["blocks"]["jump"][:, :, [0]],
        "jump_y": result["blocks"]["jump"][:, :, [1]],
        "jump_z": result["blocks"]["jump"][:, :, [2]],
    }
    surface.to_vtk(OUT_DIR + "boozerQA_finitebeta_boundary", extra_data=extra_data)


if mode in ("prescribed", "prescribed-fixed", "fixed"):
    print("Using prescribed-field finite-beta test case")
    print(
        f"Mode={mode}, grid: nphi={nphi}, ntheta={ntheta}, "
        f"ls_max_nfev={ls_max_nfev}, qa_maxiter={qa_maxiter}"
    )

    _, _, ma, nfp, _ = get_data("ncsx")
    phis = np.linspace(0, 1 / nfp, nphi, endpoint=False)
    thetas = np.linspace(0, 1, ntheta, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=3,
        ntor=3,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)

    iota_target = -0.31
    G_target = 1.4
    pressure_jump = 0.0
    vol = Volume(surface)
    vol_target = vol.J()

    def B_in(surface):
        tang = surface.gammadash1() + iota_target * surface.gammadash2()
        tang_norm_sq = np.sum(tang**2, axis=2)
        return (G_target / tang_norm_sq)[:, :, None] * tang

    def B_out(surface):
        return B_in(surface)

    B_in0 = B_in(surface)
    B_out0 = B_out(surface)

    boozer_surface = FiniteBetaBoozerSurface(
        None,
        surface,
        vol,
        vol_target,
        pressure_jump=pressure_jump,
        options={'verbose': True, 'ls_max_nfev': ls_max_nfev},
    )

    initial = boozer_surface.residual_blocks(iota=-0.1, G=G_target, I=0.0, B_in=B_in0, B_out=B_out0)
    print_blocks("Initial residual norms:", initial)

    if mode in ("prescribed-fixed", "fixed"):
        res = boozer_surface.run_code(iota=-0.1, G=G_target, I=0.0, B_in=B_in0, B_out=B_out0, optimize_G=False)
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in0,
            B_out=B_out0,
        )
        print_blocks("Solved residual norms:", solved)
        print(
            f"Solved parameters: success={res['success']}, iota={res['iota']:.6e}, "
            f"G={res['G']:.6e}, I={res['I']:.6e}, ||r||={res['residual_norm']:.6e}"
        )

        surface.x = surface.x + 1e-3 * np.random.default_rng(7).standard_normal(surface.x.shape)
        res = boozer_surface.run_code(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in0,
            B_out=B_out0,
            optimize_G=False,
            optimize_surface=True,
        )
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in0,
            B_out=B_out0,
        )
        print_blocks("Analytic fixed-field surface solve residual norms:", solved)
        print(
            f"Fixed-field surface solve: success={res['success']}, iota={res['iota']:.6e}, "
            f"mr={surface.major_radius():.6e}, ||r||={res['residual_norm']:.6e}"
        )
        write_residual_vtk(surface, solved)
        print("Prescribed fixed-field finite-beta solve complete.")

    else:
        res = boozer_surface.run_code(iota=-0.1, G=G_target, I=0.0, B_in=B_in, B_out=B_out, optimize_G=False)
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in,
            B_out=B_out,
        )
        print_blocks("Solved residual norms:", solved)
        print(
            f"Solved parameters: success={res['success']}, iota={res['iota']:.6e}, "
            f"G={res['G']:.6e}, I={res['I']:.6e}, ||r||={res['residual_norm']:.6e}"
        )

        iota0 = res["iota"]
        mr0 = surface.major_radius()

        def fun(dofs):
            surface.x = dofs
            res_local = boozer_surface.run_code(
                iota=boozer_surface.res["iota"],
                G=G_target,
                I=boozer_surface.res["I"],
                current_potential=boozer_surface.res["current_potential"],
                B_in=B_in,
                B_out=B_out,
                optimize_G=False,
            )
            field = B_in(surface)
            J_nonqs = surface_field_nonquasisymmetric_ratio(surface, field)
            J_iota = 0.5 * (res_local["iota"] - iota0)**2
            J_mr = 0.5 * (surface.major_radius() - mr0)**2
            J = J_nonqs + J_iota + J_mr
            print(
                f"J={J:.3e}, J_nonqs={J_nonqs:.3e}, iota={res_local['iota']:.3e}, "
                f"mr={surface.major_radius():.3e}, ||r||={res_local['residual_norm']:.3e}"
            )
            return J

        print("Running prescribed-field finite-beta QA optimization")
        minimize(fun, surface.x, method="L-BFGS-B", options={"maxiter": qa_maxiter})

        res = boozer_surface.run_code(
            iota=boozer_surface.res["iota"],
            G=G_target,
            I=boozer_surface.res["I"],
            current_potential=boozer_surface.res["current_potential"],
            B_in=B_in,
            B_out=B_out,
            optimize_G=False,
        )
        field = B_in(surface)
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in,
            B_out=B_out,
        )
        print(
            f"Final QA ratio={surface_field_nonquasisymmetric_ratio(surface, field):.6e}, "
            f"iota={res['iota']:.6e}, mr={surface.major_radius():.6e}"
        )
        write_residual_vtk(surface, solved)
        print("Prescribed-field finite-beta QA solve complete.")

elif mode == "vmec":
    print("Using auxiliary VMEC-backed finite-beta case")
    test_dir = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
    default_vmec_input = test_dir / "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"
    vmec_filename = Path(os.environ.get("SIMSOPT_FINITE_BETA_VMEC", str(default_vmec_input))).resolve()
    if not vmec_filename.exists():
        raise FileNotFoundError(
            f"VMEC input or wout file not found: {vmec_filename}. Set SIMSOPT_FINITE_BETA_VMEC to a valid path."
        )

    pressure_jump = float(os.environ.get("SIMSOPT_FINITE_BETA_PRESSURE_JUMP", "0.0"))
    print(f"VMEC file: {vmec_filename}")
    print(f"Grid: nphi={nphi}, ntheta={ntheta}, ls_max_nfev={ls_max_nfev}, pressure_jump={pressure_jump}")

    vmec = Vmec(str(vmec_filename))
    vmec.run()

    vc_override = os.environ.get("SIMSOPT_FINITE_BETA_VCASING")
    if vc_override:
        vc_filename = Path(vc_override).resolve()
    else:
        vc_filename = vmec_filename.with_name(vmec_filename.name.replace("input.", "vcasing_") + ".nc")

    if vc_filename.exists():
        print(f"Loading saved virtual casing result from {vc_filename}")
        vc = VirtualCasing.load(str(vc_filename))
    else:
        print("Running virtual casing calculation")
        try:
            vc = VirtualCasing.from_vmec(
                vmec,
                src_nphi=nphi,
                src_ntheta=ntheta,
                trgt_nphi=nphi,
                trgt_ntheta=ntheta,
                filename=str(vc_filename),
            )
        except ModuleNotFoundError as err:
            if err.name == "virtual_casing":
                raise ModuleNotFoundError(
                    "The auxiliary VMEC-backed mode needs the optional virtual_casing package when no cached vcasing file is available. "
                    "Install virtual_casing or set SIMSOPT_FINITE_BETA_VCASING to an existing vcasing_*.nc file."
                ) from err
            raise

    boundary_rz = SurfaceRZFourier.from_nphi_ntheta(
        mpol=vmec.wout.mpol,
        ntor=vmec.wout.ntor,
        nfp=vmec.wout.nfp,
        stellsym=not bool(vmec.wout.lasym),
        nphi=nphi,
        ntheta=ntheta,
        range="half period",
    )
    boundary_rz.x = vmec.boundary.x

    surface = SurfaceXYZTensorFourier(
        mpol=vmec.wout.mpol,
        ntor=vmec.wout.ntor,
        nfp=vmec.wout.nfp,
        stellsym=not bool(vmec.wout.lasym),
        quadpoints_phi=boundary_rz.quadpoints_phi,
        quadpoints_theta=boundary_rz.quadpoints_theta,
    )
    surface.least_squares_fit(boundary_rz.gamma())
    vol = Volume(surface)
    vol_target = vol.J()

    boozer_surface = FiniteBetaBoozerSurface(
        None,
        surface,
        vol,
        vol_target,
        pressure_jump=pressure_jump,
        options={'verbose': True, 'ls_max_nfev': ls_max_nfev},
    )

    B_in = vc.B_total - vc.B_external
    B_out = vc.B_external
    initial = boozer_surface.residual_blocks(iota=vmec.iota_edge(), G=0.0, I=0.0, B_in=B_in, B_out=B_out)
    print_blocks("Initial residual norms:", initial)

    res = boozer_surface.run_code(iota=vmec.iota_edge(), G=0.0, I=0.0, B_in=B_in, B_out=B_out, optimize_G=False)
    solved = boozer_surface.residual_blocks(
        iota=res["iota"],
        G=res["G"],
        I=res["I"],
        current_potential=res["current_potential"],
        B_in=B_in,
        B_out=B_out,
    )
    print_blocks("Solved residual norms:", solved)
    print(
        f"Solved parameters: success={res['success']}, iota={res['iota']:.6e}, "
        f"G={res['G']:.6e}, I={res['I']:.6e}, ||r||={res['residual_norm']:.6e}"
    )
    write_residual_vtk(surface, solved)
    if res["success"]:
        print("Auxiliary VMEC-backed finite-beta solve converged.")
    else:
        print("Auxiliary VMEC-backed finite-beta run finished without convergence; increase SIMSOPT_FINITE_BETA_MAX_NFEV for a tighter solve.")

else:
    raise ValueError("SIMSOPT_FINITE_BETA_MODE must be one of 'prescribed', 'prescribed-fixed', or 'vmec'.")

print("End of 2_Intermediate/boozerQA_finitebeta.py")
print("=============================================")