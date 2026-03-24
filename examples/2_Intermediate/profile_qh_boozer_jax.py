#!/usr/bin/env python

import argparse
import os
import time

import jax
import jax.numpy as jnp

import vmec_jax as vj
from booz_xform_jax.jax_api import (
    booz_xform_from_inputs,
    prepare_booz_xform_constants_from_inputs,
)

from simsopt.mhd import VmecJax, BoozerJax, QuasisymmetryJax


def _block(x):
    return jax.block_until_ready(x)


def _qs_residual_from_booz(out, s_to_index, s_used, helicity_m=1, helicity_n=1,
                           normalization="B00", weight="even"):
    bmnc_b = jnp.asarray(out["bmnc_b"])
    xm_b = jnp.asarray(out["ixm_b"])
    xn_b = jnp.asarray(out["ixn_b"]) / jnp.asarray(out["nfp_b"], dtype=bmnc_b.dtype)

    symmetry_error = []
    for s in s_to_index:
        index = s_to_index[s]
        bmnc = bmnc_b[index, :]

        if helicity_m not in (0, 1):
            raise ValueError("m for quasisymmetry should be 0 or 1.")

        if helicity_n == 0:
            symmetric = (xn_b == 0)
        elif helicity_m == 0:
            symmetric = (xm_b == 0)
        else:
            symmetric = (xm_b * helicity_n + xn_b * helicity_m == 0)

        nonsymmetric = jnp.logical_not(symmetric)

        if normalization == "B00":
            bnorm = bmnc[0]
        elif normalization == "symmetric":
            temp = bmnc[symmetric]
            bnorm = jnp.sqrt(jnp.dot(temp, temp))
        else:
            raise ValueError("Unrecognized value for normalization in Quasisymmetry")

        bmnc = bmnc / bnorm

        if weight == "even":
            symmetry_error.append(bmnc[nonsymmetric])
        elif weight == "stellopt":
            rad_sigma = jnp.asarray(s_used[s] * s_used[s], dtype=bmnc.dtype)
            symmetry_error.append((bmnc / rad_sigma)[nonsymmetric])
        elif weight == "stellopt_ornl":
            symmetry_error.append(jnp.array([jnp.linalg.norm(bmnc[nonsymmetric])]))
        else:
            raise ValueError("Unrecognized value for weight in Quasisymmetry")

    return jnp.concatenate(symmetry_error, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.nfp4_QH_warm_start")
    parser.add_argument("--mpol", type=int, default=32)
    parser.add_argument("--ntor", type=int, default=32)
    parser.add_argument("--mboz", type=int, default=32)
    parser.add_argument("--nboz", type=int, default=32)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warm-start-iters", type=int, default=20)
    parser.add_argument("--solver", default="gd", choices=["gd", "lbfgs", "initial"])
    parser.add_argument("--vmec-max-iter", type=int, default=None)
    parser.add_argument("--vmec-grad-tol", type=float, default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--jit-booz", action="store_true")
    parser.add_argument("--jit-full", action="store_true")
    args = parser.parse_args()

    filename = os.path.join(os.path.dirname(__file__), "inputs", args.input)

    vmec = VmecJax(filename, verbose=False)
    solver_use = args.solver
    if args.jit_full and solver_use != "initial":
        print("jit-full requested: using solver=initial for JIT compatibility.")
        solver_use = "initial"

    vmec.set_solver_options(
        solver=solver_use,
        warm_start_iters=args.warm_start_iters,
        max_iter=args.vmec_max_iter,
        grad_tol=args.vmec_grad_tol,
    )
    vmec.indata.mpol = args.mpol
    vmec.indata.ntor = args.ntor

    x0 = vmec.boundary.get_free_params()
    x0 = jnp.asarray(x0)

    static = vmec.get_static()
    indata = vmec._indata_raw
    s_list = [float(args.s)]
    compute_surfs, s_used = vj.optimization.surface_indices_from_static(static, s_list)
    s_to_index = {s: int(i) for i, s in enumerate(s_list)}
    s_used = {s: float(s_used[i]) for i, s in enumerate(s_list)}

    constants = None
    grids = None
    boozer = BoozerJax(vmec, mpol=int(args.mboz), ntor=int(args.nboz), jit=bool(args.jit_booz))
    qs = QuasisymmetryJax(boozer, s_list, 1, 1)

    def residuals(x_free):
        aspect = vmec.aspect_jax(x_free)
        qs_res = qs.J(x_free)
        return jnp.concatenate([jnp.array([aspect - 7.0]), qs_res])

    residuals_jit = None
    if args.jit_full:
        try:
            qs.prepare(x0)
            residuals_jit = jax.jit(residuals)
            _block(residuals_jit(x0))
        except Exception as exc:
            print(f"jit-full failed ({type(exc).__name__}: {exc}); falling back to non-jitted objective.")
            residuals_jit = None

    def run_once(x_free):
        nonlocal constants, grids
        if residuals_jit is not None:
            t0 = time.perf_counter()
            res = residuals_jit(x_free)
            _block(res)
            t1 = time.perf_counter()
            return {
                "vmec_s": float("nan"),
                "booz_s": float("nan"),
                "qs_s": float("nan"),
                "total_s": t1 - t0,
            }

        t0 = time.perf_counter()
        state = vmec._solve_state(x_free)
        _block(state.Rcos)
        t1 = time.perf_counter()

        inputs = vj.optimization.booz_xform_inputs_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=vmec._signgs,
        )
        if constants is None:
            constants, grids = prepare_booz_xform_constants_from_inputs(
                inputs=inputs,
                mboz=int(args.mboz),
                nboz=int(args.nboz),
                asym=bool(static.cfg.lasym),
            )
        out = booz_xform_from_inputs(
            inputs=inputs,
            constants=constants,
            grids=grids,
            surface_indices=jnp.asarray(compute_surfs, dtype=jnp.int32),
            jit=bool(args.jit_booz),
        )
        _block(out["bmnc_b"])
        t2 = time.perf_counter()

        qs_res = _qs_residual_from_booz(out, s_to_index, s_used)
        _block(qs_res)
        t3 = time.perf_counter()

        return {
            "vmec_s": t1 - t0,
            "booz_s": t2 - t1,
            "qs_s": t3 - t2,
            "total_s": t3 - t0,
        }

    for _ in range(int(args.warmup)):
        run_once(x0)

    if args.trace_dir:
        jax.profiler.start_trace(args.trace_dir)
    stats = run_once(x0)
    if args.trace_dir:
        jax.profiler.stop_trace()

    print("Timing (seconds):")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
