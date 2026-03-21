#!/usr/bin/env python

import argparse
import os
import time

import jax
import jax.numpy as jnp

from simsopt.mhd import VmecJax


def _block(x):
    return jax.block_until_ready(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.nfp4_QH_warm_start")
    parser.add_argument("--mpol", type=int, default=4)
    parser.add_argument("--ntor", type=int, default=4)
    parser.add_argument("--warm-start-iters", type=int, default=0)
    parser.add_argument("--vmec-max-iter", type=int, default=1)
    parser.add_argument("--vmec-grad-tol", type=float, default=1e-3)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--trace-dir", default=None)
    args = parser.parse_args()

    filename = os.path.join(os.path.dirname(__file__), "inputs", args.input)

    vmec = VmecJax(filename, verbose=False)
    vmec.set_solver_options(
        solver="gd",
        warm_start_iters=int(args.warm_start_iters),
        max_iter=int(args.vmec_max_iter),
        grad_tol=float(args.vmec_grad_tol),
    )
    vmec.indata.mpol = int(args.mpol)
    vmec.indata.ntor = int(args.ntor)

    x0 = jnp.asarray(vmec.boundary.get_free_params())

    def obj(x):
        return vmec.aspect_jax(x)

    if args.jit:
        obj = jax.jit(obj)
        _block(obj(x0))

    if args.trace_dir:
        jax.profiler.start_trace(args.trace_dir)

    t0 = time.perf_counter()
    val = obj(x0)
    _block(val)
    t1 = time.perf_counter()

    if args.trace_dir:
        jax.profiler.stop_trace()

    print(f"aspect: {float(val):.6f}")
    print(f"elapsed_s: {t1 - t0:.4f}")


if __name__ == "__main__":
    main()
