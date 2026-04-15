# QH VMEC-JAX Discrete-Adjoint Consumer Plan

Last updated: 2026-04-14
Repo: `simsopt`
Branch: `codex/discrete-adjoint-2506`
Coupled repo: `vmec_jax` on `codex/discrete-adjoint-2506`

## Purpose

This branch exists to consume a new solver-faithful discrete-adjoint derivative path
from `vmec_jax`, and to use it in a clean `QH_fixed_resolution_jax.py` that mirrors
the structure of the classic `QH_fixed_resolution.py`.

The intent is not to add another layer of optimizer tuning around a weak derivative.
The intent is to expose one correct and performant autodiff path to the existing
SIMSOPT objective/solver machinery.

## Paper-driven design decision

The method in `arXiv:2506.14792` suggests that the right long-term structure is:

- treat the equilibrium solve as the differentiated object,
- use the solver's own accepted discrete path in reverse mode,
- avoid dense finite-difference or dense Jacobian fallback paths,
- keep the outer optimization code simple.

For `simsopt`, that means the main work should stay small:

- choose the right `VmecJax` derivative backend,
- expose a clean QH example,
- keep the benchmark apples-to-apples with the classic script.

Important reality on this clean branch:

- clean `simsopt` mainline does **not** yet contain:
  - `src/simsopt/mhd/vmec_jax.py`,
  - `src/simsopt/mhd/vmec_diagnostics_jax.py`,
  - `src/simsopt/solve/jax_solve.py`,
  - `examples/2_Intermediate/QH_fixed_resolution_jax.py`.

So this branch is not just an example edit. It needs a staged reintroduction of
the JAX consumer layer once the new backend in `vmec_jax` is trustworthy.

## Required benchmark

- classic script: `examples/2_Intermediate/QH_fixed_resolution.py`
- JAX script: `examples/2_Intermediate/QH_fixed_resolution_jax.py`
- objective: `aspect + quasisymmetry`
- no `mean_iota`
- `max_mode = 1`
- VMEC `mpol = ntor = 3`
- 8 free boundary DOFs
- `max_nfev = 10`

## Planned changes

### 0. Reintroduce the minimum JAX consumer surface

- [ ] Port the minimum required files from the previous experimental branch, not
  the whole branch history.
- [ ] Keep the first port narrow:
  - `src/simsopt/mhd/vmec_jax.py`
  - `src/simsopt/mhd/vmec_diagnostics_jax.py`
  - `src/simsopt/solve/jax_solve.py`
  - exports / imports needed to surface them
  - focused tests only
- [ ] Do this only after `vmec_jax` reaches derivative Gate 4C.

Acceptance:

- clean `simsopt` can import the JAX wrapper and run a one-evaluation QH script.

### 1. Wrapper surface area

- [ ] Add an explicit `VmecJax` derivative backend option for the new discrete-adjoint
  residual path.
- [ ] Keep wrapper defaults conservative until the new backend is validated.
- [ ] Make backend selection explicit in the QH example output.

Acceptance:

- backend selection is user-visible and test-covered,
- the legacy implicit path is not silently selected by accident.

### 2. QH example cleanup

- [ ] Keep the script structure close to the classic example:
  - setup,
  - objective definition,
  - before-solve reporting,
  - solve,
  - after-solve reporting.
- [ ] Avoid branchy QA/QH scaffolding in the QH example itself.
- [ ] Avoid hidden finite-difference Jacobian fallbacks.

Acceptance:

- the example reads like the classic QH script with a JAX-backed VMEC object
  substituted for the classic VMEC object,
- the derivative backend used is explicit in script output.

### 3. Solver choice

- [ ] Re-test least-squares methods only after the new backend is correct.
- [ ] Prefer a solver that does not need hand-tuned finite step sizes once the
  Jacobian is trustworthy.
- [ ] Keep `gradient_descent` as a diagnostic only, not the target path.

Acceptance:

- no production default depends on a hand-tuned finite step size,
- the selected solver is justified by short benchmark data rather than guesswork.

### 4. Regression coverage

- [ ] Add a wrapper-level regression confirming the QH start objective and finite
  gradients on the new backend.
- [ ] Add a small QH derivative smoke test from the public `VmecJax` API.

Acceptance:

- wrapper regressions fail early if the backend becomes non-finite, changes the
  QH start state, or silently regresses derivative quality.

## Benchmarks and gates

### Consumer-side gates

- Gate S0: import and one-evaluation smoke test for the reintroduced wrapper.
- Gate S1: exact QH start objective matches the intended converged forward path.
- Gate S2: public-API directional derivative agrees with finite differences on
  the exact QH start point.
- Gate S3: `max_nfev=1` runtime and correctness benchmark.
- Gate S4: `max_nfev=2` apples-to-apples QH benchmark.
- Gate S5: `max_nfev=10` apples-to-apples QH benchmark.

### Required runtime metrics

For every benchmark script run, record:

- total wall time,
- solver method,
- derivative backend,
- objective/Jacobian call counts,
- wall time per objective/Jacobian call,
- final total objective,
- final QS objective,
- final aspect.

### Required file-level benchmark targets

- `src/simsopt/mhd/vmec_jax.py`
  - wrapper overhead and caching overhead.
- `src/simsopt/mhd/vmec_diagnostics_jax.py`
  - QS residual evaluation cost and derivative smoke checks.
- `src/simsopt/solve/jax_solve.py`
  - solver bookkeeping overhead independent of VMEC solve cost.

## Acceptance

This branch is successful only if the resulting `QH_fixed_resolution_jax.py`:

- uses autodiff / adjoints only,
- materially improves QH quasisymmetry,
- reaches objective quality in the same ballpark as the classic script,
- is competitive enough in runtime to justify the JAX path.

## Current status

- [x] Port the minimum consumer layer from the earlier `simsopt_qh` branch.
- [x] Add an explicit `VmecJax` backend selector for the new residual
  discrete-adjoint path.
- [x] Add a first public wrapper derivative gate on the exact QH setup.
- [x] Add explicit reverse-Jacobian support to `least_squares_jax_solve`.
- [x] Expose backend / Jacobian mode in `QH_fixed_resolution_jax.py`.
- [ ] Benchmark the new path on the exact apples-to-apples QH run.
- [ ] Decide the production outer-solver policy for the discrete-adjoint backend.

## Current constraints

- The new backend is now a forward-mode (`custom_jvp`) path because the QH
  benchmark has `n=8` controls and `m ~ 4.4e4` residual components, so
  forward Jacobian columns are the correct scaling.
- The production SciPy consumer should therefore use `jac="jax"` /
  `jacfwd`, not `jac="reverse"`.
- `jit=False` remains the safe default on this path until the traced residual
  solver is fully cleaned up. A traced `np.asarray(...)` precompute in the
  VMEC residual path was identified as a concrete JIT blocker and is now being
  removed incrementally.

## Activity log

- 2026-04-14:
  - Added `residual_derivative_backend` to `VmecJax` and wired a narrow
    discrete-adjoint residual solve path into `_solve_state(...)`.
  - The new wrapper path uses the validated `vmec_jax` checkpoint tape reverse
    helper plus a frozen-axis initial-state VJP over SIMSOPT's free boundary DOFs.
  - Added wrapper regressions covering backend selection and an exact QH aspect
    directional derivative against finite differences.
  - Added `jac="reverse"` / `jacrev` support to `least_squares_jax_solve`.
  - Exposed `--jac` and `--residual-derivative-backend` in
    `examples/2_Intermediate/QH_fixed_resolution_jax.py`.
  - Small end-to-end script smoke no longer fails immediately on the old
    `TracerArrayConversionError`, but the full SciPy QH smoke is still too slow
    even at tiny inner budgets, so runtime policy is still unresolved.
  - Measured the actual scaling mismatch for the first wrapper implementation:
    reverse-mode scalar objective gradients were viable, but full residual
    Jacobians were the wrong shape/cost for QH (`m >> n`).
  - Switched the wrapper discrete-adjoint backend from `custom_vjp` to
    `custom_jvp` and removed the Python payload-cache workaround.
  - Added forward tape-JVP helpers in `vmec_jax` and confirmed the wrapper
    exact-QH aspect directional-derivative gate still matches finite
    differences after the switch.
  - Revalidated that `jax.jacfwd(stage.residuals)` works on the new backend and
    returns a finite `(44353, 8)` Jacobian on the QH microcase.
  - Re-ran a small SciPy QH smoke on the new forward path
    (`max_nfev=2`, `vmec_max_iter=1`, `jac='jax'`, discrete adjoint) and saw
    real least-squares progress:
    - total objective `1.901008100532871 -> 0.5052581847050714`,
    - wall time about `23.18 s`.
  - Identified the next concrete runtime blocker for the production path:
    `jit=True` still fails in the traced residual solver due to NumPy-only
    precompute code inside `vmec_jax.solve`.
  - Switched the production SciPy path away from traced `jacfwd(stage.residuals)`
    and onto a concrete Jacobian callable built from the discrete-adjoint tape
    columns in `VmecJax`.
  - `build_vmec_objective_stage(...)` now attaches a discrete-backend
    `scipy_jacobian` override, and `least_squares_jax_solve(...)` consumes it
    when `jac='jax'`.
  - That keeps the outer optimizer on an autodiff/adjoint Jacobian while
    avoiding traced execution of the VMEC residual solver entirely.
  - Revalidated the wrapper and solver regressions after this refactor.
  - Re-ran the same QH microcase with `--jit` still enabled; because the
    SciPy Jacobian now comes from the concrete tape-column path, it no longer
    depends on tracing the VMEC solve, and the 2-evaluation run improved to:
    - total objective `1.901008100532871 -> 0.5052581847050319`,
    - wall time about `18.26 s`.
  - Split the production SciPy path again so residual values and Jacobians no
    longer pay the same solver cost on the discrete backend:
    - residual values now use a plain forward residual solve without building
      the discrete-adjoint tape,
    - Jacobians still use the concrete tape-column autodiff path.
  - Revalidated the same wrapper and solver regressions after the split.
  - Re-ran the same QH microcase with the split residual/Jacobian path:
    - total objective `1.901008100532871 -> 0.5052581847050319`,
    - wall time `17.7149871670008 s`.
  - This confirms the next exact benchmark bottleneck is the plain forward
    VMEC residual solve, not Jacobian tracing/plumbing in the outer SciPy
    path.
  - Cleaned up `examples/2_Intermediate/QH_fixed_resolution_jax.py` so its
    pre/post reporting path now uses the same concrete forward solve and
    concrete residual callable as the production SciPy residual path when the
    discrete backend is active.
  - That removes a large amount of wasted work from the exact benchmark shell
    runtime: the script now reaches `Solver settings`, `Quasisymmetry
    objective before optimization`, and `Total objective before optimization`
    quickly on the exact QH setup instead of stalling in a tape-building
    `_solve_state(...)` call before the solve begins.
  - Reworked the concrete discrete-adjoint Jacobian builder so it no longer
    retraces the same linear maps for each of the 8 control directions:
    - linearize the frozen projected-initial-state map once,
    - propagate all 8 packed-state tangents through the replay tape together,
    - linearize the residuals-from-state map once and apply it to the packed
      tangent block.
  - Revalidated the simsopt wrapper/solver regressions after that batching
    refactor, and revalidated the matching exact-QH batched tape-JVP gate in
    `vmec_jax`.
  - Measured the exact QH start-point split on the production path after the
    batching refactor:
    - concrete residual evaluation is about `12.27 s` in one run and
      `13.57 s` in a rerun,
    - the concrete 8-column Jacobian build still dominates wall time by a wide
      margin and remains well above one minute.
  - This changes the runtime diagnosis again: the outer Python loop over
    columns is no longer the main cost. The next performance target is inside
    the tape-column propagation / exact replay Jacobian path itself.
  - Re-measured the exact QH start-point concrete Jacobian path directly on the
    current branch at `vmec_max_iter=1` and found a different dominant term:
    - solve/tape build about `2.42 s`,
    - frozen initial-state linearization about `2.10 s`,
    - replay column propagation about `1.30 s`,
    - residual-side linearization about `9.42 s`,
    - total concrete Jacobian build about `15.24 s`.
  - Based on that measurement, shifted the next optimization into the wrapper's
    pure-JAX blocks rather than the replay tape itself.
  - Added a cache of JIT-compiled helper functions inside
    `VmecJax._discrete_adjoint_residual_jacobian(...)` for:
    - frozen initial-state tangent columns,
    - residual tangent columns.
  - The residual-side JIT initially exposed a traced NumPy path in the
    Boozer/QS diagnostics on the `vmec_jax` side; after that blocker was fixed,
    the cached residual block became usable.
  - With those cached helper blocks active, the repeated exact-QH concrete
    Jacobian on the same start point dropped to about `1.17 s` for a
    `(44353, 8)` Jacobian after the first compilation, with zero numerical
    difference between repeated calls.
  - Re-ran the same production QH microcase:
    - `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 2 --vmec-max-iter 1
      --timings --method scipy --jac jax
      --residual-derivative-backend discrete_adjoint --jit`
    - total objective still `1.901008100532871 -> 0.505258184704556`,
    - solve wall time dropped again from about `16.93 s` to `5.14 s`.
  - Cleaned up the example benchmark script itself to stop doing duplicate
    forward solves just for reporting:
    - before: solve state for QS/aspect reporting, then solve again through
      `stage.residuals`/`scipy_residuals` to compute total objective,
    - now: compute the total objective directly from the already-available
      state using aspect + QS residuals.
  - This does not change the numerical results, but it removes redundant shell
    benchmark overhead at the start and end of the script, which matters on the
    exact full-inner-solve QH benchmark because those extra forward solves are
    expensive.
  - Added a new QA-side experiment script,
    `examples/2_Intermediate/QA_fixed_resolution_jax.py`, mirroring the
    cleaned fixed-resolution JAX example structure but targeting:
    - input `input.nfp2_QA`,
    - max_mode `1`,
    - QA helicity `(m, n) = (1, 0)`,
    - objective tuples `(aspect, 2.0, 1.0)`,
      `(mean_iota, 0.41, 1.0)`, `(qs, 0.0, 1.0)`.
  - QA smoke result on the current discrete-adjoint `simsopt` path:
    - `QA_fixed_resolution_jax.py --max-mode 1 --max-nfev 2 --vmec-max-iter 1
      --timings --method scipy --jac jax
      --residual-derivative-backend discrete_adjoint --jit`
    - total objective `9.168100000000209 -> 5.205846451699488`,
    - solve wall time about `8.58 s`.
  - QA exact full-inner run on the same wrapper path is still heavy before the
    first SciPy iteration, so QA does not automatically bypass the current
    wrapper/Jacobian bottleneck.
  - Built a fresh mainline comparison environment:
    - cloned `vmec_jax` main to `/Users/rogeriojorge/local/vmec_jax_main_fresh`,
    - installed it in an isolated venv with system site packages.
  - Direct fixed-boundary forward solves on main vs the discrete-adjoint branch
    are effectively identical, which rules out a low-level `vmec_jax` runtime
    regression as the primary cause of the current slowdown. Measured parity:
    - QA, `max_iter=1`: branch `2.454 s`, main `2.429 s`
    - QA, `max_iter=20`: branch `0.334 s`, main `0.326 s`
    - QH, `max_iter=1`: branch `1.963 s`, main `1.912 s`
    - QH, `max_iter=20`: branch `0.238 s`, main `0.228 s`
  - Full QA forward solve parity on main vs branch is also essentially exact:
    - both converge in `113` iterations with `fsqz_last ≈ 4.39e-12`,
    - branch `3.733 s`, main `3.622 s`.
  - That shifts the current diagnosis again:
    - the `vmec_jax` core forward solver is not the source of the major
      regression,
    - the remaining heaviness is in the higher-level `simsopt`
      objective/Jacobian path used before the first SciPy iteration.
