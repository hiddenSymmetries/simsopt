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
  - Added a narrower runtime fix in the SciPy callback pair for the discrete
    backend:
    - residual and Jacobian callbacks now share the same exact
      discrete-adjoint solve payload when SciPy evaluates both at the same
      parameter vector,
    - the Jacobian builder accepts an already-solved state/tape payload instead
      of always rebuilding it.
  - Added a wrapper regression that proves `scipy_residuals(x0)` followed by
    `scipy_jacobian(x0)` only calls
    `_solve_state_discrete_adjoint_residual(...)` once on the exact same point.
  - Sequential microcase reruns on the new callback-reuse path:
    - QH:
      `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 2 --vmec-max-iter 1
      --timings --method scipy --jac jax
      --residual-derivative-backend discrete_adjoint --jit`
      still gives total objective `1.901008100532871 -> 0.505258184704556`,
      and solve wall time dropped again from about `5.14 s` to about `4.10 s`.
    - QA:
      `QA_fixed_resolution_jax.py --max-mode 1 --max-nfev 2 --vmec-max-iter 1
      --timings --method scipy --jac jax
      --residual-derivative-backend discrete_adjoint --jit`
      still gives total objective `9.168100000000209 -> 5.205846451699488`,
      and solve wall time dropped from about `8.58 s` to about `4.51 s`.
  - QA exact full-inner run on the same wrapper path is still heavy before the
    first SciPy iteration, so QA does not automatically bypass the current
    wrapper/Jacobian bottleneck.
  - Re-probed the exact full-inner QH path after the callback reuse fix with:
    - `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 1 --timings
      --method scipy --jac jax --residual-derivative-backend discrete_adjoint
      --jit`
    - the run still exited before the first SciPy iteration completed, so this
      fix materially improved the short compiled path but did not yet stabilize
      the full-inner benchmark regime.
  - Switched the production discrete-adjoint wrapper off the replay-built tape
    and onto a new direct full-solve tape path from `vmec_jax`, so the exact
    Jacobian payload is built from one residual solve with full
    `adjoint_step_trace` history instead of hundreds of `max_iter=1` replay
    solves.
  - On the exact full-inner QH benchmark path this removes the compile storm
    that previously happened before SciPy iteration 0. The exact run now gets
    through real SciPy iterations:
    - `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 1 --timings
      --method scipy --jac jax --residual-derivative-backend discrete_adjoint
      --jit`
      now reaches and completes iteration 0 with solve wall time about
      `16.58 s`.
    - `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 2 --timings
      --method scipy --jac jax --residual-derivative-backend discrete_adjoint
      --jit`
      now completes two exact full-inner evaluations with:
      - total objective `0.2983122217180416 -> 0.2651202937448159`,
      - QS objective `0.24335963941459998`,
      - aspect `7.147514929177409`,
      - solve wall time about `29.73 s`.
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
  - After the stacked-trace cache refactor landed in `vmec_jax`, reran the
    exact full-inner apples-to-apples QH benchmark:
    - `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 10 --timings
      --method scipy --jac jax --residual-derivative-backend discrete_adjoint
      --jit`
      now finishes with the same final objective
      `0.25740827662370214`, but solve wall time improves from about
      `136.70 s` to about `103.92 s` and shell real time improves from about
      `167.41 s` to about `133.02 s`.
  - Probed exact callback compile churn on `x0`, repeated `x0`, and nearby
    `x1`, and found one remaining avoidable `simsopt` helper-cache miss:
    `_initial_tangent_columns` and `_residual_tangent_columns` were
    recompiling when the tape length changed, even though those helpers do not
    depend on step count.
  - Removed `len(payload["tape"].step_traces)` from the helper-cache key in
    `VmecJax._discrete_adjoint_residual_jacobian(...)`.
  - After that change, compile logging shows each of
    `_initial_tangent_columns` and `_residual_tangent_columns` compiles only
    once across `x0/x0/x1`, while the nearby exact callback timings improve to:
    - `x0a`: solve about `7.89 s`, jac about `4.14 s`
    - `x0b`: solve about `5.77 s`, jac about `3.32 s`
    - `x1`: solve about `5.65 s`, jac about `3.49 s`
  - After the replay scan-runner cache landed in `vmec_jax`, reran the exact
    full-inner apples-to-apples QH benchmark again:
    - `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 10 --timings
      --method scipy --jac jax --residual-derivative-backend discrete_adjoint
      --jit`
      now finishes with the same final objective
      `0.25740827662357`, but solve wall improves again to about `97.79 s`
      and shell real time to about `126.34 s`.
  - Exact `max_mode=1`, `max_nfev=3` JAX benchmark on the same path:
    - final total objective `0.2581125330463535`
    - final QS objective `0.2565631893138016`
    - final aspect `7.039361703882731`
    - solve wall about `30.89 s`
    - shell real about `55.97 s`
  - Compared against the local classic reference at the same early-stop point:
    - classic `max_mode=1`, `max_nfev=3` reaches cost `1.0803e-01`, so total
      objective is already about `0.21606`;
    - the current JAX exact path is still materially worse at `0.25811`.
  - Exact `max_mode=2`, `max_nfev=3` JAX benchmark:
    - final total objective `0.30024580360290376` (no improvement)
    - solve wall about `97.98 s`
    - shell real about `130.74 s`
  - Compared against the local classic `max_mode=2`, `max_nfev=3` reference:
    - classic reaches cost `5.1158e-02`, so total objective is already about
      `0.102316`;
    - the current JAX exact path is still far slower and does not move off the
      starting point by 3 function evaluations.
  - Ran a stricter derivative audit on the exact `max_mode=1`, `max_iter=1`
    production path and isolated the earlier AD-vs-FD discrepancy:
    - the discrete-adjoint Jacobian column does **not** match central FD of the
      plain `scipy_residuals(x)` callback;
    - but it **does** match central FD almost exactly when the residual is
      finite-differenced through the same frozen-axis local map that the
      discrete-adjoint path actually linearizes.
  - On the benchmark QH start point, direction `e0`:
    - plain residual callback FD gives objective-direction about `8.52594`;
    - discrete-adjoint AD gives about `33.67235`;
    - frozen-axis local residual FD gives about `33.67455`.
  - This means the live production derivative gap is now understood as a model
    mismatch in the initialization branch (`axis_override` refreshed at each
    perturbed point in the value callback versus frozen in the local Jacobian),
    not a corruption in the replay/tape linearization itself.
  - Follow-up full-inner audit changed that diagnosis for the production path:
    at the exact full-inner benchmark (`max_iter=1500`), moving-axis FD and
    frozen-axis FD are nearly identical at both `x0` and the first accepted
    SciPy point `x1`, and both disagree with the production Jacobian.
  - On the exact benchmark path:
    - at `x0`, direction `e0`: AD objective-direction about `-3.4527`, moving
      and frozen FD both about `-1.03`;
    - at `x1`, direction `e0`: AD about `-5.5086`, moving and frozen FD both
      about `-2.964`;
    - along the local GN direction, the same pattern holds.
  - A frozen-axis iteration sweep then showed the replay Jacobian is good only
    on short tapes and drifts progressively with tape length:
    - `max_iter=1`: relative frozen-axis column error about `3.1e-4`
    - `max_iter=10`: about `1.5e-2`
    - `max_iter=20`: about `6.5e-2`
    - `max_iter=50`: about `5.7e-1`
    - `max_iter=100`: about `4.85`
  - The exact QH `max_iter=100` step trace shows all 100 steps are still
    `momentum_accept` with no restarts, so the drift is not caused by restart
    branches.
  - The remaining live hypothesis is now solver-control carry:
    `time_step` itself stays fixed, but `dt_eff`, `b1`, `fac`, and
    `force_scale` vary every step with the residual history, while the replay
    Jacobian still treats those scalars as frozen trace data from the base
    trajectory.
  - Full carry-aware replay landed on the vmec_jax branch and removed the
    long-horizon QH Jacobian drift on the production exact path.
  - Exact frozen-axis directional audit on the benchmark wrapper path now gives:
    - `max_iter=1`: relative column error about `3.0e-8`
    - `max_iter=10`: about `1.5e-6`
    - `max_iter=20`: about `2.6e-5`
    - `max_iter=50`: about `1.15e-3`
  - Exact `max_mode=1`, `max_nfev=3` benchmark improved materially:
    - final total objective `0.25541676622646603`
    - final QS objective `0.2541757712576408`
    - final aspect `7.035227758498451`
    - solve wall about `41.99 s`
  - Exact `max_mode=1`, `max_nfev=10` benchmark after the carry fix:
    - final total objective `0.22719939587753907`
    - final QS objective `0.22499609902383344`
    - final aspect `7.046939289020027`
    - solve wall about `129.20 s`
    - this is much closer to the classic `0.21378863910867005`, though still
      slower than the classic `24.75 s`.
  - Exact `max_mode=2`, `max_nfev=3` benchmark is no longer stalled:
    - final total objective `0.12191948331868585`
    - final QS objective `0.11793999150722417`
    - final aspect `7.063083213388838`
    - solve wall about `51.55 s`
    - compared with the local classic reference at about `0.102316`, this is
      now qualitatively competitive in objective reduction, though still slower.
  - Added a slow wrapper regression that locks the full-inner (`max_iter=20`)
    QH frozen-axis Jacobian against central FD on the production
    discrete-adjoint backend.
  - The production wrapper is no longer using the old frozen-axis
    initial-state surrogate. After fixing the traced moving-axis
    `initial_guess_from_boundary(...)` path in `vmec_jax`, the exact
    discrete-adjoint backend now linearizes the true moving-axis initial-state
    map on the user-facing QH objective path.
  - The replay tangent transport is also no longer linearized around a drifting
    replayed scan. It now composes the dynamic one-step map at the stored
    primal carry of each accepted step, which materially tightened the exact
    QH Jacobian audit on the real wrapper path.
  - Updated slow wrapper gates now lock:
    - exact QH start-point moving-axis Jacobian vs central FD at `max_iter=1`
    - exact QH start-point moving-axis Jacobian vs central FD at `max_iter=20`
  - Current exact full-inner start-point audit on the production path:
    - moving-axis `max_iter=20` relative column error about `2.25e-3`
    - objective-direction mismatch about `1.1e-2` relative
    - that is materially better than the earlier full-inner moving-axis drift,
      but still leaves some headroom before calling the derivative issue fully
      closed at longer horizons / later accepted iterates.
  - Exact callback RSS profiling on the benchmark QH wrapper path now shows:
    - cold exact residual at `x0`: about `11.65 s`, RSS `+3.00 GB`
    - cold exact Jacobian at `x0`: about `6.93 s`, RSS `+0.44 GB`
    - warm exact residual at nearby `x1`: about `6.02 s`, RSS `+2.36 GB`
    - warm exact Jacobian at nearby `x1`: about `5.15 s`, RSS `+0.14 GB`
    - so the remaining large memory growth is dominated by the exact forward
      solve path and executable retention, not by the replay Jacobian alone.
  - The first lean exact-tape pass reduced runtime and peak RSS materially, but
    it also exposed that the late-iteration Jacobian was still catastrophically
    wrong whenever the tape contained even one restart step; SciPy optimality
    blew up again there.
  - Exact late-point audit on the benchmark QH final iterate before the mixed
    replay fix gave objective-direction errors of order `1e19` on columns
    `0..3`.
  - After the mixed replay fix on the vmec_jax branch, the same final-point
    audit now gives:
    - column `0`: objective-direction relative error about `9.9e-3`
    - column `1`: about `2.6e-2`
    - column `2`: about `1.0e-5`
    - column `3`: about `2.9e-2`
    - so the late-iteration Jacobian blow-up is fixed.
  - Exact `max_mode=1`, `max_nfev=3` benchmark on the current branch state:
    - final total objective `0.26052325878767574`
    - final QS objective `0.2596812636714725`
    - final aspect `7.029017152103597`
    - solve wall about `36.10 s`
    - peak RSS about `17.48 GB`
  - Exact `max_mode=1`, `max_nfev=10` benchmark on the current branch state:
    - final total objective `0.24174600728278778`
    - final QS objective `0.24061022810764002`
    - final aspect `7.0337013230474374`
    - solve wall about `138.55 s`
    - peak RSS about `23.47 GB`
    - first-order optimality now stays finite at about `6.62e-1`, which is a
      major behavioral improvement over the earlier `~1e19` blow-up
  - This branch state is now derivative-stable at late iterates, but runtime is
    still not competitive with classic QH and the final objective is still
    above the earlier best branch result and above the classic reference.
