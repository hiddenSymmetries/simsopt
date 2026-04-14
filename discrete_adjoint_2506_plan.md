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

### 1. Wrapper surface area

- [ ] Add an explicit `VmecJax` derivative backend option for the new discrete-adjoint
  residual path.
- [ ] Keep wrapper defaults conservative until the new backend is validated.
- [ ] Make backend selection explicit in the QH example output.

### 2. QH example cleanup

- [ ] Keep the script structure close to the classic example:
  - setup,
  - objective definition,
  - before-solve reporting,
  - solve,
  - after-solve reporting.
- [ ] Avoid branchy QA/QH scaffolding in the QH example itself.
- [ ] Avoid hidden finite-difference Jacobian fallbacks.

### 3. Solver choice

- [ ] Re-test least-squares methods only after the new backend is correct.
- [ ] Prefer a solver that does not need hand-tuned finite step sizes once the
  Jacobian is trustworthy.
- [ ] Keep `gradient_descent` as a diagnostic only, not the target path.

### 4. Regression coverage

- [ ] Add a wrapper-level regression confirming the QH start objective and finite
  gradients on the new backend.
- [ ] Add a small QH derivative smoke test from the public `VmecJax` API.

## Acceptance

This branch is successful only if the resulting `QH_fixed_resolution_jax.py`:

- uses autodiff / adjoints only,
- materially improves QH quasisymmetry,
- reaches objective quality in the same ballpark as the classic script,
- is competitive enough in runtime to justify the JAX path.

## Immediate next tasks

- [ ] Wait for the new `vmec_jax` backend interface to exist.
- [ ] Wire it into `VmecJax`.
- [ ] Replace the current QH example defaults with the new backend once validated.
