# Finite-Beta Boozer QA Plan

## Goal

Add a new finite-beta Boozer workflow beside the current vacuum Boozer workflow, keeping the existing vacuum path unchanged. The finite-beta path should be implemented as a single-surface least-squares residual system with explicit sheet-current closure, prescribed interior and exterior magnetic fields on the surface, analytic derivatives, tests at several levels, and matching documentation. Ideal-MHD equilibrium solvers may be used to generate validation data, but they are not part of the optimization loop.

## Status

1. Completed: Python finite-beta residual blocks, fixed-surface least-squares solve, analytic derivatives for the parameter block, and targeted finite-beta tests.
2. Completed: analytic surface-dof derivatives for the fixed-field finite-beta residual blocks, together with a directional finite-difference regression test.
3. Completed: `FiniteBetaBoozerSurface.run_code()` supports analytic surface optimization for fixed prescribed fields, while reevaluable field providers use numerical surface differentiation.
4. Completed: a synthetic moving-surface QA example exercises the finite-beta solve inside a surface optimization loop without VMEC in the loop.
5. Completed: VMEC plus virtual-casing remains available as an auxiliary field-data generation and validation path, while the primary framework is the single-surface prescribed-field problem.
6. Completed: documentation and regression coverage were added in the same branch as the feature work.

## Closeout

1. Primary workflow: single-surface finite-beta Boozer solve with prescribed `B_in` and `B_out` on one surface.
2. Auxiliary workflow: VMEC plus virtual-casing can generate or validate surface field data, but is not in the optimization loop.
3. Legacy coverage preserved: the existing vacuum Boozer regression path remains green alongside the new finite-beta tests.
4. Branch outcome: the implementation is ready to land as one final PR from the dedicated working branch.

## Phases

### 1. Freeze the physics model

1. Write the governing-model note that defines the exact unknowns, equations, conventions, and limits for the finite-beta Boozer solve.
2. Freeze the unknown set for the first merge: surface dofs, `iota`, `G`, internal current `I`, and sheet-current potential coefficients `Phi_mn` on the target surface.
3. Freeze the gauge choice for `Phi`: remove the constant mode or impose zero mean on the single-valued part of `Phi`.
4. Freeze the finite-beta interface interpretation and its sign conventions.
5. Freeze the fact that the first finite-beta path is least-squares only.

### 2. Freeze the equations

1. Generalized Boozer tangential relation:
   `(G + iota I) B_in - |B_in|^2 (x_phi + iota x_theta) = 0`
2. Normal-field condition:
   `B_out · n = 0`
3. Pressure-balance jump condition:
   `|B_out|^2 - |B_in|^2 - 2 mu0 Delta p = 0`
4. Tangential jump / sheet-current closure:
   `mu0 K - n x (B_out - B_in) = 0`
5. Surface-current representation:
   `K = n x grad Phi`
6. Retain the surface label constraint.
7. Retain the geometric anchor or symmetry-fixing constraint where needed.
8. Freeze the vacuum limit: when `Delta p = 0`, `I = 0`, and `Phi_mn = 0`, the new path must reduce to the current vacuum behavior.

### 3. Source-code architecture

1. Keep the current vacuum Boozer implementation unchanged.
2. Add a new finite-beta driver class, preferably `FiniteBetaBoozerSurface`, as a sibling to `BoozerSurface`.
3. Add the finite-beta residual builder and differentiated residual helpers in `src/simsopt/geo/surfaceobjectives.py`.
4. Add minimal surface-current field support under `src/simsopt/field` if needed.
5. Treat prescribed `B_in` and `B_out` on the surface as the primary finite-beta interface. Reuse `src/simsopt/mhd/virtual_casing.py` only as an optional data-generation path when helpful.
6. Delay compiled-kernel work until the Python equations and derivative tests are stable.

### 4. Python-first implementation

1. Build the finite-beta residual assembly in Python first.
2. Keep the residual blocks separable: generalized Boozer residual, normal-field residual, pressure-balance residual, jump residual, label residual, and `Phi` gauge residual.
3. Expose both the full stacked residual and per-block diagnostics.
4. Keep the primary solve path on a single target surface with prescribed field data, and only use VMEC-backed cases as optional validation inputs.

### 5. Analytic derivatives

1. Require analytic derivatives from the start.
2. Differentiate each residual block with respect to surface dofs, `iota`, `G`, `I`, and `Phi_mn`.
3. Reuse the existing Biot-Savart derivative and adjoint patterns where possible.
4. Add compiled derivative parity only after the Python derivatives pass Taylor tests.

### 6. Validation and testing

1. Algebraic validation on manufactured limits.
2. Physics validation on fixed prescribed-field cases, with optional VMEC plus virtual-casing validation when available.
3. Cross-limit validation back to vacuum.
4. Solver validation that every residual block decreases.
5. Unit tests for residual assembly, gauge handling, and derivatives.
6. Physics tests for vacuum limit, pressure-balance sign, jump response, and gauge invariance.
7. Regression tests for the existing vacuum Boozer path.
8. Example smoke tests in CI-scale settings.

### 7. Example work

1. Create `examples/2_Intermediate/boozerQA_finitebeta.py` as a near-copy of `examples/2_Intermediate/boozerQA.py`.
2. Keep the script layout parallel to the vacuum example.
3. Make the single-surface prescribed-field workflow the primary example path, with optional VMEC-backed data loading only as an auxiliary mode.
4. Add a CI-scale mode using `in_github_actions` or similar.

### 8. Documentation

1. Add user-facing documentation for the new finite-beta Boozer example.
2. Document the new example script, required dependencies, and runtime expectations.
3. Add API documentation for the new finite-beta Boozer driver class.
4. Add API documentation for any new surface-current field representation.
5. Update `docs/source/fields.rst` if a new field type is added.
6. Update `docs/source/geo.rst` if a new Boozer or geometry API is added.
7. Keep documentation in the same PR as the feature.

### 9. PR and branch plan

1. Keep all remaining work on the implementation branch from upstream `master`.
2. Working branch: `feature/finite-beta-boozer-surface`.
3. Land the feature as one final PR once the single-surface finite-beta framework, validation, and documentation are complete.
4. In the final PR description, include motivation, governing equations, unknown set, solver choice, validation matrix, runtime impact, regression coverage, and documentation changes.

## Verification checklist

1. Done: governing equations and sign conventions are frozen in writing.
2. Done: Python finite-beta residual assembly is implemented and block diagnostics are available.
3. Done: first-derivative Taylor tests pass for the implemented new unknown groups.
4. Done: physics tests cover the vacuum limit, pressure-balance sign, jump-condition response, and gauge handling.
5. Done: existing vacuum Boozer tests remain green.
6. Done: the new example runs in CI-scale prescribed mode, with auxiliary VMEC-backed validation available.
7. Done: documentation for the new example and the new geometry API is included.
8. Deferred by design: compiled performance parity work was kept out of scope for the first implementation.

## Out of scope for the first implementation

1. Multi-surface finite-beta Boozer QA.
2. A broad free-boundary equilibrium refactor inside simsopt.
3. Replacing the virtual-casing package or redesigning its global API.
4. General-purpose surface-current infrastructure beyond what the finite-beta Boozer path strictly needs.
5. Aggressive performance optimization before correctness and tests are complete.
