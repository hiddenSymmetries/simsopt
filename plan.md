# JAX finite-beta single-stage Simsopt integration plan

Date: 2026-04-25

Local Simsopt branch: `plan/jax-finite-beta-single-stage`

This file is intentionally uncommitted. It is both the implementation plan and
the running log for integrating `vmec_jax`, `booz_xform_jax`, and
`virtual_casing_jax` into Simsopt for finite-beta, single-stage optimization.

## Current checkout state

- `simsopt`: cloned to `/Users/rogerio/local/simsopt_jax`
  - Remote: `https://github.com/hiddensymmetries/simsopt.git`
  - Branch: `plan/jax-finite-beta-single-stage`
  - Base commit inspected: `1b0cc3a9`
- `vmec_jax`: cloned to `/Users/rogerio/local/vmec_jax_simsopt`
  - Remote: `https://github.com/uwplasma/vmec_jax.git`
  - Branch: `main`
  - Commit inspected: `42155e0`
- `booz_xform_jax`: cloned to `/Users/rogerio/local/booz_xform_jax_simsopt`
  - Remote: `https://github.com/uwplasma/booz_xform_jax.git`
  - Branch: `main`
  - Commit inspected: `e29fce7`
- `virtual_casing_jax`: cloned to `/Users/rogerio/local/virtual_casing_jax_simsopt`
  - Remote: `https://github.com/uwplasma/virtual_casing_jax.git`
  - Branch: `main`
  - Commit inspected: `a4a4b5b`

## High-level goal

Add JAX-backed MHD wrappers to Simsopt that can eventually replace the current
VMEC2000, BOOZ_XFORM, and virtual-casing integrations while preserving Simsopt's
public workflow:

- `Optimizable`-style wrappers with Simsopt dependency graph semantics.
- VMEC input and `wout` compatibility where users and tests expect it.
- Existing diagnostics such as aspect ratio, volume, iota, magnetic shear,
  external current, vacuum well, quasisymmetry residuals, Boozer spectra, and
  virtual-casing fields.
- Existing coil classes, Biot-Savart machinery, squared-flux objectives, and
  coil regularization terms remain in Simsopt.
- JAX autodiff replaces finite-difference VMEC/Boozer/virtual-casing derivative
  paths in the new examples and optimization workflows.
- Parity and regression tests compare old non-JAX wrappers and new JAX wrappers
  at every layer before the old implementations are considered replaceable.

## Source and literature context

Simsopt style and workflow references inspected locally:

- `src/simsopt/mhd/vmec.py`
- `src/simsopt/mhd/boozer.py`
- `src/simsopt/mhd/virtual_casing.py`
- `src/simsopt/mhd/vmec_diagnostics.py`
- `tests/mhd/test_vmec.py`
- `tests/mhd/test_boozer.py`
- `tests/mhd/test_virtual_casing.py`
- `examples/2_Intermediate/QH_fixed_resolution.py`
- `examples/2_Intermediate/QH_fixed_resolution_boozer.py`
- `examples/2_Intermediate/B_external_normal.py`
- `examples/2_Intermediate/stage_two_optimization_finite_beta.py`
- `examples/3_Advanced/single_stage_optimization.py`
- `examples/3_Advanced/single_stage_optimization_finite_beta.py`
- Git history for the MHD wrappers and single-stage examples.

Upstream JAX source references inspected locally:

- `vmec_jax/api.py`, `driver.py`, `optimization.py`, `quasisymmetry.py`,
  `booz_input.py`, `wout.py`, `free_boundary.py`, and docs on discrete adjoints,
  Simsopt comparison, validation, and optimization.
- `booz_xform_jax/core.py`, `jax_api.py`, `vmec.py`, tests, examples, and docs.
- `virtual_casing_jax/virtual_casing.py`, `functional.py`,
  `simsopt_virtual_casing.py`, tests, examples, and docs.

Online references reviewed:

- Simsopt JOSS paper: https://joss.theoj.org/papers/10.21105/joss.03525
- Simsopt single-stage documentation:
  https://simsopt.readthedocs.io/v1.9.1/example_single_stage.html
- Single-stage stellarator optimization paper:
  https://arxiv.org/abs/2302.10622
- Virtual-casing singular quadrature paper:
  https://arxiv.org/abs/1909.07417
- Boozer coordinates primary reference:
  https://www.osti.gov/biblio/6063300
- VMEC numerical-equilibrium context:
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/an-adjointbased-method-for-optimising-mhd-equilibria-against-the-infiniten-ideal-ballooning-mode/D7DC9ACDA1C77ED15FB12615F890B9A2
- JAX autodiff cookbook:
  https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
- `vmec_jax` README and docs:
  https://github.com/uwplasma/vmec_jax
- `booz_xform_jax` README and docs:
  https://github.com/uwplasma/booz_xform_jax

## Simsopt conventions to preserve

- MHD wrappers live under `src/simsopt/mhd` and are exported from
  `src/simsopt/mhd/__init__.py`.
- Optional external codes are imported in guarded `try` blocks, with
  `logger.debug(str(e))` on import failure and clear `RuntimeError` messages
  when a user tries to instantiate an unavailable wrapper.
- Wrappers should use `Optimizable` dependencies, `need_to_run_code`, and
  `recompute_bell()` so changes to boundary/profile dofs invalidate cached
  calculations.
- The current `Vmec` owns dofs `phiedge`, `curtor`, and `pres_scale`; the
  boundary surface owns boundary dofs. `VmecJax` should preserve this split.
- Existing examples are executable scripts with user parameters near the top,
  not command-line argument parsers. The new examples should follow that style.
- Tests are primarily `unittest` under `tests/<area>`, with optional-dependency
  `skipIf` guards and shared data in `tests/test_files`.
- Existing wrappers expose in-memory attributes after a run, especially
  `vmec.wout`, `Boozer.bx`, and `VirtualCasing.B_external_normal`. The JAX
  wrappers should expose the same high-value attributes where possible.
- Existing history favors incremental compatibility patches, parity tests, and
  optional external-code handling rather than broad refactors.

## Important upstream capability map

### `vmec_jax`

Useful existing pieces:

- Public drivers: `run_fixed_boundary()`, `run_free_boundary()`,
  `wout_from_fixed_boundary_run()`, and `write_wout_from_fixed_boundary_run()`.
- Public I/O: `read_indata`, `write_indata`, `load_input`, `load_wout`,
  `read_wout`, `state_from_wout`.
- Boundary helpers: `boundary_input_from_indata`, `boundary_from_input_convention`,
  `boundary_from_indata`, `apply_boundary_params`, `boundary_param_specs`.
- Optimization helpers: `FixedBoundaryExactOptimizer`,
  `make_qs_residuals_fn`, `make_qh_residuals_fn`, `create_x_scale`.
- Direct QS diagnostics: `quasisymmetry_ratio_residual_from_state()`.
- `booz_xform_inputs_from_state()` for a direct VMEC-to-Boozer pipeline.
- Discrete-adjoint support with checkpoint replay, JVP/VJP paths, JIT caches,
  and no VMEC2000 subprocess.

Integration implications:

- Prefer in-memory `WoutData` objects over temporary `wout_*.nc` files.
- `VmecJax.run()` should cache the `FixedBoundaryRun`, solved state, and
  `WoutData`, then expose a Simsopt-like `wout` view.
- The `VmecJax` wrapper needs a small compatibility layer for the old `Struct`
  style expected by Simsopt tests and wrappers.
- A public helper may be needed upstream to make exact objective/Jacobian
  construction less tied to the standalone examples.

### `booz_xform_jax`

Useful existing pieces:

- Legacy-compatible class `Booz_xform`.
- `read_wout()`, `read_wout_data()`, `init_from_vmec()`, `write_boozmn()`.
- `run()` that populates legacy-style attributes.
- `run_jax()` and `booz_xform_jax_impl()` for JIT/differentiable workflows.
- Streamed and vectorized Fourier modes selected by
  `BOOZ_XFORM_JAX_FOURIER_MODE`.

Integration implications:

- `BoozerJax` can mirror Simsopt's `Boozer` registry and surface-index mapping,
  then feed `VmecJax.wout` directly into `Booz_xform.read_wout_data()`.
- `QuasisymmetryJax` should preserve the old `Quasisymmetry` options:
  `normalization="B00"`, `normalization="symmetric"`, `weight="even"`,
  `weight="stellopt"`, and `weight="stellopt_ornl"`.
- For optimization, a lower-level functional path should avoid object mutation
  and use surface-major JAX arrays from `run_jax()` or `booz_xform_jax_impl()`.

### `virtual_casing_jax`

Useful existing pieces:

- `VirtualCasingJAX.setup()`, `compute_external_B()`,
  `compute_external_gradB()`, off-surface field/GradB functions, batch methods,
  and JIT wrappers.
- Functional API in `functional.py` for differentiating through surface
  coordinates and field data.
- `compute_external_B_autodiff()` with a custom JVP tied to computed GradB.
- A Simsopt-compatible `VirtualCasing` adapter already exists in
  `simsopt_virtual_casing.py`.

Integration implications:

- The existing adapter is valuable but still uses Simsopt's current `Vmec` and
  `B_cartesian()` path. In Simsopt, it should become `VirtualCasingJax` and
  accept both `VmecJax` and legacy `Vmec`/`wout` inputs where possible.
- The first Simsopt integration can reuse the adapter for parity. The second
  step should replace its VMEC-side field construction with `vmec_jax` state,
  geometry, and field helpers so finite-beta single-stage derivatives can flow
  through VMEC and virtual casing.

## Proposed Simsopt API

Add new names first, without changing existing imports:

- `simsopt.mhd.VmecJax`
- `simsopt.mhd.BoozerJax`
- `simsopt.mhd.QuasisymmetryJax`
- `simsopt.mhd.QuasisymmetryRatioResidualJax`
- `simsopt.mhd.VirtualCasingJax`

Keep the existing names unchanged initially:

- `Vmec`
- `Boozer`
- `Quasisymmetry`
- `QuasisymmetryRatioResidual`
- `VirtualCasing`

After parity and downstream example coverage are mature, consider aliasing or
runtime backend selection, for example `Vmec(..., backend="jax")`, but do not
start there. A side-by-side API makes tests, examples, and user migration much
cleaner.

## Implementation workstreams

### 1. Development environment

Planned integration environment:

1. Create one Simsopt integration virtual environment at
   `/Users/rogerio/local/simsopt_jax/.venv`.
2. Install the three JAX repositories in editable mode into that environment:
   `/Users/rogerio/local/vmec_jax_simsopt`,
   `/Users/rogerio/local/booz_xform_jax_simsopt`, and
   `/Users/rogerio/local/virtual_casing_jax_simsopt`.
3. Install Simsopt editable from `/Users/rogerio/local/simsopt_jax`.
4. Keep isolated upstream test environments optional unless upstream package
   tests need incompatible dependency pins.

Open dependency issue:

- Simsopt currently declares Python `>=3.8`, while the JAX packages require
  Python `>=3.9` or `>=3.10`. The new JAX wrappers should probably live behind
  an optional extra such as `JAX_MHD = ["vmec-jax", "booz_xform_jax",
  "virtual_casing_jax", "netCDF4"]` and skip cleanly on older Pythons.

### 2. `VmecJax` wrapper

Files to add or modify:

- Add `src/simsopt/mhd/vmec_jax.py`.
- Update `src/simsopt/mhd/__init__.py`.
- Add tests in `tests/mhd/test_vmec_jax.py`.
- Add docs under `docs/source/simsopt.mhd.rst` and MHD user docs.

Core behavior:

- Constructor accepts the same important arguments as `Vmec`:
  `filename=None`, `keep_all_files=False`, `verbose=True`, `ntheta=50`,
  `nphi=50`, and `range_surface="full torus"`.
- `mpi` should be accepted for API compatibility, but initial JAX execution can
  run on each process or proc0 with broadcast only after a clear design choice.
- Input files:
  - Load VMEC namelists through `vmec_jax.read_indata` or public wrappers.
  - Initialize `SurfaceRZFourier` from boundary coefficients.
  - Preserve `indata` access for resolution, profiles, and scalar parameters.
- Wout files:
  - Load through `vmec_jax.read_wout`.
  - Initialize boundary from `wout` data.
  - Mark object not runnable from a `wout`, matching current `Vmec` behavior.
- Dofs:
  - Own `phiedge`, `curtor`, `pres_scale`.
  - Depend on `boundary`.
  - Support `get_dofs()`, `set_dofs()`, profile setters, and cache invalidation.
- Run path:
  - Convert Simsopt boundary/profile state into `vmec_jax` input/boundary data.
  - Call `vmec_jax.run_fixed_boundary()` or `run_free_boundary()` based on
    `LFREEB`.
  - Cache `run`, `state`, and a Simsopt-compatible `wout` object.
  - Support `write_input()` and `get_input()` with VMEC-style namelist output.
- Diagnostics to implement at parity with `Vmec`:
  - `aspect()`
  - `volume()`
  - `iota_axis()`
  - `iota_edge()`
  - `mean_iota()`
  - `mean_shear()`
  - `external_current()`
  - `vacuum_well()`
  - `__repr__()`

Derivative behavior:

- Expose direct residual/Jacobian helpers for least-squares optimization, using
  `vmec_jax.FixedBoundaryExactOptimizer` or a thinner upstream API.
- Avoid having Simsopt silently finite-difference `VmecJax` objectives in the
  new examples. The examples should call exact residual/Jacobian functions.
- Add Taylor tests comparing exact gradients to finite differences on small
  problems.

Likely upstream needs in `vmec_jax`:

- Stable public conversion helpers between VMEC input convention and Simsopt
  `SurfaceRZFourier`.
- Public helpers for `external_current`, `vacuum_well`, and any fields missing
  from `WoutData` parity.
- A stable, documented optimizer API for "given boundary params, return
  residuals and exact Jacobian" without copying large chunks from examples.

### 3. `QuasisymmetryRatioResidualJax`

Files to add or modify:

- Add to `src/simsopt/mhd/vmec_jax.py` or a small separate module
  `src/simsopt/mhd/vmec_jax_diagnostics.py`.
- Add tests in `tests/mhd/test_vmec_jax.py` or
  `tests/mhd/test_vmec_jax_diagnostics.py`.

Behavior:

- Mirror current `QuasisymmetryRatioResidual` constructor and methods where
  practical:
  - `vmec`
  - `surfaces`
  - `helicity_m`
  - `helicity_n`
  - weights/residuals/total/profile behavior
- Under the hood, use `vmec_jax.quasisymmetry_ratio_residual_from_state()`.
- Handle helicity sign and field-period conventions explicitly in docstrings
  and tests. `vmec_jax` docs note field-period-unit conventions; Simsopt
  conventions must be preserved at the public Simsopt layer.
- Provide exact `dJ()` for scalar totals and exact residual Jacobians for
  least-squares workflows.

### 4. `BoozerJax` and `QuasisymmetryJax`

Files to add or modify:

- Add `src/simsopt/mhd/boozer_jax.py`.
- Update `src/simsopt/mhd/__init__.py`.
- Add `tests/mhd/test_boozer_jax.py`.

Behavior:

- Mirror current `Boozer`:
  - `BoozerJax(equil, mpol=32, ntor=32, verbose=False)`
  - `s` registry as a `set`
  - `register()`
  - `run()`
  - `s_used`
  - `s_to_index`
  - `need_to_run_code`
  - `_calls` test counter
- For `equil`:
  - Support `VmecJax` first.
  - Consider supporting legacy `Vmec` and `wout` for parity tests by using
    `read_wout_data()` or `read_wout()`.
- Map requested `s` values to VMEC half-grid indices identically to current
  `Boozer`.
- Use `booz_xform_jax.Booz_xform.read_wout_data(equil.wout)` where possible.
- Preserve `bx` attributes expected by tests and downstream code:
  `bmnc_b`, `xm_b`, `xn_b`, `compute_surfs`, and related spectra.
- `QuasisymmetryJax` mirrors `Quasisymmetry.J()` behavior and normalization.

Derivative behavior:

- Use `booz_xform_jax.run_jax()` or `booz_xform_jax.jax_api` for exact
  derivatives in new Boozer optimization examples.
- Keep the object-style `run()` for diagnostics and compatibility.

Likely upstream needs in `booz_xform_jax`:

- Optional `run_jax(populate=True)` or helper to populate legacy attributes from
  the JAX-native output without rerunning.
- A documented way to build Boozer inputs directly from `vmec_jax` state via
  `vmec_jax.booz_xform_inputs_from_state()` and
  `booz_xform_jax.prepare_booz_xform_constants_from_inputs()`.

### 5. `VirtualCasingJax`

Files to add or modify:

- Add `src/simsopt/mhd/virtual_casing_jax.py`.
- Update `src/simsopt/mhd/__init__.py`.
- Add `tests/mhd/test_virtual_casing_jax.py`.

Behavior:

- Mirror current `VirtualCasing`:
  - `from_vmec()`
  - `save()`
  - `load()`
  - `plot()`
  - `src_*` and `trgt_*` grid attributes
  - `gamma`
  - `B_total`
  - `unit_normal`
  - `B_external`
  - `B_external_normal`
  - `B_external_normal_extended`
- Reuse the existing adapter in
  `/Users/rogerio/local/virtual_casing_jax_simsopt/virtual_casing_jax/simsopt_virtual_casing.py`
  as the first implementation template.
- Initial parity route:
  - Support legacy `Vmec` and wout inputs using the current Simsopt
    `B_cartesian()` path, but JAX backend for the integral equation.
- Full JAX route:
  - Support `VmecJax` by deriving `gamma`, normals, and total field from the
    `vmec_jax` state/wout without VMEC2000 or Simsopt finite differences.
  - Use `VirtualCasingJAX` or `virtual_casing_jax.functional` so derivatives
    can flow through the surface and field data.

Derivative behavior:

- For finite-beta single-stage optimization, the important derivative is the
  derivative of the stage-two target `B_external_normal` with respect to VMEC
  surface/field variables. Plan to expose a function returning target and VJP
  or exact Jacobian blocks that can be composed with coil derivatives.
- Use `compute_external_gradB()` and `compute_external_B_autodiff()` for
  geometry derivatives where applicable.
- Compare exact JAX derivatives to finite differences on small grids.

Likely upstream needs in `virtual_casing_jax`:

- A public pure function that accepts Simsopt/JAX surface coordinates,
  `B_total`, and target-grid geometry and returns `B_external_normal` in the
  Simsopt `(nphi, ntheta)` convention.
- Shape-derivative examples/tests for `B_external_normal`, not only off-surface
  `B_external`.
- Clear API for half-period vs full-field-period conventions and source/target
  axis ordering.

### 6. Examples to add

All new examples should follow the existing Simsopt example style:

- User parameters at the top.
- Same input files and output directory conventions as the non-JAX examples.
- `proc0_print` for MPI-like examples.
- No hidden command-line argument parser unless the original example had one.
- Small defaults suitable for CI or quick smoke tests.

Add:

1. `examples/2_Intermediate/QH_fixed_resolution_jax.py`
   - Follows `QH_fixed_resolution.py`.
   - Uses `VmecJax` and `QuasisymmetryRatioResidualJax`.
   - Uses exact autodiff/discrete-adjoint derivatives from `vmec_jax`, not
     `least_squares_mpi_solve(..., grad=True)` finite differences.
   - Reads `examples/2_Intermediate/inputs/input.nfp4_QH_warm_start`.

2. `examples/2_Intermediate/QH_fixed_resolution_boozer_jax.py`
   - Follows `QH_fixed_resolution_boozer.py`.
   - Uses `VmecJax`, `BoozerJax`, and `QuasisymmetryJax`.
   - Uses autodiff through `vmec_jax` and `booz_xform_jax`.

3. `examples/2_Intermediate/B_external_normal_jax.py`
   - Follows `B_external_normal.py`.
   - Uses `VirtualCasingJax`.
   - Preserves save/load demonstration and printed array snippet.

4. `examples/2_Intermediate/stage_two_optimization_finite_beta_jax.py`
   - Follows `stage_two_optimization_finite_beta.py`.
   - Uses `VirtualCasingJax` and optionally `VmecJax` for target generation.
   - Keeps all coil functionality in Simsopt.
   - Uses the same W7-X/QH reference data and current-sign logic.

5. `examples/3_Advanced/single_stage_optimization_jax.py`
   - Follows `single_stage_optimization.py`.
   - Replaces VMEC finite-difference surface gradients with exact
     `vmec_jax` derivative blocks.
   - Keeps analytic/mixed coil-surface derivative terms already in Simsopt.

6. `examples/3_Advanced/single_stage_optimization_finite_beta_jax.py`
   - Follows `single_stage_optimization_finite_beta.py`.
   - Uses `VmecJax` and `VirtualCasingJax`.
   - Replaces finite-difference surface gradient of `fun_J` with JAX autodiff
     through VMEC and virtual casing.
   - Retains the initial stage-two coil optimization to match the published
     single-stage workflow.

Note: the user mentioned `state_two_optimization_jax.pt`; this appears to mean
the existing stage-two finite-beta workflow and/or the
`virtual_casing_jax/examples/simsopt_stage_two_optimization_finite_beta.py`
template.

### 7. Tests and validation

New tests:

- `tests/mhd/test_vmec_jax.py`
  - Optional import skips for `vmec_jax`.
  - Initialization from input and wout.
  - Boundary coefficient parity with `SurfaceRZFourier`.
  - Dof get/set and cache invalidation.
  - `aspect`, `volume`, iota metrics, `mean_shear`, `external_current`,
    and `vacuum_well` parity against current `Vmec` on existing test files.
  - Exact derivative/Taylor tests on small fixed-boundary problems.

- `tests/mhd/test_boozer_jax.py`
  - Registry behavior identical to `Boozer`.
  - `s_to_index` and `compute_surfs` parity.
  - Circular tokamak and li383 Boozer spectra parity against existing
    `boozmn_*` references and/or current `Boozer`.
  - `QuasisymmetryJax` residual parity for QA, QP, QH, normalization, and
    weight options.
  - JAX functional derivative check for a small Boozer objective.

- `tests/mhd/test_virtual_casing_jax.py`
  - `from_vmec()` input, wout, and object initialization.
  - Save/load parity for all public attributes.
  - BNORM benchmark parity.
  - Vacuum case where `B_external` approximately equals `B_total` and normal
    component is approximately zero.
  - Stellarator-symmetry and full-field-period parity.
  - JAX derivative check on small grids.

- Example smoke tests, either in existing example test infrastructure or a new
  lightweight test file:
  - Each new example imports and executes with reduced iteration counts.
  - Avoid writing large artifacts outside temp directories in CI.

Validation matrix:

- Old `Vmec` vs `VmecJax`:
  - fixed-boundary low-resolution inputs,
  - wout-loaded diagnostics,
  - free-boundary cases where `vmec_jax` support is mature.
- Old `Boozer` vs `BoozerJax`:
  - circular tokamak,
  - li383,
  - asymmetric cases.
- Old `VirtualCasing` vs `VirtualCasingJax`:
  - BNORM reference,
  - vacuum reference,
  - stellsym/full-period agreement.
- Finite-beta workflows:
  - initial objective parity,
  - first derivative parity/Taylor tests,
  - short optimization trajectory sanity checks.

Performance gates:

- Always distinguish cold JIT compile time from warm repeated-solve time.
- Enable JAX x64 in tests to match VMEC/BOOZ/virtual-casing double precision.
- Exercise JIT cache reuse for repeated solve/transform/integral calls.
- Track wall time and memory for representative max-mode 1 and 2 examples.

## Open design decisions

1. Dependency policy:
   - Use an optional Simsopt extra for the JAX MHD stack first.
   - Decide later whether replacement of VMEC2000 makes these dependencies
     required.

2. MPI policy:
   - Current Simsopt VMEC runs use `MpiPartition` and parallel finite
     differences. JAX exact derivatives reduce the need for multi-process
     finite differencing, but examples still use `proc0_print` and some MPI
     plumbing. Decide whether `VmecJax` runs on every rank, proc0 plus
     broadcast, or supports both.

3. Wout compatibility:
   - Prefer in-memory `WoutData`, but some downstream code expects mutable
     attributes with SciPy NetCDF transposition conventions. A small
     compatibility adapter is safer than changing callers.

4. Objective API:
   - Existing Simsopt least-squares wrappers are finite-difference oriented for
     black-box functions. New exact-JAX examples may need a narrow
     `residual_and_jacobian` utility rather than forcing exact derivatives into
     old finite-difference paths.

5. Virtual-casing shape derivatives:
   - The full finite-beta single-stage objective requires differentiating the
     target field as the VMEC surface changes. This is the highest-risk part
     and should be built on small-grid derivative tests before examples depend
     on it.

6. Long-term replacement strategy:
   - Keep new classes side by side until parity and examples are robust.
   - Then add backend selection or aliases.
   - Only remove/replace old wrappers once downstream examples, docs, and tests
     are passing with JAX backends.

## Risks and mitigations

- Risk: JAX package Python requirements exceed Simsopt's current lower bound.
  - Mitigation: optional extra and import skips; document Python requirement.

- Risk: VMEC2000 parity differs for edge cases, free-boundary inputs, `lasym`,
  pressure/current profile modes, or output fields.
  - Mitigation: parity test matrix before using wrappers in examples.

- Risk: JIT compilation dominates short examples.
  - Mitigation: distinguish cold/warm timings, use persistent JIT cache, keep
    CI examples tiny.

- Risk: Object mutation in Simsopt wrappers conflicts with JAX functional
  transformations.
  - Mitigation: keep compatibility wrappers object-oriented, but build exact
    derivative paths from functional helper APIs.

- Risk: `VirtualCasingJax` target derivatives are not yet exposed in the exact
  Simsopt convention.
  - Mitigation: upstream a pure functional normal-field API and test against
    finite differences before the finite-beta single-stage example.

- Risk: Axis/sign conventions differ among VMEC, Simsopt, Boozer, BNORM, and
  virtual casing.
  - Mitigation: explicit sign-convention tests for helicity, current sign, and
    `B_external_normal` symmetry.

## Likely upstream PRs

Open branches and PRs in the JAX repositories only when the Simsopt wrapper
implementation demonstrates a concrete gap. Expected candidates:

- `vmec_jax`
  - Public Simsopt conversion helpers.
  - Public exact residual/Jacobian builder for Simsopt-style objectives.
  - Missing wout fields or diagnostics needed by Simsopt parity.

- `booz_xform_jax`
  - Direct `BoozXformInputs` pipeline from `vmec_jax` state.
  - Populate legacy attributes from `run_jax()` output.
  - Additional parity tests for Simsopt-specific surface registration.

- `virtual_casing_jax`
  - Pure functional `B_external_normal` API in Simsopt axis order.
  - Shape derivative/JVP tests for on-surface normal field.
  - Cleaner import path for the Simsopt-compatible adapter.

## Step-by-step execution plan

1. Environment setup and smoke imports.
   - Create and install the integration venv.
   - Run import smoke tests for all four checkouts.
   - Run selected upstream JAX package tests that do not require large assets.

2. Add `VmecJax` skeleton.
   - Imports, constructor, boundary conversion, wout loading, dofs, and
     diagnostics from loaded wout.
   - Tests for initialization and diagnostics from existing wout files.

3. Add runnable fixed-boundary `VmecJax`.
   - Input file run path through `vmec_jax`.
   - Wout compatibility adapter.
   - Parity tests vs old `Vmec` on low-resolution fixed-boundary cases.

4. Add exact `QuasisymmetryRatioResidualJax`.
   - Residual parity.
   - Total/profile parity where applicable.
   - Taylor tests for exact derivatives.

5. Add `BoozerJax` and `QuasisymmetryJax`.
   - Object compatibility first.
   - Functional derivative path second.
   - Parity against boozmn references and old `Boozer`.

6. Add `VirtualCasingJax`.
   - Port existing adapter into Simsopt style.
   - Add save/load/plot parity.
   - Add full JAX VMEC-side data path.
   - Add derivative tests.

7. Add intermediate examples.
   - `QH_fixed_resolution_jax.py`
   - `QH_fixed_resolution_boozer_jax.py`
   - `B_external_normal_jax.py`
   - `stage_two_optimization_finite_beta_jax.py`

8. Add advanced examples.
   - `single_stage_optimization_jax.py`
   - `single_stage_optimization_finite_beta_jax.py`

9. Full validation.
   - Unit tests for new wrappers.
   - Selected old-vs-new parity tests.
   - Example smoke tests.
   - Warm performance comparisons.

10. Upstream cleanup PRs.
    - Open branches/PRs in `vmec_jax`, `booz_xform_jax`, and/or
      `virtual_casing_jax` for concrete API gaps found during integration.

## Running log

### 2026-04-25

- Cloned fresh upstream repositories to the requested paths.
- Created local Simsopt branch `plan/jax-finite-beta-single-stage`.
- Inspected existing Simsopt MHD wrappers, tests, examples, docs, and relevant
  git history.
- Inspected `vmec_jax`, `booz_xform_jax`, and `virtual_casing_jax` APIs, tests,
  examples, and docs.
- Reviewed online Simsopt, single-stage optimization, virtual-casing,
  Boozer-coordinate, VMEC, and JAX autodiff references.
- Added this uncommitted `plan.md` file.
- Created `/Users/rogerio/local/simsopt_jax/.venv` with Python 3.13.7.
- Installed editable checkouts of `vmec_jax`, `booz_xform_jax`,
  `virtual_casing_jax`, and `simsopt` into that venv.
- Ran import smoke test successfully:
  - `simsopt 1.10.7.dev402+g1b0cc3a96`
  - `vmec_jax` imported successfully; package has no `__version__`.
  - `booz_xform_jax 0.1.0`
  - `virtual_casing_jax` imported successfully; package has no `__version__`.
- Added the first Simsopt `VmecJax` wrapper slice:
  - Optional `vmec_jax` import and `simsopt.mhd` export.
  - Input-file initialization without MPI or VMEC2000.
  - Attribute-style `indata` view for common VMEC scalar dofs.
  - Boundary transfer from `SurfaceRZFourier` into vmec_jax namelist data.
  - Loaded-`wout` compatibility with existing Simsopt diagnostics.
  - Runnable fixed-boundary path through `vmec_jax.run_fixed_boundary()`.
- Added focused parity tests in `tests/mhd/test_vmec_jax.py`.
- Installed `pytest` in the integration venv and ran:
  - `python -m pytest tests/mhd/test_vmec_jax.py -q`
    (`6 passed`, one NumPy binary-size runtime warning)
  - `python -m pytest tests/mhd/test_vmec.py -q`
    (`6 passed, 14 skipped`)
- Exercised `VmecJax.run()` on `tests/test_files/input.li383_low_res`
  against `wout_li383_low_res_reference.nc`:
  - aspect relative error: `1.0e-15`
  - volume relative error: `6.0e-16`
  - mean-iota absolute error: `9.0e-4`
  - residuals: `fsqr=9.8e-14`, `fsqz=1.9e-14`, `fsql=7.1e-15`
- Promoted this low-resolution solve comparison into
  `tests/mhd/test_vmec_jax.py`.
- Re-ran `python -m pytest tests/mhd/test_vmec_jax.py -q`
  (`7 passed`, two warnings from NumPy/JAX internals).
- Added the first Simsopt `BoozerJax` and `QuasisymmetryJax` wrapper slice:
  - Optional `booz_xform_jax` import and `simsopt.mhd` export.
  - Simsopt-style surface registry and cache invalidation.
  - In-memory transfer from legacy `Vmec.wout` or `VmecJax.wout` to
    `booz_xform_jax.Booz_xform`.
  - Reuse of the existing quasisymmetry residual semantics through a JAX-named
    objective class.
- Added `tests/mhd/test_boozer_jax.py`:
  - registry behavior without the legacy `booz_xform` package,
  - comparison with `tests/test_files/boozmn_li383_low_res.nc`,
  - parity between legacy `Vmec` loaded-wout input and `VmecJax` loaded-wout
    input.
- Ran:
  - `python -m pytest tests/mhd/test_boozer_jax.py -q` (`3 passed`)
  - `python -m pytest tests/mhd/test_boozer.py -q` (`5 skipped`)
- Added `QuasisymmetryRatioResidualJax` in Simsopt:
  - Same `compute()`, `residuals()`, `profile()`, and `total()` workflow as
    the existing VMEC-only quasisymmetry metric.
  - Delegates to `vmec_jax.quasisymmetry_ratio_residual_from_wout`.
  - Matches the existing Simsopt implementation at roundoff for loaded-wout
    Li383 test cases.
- Found an upstream `vmec_jax` API gap: intermediate diagnostic arrays were
  computed but not returned by `quasisymmetry_ratio_residual_from_wout`.
- Opened `vmec_jax` branch `simsopt-qs-diagnostics`, committed the diagnostic
  return fields, and opened PR https://github.com/uwplasma/vmec_jax/pull/8.
- Ran upstream `vmec_jax` test:
  - `python -m pytest tests/test_quasisymmetry.py -q`
    (`3 passed, 1 skipped`, two warnings from JAX/NumPy internals)
- Added `tests/mhd/test_vmec_diagnostics_jax.py` and ran:
  - `python -m pytest tests/mhd/test_vmec_diagnostics_jax.py -q`
    (`3 passed`)
- Added `VirtualCasingJax` in Simsopt:
  - Uses `virtual_casing_jax.VirtualCasingJAX` for the integral solve.
  - Accepts `VmecJax`, legacy `Vmec` loaded-wout objects, or file paths
    converted through `VmecJax`.
  - Preserves the existing `VirtualCasing` attributes and inherited
    save/load/plot behavior.
- Added `tests/mhd/test_virtual_casing_jax.py`:
  - legacy `Vmec` wout and `VmecJax` wout parity,
  - small-grid vacuum normal-field check,
  - save/load round trip.
- Ran:
  - `python -m pytest tests/mhd/test_virtual_casing_jax.py -q` (`3 passed`)
  - `python -m pytest tests/mhd/test_virtual_casing.py -q` (`6 skipped`)

## Immediate next actions

- Start adding JAX versions of the requested intermediate examples, beginning
  with `QH_fixed_resolution_jax.py` and `QH_fixed_resolution_boozer_jax.py`.

## Implementation log - example slice 1

- Added `examples/2_Intermediate/QH_fixed_resolution_jax.py`:
  - follows the original fixed-resolution QH example structure with top-level
    parameters and the same warm-start input file,
  - uses `vmec_jax.FixedBoundaryExactOptimizer`,
  - uses the vmec_jax exact discrete-adjoint Jacobian rather than finite
    differences,
  - saves `input.QH_fixed_resolution_jax_final`.
- Added `examples/2_Intermediate/QH_fixed_resolution_boozer_jax.py`:
  - follows the original Boozer-targeted QH example,
  - builds `vmec_jax -> booz_xform_jax` residuals in memory,
  - differentiates the VMEC solve with the exact discrete-adjoint path and the
    Boozer residual with JAX autodiff,
  - uses the existing Simsopt `Quasisymmetry` convention for nonsymmetric
    Boozer modes normalized by B00,
  - saves `input.QH_fixed_resolution_boozer_jax_final`.
- Added `examples/2_Intermediate/B_external_normal_jax.py`:
  - follows `B_external_normal.py`,
  - swaps in `VirtualCasingJax`,
  - demonstrates save/load compatibility with the existing virtual-casing file
    format.
- Ran syntax checks:
  - `python -m py_compile examples/2_Intermediate/QH_fixed_resolution_jax.py examples/2_Intermediate/QH_fixed_resolution_boozer_jax.py examples/2_Intermediate/B_external_normal_jax.py`

## Implementation log - example slice 2

- Added `examples/2_Intermediate/stage_two_optimization_finite_beta_jax.py`:
  - follows `stage_two_optimization_finite_beta.py`,
  - keeps all coil geometry/current objects and derivatives in Simsopt,
  - replaces the finite-beta target-field calculation with
    `VirtualCasingJax`,
  - uses `VmecJax` for the total-current diagnostic from the target wout.
- Added `examples/3_Advanced/single_stage_optimization_jax.py`:
  - follows `single_stage_optimization.py`,
  - keeps the Simsopt coil objective and mixed coil-surface derivative term,
  - swaps in `VmecJax` and `QuasisymmetryRatioResidualJax` for the stage-I
    equilibrium and QS objective.
- Added `examples/3_Advanced/single_stage_optimization_finite_beta_jax.py`:
  - follows `single_stage_optimization_finite_beta.py`,
  - keeps the Simsopt coil objective and coil derivatives,
  - swaps in `VmecJax`, `QuasisymmetryRatioResidualJax`, and
    `VirtualCasingJax`,
  - recomputes the virtual-casing target through the JAX path when surface
    degrees of freedom change.
- Ran syntax checks:
  - `python -m py_compile examples/2_Intermediate/stage_two_optimization_finite_beta_jax.py examples/3_Advanced/single_stage_optimization_jax.py examples/3_Advanced/single_stage_optimization_finite_beta_jax.py`
- Re-ran focused JAX wrapper tests after the example additions:
  - `python -m pytest tests/mhd/test_vmec_jax.py tests/mhd/test_boozer_jax.py tests/mhd/test_vmec_diagnostics_jax.py tests/mhd/test_virtual_casing_jax.py -q`
    (`16 passed`, one vmec_jax/JAX deprecation warning)

## Validation plots and end-to-end smoke checks

- Generated comparison plots under `results/jax_mhd_integration/`:
  - `vmec_jax_vs_reference.png`: LI383 low-resolution iota profile and scalar
    diagnostics comparing a fresh `VmecJax` solve with the existing VMEC2000
    reference wout.
  - `boozer_jax_vs_reference.png`: LI383 edge Boozer spectrum comparing
    `BoozerJax` with the existing `boozmn_li383_low_res.nc` reference.
  - `virtual_casing_jax_normal_field.png`: vacuum-equilibrium
    `VirtualCasingJax` normal-field check.
  - `metrics.json`: scalar metrics used in the plots.
- Plot metrics:
  - LI383 aspect: reference `4.354967596750808`, JAX
    `4.354967596750813`.
  - LI383 volume: reference `2.9813872701632924`, JAX
    `2.9813872701632906`.
  - LI383 mean iota: reference `0.5544911906253179`, JAX
    `0.5535895178061357`.
  - Fresh JAX residuals: `fsqr=9.765e-14`, `fsqz=1.861e-14`,
    `fsql=7.144e-15`.
  - Boozer edge-spectrum max absolute error against the reference boozmn file:
    `7.073e-16`.
  - Vacuum virtual-casing max `|B_external_normal|`: `3.709e-3`.
- Ran the smallest end-to-end exact optimization example:
  - `python examples/2_Intermediate/QH_fixed_resolution_boozer_jax.py`
  - completed successfully with `max_nfev=1`;
  - initial/final QS objective `0.0017286553289639218` as expected for a
    one-evaluation smoke test;
  - final aspect ratio `7.000345969477066`.

## Online references checked

- Simsopt repository and documentation:
  - https://github.com/hiddenSymmetries/simsopt
  - https://simsopt.readthedocs.io/v1.9.1/example_single_stage.html
- `vmec_jax` README and exact-discrete-adjoint optimization notes:
  - https://github.com/uwplasma/vmec_jax
  - https://raw.githubusercontent.com/uwplasma/vmec_jax/main/README.md
- `booz_xform_jax` repository:
  - https://github.com/uwplasma/booz_xform_jax
- Single-stage stellarator optimization paper:
  - https://arxiv.org/abs/2302.10622
- Virtual-casing quadrature references:
  - https://arxiv.org/abs/1909.07417
  - https://arxiv.org/abs/2404.02799

## Implementation log - VmecJax compatibility slice

- Compared the existing `Vmec` public method surface with `VmecJax`.
- Added Simsopt `Profile` object support to `VmecJax`:
  - `pressure_profile`, `current_profile`, and `iota_profile` are converted
    into VMEC/JAX namelist arrays before writing or running,
  - power-series and spline-like profile types follow the same workflow as the
    existing `Vmec` wrapper,
  - current-profile `curtor` is updated from the integrated profile when the
    profile specifies current derivative form.
- Made `VmecJax.get_max_mn()` safe for objects initialized from a `wout` file.
- Added tests covering profile transfer and `wout`-initialized `get_max_mn()`.
- Ran:
  - `python -m pytest tests/mhd/test_vmec_jax.py -q`
    (`9 passed`, one vmec_jax/JAX deprecation warning)
  - `python -m pytest tests/mhd/test_vmec_jax.py tests/mhd/test_boozer_jax.py tests/mhd/test_vmec_diagnostics_jax.py tests/mhd/test_virtual_casing_jax.py -q`
    (`18 passed`, one vmec_jax/JAX deprecation warning)

## Documentation log - JAX MHD integration page

- Added `docs/source/example_jax_mhd.rst` and linked it from the Tutorials
  toctree in `docs/source/index.rst`.
- Copied the generated validation plots into the documentation tree:
  - `docs/source/jax_mhd_vmec_jax_vs_reference.png`
  - `docs/source/jax_mhd_boozer_jax_vs_reference.png`
  - `docs/source/jax_mhd_virtual_casing_jax_normal_field.png`
- The page documents:
  - the new JAX wrapper interfaces,
  - the requested JAX examples,
  - validation metrics and plots,
  - the focused JAX MHD test command,
  - current limitations and remaining work.
- Installed the documented Sphinx dependencies into the integration venv and
  ran:
  - `python -m sphinx -b html docs/source docs/build/html`
    (`build succeeded`, with existing project documentation warnings)
- Added the new JAX MHD modules to `docs/source/simsopt.mhd.rst` so the API
  reference covers:
  - `simsopt.mhd.vmec_jax`
  - `simsopt.mhd.boozer_jax`
  - `simsopt.mhd.vmec_diagnostics_jax`
  - `simsopt.mhd.virtual_casing_jax`
- Rebuilt the docs after adding API coverage:
  - `python -m sphinx -b html docs/source docs/build/html`
    (`build succeeded`, with existing project documentation warnings)

## Testing log - requested JAX examples

- Added `tests/mhd/test_jax_examples.py`.
- The test compiles all requested JAX example scripts:
  - `B_external_normal_jax.py`
  - `QH_fixed_resolution_jax.py`
  - `QH_fixed_resolution_boozer_jax.py`
  - `stage_two_optimization_finite_beta_jax.py`
  - `single_stage_optimization_jax.py`
  - `single_stage_optimization_finite_beta_jax.py`
- Ran:
  - `python -m pytest tests/mhd/test_jax_examples.py -q`
    (`1 passed`, six subtests)
  - `python -m pytest tests/mhd/test_jax_examples.py tests/mhd/test_vmec_jax.py tests/mhd/test_boozer_jax.py tests/mhd/test_vmec_diagnostics_jax.py tests/mhd/test_virtual_casing_jax.py -q`
    (`19 passed`, six subtests, one vmec_jax/JAX deprecation warning)

## Implementation log - vectorized BoozerJax backend

- Updated `BoozerJax.run()` to prefer `booz_xform_jax.Booz_xform.run_jax()`
  for stellarator-symmetric equilibria and populate the same Booz_xform
  attributes consumed by `QuasisymmetryJax`.
- Kept the compatibility `run()` path for asymmetric equilibria until the
  vectorized upstream JAX API returns all asymmetric sine spectra.
- Found an upstream `booz_xform_jax` API gap: the vectorized JAX kernel
  computes `gmnc_b` but did not return it.
- Opened `booz_xform_jax` branch `simsopt-run-jax-output`, committed
  `Return gmnc_b from JAX API`, and opened
  https://github.com/uwplasma/booz_xform_jax/pull/1.
- Ran upstream:
  - `python -m pytest tests/test_jax_api.py -q` (`3 passed`)
- Added a Simsopt test assertion that `BoozerJax` used the vectorized JAX
  output path for the stellsym LI383 case.
- Added a compatibility guard in `BoozerJax`: if the installed upstream
  `run_jax()` API lacks `gmnc_b`, the wrapper logs a warning and falls back
  to the attribute-populating `run()` path instead of failing with a
  `KeyError`.
- Updated the Simsopt JAX MHD documentation page to tie the Boozer plot to
  the vectorized backend and document the `gmnc_b` upstream API requirement.
- Ran:
  - `python -m pytest tests/mhd/test_boozer_jax.py -q` (`3 passed`)
  - `python -m pytest tests/mhd/test_jax_examples.py tests/mhd/test_vmec_jax.py tests/mhd/test_boozer_jax.py tests/mhd/test_vmec_diagnostics_jax.py tests/mhd/test_virtual_casing_jax.py -q`
    (`19 passed`, six subtests, one vmec_jax/JAX deprecation warning)

## Implementation log - exact single-stage stage-I gradient

- Updated `examples/3_Advanced/single_stage_optimization_jax.py` so the
  stage-I VMEC/QS objective uses
  `vmec_jax.FixedBoundaryExactOptimizer.objective_and_gradient_fun()` instead
  of `MPIFiniteDifference`.
- Added an explicit mapping from Simsopt `SurfaceRZFourier` free-DOF names to
  `vmec_jax` boundary parameter specs. The exact gradient is returned in
  `vmec_jax` parameter order and scattered back into the Simsopt surface DOF
  order before adding the existing mixed Biot-Savart surface derivative.
- Left coils, coil regularization derivatives, and the mixed
  squared-flux/surface derivative in native Simsopt code, matching the original
  single-stage workflow.
- Added a regression check that the exact-stage parameter specs for
  `input.nfp4_QH_warm_start` match the active Simsopt surface DOFs.
- Updated the documentation page to state which advanced example now uses the
  exact stage-I gradient and to mark the remaining finite-beta virtual-casing
  shape-derivative gap explicitly.
- Ran:
  - `python -m py_compile examples/3_Advanced/single_stage_optimization_jax.py`
  - `python -m pytest tests/mhd/test_jax_examples.py -q`
    (`2 passed`, six subtests, one vmec_jax/JAX deprecation warning)

## Upstream log - virtual-casing normal-field JVP API

- Investigated `virtual_casing_jax` for the finite-beta shape-derivative
  blocker in `single_stage_optimization_finite_beta_jax.py`.
- Found that the existing functional API differentiates off-surface fields
  with respect to surface coordinates, but there was no direct on-surface
  `B_external_normal` functional helper.
- Added upstream branch `simsopt-normal-field-functional` in
  `virtual_casing_jax` with:
  - `target_surface_normal(...)`,
  - `compute_external_B_normal_functional(...)`,
  - projection parity tests,
  - a forward-mode JVP Taylor test for the on-surface normal field.
- Confirmed that forward-mode JVPs through the on-surface singular quadrature
  are finite and match finite differences on the small test geometry.
- Also found that reverse-mode gradients through the same singular quadrature
  currently produce NaNs, so the Simsopt finite-beta exact path should use JVP
  columns for now rather than reverse-mode gradients.
- Ran upstream:
  - `python -m pytest tests/test_functional_api.py -q` (`4 passed`)
- Opened upstream PR:
  - https://github.com/uwplasma/virtual_casing_jax/pull/1
- Updated the Simsopt documentation page to reference this upstream API as the
  next required piece for replacing the finite-beta whole-objective finite
  differences.

## Implementation log - Simsopt virtual-casing normal-field JVP bridge

- Revisited the full implementation state against this plan. The side-by-side
  wrappers, diagnostics, requested JAX examples, validation plots, docs, and
  upstream PRs are in place. The remaining highest-value workstream is still
  the finite-beta single-stage derivative path.
- Added Simsopt-level helpers in `src/simsopt/mhd/virtual_casing_jax.py`:
  - `B_external_normal_from_data(...)`
  - `B_external_normal_jvp_from_data(...)`
- These helpers expose the upstream functional `virtual_casing_jax` API in
  Simsopt array convention, with optional explicit `unit_normal` and
  `tangent_unit_normal` inputs. This is important because the existing
  `VirtualCasingJax.from_vmec()` projection uses the Simsopt target-surface
  normal, which is not identical to the upstream functional target normal on
  the half-period grid.
- Added tests showing:
  - `B_external_normal_from_data(..., unit_normal=vc.unit_normal)` reproduces
    the `VirtualCasingJax.from_vmec()` stored `B_external_normal`.
  - `B_external_normal_jvp_from_data(...)` matches finite differences for a
    scalar normal-field objective, including perturbations to source geometry,
    total field, and target normal.
- This keeps the plan on track: the new helper is the first Simsopt-owned
  derivative bridge needed before replacing the finite-beta example's
  whole-objective `MPIFiniteDifference` path.
- Immediate next steps:
  - Compose `B_external_normal_jvp_from_data(...)` with Simsopt surface normal
    derivatives and VMEC-JAX state/field tangents.
  - Replace the virtual-casing target part of
    `single_stage_optimization_finite_beta_jax.py` with exact directional
    derivatives column-by-column.
  - Add a short optimization regression comparing the first finite-beta
    objective and gradient against the current finite-difference path.

## Implementation log - virtual-casing surface-Jacobian assembly

- Inspected `examples/3_Advanced/single_stage_optimization_finite_beta_jax.py`
  against the non-JAX finite-beta example and the vacuum
  `single_stage_optimization_jax.py` exact-gradient implementation.
- Confirmed that the finite-beta JAX example still mirrors the original
  whole-objective `MPIFiniteDifference` path for surface variables. This is
  plan-aligned as a temporary compatibility step, but it is now the main
  derivative replacement target.
- Added `B_external_normal_jacobian_from_surface(...)` in
  `src/simsopt/mhd/virtual_casing_jax.py`.
  - It uses `surface.dgamma_by_dcoeff()` and
    `surface.dunitnormal_by_dcoeff()` to assemble the surface-coefficient
    columns of the virtual-casing target derivative.
  - It accepts optional `B_total_tangents` columns so VMEC-JAX field-state
    tangents can be inserted without changing the projection logic.
  - It currently assumes the virtual-casing source and target grids are the
    same surface quadrature grid, which matches the finite-beta JAX example.
- Added a Taylor test on a small `SurfaceRZFourier` that checks the assembled
  `B_external_normal` Jacobian against finite differences while perturbing
  source geometry, total field, and target normal.
- This completes the first Simsopt-side composition of the upstream
  virtual-casing JVP with Simsopt surface derivative machinery.
- Next steps:
  - Obtain or build VMEC-JAX tangent columns for `B_total` on the target
    surface.
  - Use those `B_total_tangents` in
    `B_external_normal_jacobian_from_surface(...)`.
  - Add the resulting target derivative to the finite-beta `SquaredFlux`
    surface gradient and compare against `MPIFiniteDifference`.

## Implementation log - VMEC-JAX boundary field parity and Simsopt wiring

- Found a concrete upstream parity gap while trying to route
  `VirtualCasingJax.from_vmec(VmecJax(...))` through the new VMEC-JAX
  boundary-field helper:
  - the initial helper evaluated the last half-mesh `bsup` slice directly,
  - Simsopt's legacy `B_cartesian(...)` uses VMEC's boundary extrapolation
    `1.5 * edge - 0.5 * previous`.
- Updated the VMEC-JAX upstream branch and open PR:
  - https://github.com/uwplasma/vmec_jax/pull/8
  - Added `b_cartesian_from_state(...)` with public API exports and tests.
  - Added boundary extrapolation for the selected outer surface.
  - Verified the helper matches Simsopt's legacy `B_cartesian(...)` vector
    field to roundoff on a loaded-wout QA low-resolution grid.
- Added Simsopt `B_cartesian_jax(...)` and `VmecJax.B_cartesian(...)`.
  - The function mirrors the legacy `B_cartesian(...)` grid arguments and
    return convention.
  - Loaded-wout calls use the stored-wout `bsup` parity path.
  - Runnable `VmecJax` calls use the solved VMEC-JAX state/input path by
    default, keeping the value path aligned with the future tangent path.
- Updated `VirtualCasingJax.from_vmec(...)` so a `VmecJax` input obtains
  `B_total` through `B_cartesian_jax(...)` rather than the legacy VMEC
  diagnostic.
- Added regression coverage:
  - `B_cartesian_jax(...)` and `VmecJax.B_cartesian(...)` match legacy
    `B_cartesian(...)` for a loaded-wout QA low-resolution case.
  - `VirtualCasingJax.from_vmec(VmecJax(...))` still matches the legacy
    `Vmec` path after the routing change.
- This keeps us on the plan:
  - completed the value-path replacement of `B_cartesian` for `VmecJax`,
  - upstreamed the parity fix required for safe replacement,
  - left the remaining derivative target as VMEC-JAX state tangent columns
    for `B_total`, then insertion into
    `B_external_normal_jacobian_from_surface(...)`.

## Implementation log - finite-beta exact target-gradient path

- Added `B_cartesian_jax_tangent_columns(...)` in
  `src/simsopt/mhd/vmec_jax.py`.
  - It accepts a `vmec_jax.FixedBoundaryExactOptimizer`, parameter vector, and
    a Simsopt grid.
  - It returns the VMEC-JAX boundary Cartesian field with shape
    `(nphi, ntheta, 3)` and exact accepted-point tangent columns with shape
    `(nphi, ntheta, 3, nparams)`.
  - The tangent convention matches VMEC-JAX's exact optimizer Jacobian:
    accepted-point tape replay plus the frozen-axis initial-state derivative
    used by `FixedBoundaryExactOptimizer.jacobian_fun(...)`.
- Added a focused Simsopt regression that builds a VMEC-JAX exact optimizer
  whose residual is exactly the boundary Cartesian field, then verifies
  `B_cartesian_jax_tangent_columns(...)` reproduces both the residual vector
  and exact Jacobian columns.
- Updated
  `examples/3_Advanced/single_stage_optimization_finite_beta_jax.py`:
  - Removed the whole-objective `MPIFiniteDifference` surface-gradient path.
  - Added the same exact VMEC-JAX stage-I optimizer pattern used by the vacuum
    JAX single-stage example.
  - Composed VMEC-JAX `B_total` tangent columns with
    `B_external_normal_jacobian_from_surface(...)`.
  - Added the resulting target derivative to the local `SquaredFlux` surface
    gradient while keeping native Simsopt coil objects and coil derivatives.
- Updated the JAX MHD documentation page to describe the new tangent helper and
  the finite-beta exact target-gradient path.
- Generated a documentation validation plot:
  - `docs/source/jax_mhd_finite_beta_target_jacobian.png`
  - It compares the exact assembled virtual-casing target derivative against a
    central finite difference on a reduced grid.
  - Maximum absolute derivative error in the plot data:
    `1.390e-10`.
- Upstreamed the remaining VMEC-JAX API gap on the existing
  `simsopt-qs-diagnostics` branch:
  - Added `FixedBoundaryExactOptimizer.state_tangent_columns_fun(...)`.
  - Added `FixedBoundaryExactOptimizer.b_cartesian_tangent_columns_fun(...)`.
  - Added upstream regression coverage in `tests/test_boundary_field.py`.
  - Updated Simsopt `B_cartesian_jax_tangent_columns(...)` to prefer the new
    public VMEC-JAX method and keep the private replay only as compatibility
    fallback for older local installs.
  - Commit pushed to the open VMEC-JAX PR:
    `98f9585 Expose exact boundary field tangent columns`.
  - PR #8 comment posted:
    https://github.com/uwplasma/vmec_jax/pull/8#issuecomment-4327126375
- Focused checks run:
  - `python -m pytest tests/mhd/test_jax_examples.py -q`
    (`2 passed`, six example subtests, one upstream JAX warning)
  - `python -m pytest tests/mhd/test_vmec_jax.py::VmecJaxInitializedFromInput::test_B_cartesian_tangent_columns_match_exact_jacobian tests/mhd/test_virtual_casing_jax.py -q`
    (`7 passed`, one upstream JAX warning)
  - `cd /Users/rogerio/local/vmec_jax_simsopt && ../simsopt_jax/.venv/bin/python -m pytest tests/test_boundary_field.py -q`
    (`3 passed`, one upstream JAX warning)
  - `cd /Users/rogerio/local/vmec_jax_simsopt && ../simsopt_jax/.venv/bin/python -m pytest tests/test_boundary_field.py tests/test_booz_input.py tests/test_step4_field_cartesian.py tests/test_vmec_bcovar_smoke.py -q`
    (`6 passed`, `1 skipped`, one upstream JAX warning)
  - `python -m pytest tests/mhd/test_jax_examples.py tests/mhd/test_vmec_jax.py tests/mhd/test_boozer_jax.py tests/mhd/test_vmec_diagnostics_jax.py tests/mhd/test_virtual_casing_jax.py -q`
    (`25 passed`, six example subtests, one upstream JAX warning)
  - `python -m sphinx -b html docs/source docs/build/html`
    (build succeeded with 28 pre-existing C++ API documentation warnings)
- Immediate next steps:
  - Build an end-to-end reduced finite-beta objective-gradient smoke test that
    exercises the exact target-gradient path without running the full advanced
    optimization script.
  - Re-run the full focused Simsopt JAX MHD suite after committing the upstream
    public VMEC-JAX tangent API and pushing both branches.
