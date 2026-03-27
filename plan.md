# BoozerSurface QI Implementation Plan

## Goal

Implement a new quasi-isodynamic Boozer-surface objective in this clone, preserving the coil/Biot-Savart workflow of `boozerQA.py` while porting the legacy single-surface well-shuffling QI residual onto the Boozer coordinates already enforced by `BoozerSurface`.

## Branch Order

1. `rogerio/boozer-qi-fixtures-and-math-spec`
2. `rogerio/boozer-qi-objective`
3. `rogerio/boozer-qi-example-and-validation`
4. `rogerio/boozer-qi-docs`

Each branch starts from fresh `master` after the previous PR merges.

## Milestone 1 Math Contract

For a single Boozer surface, define physical Boozer angles

$$
\phi_B = 2\pi\,\varphi, \qquad \theta_B = 2\pi\,\theta,
$$

and evaluate the residual over one field period $[\phi_{\min}, \phi_{\max}]$ with

$$
\phi_{\max} = \phi_{\min} + \frac{2\pi}{n_{fp}}.
$$

For field-line labels $\alpha_j = 2\pi j / n_{\alpha}$, sample

$$
\theta_B(\phi_B; \alpha_j) = \alpha_j + \iota (\phi_B - \phi_{\min}),
$$

with $\iota = \text{boozer\_surface.res['iota']}$, and define the field magnitude

$$
B_j(\phi_B) = |B(\phi_B, \theta_B(\phi_B; \alpha_j))|.
$$

Apply the legacy affine normalization globally on the surface:

$$
B_{\min} = \min_{j,\phi} B_j(\phi), \qquad
B_{\max} = \max_{j,\phi} B_j(\phi),
$$

$$
\widehat{B}_j(\phi) = \frac{B_j(\phi) - B_{\min}}{\max(B_{\max} - B_{\min}, \varepsilon_B)}.
$$

For each field line, split at the minimum, apply the legacy left/right squash rules, apply the cosine-power stretch functions with `pmax = 50` and `pmin = 15`, compute bounce branches using the legacy `GetBranches` edge handling, shift the branches to equalize weighted bounce distances, reconstruct the shuffled target $\widehat{B}_{QI,j}$, and define the residual samples

$$
r_{jm} = \frac{\widehat{B}_{QI,j}(\phi_m) - \widehat{B}_j(\phi_m)}{\sqrt{n_{\alpha} n_{\phi,\mathrm{out}}}}.
$$

The milestone-1 scalar objective is

$$
J_{QI} = \sum_{j,m} r_{jm}^2.
$$

No extra denominator is introduced in milestone 1. Exact-value parity of $J_{QI}$ with the legacy single-surface residual is the primary correctness target.

## New Code Layout

Primary implementation target:

- `src/simsopt/geo/surfaceobjectives.py`

New public class:

- `NonQuasiIsodynamicRatio(boozer_surface, bs, sDIM=20, nphi=151, nalpha=31, nBj=51, nphi_out=2000, phi_shift=None, smoothing=None)`

Recommended private helpers in `surfaceobjectives.py`:

- `_make_qi_aux_surface`
- `_make_qi_sampling_grid`
- `_sample_modB_on_boozer_lines`
- `_normalize_modB_global`
- `_squash_left_branch`
- `_squash_right_branch`
- `_stretch_left_branch`
- `_stretch_right_branch`
- `_get_branches`
- `_build_template_well`
- `_shuffle_well_coordinates`
- `_make_shuffled_target_values`
- `_qi_residual_vector_single_surface`
- `_qi_objective_single_surface`
- `_dJ_by_dB_qi`
- `_dJ_by_dsurfacecoefficients_qi`

The public class should mirror `NonQuasiSymmetricRatio` structurally: constructor, `recompute_bell`, `J`, `dJ`, and `compute`.

## PR Scope

### PR 1

Branch: `rogerio/boozer-qi-fixtures-and-math-spec`

Deliverables:

- frozen branch-crossing fixtures
- frozen transformed-well fixtures
- frozen shuffled-target fixtures
- frozen scalar references
- helper tests for squash, stretch, `GetBranches`, monotonicity repair, and affine-normalization invariance

### PR 2

Branch: `rogerio/boozer-qi-objective`

Deliverables:

- helper stack in `surfaceobjectives.py`
- `NonQuasiIsodynamicRatio`
- import/export wiring
- focused unit tests and Taylor tests

### PR 3

Branch: `rogerio/boozer-qi-example-and-validation`

Deliverables:

- `examples/2_Intermediate/boozerQI.py`
- reduced-runtime regression and physics validation
- optional VMEC geometry initializer

### PR 4

Branch: `rogerio/boozer-qi-docs`

Deliverables:

- `docs/source/example_quasiisodynamic.rst`
- docs index updates
- user-facing explanation of the objective and limitations

## Tests

Unit and regression coverage should include:

- `_get_branches` edge cases
- left/right squash behavior
- left/right stretch values
- monotonicity-repair logic
- affine-normalization invariance
- scalar residual assembly
- later, `NonQuasiIsodynamicRatio` Taylor tests beside `NonQSRatioTests`

## Documentation Scope

Add a dedicated Boozer QI tutorial page documenting:

- the single-surface scalar objective
- the well squash/stretch and branch-shift construction
- the difference from the older VMEC plus `booz_xform` workflow
- milestone-1 limitations