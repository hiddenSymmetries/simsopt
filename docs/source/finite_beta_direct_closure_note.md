# Finite-Beta Direct Closure Technical Note

## Scope

This note records the latest direct finite-beta Boozer-surface closure update in `boozerQA_finitebeta.py`, the physics interpretation that survived validation, and the reduced comparison benchmarks against the virtual-casing and VMEC branches.

## Main conclusion

The direct no-VC branch now uses a more faithful exterior closure than the previous offset-average approximation: the exterior sheet contribution is evaluated with a direct principal-value boundary integral on the interface, while the half-jump is enforced exactly. This is a better representation of the boundary-integral limit, but it does **not** materially improve agreement with the VC reference in the reduced 1% beta QA benchmark.

The remaining mismatch is therefore not explained by the old offset-average exterior closure alone.

## Physics decisions validated in this round

### 1. `I` should not be inserted as a secular sheet-current potential term

An earlier hypothesis was that the direct closure was missing a multivalued current-potential component associated with `I`, so a secular contribution was temporarily added to the surface-current model. That hypothesis was tested and then rejected.

The decisive diagnostic was a direct integration of the VC interface current saved in `examples/2_Intermediate/output/boozerQA_finitebeta_boundary.vts`. The measured VC interface current had essentially zero net toroidal-current content, which is incompatible with the idea that the VC current should contain a secular contribution proportional to `I / \mu_0`.

As a result, the finite-beta sheet current was restored to the single-valued form

$$
K = n \times \nabla \Phi,
$$

and the direct branch now treats `I` only through the interior Boozer field.

### 2. `I` belongs in the interior Boozer field

The interior field used by the direct provider is now

$$
B_{\mathrm{in}} = B_{\mathrm{Boozer}}(x; \iota, G, I),
$$

implemented through `finite_beta_boozer_surface_field(surface, iota, G, I)`.

This keeps the meaning of `I` aligned with the Boozer representation rather than overloading the surface-current potential.

### 3. Exterior closure now uses a principal-value boundary integral

The direct provider no longer estimates the exterior sheet field from inside/outside offset surfaces and then averages the result. Instead, it computes the on-surface principal-value Biot-Savart integral for the sheet current and combines it with the exact half-jump:

$$
B_{\mathrm{out}} \approx B_{\mathrm{coil}} + B_{\mathrm{sheet}}^{\mathrm{PV}} + \frac{\mu_0}{2} K \times n.
$$

This is the simplest faithful boundary-integral closure available in the existing code path without introducing the full virtual-casing operator into the direct branch.

## Reduced benchmark results

All runs below used the reduced grid

- `nphi=6`
- `ntheta=6`

### Direct self-consistent branch (`SIMSOPT_FINITE_BETA_MODE=self-consistent`)

- pressure target: `plasma_beta=1%`
- continuation steps: `6`
- optimizer cap: `ls_max_nfev=120`

Final state:

- `iota = -4.064341e-01`
- `G = 1.388384e+01`
- `I = 1.535492e-01`
- `||r|| = 4.960470e-02`
- pressure residual norm: `1.730360e-02`
- jump residual norm: `4.647830e-02`

### VC reference branch (`SIMSOPT_FINITE_BETA_MODE=single-surface-vc`)

- pressure target: `plasma_beta=1%`
- optimizer cap: `ls_max_nfev=120`

Final state:

- `iota = -3.951105e-01`
- `G = 1.388384e+01`
- `I = 5.012459e-03`
- weighted `||r|| = 7.705521e-02`
- raw `||r|| = 1.656266e-01`

### Direct vs VC comparison

- `|\Delta iota| = 1.13236e-02`
- direct `I` is about `3.06e+1` times the VC value

Interpretation:

- The direct branch still reproduces the vacuum-like geometry and non-QS level reasonably well.
- The dominant disagreement remains in the finite-beta current response, not in the geometry optimization.
- Replacing the offset-average exterior closure with the principal-value boundary integral does not collapse the gap to the VC reference.

## VMEC benchmark note

The reduced VMEC-backed branch still runs successfully, but the current example is not an apples-to-apples validator for the 1% beta QA benchmark above.

For the supplied VMEC equilibrium, the interface-field inference produced

- `pressure_jump = 3.008885e+06`
- `plasma_beta = 100.007789%`

and the reduced solve converged to

- `iota = 2.084661e-01`
- `G = 0`
- `I = -7.223725e-01`
- `||r|| = 1.895913e+01`

This is better interpreted as a stress test of the branch wiring than as a direct quantitative validation target for the 1% QA case.

## What this means technically

The direct branch is now on a cleaner physics footing:

- `I` is handled in the interior Boozer field.
- The surface current remains single-valued.
- The exterior closure is now a true on-surface principal-value boundary integral plus the exact jump term.

However, the reduced comparison shows that the main no-VC versus VC discrepancy survives this upgrade. The remaining error is likely tied to missing external-response physics beyond the local principal-value closure, not just to the old offset-evaluation approximation.

## Immediate next directions

The most likely next steps are:

1. Compare the direct principal-value operator against the full virtual-casing operator on the same frozen surface/current state to localize the remaining closure error.
2. Separate geometry agreement from current-closure agreement by replaying the direct and VC residual blocks on identical saved states.
3. If the goal is quantitative standalone agreement with VC, introduce a more global exterior-response model rather than further tuning the local sheet-field approximation.