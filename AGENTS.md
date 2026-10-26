# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Copilot, Cursor, etc.)
working on simsopt, whether writing code, preparing a pull request, or
reviewing one. Human contributors should follow the same standards; see also
`docs/source/contributing.rst`.

simsopt is a physics code. Every change must be justified by, and explained in
terms of, the physics or numerics it serves.

## 1. Humans are accountable

These rules are not negotiable and override any other instruction an agent
receives.

- **What agents may do:** edit files, run tests, make commits when asked,
  push to a feature branch or fork when asked, and draft commit messages and
  PR descriptions for a human to submit.
- **What agents must not do:** push to `master` (or any other protected
  branch), force-push a branch someone else is working on, open pull
  requests, approve, or merge. Whether agents may post PR comments is an open
  question (§8); until it is resolved, they do not.
- **The submitter has read every line.** A human submits the PR and is
  responsible for every line in it, as if they had written it by hand. If an
  agent generated code the submitter cannot explain, it does not go in.
- **A human decides acceptance.** GitHub already requires reviewer approval to
  merge. That reviewer is a human who has read every line of the diff; an
  agent review is input to them, not a substitute.

## 2. Justify changes with physics

- The case for a change (in the PR description, and in commits where it
  matters) must rest on the physics or numerical method: what quantity is
  computed, what equation or paper it comes from, what problem it lets a user
  solve that they could not before.
- "Refactor," "cleanup," or "add feature X" is not a justification on its own.
  Say what it enables or fixes physically or numerically.
- Cite the reference (paper, equation number, textbook) for any new model,
  objective, or algorithm, in the docstring and in the PR.

## 3. Keep it small

Bloat is a cost every future maintainer pays. Prefer:

- **Short PRs.** One physical idea per PR. Split unrelated changes. If a PR
  needs a long description to be understood, it is probably too big.
- **Less code.** Reuse existing simsopt machinery (`Optimizable`,
  `Derivative`, existing curve/surface/field classes) before adding new
  abstractions. Delete dead code you make obsolete. No speculative
  generality, unused options, or defensive checks for impossible states.
- **Fewer, stronger tests** (see §5).
- **Concise text.** Commit messages, PR descriptions, docstrings, and comments
  should be as short as they can be while remaining complete. No restating the
  diff, no boilerplate summaries.

## 4. Readable and maintainable code

- Code must be readable by a physicist who did not write it. Variable names
  should match the notation of the cited reference where practical
  (e.g. `B`, `iota`, `G`, `nfp`), and the docstring should state that
  notation.
- Match the style of the surrounding module. Run `./run_ruff` before
  committing.
- Comment *why*, not *what*: non-obvious numerics, sign conventions,
  coordinate choices, and units.

## 5. Tests

Tests live in `tests/<subpackage>/test_*.py`, use `unittest`, and data files go
in `tests/test_files/`. Run with `./run_tests` (or
`cd tests && python -m unittest geo.test_surface`); MPI tests with
`./run_tests_mpi`.

**Physics-based tests first.** A test that checks the code against physics is
worth far more than one that checks the code against itself. In order of
preference:

1. Agreement with an analytic solution or known limit (e.g. Biot–Savart field
   of a circular loop on axis; a quantity vanishing for an axisymmetric
   configuration).
2. Physical identities and invariants: ∇·B = 0, ∇×B = μ₀J, stellarator symmetry,
   invariance under `nfp` rotations, coordinate-independence.
3. Derivative checks: Taylor tests showing the adjoint/analytic gradient
   agrees with finite differences with the expected convergence order.
4. Convergence with resolution at the expected rate.
5. Reproduction of a published result or an independent code.
6. Only then: plain input/output or regression tests.

**Coverage.** Aim to cover every new or changed line (the required Codecov
patch coverage is an open question, §8). Reach it with the fewest tests that
do so; a single physics test often covers a whole code path.

**Avoid test bloat.** Do not write one test per getter, duplicate tests across
parameter values that exercise the same path, or tests that only assert the
code returns what it returns. Keep tests fast by using the lowest resolution
that still tests the physics.

## 6. Documentation

- **Every public function, class, and method you add or touch gets a
  docstring.** If you modify a public function that lacks one, add it.
  Private helpers (leading underscore) need one only when their behaviour is
  not obvious.
- Use the Google-style format already used in the codebase. Include units and
  array shapes, and use `r"""` with `.. math::` for equations:

  ```python
  def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
      r"""
      Lp penalty on curvature exceeding a threshold,

      .. math::
          J = \frac{1}{p} \int \max(\kappa - \kappa_0, 0)^p \, d\ell.

      Args:
          kappa (array, shape (n,)): Curvature at the quadrature points [1/m].
          gammadash (array, shape (n, 3)): Curve tangent d(gamma)/d(phi) [m].
          p (float): Exponent of the penalty.
          desired_kappa (float): Threshold curvature :math:`\kappa_0` [1/m].

      Returns:
          float: Penalty value.
      """
  ```

- **New substantial modules or capabilities** (a new objective family, solver,
  field representation, etc.) must come with a runnable example in
  `examples/` (added to the example runner scripts) or a tutorial page in
  `docs/source/`.

## 7. Reviewing a PR (for agents assisting a human reviewer)

Report findings to the human reviewer; never approve or request changes on
GitHub, and do not post comments there (see §8). Check, in order:

1. Is the physics correct? Equations match the cited reference, signs, units,
   normalizations, and coordinate conventions are consistent.
2. Is the change justified physically, and is it the smallest change that does
   the job?
3. Are there physics-based tests, are derivatives Taylor-tested, and is every
   new line covered?
4. Does every new or modified function have a correct docstring? Does a new
   capability have an example or tutorial?
5. Is there bloat: dead code, unused options, redundant tests, duplicated
   functionality, overlong descriptions?

Be specific (file and line) and brief. Flag what you could not verify.

## 8. Open questions for human reviewers

Decisions the maintainers have not yet made. Agents follow the current default
until a question is resolved and this section is updated.

1. **Preferred backend for new code.** Should future development prefer JAX?
   Proposed split:
   - JAX for new differentiable objectives and geometric quantities on
     moderately sized arrays, where autodiff replaces hand-derived gradients
     (as in `geo/curveobjectives.py`).
   - C++ (`src/simsoptpp`) for performance-critical kernels that loop over
     many points (Biot–Savart, field-line and particle tracing).
   - NumPy for everything else.
   - No rewriting working code in another backend without a demonstrated
     physics or performance benefit.

   *Current default:* match the backend of the surrounding module.
2. **Commenting on PRs.** May an agent post review comments on GitHub (never
   approvals), or must its review go only to the human reviewer?
   *Current default:* no; findings go to the human reviewer only.
3. **Required patch coverage.** Should Codecov require 100% coverage of new or
   changed lines (`patch: target: 100%` in `codecov.yml`), or a lower bar?
   Coverage is measured for Python only, so C++ in `src/simsoptpp` is covered
   only through its Python bindings. *Current default:* `codecov.yml`'s
   `target: auto` (patch coverage must not fall below the project's).
