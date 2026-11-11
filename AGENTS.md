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

- **What agents may do:** edit files, run tests, and make commits, push, and
  open pull requests when asked.
- **What agents must not do:** force-push a branch someone else is working
  on, post PR comments, approve, or merge.
- **A human has read every line.** A human is responsible for every line in a
  PR, as if they had written it by hand, and reads every line before the PR
  goes to review. If an agent generated code that human cannot explain, it
  does not go in.
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
  Minimize the use of small single-use helper functions; they make the code
  harder to follow.
- **Fewer, stronger tests** (see §6).
- **Concise text.** Commit messages, PR descriptions, docstrings, and comments
  should be as short as they can be while remaining complete. No restating the
  diff, no boilerplate summaries.

## 4. Readable and maintainable code

- Code must be readable by a physicist who did not write it. Variable names
  should match the notation of the cited reference where practical
  (e.g. `B`, `iota`, `G`, `nfp`), and the docstring should state that
  notation.
- Match the style and backend (NumPy, JAX, or C++) of the surrounding
  module. Run `./run_ruff` before committing.
- Comment *why*, not *what*: non-obvious numerics, sign conventions,
  coordinate choices, and units.
- Comments describe the code as it is, not its history ("instead of X, we now
  do Y"). Explain what changed and why in the commit message or PR instead.
- Code that runs on both CPU and GPU (e.g. JAX) can behave very differently
  on each: a change that speeds up the CPU can slow down the GPU, so check
  both when you can.

## 5. Changing existing code

When editing existing code that does not follow these conventions, make the
new code follow them. *Do not* bloat the diff by reformatting or rewriting the
whole file. *Do* improve the parts you touch, for example by adding or
improving docstrings.

## 6. Tests

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
4. Convergence with resolution at the expected rate (e.g. spectral
   convergence of the trapezoidal rule on a periodic integrand).
5. Reproduction of a published result or an independent code.
6. Only then: plain input/output or regression tests.

**Coverage.** Cover new and changed code with the fewest tests that do so; a
single physics test often covers a whole code path.

**Avoid test bloat.** Do not write one test per getter, duplicate tests across
parameter values that exercise the same path, or tests that only assert the
code returns what it returns. Use parametrized tests (`self.subTest`) to test
variants.

**Keep run time reasonable.** A single test should not take over a minute,
except for a small number of first-principles physics tests. Relax
tolerances and lower the numerical resolution when that does not affect the
physics the test checks.

## 7. Documentation

- **Every public function, class, and method you add or touch gets a
  docstring.** Private helpers (leading underscore) need one only when their
  behaviour is not obvious.
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
  `examples/` (added to the example runner scripts). For large modules, also
  consider a tutorial page in `docs/source/`.

## 8. PR descriptions

Follow `.github/pull_request_template.md`. A PR description should say what
the code cannot: why the change is needed and how it affects users. Do not
list changed files or tests; the diff shows them.

- **Summary**: one to three sentences on what the PR does and the physics or
  numerical problem it solves. State any breaking change and who it affects.
- **Context**: background a reviewer needs, such as the physics, the bug and
  its effect on users, or references. Omit for small PRs.
- **Questions**: anything you are unsure of, @mentioning the people who would
  know.

An agent writing a PR description ticks "I used AI to generate this PR" and
leaves "I have read every diffed line" unticked for the human.

## 9. Reviewing a PR (for agents assisting a human reviewer)

Report findings to the human reviewer; never approve, request changes, or
post comments on GitHub. Check that the physics is correct (equations match
the cited reference; signs, units, normalizations, and coordinate conventions
are consistent) and that sections 2–7 are respected, particularly
conciseness, maintainability, tests, and documentation. Be specific (file and
line) and brief. Flag what you could not verify.
