# v54 — Independent exact general-sum Nash reference

This campaign executes Gambit's *rational*
`pygambit.nash.enummixed_solve(game, rational=True)` against
GameTheory's exhaustive `general_exact_search.search_exact_nash_cells`,
not its numerical or heuristic portfolio. The independent code and Gambit
have no shared Nash-implementation internals.

**Input contract:** bounded small two-player *integer general-sum* normal-form
games, including pure, unique mixed, rectangular, ties, and a continuous
equilibrium family. Both engines receive the same matrices. Every Gambit
`NFG 1 R` import is checked coefficient by coefficient before solving.
The exact search runs all `(2^m - 1)(2^n - 1)` support pairs with an
explicit limit of 2,000,000 rational operations, and refuses incomplete
or truncated results.

**Independent exact predicates:**
- Each GameTheory witness and each Gambit extreme equilibrium must
  satisfy exact rational simplex constraints and both players'
  *original-payoff* unilateral best-response equalities.
- Every Gambit point must lie in **exactly one** GameTheory strict-support
  product H-polytope. The membership check evaluates rational equalities,
  inequalities, positivity, and probability normalization without
  invoking the GameTheory search or LP implementation.
- Direct independent pure-strategy Nash scanning must agree with the
  GameTheory pure-cell set. Gambit must enumerate these pure points too.
- Exact profile counts and GameTheory family counts are intentionally
  **not equated**, because degenerate games have continuous equilibrium
  sets with infinitely many profiles. The Gambit result is a set of
  **extreme equilibria**, while GameTheory's cells are relatively open.
  Containment of sampled/extreme points is not a formal converse
  or independent completeness proof for every cell.
- A failed reference solver, incomplete exact search, mismatched
  payoff, invalid original-payoff certificate or missing family
  fails CI; no fallback or silent skipped fixtures.

## CI/reproduction

The `external-gambit` job retains v52 and adds its own new v54 test
and `benchmark_gambit_general_v54.py` run with pinned PyGambit
17.0.0b2 (prerelease). 3 repeated measurements are recorded per case,
including all timing vectors, support enumeration statistics,
payoff/strategy validation and fixture SHA-256. Core tests for the
independent validator stay in normal Python 3.13 strict CI, without
adding Gambit to the application's dependency set.

From `backend/` with the pinned packages installed:

    python -m pytest -q -W error test_gambit_general_contract_v54.py benchmark_gambit_general_external_v54.py
    python benchmark_gambit_general_v54.py --repetitions 3

## Evidence/limitations

This is independent exact **reference containment** on tested games,
not a proof that GameTheory discovers all Nash families on arbitrary
inputs, nor that Gambit returns a unique equilibrium for each family.
The cost of Gambit's extreme enumeration and GameTheory's support-cell
construction differs, so median times are descriptive without broad
SOTA performance inferences.

Remaining tasks include formal complete-set equivalence on adversarial
family geometries, more statistical repetitions, standalone
large-scale comparison under equal time budgets, robustness of rational
bit growth, and release readiness/production deployment audits.

Prior art:
https://gambitproject.readthedocs.io/en/latest/api/pygambit.nash.enummixed_solve.html
https://gambitproject.readthedocs.io/en/latest/api/pygambit.nash.EnumMixedResult.html
