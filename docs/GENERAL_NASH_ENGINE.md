# General Nash Engine v45 — Algorithm and evidence contract

## Objective

Expose one reusable, size-independent bimatrix-game computational core
without expanding matrix limits version by version. The FastAPI input
validator still enforces **20×20** as a local-service safeguard. The
Python core is not hardcoded to 4×4, 5×5, or 20×20; computational
budgets and practical memory limits apply.

`POST /api/equilibria/general` accepts all existing Game matrix fields
and optional: `mode` (`auto`, `approximate`, `exact`),
`iterations`, `restarts`, `seed`, `max_pure_cells`,
`max_exact_pairs`, `max_exact_operations`, and `max_exact_support`.

## Algorithm dispatch

* **Iterated strict pure-strategy dominance.** An action is deleted only
  if another *retained pure* action has strictly better payoff against
  **every** currently retained opponent action. On any Nash equilibrium,
  a strictly dominated action gets probability zero. Iterating such
  removals preserves all original Nash equilibria, and equilibria of
  the reduced game embed back into the original game with zeros on
  removed actions. The return value retains action maps and a
  witness for each deletion.
* **Auto:** Scan pure best responses (if within `max_pure_cells`).
  Otherwise, detect constant-sum games and solve row-player primal
  and column-player dual minimax LPs in well-conditioned coordinates.
  On small reduced games try the independent Lemke–Howson pivot
  solver. If none yields a usable candidate, run seeded fictitious
  play. Every returned numerical profile is checked against the
  original, unreduced payoff matrices.
* **Approximate:** Run deterministic-seed multistart fictitious play
  and report the resulting unilateral-deviation certificate. Its
  success does *not* establish global convergence or uniqueness.
* **Exact:** For supports `(I,J)`, use exact rational vertex-basis
  feasibility in increasing support-cardinality order to test whether
  each player has a *strictly positive* mixture making the opponent's
  supported actions best responses. The output contains an exact
  rational Nash witness and the product H-polytope constraints.
  Exhaustion of every nonempty support pair proves complete
  equilibrium-set coverage. A bounded subset is **partial**.

## Completeness and certificate semantics

There are `(2**m - 1)*(2**n - 1)` nonempty support pairs for a
`m×n` reduced game. Exact support enumeration is generally
exponential. A `complete_equilibrium_set` claim is permitted only if
the exact search actually visits every such pair; each feasible pair
returns the exact strict-support cell defined by best-response
equalities, weak inequalities, simplex normalization and strict
positivity. The union equals the full Nash set by the proof in
`EXACT_NASH_GEOMETRY.md`.

All exact rational strings refer to `Fraction(str(float(payoff)))`
after validating finite float64 inputs. The support representation is
in **reduced** action coordinates; `retained_actions` maps strategies
back to the original game. A returned floating projection of a
rational witness is informative only; near subnormal probabilities
can underflow, so rational witnesses remain authoritative.

`certificate.valid` from the numerical verifier means the
provided strategy satisfies declared probability and relative/absolute
unilateral-gain tolerances, **not** an exact mathematical proof.
Pure best responses are exact comparisons in the validated input
model. Solver outcomes distinguish `certified_candidate`,
`uncertified_candidate`, `partial_exact_search`,
`complete_exact_set`, and `inconclusive`.
No solver failure or exhausted budget is silently marked complete.

## Computational limitations and research priorities

* Default exact scan: 96 support pairs, 20,000 counted arithmetic
  basis operations, support sizes up to 3. API hard maxima: 4,096
  pairs, 2,000,000 operations and support size 20.
* The exact implementation currently enumerates rational polytope
  vertices. Its worst-case cost is combinatorial. These counters do
  not prove a hard bound on integer bit complexity or wallclock time.
* The general auto-mode uses existing numerical solvers; it does not
  promise a polynomial-time exact Nash solution for general games.
* The complete-set certificate only applies to the whole reduced
  game when **all** supports were processed. A report with a truncated
  support search is provisional even if it returns valid equilibria.
* Future optimization should focus on an exact rational simplex or
  complementary-pivot method with independently checked feasibility,
  stronger structure-aware pruning, and empirical large-game
  comparisons—**not** adding per-dimension branches.

## Reproduction

Run the existing GitHub Actions backend tests with strict warning
handling and full statement/branch coverage; use
`backend/test_general_nash_v45.py` and
`backend/test_general_nash_api_v45.py` for direct differential
checks, regressions on degenerate/rectangular inputs, and
budget-exhaustion semantics. Do not interpret high coverage as a proof
of every numerical corner case.
