# General Nash Engine v47 — Algorithm and evidence contract

## Objective

Expose one reusable, size-independent bimatrix-game computational core
without expanding matrix limits version by version. The FastAPI input
validator still enforces **20×20** as a local-service safeguard. The
Python core is not hardcoded to 4×4, 5×5, or 20×20; computational
budgets and practical memory limits apply.

`POST /api/equilibria/general` accepts all existing Game matrix fields
and optional: `mode` (`auto`, `approximate`, `exact`),
`iterations`, `restarts`, `seed`, `max_pure_cells`,
`max_exact_pairs`, `max_exact_operations`, `max_exact_support`,
  `exact_backend` (`simplex` default or `vertices` reference),
  and `prune_supports` (`true` default, `false` exact baseline).

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
* **Exact:** The solver performs exact-sign, support-conditional dominance
  pruning before launching an LP. It retains the exact mathematical
  completeness condition and allows `prune_supports=false` for an
  unpruned baseline. The default feasibility backend is an exact rational two-phase
  simplex LP that maximizes the minimum supported probability. Strict
  support exists precisely when the exact optimum is positive. A bounded
  vertex-enumeration oracle remains selectable via `exact_backend=vertices`
  for independent comparisons. Both backends verify feasible strategies
  against rational best-response inequalities before producing cells.
  For supports `(I,J)`, use exact rational vertex-basis
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
* Default exact feasibility now uses rational two-phase simplex (Bland-style
  pivot tie-breaking) rather than enumerating vertices. Exact simplex can
  still take exponential time in the worst case. The operation counters
  bound tableau pivot-update work, **not** coefficient bit complexity,
  memory, or wallclock time. The vertex backend remains available for
  differential testing, not as a guaranteed scalable fallback.
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

## v47: provably safe conditional support pruning

For a candidate support pair `(I,J)`, every equilibrium strategy
`y` with support `J` has **strictly positive** probabilities
`y_j>0` for every `j in J`. If some supported row action `i in I`
has an alternative pure action `k` with
`A[k,j] >= A[i,j]` for every `j in J` and strict inequality for at
least one `j in J`, then

```text
sum_{j in J} (A[k,j] - A[i,j]) * y[j] > 0
```

for *every* admissible `y`. Thus `i` cannot be a best response,
so the entire strict-support cell is empty. Apply the symmetric
comparison for supported column `j in J` against an alternative
column `k` on every `i in I`.

This is a necessary-condition filter, not an approximate equilibrium
test. If no exclusion witness exists, the exact LP still decides
feasibility. A strict advantage **only outside** the opponent support
does not justify pruning, nor does weak equality on the entire support.

The implementation precomputes comparison-sign bitmasks and caches
support-conditioned dominated-action maps. Finite validated float64
value comparisons preserve the ordering of their exact decimal-string
rational interpretations. A successful classification by this rule
counts as a checked infeasible support, with up to 12 illustrative
exclusion certificates returned as `pruning_witnesses_sample`.

Completeness bookkeeping: `support_pairs_checked` counts both
LP-classified supports and dominance-pruned supports.
`dominance_pruned_pairs` counts the latter, and `lp_pairs_attempted`
counts the former's attempted exact LP calls (including a call that
exhausts the operation budget). The existing support-pair ceiling
still applies to every classified support. Only classifying every
support pair establishes `complete_equilibrium_set=true`.

Precomputing mask comparisons costs approximately
`O(m*m*n + n*n*m)` payoff comparisons, and caching may use memory
proportional to distinct encountered opponent support masks.
This preprocessing is **not** part of the
`max_exact_operations` rational-simplex budget. The rule may add
overhead when little can be pruned; it does not improve worst-case
complexity or claim polynomial-time complete Nash enumeration.

The paired, reproducible `backend/benchmark_support_pruning_v47.py`
runs identical support prefixes with `prune_supports=true/false`
on globally dominated, cyclic, and seeded random games of varying
sizes, comparing feasible support sets and report semantics before
reporting measured times and exact work counts. Independent regression
tests compare complete/partial classifications against the unpruned
rational LP and direct exact Nash deviation verification.
