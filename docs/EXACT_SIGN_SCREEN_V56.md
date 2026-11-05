# v56 — Exact sign-infeasibility screen and rational resource evidence

## Algorithm and proof obligation

Given supported positive strategy probabilities `p_j > 0`, fix a required
best-response action `base`. For each action `i`, let
`d_j = payoff(base,j) - payoff(i,j)` on the opponent support.

- A **required best-response tie** demands `Σ p_j d_j = 0`.
  When all `d_j >= 0` and some `d_j > 0`, the sum is strictly
  positive; when all `d_j <= 0` and some `d_j < 0`, it is strictly
  negative. Either case is impossible for strictly positive `p_j`.
- A **nonrequired deviation** demands `Σ p_j d_j >= 0`.
  When all `d_j <= 0` and some `d_j < 0`, the sum must be negative,
  so the support is impossible.
- All other sign patterns are *inconclusive* and must be delegated to
  the existing exact rational LP. No approximate comparisons and no
  relaxation of Nash or strict-support conditions occur.

The sign-only screen is applied after the exact uniform-witness
check, and never replaces the LP when the sign pattern is inconclusive.
It is explicitly optional via
`search_exact_nash_cells(..., feasibility_sign_screen=False)`.
`feasibility_fast_path=False` remains the pure original simplex
baseline. The screen charges `(number_of_actions - 1) × support_size`
inspection operations **before** using the result. No computation is
free against the configured exact work budget; a newly exhausted
budget is explicitly classified as partial, never complete.

## Validation

- Complete exhaustive 2×2 payoff-grid comparison against the unchanged
  rational LP; supported-action subsets and opponent supports included.
- Seeded 2×2 through 4×4 bimatrix full-support-pair differential tests,
  including tied and degenerate games, comparing exact feasible cells.
- Independent negative tests for impossible ties, impossible deviations,
  nonuniform feasible supports, valid uniform witnesses, malformed
  flags, operation-budget exhaustion, and opt-in instrumentation.
- Existing unmodified Gambit, OpenSpiel, Nashpy, backend strict
  branch-coverage, and frontend checks remain required.

## Reproducible benchmark and resources

`backend/benchmark_sign_screen_v56.py` executes the two modes on
15 deterministic cases (all ties, dominance, seeded random, 2–8
actions) with exactly matched support search order, budgets, and
pruning disabled. Both raw wall-clock samples, exact LP calls, explicit
operation counts, support-set identity, completeness/partial status,
and SHA-256 of fixture payloads are recorded.

Opt-in `track_rational_bits=True` scans the actual rational simplex
tableau at initialization and after each pivot, recording maxima of
Fraction numerator and denominator bit lengths. A **separate untimed**
`tracemalloc` run captures tracked Python allocation peaks. This is
not OS process RSS; sample runs intentionally exclude these diagnostics
from wall-clock timings. Neither charged operations nor numerator
bit lengths are a proof of bit-complexity bounds.

Reproduce from `backend/`:

```bash
python -m pytest -q -W error test_sign_screen_v56.py test_sign_screen_benchmark_v56.py
python benchmark_sign_screen_v56.py --repetitions 3 --seed 56056 --output benchmark_sign_screen_v56_results.json
```

A CI artifact named `exact-sign-screen-v56-evidence` contains the
full JSON, and coverage must remain at least 99%.

## Limitations

The exact infeasibility proof is local and necessary, not complete;
inconclusive cases still require the simplex. The paired workload is
bounded and ordered; speed can regress when almost all supports require
the LP. Memory is Python-traced, not complete process or native-memory
usage. No claim of universal speed superiority, novelty, or global SOTA.
