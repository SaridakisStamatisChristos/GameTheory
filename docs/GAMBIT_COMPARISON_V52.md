# v52 — Independent Gambit Exact-Rational Comparison

**Goal:** Compare finite two-player **zero-sum matrix games** with a
separately maintained exact-rational baseline before any broad SOTA claim.

## Verification contract

The comparator uses Gambit **17.0.0b2** (PyGambit
`pygambit.nash.lp_solve(game, rational=True, use_strategic=True)`),
not a second algorithm written using GameTheory's own rational simplex.

All fixtures are exactly representable **integer** payoff matrices,
so GameTheory's float64 -> Fraction(str(value)) transformation and Gambit's
rational payoffs describe **the same game**. The comparator explicitly checks
Gambit's standard `.nfg` payoff-format import round trip before comparing outcomes.

For **both** engine outputs, a separately authored verifier calculates from
the source integer payoffs:

    min_j Σ_i x_i A_ij = v = max_i Σ_j A_ij y_j = x.T @ A @ y

All normalization and nonnegativity checks are rational. No tolerance or
comparison of strategy vectors is necessary. Different equilibria can be
equally correct in degenerate games. The **exact game value** must match,
and **both independent saddle-point certificates** must pass.

Fixture generator seed: 52052, with canonical, rectangular, tie-heavy,
single pure-saddle and seeded random 3x3, 5x5, 8x8 games. Each case carries
a fixture-set SHA-256 digest, all repetitions' measured seconds and medians.
The pinned Python version, OS, NumPy, PyGambit version, solver arguments,
and wallclock scope are reported in the artifact.

## Reproduction and status

The separate `external-gambit` GitHub Actions job installs the explicitly pinned **prerelease**
`pygambit==17.0.0b2`, runs unit/real-engine tests, performs 3 repeated
measurements on each case, and uploads
`backend/benchmark_gambit_v52_results.json`. A missing engine, a changed
input, a failed certificate, any disagreed value, or a failed run is an
explicit CI **FAIL**—not 'unavailable but passed'. Core/test coverage
jobs retain their previous constrained environment.

Locally, with the pinned packages available:

    python -m pip install pygambit==17.0.0b2
    cd backend
    python -m pytest -q -W error test_benchmark_gambit_v52.py
    python benchmark_gambit_v52.py --repetitions 3

## Scope limits and next gates

- Timing comparisons are **descriptive only**, not statistically robust
  speed rankings. Construction, solver execution, extraction and certificate
  checking are included, but their internal workloads are different.
- CI does not yet independently profile peak memory, rational-bit growth,
  or asymptotic scaling. Worst-case rational-simplex work may be large.
- OpenSpiel and Gambit's **general-sum** engines need their own model-matched
  comparisons. This validation does not establish optimality of the whole
  GameTheory portfolio.
- Artifact and CI URLs must refer to the **exact branch/commit** under review.
- A successful externally validated comparator is a stronger evidence state,
  but **not a global SOTA demonstration**.

Official sources:
https://gambitproject.readthedocs.io/en/v17.0.0b2/api/pygambit.nash.lp_solve.html
https://gambitproject.readthedocs.io/en/v17.0.0b2/api/pygambit.gambit.Game.from_arrays.html

## Gambit 17 compatibility adaptation

The installable 17.0.0 beta 2 wheel uses player/strategy labels (rather
than the removed public Player/Strategy objects), and the release's
`Game.from_arrays` entry point is unavailable. The benchmark therefore
writes the stable **NFG 1 R payoff format** in memory, with row player
actions changing fastest in each payoff pair. It imports that format via
`pygambit.read_nfg(BytesIO(...))` and recovers probabilities using
`profile[player_label][strategy_label]` and
`game.get_strategies(player_label)`.
`Game.to_arrays()` verifies that no transposition or payoff conversion
occurred before the solver is run. A parser failure or unexpected
strategy ordering is a CI failure, never a fallback to our solver.
