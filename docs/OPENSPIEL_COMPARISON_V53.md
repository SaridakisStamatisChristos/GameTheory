# v53 — Independent OpenSpiel zero-sum matrix-game LP campaign

## Acceptance contract

This release adds a **separate external comparison**, not another solver
inside GameTheory. It executes OpenSpiel 2.0.2's
`open_spiel.python.algorithms.lp_solver.solve_zero_sum_matrix_game`
on the exact same 16 integer game matrices used by the Gambit v52
campaign, plus three seeded 4×7, 7×4 and 10×10 fixtures.

**Game model:** two-player, simultaneous-move, zero-sum matrix games.
For an integer matrix A, OpenSpiel receives the strategically
equivalent matrix A - min(A) (a constant translation), while
GameTheory's exact rational solver receives A unchanged. This preserves
all equilibria, and values are mapped back by adding the translation.
The OpenSpiel `pyspiel.MatrixGame` is explicitly round-trip checked,
entry by entry, before invoking the third-party LP.

**Independent original-game certificate:** Each returned numerical
strategy must be finite, approximately simplex-normalized, and
nonnegative (1e-7); the maximum unilateral gap and absolute error
from the independently exact rational game value are each at most 2e-5
original payoff units. OpenSpiel-reported player values are checked,
with the opposite sign for the column player.

**No exact OpenSpiel witness is claimed.** The exact comparison is
possible for Gambit v52, not numerical OpenSpiel v53. A narrow
numerical tolerance is an empirical acceptance policy, not a proof
that the numerical algorithm always converges on larger inputs.

## Reproduction

The dedicated `external-openspiel` GitHub Actions job uses Python
3.12 rather than 3.13 because the historical ECOS 2.0.14 wheel has no
CPython 3.13 wheel. It installs exact versions `open-spiel==2.0.2`,
`cvxpy==1.9.3`, and `ecos==2.0.14`. The main project remains on
Python 3.13. A missing dependency or failed numerical certificate
is a CI failure, not a skipped result.

From `backend/` in the compatible environment:

    python -m pytest -q test_open_spiel_contract_v53.py benchmark_open_spiel_external_validation_v53.py
    python benchmark_openspiel_v53.py --repetitions 5

A valid benchmark writes JSON only after **all fixtures and all
replicates** pass; the uploaded artifact includes the SHA-256 of
fixtures, package versions, all timing samples and medians, diagnostics,
and exact game values.

## Limitations

- Speed comparisons are **descriptive**, because one engine uses a
  translation, exact and numerical solvers have fundamentally
  different precision semantics, and matrix construction/extraction
  overheads are not identical. No speedup claim is made.
- Five repeats per fixture are not a statistically powered study.
  Neither paired execution nor error bars establish broad SOTA.
- Non-zero-sum / general-sum and extensive-form games still need
  matching independent comparators. Memory and coefficient bit
  growth, throughput at larger matrix sizes, and adversarial
  ill-conditioned games remain open.
- OpenSpiel's Python LP implementation uses CVXPY and ECOS.
  This is a real independent engine, not a SciPy wrapper written
  within GameTheory.

Primary implementation reference:
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/lp_solver.py
