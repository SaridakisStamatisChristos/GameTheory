# v59 — Multiplayer mixed-Nash search with rational incentive certificates

## Scope

This is the first genuinely **mixed-strategy Nash** engine for
finite, three-to-five-player general-sum normal-form games in GameTheory.
The output is a bounded collection of independently certified candidate
profiles, **not a complete enumeration of mixed Nash equilibria**.

Endpoint: POST /api/multiplayer/mixed

The payoff input is the same as v58: one matching N-dimensional payoff
tensor per player, with 3–5 players, 1–4 actions each, and at most 36
joint action profiles. Inputs are validated and interpreted as decimal
rationals of float64 payoffs by the existing multiplayer _game model.

Example (three players; third has one action, first two play matching
pennies):

~~~json
{
  "payoffs": [
    [[[1],[-1]],[[-1],[1]]],
    [[[-1],[1]],[[1],[-1]]],
    [[[0],[0]],[[0],[0]]]
  ],
  "tolerance": 1e-8,
  "seed": 59059,
  "max_support_patterns": 96,
  "starts_per_pattern": 4,
  "max_evaluations": 10000,
  "max_candidates": 16
}
~~~

A known exact mixed Nash equilibrium is (1/2, 1/2) for the first
two players and (1) for the third; there are no pure Nash equilibria.

## Actual algorithm

For each tuple of nonempty player supports, search for probabilities
with strictly positive support and zero elsewhere. The root equations
require every supported action of each player to yield the same
expected payoff against the other players' independent mixed strategies:

    U_p(a; x_-p) - U_p(base; x_-p) = 0, for a in S_p minus {base}.

These are **multilinear polynomial equations** for N>2, so the
two-player LP formulation cannot be assumed to solve them.
The solver runs SciPy/Hybr nonlinear roots from a uniform start and
seeded Dirichlet restarts, in increasing total-support-cardinality
order. It enforces independent limits on patterns, starts, root
function evaluations and accepted candidates, and explicitly exposes
which limit stopped the search. Root convergence or lack of a found
root is **not** a mathematical completeness certificate.

A candidate must have every supported action strictly positive
(the numerical probe rejects values <= 1e-8), with all excluded actions
assigned zero. Convert its supported decimal float probabilities to
Fraction exactly, then set the last supported probability to one
minus the exact sum of preceding supported probabilities. This
constructs a rational, *exactly normalized* product mixed strategy.

## Independent mathematical certificate

For a rational candidate strategy x and rational payoff tensors u,
the independent verifier recomputes from the original game:

    U_p(a; x_-p) = sum_{a_-p} u_p(a,a_-p)
                    product_{q != p} x_q(a_q)
    R_p(x) = max_{a} U_p(a;x_-p)
                  - sum_a x_p(a) U_p(a;x_-p)
    epsilon(x) = max_p R_p(x).

Every product, expectation, and unilateral pure-action deviation
is recomputed with exact rational arithmetic. The candidate is
returned only if epsilon(x) <= the requested tolerance interpreted
as a rational decimal string and all supported-action payoff
spreads are <= twice that tolerance.

Because maximizing expected utility against fixed opponents'
mixed strategies is a linear function of the deviating player's
own mixed strategy, checking every pure deviation also checks
all mixed deviations. This establishes a true **epsilon-Nash
certificate for the represented rational profile and game**.
If epsilon is exactly zero as a Fraction, the profile is
certified as an exact rational Nash equilibrium. Otherwise the
output is a mathematically verified epsilon-Nash equilibrium;
numerical root convergence alone is never treated as a proof.

The exact certificate validates only a returned candidate, not
the absence, uniqueness, count, or completeness of other mixed
equilibria. The entire pure Nash set remains exhaustively
enumerated via the previous v58 pure solver.

## Search budgets and statuses

- Players 3–5; actions 1–4 each; at most 36 joint profiles.
- Tolerance between 1e-12 and 0.1 in **raw payoff units**.
- Seed within [0, 2^31-1].
- max_support_patterns: 1–1024, default 96.
- starts_per_pattern: 1–12, default 4.
- max_evaluations: 1–100,000, default 10,000.
- max_candidates: 1–128, default 16.

The response contains pure_nash_complete=true but always
mixed_nash_complete=false, even if all support patterns were
attempted. Function-evaluation, candidate, and support-pattern
budgets return partial metadata, not a false negative theorem.
The exact rational certificate is computed only for feasible
probability vectors; it is not a global numerical root guarantee.

## Independent evidence and honest limitations

Tests independently reconstruct exact rational unilateral gains,
including a three-player matching-pennies extension, a three-player
parity interaction, four-player all-ties and five-player dominated
games. Adversarial injected off-simplex and nonfinite roots must
be rejected; perturbed roots retain nonzero exact epsilon and
cannot be falsely promoted to exact Nash.

The v59 10-case reproducible campaign records deterministic seeds
and fixture SHA-256, raw timing distributions, support attempts,
root function calls, incomplete reasons, and independent rational
Nash-deviation checks on every returned candidate.
It is archived as multiplayer-mixed-v59-evidence in GitHub Actions.

This is a research-grade multistart support-root method, not
an independently benchmarked state-of-the-art mixed Nash solver.
No guaranteed solution to all multiplayer general-sum games is claimed.
The external Gambit/OpenSpiel checks from prior versions remain active,
but they are **not** independent multiplayer mixed-equilibrium
reference engines in this release.

Reproduce:

~~~bash
cd backend
python -m pytest -q -W error test_multiplayer_mixed_v59.py test_multiplayer_mixed_http_v59.py test_multiplayer_mixed_benchmark_v59.py
python benchmark_multiplayer_mixed_v59.py --repetitions 3 --seed 59059
~~~
