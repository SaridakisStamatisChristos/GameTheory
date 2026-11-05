# v58 — Exact pure Nash and correlated equilibrium for 3–5 players

This version adds the first **finite multiplayer general-sum** solver
to GameTheory. It supports a game with 3–5 players, each with 1–4
actions, **at most 36 joint action profiles**, and finite payoffs
with absolute magnitudes at most 1,000,000. Inputs use one rectangular
N-dimensional payoff tensor per player, with shape
`[players, actions_player0, ..., actions_playerN-1]`.

## API

`POST /api/multiplayer/exact`:

```json
{
  "payoffs": [
    [[[2,0],[0,2]],[[0,0],[0,0]]],
    [[[2,0],[0,2]],[[0,0],[0,0]]],
    [[[2,0],[0,2]],[[0,0],[0,0]]]
  ],
  "mode": "correlated",
  "objective": "welfare",
  "max_operations": 200000
}
```

The example uses identical 2×2×2 payoff tensors for 3 players.
For general applications provide `payoffs[p][a0]...[aN-1]`
for each player `p`.

The `mode` options:
- `pure`: enumerate every pure action profile and check *every*
  player's unilateral pure-action deviations using exact rationals.
  A complete pure Nash set is reported, including the empty set.
- `correlated` (default): compute the complete pure Nash set and
  **one** exact rational correlated-equilibrium joint distribution.
  The `objective` is `welfare` (maximize raw sum of expected player
  payoffs) or `feasible` (find any valid CE). In welfare mode,
  the objective's optimality is certified with a separate rational dual.

Correlated distributions are **flattened** in lexicographic
`itertools.product` order; `action_profiles` gives the exact
corresponding action tuples. This avoids ambiguous nested indexing.
The response includes `joint_distribution_exact`,
`expected_payoffs_exact`, `maximum_obedience_gain_exact`,
the ordered rational dual multipliers, and the number of
recommendation-obedience inequalities evaluated.

## Mathematical model

For every player `p`, recommended action `i`, and alternative `k`:

```text
sum_{a_-p} q(i,a_-p) [u_p(k,a_-p) - u_p(i,a_-p)] <= 0
q(a) >= 0
sum_a q(a) = 1
```

These are unconditional obedience inequalities; they also hold when
the probability of recommendation `i` is zero. The rational LP
maximizes `sum_a q(a) sum_p u_p(a)`, or zero in feasibility mode.
The existing phase-I/II rational simplex is reused, with a declared
work budget of at most 2 million charged tableau updates.

**Independent output checks** re-evaluate normalized probabilities,
every player's actual gain from every recommendation deviation,
and every expected payoff using original game payoffs rather than
the LP coefficient rows.

For the welfare objective, the separately solved exact dual satisfies

```text
y >= 0
A^T y >= c
b^T y = c^T q
```

with `A q <= b, q >= 0`. These original exact rational
primal/dual feasibility and zero-gap equalities certify
optimality *for the represented rational input game*.
Feasible-objective mode has zero objective with a trivially valid
zero dual witness.

**Exactness scope:** all input payoff entries are first validated as
finite float64, then treated as `Fraction(str(float(value)))`.
The solver does not preserve arbitrary-precision original JSON
literals that round during Pydantic/NumPy conversion.

## Validation and evidence

- Exhaustively scan all pure strategy profiles and their unilateral
  deviations, independently rechecked for every benchmark fixture.
- Check exact CE joint-probability sums, nonnegativity, expectation
  and every recommendation-obedience inequality independently.
- Reconstruct rational primal–dual constraints in the test harness;
  test exact optimality against an independently built SciPy/HiGHS
  numerical LP on matched integer inputs.
- Include 3-, 4-, and 5-player games; coordination, all-ties, seeded
  random, no-pure-equilibrium, and degenerate action-count cases.
- Inject bad primal distributions, violated obedience and invalid
  dual multipliers; reject invalid payloads and work exhaustion.
- Run a seeded 12-case performance/correctness campaign and archive
  the evidence under `multiplayer-exact-v58-evidence` in Actions.

## Boundaries

This is **not** a solver for general multiplayer *mixed Nash equilibria*.
A correlated equilibrium is a distinct, generally broader solution
concept. The pure Nash subset can be empty even though a correlated
equilibrium exists. Returning **one** correlated distribution is not
enumeration of the full correlated-equilibrium polytope. Exponential
profile growth and exact rational coefficient size limit scalability;
operation counts do not bound bit growth, CPU time or memory usage.

The old two-player correlated-equilibrium endpoint remains unchanged.
No Gambit/OpenSpiel multiplayer reference is claimed in v58; SciPy/
HiGHS provides a numerical independent LP oracle on the same instances.
A global SOTA claim requires broader game classes, comparator evidence,
and large-instance benchmarks.
