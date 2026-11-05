# v57 — Discounted two-player zero-sum stochastic games

## Added capability

`POST /api/stochastic/zero-sum/discounted` solves bounded *finite,
simultaneous-action, two-player zero-sum Markov games* using Shapley's
discounted value iteration. This introduces action-dependent stochastic
state transitions, rather than reusing a single static matrix game.

Input `rewards[s][row_action][col_action]` is the row player's
immediate reward; the column player's reward is its negative.
`transitions[s][i][j][next_state]` is a probability distribution.
The return criterion uses **unnormalized** discounted payoffs
`sum(t>=0) discount**t * reward[t]`.

The local Shapley stage matrix is

```text
Q_s(V)[i,j] = rewards[s,i,j] + gamma * sum_t transitions[s,i,j,t] * V[t]
(T V)[s] = value_of_zero_sum_matrix_game(Q_s(V))
```

The solver uses *both* SciPy/HiGHS minimax LP directions for each
nonconstant matrix, centered and scaled for conditioning.
It validates the resulting mixed strategies in the normalized
stage payoff model and recomputes their security levels in original
payoff units, using higher precision for the security dot products.

## Mathematical residual and honest completion state

Let `L_s(V) = min_j sum_i x_s[i] Q_s(V)[i,j]` and
`U_s(V) = max_i sum_j Q_s(V)[i,j] y_s[j]` for the returned
stationary policies. Any exact stage-game value lies between
these two security quantities. Therefore the Shapley residual is
bounded by

```text
rho = max_s(max(abs(L_s(V) - V[s]), abs(U_s(V) - V[s])))
||V - V*||_infinity <= rho / (1 - gamma)
stationary-policy unilateral regret <= 2 * rho / (1 - gamma)
```

The last inequality follows from the lower-security and upper-ceiling
discounted fixed-point operators for stationary row and column
policies. These operators are monotone and gamma-contractive;
the security gap between them bounds deviation gains against
arbitrary opponent behavior. This is a mathematical statement for
exact expectations. **Computed bounds are numerical**, relying on
IEEE-754 arithmetic and toleranced HiGHS stage certificates;
they are *not* machine-verified, outward-rounded interval proofs.

`status='converged_numerical'` is returned only when the numerical
`value_error_bound <= tolerance`. If the iteration cap is
reached first, `status='partial_bound'` and
`tolerance_met=false`, and the unresolved bound is disclosed.

## Limits and input semantics

- Max 8 states, 6 row actions, 6 column actions, max 500 value
  updates, and at most 2,400 states×iteration work.
- Finite immediate rewards with absolute magnitude <=1,000,000;
  `0 <= discount < 1`, `1e-10 <= tolerance <= 1`.
- Probabilities must be nonnegative and sum to one within 1e-12
  per (state, row action, column action) distribution.
  Tiny accepted rounding deviations are normalized internally and
  their maximum row-sum correction is reported.
- The value estimate is for the **normalized accepted transition
  model**, not for arbitrary unnormalized input probabilities.
- Constant matrix games and 1×1 stages bypass the LP while
  retaining correct security bounds.
- Work cap controls iteration count, not every physical CPU,
  solver or memory resource. Near-undiscounted games can require
  more iterations than permitted, returning a partial bound.

Example request:

```json
{
  "rewards": [[[1,-1],[-1,1]]],
  "transitions": [[[[1],[1]],[[1],[1]]]],
  "discount": 0.9,
  "tolerance": 0.000001,
  "max_iterations": 200
}
```

Matching pennies in one absorbing state has value zero and
half-half equilibrium strategies for both players.

## Independent verification and evidence

- A one-state matrix game reduces to ordinary minimax;
  known 2×2 rational equilibrium values are checked independently.
- A single-action Markov game reduces to the linear system
  `(I - gamma*P)V = reward`; direct dense linear algebra verifies
  the result without any minimax or Shapley implementation shared
  with the solver.
- Action-independent transitions reduce to independently known
  matrix-game values plus a linear discounted transition equation.
- Seeded 2–4-state action-dependent games verify both players'
  actual stage best-response security values.
- Adversarial invalid input, truncated iteration and corrupted LP
  tests require explicit failure or partial-result semantics.
- A pinned, deterministic 10-case performance/correctness campaign
  records fixture SHA-256, iteration counts, raw timing samples,
  stage saddle gaps, bounds and analytic reference discrepancies.

Reproduce from `backend/`:

```bash
python -m pytest -q -W error test_stochastic_zero_sum_v57.py test_stochastic_http_v57.py test_stochastic_benchmark_v57.py
python benchmark_stochastic_v57.py --repetitions 3 --seed 57057
```

CI preserves backend >=99% statement/branch coverage, Gambit,
OpenSpiel and frontend checks. Artifact:
`stochastic-zero-sum-v57-evidence`.

## Remaining limitations

This is **not** a solver for general-sum stochastic games, three-player
games, private information, partially observable Markov games, or
average-reward undiscounted stochastic games. No full stochastic-game
polytope enumeration or formally certified interval bound is claimed.
No comparative speedup over OpenSpiel is asserted. Future work should
test independent stochastic-game reference solvers, larger instances,
multiplayer general-sum games, and true high-precision interval
verification.

## Mathematical reference

Shapley value-iteration contraction and residual derivation:
MIT 6.7980 supplementary reading,
https://www.mit.edu/~6.7980/stochastic_games.html
