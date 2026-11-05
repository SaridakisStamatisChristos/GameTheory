# Verification, numerical contracts and limitations

Game Theory Lab is a **local-first research workbench** for finite games, not a general mathematical proof assistant. The repository provides reproducible experiments, explicit input budgets, and independently checked numerical diagnostics. Algorithm behavior must be judged by the declared game model and solver method.

## Reproduce the checks

Recommended environment: Python 3.13 with dependencies installed from `backend/requirements.txt` under `backend/constraints-audit-v38.txt`; Node.js 22 with `frontend/package-lock.json`.

```bash
python -m pip install -r backend/requirements.txt -c backend/constraints-audit-v38.txt
python -m pip install -c backend/constraints-audit-v38.txt nashpy==0.0.43 pytest-cov==7.0.0
cd backend
python -m pytest -q -W error --cov=. --cov-config=.coveragerc --cov-branch --cov-report=term-missing --cov-fail-under=99
python -m compileall -q .
```

For the independent frontend build:

```bash
cd frontend
npm ci --no-audit --no-fund
npm run build
```

The CI workflow runs these checks on pushes and pull requests. The earlier v40 audit established a baseline of **251 passing tests** and **100% statement and branch coverage** over 18 core modules. The current CI run is the authority for the enlarged suite and scope. CI enforces at least **99% combined coverage**. The denominator excludes tests, standalone benchmarks, historical baselines and benchmark drivers. It does **not** measure real-browser behavior or prove all input combinations correct.

## Numerical meaning of results

- **Pure and mixed Nash.** The `pure_search_exhausted` field now means the pure-pair scan actually visited every reduced pure cell; `pure_cells_checked` discloses the number inspected.  Returned profiles are checked against best responses where the solver provides such certification. General finite bimatrix equilibrium *enumeration* is not proven complete, especially under degeneracy or severe conditioning.
- **2×2 analysis.** The analytic endpoint now supplies `equilibrium_components`, an exact-rational finite union of points, segments, and rectangles that covers the complete Nash set even for degenerate 2×2 games. Its legacy `equilibria` list contains representative points only. Exactness applies to the decimal strings of validated float64 input payoffs, not the original arbitrary-precision JSON text or all downstream floating-point payoff diagnostics.
- **Generalized Nash solver v48.** `/api/equilibria/general` uses a dimension-independent computational core. In auto mode it tries exact pure-best-response detection, sound iterated strict pure-strategy dominance reduction, constant-sum primal/dual minimax LP, bounded independent Lemke–Howson, and seeded approximate fictitious play. All returned profiles are evaluated on the **original** payoff matrices using the existing numerical deviation certificate; the route never silently equates finding one candidate with completing the Nash set. In `mode="exact"`, it explores feasible rational support cells (product H-polytopes), with no hardcoded 5×5 core limit, but strict support-pair, support-cardinality and rational-work budgets. Only exhaustive full support processing yields `equilibrium_set_complete=true`. Its v48 exact-search path defaults to a two-phase rational simplex with Bland pivot selection and provably safe, support-conditional pure weak-dominance pruning; `prune_supports=false` is the comparison baseline. An additional opt-in `mixed_pruning=true` tests exact-support mixed-strategy weak dominance via rational LP, reports rational mixture certificates, and charges every LP pivot to the same work budget. Exhaustion is reported as partial; it never certifies an unexamined support. This stronger pruning can be slower than pure pruning and must be benchmarked case-by-case. A pruned support counts as *mathematically classified infeasible* but not as an LP solve. The `support_pairs_checked` count includes both LP-classified and pruned supports, while `lp_pairs_attempted` counts attempted feasibility checks. Partial searches never claim full completeness. The optional pruning precomputation is not charged against the exact arithmetic operation budget and can impose overhead on small or tie-heavy games; the vertex-enumeration backend remains selectable for independent differential testing. The exact arithmetic is over the decimal strings of validated float64 payoffs. Exact simplex pivot counts do **not** bound coefficient bit growth or runtime. See [general engine contract](GENERAL_NASH_ENGINE.md).
- **Exact closure geometry.** Request `include_vertices=true` on `/api/equilibria/exact-nash-set` to enumerate all rational vertices of each factor-polytope closure and compute exact affine dimensions. The product dimension is the sum of factor dimensions. Closure vertices may have zero probability on actions required positive by the original strict-support cell; their status is disclosed. Candidate-basis exhaustion fails closed with HTTP 422. The vertex algorithm is exact for its finite dimensions, but worst-case cost is combinatorial.
- **Complete Nash-set representation (4×4 default; opt-in 5×5).** The new `/api/equilibria/exact-nash-set` endpoint returns a finite union of exact rational product-polytopal families, one for each feasible strict-support pair. The linear equalities/inequalities and positivity constraints describe **every Nash strategy pair**, including infinitely many equilibria in degenerate games, relative to the validated float64-decimal payoff interpretation. It reuses the exact feasibility work budget and fails closed if enumeration cannot finish. See [the mathematical derivation](EXACT_NASH_GEOMETRY.md). This is a compact symbolic characterization, not numerical sampling or an explicit list of all points.
- **Exact support feasibility (4×4 default; opt-in 5×5).** The new `/api/equilibria/exact-support` endpoint enumerates every nonempty row/column support pair and uses exact rational Fourier–Motzkin feasibility with **strictly positive** support probabilities, followed by a direct exact Nash-deviation certificate. A successful response guarantees one exact Nash witness for **every feasible support pair** of the validated decimal float64 game. It does **not** describe all equilibria within each support (which may form a continuum). The solver is exponential and fails explicitly with HTTP 422 if its support-pair or elimination-work budget is exhausted. Results include authoritative rational probability strings; floating projections may underflow and are flagged.
- **Support-LP and hybrid methods.** The existing numerical solvers are unchanged. Visiting all support pairs numerically does not prove they were all resolved; numerical LP feasibility and exact support completeness are distinct.
- **Correlated equilibrium.** Output is a numerical candidate checked against original-payoff unilateral obedience deviations, plus per-player-normalized residuals. Its welfare objective uses the sum of **individually normalized** player utilities, not necessarily raw utilitarian welfare.
- **Zero-sum minimax.** Payoffs are centered/scaled before LP optimization and the reported certificate includes independent primal/dual diagnostics; numerical tolerances still apply.
- **Approximate Nash.** Fictitious-play search is heuristic in general-sum games. Reported epsilon is a maximum unilateral-deviation gain for the *returned* strategy profile, not a universal convergence guarantee.
- **Extensive-form and CFR/CFR+.** Imperfect-information solvers assume finite two-player zero-sum games with the required perfect-recall properties. Pure best-response exploitability depends on the modeled information sets and budget; the game specification itself must be valid.
- **Bayesian games.** Private-type evaluation validates *provided pure policies* through conditional deviations; it does not enumerate general mixed Bayesian equilibria.
- **Evolutionary and repeated games.** Replicator integration preserves zero-supported strategies; repeated-strategy simulation reports payoffs, not necessarily equilibria.

## Local service budgets

The service is intended for trusted local research. Typical configured limits include:

| Control | Current policy |
| --- | --- |
| HTTP request body | 256 KiB |
| Normal-form matrices | At most 20 × 20 |
| Local per-process rate | 1,000 requests/minute per source |
| Remote per-process rate | 120 requests/minute per source |
| Concurrent expensive solver requests | 2 |
| CFR / CFR+ work | 2,000,000 nodes × iterations |
| Approximate Nash work | 500,000 iterations × restarts |
| Numerical support-LP API budget | 5,000 pairs |
| Exact support and exact Nash-set APIs | 4×4 default (225 pairs, max 500,000 operations); opt-in 5×5 (961 pairs, max 2,000,000 operations) |
| Exact closure vertex enumeration | Optional `include_vertices=true`, default 50,000 / maximum 200,000 candidate bases |
| General solver API | Up to 20×20 matrices; exact mode max 4,096 support pairs, max 2 million counted exact operations, max support cardinality 20 |
| General solver approximation | At most 500,000 iteration × restart units, with independent deviation certificate |
| Candidate fast-support work | 10,000 |
| Local experiment store | 1,000 records and approximately 10 MB |

These protections are local-service safeguards, not hardened multi-tenant authorization or proof of denial-of-service resistance. Run Uvicorn bound to `127.0.0.1` by default. If `GAME_LAB_API_TOKEN` is set, remote requests require an authorization bearer token, and experiment routes require it even locally. See `backend/api_guard.py` for the authoritative implementation.

## Benchmark interpretation and remaining gaps

Timing results in `backend/benchmark_*.json` are reproducible experiments, not hardware-independent performance guarantees. Benchmark comparison requires matching workload, hardware, solver semantics, warmup, repetition and error metrics.

Current missing evidence includes full browser/API end-to-end tests, polynomial-time guaranteed exact methods for general bimatrix games (not expected in general complexity models), efficient exact support-space pruning and independent large-game rational LP verification, global topology/connected-component extraction (as distinct from the per-cell geometry), independent large-game differential benchmarks, and production deployment assessment. Optional vertex computation reports vertices of **closed** factor polytopes, not necessarily points within each original strict-support cell. Full Nash-set completeness still means exact H-representation, not explicit enumeration of infinitely many points. Code coverage alone does not prove algorithmic correctness.

## v57 — Discounted zero-sum stochastic/Markov games

The new `/api/stochastic/zero-sum/discounted` API accepts finite
simultaneous-action Markov games with stochastic transitions and a
discount factor strictly less than one. The algorithm is Shapley
value iteration, with per-state primal/dual numerical zero-sum
matrix LPs and original-unit saddle security checks. The reported
numerical residual divided by `1-discount` bounds remaining value
error *under its floating-point LP model*; its doubled value bounds
stationary-policy unilateral regret. This is not formal interval
arithmetic. Iteration work exhaustion returns `partial_bound`
rather than falsely claiming convergence. The new API cap is
8 states × up to 6×6 actions, max 500 updates and a joint
state×iteration budget of 2,400. Tests compare one-state values
with known matrix games and action-independent transitions with
independent linear system oracles. See `STOCHASTIC_ZERO_SUM_V57.md`.

## v58 — Finite multiplayer general-sum games

`/api/multiplayer/exact` adds 3–5 player finite normal-form games
with up to 4 actions per player, at most 36 joint profiles.
All pure Nash equilibria are exhaustively classified with exact
rational unilateral-deviation checks. An optional exact LP computes
one correlated equilibrium with independent rational obedience
verification, and a rational primal–dual optimum when optimizing
raw sum of payoffs. A correlated equilibrium is **not** equivalent
to a mixed Nash equilibrium, and general multiplayer mixed Nash
enumeration remains unimplemented. An exact work budget of 1..2m
charged tableau updates applies, not a bounded coefficient bit size.
See `MULTIPLAYER_EXACT_V58.md`.
