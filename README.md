> **Current repair release: v38 (2026-11-03).** Read [PHASE38.md](PHASE38.md) for correctness fixes, model changes, request/solver budgets and limitations. Run the API bound to `127.0.0.1`; remote clients require `GAME_LAB_API_TOKEN` with bearer authorization. A numerical support enumeration result is **not** proof of complete Nash enumeration. The backend test suite passes locally; the frontend production build and remote CI are not verified in this archive. This is a trusted-local research/educational laboratory, not a public multi-tenant service.

# Game Theory Lab

A runnable React + FastAPI full-stack game theory analysis studio.

## Features
- Editable bimatrix normal-form games and four example games
- Pure-strategy Nash equilibrium enumeration
- Strict pure-strategy dominance detection
- Interior 2×2 mixed Nash equilibrium when nondegenerate
- General finite two-player zero-sum minimax solver (linear programming, up to 20×20)
- API input validation, API docs, backend tests

## Run

Requires Python 3.10+ and Node 20+.

Terminal 1:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Terminal 2:
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173. API docs: http://localhost:8000/docs.

## Tests
```bash
cd backend
pytest -q
```

## Scope
This is a functional MVP, not a complete game-theory research platform. It does not yet compute all mixed Nash equilibria in general bimatrix games, extensive-form subgame-perfect equilibria, Bayesian games, evolutionary dynamics, repeated games, or multiplayer games. The zero-sum solver assumes the second player's payoffs equal the negation of the first player's payoffs; the frontend enforces this constraint. No persistent database or user accounts are included.

## Phase 2 extensions
- `POST /api/equilibria`: verified equal-size support enumeration (up to support size 4); incomplete for degenerate games.
- `POST /api/evolution`: simplex-preserving exponential replicator dynamics, with trajectory data.
- Frontend controls and evolutionary trajectory chart.

## Phase 3: extensive-form and repeated games

### Backward induction
`POST /api/tree/backward-induction` with a root ID and nodes. Nonterminal nodes specify `player` (zero-indexed) and `children`; terminal nodes specify `payoffs`. Supports 2–8 players and at most 500 nodes. All nodes must be reachable. Returns one pure subgame-perfect equilibrium selection with deterministic first-child tie-breaking, plus all optimal children at each decision node. Only perfect-information finite trees are supported.

### Repeated games
`POST /api/repeated/simulate` with `row_payoffs`, `col_payoffs`, `row_strategy`, `col_strategy`, `rounds`, and `discount`. Strategy lists are repeated periodically. The API reports discounted cumulative payoffs; it does not claim strategic equilibrium. Traces are returned for up to 200 rounds.

### Test
From `backend`, run `python -m pytest -q` (10 tests passing in the development environment). Frontend production build and deployment remain unverified.

## Phase 4: correlated equilibria, public-state games, persistence
- `POST /api/correlated-equilibrium`: computes welfare-maximizing correlated equilibrium using LP obedience constraints and returns numerical violation.
- `POST /api/bayesian/evaluate`: evaluates state-contingent pure actions when both players observe the state. **Not** a private-information Bayesian Nash solver.
- `POST /api/experiments`, `GET /api/experiments`, `GET /api/experiments/{id}`: SQLite persistence; set `GAME_LAB_DB` for database location.
- Frontend includes correlated-equilibrium action and save experiment action.
- Tests: 15 backend tests passed in the development environment. Frontend build remains unverified due to npm installation timeout.

### Security and deployment limitations
Experiment APIs have no authentication and no multi-user isolation. Do not expose this build to the public internet without adding authentication, rate limits, request-size limits, migration management, and a production deployment configuration. General mixed-equilibrium support enumeration remains incomplete for degenerate games; no claim of global SOTA is made.

## Phase 5: private-information Bayesian games
`POST /api/bayesian/private-evaluate` accepts `joint_prior[type_row][type_col]`, payoff tensors `[type_row][type_col][action_row][action_col]`, and pure policies (one action per private type). It returns expected payoffs and per-type prior-weighted unilateral deviation gains. This verifies a *specified pure Bayesian Nash equilibrium*, not enumeration of all Bayesian equilibria or mixed Bayesian strategies. Zero-probability types are treated as unconstrained. Input dimensions and finite probabilities are validated.

### Evidence and remaining gaps
17 backend pytest tests passed locally on 2026-11-03. Frontend build and browser integration have not been verified. General degenerate Nash enumeration, imperfect-information extensive-form games, mixed Bayesian equilibrium solving, benchmarks, and production deployment remain open. SOTA status is **not** claimed.

## Phase 6 — Approximate Nash profiles

`POST /api/equilibria/approximate` accepts `row_payoffs`, `col_payoffs`,
`iterations` (100–200000), `restarts` (1–64), and `seed`.
Returns the best observed empirical strategy profile and a verified unilateral
regret certificate (`epsilon`) in original payoff units. Simultaneous fictitious
play is a heuristic for general-sum games and does **not** guarantee convergence.

Verification: `cd backend && pytest -q` — 21 tests passed on 2026-11-03.
Frontend production build and independent solver benchmarks remain outstanding.

## Phase 7: Approximate Nash UI
The frontend now exposes the `/api/equilibria/approximate` endpoint with adjustable iterations, restarts and seed. It displays both mixed strategies, expected payoffs, and an independently calculated maximum unilateral-deviation gain (epsilon) in the original payoff units. The method is heuristic and does not guarantee convergence in general-sum games.

## Phase 8: Imperfect-information CFR
`POST /api/imperfect-information/cfr` accepts a rooted finite extensive-form tree with terminal `payoff` (row-player utility), chance nodes (`player: "chance"`, `actions`, `probabilities`), and player nodes (`player: 0` or `1`, `infoset`, `actions`). All nodes must be reachable and acyclic; nodes sharing an information set must have identical actions and player. Use unique node IDs. The model assumes zero-sum utilities and perfect recall; it does not automatically verify perfect recall or calculate exploitability. Full-tree vanilla CFR is intended for small games; the `iterations` maximum is 20,000. The solver returns average behavioral strategies and their expected row payoff. Example matching-pennies convergence is covered by tests.

## Phase 9: CFR+ and bounded exact exploitability

`POST /api/imperfect-information/cfr-plus` accepts the same tree format as the CFR endpoint, plus `max_br_profiles` (default 100000). It performs alternating regret-matching+ updates and linear averaging, validates tree structure and own-action perfect recall, and computes an exact pure best response for each player when the number of pure information-set policies fits the requested bound. `exploitability` is the sum of both unilateral improvement opportunities in player-zero payoff units; `certificate_status` explicitly reports when computation was skipped. The algorithm assumes finite two-player zero-sum games and perfect recall. Validation checks own-action recall but does not constitute a complete formal proof of perfect recall for arbitrary representations. Larger games need sequence-form best responses, sampling, and performance benchmarking.

Run backend checks with `cd backend && python -m pytest -q`.

## CFR+ web laboratory (v11)

The frontend now exposes the CFR+ solver through an editable JSON game-tree editor. The bundled Matching Pennies example includes a shared information set for the second player. The result panel displays average behavioral strategies, expected payoff, and the exact-tree best-response exploitability gap. The backend enforces perfect recall, finite terminal payoffs, and a rooted tree.

Start the API with `cd backend && uvicorn main:app --reload`; start the frontend with `cd frontend && npm install && npm run dev`. The Vite dev server proxies `/api` to port 8000.

**Scope:** CFR+ is for finite, two-player zero-sum perfect-recall trees; it is not a general multi-player or general-sum solver. This release does not establish SOTA benchmark performance.

## Phase 14: Hybrid Nash solver and reproducible benchmark

`POST /api/equilibria/hybrid` accepts the usual payoff matrices plus `max_pairs`,
`fast_support`, and `exhaustive`. Fast-path output is **not exhaustive**. Identical
payoff rows/columns for both players trigger LP fallback when within budget;
this heuristic does not detect every form of degeneracy. `exhaustive=true` runs
all support pairs within the budget, returning one representative per feasible
support pair, **not** the full continuum of equilibria.

Run from `backend/`: `python -m pytest -q` and `python benchmark_hybrid.py`.
The recorded measurements are in `backend/benchmark_hybrid_results.json`.
They are local wall-clock measurements, not cross-library SOTA evidence.

## Phase 15: independent equilibrium comparison

From `backend/`, install optional benchmark dependency `pip install nashpy==0.0.43`, then run `python reference_benchmark.py --output benchmark_reference.json`. The deterministic 16-game campaign measures the hybrid solver against Nashpy support enumeration and independently computes Nash gaps. Nashpy support enumeration is not a completeness oracle; it can return no equilibria for some games. Run `pytest -q` for regression tests. The frontend is unchanged.

## Phase 16: Lemke–Howson comparison

An optional Nashpy-backed Lemke–Howson pivot-path adapter is available at
`POST /api/equilibria/lemke-howson`. Install `nashpy==0.0.43` for this endpoint.
Results are checked for simplex validity and independent Nash gap. Degenerate
pivot failures are returned as diagnostics. The solver is not exhaustive.

From `backend/`, run `pytest -q` and `python benchmark_lemke.py` to reproduce
`benchmark_lemke_results.json`. The benchmark compares hybrid and Lemke–Howson
on deterministic games with five repetitions and records solver failures.
Neither runtime measurements nor passing tests establish SOTA performance.
