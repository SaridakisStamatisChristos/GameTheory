<div align="center">

# Game Theory Lab

**A local-first computational laboratory for strategic reasoning.**

Explore equilibria, incentives, learning dynamics, and imperfect-information games through an interactive React interface and a research-oriented Python API.

[![Verification](https://github.com/SaridakisStamatisChristos/GameTheory/actions/workflows/verification.yml/badge.svg)](https://github.com/SaridakisStamatisChristos/GameTheory/actions/workflows/verification.yml)
![Python](https://img.shields.io/badge/Python-FastAPI-3776AB?logo=python&logoColor=white)
![Frontend](https://img.shields.io/badge/React-Vite-646CFF?logo=react&logoColor=white)
![Scope](https://img.shields.io/badge/scope-research%20laboratory-336B65)

[Quick start](#quick-start) · [What you can explore](#what-you-can-explore) · [API](#api-at-a-glance) · [Verification](#verification-and-benchmarks) · [Limitations](#research-integrity-and-limitations)

</div>

---

## Why Game Theory Lab?

Game theory becomes more useful when you can **change a payoff, run a solver, and inspect the incentive to deviate**. Game Theory Lab brings several complementary methods into one workspace: analytic calculations, linear programming, support enumeration, iterative learning, counterfactual regret minimization, and numerical certificates.

The project is designed for **personal research, education, and experimentation**—not as a hosted multi-user SaaS product. It emphasizes inspectable algorithms, reproducible experiments, and honest reporting of approximation and solver limits.

## What you can explore

| Research area | Available capabilities |
|---|---|
| **Normal-form games** | Editable two-player payoff matrices; pure Nash equilibria; strict pure-strategy dominance; analytic 2×2 equilibria |
| **Mixed Nash equilibria** | Adaptive dimension-independent solver core with certified candidates and budgeted exact support search; existing complete exact polyhedral geometry (up to opt-in 5×5) |
| **Approximate equilibria** | Restarted fictitious-play search with independently evaluated unilateral-regret (`ε`) diagnostics |
| **Zero-sum games** | Linear-programming minimax strategies with duality-gap checks |
| **Correlated equilibrium** | Obedience-constrained LP and checks of incentive violations in original payoff units |
| **Evolutionary games** | Replicator-dynamics trajectories, including preservation of initially absent strategies |
| **Extensive-form games** | Backward induction for finite perfect-information games; vanilla CFR and CFR+ for finite two-player zero-sum imperfect-information trees |
| **Bayesian games** | State-observable policy evaluation and separate private-type pure-policy deviation analysis |
| **Repeated games** | Discounted payoff simulation for specified periodic strategy sequences |
| **Experiments** | Local SQLite experiment saving and retrieval, with storage quotas |
| **Research evidence** | Backend regression and adversarial tests, numerical-certificate checks, and reproducible benchmark scripts |

### The interactive studio

The browser interface currently includes:

- An editable payoff matrix and four familiar games: **Prisoner's Dilemma**, **Matching Pennies**, **Battle of the Sexes**, and **Rock–Paper–Scissors**.
- One-click analysis of pure Nash equilibria, strict dominance, mixed-support candidates, minimax strategies, approximate Nash profiles, correlated equilibrium, and evolutionary dynamics.
- Mixed-strategy probability displays, numerical regret/gap diagnostics, a simple evolution chart, and an experiment-save action.
- An editable JSON tree workspace for **CFR+**, with average strategies and exploitability diagnostics.

Some more specialized algorithms are available **through the API rather than the browser UI**.

## Quick start

**Recommended development environment:** Python 3.13, Node.js 22, and Git. The Python dependencies include FastAPI, NumPy, SciPy, Pydantic, pytest, and HTTPX.

### 1. Clone the repository

```bash
git clone https://github.com/SaridakisStamatisChristos/GameTheory.git
cd GameTheory
```

### 2. Start the backend

Open a terminal:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -c constraints-audit-v38.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API is served at **http://127.0.0.1:8000**.

- Interactive API reference: **http://127.0.0.1:8000/docs**
- Health check: **http://127.0.0.1:8000/api/health**

### 3. Start the frontend

In a **second** terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open **http://localhost:5173**. Vite proxies `/api` requests to the backend on port 8000.

> **Dependency reproducibility:** `frontend/package-lock.json` is committed from a successful GitHub Actions dependency resolution. The workflow uses `npm ci`, and the Vite production bundle has passed CI. Browser/API end-to-end behavior remains a separate verification target.

## Your first experiment

1. Open the browser and select **Matching pennies**.
2. Run **Analyze Nash equilibria**: the game has no pure-strategy equilibrium, but does have an interior mixed equilibrium.
3. Run **Solve zero-sum game** to inspect minimax strategies and the reported duality gap.
4. Change a payoff and see how best responses and equilibrium predictions change.
5. Try the **CFR+** example in the lower panel to examine average behavioral strategies and exploitability in a small imperfect-information tree.

You can also call the API directly:

```bash
curl -sS http://127.0.0.1:8000/api/health

curl -sS -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"row_payoffs":[[1,-1],[-1,1]],"col_payoffs":[[-1,1],[1,-1]]}'
```

The second request analyzes Matching Pennies, with separate row-player and column-player payoff matrices.

## API at a glance

FastAPI provides request schemas and live examples at [`/docs`](http://127.0.0.1:8000/docs) when the server is running.

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Pure Nash, strict pure dominance, interior 2×2 mixed equilibrium |
| `POST /api/equilibria/analytic-2x2` | Exact-rational complete 2×2 Nash-set geometry (points, segments, rectangles) and representative profiles |
| `POST /api/equilibria/support-lp` | Support-pair LP search with explicit unresolved/completeness diagnostics |
| `POST /api/equilibria/exact-support` | Exact feasible-support classification (4×4 default; opt-in 5×5) |
| `POST /api/equilibria/exact-nash-set` | Exact Nash-set families and optional closure vertices/dimensions (4×4 default; opt-in 5×5) |
| `POST /api/equilibria/general` | Adaptive solver with exact rational simplex, proven pure-dominance pruning, optional exact mixed-dominance pruning (`mixed_pruning=true`), and explicit completeness states |
| `POST /api/equilibria/hybrid` | Fast support search combined with optional LP exploration |
| `POST /api/equilibria/lemke-howson` | Lemke–Howson equilibrium method |
| `POST /api/equilibria/independent-lemke-howson` | Independent Lemke–Howson implementation |
| `POST /api/equilibria/approximate` | Heuristic approximate Nash search and `ε` certificate |
| `POST /api/zero-sum` | Minimax LP and numerical duality-gap diagnostics |
| `POST /api/correlated-equilibrium` | Numerical correlated-equilibrium candidate and obedience checks |
| `POST /api/evolution` | Replicator dynamics |
| `POST /api/tree/backward-induction` | Finite perfect-information backward induction |
| `POST /api/repeated/simulate` | Specified repeated-strategy payoff simulation |
| `POST /api/imperfect-information/cfr` | Vanilla CFR for finite zero-sum trees |
| `POST /api/imperfect-information/cfr-plus` | CFR+ and bounded exact pure best-response evaluation |
| `POST /api/bayesian/evaluate` | Common-observed-state policy evaluation |
| `POST /api/bayesian/private-evaluate` | Private-type pure-policy evaluation and conditional deviations |
| `POST /api/experiments`, `GET /api/experiments` | Save and list local experiments |
| `GET /api/experiments/{experiment_id}` | Retrieve a saved experiment |

**2×2 completeness:** The `equilibrium_components` response field represents the entire 2×2 Nash set as a union of closed probability rectangles, including degenerate line and surface equilibria. Its `*_bounds_exact` fields are rational strings. The legacy `equilibria` list is still representative only (`complete: false`); `geometry_complete: true` applies solely to the new geometry. Exactness is relative to the decimal strings of validated float64 payoffs, not arbitrary-precision original JSON literals.

**Important distinctions:** The v48 general endpoint's Python core is dimension-independent and uses exact rational simplex plus exact-sign pure support dominance pruning by default; opt-in `mixed_pruning=true` adds an exact LP test for mixed-strategy dominance, which may improve pruning but also increase computation; the local API retains its 20×20 matrix size safety cap. Its default automatic mode returns a verified candidate when possible, **not** proof of a complete equilibrium set. Explicit `mode="exact"` searches rational support families under budgets and claims completeness only when **every** support is exhausted. The exact-support endpoint still returns one rational witness per feasible support. The separate exact-nash-set endpoint characterizes all equilibrium families as finite rational constraint systems, including continuous families; it does not enumerate every continuum point. Optional `include_vertices=true` computes exact vertices and affine dimensions **of each strict-support cell's closure**, and `max_actions=5` with `max_pairs=961` explicitly enables bounded 5×5 games. These are not global connected-component counts. A low reported `ε` is evidence about a *returned profile*, not proof that a heuristic will converge on every game. Similarly, visiting all enumerated support pairs is not proof that every equilibrium has been found. CFR and CFR+ assume a valid finite two-player zero-sum model with the required recall properties.

## Verification and benchmarks

Run from the backend directory after installing dependencies. The repository's `pytest.ini` enforces **warnings-as-errors with no exemptions**:

```bash
cd backend
python -m compileall -q .
python -m pytest -q -W error --cov=. --cov-config=.coveragerc --cov-branch --cov-report=term-missing --cov-fail-under=99
```

The [CI workflow](.github/workflows/verification.yml) runs the backend regression, adversarial and exact-game differential suites; enforces at least **99% measured combined statement/branch coverage** under warnings-as-errors; checks dependency consistency; and builds the React/Vite frontend from the committed npm lockfile. The earlier v40 baseline measured 100% over 18 modules; the current report is the authority for coverage after subsequent changes. Test files, benchmark scripts and historical baselines are excluded from the denominator.

A reproducible benchmark entry point is included:

```bash
cd backend
python benchmark_algorithms.py --seed 1729 --repeats 5 --iterations 1000 --output benchmark-results.json
```

Benchmark timings depend on the machine and configuration. Treat committed benchmark JSON files as **historical experimental artifacts**, not universal speed or state-of-the-art claims.

For mathematical assumptions, numerical caveats, resource limits and test-scope details, see **[Validation and limitations](docs/VALIDATION.md)**, the [exact Nash-set proof and representation](docs/EXACT_NASH_GEOMETRY.md), the [general engine contract](docs/GENERAL_NASH_ENGINE.md), the [SOTA evidence gates](docs/SOTA_ENGINEERING_GATES.md), and the [reproducible Nashpy comparator](backend/benchmark_independent_nashpy_v49.py).

## Project layout

```text
GameTheory/
├── backend/
│   ├── main.py                  # FastAPI routes and input validation
│   ├── analytic_2x2.py          # Exact-input 2×2 reasoning
│   ├── support_lp.py            # Support-based LP solving
│   ├── lemke_howson.py          # Pivot-based mixed-equilibrium solver
│   ├── independent_lh.py        # Independent Lemke–Howson path
│   ├── approximate.py           # Approximate Nash search
│   ├── cfr.py / cfr_plus.py     # Imperfect-information solvers
│   ├── research.py              # Correlated equilibrium and experiments
│   ├── api_guard.py             # Local-service request/work budgets
│   ├── benchmark_*.py           # Reproducible experiment scripts
│   └── test_*.py                # Regression and adversarial tests
├── frontend/
│   ├── src/main.jsx             # Interactive research studio
│   ├── src/style.css            # UI styles
│   └── vite.config.js           # Local API proxy
├── .github/workflows/
│   └── verification.yml         # Verification workflow
└── docs/VALIDATION.md           # Numerical contracts and verification scope
```

## Local-first operation

By default, launch Uvicorn bound to `127.0.0.1`. The service implements request-size, rate, solver-work, concurrency, and local experiment-storage budgets. If `GAME_LAB_API_TOKEN` is configured, remote API requests require a bearer token; experiment routes require it even locally. These controls support a **trusted local research service**, not a secure public multi-user deployment.

The optional SQLite database defaults to `backend/game_lab.sqlite3` when the server is launched from `backend`; it is ignored by Git. Set `GAME_LAB_DB` to choose a different location.

## Research integrity and limitations

Game Theory Lab is a **research workbench, not an all-games equilibrium oracle**.

- General finite bimatrix equilibrium enumeration is **not proven complete**, especially for degenerate or ill-conditioned cases. Support-LP results distinguish explored support pairs from unresolved feasible supports.
- Correlated-equilibrium output is a **numerically certified candidate** within stated tolerances. The welfare objective uses separately normalized player utilities rather than raw-payoff social welfare.
- Approximate Nash methods are **heuristic**; a measured unilateral-regret gap does not imply universal convergence.
- CFR/CFR+ models require **two-player, zero-sum, perfect-recall** games. Certificate availability and cost depend on game representation and size.
- Passing backend tests and the CI frontend build do **not** establish browser-level end-to-end behavior, arbitrary large-game correctness, formal verification, or comparative performance superiority.

See [Validation and limitations](docs/VALIDATION.md) for the implemented resource limits, mathematical assumptions, test boundaries, and validation commands.

## Roadmap

Future work focuses on **algorithm quality before infrastructure**: harder differential-test oracles; more degenerate-game handling; numerical sensitivity analysis; stronger exploitability and solution certificates; broader game classes; credible performance baselines; and browser-level integration testing.

## Licensing

**Proprietary — all rights reserved.** © 2026 Stamatis-Christos Saridakis. Public repository visibility does **not** grant a general license to use, modify, distribute, sublicense, or sell the original software. See [LICENSE](LICENSE) for the notice and contact the repository owner to request commercial, research, or evaluation rights. Third-party dependencies retain their own licenses.

---

<div align="center">

**Game Theory Lab** · Built for studying strategies, not just displaying answers.

</div>
