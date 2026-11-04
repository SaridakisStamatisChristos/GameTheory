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

- **Pure and mixed Nash.** Returned profiles are checked against best responses where the solver provides such certification. General finite bimatrix equilibrium *enumeration* is not proven complete, especially under degeneracy or severe conditioning.
- **2×2 analysis.** The analytic endpoint now supplies `equilibrium_components`, an exact-rational finite union of points, segments, and rectangles that covers the complete Nash set even for degenerate 2×2 games. Its legacy `equilibria` list contains representative points only. Exactness applies to the decimal strings of validated float64 input payoffs, not the original arbitrary-precision JSON text or all downstream floating-point payoff diagnostics.
- **Exact support feasibility (up to 4×4).** The new `/api/equilibria/exact-support` endpoint enumerates every nonempty row/column support pair and uses exact rational Fourier–Motzkin feasibility with **strictly positive** support probabilities, followed by a direct exact Nash-deviation certificate. A successful response guarantees one exact Nash witness for **every feasible support pair** of the validated decimal float64 game. It does **not** describe all equilibria within each support (which may form a continuum). The solver is exponential and fails explicitly with HTTP 422 if its support-pair or elimination-work budget is exhausted. Results include authoritative rational probability strings; floating projections may underflow and are flagged.
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
| Exact support API | Up to 4×4, 225 pairs and default 100,000 elimination-work units |
| Candidate fast-support work | 10,000 |
| Local experiment store | 1,000 records and approximately 10 MB |

These protections are local-service safeguards, not hardened multi-tenant authorization or proof of denial-of-service resistance. Run Uvicorn bound to `127.0.0.1` by default. If `GAME_LAB_API_TOKEN` is set, remote requests require an authorization bearer token, and experiment routes require it even locally. See `backend/api_guard.py` for the authoritative implementation.

## Benchmark interpretation and remaining gaps

Timing results in `backend/benchmark_*.json` are reproducible experiments, not hardware-independent performance guarantees. Benchmark comparison requires matching workload, hardware, solver semantics, warmup, repetition and error metrics.

Current missing evidence includes full browser/API end-to-end tests, a constructive description of the full Nash equilibrium continuum for **3×3 and larger degenerate games**, broad independent large-game differential benchmarks, and production deployment assessment. Exact support feasibility is a narrower mathematical claim than full equilibrium-set enumeration. Complete statement/branch coverage alone does not provide those guarantees.
