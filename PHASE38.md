# Phase 38 — Correctness and controlled local-service hardening

Source: supplied `game-theory-lab-v37.zip` and `Game_Theory_Lab_v37_Adversarial_Evidence.zip` (2026-11-03). This release is an audited repair candidate, not a certified public deployment or SOTA-performance claim.

## Verified repairs

- Strict-versus-weak dominance and exact pure best-response classifications fixed.
- Normal-form 2×2 indifference uses rational input comparisons; separate analytic solver also uses rational best-response geometry. Missing rare interior supports and scale-dependent pure equilibria repaired.
- Replicator discretization preserves precisely absent strategies.
- Private-type Bayesian Nash assessment now uses conditional expected action advantages for positive-probability types; separate ex-ante gain diagnostics preserved.
- Support-pair LP uses per-player normalization, explicit unresolved-support status, and exact rational 2×2 interior witness recovery. `complete_support_coverage` means *combinatorial pairs visited*, not proven complete equilibrium discovery.
- Correlated-equilibrium LP uses per-player normalization and **independent original-payoff obedience checks**. Result is labeled a numerically certified candidate with disclosed tolerance; its welfare objective is the sum of individually normalized player payoffs, not raw utility welfare.
- Backward induction selects exact highest-payoff actions, avoiding absolute-tolerance promotion of strictly inferior branches.
- Vanilla CFR rejects aliased-node DAGs, recursion cycles and excessive node×iteration work.
- Zero-sum minimax LP centers/scales before optimization and checks independent maximin/minimax outcomes; extreme finite fictitious-play payoffs use extended precision and JSON-safe diagnostics.
- Frontend formats finite diagnostics encoded as strings without invoking number-only methods.
- Local API: bounded request bodies, local/remote isolation, optional bearer-token remote access, experiment-record quota, request-rate limit, heavy-solver concurrency limit and per-solver work budgets. Distinct helper `api_guard.py`.
- Formerly ambiguous `cfr_plus_compiled` descriptor corrected to *pre-indexed pure-Python traversal*; no native compiled performance claim.

## API and operational changes

- Run locally: `cd backend && uvicorn main:app --host 127.0.0.1 --port 8000`.
- The default interface is local-only. A remote client receives HTTP 403 unless `GAME_LAB_API_TOKEN` is set and the request includes `Authorization: Bearer <token>`. When the variable is set, experiment routes require that header even from localhost. No end-user authentication or tenant isolation is implied. Do not deploy as a shared public internet service.
- POST body limit: 256 KiB. Local request-rate budget: 1000/minute; remote: 120/minute, per process and peer IP. Solver concurrency: two in-flight expensive endpoints per process. SQLite experiment count: 1000; total approximate record bytes: 10 MB. Some excessive requests return 413, 422, 429 or 507.
- CFR and CFR+ impose a 2,000,000 node×iteration budget. Fictitious play imposes 500,000 iteration×restart budget. Support-LP API enumeration is capped at 5000 support pairs, and the fast-support candidate budget is 10,000.
- These are local-service safeguards, not proof of anti-DoS resistance behind arbitrary reverse proxies or distributed infrastructure.
- `correlated-equilibrium.solution` is now a *numerically certified candidate* status; `violation_units` means original payoff units. `objective_units` documents independently normalized welfare.
- `support_lp.all_feasible_supports_resolved` is explicitly `false`: a numerical LP does not establish complete support feasibility. `complete_support_coverage` retains legacy meaning of visiting every support pair; consumers must not interpret it as a completeness proof.

## Reproduction

```sh
cd backend
python -m pytest -q -W error
python -m compileall -q .
```

The backend includes `test_adversarial_v38.py` (10 prior witnesses) and `test_security_v38.py` (16 additional security, numerical, random oracle, work-budget and overflow tests). Independent brute-force CFR+ best response oracle is retained in the evidence ZIP provided separately, not imported into production.

## Unverified and remaining gates

- Frontend Vite bundling could **not** run in this offline container: npm registry resolution returned `EAI_AGAIN`, and the source archive has no npm lockfile. JSX syntax parsing with installed TypeScript did succeed. The added CI workflow has not executed here and installs frontend dependencies without a lockfile. Generate and commit `frontend/package-lock.json`, change CI to `npm ci`, run browser/API integration tests, and examine the actual CI logs before claiming full-stack readiness.
- No independent production deployment, authorization penetration testing, load testing or fault-injection exercise was executed. This remains a trusted-local educational/research tool, not a validated multi-tenant service.
- General finite-game Nash completeness, continuity of degenerate equilibrium sets, solver optimality for arbitrary ill-conditioned games, global SOTA and comprehensive formal verification are **not** established.
