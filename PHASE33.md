# Phase 33 — Differential verification and timing

Baseline: v32. Frontend unchanged.

## Changes
- `backend/test_phase33_differential.py`: 60 seeded random bimatrix cases, independent certificate validation, and affine-transformation regression.
- `backend/benchmark_phase33.py`: deterministic seed 3302, 8 cases per dimension, 5 repetitions, warmup, 4 initial labels maximum.
- `backend/benchmark_phase33_results.json`: raw summary, environment and latency quantiles.

## Executed verification
`cd backend && python -m pytest -q -W error`: 107 passed in 15.55 seconds, zero warnings.

## Caveats
The 60 randomized games are a small correctness sample, not a proof of universal completeness. Timing benchmark measures one implementation, not a SOTA comparison. All 200 timing runs returned at least one certified equilibrium. No claims about full enumeration, worst-case complexity or universal superiority.

## Next work
Differential oracle comparisons with independent equilibrium methods, degeneracy minimization, and CFR+ convergence/exploitability stress tests.
