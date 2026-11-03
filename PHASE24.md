# Phase 24 — independent numerical certificates

Backend only. Added `backend/numerical_certificate.py`, independent tests, and `benchmark_certificate_v24.py` with raw JSON results.

## Verified locally
- `python -m pytest -q --disable-warnings`: 67 passed, 3 warnings.
- Benchmark: 60 seeded instances, 3 repetitions per instance, 60/60 with at least one certified equilibrium, zero invalid returned certificates; median per-instance latency 0.5475 ms.
- Families: integer, zero-sum, asymmetrically scaled; dimensions 2, 3, 5, 8.

## Caveats
- Certificate uses floating-point arithmetic and is an a posteriori tolerance check, not a formal proof.
- Payoff range scaling is translation-invariant, but the absolute tolerance can be too permissive for very small payoffs; report thresholds alongside verdicts.
- Benchmark does not establish exhaustive equilibrium discovery or SOTA.
- The pivot engine still uses a fixed absolute tolerance for accepting equilibria. Future work: integrate scale-aware checks into solver without compromising numerical stability; benchmark repeated randomized-order differential runs against Nashpy.
