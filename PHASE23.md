# Phase 23 — Conditioning and differential pivot audit

Frontend unchanged. Added `backend/benchmark_conditioning_v23.py` and `backend/test_conditioning_v23.py`.

Benchmark: 100 seeded matrices, dimensions 2,3,5,8,10, five families, four cases per family, first three initial labels. Single timed execution per solver per instance, sequential baseline then optimized (order bias possible). Both algorithms found at least one verified equilibrium in all 100 cases, with zero uncaught exceptions. Baseline median 0.8477 ms (p90 2.2116 ms); optimized median 0.6282 ms (p90 1.597 ms). These are exploratory timings, not controlled SOTA evidence. Full raw records: `backend/benchmark_conditioning_v23.json`.

Tests: 63 passed, 3 existing warnings. Test suite count increased from the 62 tests present in the v22 archive. Prior v21 reported 64 tests; that historical discrepancy is not resolved by this release.

Limits: tolerance uses absolute payoff units, which can distort validity decisions under very large or tiny payoff scales. No exhaustive initial-label coverage, no statistical confidence intervals, no external benchmark in this phase. Future work: scale-aware numerical certificates, randomized execution ordering, repeat timings, independent oracle comparisons.
