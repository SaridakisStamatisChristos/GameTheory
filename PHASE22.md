# Phase 22 — independent LH pivot optimization

Changes: pivot selection now solves a single multi-RHS linear system rather than computing the inverse explicitly. Baseline preserved in `backend/independent_lh_v21_baseline.py`. The frontend was not modified.

Verification: `cd backend && python -m pytest -q` produced 62 passed, 3 warnings. Previous conversation reported 64 for v21; discrepancy has not been resolved and is not claimed as parity.

Benchmark: `cd backend && python benchmark_pivot_v22.py` writes `benchmark_pivot_v22.json`. Five seeded random integer games per matrix size, three sequential timing repetitions, first three labels only, no warmup. These are exploratory microbenchmarks, not SOTA evidence. See JSON for raw records. Additional tests of numerical conditioning and performance are needed.
