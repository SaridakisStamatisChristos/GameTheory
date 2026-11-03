# Phase 34 — CFR+ correctness and convergence campaign

- Frontend unchanged.
- Added deterministic convergence/latency benchmark for Matching Pennies.
- Added regression tests for cross-player infoset identifier collisions, strategy normalization, and benchmark reproducibility.
- Added defensive HTTP 422 handling for excessively deep game trees during inspection.
- Verification: `cd backend && python -m pytest -q -W error`: 110 passed, zero warnings.
- Benchmark: `cd backend && python benchmark_cfr_phase34.py --repeats 3`.
- Data: `backend/benchmark_cfr_phase34_results.json`.

Caveats: This is one small perfect-recall game, not an independent CFR+ implementation comparison. No general SOTA claim follows. The Python recursive traversal can still be limited by depth during iterative solving, and the benchmark is not a statistically powered cross-platform study.
