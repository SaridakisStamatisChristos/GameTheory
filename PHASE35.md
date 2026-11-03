# Phase 35 — Kuhn Poker CFR+ validation

- Added `backend/benchmark_kuhn_phase35.py`: 55-node Kuhn poker game generator and reproducible CFR+ convergence benchmark.
- Added `backend/test_phase35_kuhn.py`: theoretical value and exploitability regression, normalization, benchmark schema.
- Verified: `python -m pytest -q -W error` → **112 passed in 16.77 s**.
- Benchmark: 3 repetitions each, 100/500/2000 iterations. See `backend/benchmark_kuhn_phase35_results.json` for measured latencies, exploitability and value errors.
- Kuhn poker theoretical row value: -1/18. At 2000 iterations observed -0.0555556447, exploitability 0.000087297.
- Frontend unchanged. No independent SOTA claim; larger poker games, independent CFR implementations and profiling remain open.
