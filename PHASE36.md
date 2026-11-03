# Phase 36 — Compiled-tree CFR+ optimization

## Changes
- Added `backend/cfr_plus_compiled.py`: indexed tree, indexed information sets, precomputed action positions and chance weights.
- Preserved `backend/cfr_plus.py` as the reference and validation oracle.
- Added five differential and invalid-input tests in `backend/test_phase36_compiled.py`.
- Added `backend/benchmark_phase36.py` and raw `backend/benchmark_phase36_results.json`.

## Evidence
- `python -m pytest -q -W error`: **117 passed**, zero warnings.
- Kuhn Poker benchmark, five measured repetitions after warm-up:
  - 100 iterations: reference 10.709 ms; compiled 9.179 ms.
  - 500 iterations: reference 52.545 ms; compiled 41.935 ms.
  - 2000 iterations: reference 210.985 ms; compiled 174.747 ms.
- Differential tests compare strategies, expected value and exploitability to 1e-10 for 1, 10, 100 and 500 iterations.

## Limitations
- Compiled solver currently validates by invoking one reference CFR+ iteration, including a best-response certificate. This introduces fixed overhead; the benchmark includes it.
- No independent external CFR oracle or large-game scaling evidence yet.
- Frontend untouched. No universal SOTA claim.
