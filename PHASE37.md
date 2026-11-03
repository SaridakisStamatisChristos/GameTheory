# Phase 37 — CFR+ validation decoupling

Extracted `validate_cfr_game` from the reference CFR+ solver. Both reference and compiled implementations now share the same validation logic, while the compiled implementation no longer executes an unnecessary training iteration merely to validate the game.

Verification: `python -m pytest -q -W error` — 117 passed. See `backend/benchmark_phase37_results.json` for local timings. No SOTA claim.

The frontend is unchanged. Next: independent CFR+ comparison, larger imperfect-information game fixtures, and stronger performance profiling.
