# Phase 30 — numerical diagnostic repair and adversarial regression

Baseline: v29 archive. Frontend unchanged.

## Repair
The independent Lemke–Howson solver's rejection diagnostic used `gap={gap:.6g}` even though the JSON-safe certificate intentionally returns a string for numbers exceeding float64. Replaced with `gap={gap}` so failed certificates are reported as solver failures instead of raising `ValueError`/`TypeError` in formatting.

## Verification
`python -m pytest -q -W error` from backend: 80 passed in 15.22s. Added tests for extreme Nash gaps, JSON-safe independent solver results, and affine translation invariance. Existing tests: 77 passed before additions.

## Scope and limitations
No performance benchmark was run in this phase. The tests are not a formal proof of numerical stability or completeness. In particular, lexicographic pivot degeneracy, platform-specific longdouble precision, full HTTP API serialization and broad independent-oracle differential campaigns remain open.
