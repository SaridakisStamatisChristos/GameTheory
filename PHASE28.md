# Phase 28 — Extreme-scale numerical certificate repair

- Center payoffs using `numpy.longdouble` before subtracting extremes.
- Normalize to unit payoff range before calculating best-response gains.
- Compare relative gains in normalized coordinates, avoiding overflow in tolerance calculation.
- Retain original-unit diagnostics (which can be `Infinity` when outside float64 range). Consumers requiring strict JSON should sanitize these diagnostics before serialization.
- Reject nonfinite tolerance inputs.
- Frontend untouched.

## Verification
`python -m pytest -q -W error`: **74 passed**, no warnings.

## Remaining risks
The Lemke–Howson solver's `expected_payoffs` are still computed using float64 products on original payoffs and may overflow for extreme values. Pivot conditioning and exact degeneracy handling remain separate research tasks. No SOTA claim.
