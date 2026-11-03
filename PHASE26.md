# Phase 26 — Integrated pivot certification and numerical normalization

## Changes
- Independent Lemke–Howson now calls `certify_bimatrix` for every accepted equilibrium, and attaches the certificate to each result.
- Positive payoff coefficients normalized to [1,2] for nonconstant matrices, retaining best responses under positive affine transformations.
- Added tests for certificate integration and seeded positive affine payoff transformations.
- Added paired, deterministic old-vs-new benchmark with alternating execution order, warmups and 3 measured repetitions per case.

## Verification
`python -m pytest -q -W error`: **69 passed**. No warning output.

## Paired exploratory benchmark
Six games per dimension, 3 measured repetitions per game and solver; timings in milliseconds.

| Size | Baseline median | New median | Valid old/new |
|---|---:|---:|---:|
| 2x2 | 0.3864 | 0.4009 | 6/6, 6/6 |
| 3x3 | 0.3992 | 0.4661 | 6/6, 6/6 |
| 5x5 | 0.6388 | 0.6067 | 6/6, 6/6 |
| 8x8 | 1.1640 | 0.9498 | 6/6, 6/6 |

Results do not show universal speedups. The independent numerical certificate improves reporting and rejection of invalid outputs, but floating-point tolerance does not constitute a formal proof. No exhaustive equilibrium enumeration guarantee. Numerical stability at extreme dynamic ranges remains open. Benchmarks are small and not SOTA evidence.
