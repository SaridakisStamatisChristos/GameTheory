# v55 — Exact feasibility scaling and differential performance

The preexisting `feasible_strict_strategy(..., fast_path=True)` probes an exact
uniform witness, rejects infeasible singleton supports without a simplex call,
and falls back to the original rational simplex when needed.

This release **does not invent a second optimization**. It introduces
paired, seeded, reproducible comparisons against the same public solver
with `feasibility_fast_path=False`. Cases include all ties, strictly
dominated and random integer games, from 2x2 through 8x8. All use the
same deterministic support order, max 96 pairs, max support 3, and
2,000,000 charged rational operations per run.

The independent differential contract checks support-cell identities,
complete/partial status, checked pair count, incomplete reason, and
repeatability. Witnesses need not be identical inside continuous
equilibrium families. The CI artifact records per-case raw time
samples, medians, exact charged operations, LP counts, uniform
witness hits and singleton rejections, seed, and fixture SHA-256.

The operation counter is **not** a bit-complexity, CPU-cycle, or peak
memory measurement. Timings are machine-dependent and this is not a
global speedup/SOTA claim. Next research: peak memory, Fraction
numerator/denominator bit lengths, robust randomized order, larger
workloads and cost-normalized external reference comparisons.
