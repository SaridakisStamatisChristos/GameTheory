# Phase 32 — Adversarial HTTP contracts and correlated-equilibrium scaling

- Baseline: v31 archive, 83 passing tests.
- Added deterministic malformed matrix and seeded 2x2 API fuzz cases across seven endpoints.
- Discovered HTTP 500 for a finite 1e300 payoff correlated-equilibrium instance.
- Fixed correlated equilibrium LP by using a common affine normalization for both payoff matrices, maintaining relative welfare weights and obedience constraints.
- Added normalized obedience violation verification and JSON-safe expected payoff diagnostics.
- Added nonstandard NaN JSON literal rejection tests and an extreme-payoff equilibrium regression.
- Run: `cd backend && python -m pytest -q -W error`.
- Caveats: not a SOTA claim; finite randomized samples do not prove universal numerical correctness. Frontend intentionally unchanged.
