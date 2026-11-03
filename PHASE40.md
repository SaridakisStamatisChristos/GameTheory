# Phase 40 — Failure-path and branch-coverage expansion

## Purpose and scope

This changeset strengthens **Game Theory Lab as a trusted, local research application**, not as a multi-tenant service. No new login, tenant or deployment architecture is introduced. The only production-code change is correcting vanilla CFR's player-ID validation to reject Python booleans, which otherwise compare equal to integers.

## New evidence and contracts

- Reproducible, permanent negative tests for ASGI request framing, partial bodies and disconnects, request-source exhaustion, rate budgets, remote bearer gates and compute-slot release even on exceptions.
- Invalid extensive-form trees (missing children, DAG aliases, cycles, impossible probabilities, inconsistent information sets, and deep recursion), zero-probability Bayesian types, perfect-information tree failures, repeated-game action/trace limits.
- Fault injection at the LP and Lemke–Howson boundaries to prove invalid probabilities, invalid best responses, numerical certificates, pivot budgets and corrupt pivot state are rejected rather than reported as equilibria.
- End-to-end API error translation, real isolated SQLite quota/roundtrip tests, independent best-response information-set dependency ordering, and near-degenerate mixed-strategy tolerance behavior.

## Coverage policy

`backend/.coveragerc` enables **statement and branch instrumentation** for the 18 core backend Python modules. Test modules, standalone benchmark scripts, historical baseline code, and the benchmark driver are excluded from the denominator. The exclusions are documented rather than hidden.

From `backend`:

```bash
python -m pytest -q -W error --cov=. --cov-config=.coveragerc --cov-branch --cov-report=term-missing --cov-report=json:coverage-audit.json --cov-report=xml:coverage-audit.xml --cov-fail-under=99
```

The CI workflow enforces **at least 99% combined statement/branch coverage** and uploads machine-readable reports. It does not enforce 100% on every future edit; it guards against material regression while allowing additional code to be tested incrementally.

## Local evidence prior to pushing

Python 3.13.5 and the local v38-compatible dependencies: **251 passed**, **1,173/1,173 executable statements covered**, **458/458 branch opportunities covered**, **100.00% combined coverage** with the scoped configuration. The full backend + frontend upgrade CI on the source commit must run separately; these are not interchangeable pieces of evidence.

## Important limitations

- 100% coverage demonstrates that defined control-flow paths executed; it does not prove exhaustiveness of numerical input values, global Nash enumeration, solver convergence, formal correctness, UI/browser behavior, or code performance relative to published algorithms.
- Dependency-specific warnings in the old local FastAPI/Starlette environment required a local warning-ignore switch. The GitHub CI has the upgraded dependencies and retains **all warnings as errors**, without exceptions.
- Deliberately invalid/mock responses are used only to exercise verification boundaries; no coverage threshold is met by disabling tests or catching every exception indiscriminately.
