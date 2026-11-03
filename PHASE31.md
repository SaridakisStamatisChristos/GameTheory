# Phase 31 — HTTP Contract Audit

Baseline: v30 ZIP, 80 tests passing.

## Fixed
- Independent Lemke–Howson endpoint incorrectly accessed `req.payoff_a` and `req.payoff_b`; the actual Pydantic schema provides `row_payoffs` and `col_payoffs`. Replaced direct access with `validate(req)`.
- Zero-sum endpoint now converts ragged/nonrectangular matrix conversion errors into HTTP 422 rather than uncaught exceptions.

## Evidence
`cd backend && python -m pytest -q -W error` -> 83 passed in 14.51s, no warnings.
Three new HTTP regression tests in `backend/test_http_contract_phase31.py`.

## Remaining
No general guarantee of numerical robustness or SOTA performance; broader HTTP fuzzing, cross-engine consistency checks, and reproducible performance experiments are still needed. Frontend unchanged.
