# Phase 39 — Framework dependency modernization

## Rationale

The previous CI repair suppressed one Starlette/AnyIO deprecation. Starlette 1.7.0 (2026-09-23) fixes the actual import, and FastAPI 0.159.1 supports Starlette 1.x. This revision updates the framework instead of retaining a long-term warning filter.

## Changes

- Upgrade FastAPI 0.128.2 -> 0.159.1 and Starlette 0.50.0 -> 1.7.0.
- Keep AnyIO 4.16.0 and install HTTPX2 2.16.0 for Starlette TestClient; retain HTTPX 0.28.1 for independent legacy usages.
- Remove the pytest warning exception; both collection and test execution run with all warnings as errors.
- Preserve game-theoretic algorithm code and API surface unchanged.
- Capture dependency resolutions through `pip freeze` and use `pip check` for consistency.

## Verification

GitHub Actions #7 passed on upgrade candidate `83ac01b`: **143/143 backend tests passed** with `python -m pytest -q -W error` and no warning filter, `pip check` passed, and the Vite frontend production build passed. The full 33-package version snapshot is now committed for reproducibility. The revised snapshot commit must pass CI independently before merging to `main`.

## Sources

- https://starlette.dev/release-notes/ (Starlette 1.7.0; AnyIO imports #3498)
- https://fastapi.tiangolo.com/release-notes/ (FastAPI 0.159.1)
- https://starlette.dev/testclient/ (HTTPX2 requirement)
