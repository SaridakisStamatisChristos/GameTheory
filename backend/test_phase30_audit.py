"""Regression checks for numerical certificate diagnostics and API-safe output."""
import json
import numpy as np
from numerical_certificate import certify_bimatrix
from independent_lh import solve_independent_lh


def test_extreme_gap_certificate_is_json_safe():
    a = np.array([[1e300, -1e300], [-1e300, 1e300]])
    b = -a
    cert = certify_bimatrix(a, b, [1, 0], [1, 0])
    assert not cert['valid']
    assert float(cert['nash_gap']) > 1e299
    json.dumps(cert, allow_nan=False)


def test_independent_solver_results_json_safe_extreme():
    a = np.array([[1e300, -1e300], [-1e300, 1e300]])
    result = solve_independent_lh(a, -a)
    assert result['equilibria']
    assert all(e['certificate']['valid'] for e in result['equilibria'])
    json.dumps(result, allow_nan=False)


def test_certificate_affine_translation_invariance():
    a = np.array([[3., 0.], [0., 2.]])
    b = np.array([[2., 0.], [0., 3.]])
    x = [0.6, 0.4]
    y = [0.4, 0.6]
    baseline = certify_bimatrix(a, b, x, y)
    translated = certify_bimatrix(a + 1e8, b - 1e8, x, y)
    assert baseline['valid'] == translated['valid']
    assert abs(float(baseline['nash_gap']) - float(translated['nash_gap'])) < 1e-6
