import json
import numpy as np
from safe_numeric import expected_payoffs
from numerical_certificate import certify_bimatrix
from independent_lh import solve_independent_lh


def test_extreme_expected_payoffs_json_safe():
    a = np.array([[1e308, -1e308], [-1e308, 1e308]])
    result = expected_payoffs(a, -a, [1., 0.], [1., 0.])
    assert result == [1e308, -1e308]
    assert 'Infinity' not in json.dumps(result, allow_nan=False)


def test_extreme_certificate_json_safe():
    a = np.array([[1e308, -1e308], [-1e308, 1e308]])
    c = certify_bimatrix(a, -a, [1., 0.], [1., 0.])
    json.dumps(c, allow_nan=False)
    assert not c['valid']
    assert isinstance(c['col_deviation_gain'], str)


def test_solver_extreme_json_safe():
    a = np.array([[1e308, -1e308], [-1e308, 1e308]])
    r = solve_independent_lh(a, -a)
    json.dumps(r, allow_nan=False)
    assert r['equilibria']
