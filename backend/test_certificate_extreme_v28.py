import numpy as np
import pytest
from numerical_certificate import certify_bimatrix


def test_opposite_extreme_payoffs_do_not_overflow_certificate():
    a = np.array([[1e308, -1e308], [-1e308, 1e308]])
    b = -a
    cert = certify_bimatrix(a, b, [0.5, 0.5], [0.5, 0.5])
    assert cert['valid']
    assert cert['nash_gap'] == 0
    assert np.longdouble(cert['row_payoff_range']) > np.finfo(float).max  # JSON-safe scientific notation


def test_extreme_non_equilibrium_rejected():
    a = np.array([[1e308, -1e308], [-1e308, 1e308]])
    cert = certify_bimatrix(a, -a, [1., 0.], [1., 0.])
    assert not cert['valid']


def test_nonfinite_tolerance_rejected():
    with pytest.raises(ValueError):
        certify_bimatrix([[1.]], [[1.]], [1.], [1.], relative_tolerance=float('inf'))
