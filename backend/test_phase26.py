import numpy as np
from independent_lh import solve_independent_lh
from numerical_certificate import certify_bimatrix


def test_certificates_integrated():
    a = np.array([[3., 0.], [0., 2.]])
    b = np.array([[2., 0.], [0., 3.]])
    result = solve_independent_lh(a, b)
    assert result['equilibria']
    for eq in result['equilibria']:
        assert eq['certificate']['valid']
        assert eq['certificate']['nash_gap'] == eq['nash_gap']
        assert certify_bimatrix(a, b, eq['row_probabilities'], eq['col_probabilities'])['valid']


def test_positive_affine_invariance_seeded():
    rng = np.random.default_rng(2601)
    for n in (2, 3, 4):
        for _ in range(8):
            a = rng.integers(-8, 9, (n, n)).astype(float)
            b = rng.integers(-8, 9, (n, n)).astype(float)
            base = solve_independent_lh(a, b, max_labels=2)
            shifted = solve_independent_lh(3*a+1e5, 7*b-1e5, max_labels=2)
            assert bool(base['equilibria']) == bool(shifted['equilibria'])
            for res, aa, bb in ((base, a, b), (shifted, 3*a+1e5, 7*b-1e5)):
                for eq in res['equilibria']:
                    assert certify_bimatrix(aa, bb, eq['row_probabilities'], eq['col_probabilities'])['valid']
