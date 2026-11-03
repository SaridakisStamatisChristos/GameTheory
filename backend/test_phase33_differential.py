"""Independent solver differential and affine-invariance regression tests."""
import numpy as np
from independent_lh import solve_independent_lh
from numerical_certificate import certify_bimatrix


def test_random_games_have_certified_solutions():
    rng = np.random.default_rng(3301)
    for size in (2, 3, 4, 5):
        for _ in range(15):
            a = rng.integers(-9, 10, size=(size, size)).astype(float)
            b = rng.integers(-9, 10, size=(size, size)).astype(float)
            result = solve_independent_lh(a, b, max_labels=2 * size)
            assert result['equilibria'], (size, result['failures'])
            for e in result['equilibria']:
                assert certify_bimatrix(a, b, e['row_probabilities'], e['col_probabilities'])['valid']


def test_large_affine_transform_preserves_certified_equilibrium():
    a = np.array([[3., 0.], [1., 2.]])
    b = np.array([[1., 4.], [3., 0.]])
    baseline = solve_independent_lh(a, b)
    transformed = solve_independent_lh(1e120 * a + 1e120, 1e-100 * b - 1e-100)
    assert baseline['equilibria'] and transformed['equilibria']
    for e in transformed['equilibria']:
        assert e['certificate']['valid']
