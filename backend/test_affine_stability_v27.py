import numpy as np
from independent_lh import solve_independent_lh


def test_extreme_finite_scale_does_not_overflow():
    a = np.array([[1e300, -1e300], [-1e300, 1e300]])
    b = -a
    result = solve_independent_lh(a, b, max_labels=4)
    assert result['equilibria'], result['failures']
    eq = result['equilibria'][0]
    assert np.allclose(eq['row_probabilities'], [.5,.5], atol=1e-7)
    assert np.allclose(eq['col_probabilities'], [.5,.5], atol=1e-7)


def test_translation_invariance_large_offset():
    a = np.array([[4., 0.], [0., 2.]])
    b = np.array([[3., 0.], [0., 5.]])
    original = solve_independent_lh(a, b)
    shifted = solve_independent_lh(a + 1e9, b - 1e9)
    assert original['equilibria'] and shifted['equilibria']
    assert any(np.allclose(x['row_probabilities'], y['row_probabilities'], atol=1e-6) and np.allclose(x['col_probabilities'], y['col_probabilities'], atol=1e-6) for x in original['equilibria'] for y in shifted['equilibria'])
