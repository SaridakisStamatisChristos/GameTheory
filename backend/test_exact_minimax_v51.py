"""Independent minimax references and fault injection for bounded exact zero-sum LP."""
from fractions import Fraction as F
from itertools import product
import numpy as np
import pytest
from scipy.optimize import linprog
import exact_minimax
from exact_minimax import solve_exact_zero_sum as solve
from exact_support import ExactWorkLimit


def _check_original_saddle(a, result):
    matrix = np.asarray(a, dtype=float)
    m, n = matrix.shape
    af = [[F(str(v)) for v in row] for row in matrix]
    x = list(map(F, result['row_strategy_exact']))
    y = list(map(F, result['col_strategy_exact']))
    value = F(result['value_exact'])
    assert len(x) == m and len(y) == n
    assert sum(x) == sum(y) == F(1)
    assert min((*x, *y)) >= 0
    security = min(sum(x[i]*af[i][j] for i in range(m)) for j in range(n))
    ceiling = max(sum(af[i][j]*y[j] for j in range(n)) for i in range(m))
    assert security == ceiling == value
    assert result['row_security_value_exact'] == str(value)
    assert result['col_security_value_exact'] == str(value)
    assert result['minimax_duality_gap_exact'] == '0'
    assert result['status'] == 'certified_exact_minimax'
    assert result['all_equilibria_enumerated'] is False
    return value


def _scipy_value(a):
    a = np.asarray(a, dtype=float)
    m, n = a.shape
    result = linprog(np.r_[np.zeros(m), -1.],
                     A_ub=np.c_[-a.T, np.ones(n)], b_ub=np.zeros(n),
                     A_eq=np.r_[np.ones(m), 0.][None, :], b_eq=[1],
                     bounds=[(0, None)]*m + [(None, None)], method='highs')
    assert result.success, result.message
    return result.x[-1]


def test_canonical_games_and_extreme_float64_scale():
    games = (
        [[1, -1], [-1, 1]],
        [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
        [[0, 0], [0, 0]],
        [[-7]],
        [[-5, 7, 3]],
        [[3], [1], [5]],
        [[-5, 10], [0, -1]],
        [[1e308, -1e308], [-1e308, 1e308]],
        [[1e-300, 0], [0, 2e-300]],
    )
    for a in games:
        result = solve(a)
        _check_original_saddle(a, result)
        assert result['exact_operations'] > 0
        if np.max(np.abs(a)) < 1e12:
            assert abs(float(F(result['value_exact'])) - _scipy_value(a)) < 1e-8


def test_exhaustive_tiny_integer_games_and_seeded_independent_highs():
    for entries in product((-1, 0, 1), repeat=4):
        game = np.asarray(entries).reshape(2, 2)
        value = _check_original_saddle(game, solve(game))
        assert abs(float(value) - _scipy_value(game)) < 1e-8
    rng = np.random.default_rng(51042)
    for shape in ((2, 3), (3, 2), (3, 3), (4, 4), (6, 6), (8, 8)):
        for _ in range(8):
            game = rng.integers(-9, 10, size=shape)
            value = _check_original_saddle(game, solve(game))
            assert abs(float(value) - _scipy_value(game)) < 1e-8


def test_invalid_model_and_bounded_failures():
    for invalid in ([], [[]], [[float('inf')]], [[float('nan')]],
                    [[1], [1, 2]], [['abc']]):
        with pytest.raises(ValueError):
            solve(invalid)
    with pytest.raises(ValueError, match='100 action profiles'):
        solve(np.zeros((11, 10)))
    for budget in (True, 0, -10, 2000001, 2.5):
        with pytest.raises(ValueError, match='work budget'):
            solve([[0]], max_operations=budget)
    with pytest.raises(ExactWorkLimit):
        solve([[1, -1], [-1, 1]], max_operations=1)


def test_malicious_simplex_results_rejected(monkeypatch):
    class Broken:
        data = None
        counter = 0
        def __init__(self, *_args):
            pass
        def solve(self):
            out = self.data[self.counter % 2]
            self.__class__.counter += 1
            return out
    normal = exact_minimax._RationalSimplex
    try:
        for left, right, message in (
            (None, [F(1), F(1)], 'infeasible'),
            ([F(1), F(1)], None, 'infeasible'),
            ([F(1)], [F(1), F(1)], 'invalid scaled'),
            ([F(-1), F(2)], [F(1), F(1)], 'invalid scaled'),
            ([F(0), F(0)], [F(0), F(0)], 'primal-dual'),
            ([F(1), F(0)], [F(1), F(1)], 'primal-dual'),
            ([F(1), F(0)], [F(1), F(0)], 'original-payoff Nash'),
        ):
            Broken.data = (left, right)
            Broken.counter = 0
            monkeypatch.setattr(exact_minimax, '_RationalSimplex', Broken)
            with pytest.raises(AssertionError, match=message):
                solve([[1, -1], [-1, 1]])
    finally:
        monkeypatch.setattr(exact_minimax, '_RationalSimplex', normal)
