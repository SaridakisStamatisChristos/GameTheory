"""Exact correlated-equilibrium checks against an independent numerical LP and HTTP contracts."""
from fractions import Fraction as F
import numpy as np
import pytest
from scipy.optimize import linprog
import exact_correlated
from exact_correlated import solve_exact_correlated_equilibrium as solve
from exact_support import ExactWorkLimit
from fastapi.testclient import TestClient
from main import app


def check_obedience(a, b, answer):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    m, n = a.shape
    q = [[F(v) for v in row] for row in answer['joint_distribution_exact']]
    assert sum(map(sum, q)) == 1
    assert all(p >= 0 for row in q for p in row)
    gains = ([sum(q[i][j] * (a[k, j]-a[i, j]) for j in range(n))
              for i in range(m) for k in range(m) if i != k] +
             [sum(q[i][j] * (b[i, k]-b[i, j]) for i in range(m))
              for j in range(n) for k in range(n) if j != k])
    assert all(g <= 0 for g in gains)
    assert answer['obedience_inequalities_checked'] == len(gains)
    expected = [sum(q[i][j]*payoff[i, j] for i in range(m) for j in range(n))
                for payoff in (a, b)]
    assert [F(v) for v in answer['expected_payoffs_exact']] == expected
    assert answer['maximum_obedience_violation_exact'] == '0'
    assert not answer['complete_equilibrium_set']
    return q


def test_canonical_and_rectangular_games():
    games = [
        ([[1, -1], [-1, 1]], [[-1, 1], [1, -1]]),
        ([[2, 0], [0, 1]], [[1, 0], [0, 2]]),
        ([[-1, -3], [0, -2]], [[-1, 0], [-3, -2]]),
        ([[0]*3]*3, [[0]*3]*3),
        ([[0]], [[4]]),
        ([[1, 3]], [[0, 2]]),
        ([[1], [3]], [[2], [0]]),
        ([[1, 2, 0], [3, -1, 4]], [[-1, 2, 2], [1, 3, 0]]),
    ]
    for a, b in games:
        for objective in ('welfare', 'feasible'):
            out = solve(a, b, objective=objective)
            check_obedience(a, b, out)
            assert out['objective'] == objective
            assert out['exact_operations'] > 0
            assert out['objective_value_exact'] == (
                str(sum(map(F, out['expected_payoffs_exact'])))
                if objective == 'welfare' else '0')


def test_seeded_differential_optimality_against_independent_highs():
    rng = np.random.default_rng(50023)
    for shape in ((2, 2), (2, 3), (3, 2), (3, 3), (4, 4)):
        m, n = shape
        for _ in range(8):
            a = rng.integers(-6, 7, size=shape)
            b = rng.integers(-6, 7, size=shape)
            answer = solve(a, b, max_operations=1000000)
            check_obedience(a, b, answer)
            inequalities = []
            for i in range(m):
                for k in range(m):
                    if i == k:
                        continue
                    row = np.zeros(shape)
                    row[i, :] = a[k, :] - a[i, :]
                    inequalities.append(row.ravel())
            for j in range(n):
                for k in range(n):
                    if j == k:
                        continue
                    row = np.zeros(shape)
                    row[:, j] = b[:, k] - b[:, j]
                    inequalities.append(row.ravel())
            reference = linprog(-(a+b).ravel().astype(float),
                                A_ub=np.asarray(inequalities) if inequalities else None,
                                b_ub=np.zeros(len(inequalities)) if inequalities else None,
                                A_eq=[np.ones(m*n)], b_eq=[1],
                                bounds=(0, None), method='highs')
            assert reference.success
            assert abs(float(F(answer['objective_value_exact']))+reference.fun) < 1e-8


def test_input_validation_and_budget_fail_closed():
    bad = [([], []), ([[0]], [[0, 1]]), ([[float('nan')]], [[0]]),
           ([[float('inf')]], [[0]]), ([[0], [0, 1]], [[0]]),
           ([['invalid']], [[0]])]
    for a, b in bad:
        with pytest.raises(ValueError):
            solve(a, b)
    with pytest.raises(ValueError, match='36 action profiles'):
        solve(np.zeros((7, 6)), np.zeros((7, 6)))
    with pytest.raises(ValueError, match='objective'):
        solve([[0]], [[0]], objective='unknown')
    for budget in (True, 0, -3, 2000001, 1.5):
        with pytest.raises(ValueError, match='budget'):
            solve([[0]], [[0]], max_operations=budget)
    with pytest.raises(ExactWorkLimit):
        solve([[1, -1], [-1, 1]], [[-1, 1], [1, -1]], max_operations=1)


def test_adversarial_fault_injection_detects_invalid_lp_output(monkeypatch):
    base = exact_correlated._RationalSimplex
    class Broken:
        def __init__(self, *_args):
            pass
        def solve(self):
            return self.distribution
    try:
        for distribution, message in ((None, 'no correlated equilibrium'),
                                      ([F(0)]*4, 'probability-simplex'),
                                      ([F(-1), F(2), F(0), F(0)], 'probability-simplex'),
                                      ([F(1), F(0), F(0), F(0)], 'obedience')):
            Broken.distribution = distribution
            monkeypatch.setattr(exact_correlated, '_RationalSimplex', Broken)
            with pytest.raises(AssertionError, match=message):
                solve([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    finally:
        monkeypatch.setattr(exact_correlated, '_RationalSimplex', base)


def test_http_exact_correlated_contract_and_fail_closed():
    base = {'row_payoffs': [[1, -1], [-1, 1]],
            'col_payoffs': [[-1, 1], [1, -1]]}
    with TestClient(app, raise_server_exceptions=False) as client:
        result = client.post('/api/correlated-equilibrium/exact', json=base)
        assert result.status_code == 200, result.text
        body = result.json()
        assert body['joint_distribution_exact'] == [['1/4', '1/4'], ['1/4', '1/4']]
        assert body['maximum_obedience_violation_exact'] == '0'
        exhausted = client.post('/api/correlated-equilibrium/exact', json={
            **base, 'max_operations': 1})
        assert exhausted.status_code == 422, exhausted.text
        assert 'budget' in exhausted.text.lower()
        large = client.post('/api/correlated-equilibrium/exact', json={
            'row_payoffs': [[0]*6 for _ in range(7)],
            'col_payoffs': [[0]*6 for _ in range(7)]})
        assert large.status_code == 422, large.text
        assert '36 action profiles' in large.text
        bad_objective = client.post('/api/correlated-equilibrium/exact', json={
            **base, 'objective': 'not_an_objective'})
        assert bad_objective.status_code == 422
        assert client.post('/api/correlated-equilibrium', json=base).status_code == 200
