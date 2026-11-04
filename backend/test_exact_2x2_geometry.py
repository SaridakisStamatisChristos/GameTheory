"""Independent exact-rational verification of the full 2x2 Nash set."""
from fractions import Fraction as F
from itertools import product
import numpy as np
from fastapi.testclient import TestClient
from main import app
from analytic_2x2 import solve_2x2
from exact_2x2_geometry import complete_nash_geometry


def _oracle(a, b, x, y):
    """Direct payoffs: no implementation formulas or solver calls."""
    row0 = a[0][0] * y + a[0][1] * (1-y)
    row1 = a[1][0] * y + a[1][1] * (1-y)
    col0 = b[0][0] * x + b[1][0] * (1-x)
    col1 = b[0][1] * x + b[1][1] * (1-x)
    return ((x == 0 or row0 >= row1) and (x == 1 or row1 >= row0)
            and (y == 0 or col0 >= col1) and (y == 1 or col1 >= col0))


def _intervals(component):
    return tuple(tuple(F(q) for q in component[key]) for key in
                 ('row_probability_bounds_exact', 'col_probability_bounds_exact'))


def _contains(component, x, y):
    (xl, xh), (yl, yh) = _intervals(component)
    return xl <= x <= xh and yl <= y <= yh


def test_all_6561_ternary_payoff_games_match_independent_oracle():
    grid = [F(0), F(1,4), F(1,2), F(3,4), F(1)]
    for v in product((-1, 0, 1), repeat=8):
        a = ((F(v[0]), F(v[1])), (F(v[2]), F(v[3])))
        b = ((F(v[4]), F(v[5])), (F(v[6]), F(v[7])))
        parts = complete_nash_geometry(a, b)
        assert parts, (a,b)
        for part in parts:
            (xl, xh), (yl, yh) = _intervals(part)
            assert part['dimension'] == int(xl < xh) + int(yl < yh)
            for x, y in product((xl,(xl+xh)/2,xh), (yl,(yl+yh)/2,yh)):
                assert _oracle(a, b, x, y), (a,b,part,x,y)
        for x, y in product(grid, repeat=2):
            assert any(_contains(part,x,y) for part in parts) == _oracle(a,b,x,y), (a,b,x,y)


def test_entire_square_for_fully_indifferent_game():
    zero = ((F(0),F(0)),(F(0),F(0)))
    parts = complete_nash_geometry(zero,zero)
    assert len(parts) == 1
    assert _intervals(parts[0]) == ((F(0),F(1)),(F(0),F(1)))
    assert parts[0]['dimension'] == 2


def test_full_segment_for_strictly_dominant_row_and_indifferent_column():
    a = ((F(1),F(1)),(F(0),F(0)))
    b = ((F(0),F(0)),(F(0),F(0)))
    parts = complete_nash_geometry(a,b)
    assert len(parts) == 1
    assert _intervals(parts[0]) == ((F(1),F(1)),(F(0),F(1)))
    assert parts[0]['dimension'] == 1


def test_three_exact_points_for_coordination_game():
    a = ((F(2),F(0)),(F(0),F(1)))
    parts = complete_nash_geometry(a,a)
    assert len(parts) == 3
    assert all(q['dimension']==0 for q in parts)
    assert {( _intervals(q)[0][0], _intervals(q)[1][0]) for q in parts} == {
        (F(0),F(0)),(F(1),F(1)),(F(1,3),F(1,3))}


def test_vertical_and_horizontal_indifference_components():
    a = ((F(0),F(0)),(F(0),F(0)))
    b = ((F(1),F(0)),(F(0),F(1)))
    parts = complete_nash_geometry(a,b)
    assert any(_intervals(c)==((F(1,2),F(1,2)),(F(0),F(1))) for c in parts)
    for x,y in product([F(0),F(1,4),F(1,2),F(3,4),F(1)],repeat=2):
        assert any(_contains(c,x,y) for c in parts)==_oracle(a,b,x,y)


def test_extremely_small_interior_probability_is_not_rounded_away():
    a = ((F(1),F(0)),(F(0),F('1e-12')))
    b = ((F(2),F(0)),(F(0),F(1)))
    parts = complete_nash_geometry(a,b)
    assert any(_intervals(c)==((F(1,3),F(1,3)),
               (F(1,1000000000001),F(1,1000000000001))) for c in parts)


def test_positive_affine_payoff_invariance():
    for v in list(product((-1,0,1),repeat=8))[::41]:
        a = ((F(v[0]),F(v[1])),(F(v[2]),F(v[3])))
        b = ((F(v[4]),F(v[5])),(F(v[6]),F(v[7])))
        shifted_a = tuple(tuple(3*q + 11 for q in row) for row in a)
        shifted_b = tuple(tuple(7*q - 9 for q in row) for row in b)
        assert complete_nash_geometry(a,b) == complete_nash_geometry(shifted_a,shifted_b)


def test_analytic_endpoint_preserves_representatives_and_adds_geometry():
    with TestClient(app) as client:
        response = client.post('/api/equilibria/analytic-2x2',json={
            'row_payoffs':[[2,0],[0,1]],
            'col_payoffs':[[2,0],[0,1]]})
    assert response.status_code == 200
    result = response.json()
    assert result['geometry_complete'] is True
    assert result['complete'] is False  # legacy representative list semantics
    assert len(result['equilibria']) == 3
    assert len(result['equilibrium_components']) == 3
    assert any(c['row_probability_bounds_exact'] == ['1/3','1/3']
               and c['col_probability_bounds_exact']==['1/3','1/3']
               for c in result['equilibrium_components'])


def test_large_finite_payoffs_preserve_geometry():
    a = np.array([[1e308,0],[0,1e308]])
    result = solve_2x2(a,a)
    assert result['geometry_complete'] is True
    assert len(result['equilibrium_components']) == 3
    assert any(c['row_probability_bounds_exact']==['1/2','1/2']
               and c['col_probability_bounds_exact']==['1/2','1/2']
               for c in result['equilibrium_components'])
