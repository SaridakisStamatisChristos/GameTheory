"""Independent rational best-response checks of exact Nash-family membership."""
from fractions import Fraction
from itertools import product
import numpy as np
import pytest
from exact_nash_set import solve_exact_nash_set
from exact_support import ExactWorkLimit

F = Fraction


def simplex(n, denominator):
    def compositions(total, dim):
        if dim == 1:
            yield (total,)
        else:
            for k in range(total + 1):
                for tail in compositions(total - k, dim - 1):
                    yield (k,) + tail
    for t in compositions(denominator, n):
        yield tuple(F(i, denominator) for i in t)


def is_nash(a, b, x, y):
    m, n = len(a), len(a[0])
    row = [sum(F(a[i][j])*y[j] for j in range(n)) for i in range(m)]
    col = [sum(x[i]*F(b[i][j]) for i in range(m)) for j in range(n)]
    return (sum(x[i]*row[i] for i in range(m)) == max(row)
            and sum(y[j]*col[j] for j in range(n)) == max(col))


def in_polytope(desc, full_strategy):
    active = desc['variable_actions']
    mix = [full_strategy[i] for i in active]
    if any(full_strategy[i] != 0 for i in range(len(full_strategy)) if i not in active):
        return False
    if sum(mix) != 1 or not all(p > 0 for p in mix):
        return False
    for item in desc['equalities_dot_coefficients_eq_zero']:
        if sum(F(c)*p for c,p in zip(item['coefficients_exact'],mix)) != 0:
            return False
    for item in desc['inequalities_dot_coefficients_ge_zero']:
        if sum(F(c)*p for c,p in zip(item['coefficients_exact'],mix)) < 0:
            return False
    return True


def membership(families, x, y):
    return sum(in_polytope(f['row_strategy_polytope'],x)
               and in_polytope(f['col_strategy_polytope'],y)
               for f in families)


def verify_witnesses(a, b, result):
    families=result['equilibrium_families']
    assert result['equilibrium_set_complete'] is True
    assert result['nonempty_support_cells'] == len(families)
    for cell in families:
        rx=cell['row_strategy_polytope']
        cy=cell['col_strategy_polytope']
        x=[F(0)]*len(a)
        y=[F(0)]*len(a[0])
        for i,p in zip(rx['variable_actions'],rx['witness_probabilities_exact']):
            x[i]=F(p)
        for i,p in zip(cy['variable_actions'],cy['witness_probabilities_exact']):
            y[i]=F(p)
        assert is_nash(a,b,x,y)
        assert membership(families,x,y) == 1
        assert [i for i,p in enumerate(x) if p>0]==cell['support']['row']
        assert [i for i,p in enumerate(y) if p>0]==cell['support']['col']


def test_all_zero_three_and_four_action_games():
    for n in (3,4):
        a=b=[[0]*n for _ in range(n)]
        result=solve_exact_nash_set(a,b)
        verify_witnesses(a,b,result)
        assert result['nonempty_support_cells']==(2**n-1)**2
        for x in simplex(n,3):
            for y in simplex(n,3):
                assert membership(result['equilibrium_families'],x,y)==1


def test_all_256_two_valued_two_by_two_games_against_independent_nash_predicate():
    points=list(simplex(2,4))
    for values in product((-1,1), repeat=8):
        a=[values[:2],values[2:4]]
        b=[values[4:6],values[6:]]
        result=solve_exact_nash_set(a,b)
        verify_witnesses(a,b,result)
        for x in points:
            for y in points:
                assert (membership(result['equilibrium_families'],x,y)==1)==is_nash(a,b,x,y)


def test_seeded_three_by_three_games_vs_direct_best_responses():
    rng=np.random.default_rng(4301)
    points=list(simplex(3,3))
    for _ in range(30):
        a=rng.integers(-3,4,size=(3,3)).tolist()
        b=rng.integers(-3,4,size=(3,3)).tolist()
        result=solve_exact_nash_set(a,b)
        verify_witnesses(a,b,result)
        for x in points:
            for y in points:
                assert (membership(result['equilibrium_families'],x,y)==1)==is_nash(a,b,x,y)


def test_random_four_by_four_and_rectangular_games():
    rng=np.random.default_rng(4313)
    for m,n in ((1,3),(3,2),(4,3),(4,4)):
        a=rng.integers(-4,5,size=(m,n)).tolist()
        b=rng.integers(-4,5,size=(m,n)).tolist()
        result=solve_exact_nash_set(a,b)
        verify_witnesses(a,b,result)
        for x in simplex(m,2):
            for y in simplex(n,2):
                assert (membership(result['equilibrium_families'],x,y)==1)==is_nash(a,b,x,y)


def test_continuous_equilibrium_line_and_support_boundaries():
    a=[[0,0,0],[0,0,0],[-1,-1,-1]]
    b=[[3,0,0],[3,0,0],[3,0,0]]
    result=solve_exact_nash_set(a,b)
    assert result['nonempty_support_cells']==3
    for p in (F(0),F(1,4),F(1,2),F(3,4),F(1)):
        assert membership(result['equilibrium_families'],
                          (p,1-p,F(0)), (F(1),F(0),F(0)))==1
    assert membership(result['equilibrium_families'],
                      (F(0),F(0),F(1)), (F(1),F(0),F(0)))==0


def test_fractional_payoff_coefficients_and_exact_witness():
    a=[[F('0.1'),F('0.2')],[F('0.2'),F('0.1')]]
    b=[[F('0.4'),F('0.3')],[F('0.3'),F('0.4')]]
    result=solve_exact_nash_set(a,b)
    assert membership(result['equilibrium_families'],
                      (F(1,2),F(1,2)),(F(1,2),F(1,2)))==1
    verify_witnesses(a,b,result)


def test_budget_exhaustion_and_invalid_data_fail_closed():
    with pytest.raises(ExactWorkLimit):
        solve_exact_nash_set([[0]*4]*4,[[0]*4]*4,max_pairs=100)
    with pytest.raises(ExactWorkLimit):
        solve_exact_nash_set([[0]*3]*3,[[0]*3]*3,max_operations=1)
    with pytest.raises(ValueError):
        solve_exact_nash_set([[0]*5]*5,[[0]*5]*5)
    with pytest.raises(ValueError):
        solve_exact_nash_set([[0,float('nan')]],[[0,0]])
