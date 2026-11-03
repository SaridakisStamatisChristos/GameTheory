import numpy as np
import pytest
from hybrid_nash import solve_hybrid


def test_fast_matching_pennies():
    a=np.array([[1.,-1.],[-1.,1.]])
    result=solve_hybrid(a,-a)
    assert result['method']=='fast_support'
    assert result['equilibria'][0]['nash_gap']<1e-9
    assert not result['complete_support_coverage']


def test_degenerate_fallback():
    a=np.zeros((2,3))
    result=solve_hybrid(a,a)
    assert result['method']=='support_lp'
    assert result['complete_support_coverage']
    assert result['equilibria']


def test_exhaustive_forces_lp():
    a=np.array([[1.,-1.],[-1.,1.]])
    result=solve_hybrid(a,-a,exhaustive=True)
    assert result['method']=='support_lp'


def test_budget_not_false_nonexistence():
    a=np.zeros((8,8))
    result=solve_hybrid(a,a,max_pairs=100)
    assert not result['complete_support_coverage']
    assert result['method']=='fast_support_budget_limited'
    with pytest.raises(ValueError,match='exceed'):
        solve_hybrid(a,a,max_pairs=100,exhaustive=True)


def test_validation():
    with pytest.raises(ValueError):
        solve_hybrid([[float('nan')]],[[0.]])
