import numpy as np
import pytest
from support_lp import solve_support_lp

def test_matching_pennies():
    a=np.array([[1,-1],[-1,1.]])
    r=solve_support_lp(a,-a)
    assert len(r['equilibria'])==1
    assert np.allclose(r['equilibria'][0]['row_probabilities'],[.5,.5])
    assert r['equilibria'][0]['nash_gap']<1e-7

def test_degenerate_constant():
    a=np.zeros((2,3))
    r=solve_support_lp(a,a)
    assert r['equilibria']
    assert any(len(z['support']['row'])!=len(z['support']['col']) for z in r['equilibria'])

def test_pure_prisoners():
    a=np.array([[3,0],[5,1.]])
    b=np.array([[3,5],[0,1.]])
    r=solve_support_lp(a,b)
    assert any(np.allclose(z['row_probabilities'],[0,1]) and np.allclose(z['col_probabilities'],[0,1]) for z in r['equilibria'])

def test_budget():
    with pytest.raises(ValueError,match='exceed'):
        solve_support_lp(np.zeros((8,8)),np.zeros((8,8)),max_pairs=100)
