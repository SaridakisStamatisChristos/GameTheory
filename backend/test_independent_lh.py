import numpy as np
from independent_lh import solve_independent_lh

def test_classics():
    for a,b in [([[-1,1],[1,-1]],[[1,-1],[-1,1]]),([[3,0],[0,2]],[[3,0],[0,2]]),([[1,1],[1,1]],[[1,1],[1,1]])]:
        r=solve_independent_lh(a,b)
        assert r['equilibria'] and all(e['nash_gap']<1e-7 for e in r['equilibria'])

def test_random_differential():
    import nashpy as nash
    rng=np.random.default_rng(20261103)
    for size in (2,3,4,5):
        for _ in range(12):
            a=rng.integers(-5,6,(size,size)).astype(float)
            b=rng.integers(-5,6,(size,size)).astype(float)
            result=solve_independent_lh(a,b)
            assert result['equilibria'], (size,result['failures'])
            for e in result['equilibria']:
                x=np.array(e['row_probabilities']); y=np.array(e['col_probabilities'])
                assert max(np.max(a@y)-x@a@y,np.max(x@b)-x@b@y)<1e-7

def test_invalid():
    import pytest
    with pytest.raises(ValueError): solve_independent_lh([[1,2]],[[1]])
