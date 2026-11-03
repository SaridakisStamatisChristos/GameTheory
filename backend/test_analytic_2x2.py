import numpy as np
import pytest
from analytic_2x2 import solve_2x2

@pytest.mark.parametrize('a,b',[
    ([[1,-1],[-1,1]],[[-1,1],[1,-1]]),
    ([[3,0],[5,1]],[[3,5],[0,1]]),
    ([[2,0],[0,1]],[[2,0],[0,1]]),
    ([[0,0],[0,0]],[[0,0],[0,0]]),
    ([[2,2],[2,2]],[[1,1],[1,1]]),
    ([[1,1],[0,0]],[[1,0],[1,0]]),
])
def test_known(a,b):
    result=solve_2x2(a,b)
    assert result['equilibria']
    assert all(e['nash_gap']<1e-8 for e in result['equilibria'])

def test_random_against_nashpy():
    nashpy=pytest.importorskip('nashpy')
    rng=np.random.default_rng(1977)
    for _ in range(100):
        a=rng.integers(-3,4,(2,2)).astype(float)
        b=rng.integers(-3,4,(2,2)).astype(float)
        result=solve_2x2(a,b)
        assert result['equilibria']
        for e in result['equilibria']:
            p=np.array(e['row_probabilities']);q=np.array(e['col_probabilities'])
            assert max(np.max(a@q)-p@a@q,np.max(p@b)-p@b@q)<1e-8

def test_invalid():
    with pytest.raises(ValueError): solve_2x2([[1]],[[1]])
