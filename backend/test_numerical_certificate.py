import numpy as np
import pytest
from numerical_certificate import certify_bimatrix
from independent_lh import solve_independent_lh


def test_matching_pennies():
    a=np.array([[1.,-1.],[-1.,1.]])
    c=certify_bimatrix(a,-a,[.5,.5],[.5,.5])
    assert c['valid'] and c['nash_gap']==0
    assert not certify_bimatrix(a,-a,[1.,0.],[1.,0.])['valid']


def test_translation_invariance():
    a=np.array([[2.,0.],[0.,1.]])
    x=[.5,.5]; y=[.5,.5]
    p=certify_bimatrix(a,a,x,y)
    q=certify_bimatrix(a+1000,a-2000,x,y)
    assert np.isclose(p['nash_gap'],q['nash_gap'])
    assert p['valid']==q['valid']


def test_invalid_inputs():
    with pytest.raises(ValueError): certify_bimatrix([[1]],[[1]],[float('nan')],[1])
    with pytest.raises(ValueError): certify_bimatrix([[1]],[[1]],[1,0],[1])


def test_solver_certificate_random():
    rng=np.random.default_rng(2401)
    for n in (2,3,4):
        for _ in range(4):
            a=rng.integers(-5,6,(n,n)).astype(float)
            b=rng.integers(-5,6,(n,n)).astype(float)
            result=solve_independent_lh(a,b,max_labels=2*n)
            assert result['equilibria'],result['failures']
            for eq in result['equilibria']:
                assert certify_bimatrix(a,b,eq['row_probabilities'],eq['col_probabilities'])['valid']
