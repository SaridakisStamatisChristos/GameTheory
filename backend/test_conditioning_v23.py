import numpy as np
from independent_lh import solve_independent_lh
from benchmark_conditioning_v23 import cases

def test_adversarial_conditioning_small():
    for n,family,k,a,b in cases(per_family=2):
        if n>5: continue
        result=solve_independent_lh(a,b,max_labels=min(3,2*n))
        assert result['equilibria'], (n,family,k,result['failures'])
        for eq in result['equilibria']:
            x=np.asarray(eq['row_probabilities']); y=np.asarray(eq['col_probabilities'])
            assert np.isclose(x.sum(),1) and np.isclose(y.sum(),1)
            assert min(x.min(),y.min())>=-1e-7
            gap=max(0.,np.max(a@y)-x@a@y,np.max(x@b)-x@b@y)
            assert gap<=1e-7, (n,family,k,gap)
