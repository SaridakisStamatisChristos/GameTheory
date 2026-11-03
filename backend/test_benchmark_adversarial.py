import numpy as np
from benchmark_adversarial import cases, certify, run

def test_determinism():
    a=list(cases(11,1)); b=list(cases(11,1))
    assert len(a)==len(b)
    assert all(x[0]==y[0] and np.array_equal(x[1],y[1]) and np.array_equal(x[2],y[2]) for x,y in zip(a,b))

def test_certificate_rejects_false_equilibrium():
    a=np.array([[1.,-1.],[-1.,1.]])
    assert certify(a,-a,{'row_probabilities':[1,0],'col_probabilities':[1,0]})>0
    assert certify(a,-a,{'row_probabilities':[.5,.5],'col_probabilities':[.5,.5]})<1e-10
    assert certify(a,-a,{'row_probabilities':[2,0],'col_probabilities':[1,0]}) is None

def test_campaign_smoke():
    result=run(random_per_size=0,repeats=1)
    assert len(result['records'])==32
    assert all(r['valid_runs']+len(r['failures'])==1 for r in result['records'])
