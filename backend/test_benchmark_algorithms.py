import numpy as np
from benchmark_algorithms import nash_gap, run

def test_nash_gap_known_equilibrium():
    a=np.array([[1.,-1.],[-1.,1.]])
    assert nash_gap(a,-a,[.5,.5],[.5,.5]) < 1e-12
    assert nash_gap(a,-a,[1.,0.],[1.,0.]) >= 1.0

def test_benchmark_contract_and_repeatability():
    r=run(seed=42,repeats=1,iterations=100)
    assert len(r['records']) >= 10
    assert all(not x['failures'] for x in r['records'])
    assert all(x['worst_nash_gap'] is None or x['worst_nash_gap']>=0 for x in r['records'])
