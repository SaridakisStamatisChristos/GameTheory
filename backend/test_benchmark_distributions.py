import numpy as np
import pytest
from benchmark_distributions import summary, benchmark
from reference_benchmark import gap


def test_summary():
    s=summary([1,2,3,4,5])
    assert s['median_ms']==3 and s['n']==5 and s['p10_ms']<=s['median_ms']<=s['p90_ms']


def test_invalid_repetitions():
    with pytest.raises(ValueError): benchmark(repeats=0)


def test_independent_gap():
    a=np.array([[1.,-1.],[-1.,1.]])
    assert gap(a,-a,[.5,.5],[.5,.5])<1e-12
    assert gap(a,-a,[1.,0.],[1.,0.])>0
