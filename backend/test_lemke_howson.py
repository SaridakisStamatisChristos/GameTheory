import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app
from lemke_howson import solve_lemke_howson


def test_matching_pennies():
    a=np.array([[1.,-1.],[-1.,1.]])
    r=solve_lemke_howson(a,-a)
    assert r['equilibria']
    assert min(e['nash_gap'] for e in r['equilibria']) < 1e-7
    assert r['complete'] is False


def test_invalid():
    with pytest.raises(ValueError):
        solve_lemke_howson([[1,2]],[[1]])


def test_api():
    r=TestClient(app).post('/api/equilibria/lemke-howson',json={'row_payoffs':[[1,-1],[-1,1]],'col_payoffs':[[-1,1],[1,-1]]})
    assert r.status_code==200
    assert r.json()['equilibria']
