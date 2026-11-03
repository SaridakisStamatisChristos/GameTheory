"""Deterministic API contract fuzzing; no external property-testing dependency."""
import math
import random
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app, raise_server_exceptions=False)
ENDPOINTS = ['/api/analyze','/api/equilibria','/api/correlated-equilibrium',
             '/api/equilibria/analytic-2x2','/api/equilibria/hybrid',
             '/api/equilibria/support-lp','/api/equilibria/independent-lemke-howson']

@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_malformed_matrix_never_500(endpoint):
    rng = random.Random(3201)
    bad = [None, [], [[], []], [[1,2],[3]], [[1,'x'],[2,3]],
           [[1,None],[2,3]], [[1e300,1e300],[0,0]],
           [[True,False],[False,True]], {'a': 1}, 'garbage']
    for _ in range(30):
        rows = rng.randint(1,4)
        bad.append([[rng.choice([None,'oops',{},[],1]) for _ in range(rng.randint(0,4))] for _ in range(rows)])
    for matrix in bad:
        response = client.post(endpoint,json={'row_payoffs':matrix,'col_payoffs':[[1,2],[3,4]]})
        assert response.status_code in (200,422), (endpoint,matrix,response.status_code,response.text[:300])
        assert 'NaN' not in response.text and 'Infinity' not in response.text

@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_random_finite_2x2_no_server_error(endpoint):
    rng=random.Random(3202)
    for _ in range(12):
        a=[[rng.randint(-5,5) for _ in range(2)] for _ in range(2)]
        b=[[rng.randint(-5,5) for _ in range(2)] for _ in range(2)]
        r=client.post(endpoint,json={'row_payoffs':a,'col_payoffs':b,'max_pairs':100})
        assert r.status_code != 500, (endpoint,a,b,r.text[:300])
        assert 'NaN' not in r.text and 'Infinity' not in r.text

@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_nonfinite_json_literal_rejected(endpoint):
    response=client.post(endpoint,content='{"row_payoffs":[[NaN,0],[0,1]],"col_payoffs":[[0,1],[1,0]]}',headers={'content-type':'application/json'})
    assert response.status_code in (400,422), (endpoint,response.status_code,response.text[:200])

def test_extreme_correlated_equilibrium_obedience():
    response=client.post('/api/correlated-equilibrium',json={
        'row_payoffs':[[1e300,1e300],[0,0]],'col_payoffs':[[1,2],[3,4]]})
    assert response.status_code==200,response.text
    q=response.json()['joint_distribution']
    assert abs(sum(map(sum,q))-1)<1e-9
    assert sum(q[1])<1e-7
