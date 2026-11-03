from fastapi.testclient import TestClient
from main import app
c=TestClient(app)
def test_prisoners_dilemma():
    r=c.post('/api/analyze',json={'row_payoffs':[[-1,-3],[0,-2]],'col_payoffs':[[-1,0],[-3,-2]]})
    assert r.status_code==200
    assert [(x['row'],x['col']) for x in r.json()['pure_nash']]==[(1,1)]
def test_matching_pennies():
    r=c.post('/api/analyze',json={'row_payoffs':[[1,-1],[-1,1]],'col_payoffs':[[-1,1],[1,-1]]})
    assert not r.json()['pure_nash']
    assert abs(r.json()['interior_mixed_nash_2x2'][0]['row_probabilities'][0]-.5)<1e-8
def test_zero_sum():
    r=c.post('/api/zero-sum',json={'payoffs':[[1,-1],[-1,1]]})
    assert r.status_code==200
    assert abs(r.json()['value'])<1e-8
    assert r.json()['duality_gap']<1e-8
def test_invalid():
    assert c.post('/api/analyze',json={'row_payoffs':[[1,2],[3]],'col_payoffs':[[1,2],[3,4]]}).status_code==422
