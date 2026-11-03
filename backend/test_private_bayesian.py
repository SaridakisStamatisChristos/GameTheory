from fastapi.testclient import TestClient
from main import app
client=TestClient(app)
def test_private_equilibrium():
    g={'joint_prior':[[.25,.25],[.25,.25]],'row_payoffs':[[[[1,1],[0,0]],[[1,1],[0,0]]],[[[1,1],[0,0]],[[1,1],[0,0]]]],'col_payoffs':[[[[2,0],[2,0]],[[2,0],[2,0]]],[[[2,0],[2,0]],[[2,0],[2,0]]]],'row_policy':[0,0],'col_policy':[0,0]}
    r=client.post('/api/bayesian/private-evaluate',json=g)
    assert r.status_code==200,r.text
    assert r.json()['is_bayesian_nash']
    g['row_policy']=[1,0]
    assert not client.post('/api/bayesian/private-evaluate',json=g).json()['is_bayesian_nash']
def test_invalid_prior():
    g={'joint_prior':[[1,1]],'row_payoffs':[[[[1]] ,[[1]]]],'col_payoffs':[[[[1]],[[1]]]],'row_policy':[0],'col_policy':[0,0]}
    assert client.post('/api/bayesian/private-evaluate',json=g).status_code==422
