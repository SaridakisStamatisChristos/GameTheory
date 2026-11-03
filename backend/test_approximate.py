import numpy as np
from fastapi.testclient import TestClient
from main import app
from approximate import approximate_nash, nash_gap

client = TestClient(app)

def test_certificate_exact_matching_pennies():
    a = np.array([[1., -1.], [-1., 1.]])
    c = nash_gap(a, -a, np.array([.5,.5]), np.array([.5,.5]))
    assert c['epsilon'] == 0

def test_dominant_strategy_convergence():
    a = np.array([[3.,3.],[0.,0.]])
    b = np.array([[2.,0.],[2.,0.]])
    result = approximate_nash(a,b,iterations=1000,restarts=2,seed=42)
    assert result['profile']['certificate']['epsilon'] < .02

def test_reproducibility():
    a=np.array([[1.,-1.],[-1.,1.]])
    first=approximate_nash(a,-a,iterations=200,restarts=2,seed=4)
    second=approximate_nash(a,-a,iterations=200,restarts=2,seed=4)
    assert first==second

def test_endpoint_and_invalid_input():
    data={'row_payoffs':[[1,-1],[-1,1]],'col_payoffs':[[-1,1],[1,-1]],'iterations':200,'restarts':2}
    response=client.post('/api/equilibria/approximate',json=data)
    assert response.status_code==200
    assert response.json()['profile']['certificate']['epsilon'] >= 0
    data['iterations']=1
    assert client.post('/api/equilibria/approximate',json=data).status_code==422
