"""HTTP contract regression tests for previously untested solver entrypoints."""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_independent_lh_http_matching_pennies():
    response = client.post('/api/equilibria/independent-lemke-howson', json={
        'row_payoffs': [[1,-1],[-1,1]],
        'col_payoffs': [[-1,1],[1,-1]],
        'max_labels': 4,
    })
    assert response.status_code == 200, response.text
    assert 'NaN' not in response.text and 'Infinity' not in response.text

def test_independent_lh_http_ragged_rejected():
    response = client.post('/api/equilibria/independent-lemke-howson', json={
        'row_payoffs': [[1,2],[3]], 'col_payoffs': [[1,2],[3,4]]})
    assert response.status_code == 422

def test_zero_sum_http_ragged_rejected():
    response = client.post('/api/zero-sum', json={'payoffs': [[1,2],[3]]})
    assert response.status_code == 422
