"""API-level exact equilibrium support contract and budget limits."""
from fractions import Fraction
from fastapi.testclient import TestClient
from main import app

def test_exact_support_api_complete_degenerate_3x3():
    client=TestClient(app,raise_server_exceptions=False)
    payload={'row_payoffs':[[0]*3 for _ in range(3)],
             'col_payoffs':[[0]*3 for _ in range(3)]}
    r=client.post('/api/equilibria/exact-support',json=payload)
    assert r.status_code==200,r.text
    data=r.json()
    assert data['support_pairs_checked']==49
    assert data['complete_support_feasibility'] is True
    assert data['all_equilibria_enumerated'] is False
    assert len(data['equilibria'])==49
    for eq in data['equilibria']:
        assert sum(Fraction(p) for p in eq['row_probabilities_exact'])==1
        assert sum(Fraction(p) for p in eq['col_probabilities_exact'])==1

def test_exact_support_api_guards_failed_budgets():
    client=TestClient(app,raise_server_exceptions=False)
    three={'row_payoffs':[[0]*3 for _ in range(3)],
           'col_payoffs':[[0]*3 for _ in range(3)]}
    assert client.post('/api/equilibria/exact-support',json={**three,'max_pairs':5}).status_code==422
    assert client.post('/api/equilibria/exact-support',json={**three,'max_operations':1}).status_code==422
    five={'row_payoffs':[[0]*5 for _ in range(5)],
          'col_payoffs':[[0]*5 for _ in range(5)]}
    assert client.post('/api/equilibria/exact-support',json=five).status_code==422
    assert client.post('/api/equilibria/exact-support',json={**three,'max_pairs':-1}).status_code==422
    assert client.post('/api/equilibria/exact-support',json={**three,'max_operations':0}).status_code==422

def test_existing_numeric_support_lp_endpoint_remains_compatible():
    client=TestClient(app,raise_server_exceptions=False)
    payload={'row_payoffs':[[1,-1],[-1,1]],
             'col_payoffs':[[-1,1],[1,-1]]}
    exact=client.post('/api/equilibria/exact-support',json=payload)
    assert exact.status_code==200,exact.text
    assert exact.json()['equilibria'][0]['row_probabilities_exact']==['1/2','1/2']
    legacy=client.post('/api/equilibria/support-lp',json=payload)
    assert legacy.status_code==200,legacy.text
