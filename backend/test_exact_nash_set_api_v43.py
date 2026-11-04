"""HTTP contracts for the complete Nash-family geometry endpoint."""
from fractions import Fraction
from fastapi.testclient import TestClient
from main import app


def test_exact_nash_set_api_three_by_three_continuum():
    with TestClient(app,raise_server_exceptions=False) as client:
        response=client.post('/api/equilibria/exact-nash-set',json={
            'row_payoffs':[[0]*3 for _ in range(3)],
            'col_payoffs':[[0]*3 for _ in range(3)]})
    assert response.status_code==200,response.text
    data=response.json()
    assert data['equilibrium_set_complete'] is True
    assert data['all_equilibria_enumerated_as_points'] is False
    assert data['nonempty_support_cells']==49
    assert data['support_pairs_checked']==49
    assert any(len(f['support']['row'])==3 and len(f['support']['col'])==3
               for f in data['equilibrium_families'])
    assert all(sum(Fraction(w) for w in
                   f['row_strategy_polytope']['witness_probabilities_exact'])==1
               for f in data['equilibrium_families'])


def test_exact_nash_set_api_rejects_oversize_and_budget_exhaustion():
    small={'row_payoffs':[[0]*3 for _ in range(3)],
           'col_payoffs':[[0]*3 for _ in range(3)]}
    large={'row_payoffs':[[0]*5 for _ in range(5)],
           'col_payoffs':[[0]*5 for _ in range(5)]}
    with TestClient(app,raise_server_exceptions=False) as client:
        assert client.post('/api/equilibria/exact-nash-set',json={
            **small,'max_pairs':8}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={
            **small,'max_operations':1}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json=large).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={
            **small,'max_pairs':0}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={
            **small,'max_operations':500001}).status_code==422


def test_existing_exact_support_endpoint_preserved():
    game={'row_payoffs':[[1,-1],[-1,1]],
          'col_payoffs':[[-1,1],[1,-1]]}
    with TestClient(app,raise_server_exceptions=False) as client:
        old=client.post('/api/equilibria/exact-support',json=game)
        new=client.post('/api/equilibria/exact-nash-set',json=game)
    assert old.status_code==new.status_code==200
    assert old.json()['feasible_support_pairs']==new.json()['nonempty_support_cells']==1
