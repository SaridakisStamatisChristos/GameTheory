"""v45 HTTP integration checks, existing route isolation and truthful statuses."""
from fastapi.testclient import TestClient
from main import app


def post(game,**opts):
    with TestClient(app,raise_server_exceptions=False) as client:
        return client.post('/api/equilibria/general',json={
            'row_payoffs':game[0],'col_payoffs':game[1],**opts})


def test_auto_and_exact_http_contracts():
    a=[[1.,-1.],[-1.,1.]]
    b=[[-1.,1.],[1.,-1.]]
    r=post((a,b))
    assert r.status_code==200,r.text
    assert r.json()['status']=='certified_candidate'
    assert r.json()['best_candidate']['certificate']['valid']
    assert r.json()['equilibrium_set_complete'] is False
    e=post((a,b),mode='exact',max_exact_pairs=9,max_exact_support=2)
    assert e.status_code==200,e.text
    assert e.json()['status']=='complete_exact_set'
    assert e.json()['equilibrium_set_complete'] is True
    assert e.json()['exact_search']['support_pairs_checked']==9


def test_large_dimension_http_uses_existing_input_bound():
    n=20
    a=[[0]*n for _ in range(n)]
    r=post((a,a))
    assert r.status_code==200,r.text
    assert r.json()['original_shape']==[20,20]
    assert r.json()['best_candidate']['evidence']=='exact_pure_equilibrium'
    too_large=[[0]*21 for _ in range(21)]
    assert post((too_large,too_large)).status_code==422


def test_http_exact_partial_budget_exposed():
    z=[[0,0],[0,0]]
    r=post((z,z),mode='exact',max_exact_pairs=1)
    assert r.status_code==200,r.text
    assert r.json()['equilibrium_set_complete'] is False
    assert r.json()['exact_search']['incomplete_reason']=='pair_budget'
    assert r.json()['status']=='partial_exact_search'


def test_http_budget_and_malformed_requests_fail():
    z=[[0]]
    for opts in ({'mode':'bad'},{'iterations':1},{'max_pure_cells':0},
                 {'max_exact_pairs':0},{'max_exact_operations':0},
                 {'max_exact_support':21}):
        response=post((z,z),**opts)
        assert response.status_code==422,(opts,response.text)


def test_other_v44_endpoint_still_exists():
    with TestClient(app,raise_server_exceptions=False) as client:
        r=client.post('/api/equilibria/exact-nash-set',json={
            'row_payoffs':[[0,0],[0,0]],
            'col_payoffs':[[0,0],[0,0]]})
    assert r.status_code==200,r.text
    assert r.json()['equilibrium_set_complete']
