"""v44 HTTP surface: 5x5 opt-in, vertices and bounded rejection."""
from fastapi.testclient import TestClient
from main import app


def test_optin_five_by_five_and_exact_geometry_api():
    z=[[0]*5 for _ in range(5)]
    with TestClient(app,raise_server_exceptions=False) as client:
        data={'row_payoffs':z,'col_payoffs':z}
        assert client.post('/api/equilibria/exact-nash-set',json=data).status_code==422
        assert client.post('/api/equilibria/exact-support',json=data).status_code==422
        opts=dict(data,max_actions=5,max_pairs=961,include_vertices=True)
        result=client.post('/api/equilibria/exact-nash-set',json=opts)
        assert result.status_code==200,result.text[:2000]
        r=result.json()
        assert r['nonempty_support_cells']==961 and r['vertices_computed']
        assert max(f['product_affine_dimension'] for f in r['equilibrium_families'])==8
        legacy=client.post('/api/equilibria/exact-support',json=dict(data,max_actions=5,max_pairs=961))
        assert legacy.status_code==200,legacy.text[:200]
        assert legacy.json()['feasible_support_pairs']==961


def test_api_vertex_budget_and_invalid_input_reject():
    z=[[0,0],[0,0]]
    with TestClient(app,raise_server_exceptions=False) as client:
        data={'row_payoffs':z,'col_payoffs':z,'include_vertices':True}
        assert client.post('/api/equilibria/exact-nash-set',json={**data,'max_vertex_candidates':1}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={**data,'max_actions':6}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={**data,'max_actions':3}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={**data,'max_pairs':962}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={**data,'max_operations':2000001}).status_code==422
        assert client.post('/api/equilibria/exact-nash-set',json={**data,'include_vertices':False}).status_code==200
