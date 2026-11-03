"""Independent-facing HTTP checks for v38's local-lab security boundary."""
from fastapi.testclient import TestClient
import pytest
from main import app


def test_anonymous_remote_does_not_reach_experiments():
    with TestClient(app, base_url='http://example.org', client=('198.51.100.30', 51111)) as client:
        r=client.get('/api/experiments')
    assert r.status_code==403


def test_auth_protects_local_experiments_when_configured(monkeypatch,tmp_path):
    monkeypatch.setenv('GAME_LAB_API_TOKEN','test-secret-token-only')
    monkeypatch.setenv('GAME_LAB_DB',str(tmp_path/'guard-test.sqlite'))
    with TestClient(app) as client:
        assert client.get('/api/experiments').status_code==401
        assert client.get('/api/experiments',headers={'Authorization':'Bearer invalid'}).status_code==401
        headers={'Authorization':'Bearer test-secret-token-only'}
        saved=client.post('/api/experiments',json={'title':'test','game':{'a':1}},headers=headers)
        assert saved.status_code==200
        got=client.get('/api/experiments/'+saved.json()['id'],headers=headers)
        assert got.status_code==200 and got.json()['game']=={'a':1}


def test_remote_requires_token(monkeypatch):
    monkeypatch.setenv('GAME_LAB_API_TOKEN','test-secret-token-only')
    with TestClient(app,base_url='http://remote.example',client=('203.0.113.21',7000)) as client:
        assert client.get('/api/experiments').status_code==403
        assert client.get('/api/experiments',headers={'Authorization':'Bearer test-secret-token-only'}).status_code==200


def test_large_request_rejected_before_json_decode():
    with TestClient(app) as client:
        r=client.post('/api/analyze',content=b'['+b'0'*300000+b']',headers={'Content-Type':'application/json'})
        assert r.status_code==413


def test_large_support_enumeration_rejected():
    zeros=[[0]*9 for _ in range(9)]
    with TestClient(app) as client:
        r=client.post('/api/equilibria/support-lp',json={'row_payoffs':zeros,'col_payoffs':zeros})
        assert r.status_code==422


def test_large_approximate_nash_budget_rejected():
    with TestClient(app) as client:
        r=client.post('/api/equilibria/approximate',json={'row_payoffs':[[1]],'col_payoffs':[[1]],'iterations':200000,'restarts':64})
        assert r.status_code==422


def test_large_cfr_plus_budget_rejected():
    from cfr_plus import CFRPlusRequest, validate_cfr_game
    from fastapi import HTTPException
    # 102 nodes * 20000 iterations exceeds the hard node-iteration work budget.
    nodes={f'n{k}':{'player':'chance','actions':{'a':f'n{k+1}'},'probabilities':{'a':1.}} for k in range(101)}
    nodes['n101']={'payoff':0}
    with pytest.raises(HTTPException, match='budget'):
        validate_cfr_game(CFRPlusRequest(root='n0',nodes=nodes,iterations=20000))


def test_2x2_exact_interior_oracle_detects_tiny_positive_support():
    from support_lp import solve_support_lp
    a=[[1,-1],[-1,1]]
    b=[[0,1e9],[0,-1e-8]]
    result=solve_support_lp(a,b)
    assert any(0<e['row_probabilities'][0]<1e-12 and 0<e['col_probabilities'][0]<1
               and e.get('certificate')=='exact rational indifference'
               for e in result['equilibria'])
    assert result['complete_support_coverage']
    assert not result['all_feasible_supports_resolved']


def test_correlated_equilibrium_obedience_randomized():
    import numpy as np
    from research import correlated_equilibrium
    rng=np.random.default_rng(581)
    for _ in range(30):
        a=rng.integers(-5,6,size=(2,2)).astype(float)
        b=rng.integers(-5,6,size=(2,2)).astype(float)
        result=correlated_equilibrium(a,b)
        q=np.asarray(result['joint_distribution'])
        for i in range(2):
            for k in range(2):
                assert q[i] @ (a[k]-a[i]) < 1e-7
        for j in range(2):
            for k in range(2):
                assert q[:,j] @ (b[:,k]-b[:,j]) < 1e-7


def test_replicator_support_invariant_under_randomized_payoff_scales():
    import numpy as np
    from advanced import replicator
    for scale in (1e-13,1e-3,1.,1e12,1e200):
        result=replicator([[0,0],[scale,-scale]],initial=[1,0],steps=5,dt=.25)
        assert result['final_distribution']==[1.,0.]


def test_analytic_scaled_game_rejects_spurious_pure_solutions():
    from analytic_2x2 import solve_2x2
    for scale in (1e-13,1.,1e100):
        a=[[scale,-scale],[-scale,scale]]
        b=[[-v for v in row] for row in a]
        result=solve_2x2(a,b)
        assert len(result['equilibria'])==1
        assert result['equilibria'][0]['row_probabilities']==[.5,.5]
        assert result['equilibria'][0]['col_probabilities']==[.5,.5]


def test_fast_support_pure_nash_scale_invariant():
    from advanced import support_equilibria
    for scale in (1e-13,1.,1e100):
        a=[[scale,-scale],[-scale,scale]]
        b=[[-v for v in row] for row in a]
        result=support_equilibria(a,b)
        assert all(len(z['support']['row'])>1 for z in result)
        assert any(z['support']=={'row':[0,1],'col':[0,1]} for z in result)


def test_remote_rate_budget_applies():
    with TestClient(app, base_url='http://remote.example',client=('198.51.100.66', 10000)) as client:
        results=[client.get('/api/health').status_code for _ in range(122)]
    assert results[:120]==[200]*120
    assert results[-2:]==[429,429]


def test_public_statewise_nash_not_fooled_by_tiny_payoffs():
    from research import BayesianGame,bayesian_evaluate
    result=bayesian_evaluate(BayesianGame(prior=[1],row_payoffs=[[[0],[1e-13]]],
                       col_payoffs=[[[0],[0]]],row_strategy=[0],col_strategy=[0]))
    assert result['is_statewise_nash'] is False


def test_evolution_rejects_ragged_payoffs_with_422():
    with TestClient(app) as client:
        result=client.post('/api/evolution',json={'payoffs':[[1,2],[3]]})
        assert result.status_code==422


def test_repeated_game_extreme_finite_cumulative_is_json_safe():
    from extended import RepeatedGame,repeated_payoffs
    result=repeated_payoffs(RepeatedGame(row_payoffs=[[1e308]],col_payoffs=[[-1e308]],
                                row_strategy=[0],col_strategy=[0],rounds=1000,discount=1))
    assert float(result['discounted_payoffs'][0])>1e308
    assert float(result['discounted_payoffs'][1])< -1e308
